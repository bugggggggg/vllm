# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QK offload for Whisper word timestamps in the GPU model runner.

The model writes the alignment heads' query rows (and, on a request's first
step, the encoder keys) into a small static GPU staging buffer every step (see
``vllm/model_executor/models/whisper_word_timing.py``). This module:

1. ``before_forward``: records which staged rows belong to which request that
   asked for word timestamps;
2. ``after_forward``: enqueues an async D2H copy of the stage into a pinned
   ring slot on the compute stream (so the next forward cannot overwrite the
   stage before it is read) and hands the slot to a host thread, which waits
   on the copy and scatters rows into per-request CPU buffers;
3. ``make_readout_fn``: when a request emits its last token, reconstructs
   cross-attention on the CPU and returns per-token start times.

No per-request state lives on the GPU, so capture never competes with the KV
cache, and the readout launches no GPU work.
"""

import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Number of in-flight step copies; more only helps if the host thread lags.
RING_SIZE = 4


class QKOffloadWordTimestamps:
    """Per-request word-timestamp capture with Q/K offloaded to host memory."""

    def __init__(self) -> None:
        self.enabled = False
        # Requests that asked for word timestamps (``extra_args``).
        self.wants: set[str] = set()
        self.prompt_len: dict[str, int] = {}
        self.max_out: dict[str, int] = {}
        # Decoder positions consumed so far (= captured query rows).
        self.npos: dict[str, int] = {}
        # Host buffers, filled by the scatter thread.
        self.q_host: dict[str, torch.Tensor] = {}
        self.k_host: dict[str, torch.Tensor] = {}
        self.nframes: dict[str, int] = {}
        self._pending: tuple | None = None
        self._lock = threading.Lock()
        self._done = threading.Condition()
        self._seq_submitted = 0
        self._seq_done = 0
        self.num_dropped = 0

    # ------------------------------------------------------------------ setup
    def init(self, runner: "GPUModelRunner") -> None:
        model_config = runner.model_config
        if not model_config.enable_word_timestamps:
            return
        model = runner.model
        if not getattr(model, "supports_qk_offload_word_timestamps", False):
            logger.warning(
                "--enable-word-timestamps is set but %s does not support it.",
                type(model).__name__,
            )
            return
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(
            model_config.model, revision=model_config.revision
        )
        hf_config = model_config.hf_config
        dtype = model_config.dtype
        assert isinstance(dtype, torch.dtype)
        self.max_frames = int(hf_config.max_source_positions)
        self.max_model_len = int(model_config.max_model_len)
        eos = gen_config.eos_token_id
        self.eos = eos[0] if isinstance(eos, list) else int(eos)
        self.mfw = int(getattr(hf_config, "median_filter_width", 7))
        # Upper bound on new audio items per step: the encoder token budget
        # over the encoder length per item (1500 for Whisper).
        enc_budget = int(
            getattr(runner.scheduler_config, "max_num_encoder_input_tokens", 0)
            or runner.max_num_tokens
        )
        self.max_items = max(
            1, min(runner.max_num_reqs, enc_budget // self.max_frames + 1)
        )
        self.qstage, self.kstage, self.nfstage = model.enable_qk_offload(
            alignment_heads=gen_config.alignment_heads,  # type: ignore[attr-defined]
            max_tokens=runner.max_num_tokens,
            max_items=self.max_items,
            device=runner.device,
            dtype=dtype,
        )
        self.n_heads, self.head_dim = self.qstage.shape[1], self.qstage.shape[2]
        self.dtype = dtype

        pin = is_pin_memory_available()

        def host(shape, dt=dtype):
            return torch.empty(shape, dtype=dt, device="cpu", pin_memory=pin)

        self._q_ring = [host(self.qstage.shape) for _ in range(RING_SIZE)]
        self._k_ring = [host(self.kstage.shape) for _ in range(RING_SIZE)]
        self._nf_ring = [host((self.max_items,), torch.long) for _ in range(RING_SIZE)]
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(RING_SIZE):
            self._free.put(i)
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._scatter_loop, name="qk-offload-scatter", daemon=True
        ).start()
        self._pool = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4), thread_name_prefix="qk-readout"
        )
        self.enabled = True
        per_req = (
            (self.max_model_len + self.max_frames)
            * self.n_heads
            * self.head_dim
            * self.qstage.element_size()
        )
        logger.info(
            "Word timestamps (QK offload): %d alignment heads, %.1f MiB GPU "
            "staging, up to %.2f MiB host memory per request.",
            self.n_heads,
            (self.qstage.nbytes + self.kstage.nbytes) / 2**20,
            per_req / 2**20,
        )

    # ------------------------------------------------------- request lifecycle
    def add_request(self, req_id: str, sampling_params, prompt_len: int) -> None:
        if not self.enabled or sampling_params is None:
            return
        extra = sampling_params.extra_args
        if not (extra and extra.get("word_timestamps")):
            return
        with self._lock:
            self.wants.add(req_id)
        self.prompt_len[req_id] = prompt_len
        max_tokens = sampling_params.max_tokens or self.max_model_len
        self.max_out[req_id] = min(max_tokens, self.max_model_len - prompt_len)

    def remove_request(self, req_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.wants.discard(req_id)
            self.q_host.pop(req_id, None)
            self.k_host.pop(req_id, None)
            self.nframes.pop(req_id, None)
        self.prompt_len.pop(req_id, None)
        self.max_out.pop(req_id, None)
        self.npos.pop(req_id, None)

    # ------------------------------------------------------------- per step
    def before_forward(self, input_batch: InputBatch) -> None:
        """Map this step's staged rows to the requests that want timestamps."""
        self._pending = None
        if not self.enabled or not self.wants:
            return
        qsl = input_batch.query_start_loc_np
        ncomp = input_batch.num_computed_tokens_np
        nsched = input_batch.num_scheduled_tokens
        q_rows: list[tuple[str, int, int, int]] = []
        k_rows: list[tuple[str, int]] = []
        n_new = 0  # requests on their first step, in batch order (= K order)
        for i, req_id in enumerate(input_batch.req_ids):
            first_step = int(ncomp[i]) == 0
            if req_id in self.wants:
                p0 = int(ncomp[i])
                q_rows.append((req_id, int(qsl[i]), int(qsl[i + 1]), p0))
                self.npos[req_id] = p0 + int(nsched[i])
                if first_step:
                    k_rows.append((req_id, n_new))
            if first_step:
                n_new += 1
        if not q_rows:
            return
        if k_rows and n_new > self.max_items:
            # More new audio than the K stage holds; the stage op skipped K.
            self.num_dropped += len(k_rows)
            logger.warning_once(
                "Word timestamps: %d new audio items in one step exceed the "
                "staging capacity (%d); some requests get no timestamps.",
                n_new,
                self.max_items,
            )
            k_rows = []
        self._pending = (q_rows, k_rows, input_batch.num_tokens, n_new)

    def after_forward(self) -> None:
        """Enqueue the async D2H copy of this step's stage."""
        if self._pending is None:
            return
        q_rows, k_rows, num_tokens, n_new = self._pending
        self._pending = None
        slot = self._free.get()  # back-pressure if the scatter thread lags
        self._q_ring[slot][:num_tokens].copy_(
            self.qstage[:num_tokens], non_blocking=True
        )
        if k_rows:
            nk = n_new * self.max_frames
            self._k_ring[slot][:nk].copy_(self.kstage[:nk], non_blocking=True)
            self._nf_ring[slot][:n_new].copy_(self.nfstage[:n_new], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._seq_submitted += 1
        self._jobs.put((self._seq_submitted, slot, event, q_rows, k_rows))

    def _scatter_loop(self) -> None:
        while True:
            seq, slot, event, q_rows, k_rows = self._jobs.get()
            event.synchronize()
            q_ring, k_ring = self._q_ring[slot], self._k_ring[slot]
            nf_ring = self._nf_ring[slot]
            f = self.max_frames
            with self._lock:
                for req_id, start, end, p0 in q_rows:
                    if req_id not in self.wants:
                        continue
                    n = min(end - start, self.max_model_len - p0)
                    if n <= 0:
                        continue
                    buf = self.q_host.get(req_id)
                    if buf is None:
                        buf = torch.empty(
                            (self.max_model_len, self.n_heads, self.head_dim),
                            dtype=self.dtype,
                        )
                        self.q_host[req_id] = buf
                    buf[p0 : p0 + n].copy_(q_ring[start : start + n])
                for req_id, i in k_rows:
                    if req_id in self.wants:
                        self.k_host[req_id] = k_ring[i * f : (i + 1) * f].clone()
                        self.nframes[req_id] = int(nf_ring[i])
            self._free.put(slot)
            with self._done:
                self._seq_done = seq
                self._done.notify_all()

    # ---------------------------------------------------------------- readout
    def make_readout_fn(self) -> Callable | None:
        """Bind this step's state; the returned fn runs once tokens are known."""
        if not self.enabled or not self.wants:
            return None
        target = self._seq_submitted
        # Snapshot everything the readout needs: with async scheduling it runs
        # after later steps may already have removed a finished request.
        npos = {
            r: (self.npos[r], self.prompt_len[r], self.max_out[r])
            for r in self.wants
            if r in self.npos
        }

        def readout(
            req_ids: list[str], sampled: list[list[int]]
        ) -> dict[str, list[float]] | None:
            from vllm.model_executor.models.whisper_word_timing import token_onsets

            jobs = []
            for i, req_id in enumerate(req_ids):
                state = npos.get(req_id)
                toks = sampled[i] if i < len(sampled) else None
                if state is None or not toks:
                    continue
                n, prompt_len, max_out = state
                n_out = n - prompt_len + 1  # output tokens including this one
                if (
                    toks[-1] != self.eos
                    and n_out < max_out
                    and n + 1 < self.max_model_len
                ):
                    continue
                jobs.append((req_id, n, prompt_len))
            if not jobs:
                return None
            # Wait until this step's rows have reached host memory.
            with self._done:
                self._done.wait_for(lambda: self._seq_done >= target)
            work = []
            with self._lock:
                for req_id, n, prompt_len in jobs:
                    q = self.q_host.get(req_id)
                    k = self.k_host.get(req_id)
                    if q is None or k is None:
                        continue
                    work.append((req_id, q[:n], k, self.nframes[req_id], prompt_len))
            if not work:
                return None
            times = self._pool.map(
                lambda w: token_onsets(w[1], w[2], w[3], w[4], self.mfw), work
            )
            return {w[0]: t for w, t in zip(work, times)}

        return readout

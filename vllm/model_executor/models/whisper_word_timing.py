# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper word-level timestamps from offloaded cross-attention Q/K.

Fused attention kernels never materialize attention probabilities, but word
timing in Whisper is read from the cross-attention of a few "alignment heads".
Instead of keeping attention matrices (or full-width Q/K) on the GPU, the model
writes only the alignment heads' query rows (and, on a request's first step,
the encoder keys) into a small static staging buffer. The model runner copies
that buffer to pinned host memory asynchronously each step (see
``vllm/v1/worker/gpu/qk_offload.py``), and when a request finishes its
attention weights are reconstructed and aligned on the CPU.

The reconstruction follows OpenAI Whisper's ``timing.find_alignment``: crop to
the real audio frames, softmax, standardize over tokens, median filter, average
the alignment heads, then DTW against the negated matrix.
"""

import string

import numpy as np
import torch
import torch.nn.functional as F

# Staging buffers, allocated by ``enable`` before compilation. Columns are the
# alignment heads flattened across layers in sorted (layer, head) order; row i
# is batch token i (Q) or encoder frame i of this step's new audio (K).
ENABLED = False
Q_STAGE: torch.Tensor | None = None  # [max_num_tokens, n_align_heads, head_dim]
K_STAGE: torch.Tensor | None = None  # [max_items * max_frames, n_align_heads, head_dim]
NF_STAGE: torch.Tensor | None = None  # [max_items] real encoder frames per audio

TIME_PER_FRAME = 0.02  # seconds per encoder frame (2 mel hops of 10 ms)


@torch.library.custom_op("whisper_qk_offload::stage", mutates_args={"qstage", "kstage"})
def _stage(
    q: torch.Tensor,
    k: torch.Tensor | None,
    qstage: torch.Tensor,
    kstage: torch.Tensor,
    heads: torch.Tensor,
    col0: int,
) -> None:
    """Copy this layer's alignment heads into columns [col0, col0 + len(heads)).

    Fixed buffers and no host synchronization, so it is safe inside CUDA graphs.
    """
    nq = q.shape[0]
    if nq > qstage.shape[0]:  # oversized profiling batch
        return
    hd = qstage.shape[2]
    nh = heads.shape[0]
    qstage[:nq, col0 : col0 + nh].copy_(q.view(nq, -1, hd).index_select(1, heads))
    if k is not None:
        nk = k.shape[0]
        if nk <= kstage.shape[0]:
            kstage[:nk, col0 : col0 + nh].copy_(
                k.view(nk, -1, hd).index_select(1, heads)
            )


@_stage.register_fake
def _(q, k, qstage, kstage, heads, col0):
    return None


def stage(q: torch.Tensor, k: torch.Tensor | None, heads: torch.Tensor, col0: int):
    torch.ops.whisper_qk_offload.stage(q, k, Q_STAGE, K_STAGE, heads, col0)


def allocate(
    n_align_heads: int,
    head_dim: int,
    max_tokens: int,
    max_k_frames: int,
    max_items: int,
    max_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global ENABLED, Q_STAGE, K_STAGE, NF_STAGE
    Q_STAGE = torch.zeros(max_tokens, n_align_heads, head_dim, device=device, dtype=dtype)
    K_STAGE = torch.zeros(
        max_k_frames, n_align_heads, head_dim, device=device, dtype=dtype
    )
    NF_STAGE = torch.full((max_items,), max_frames, device=device, dtype=torch.long)
    dtw_jump_times(np.zeros((2, 2)))  # compile the DTW kernel off the hot path
    ENABLED = True
    return Q_STAGE, K_STAGE, NF_STAGE


def record_real_frames(features: torch.Tensor | list[torch.Tensor], max_frames: int):
    """Stage each audio item's real (pre-padding) encoder-frame count.

    Whisper pads audio to 30 s; the padded log-mel columns all equal the last
    one, so the last column that differs marks the end of the real audio.
    Items arrive in the same order as this step's new requests.
    """
    assert NF_STAGE is not None
    fs = features if isinstance(features, torch.Tensor) else torch.stack(list(features))
    differs = (fs != fs[:, :, -1:]).any(dim=1)  # [N, n_mel_frames]
    idx = torch.arange(differs.shape[1], device=fs.device)
    n_mel = torch.where(differs, idx, idx.new_zeros(())).amax(dim=1) + 1
    nf = (n_mel // 2).clamp_(1, max_frames)
    if nf.shape[0] <= NF_STAGE.shape[0]:
        NF_STAGE[: nf.shape[0]].copy_(nf)


# ----------------------------------------------------------------------------
# CPU reconstruction
# ----------------------------------------------------------------------------


def median_filter(x: torch.Tensor, width: int) -> torch.Tensor:
    """Median filter along the last dim with reflect padding (odd ``width``)."""
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    xp = F.pad(x, (pad, pad), mode="reflect")
    return xp.unfold(-1, width, 1).sort(dim=-1).values[..., pad]


def _dtw_jump_times(cost_in: np.ndarray) -> np.ndarray:
    """DTW over ``cost_in`` [n_tokens, n_frames]; return each token's first frame.

    Same recurrence, tie-breaking and backtrace as OpenAI Whisper's ``dtw_cpu``.
    Compiled with numba (no GIL) by ``dtw_jump_times``.
    """
    n, m = cost_in.shape
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c = c0
                t = 0
            elif c1 < c0 and c1 < c2:
                c = c1
                t = 1
            else:
                c = c2
                t = 2
            cost[i, j] = cost_in[i - 1, j - 1] + c
            trace[i, j] = t
    trace[0, :] = 2
    trace[:, 0] = 1
    # Walk back from (n, m); the last frame visited for each token while walking
    # backwards is that token's first frame along the forward path.
    first = np.zeros(n, dtype=np.int64)
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0:
            first[i - 1] = j - 1
        t = trace[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return first


_DTW = None


def dtw_jump_times(cost: np.ndarray) -> np.ndarray:
    global _DTW
    if _DTW is None:
        try:
            from numba import njit

            _DTW = njit(cache=True, nogil=True)(_dtw_jump_times)
        except ImportError:
            _DTW = _dtw_jump_times
    return _DTW(np.ascontiguousarray(cost, dtype=np.float64))


def token_onsets(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    prompt_len: int,
    median_filter_width: int = 7,
) -> list[float]:
    """Start time (s) of every generated token of one request.

    ``q``: [n_positions, n_align_heads, head_dim], the alignment heads' queries
    for decoder positions 0..n_positions-1 (every input position the request
    has run). ``k``: [max_frames, n_align_heads, head_dim], the encoder keys.
    Row ``prompt_len - 1 + j`` is the query that predicted generated token j,
    so the result has ``n_positions - prompt_len + 1`` entries.
    """
    hd = q.shape[-1]
    qf = q.float().transpose(0, 1)  # [A, n, D]
    kf = k[:num_frames].float().permute(1, 2, 0)  # [A, D, F]
    w = torch.bmm(qf, kf).mul_(hd**-0.5).softmax(dim=-1)
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    w = median_filter((w - mean) / std, median_filter_width)
    matrix = w.mean(dim=0)[prompt_len - 1 :]
    first = dtw_jump_times(-matrix.double().numpy())
    return (first * TIME_PER_FRAME).tolist()


# ----------------------------------------------------------------------------
# Tokens -> words
# ----------------------------------------------------------------------------

# String (not set) membership, as in OpenAI Whisper: "" also counts.
_PUNCTUATION = string.punctuation
# Languages written without spaces between words: split on characters.
NO_SPACE_LANGUAGES = {"zh", "ja", "th", "lo", "my", "yue"}


def _subwords(tokenizer, tokens: list[int]) -> list[tuple[str, list[int]]]:
    """Group tokens so that each group decodes to whole unicode characters."""
    out: list[tuple[str, list[int]]] = []
    cur: list[int] = []
    for tok in tokens:
        cur.append(tok)
        text = tokenizer.decode(cur, clean_up_tokenization_spaces=False)
        if "�" not in text:
            out.append((text, cur))
            cur = []
    if cur:
        out.append((tokenizer.decode(cur, clean_up_tokenization_spaces=False), cur))
    return out


def split_words(
    tokenizer,
    token_ids: list[int],
    onsets: list[float],
    eot_token_id: int,
    split_on_characters: bool = False,
    offset: float = 0.0,
    merge_punctuation: bool = True,
) -> list[tuple[str, float, float]]:
    """Turn generated tokens and their onsets into (word, start, end) triples.

    A word starts at its first token's onset and ends at the onset of the token
    that follows it (the next word, a special/timestamp token, or EOT), as in
    OpenAI Whisper. Special tokens (id >= EOT) are boundaries, never words.
    With ``merge_punctuation`` (OpenAI's default post-processing), standalone
    punctuation is folded into the neighbouring word; words are stripped.
    """
    n = min(len(token_ids), len(onsets))
    words: list[tuple[str, float, float]] = []
    i = 0
    while i < n:
        if token_ids[i] >= eot_token_id:
            i += 1
            continue
        j = i
        while j < n and token_ids[j] < eot_token_id:
            j += 1
        # tokens[i:j] is a run of text tokens; split it into words.
        pos = i
        groups: list[tuple[str, int, int]] = []  # (text, start_idx, end_idx)
        for text, toks in _subwords(tokenizer, token_ids[i:j]):
            start = pos
            pos += len(toks)
            new_word = (
                split_on_characters
                or not groups
                or text.startswith(" ")
                or text.strip() in _PUNCTUATION
            )
            if new_word:
                groups.append((text, start, pos))
            else:
                prev_text, prev_start, _ = groups[-1]
                groups[-1] = (prev_text + text, prev_start, pos)
        for text, s, e in groups:
            end_time = onsets[e] if e < n else onsets[n - 1]
            words.append((text, offset + onsets[s], offset + end_time))
        i = j
    if merge_punctuation:
        words = _merge_punctuations(words)
    return [(w.strip(), s, e) for w, s, e in words if w.strip()]


_PREPEND = "\"'\u201c\u00bf([{-"
_APPEND = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001"


def _merge_punctuations(
    words: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """OpenAI Whisper's ``merge_punctuations``: opening punctuation joins the
    next word, closing punctuation joins the previous one. Merged words keep the
    timing of the word they join."""
    w = [list(x) for x in words]
    # Prepend, scanning right to left.
    i = len(w) - 2
    j = len(w) - 1
    while i >= 0:
        prev, nxt = w[i], w[j]
        if prev[0].startswith(" ") and prev[0].strip() in _PREPEND:
            nxt[0] = prev[0] + nxt[0]
            prev[0] = ""
        else:
            j = i
        i -= 1
    # Append, scanning left to right.
    i, j = 0, 1
    while j < len(w):
        prev, nxt = w[i], w[j]
        if not prev[0].endswith(" ") and nxt[0] in _APPEND:
            prev[0] = prev[0] + nxt[0]
            nxt[0] = ""
        else:
            i = j
        j += 1
    return [(x[0], x[1], x[2]) for x in w if x[0]]

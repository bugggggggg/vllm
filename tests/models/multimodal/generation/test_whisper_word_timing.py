# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Whisper word timing (QK offload readout)."""

import numpy as np
import pytest
import torch

from vllm.model_executor.models import whisper_word_timing as wt


def _reference_dtw(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenAI Whisper's ``dtw_cpu`` + ``backtrace``, transcribed verbatim."""
    n, m = x.shape
    cost = np.ones((n + 1, m + 1), dtype=np.float32) * np.inf
    trace = -np.ones((n + 1, m + 1), dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    i, j = trace.shape[0] - 1, trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1
    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
    path = np.array(result)[::-1, :].T
    return path[0], path[1]


def _reference_jump_frames(x: np.ndarray) -> np.ndarray:
    text_idx, time_idx = _reference_dtw(x)
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    return time_idx[jumps]


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 3), (12, 40), (30, 200)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dtw_matches_openai(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    np.testing.assert_array_equal(wt.dtw_jump_times(x), _reference_jump_frames(x))


def test_dtw_handles_ties():
    x = np.zeros((6, 9))
    np.testing.assert_array_equal(wt.dtw_jump_times(x), _reference_jump_frames(x))


def test_median_filter_matches_sort_median():
    x = torch.randn(3, 4, 50)
    out = wt.median_filter(x, 7)
    xp = torch.nn.functional.pad(x, (3, 3), mode="reflect")
    ref = xp.unfold(-1, 7, 1).sort(dim=-1).values[..., 3]
    torch.testing.assert_close(out, ref)
    # Too short to reflect-pad: returned unchanged.
    short = torch.randn(2, 3)
    assert wt.median_filter(short, 7) is short


def test_token_onsets_shape_and_monotonic():
    torch.manual_seed(0)
    n_pos, frames, heads, dim, prompt_len = 20, 1500, 6, 64, 4
    # Plant a diagonal: query p attends to frame 10 * p.
    k = torch.randn(frames, heads, dim) * 0.1
    q = torch.zeros(n_pos, heads, dim)
    for p in range(n_pos):
        q[p] = k[10 * p] * 50
    onsets = wt.token_onsets(q, k, num_frames=300, prompt_len=prompt_len)
    assert len(onsets) == n_pos - prompt_len + 1
    assert all(b >= a for a, b in zip(onsets, onsets[1:]))


class _FakeTokenizer:
    """Maps token ids to fixed strings; ids >= 100 are special."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.vocab[i] for i in ids)


def test_split_words_spaces_punctuation_and_specials():
    tok = _FakeTokenizer(
        {1: " Hello", 2: ",", 3: " wor", 4: "ld", 5: ".", 100: "<|eot|>", 101: "<|1.00|>"}
    )
    ids = [1, 2, 3, 4, 5, 101, 100]
    onsets = [0.0, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2]
    raw = wt.split_words(tok, ids, onsets, eot_token_id=100, merge_punctuation=False)
    assert raw == [
        ("Hello", 0.0, 0.4),
        (",", 0.4, 0.5),
        ("world", 0.5, 0.9),
        (".", 0.9, 1.0),
    ]
    merged = wt.split_words(tok, ids, onsets, eot_token_id=100, offset=10.0)
    assert merged == [("Hello,", 10.0, 10.4), ("world.", 10.5, 10.9)]


def test_split_words_no_eot_uses_last_onset():
    tok = _FakeTokenizer({1: " a", 2: " b"})
    words = wt.split_words(tok, [1, 2], [0.1, 0.3], eot_token_id=100)
    assert words == [("a", 0.1, 0.3), ("b", 0.3, 0.3)]

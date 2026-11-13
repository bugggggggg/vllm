# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Whisper transcription with word-level timestamps (QK offload)."""

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.model_executor.models.whisper_word_timing import split_words

llm = LLM(
    "openai/whisper-large-v3-turbo",
    enable_word_timestamps=True,
    limit_mm_per_prompt={"audio": 1},
)
prompt = {
    "encoder_prompt": {
        "prompt": "",
        "multi_modal_data": {"audio": AudioAsset("mary_had_lamb").audio_and_sample_rate},
    },
    "decoder_prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
}
params = SamplingParams(
    temperature=0, max_tokens=440, extra_args={"word_timestamps": True}
)
out = llm.generate(prompt, params)[0].outputs[0]

tokenizer = llm.get_tokenizer()
print(out.text)
for word, start, end in split_words(
    tokenizer, list(out.token_ids), out.token_timestamps, tokenizer.eos_token_id
):
    print(f"{start:6.2f} {end:6.2f}  {word}")

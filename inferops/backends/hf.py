"""Baseline engine: Hugging Face model with a hand-written decode loop.

We deliberately avoid `model.generate()` so every inference concept is visible:

    prefill  - one forward pass over the whole prompt; fills the KV cache and
               produces the first token. Compute-bound. Its duration is TTFT.
    decode   - one forward pass per new token, feeding only that token plus the
               cached keys/values. Memory-bandwidth-bound. Its duration is TPOT.

What is intentionally naive here (and fixed in later weeks):
    * one request at a time (no batching)      -> week 4: continuous batching
    * KV cache grows as contiguous tensors      -> week 5: paged KV / prefix cache
    * full re-detokenization every step, O(n^2) -> good enough for now
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferops.backends.base import Backend, GenerationParams, Prompt, StreamEvent, iterate_in_thread
from inferops.config import Settings


class HFBackend(Backend):
    name = "hf"

    def __init__(self, settings: Settings):
        if settings.torch_threads:
            torch.set_num_threads(settings.torch_threads)
        self.model_id = settings.model_id
        self.device = torch.device(settings.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
        dtype = getattr(torch, settings.dtype)
        if settings.quantization == "none":
            self.precision = settings.dtype
            self.model = AutoModelForCausalLM.from_pretrained(settings.model_id, dtype=dtype).to(self.device)
        else:
            # Quantized weights are placed by accelerate (device_map), not .to():
            # moving a bitsandbytes model after loading is not supported.
            self.precision = f"bnb-{settings.quantization}"
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_id, dtype=dtype, device_map={"": self.device},
                quantization_config=_bnb_config(settings.quantization, dtype),
            )
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.eos_ids = {i for i in [*eos_ids, self.tokenizer.eos_token_id] if i is not None}
        self._slots = asyncio.Semaphore(settings.max_concurrency)

    def encode(self, messages: list[dict]) -> Prompt:
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):  # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
        return Prompt(token_ids=list(ids))

    async def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[StreamEvent]:
        # Requests queue here. Under load, this wait dominates TTFT.
        async with self._slots:
            async for event in iterate_in_thread(lambda cancel: self._generate(prompt, params, cancel)):
                yield event

    @torch.inference_mode()
    def _generate(self, prompt: Prompt, params: GenerationParams, cancel: threading.Event) -> Iterator[StreamEvent]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(params.seed if params.seed is not None else time.time_ns())

        # Prefill: process all prompt tokens in parallel, build the KV cache.
        input_ids = torch.tensor([prompt.token_ids], device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values

        generated: list[int] = []
        emitted = ""
        finish_reason = "length"
        while len(generated) < params.max_tokens:
            if cancel.is_set():
                finish_reason = "cancelled"
                break
            next_id = _sample(out.logits[0, -1], params.temperature, params.top_p, generator)
            if next_id in self.eos_ids and not params.ignore_eos:
                finish_reason = "stop"
                break
            generated.append(next_id)

            # Decode the whole sequence and emit only the new suffix: tokens are
            # not characters, and one character may span several tokens.
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if not text.endswith("�"):  # wait for an incomplete UTF-8 sequence
                yield StreamEvent(text=text[len(emitted):])
                emitted = text

            # Decode step: feed one token; attention reads the cached K/V of all
            # previous tokens instead of recomputing them.
            out = self.model(input_ids=torch.tensor([[next_id]], device=self.device), past_key_values=past, use_cache=True)
            past = out.past_key_values

        yield StreamEvent(
            finish_reason=finish_reason,
            prompt_tokens=len(prompt.token_ids),
            completion_tokens=len(generated),
        )


def _bnb_config(mode: str, compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    if mode == "int8":
        # LLM.int8(): int8 matmul, with outlier feature columns kept in FP16.
        return BitsAndBytesConfig(load_in_8bit=True)
    # NF4: 4-bit codes on a normal-distribution grid (weights are roughly
    # Gaussian), one scale per 64 weights; dequantized to compute_dtype per matmul.
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=compute_dtype)


def _sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    logits = logits.float()
    if temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        # Keep the smallest set of tokens whose cumulative probability reaches top_p.
        sorted_probs[sorted_probs.cumsum(-1) - sorted_probs > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return int(torch.multinomial(probs, 1, generator=generator))

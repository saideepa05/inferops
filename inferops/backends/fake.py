"""Model-free backend that mimics an engine's timing profile.

Prefill cost scales with prompt length, decode cost is per token, and requests
queue behind a fixed number of slots. Lets you test the API and load generator
in milliseconds, and sanity-check that the benchmark measures what you expect.
"""

import asyncio
from collections.abc import AsyncIterator

from inferops.backends.base import Backend, GenerationParams, Prompt, StreamEvent
from inferops.config import Settings


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, settings: Settings):
        self.model_id = "fake-model"
        self.prefill_s = settings.fake_prefill_ms_per_token / 1000
        self.decode_s = settings.fake_decode_ms_per_token / 1000
        self._slots = asyncio.Semaphore(settings.max_concurrency)

    def encode(self, messages: list[dict]) -> Prompt:
        words = " ".join(m["content"] for m in messages).split()
        return Prompt(token_ids=list(range(len(words))))

    async def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[StreamEvent]:
        async with self._slots:
            await asyncio.sleep(self.prefill_s * len(prompt.token_ids))
            for i in range(params.max_tokens):
                if i:
                    await asyncio.sleep(self.decode_s)
                yield StreamEvent(text=f"tok{i} ")
            yield StreamEvent(
                finish_reason="length",
                prompt_tokens=len(prompt.token_ids),
                completion_tokens=params.max_tokens,
            )

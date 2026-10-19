"""Backend interface: every engine (HF baseline, vLLM, our own) implements this."""

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Prompt:
    token_ids: list[int]


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False


@dataclass
class StreamEvent:
    """One decoded text delta. The last event carries finish_reason and usage."""

    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(ABC):
    name: str
    model_id: str
    precision: str = "n/a"  # e.g. "float16", "bnb-nf4"; recorded in benchmark reports

    @abstractmethod
    def encode(self, messages: list[dict]) -> Prompt:
        """Apply the chat template and tokenize. Cheap; runs before streaming starts."""

    @abstractmethod
    def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[StreamEvent]:
        """Yield text deltas as tokens are generated."""


async def iterate_in_thread(gen_factory: Callable[[threading.Event], Iterator[T]]) -> AsyncIterator[T]:
    """Run a blocking generator in a worker thread and consume it asynchronously.

    PyTorch forward passes block, so running them on the event loop would freeze
    the whole server (health checks, other clients, /metrics). The worker gets a
    cancel event it must check between steps, so a client disconnect stops
    generation instead of wasting compute on tokens nobody will read.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    done = object()

    def worker() -> None:
        try:
            for item in gen_factory(cancel):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # surface model errors to the request
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while (item := await queue.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()
        # Wait for the current forward pass to finish so the next request
        # never shares the model with a zombie generation.
        await asyncio.to_thread(thread.join)

from inferops.backends.base import Backend, GenerationParams, Prompt, StreamEvent
from inferops.config import Settings


def load_backend(settings: Settings) -> Backend:
    if settings.backend == "fake":
        from inferops.backends.fake import FakeBackend

        return FakeBackend(settings)
    from inferops.backends.hf import HFBackend

    return HFBackend(settings)


__all__ = ["Backend", "GenerationParams", "Prompt", "StreamEvent", "load_backend"]

"""OpenAI-compatible request schema (the subset we support)."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    # vLLM extension: keep decoding past EOS so benchmarks get fixed-length outputs.
    ignore_eos: bool = False

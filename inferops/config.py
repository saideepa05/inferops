"""Runtime configuration, read from INFEROPS_* environment variables (or .env)."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_", env_file=".env", protected_namespaces=())

    backend: Literal["hf", "fake"] = "hf"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    # float32 on CPU: most laptop CPUs have no fast bf16/fp16 matmul, so lower
    # precision is often *slower* there. On a T4 GPU use float16 (no bf16 support).
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    torch_threads: int | None = None
    # Weight-only quantization via bitsandbytes (CUDA only). Weights are stored
    # in 8 or 4 bits and dequantized to `dtype` inside every matmul, so memory
    # drops but speed often does not improve. See docs/week3.md.
    quantization: Literal["none", "int8", "nf4"] = "none"

    # The baseline engine runs one request at a time. Raising this does not
    # batch anything; it just lets requests fight over the same model.
    max_concurrency: int = 1

    max_prompt_tokens: int = 4096
    max_new_tokens_limit: int = 1024

    # Fake backend timings, used for tests and load-generator development.
    fake_prefill_ms_per_token: float = 0.2
    fake_decode_ms_per_token: float = 20.0

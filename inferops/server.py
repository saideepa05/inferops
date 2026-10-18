"""OpenAI-compatible HTTP API.

Run:  uvicorn inferops.server:app --port 8000
Any OpenAI client works against it (base_url="http://localhost:8000/v1").
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import make_asgi_app

from inferops.backends import Backend, GenerationParams, StreamEvent, load_backend
from inferops.config import Settings
from inferops.metrics import instrument
from inferops.schemas import ChatCompletionRequest


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend or load_backend(settings)
        yield

    app = FastAPI(title="InferOps", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    async def health(request: Request):
        b: Backend = request.app.state.backend
        return {"status": "ok", "backend": b.name, "model": b.model_id}

    @app.get("/v1/models")
    async def models(request: Request):
        return {"object": "list", "data": [{"id": request.app.state.backend.model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        b: Backend = request.app.state.backend
        prompt = b.encode([m.model_dump() for m in req.messages])
        if len(prompt.token_ids) > settings.max_prompt_tokens:
            raise HTTPException(400, f"Prompt has {len(prompt.token_ids)} tokens; limit is {settings.max_prompt_tokens}")

        params = GenerationParams(
            max_tokens=min(req.max_tokens, settings.max_new_tokens_limit),
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            ignore_eos=req.ignore_eos,
        )
        events = instrument(b.stream(prompt, params))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _sse(events, completion_id, created, b.model_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        parts, final = [], StreamEvent()
        async for event in events:
            parts.append(event.text)
            if event.finish_reason:
                final = event
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": b.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(parts)},
                "finish_reason": final.finish_reason,
            }],
            "usage": _usage(final),
        }

    return app


async def _sse(events: AsyncIterator[StreamEvent], completion_id: str, created: int, model: str) -> AsyncIterator[str]:
    def chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n"

    yield chunk({"role": "assistant"})
    async for event in events:
        if event.text:
            yield chunk({"content": event.text})
        if event.finish_reason:
            yield chunk({}, event.finish_reason, _usage(event))
    yield "data: [DONE]\n\n"


def _usage(event: StreamEvent) -> dict:
    return {
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.prompt_tokens + event.completion_tokens,
    }


app = create_app()

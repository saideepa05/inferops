import json

import pytest
from fastapi.testclient import TestClient

from inferops.config import Settings
from inferops.server import create_app


@pytest.fixture
def client():
    settings = Settings(backend="fake", fake_prefill_ms_per_token=0, fake_decode_ms_per_token=0,
                        max_prompt_tokens=50, max_new_tokens_limit=8)
    with TestClient(create_app(settings)) as c:
        yield c


def chat(**overrides) -> dict:
    return {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 3} | overrides


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "backend": "fake", "model": "fake-model",
                                           "precision": "n/a"}


def test_non_streaming_completion(client):
    body = client.post("/v1/chat/completions", json=chat()).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "tok0 tok1 tok2 "
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_streaming_follows_openai_sse_format(client):
    with client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in resp.iter_lines() if line]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "tok0 tok1 tok2 "
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 3


def test_max_tokens_is_capped_by_server_limit(client):
    body = client.post("/v1/chat/completions", json=chat(max_tokens=1000)).json()
    assert body["usage"]["completion_tokens"] == 8


def test_prompt_over_limit_is_rejected_before_generation(client):
    resp = client.post("/v1/chat/completions", json=chat(messages=[{"role": "user", "content": "word " * 51}]))
    assert resp.status_code == 400


@pytest.mark.parametrize("bad", [{"messages": []}, chat(temperature=5), chat(max_tokens=0),
                                 chat(messages=[{"role": "robot", "content": "hi"}])])
def test_invalid_requests_return_422(client, bad):
    assert client.post("/v1/chat/completions", json=bad).status_code == 422


def test_metrics_are_exported(client):
    client.post("/v1/chat/completions", json=chat())
    metrics = client.get("/metrics/").text
    assert "inferops_ttft_seconds_bucket" in metrics
    assert "inferops_completion_tokens_total" in metrics

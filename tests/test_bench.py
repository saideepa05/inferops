import asyncio
import threading
import time

import pytest

from bench.loadgen import RequestResult, summarize
from bench.prompts import build_dataset
from inferops.backends.base import iterate_in_thread


def test_tpot_uses_server_token_count():
    r = RequestResult(ok=True, completion_tokens=11, ttft=0.5, e2e=1.5)
    assert r.tpot == pytest.approx(0.1)


def test_goodput_counts_only_requests_meeting_slo():
    fast = RequestResult(ok=True, completion_tokens=11, ttft=0.1, e2e=0.6)   # TPOT 50 ms
    slow_start = RequestResult(ok=True, completion_tokens=11, ttft=2.0, e2e=2.5)
    slow_decode = RequestResult(ok=True, completion_tokens=11, ttft=0.1, e2e=3.1)  # TPOT 300 ms
    failed = RequestResult(error="boom")
    s = summarize([fast, slow_start, slow_decode, failed], wall=2.0, concurrency=4, slo_ttft=1.0, slo_tpot=0.1)
    assert s["errors"] == 1
    assert s["rps"] == pytest.approx(1.5)
    assert s["goodput_rps"] == pytest.approx(0.5)
    assert s["slo_attainment"] == pytest.approx(0.25)


def test_datasets_are_deterministic_and_sized():
    assert build_dataset("long", n=3, seed=1) == build_dataset("long", n=3, seed=1)
    short = len(build_dataset("short", n=1)[0][0]["content"].split())
    long = len(build_dataset("long", n=1)[0][0]["content"].split())
    assert short < 20 < 1500 <= long


def test_thread_bridge_stops_worker_when_consumer_leaves():
    stopped = threading.Event()

    def produce(cancel):
        i = 0
        while not cancel.is_set():
            yield i
            i += 1
            time.sleep(0.001)
        stopped.set()

    async def consume():
        gen = iterate_in_thread(produce)
        async for item in gen:
            if item == 3:
                break
        await gen.aclose()

    asyncio.run(consume())
    assert stopped.is_set()


def test_thread_bridge_propagates_worker_errors():
    def produce(cancel):
        yield 1
        raise RuntimeError("model crashed")

    async def consume():
        return [x async for x in iterate_in_thread(produce)]

    with pytest.raises(RuntimeError, match="model crashed"):
        asyncio.run(consume())

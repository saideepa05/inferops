"""Server-side Prometheus metrics.

Server-side TTFT excludes network time; the load generator measures the
client-side view. Comparing the two tells you where latency comes from.
"""

import time
from collections.abc import AsyncIterator

from prometheus_client import Counter, Gauge, Histogram

from inferops.backends.base import StreamEvent

_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_TOKEN_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.25, 0.5, 1)

TTFT = Histogram("inferops_ttft_seconds", "Time to first token", buckets=_LATENCY_BUCKETS)
ITL = Histogram("inferops_inter_token_latency_seconds", "Gap between streamed tokens", buckets=_TOKEN_BUCKETS)
E2E = Histogram("inferops_e2e_latency_seconds", "Request latency", buckets=_LATENCY_BUCKETS)
PROMPT_TOKENS = Counter("inferops_prompt_tokens_total", "Prompt tokens processed")
COMPLETION_TOKENS = Counter("inferops_completion_tokens_total", "Tokens generated")
REQUESTS = Counter("inferops_requests_total", "Finished requests", ["finish_reason"])
IN_FLIGHT = Gauge("inferops_requests_in_flight", "Requests queued or generating")


async def instrument(events: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
    start = last = time.perf_counter()
    first = True
    finish_reason = "error"
    IN_FLIGHT.inc()
    try:
        async for event in events:
            now = time.perf_counter()
            if event.text:
                if first:
                    TTFT.observe(now - start)
                    first = False
                else:
                    ITL.observe(now - last)
                last = now
            if event.finish_reason:
                finish_reason = event.finish_reason
                PROMPT_TOKENS.inc(event.prompt_tokens)
                COMPLETION_TOKENS.inc(event.completion_tokens)
            yield event
    except BaseException:
        finish_reason = "cancelled" if finish_reason == "error" else finish_reason
        raise
    finally:
        IN_FLIGHT.dec()
        E2E.observe(time.perf_counter() - start)
        REQUESTS.labels(finish_reason).inc()

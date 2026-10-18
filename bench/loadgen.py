"""Load generator: closed-loop benchmark of any OpenAI-compatible endpoint.

Runs N requests at each concurrency level and reports client-side TTFT, TPOT,
inter-token latency, throughput and goodput (requests meeting the SLO).

    python -m bench.loadgen --url http://127.0.0.1:8000 --dataset short --concurrency 1,4,8

"Closed loop" means each simulated user sends its next request only after the
previous one finishes. Week 7 adds open-loop (fixed arrival rate) traffic.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from bench.prompts import build_dataset


@dataclass
class RequestResult:
    ok: bool = False
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft: float | None = None
    e2e: float | None = None
    itl: list[float] = field(default_factory=list)

    @property
    def tpot(self) -> float | None:
        # Uses the server's token count, not SSE chunk count: one chunk can
        # carry several tokens (e.g. a multi-token emoji).
        if self.ttft is None or self.e2e is None or self.completion_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.completion_tokens - 1)


async def send_request(client: httpx.AsyncClient, url: str, messages: list[dict], max_tokens: int,
                       ignore_eos: bool) -> RequestResult:
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
               "stream": True, "ignore_eos": ignore_eos}
    result = RequestResult()
    start = last = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {(await resp.aread()).decode()[:200]}"
                return result
            # Read to the end of the body (don't break on [DONE]): a half-read
            # response can't go back to the connection pool, so every request
            # would pay a fresh TCP connect and inflate TTFT.
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                    if result.ttft is None:
                        result.ttft = now - start
                    else:
                        result.itl.append(now - last)
                    last = now
                if usage := chunk.get("usage"):
                    result.prompt_tokens = usage["prompt_tokens"]
                    result.completion_tokens = usage["completion_tokens"]
        result.e2e = time.perf_counter() - start
        result.ok = True
    except httpx.HTTPError as exc:
        result.error = repr(exc)
    return result


async def run_level(url: str, prompts: list[list[dict]], concurrency: int, num_requests: int,
                    max_tokens: int, ignore_eos: bool, timeout: float) -> tuple[list[RequestResult], float]:
    jobs = iter(prompts[i % len(prompts)] for i in range(num_requests))
    results: list[RequestResult] = []
    limits = httpx.Limits(max_connections=concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def user() -> None:
            for messages in jobs:  # shared iterator: each user pulls the next job
                results.append(await send_request(client, url, messages, max_tokens, ignore_eos))

        start = time.perf_counter()
        await asyncio.gather(*(user() for _ in range(concurrency)))
        wall = time.perf_counter() - start
    return results, wall


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values)
    return {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99)} | {"mean": float(arr.mean())}


def summarize(results: list[RequestResult], wall: float, concurrency: int,
              slo_ttft: float, slo_tpot: float) -> dict:
    ok = [r for r in results if r.ok and r.ttft is not None]
    good = [r for r in ok if r.ttft <= slo_ttft and (r.tpot is None or r.tpot <= slo_tpot)]
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall,
        "rps": len(ok) / wall,
        "goodput_rps": len(good) / wall,
        "slo_attainment": len(good) / len(results) if results else 0.0,
        "output_tok_s": sum(r.completion_tokens for r in ok) / wall,
        "input_tok_s": sum(r.prompt_tokens for r in ok) / wall,
        "ttft_s": _percentiles([r.ttft for r in ok]),
        "tpot_s": _percentiles([r.tpot for r in ok if r.tpot is not None]),
        "itl_s": _percentiles([x for r in ok for x in r.itl]),
        "e2e_s": _percentiles([r.e2e for r in ok]),
    }


def print_table(levels: list[dict]) -> None:
    ms = lambda d, k: f"{d[k] * 1000:8.0f}" if k in d else "       -"  # noqa: E731
    print(f"{'conc':>5} {'rps':>6} {'out tok/s':>9} {'TTFT p50':>8} {'p95':>8} {'p99':>8} "
          f"{'TPOT p50':>8} {'p95':>8} {'goodput':>8} {'SLO%':>5} {'err':>4}")
    for s in levels:
        print(f"{s['concurrency']:>5} {s['rps']:6.2f} {s['output_tok_s']:9.1f} "
              f"{ms(s['ttft_s'], 'p50')} {ms(s['ttft_s'], 'p95')} {ms(s['ttft_s'], 'p99')} "
              f"{ms(s['tpot_s'], 'p50')} {ms(s['tpot_s'], 'p95')} "
              f"{s['goodput_rps']:8.2f} {s['slo_attainment'] * 100:5.0f} {s['errors']:>4}")
    print("(latencies in ms; goodput in requests/s meeting the SLO)")


async def main(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    async with httpx.AsyncClient(timeout=10) as client:
        server_info = (await client.get(f"{url}/health")).json()
    prompts = build_dataset(args.dataset, n=max(args.num_requests, 50), seed=args.seed)

    for _ in range(args.warmup):  # first request pays one-off costs (lazy init, caches)
        await run_level(url, prompts, 1, 1, args.max_tokens, args.ignore_eos, args.timeout)

    levels, raw = [], {}
    for c in args.concurrency:
        results, wall = await run_level(url, prompts, c, args.num_requests, args.max_tokens,
                                        args.ignore_eos, args.timeout)
        levels.append(summarize(results, wall, c, args.slo_ttft_ms / 1000, args.slo_tpot_ms / 1000))
        raw[c] = [asdict(r) for r in results]
        print(f"concurrency={c}: {levels[-1]['rps']:.2f} req/s, {levels[-1]['errors']} errors")

    print()
    print_table(levels)

    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "server": server_info,
            "label": args.label,
            **{k: getattr(args, k) for k in ("dataset", "num_requests", "max_tokens", "ignore_eos",
                                            "slo_ttft_ms", "slo_tpot_ms", "seed")},
        },
        "levels": levels,
        "requests": raw,
    }
    out = Path(args.out or f"results/{datetime.now():%Y%m%d-%H%M%S}_{args.label or server_info['backend']}_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and
    # falls back to IPv4 after ~250 ms, which shows up as fake TTFT.
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--dataset", default="short", choices=["short", "medium", "long", "mixed"])
    p.add_argument("--concurrency", default="1,4,8", type=lambda s: [int(x) for x in s.split(",")])
    p.add_argument("--num-requests", type=int, default=32, help="requests per concurrency level")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                   help="force exactly max-tokens outputs so runs are comparable")
    p.add_argument("--slo-ttft-ms", type=float, default=1000)
    p.add_argument("--slo-tpot-ms", type=float, default=100)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="", help="name for this run, e.g. hf-fp32-cpu")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

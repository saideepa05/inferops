"""Quality check: GSM8K grade-school math accuracy through any OpenAI-compatible API.

Speed numbers alone can't justify an optimization; quantization, speculative
decoding or a new engine must also show they didn't make the answers worse.

    python -m bench.quality --url http://127.0.0.1:8000 --n 200 --label hf-fp16

Why GSM8K: multi-step arithmetic is fragile. One wrong digit anywhere in the
chain gives a wrong final answer, so it surfaces precision loss that a "does the
text look fine?" check would miss. Answers are a single number, so scoring is
exact and automatic.

The same first-n questions are used for every run (no shuffling), so runs are
paired: differences come from the model, not from easier questions.
"""

import argparse
import asyncio
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bench.loadgen import get_server_info

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SYSTEM = "Solve the math problem step by step. End your reply with: The answer is <number>."
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def load_gsm8k(n: int, cache: Path = Path("results/cache/gsm8k_test.jsonl")) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(httpx.get(GSM8K_URL, timeout=60, follow_redirects=True).raise_for_status().text,
                         encoding="utf-8")
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Gold answers end with "#### <number>" after the worked solution.
    return [{"question": r["question"], "gold": r["answer"].split("####")[-1].strip()} for r in rows[:n]]


def _to_number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Prefer the requested 'The answer is N'; otherwise take the last number in the reply."""
    if m := re.search(rf"answer is[:\s]*\**\s*({_NUMBER})", text, re.IGNORECASE):
        return _to_number(m.group(1))
    numbers = re.findall(_NUMBER, text)
    return _to_number(numbers[-1]) if numbers else None


def is_correct(pred: float | None, gold: str) -> bool:
    g = _to_number(gold)
    return pred is not None and g is not None and math.isclose(pred, g, rel_tol=0, abs_tol=1e-6)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy of k/n.

    With n=200, an accuracy of 80% is really "somewhere in 74-85%". Two configs
    whose intervals overlap heavily are not shown to differ by this test.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


async def ask(client: httpx.AsyncClient, url: str, model: str, question: str, max_tokens: int) -> dict:
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]}
    try:
        resp = await client.post(f"{url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        return {"output": body["choices"][0]["message"]["content"],
                "finish_reason": body["choices"][0].get("finish_reason"),
                "completion_tokens": body["usage"]["completion_tokens"], "error": None}
    except (httpx.HTTPError, KeyError) as exc:
        return {"output": "", "finish_reason": None, "completion_tokens": 0, "error": repr(exc)}


async def evaluate(url: str, items: list[dict], concurrency: int, max_tokens: int, timeout: float) -> list[dict]:
    model = (await get_server_info(url))["model"]
    slots = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=concurrency)) as client:
        async def one(i: int, item: dict) -> dict:
            async with slots:
                r = await ask(client, url, model, item["question"], max_tokens)
            pred = extract_answer(r["output"])
            return {"id": i, **item, **r, "pred": pred, "correct": is_correct(pred, item["gold"])}

        return await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))


async def main(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    server_info = await get_server_info(url)
    items = load_gsm8k(args.n)
    start = time.perf_counter()
    records = await evaluate(url, items, args.concurrency, args.max_tokens, args.timeout)
    wall = time.perf_counter() - start

    k, n = sum(r["correct"] for r in records), len(records)
    lo, hi = wilson_interval(k, n)
    summary = {
        "accuracy": k / n, "ci95": [lo, hi], "correct": k, "n": n,
        "errors": sum(r["error"] is not None for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in records) / n,
        "wall_s": wall,
    }
    print(f"{args.label or server_info.get('model')}: accuracy {k}/{n} = {k / n:.1%} "
          f"(95% CI {lo:.1%}-{hi:.1%}), {summary['truncated']} truncated, "
          f"{summary['errors']} errors, {wall:.0f}s")

    out = Path(args.out or f"results/{datetime.now():%Y%m%d-%H%M%S}_quality_{args.label or 'server'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "url": url, "server": server_info,
                 "label": args.label, "benchmark": "gsm8k-test", "max_tokens": args.max_tokens},
        "summary": summary, "records": records}, indent=2))
    print(f"Saved {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=200, help="first n GSM8K test questions")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

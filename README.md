# InferOps

A production-style LLM serving platform, built to find and fix the problems that
show up when real traffic hits a model: queueing under load, unpredictable
latency on long prompts, and GPU cost.

Every optimization has to prove itself with a controlled benchmark. Claims like
"faster" need a before/after report showing TTFT, TPOT, goodput, memory and quality.

## Architecture

```
 load generator ──►  FastAPI gateway (OpenAI-compatible /v1/chat/completions, SSE streaming)
                         │
                         ▼
                  Backend interface ── encode() / stream()
                    ├── fake      timing simulator (tests, laptop dev)
                    ├── hf        baseline: hand-written prefill/decode loop
                    ├── vllm      week 4: continuous batching, paged KV cache
                    └── nano      engine track: our own batching scheduler
                         │
                         ▼
                  /metrics (Prometheus) ──► Grafana (week 6)
```

Using the OpenAI API format means any OpenAI SDK, vLLM, llama.cpp or SGLang
server can be benchmarked with the same tool. Swapping engines doesn't touch
the client.

## Roadmap

| Week | Focus | Deliverable | Runs on |
|---|---|---|---|
| 1 | Baseline server, streaming, load generator | Working endpoint and benchmark tool | Laptop |
| 2 | Benchmark and find bottlenecks | Baseline report: TTFT/TPOT percentiles across concurrency and prompt length | Kaggle T4 |
| 3 | Precision: FP16 vs INT8 vs 4-bit (AWQ/GPTQ) | Quality-vs-speed table and decision record | Kaggle T4 |
| 4 | vLLM, continuous batching, scheduler tuning | Naive vs batched goodput comparison | Kaggle T4 |
| 5 | Long context: KV memory, prefix caching, chunked prefill | Cache hit/miss study | Kaggle T4 |
| 6 | Reliability: auth, rate limits, timeouts, load shedding, dashboards | Failure tests and Grafana dashboard | Laptop |
| 7 | Cost and capacity: open-loop traffic, $/1M tokens, online vs batch | Capacity plan | Kaggle T4 |
| 8 | CI with performance-regression gates, release, rollback | Versioned release and architecture doc | GitHub Actions |
| Engine track | Write KV cache, continuous batching and speculative decoding yourself | `nano` backend | Laptop |

## Quickstart

```powershell
cd inferops
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

**Fake engine** (no model; about 100 MB RAM):

```powershell
$env:INFEROPS_BACKEND = "fake"
.\.venv\Scripts\uvicorn inferops.server:app --port 8000
# second terminal
.\.venv\Scripts\python -m bench.loadgen --dataset mixed --concurrency 1,4,8 --num-requests 16 --max-tokens 32
```

**Real model on a laptop CPU** (tiny model, capped threads, about 1 GB RAM):

```powershell
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -e ".[hf]"
$env:INFEROPS_BACKEND = "hf"
$env:INFEROPS_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
$env:INFEROPS_TORCH_THREADS = "4"
.\.venv\Scripts\uvicorn inferops.server:app --port 8000
```

**Kaggle GPU:** import a notebook from [notebooks/](notebooks/)
([week 2 baseline](notebooks/week2_baseline_gpu.ipynb), [week 3 quantization](notebooks/week3_quantization_gpu.ipynb))
into a Kaggle notebook with GPU T4 x2 and Internet enabled. On a T4, use
`INFEROPS_DTYPE=float16`; T4s have no bfloat16 support. 8-bit and 4-bit weights:
`INFEROPS_QUANTIZATION=int8|nf4` (needs `pip install -e ".[hf,quant]"`).

**Quality check** (GSM8K math accuracy with a 95% confidence interval, against any OpenAI-compatible server):

```powershell
.\.venv\Scripts\python -m bench.quality --url http://127.0.0.1:8000 --n 200 --label my-run
```

## Configuration

All settings are environment variables prefixed `INFEROPS_`; see
[inferops/config.py](inferops/config.py).

## Repository layout

```
inferops/            serving code
  server.py          HTTP API and SSE streaming
  backends/          engine implementations behind one interface
  metrics.py         Prometheus metrics
bench/               load generator, prompt datasets, GSM8K quality check
tests/               API contract and benchmark-math tests
docs/                weekly learning notes and engineering decision records
results/             benchmark JSON files (committed when cited in a report)
```

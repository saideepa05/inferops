# Week 4: Continuous batching

**Question:** how many users can one T4 serve within the SLO, and what does batching buy you?

Run: [notebooks/week4_batching_gpu.ipynb](../notebooks/week4_batching_gpu.ipynb) on Kaggle (T4 x2, about 60-75 GPU minutes).

## The idea in one line

A decode step reads every weight once whether it produces 1 token or 64, so decoding 64 requests together costs
almost the same as decoding one. Our HF engine wastes that: it runs one request while the rest queue.

## Static vs continuous batching

| | No batching (our HF engine) | Static batching | Continuous batching (vLLM) |
|---|---|---|---|
| Unit of scheduling | one request | a fixed batch, start to finish | one decode step |
| New request arrives | waits for every earlier request | waits for the whole batch to finish | joins at the next step |
| A request finishes early | n/a | its slot sits idle until the batch ends | its slot is refilled at the next step |
| TTFT under load | grows with queue length | grows with the longest request in the batch | stays low until the GPU is saturated |

Continuous batching only works if KV-cache memory can be handed out per request without fragmenting, which is
what **PagedAttention** does: the cache lives in fixed-size blocks, like virtual-memory pages. Week 5 goes deeper.

## What else changes from week 3

- **CUDA graphs on.** Week 3 ran vLLM with `--enforce-eager` by mistake. In eager mode, Python launches every
  kernel of every layer on every step; with a 3B model at small batch sizes the GPU spends much of each step
  waiting. A CUDA graph records the step once at startup and replays it with one launch. The cost is startup time.
- **Prefix caching off.** vLLM enables it by default, and our load generator reuses the same prompts at every
  concurrency level, so later levels skipped prefill and TTFT looked too good. Week 5 measures it on purpose.
- **Warm-up at full concurrency.** Triton compiles some attention kernels the first time it sees a batch shape.
  A throwaway pass at concurrency 128 keeps that out of the measurements.

## The scheduler knobs

- `--max-num-seqs`: most requests decoded together. Lower it and TPOT drops while queued requests wait longer for
  their first token. It's the TTFT-vs-TPOT dial.
- `--max-num-batched-tokens`: token budget per step, shared between prefill chunks and decode tokens. Matters
  with long prompts (chunked prefill, week 5).

## Goodput and capacity

Raw throughput keeps climbing past the point where users would give up. **Goodput** counts only requests that met
the SLO (TTFT <= 1 s and TPOT <= 100 ms here). **Capacity** in this project is the highest concurrency at which
at least 90% of requests meet it. Capacity planning (week 7) starts from that number.

## Trusting the benchmark

Part C runs `vllm bench serve` against the same server as our load generator. Same lengths, same concurrency,
same closed loop, same TPOT definition. If the two agree within a few percent, earlier results stand. If not,
find out why before publishing any number.

## Concepts to be able to explain

1. Why batching decode is almost free until the GPU becomes compute-bound, and roughly where that happens.
2. Static vs continuous batching, and why continuous batching needs paged KV memory.
3. What CUDA graphs remove, and why the gain is largest at small batch sizes.
4. Goodput vs throughput, and why capacity must be stated against an SLO.
5. The `--max-num-seqs` tradeoff between TTFT and TPOT.

## Interview questions

- "Throughput doubled but users complain the app got slower. What happened?"
- "Explain continuous batching to someone who knows static batching."
- "How would you find the right `max-num-seqs` for a chat product?"
- "How do you know your load generator is measuring correctly?"

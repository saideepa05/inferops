# Week 1: Baseline server and load generator

## What you built

1. An OpenAI-compatible API with SSE streaming ([server.py](../inferops/server.py))
2. A baseline engine with a hand-written prefill/decode loop ([backends/hf.py](../inferops/backends/hf.py))
3. A timing simulator for model-free testing ([backends/fake.py](../inferops/backends/fake.py))
4. A load generator that reports TTFT, TPOT, throughput and goodput ([bench/loadgen.py](../bench/loadgen.py))

## Concepts, and where they live in the code

| Concept | Where to look | What to notice |
|---|---|---|
| Chat template and tokenization | `HFBackend.encode` | The model never sees your messages, only token IDs wrapped in special tokens like `<\|im_start\|>`. Print `tokenizer.decode(prompt.token_ids)` to see what it actually receives. |
| Prefill | the first `self.model(...)` call in `_generate` | One forward pass over all prompt tokens. Its duration is TTFT. |
| KV cache | `past_key_values` | Each decode step feeds **one** token. Attention reuses the cached keys and values of every earlier token. Without the cache, step *n* would recompute *n* tokens. |
| Decode | the `while` loop | One forward pass per token, each reading every weight from memory, which makes it bandwidth-bound. Its duration is TPOT. |
| Sampling | `_sample` | Temperature reshapes the distribution; top-p truncates it. `temperature=0` is greedy decoding. |
| Incremental detokenization | `text.endswith("�")` | Tokens are not characters. One emoji can be 2–3 tokens, so we hold output until the UTF-8 sequence is complete. |
| Blocking work and async servers | `iterate_in_thread` | A forward pass blocks. Running it on the event loop would freeze `/health` and every other client. |
| Cancellation | the `cancel` event | If a client disconnects, stop generating. Otherwise you burn GPU time on tokens nobody reads. |
| Queueing | `asyncio.Semaphore(max_concurrency)` | The baseline serves one request at a time; everyone else waits here. |

## First result: what serial serving does under load

Fake engine, mixed prompts, 32 output tokens (laptop):

```
 conc    rps out tok/s TTFT p50      p95      p99 TPOT p50  goodput  SLO%
    1   1.02      32.5       32      131      280       30     1.02   100
    4   1.02      32.7     2926     3152     3162       30     0.06     6
    8   1.02      32.5     6683     6987     7020       30     0.06     6
```

- **Throughput is flat.** More users don't add capacity, because the engine
  handles one request at a time.
- **TTFT grows linearly with concurrency.** It's time spent in the queue, not compute.
- **TPOT stays constant.** Once a request gets the engine, it decodes at full speed.
- **Goodput collapses** from 100% to 6%, even though the server never errored.
  That's why goodput matters more than raw throughput.

Continuous batching (week 4) is the fix: decode many requests in one forward
pass, because decode is bandwidth-bound and leaves compute idle.

## Two measurement bugs the benchmark caught

Both are worth telling in an interview, because they show you validate your tools.

1. **About 260 ms of fake TTFT on every request.** On Windows, `localhost`
   resolves to IPv6 first and falls back to IPv4 only after a timeout. The
   client also exited the stream at `[DONE]` without reading the end of the
   response, so the connection couldn't be reused and every request paid for
   a fresh connect. Fix: use `127.0.0.1` and read every response to the end.
   TTFT dropped from 262 ms to 7 ms.
2. **TPOT was 30 ms when the fake engine was set to 20 ms.** Windows timers
   have roughly 15 ms granularity, so `asyncio.sleep(0.02)` really sleeps
   about 25–31 ms. Lesson: calibrate the harness against a known input before
   trusting it.

## Exercises (do these yourself)

1. **See the chat template.** In a Python shell, load the tokenizer and print
   `tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)`.
2. **Remove the KV cache.** Copy `_generate` and, each step, feed the full
   sequence with no `past_key_values`. Plot TPOT against output length for both
   versions. Without the cache, TPOT should grow with output length; with it,
   TPOT stays roughly flat.
3. **Separate prefill and decode time.** Time the first forward pass and the
   loop separately for short and long prompts. Which one scales with prompt length?
4. **Change the queue.** Set `INFEROPS_MAX_CONCURRENCY=4` on the fake backend
   and rerun the benchmark. Then ask: why would this *not* help the real HF
   backend? (Hint: four threads sharing one model on one CPU.)
5. **Client-side vs server-side TTFT.** Compare the load generator's TTFT with
   `inferops_ttft_seconds` at `/metrics/`. Where does the difference come from?

## Interview questions you should now be able to answer

- Why is prefill compute-bound and decode bandwidth-bound?
- What does the KV cache store, and why does it grow with sequence length?
- Why can two systems with the same throughput feel completely different to users?
- Why report p95/p99 instead of the mean?
- What happens to a streaming request when the client disconnects?

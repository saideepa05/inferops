# Week 3: Quantization

**Question:** if each weight takes fewer bytes, is the model faster and cheaper, and what does it cost in quality?

Run: [notebooks/week3_quantization_gpu.ipynb](../notebooks/week3_quantization_gpu.ipynb) on Kaggle (T4 x2, about 90 GPU minutes).

## The idea in one line

Week 2 showed that decode is **memory-bandwidth bound**: every token reads all 6.2 GB of FP16 weights.
Store weights in 4 bits and the same token reads about 1.6 GB. In theory decode gets ~4x faster
and the freed memory holds more KV cache, which means more concurrent requests.

Whether that theory holds depends on the **kernel**, the code that does the matrix multiply.

## Two ways to run a quantized model

| | bitsandbytes (track A) | AWQ / GPTQ in vLLM (track B) |
|---|---|---|
| When weights are quantized | at load time, from the FP16 checkpoint | offline, once; you download a quantized checkpoint |
| What the matmul does | unpacks 4/8-bit weights to FP16, then runs a normal FP16 matmul | a fused kernel reads packed 4/8-bit weights and dequantizes inside registers |
| Memory | about 4x less (NF4) | about 4x less (4-bit) |
| Speed | often **slower** than FP16: extra unpack work, separate kernels | often **faster**: fewer bytes read per token, no extra pass |
| Built for | fitting a big model on a small GPU, fine-tuning (QLoRA) | serving |

Formats you'll see:
- **INT8 (LLM.int8())**: 8-bit weights; outlier feature columns are kept in FP16 because a few huge values would wreck the scale.
- **NF4**: 4-bit codes placed on a normal-distribution grid (weights are roughly Gaussian), one scale per block of 64 weights.
- **GPTQ**: chooses rounding layer by layer to minimize the error of the layer's *output*, using a small calibration set.
- **AWQ**: finds the ~1% of weight channels that matter most (by activation size) and scales them before rounding, so they lose less precision.

All of these are **weight-only**: activations and the KV cache stay FP16. (FP8 weights plus activations need Ada/Hopper GPUs; a T4 has no FP8.)

## Measuring quality honestly

[bench/quality.py](../bench/quality.py) runs GSM8K math through the API and scores exact final answers.

- **Why math:** one wrong digit anywhere in a multi-step chain makes the final answer wrong, so small precision losses show up.
- **Same questions every run:** the first N test questions, so configs are compared on identical inputs.
- **Greedy decoding** (temperature 0): differences come from the weights, not from sampling luck.
- **Confidence intervals:** 160/200 correct is "74-85%", not "80%". With n=50 the interval is about ±11 points.
  An optimization "lost quality" only if the drop is larger than that noise. To detect a 2-point drop
  you need thousands of questions, or a paired test on the same questions.

## Concepts to be able to explain

1. Why decode speed tracks bytes read per token, and the ceiling: `bandwidth / (params x bytes per param)`.
2. Why bitsandbytes NF4 can be slower than FP16 while AWQ is faster: same storage, different kernels.
3. Weight-only vs weight-and-activation quantization, and why the T4 limits you to the first.
4. How quantization buys concurrency: smaller weights leave more memory for KV cache (watch vLLM's "KV cache size" log line).
5. Why one accuracy number without a confidence interval proves nothing.

## Interview questions

- "We quantized to 4-bit and it got slower. Why, and what would you try?"
- "How do you know a quantized model is good enough to ship?"
- "Your GPU has 16 GB. How does quantization change how many users one GPU can serve?"
- "AWQ vs GPTQ vs bitsandbytes: which would you use for serving, and which for fine-tuning?"

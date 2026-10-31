# Test 29: how many long sessions one GPU holds, a kept chat against AXI
## Pre-registered 31 Oct 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
A GPU that serves a session has to keep that session's prompt in its memory (the "KV cache"). The longer the prompt, the more memory, in a straight line. A kept chat keeps its whole history: in our Gemini 3.8 Flash long run the prompt averaged 476,835 tokens and reached 949,591. AXI sends a fresh prompt of 2,824 tokens a step. This test asks a plain question: **on one GPU, how many sessions at each of those prompt lengths fit before the memory runs out?** The capacity shortage the big cloud companies describe is a shortage of exactly this memory and these chips.

## Read this first: what this test can and cannot say
* **Memory grows in a straight line with tokens, so the result is close to the token ratio (about 169 times) by construction.** This is a measured memory demonstration, not a discovery. Its worth is that the numbers are real allocations on a real GPU, and that anyone can rerun it.
* **Memory only.** It says nothing about speed, throughput or quality. Real servers also evict, offload to the CPU, share a common prefix between sessions and page memory; a real kept-chat deployment would not hold 950,000-token sessions in GPU memory for idle users. The count is "sessions in flight at once".
* **A small model is measured; larger ones are arithmetic.** The real run uses Qwen2.5-0.5B (24 layers, 2 key-value heads, head size 64) so that it fits any Colab GPU. Qwen2.5-7B and Llama-3.1-70B figures are computed from their published shapes and are labelled arithmetic, not measured.
* A 0.5B model does not accept 950,000 tokens; the allocation is of cache-shaped memory of that size, to count what fits. It is not a run of that model on that prompt.

## Setup
* Step 1: a forward pass over 1,024 and 2,824 random tokens; the real bytes of cache are counted and divided by the tokens; compared with the formula 2 x layers x kv heads x head size x 2 bytes (12,288 bytes a token for this model).
* Step 2: for each prompt length (AXI step 2,824; kept chat at 50,000, 100,000, 250,000; the mean 476,835; the largest 949,591) real 16-bit tensors of the cache shape are allocated, session after session, until the GPU raises out of memory; the number held is written down. The formula's number for the same free memory is saved next to it.
* The GPU is whatever Colab gives (its name is saved). One run per length. Results are in `docs/results/gpu_sessions.json`.

## Predictions (written before the run; probability that the line reads YES)
1. The real bytes a token costs are within 2 % of the formula at both lengths: **90 %**.
2. The sessions held at the AXI length are at least 100 times those held at the kept-chat mean: **85 %**.
3. For every length, the number held is within 5 % of the formula's number: **80 %**.
4. At the largest kept-chat length (949,591 tokens) the GPU holds at most one session: **90 %**.
5. The code runs to the end on the first try on Colab without a change: **65 %** (it has not run on a GPU yet; the cache reading was written to handle several versions of the library).

## How the result will be read
The counts come from the saved JSON, recomputed by a test. Each line reads YES or NO. The arithmetic table for 7B and 70B models is shown separately and marked as arithmetic. No claim is made about throughput, latency or real deployments.

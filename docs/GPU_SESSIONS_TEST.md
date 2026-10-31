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

## Result
*Run on 31 Oct 2026 in Colab on a Tesla T4 (15.47 GB free after the model was removed), model Qwen2.5-0.5B-Instruct. Figures are recomputed by a test from `docs/results/gpu_sessions.json`.*

**Real cost of a token:** 12,288 bytes at 1,024 tokens and at 2,824 tokens, exactly the formula (2 x 24 layers x 2 kv heads x 64 x 2 bytes).

| prompt kept per session | tokens | sessions held before out of memory (real allocation) | by the formula |
|---|---|---|---|
| AXI step | 2,824 | **430** | 445 |
| kept chat at 50k | 50,000 | 23 | 25 |
| kept chat at 100k | 100,000 | 12 | 12 |
| kept chat at 250k | 250,000 | 4 | 5 |
| kept chat, mean of the long run | 476,835 | **2** | 2 |
| kept chat, largest of the long run | 949,591 | **1** | 1 |

### The five lines
1. Bytes a token costs within 2 % of the formula: **YES** (exact).
2. Sessions at the AXI length at least 100 times those at the kept-chat mean: **YES** (430 against 2 = 215 times; see the note below).
3. Every length within 5 % of the formula's count: **NO**. The AXI length held 430 of 445 (3.4 % under), but kept chat at 50k held 23 of 25 (8 % under) and at 250k 4 of 5 (20 % under). The allocator keeps a little memory for itself, and with counts this small one session is a big percentage.
4. At the largest kept-chat length at most one session: **YES** (exactly one).
5. The code ran to the end on the first try on Colab without a change: **YES**.

4 of 5 lines read YES, 1 reads NO.

### What this says, and what it does not
* On one T4, a kept chat at the long run's mean length held **2 sessions** and at its largest length **1**, while AXI-sized prompts held **430**. In memory, that is the token ratio: 476,835 / 2,824 = 169 times. The 215 times above is larger only because the kept-chat count is rounded down from 2.64 to 2 whole sessions. I would quote **about 169 times in memory**, and not 215.
* By formula (arithmetic, not measured): Qwen2.5-7B on an 80 GB GPU with 15 GB of weights, 401 AXI-sized sessions against 2 at the kept-chat mean and 1 at the largest. Llama-3.1-70B on 8 x 80 GB with 140 GB of weights: 540 against 3 and 1. Both assume 16-bit cache.
* **This is memory only.** It does not show speed, quality or throughput. Real servers evict, offload to the CPU, share a prefix between sessions and page memory, and would not keep idle 950,000-token sessions in GPU memory. The count means sessions in flight at once. A 0.5B model does not accept 950,000 tokens; the memory was allocated to count what fits.
* What it does support: the part of the capacity shortage that comes from keeping long prompts in GPU memory is a direct function of prompt length, and AXI's flat prompt is a direct cut to it. That follows from the measured prompt sizes (2,824 against 476,835 on average) and a memory formula we verified on a real GPU.

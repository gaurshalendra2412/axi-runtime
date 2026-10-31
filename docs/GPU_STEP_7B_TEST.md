# Test 33: the whole step on an L4, with the prefill that Test 32 left out
## Pre-registered 1 Nov 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
Test 32 timed only the decode part of a step (the chip making reply tokens) and found 366 tokens a second for 32 AXI sessions against 8.6 for one kept-chat session at 100,000 tokens. It left out something that works against AXI: **AXI sends a fresh prompt of 2,824 tokens every step, so every step starts with a prefill of those tokens, and prefill is compute-bound.** A kept chat that reuses its cache only has to prefill its newest few tokens. This test times the prefill on the same L4 with the same 7B-shape model, adds it to the decode times saved in Test 32, and gives the chip time of a whole step: prefill, then R reply tokens.

## Read this first
* **Estimate by parts** from two saved measurements (Test 32's decode steps and this test's prefill times), the same method as before; not a serving engine. A real engine overlaps prefill with decode.
* The prefill is timed for one prompt at a time, no cache, causal attention. Each session pays its own prefill (compute does not batch away). A prefill is measured up to 32,768 tokens; longer prompts are not extrapolated.
* **Assumptions, not measurements:** a kept chat adds 100 new tokens a step; a step's reply is 10, 40 or 100 tokens (shown for all three, because the reply length decides the answer); a kept chat that reuses its cache pays one extra read of its cache (the extra decode time of its length over the AXI length at one session).
* **Not measured here, and the other way around from what favours AXI:** a hosted service with a cache keeps most of a kept chat's prefill free; a kept chat whose cache was lost (idle too long) pays the prefill of its whole prompt, which here is only a lower bound (32,768 tokens).
* L4, 7B shape, constant weights, one run per point; says nothing about quality or other chips. The Gemini speed figures (3.07 s against 7.13 s a step, first token 2.27 s against 5.06 s) are a different, hosted measurement and are not tested here.
* The code has never run on a GPU.

## Predictions (written before the run; probability that the line reads YES)
1. The code runs to the end on the first try without a change: **60 %**.
2. The prefill of 2,824 tokens takes between 0.3 s and 1.5 s: **80 %**.
3. The prefill of 8,192 tokens takes between 2.5 and 4 times that of 2,824 tokens (about linear): **70 %**.
4. With 40-token replies, AXI at 32 sessions gets at least 3 times the whole steps per chip-second of one kept-chat session at 100,000 tokens that keeps its cache: **70 %**.
5. The same ratio is at least 10 times: **30 %**.
6. The ratio in line 4 grows with the reply length (10 tokens < 40 tokens < 100 tokens): **90 %**.
7. **Against AXI:** with 10-token replies, a kept chat that keeps its cache at 50,000 tokens with 2 sessions needs less chip time per step than AXI at 32 sessions: **60 %**.
8. **Against AXI:** the time to the first reply token at one session is shorter for a kept chat that keeps its cache (at 50,000 and at 100,000 tokens) than for AXI: **75 %**.
9. A kept chat whose cache is lost, at only 32,768 tokens, needs at least 10 times the prefill time of AXI's 2,824 tokens: **85 %**.

## How the result will be read
All figures from `docs/results/gpu_step7b.json` and `docs/results/gpu_decode7b.json`, recomputed by a test with the `combine` and `first_token` functions. Each line reads YES or NO, including the two that go against AXI. No claim about a real serving engine, price or quality.

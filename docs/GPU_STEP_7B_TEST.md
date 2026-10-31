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

## Result
*Run on 31 Oct 2026 in Colab on an NVIDIA L4 (PyTorch 2.11.0), 7B-shape model. Prefill times from `docs/results/gpu_step7b.json`; decode times from `docs/results/gpu_decode7b.json` (Test 32). Every figure below is recomputed by a test with the `combine` and `first_token` functions.*

**Prefill on the L4** (one prompt, no cache): 100 tokens 60 ms; 512 tokens 119 ms; **2,824 tokens 623 ms**; 8,192 tokens 1,957 ms; 32,768 tokens 9,820 ms (the chip did 51 to 69 trillion operations a second, against a published peak of about 121; short prompts are less efficient).

**Chip seconds for one whole step, per session** (prefill, then R reply tokens; the decode steps are shared by the sessions in a batch, the prefill is not):

| | R = 10 | R = 40 | R = 100 |
|---|---|---|---|
| AXI, 1 session | 1.271 s | 3.217 s | 7.107 s |
| AXI, 16 sessions | 0.670 s | 0.811 s | 1.093 s |
| **AXI, 32 sessions** | **0.650 s** | **0.732 s** | **0.896 s** |
| kept chat at 50k, 1 session, cache kept | 0.986 s | 3.688 s | 9.092 s |
| kept chat at 50k, 2 sessions, cache kept | 0.573 s | 2.036 s | 4.962 s |
| kept chat at 100k, 1 session, cache kept | 1.272 s | 4.754 s | 11.719 s |

**AXI at 32 sessions against a kept chat that keeps its cache** (whole steps per chip second, ratio):

| | R = 10 | R = 40 | R = 100 |
|---|---|---|---|
| against 100k, 1 session | 1.96 times | **6.49 times** | 13.08 times |
| against 50k, 2 sessions (the best a kept chat fits at 50k) | **0.88 times** | 2.78 times | 5.54 times |

**Time to the first reply token, one session:** AXI 688 ms; kept chat at 50k with its cache 150 ms; at 100k with its cache 176 ms; a kept chat whose cache was lost, at only 32,768 tokens, at least **9,884 ms**.

### The nine lines
1. The code ran to the end on the first try without a change: **YES**.
2. Prefill of 2,824 tokens between 0.3 s and 1.5 s: **YES** (0.623 s).
3. Prefill of 8,192 tokens 2.5 to 4 times that of 2,824: **YES** (3.14 times).
4. With 40-token replies, AXI at 32 sessions at least 3 times a cached kept chat at 100,000 tokens: **YES** (6.49 times).
5. The same ratio at least 10 times: **NO** (6.49 times).
6. The ratio grows with reply length: **YES** (1.96, 6.49, 13.08).
7. Against AXI: at 10-token replies a cached kept chat at 50,000 tokens with 2 sessions needs less chip time per step than AXI at 32 sessions: **YES** (0.573 s against 0.650 s).
8. Against AXI: the first reply token comes sooner for a kept chat that keeps its cache: **YES** (150 and 176 ms against 688 ms).
9. A kept chat whose cache is lost, at only 32,768 tokens, needs at least 10 times AXI's prefill of 2,824 tokens: **YES** (15.8 times).

8 of 9 lines read YES, 1 reads NO.

### What this says, and what it does not
* **The decode-only 42 times of Test 32 shrinks to 2 to 13 times for a whole step, depending on the reply length and on the kept chat's length.** At 40-token replies AXI at 32 sessions does 6.5 times the whole steps per chip second of a 100,000-token kept chat that keeps its cache, and 2.8 times those of the best a kept chat fits at 50,000 tokens. My rough sizing before the run said "nearer 7 times"; the measured figure at 40 tokens is 6.5.
* **Where AXI does not win on this chip:** with very short replies (10 tokens) and a kept chat of 50,000 tokens that keeps its cache, the kept chat uses about 12 % less chip time per step (0.88 times). A kept chat that keeps its cache also gets its first token about four times sooner than AXI's fresh prompt (150 to 176 ms against 688 ms).
* **Where AXI wins clearly:** longer replies (the prefill is a fixed cost and the decode saving grows with the reply), longer kept chats, and any kept chat whose cache is lost or evicted (idle sessions, agents waiting on tools): at only 32,768 tokens that prefill alone is 15.8 times AXI's. Whether a kept chat's cache survives between steps decides which half of this table applies; this test does not measure how often it does.
* **Not in this test:** AXI's prompt may have a static part that a server can cache, which would cut its prefill and favour it further; batched prefill; real serving engines; other chips. The reply lengths and the 100 new tokens a step are assumptions. The Gemini speed figures (3.07 s against 7.13 s a step; first token 2.27 s against 5.06 s) are a separate, hosted measurement at much longer prompts and are not tested here.
* Estimate by parts from two separate measurements on one L4 with constant weights; one run per point.

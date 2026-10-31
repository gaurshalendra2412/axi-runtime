# Test 31: tokens a second one Google TPU chip makes with a 7B-shape model, kept-chat prompts against AXI prompts
## Pre-registered 31 Oct 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
Tests 29 and 30 counted how many sessions fit in a chip's memory. Memory is one limit; the others are compute and **memory bandwidth**. Each token a session generates makes the chip read all the model's weights and that session's whole cache again. This test times it on the **Google TPU v6e-1 chip** with a model of the **exact shape of Qwen2.5-7B**, the largest class that fits one chip with room for sessions (the weights are 15.2 GB of the chip's 33.5 GB). It times one decode step for B sessions at AXI-sized and kept-chat-sized prompts, and gives the tokens a second the whole chip makes.

## Read this first
* **Random (constant) weights of the real shape.** How long a step takes does not depend on the weight values, so nothing is said about quality, and the model is not Qwen. It is a systems measurement.
* One jitted decode step in JAX. The cache is read, and the new token's own key and value are used, but not written back (a small cost left out). No paging, no scheduler, no offloading, no prefix sharing: a real serving engine does more and would be faster in places.
* One chip, one run per row, median of 15 steps after compilation. Rows that do not fit in memory are reported as not fitting.
* The session counts are the largest that fit with the weights. A kept chat at the long run's mean (476,835 tokens) needs 27 GB of cache for one session and is expected not to fit next to the weights; that is a result.
* Memory and bandwidth for this shape on one chip. Other chips, bigger models split over chips, and real traffic are not tested.

## Predictions (written before the run; probability that the line reads YES)
1. The code runs to the end on the first try without a change: **45 %** (new, large JAX code).
2. The AXI-length rows up to 64 sessions fit and give a time: **85 %**.
3. One session of the kept chat at the long run's mean length does not fit next to the 7B weights: **90 %**.
4. The chip's tokens a second at the AXI length with 64 sessions are at least 20 times those at 100,000 tokens with 2 sessions: **75 %**.
5. With one session, the step at 250,000 tokens takes at least 1.5 times the step at the AXI length: **75 %**.
6. At 50,000 tokens with 4 sessions, weights plus cache bytes moved divided by the step time is at least 600 GB/s: **65 %**.
7. Tokens a second at the AXI length grow at least 10 times from 1 to 64 sessions: **85 %**.

## How the result will be read
All figures from `docs/results/tpu_decode.json`, recomputed by a test. Each line reads YES or NO. No claim about a real serving engine, price, quality or other chips.

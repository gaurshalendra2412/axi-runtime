# Test 32: tokens a second one NVIDIA L4 GPU makes with a 7B-shape model, kept-chat prompts against AXI prompts
## Pre-registered 31 Oct 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
Test 31 was written for a Google TPU v6e-1; its first run was on a v5e, whose 16.9 GB left no room for sessions next to a 7B model. This test asks the same question on **an NVIDIA L4 GPU (24 GB) in Colab**, with a model of the **exact shape of Qwen2.5-7B** (weights 15.2 GB, so about 7 GB are left for sessions). It times one decode step for B sessions at AXI-sized and kept-chat-sized prompts, and gives the tokens a second the whole chip makes. Each token a session generates makes the chip read all the weights and that session's whole cache again; this is the memory-bandwidth side of the story.

## Read this first
* **Constant weights of the real shape.** Step time does not depend on the values, so nothing is said about quality; the model is not Qwen. It is a systems measurement.
* One PyTorch decode step, eager, fp16. The cache is read and the new token's own key and value are used, not written back (a small cost left out). No paging, no scheduler, no offloading, no prefix sharing: a real serving engine does more.
* One chip, one run per row, median of 15 steps after warm-up. Rows that do not fit are reported as not fitting.
* The L4 is a small inference GPU (about 300 GB/s of memory bandwidth by NVIDIA's published figure). The larger data-centre chips are not tested; anything about them would be arithmetic.
* The code has never run on a GPU. The first run is also its smoke test.

## Predictions (written before the run; probability that the line reads YES)
1. The code runs to the end on the first try without a change: **60 %**.
2. The AXI-length rows up to 32 sessions fit and give a time: **80 %**.
3. One session of the kept chat at 250,000 tokens does not fit next to the weights: **90 %**.
4. The chip's tokens a second at the AXI length with 32 sessions are at least 15 times those at 100,000 tokens with one session: **75 %**.
5. With one session, the step at 100,000 tokens takes at least 1.2 times the step at the AXI length: **75 %**.
6. At 50,000 tokens with 2 sessions, weights plus cache bytes moved divided by the step time is at least 200 GB/s: **70 %**.
7. Tokens a second at the AXI length grow at least 10 times from 1 to 32 sessions: **85 %**.

## How the result will be read
All figures from `docs/results/gpu_decode7b.json`, recomputed by a test. Each line reads YES or NO. No claim about a real serving engine, price, quality or other chips.

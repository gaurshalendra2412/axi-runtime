# Test 30: how many long sessions one Google TPU chip holds, a kept chat against AXI
## Pre-registered 31 Oct 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
Test 29 counted how many sessions fit in the memory of one NVIDIA T4 GPU (kept chat at the long run's mean prompt: 2 sessions; at its largest: 1; AXI's 2,824-token prompt: 430). This test asks the same question on **a Google TPU chip (v6e, one chip, in Colab)**, the kind of chip Google runs its own models on. Same prompt lengths, same model shape, same method: real cache-shaped arrays put on the chip, session after session, until it reports out of memory.

## Read this first
* **As in Test 29, memory grows in a straight line with tokens, so the ratio will be close to the token ratio (about 169 times) by construction.** What is new here is the chip: its real memory limit, and the real bytes a session takes. TPUs pad array shapes to fixed tile sizes, so a head size of 64 may take more memory than 64 numbers; the test records that.
* Memory only. No speed, quality or throughput. Real servers evict, offload, share prefixes and page memory. The count means sessions in flight at once.
* A small model's shape (Qwen2.5-0.5B), bf16 cache, one run per length. No model is run; cache-shaped arrays are allocated.
* The code has never run on a TPU. The first run is also its smoke test.

## Predictions (written before the run; probability that the line reads YES)
1. The code runs to the end on the first try without a change: **55 %**.
2. The chip reports a memory limit of at least 30 GB: **75 %**.
3. The real bytes a session takes are within 2 % of the nominal bytes (no padding) at every length: **35 %** (I expect padding of the head size to take it above).
4. Sessions held at the AXI length are at least 100 times those held at the kept-chat mean: **80 %**.
5. At the largest kept-chat length (949,591 tokens) the chip holds at most 2 sessions: **85 %**.
6. For every length, the number held is within 5 % of the count computed from the real bytes per session: **70 %**.

## How the result will be read
Counts come from the saved JSON (`docs/results/tpu_sessions.json`), recomputed by a test. Each line reads YES or NO. The T4 and TPU counts will be shown side by side, and nothing is claimed about speed, price or real deployments.

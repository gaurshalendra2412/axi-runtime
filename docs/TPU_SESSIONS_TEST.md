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

## Result
*Run on 31 Oct 2026 in Colab on a TPU v6e-1 ("TPU v6 lite"), JAX 0.11.2. The chip reports a memory limit of 33,546,042,880 bytes (about 33.5 GB). Figures are recomputed by a test from `docs/results/tpu_sessions.json`.*

| prompt kept per session | tokens | TPU v6e-1: sessions held | T4 GPU (Test 29): sessions held |
|---|---|---|---|
| AXI step | 2,824 | **927** | 430 |
| kept chat at 50k | 50,000 | 54 | 23 |
| kept chat at 100k | 100,000 | 27 | 12 |
| kept chat at 250k | 250,000 | 10 | 4 |
| kept chat, mean of the long run | 476,835 | **5** | 2 |
| kept chat, largest of the long run | 949,591 | **2** | 1 |

**Real bytes a session takes on the chip against the plain count:** 1.0454 times at the AXI length (36,277,760 against 34,701,312 bytes), and between 1.0001 and 1.0011 times at every kept-chat length.

### The six lines
1. The code ran to the end on the first try without a change: **YES**.
2. The chip reports at least 30 GB: **YES** (33.5 GB).
3. Real bytes within 2 % of the nominal at every length: **NO**. The AXI length is 4.5 % over. I guessed the padding would come from the head size of 64; it did not (the long lengths show almost none). The extra at 2,824 tokens is about what rounding the token axis up to a tile size would give (2,824 to 2,944 is 4.2 %); I did not confirm that cause.
4. AXI sessions at least 100 times the kept-chat mean: **YES** (927 against 5 = 185 times; the memory ratio is 169, and the 185 is larger because the kept-chat count is rounded down from 5.5 to 5).
5. At the largest kept-chat length at most 2 sessions: **YES** (exactly 2).
6. Every length within 5 % of the count from the real bytes per session: **YES** (927 against 924 at the AXI length, exact at the others).

5 of 6 lines read YES, 1 reads NO.

### What this says, and what it does not
* The TPU holds more sessions than the T4 at every length mainly because it has more memory (33.5 GB against 15.5 GB), not because of any speed. The ratio between AXI and a kept chat is the same on both chips, as it must be: about **169 times in memory** (the token ratio 476,835 / 2,824).
* So on a Google v6e chip, a kept chat at the long run's mean length takes the memory of about 169 AXI-sized sessions, and at the largest length about 336 (949,591 / 2,824).
* **This is memory only.** Nothing here shows speed, quality, throughput or price, and real servers evict, offload, share prefixes and page memory. The count means sessions in flight at once. No model was run; cache-shaped arrays were allocated.

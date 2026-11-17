# Test 41: the rest of the size ladder: Qwen2.5 1.5B, 14B and 32B on the cells of Tests 35 and 36 (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before these cells are run on any of the three models; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Tests 35 and 36 ran five models from 3B to 8B and left out the two ends of the size ladder that the question is really about: the smallest Qwen (1.5B) and the largest one a single card can serve (32B). **The question a company serving its own models asks: does the exact-state arm still hold the right state on every step at the two ends of the ladder (the grammar that names the coordinates), does the kept chat still cost more prompt tokens, and where does the exact share fall on the looser cells (capped expression, names list, ordinary words) as the model grows from 1.5B to 32B?** The 14B is run again here with the same harness and cells as the others, so that the whole ladder (1.5B, 3B, 7B, 14B, 32B) is one table from one harness; the 14B figures from Test 21 (`P0` 94.9 %, `X0` 84.0 %, `XW` 47.8 %, `XR` 90.1 %, on an L4, another harness) are the reference to compare with, not part of the new table.

## What is run
The code of Tests 35 and 36, unchanged (`python/axi/experiments/provider_side_big.py`, repo commit `55631ec`); only the model name changes. vLLM 0.31.0, cache 32,000 tokens (2,000 blocks), prefix caching on, greedy decoding, no retry, the sixth run's jobs 192 to 223 (100 steps each). Seven cells per model: `PN`, `P4`, `PC` (Test 35) and `P0`, `X0`, `XW`, `XR` (Test 36).

| step | model | weights | chip | notes |
|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-1.5B-Instruct` | fp16 | L4 | the smallest model of the ladder |
| 2 | `Qwen/Qwen2.5-14B-Instruct-AWQ` | 4-bit AWQ | L4 | 4-bit, not the 8-bit of the first plan: the 4-bit copy is the one that fits an L4 with a cache |
| 3 | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 4-bit AWQ | a chip with at least 40 GB (A100 or larger), if Colab allocates one | about 19 GB of weights and 8 GB for the 32,000-token cache; it does not fit an L4. If no such chip is allocated it is reported as not run |

The order is the order of the table. A model that does not load, or whose request is refused by the engine, is reported as not run with the error; if a setting has to be changed to make a model run (as happened with Gemma in Test 37), the change is named in the Result and the run is called what it is.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error. Lines 2 to 12 are about the models that ran.

1. [instrument] the 1.5B and the 14B run all seven cells without an engine error (85 %); the 32B runs all seven (50 %)
2. [claim] `PN` exact share is at least 99 percent for every model that ran (85 %)
3. [claim] `P4` exact share is at least 99 percent for every model that ran (85 %)
4. [claim] the kept chat `PC` stops before its 100th step on every model that ran (92 %)
5. [claim] the kept chat computes at least 1.2 times as many prompt tokens per step as the exact-state arm (`PC` against `P4`) on every model that ran (80 %)
6. [claim] steps per hour of `P4` are at least 1.3 times those of `PC` on every model that ran (60 %)
7. [claim] `P0` exact share of the 1.5B is below 100 percent (90 %), and `P0` of the 32B is at least 95 percent (60 %)
8. [claim] `X0` exact share of the 1.5B is below the 3B's 56.59 percent (70 %)
9. [claim] `XW` exact share is below 60 percent for every model that ran (65 %): the 14B was 47.8 percent in Test 21
10. [claim] `XR` is above `XW` by at least 15 points on every model that ran (70 %)
11. [claim] `XR` exact share of the 32B is at least 85 percent (55 %): the 14B reached 90.1 percent in Test 21, with a different harness
12. [claim] the 14B in this harness is within 10 points of its Test 21 figure on each of `P0`, `X0`, `XW` and `XR` (55 %)
13. [claim] among the five Qwen2.5 sizes (1.5B, 3B, 7B, 14B, 32B), `XW` exact share rises with size at every step (each model above the next smaller one; the 3B and 7B figures are those of Test 36) (35 %): Test 36 showed that size alone does not order models

What would count against AXI here: a model whose `PN` exact share is below 95 percent, or a model for which the exact-state arm computes no fewer prompt tokens per step than the kept chat. Either is reported as it is.

## Honest limits
One card type per model (an L4 for the 1.5B and 14B; a larger card for the 32B), greedy decoding, one engine and its version, synthetic service graphs, 100-step jobs (the kept chat is limited by the 8,192-token context set in the test, not by the model), quantised copies for the 14B and 32B. A different chip gives different seconds and steps per hour, so those two figures of the 32B are not comparable with the L4 rows except as ratios inside one model. The Qwen2.5 family only; Test 37 is the other families. The per-cell files stay on the Colab machine; a compact export of the printed lines and hashes is saved with the Result. Nothing here says AXI makes a weak model exact on a loose grammar or on ordinary words; Tests 35 and 36 showed it does not.

---

# Result (written after the run, 17 Nov 2026; nothing above this line is edited)

*The text above this line is the plan as it was locked: its 5,481 bytes have sha256 `f8f5934d140d25cefd5f90dcd7839d34a83a6f1b559ec15485790d5b46a33d31`, recomputed before this Result was appended. The architecture is Rajnish Choubey's; the tests were built and run by Shalendra Gaur with an AI assistant as a tool. AXI is a layer on the host side: the model is unchanged and every figure below is about the state the host holds.*

## What was run, and what differs from the plan
* **1.5B and 14B: as planned.** `Qwen/Qwen2.5-1.5B-Instruct` (fp16) and `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit) on an NVIDIA L4, vLLM 0.31.0, repo commit `55631ec`, harness file unchanged, all seven cells each, 16 Nov. Compact export with the raw byte count and sha256 of every cell file: `docs/results/axi_model_ladder_complete_compact_stage1.json` (sha256 `c956d201c1ac6fdc7fd2d3300e94470214156a33978eeb4cfbea91412f0a538a`).
* **32B: NOT on the chip the plan named, and with one setting changed.** The plan asked for a chip with at least 40 GB and said that without one the 32B is reported as not run. Colab was asked for an A100 and for a G4 and both requests ended on an L4 (23 GB) after 10 to 15 minutes. The 18 GiB 4-bit checkpoint does not fit an L4 with the 32,000-token cache, so the 32B was run on the L4 with **10 GB of the weights held in ordinary memory (`cpu_offload_gb` 10)**. The setting is added by the driver, in memory, to the engine's constructor; the harness file is unchanged (driver `docs/results/axi_model_ladder_complete_driver_q32.py`, sha256 `bc32873862d0d5dc239001131c84a2eb49f66bc48b43bc82d89656d363859071`). The 32B rows are therefore "32B on an L4 with weights partly in host memory", and the Result calls them that.
* **What the offload does to the figures.** Every pass streams the offloaded weights over the card's link, so seconds, steps per hour and energy of the 32B rows are **not comparable** with the 1.5B and 14B rows. Ratios inside the 32B are readable (the plan says so). The exact, parsed, computed-token and cached figures are counted by the harness and should not depend on where the weights sit, but this was not measured: there is no 32B run without the offload.
* **Which machine ran what, and what was lost.** The 32B cells ran on six Colab machines over about 13 hours (16 Nov 17:10 UTC to 17 Nov 05:50 UTC); the screen lines were copied off as they printed. The files of a machine went with it in three events: a reconnect at 19:28 UTC gave a fresh, empty machine; Colab replaced another machine at 23:36 UTC; and two machines were released while the owner's computer was asleep and the browser not connected (about 02:20 UTC, 17 Nov). Because of this only **PN, PC and XR have a saved cell file with a byte count and sha256** (machines C and D, re-run on 17 Nov, listed in `docs/results/axi_model_ladder_complete_compact_stage2.json`, sha256 `07c77fa91166f652fa107439d3925a25e4b29f11cb43aa3d13da1596832c4200`). The 32B cells **P0, X0, P4 and XW rest on the CELL line printed by the run and copied from the screen; their cell files are gone.** The stage 2 file keeps those lines under `screen_lines_only`, marked as such. P4 and PC were run twice (the first run on a machine that was lost at 19:28 UTC): the two runs agree to the printed digits (P4 6.751 and 6.749 s a step, 533 steps an hour both, 125.4 computed tokens, 67.11 % cached; PC 184 steps both, 8.019 and 8.027 s a step, 155.5 computed tokens, 96.23 % cached), which is some evidence that a re-run reproduces; XW, X0 and P0 were not re-run.

## The numbers
Every row: 100-step jobs, jobs 192 to 223 (`PN`, `P0`, `X0`, `XW`, `XR`: 32 jobs, 3,200 steps; `P4`, `PC`: 4 jobs). *exact* is the share of steps on which the state the host held was exactly the true state. The 32B seconds and steps per hour are with the offload (see above). `PC` ends when the kept chat reaches the 7,600-token limit set in the harness; "stopped" is the number of its four chats that stopped there.

| model | cell | steps | exact % | corrupted | parsed % | computed prompt tokens a step | cached % | s a step | steps an hour | time growth | stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B | PN | 3200 | 100.00 | 0 | 100.00 | 97.1 | 70.87 | 0.014 | 258,284 | 0.98 | 0 |
| 1.5B | P4 | 400 | 100.00 | 0 | 100.00 | 125.4 | 67.11 | 0.060 | 60,451 | 1.04 | 0 |
| 1.5B | PC | 187 | 100.00 | 0 | 100.00 | 154.9 | 96.22 | 0.088 | 41,055 | 2.14 | 4 |
| 1.5B | P0 | 3200 | 88.91 | 0 | 100.00 | 96.8 | 71.01 | 0.014 | 262,040 | 0.98 | 0 |
| 1.5B | X0 | 3200 | 32.56 | 0 | 100.00 | 157.3 | 65.41 | 0.019 | 193,016 | 1.01 | 0 |
| 1.5B | XW | 3200 | 2.31 | 0 | 100.00 | 118.8 | 72.31 | 0.016 | 225,706 | 0.90 | 0 |
| 1.5B | XR | 3200 | 15.69 | 0 | 99.97 | 137.7 | 68.34 | 0.018 | 194,782 | 0.93 | 0 |
| 14B | PN | 3200 | 100.00 | 0 | 100.00 | 97.1 | 70.87 | 0.110 | 32,854 | 0.89 | 0 |
| 14B | P4 | 400 | 100.00 | 0 | 100.00 | 125.4 | 67.11 | 0.269 | 13,380 | 0.84 | 0 |
| 14B | PC | 178 | 100.00 | 0 | 100.00 | 155.2 | 96.25 | 0.488 | 7,383 | 1.44 | 4 |
| 14B | P0 | 3200 | 95.16 | 0 | 100.00 | 97.4 | 70.83 | 0.112 | 32,081 | 0.89 | 0 |
| 14B | X0 | 3200 | 84.03 | 0 | 100.00 | 156.8 | 64.66 | 0.153 | 23,493 | 0.88 | 0 |
| 14B | XW | 3200 | 47.72 | 0 | 99.97 | 157.4 | 64.81 | 0.172 | 20,901 | 0.98 | 0 |
| 14B | XR | 3200 | 90.72 | 0 | 100.00 | 154.8 | 64.90 | 0.160 | 22,484 | 0.94 | 0 |
| 32B (L4, offload) | PN | 3200 | 100.00 | 0 | 100.00 | 97.1 | 70.87 | 2.447 | 1,471 | 0.87 | 0 |
| 32B (L4, offload) | P4 | 400 | 100.00 | 0 | 100.00 | 125.4 | 67.11 | 6.749 | 533 | 0.70 | 0 |
| 32B (L4, offload) | PC | 184 | 100.00 | 0 | 100.00 | 155.5 | 96.23 | 8.027 | 448 | 1.82 | 4 |
| 32B (L4, offload) | P0 (line only) | 3200 | 99.69 | 0 | 100.00 | 97.1 | 70.88 | 2.444 | 1,473 | 0.87 | 0 |
| 32B (L4, offload) | X0 (line only) | 3200 | 100.00 | 0 | 100.00 | 156.5 | 64.67 | 2.718 | 1,324 | 0.98 | 0 |
| 32B (L4, offload) | XW (line only) | 3200 | 97.91 | 0 | 100.00 | 155.6 | 64.82 | 3.060 | 1,176 | 1.02 | 0 |
| 32B (L4, offload) | XR | 3200 | 100.00 | 0 | 100.00 | 155.6 | 64.80 | 2.809 | 1,282 | 1.01 | 0 |

## The 13 lines, read by the page's own rules
Computed by `docs/results/axi_model_ladder_complete_readings.py` from the stage files; none typed by hand. **10 read YES, 3 read NO.**

1. **YES**, with the change named above. The 1.5B and the 14B ran all seven cells without an engine error; the 32B also ran all seven, but on an L4 with 10 GB of weights offloaded, not on the planned chip, and four of its seven cells have no saved cell file.
2. **YES.** `PN` exact share: 1.5B 100.00, 14B 100.00, 32B 100.00.
3. **YES.** `P4` exact share: 100.00 on all three.
4. **YES.** The kept chat `PC` stopped before its 100th step on every model: all four chats on each, at 28 to 64 steps (1.5B), 27 to 61 (14B) and 28 to 61 (32B).
5. **YES.** Computed prompt tokens a step, `PC` over `P4`: 1.236 (1.5B), 1.238 (14B), 1.240 (32B).
6. **NO.** Steps an hour of `P4` over `PC`: 1.472 (1.5B), 1.812 (14B), **1.188 (32B)**, which is below 1.3. The ratio is inside one model, so the plan makes it readable. A possible reason, not tested here: with the weights streamed over the link, every pass costs about the same whatever the number of tokens, which compresses the difference between the two arms.
7. **YES**, both parts. `P0` of the 1.5B is 88.91 (below 100); `P0` of the 32B is 99.69 (at least 95; line only).
8. **YES.** `X0` of the 1.5B is 32.56, below the 3B's 56.59.
9. **NO.** `XW` exact share: 1.5B 2.31 and 14B 47.72 are below 60, **32B 97.91 is not** (line only).
10. **NO.** `XR` minus `XW`: **1.5B 13.38** (below 15), 14B 43.00, **32B 2.09**. The 32B's `XW` is already 97.91, so `XR` (100.00) has no room for 15 points; the line is read as written.
11. **YES.** `XR` of the 32B is 100.00 (at least 85); the 14B 90.72 and the 1.5B 15.69.
12. **YES.** The 14B in this harness against Test 21: `P0` +0.26, `X0` +0.03, `XW` -0.08, `XR` +0.62 points.
13. **YES.** `XW` by size: 1.5B 2.31, 3B 7.47, 7B 41.53, 14B 47.72, 32B 97.91, rising at every step (3B and 7B from Test 36).

## Against AXI
What the page said would count against AXI: a model whose `PN` exact share is below 95 percent, or a model for which the exact-state arm computes no fewer prompt tokens a step than the kept chat. **Neither happened.** `PN` was 100.00 on all three models, and `P4` computed 125.4 tokens a step against 154.9, 155.2 and 155.5 for the kept chat.

## In plain words
* The state the host holds stayed exact on every step of `PN` (3,200 of 3,200) and `P4` (400 of 400) on all three models, from 1.5B to 32B. That figure does not depend on the size of the model.
* What does change with size is how often the model's own words match a looser grammar. On the capped expression (`P0`) the exact share goes 88.91, 95.16, 99.69; on the names list (`X0`) 32.56, 84.03, 100.00; on ordinary words with the model reading the name (`XW`) 2.31, 47.72, 97.91. With the host reading the sentence (`XR`) it is 15.69, 90.72, 100.00. A weak model does not become exact on a loose grammar or on ordinary words because of AXI; Tests 35 and 36 said so and the 1.5B rows say it again.
* The kept chat cost about 24 percent more prompt tokens a step than the exact-state arm on all three models, and all its chats hit the context limit at 27 to 64 steps.
* The 32B's two surprising figures (`XW` 97.91, line 9 NO; `XR` over `XW` only 2.09, line 10 NO) are what the run printed, and `XW` is the one that has no saved cell file; a re-run (about 3 hours on this setup) is how to get it.

## Limits
One card type per run (an L4); the 32B with weights partly in host memory, so its seconds, steps an hour and energy are not comparable; greedy decoding; one engine and its version; synthetic service graphs; 100-step jobs (the kept chat stops at the 7,600-token limit of the harness, not by the model); quantised copies for the 14B and 32B. Four of the seven 32B cells have only the printed line. The Qwen2.5 family only. The plan's remark that the per-cell files stay on the Colab machine held, and for the 32B several of those machines were released by Colab before the files could be copied; the compact exports keep what was copied.

## Files
`docs/results/axi_model_ladder_complete_compact_stage1.json` (1.5B and 14B, sha256 `c956d201…a538a`), `docs/results/axi_model_ladder_complete_compact_stage2.json` (32B: summary and raw sha256 for PN, PC, XR; printed lines; the screen-line-only cells; notes), `docs/results/axi_model_ladder_complete_q32_screen_lines.json` (sha256 `ab5a2edd5532c8f410690c961840ca849d9ef89b02ca1848a64d0d949b3be750`), `docs/results/axi_model_ladder_complete_driver_q32.py` (sha256 `bc32873862d0d5dc…59071`), `docs/results/axi_model_ladder_complete_export_cell.py` (sha256 `bc68fad3da10959214afbbb67045aae8e3df76e5307f54cc7f9ded0e992ed574`), `docs/results/axi_model_ladder_complete_readings.py` (sha256 `d88905086721d8119232a5a122bd3ad17678d9366961423ee8f2b02e30870452`).

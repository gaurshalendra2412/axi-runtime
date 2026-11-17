# Test 37: the same cells on three more families (DeepSeek-R1 distill, Mistral, Gemma) (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before these cells are run on any of the three models; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Tests 35 and 36 ran five models from three families (Qwen, Microsoft's Phi, Meta's Llama copy). With the grammar that names every coordinate, all five held the exact state on every step. On the looser cells the exact share depended on the model, by a lot. A company that serves DeepSeek, Mistral or Google's Gemma asks whether the same holds for its model. **The question: on these three more families, with the same jobs, gate, repair and grammar, does the exact-state arm hold the right state on every step where the grammar names the coordinates, and where does the exact share fall on the looser cells?**

## What is run
The code of Tests 35 and 36, unchanged (`python/axi/experiments/provider_side_big.py`, repo commit `55631ec`); only the model name changes. vLLM 0.31.0, one NVIDIA L4, cache 32,000 tokens, prefix caching on, greedy decoding, no retry, the sixth run's jobs 192 to 223 (100 steps each). Seven cells per model: `PN`, `P4`, `PC` as in Test 35, and `P0`, `X0`, `XW`, `XR` as in Test 36.

| step | model | weights | notes |
|---|---|---|---|
| 1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | fp16 | a model distilled from DeepSeek-R1 onto a Qwen2.5 7B base. The grammar fixes the form of every reply, so the model has no room for thinking text: **this test does not measure the cost of thinking tokens**; that is a different test |
| 2 | `unsloth/mistral-7b-instruct-v0.3` | fp16 | Mistral 7B Instruct v0.3, a public copy of the weights |
| 3 | `unsloth/gemma-3-4b-it` | fp16 | Google's Gemma 3 4B instruction-tuned, a public copy of the weights |

The copies of Mistral and Gemma are used because the originals ask for a licence click. The weights are the same models; a copy is not the lab's own repository. The order is the order of the table; a model that does not load, or whose chat template refuses the request, is reported as not run, with the error, and the next one is run. Nothing is changed to make a model work.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error. Lines 2 to 8 are about the models that ran.

1. [instrument] at least two of the three models run all seven cells without an engine error (70 %)
2. [claim] `PN` exact share is at least 99 percent for every model that ran (65 %)
3. [claim] the kept chat `PC` stops before its 100th step on every model that ran (90 %)
4. [claim] the kept chat computes at least 1.2 times as many prompt tokens per step as the exact-state arm (`PC` against `P4`) on every model that ran (70 %)
5. [claim] `P0` exact share is below 100 percent on every model that ran (85 %)
6. [claim] `XW` exact share is below 60 percent on every model that ran (80 %)
7. [claim] `XR` is above `XW` by at least 15 points on every model that ran (60 %)
8. [claim] on the DeepSeek-R1 distill, at least 99.5 percent of the replies of every cell are accepted by the strict parser (70 %)

What would count against AXI here: a model whose `PN` exact share is below 95 percent, or a model for which the exact-state arm computes no fewer prompt tokens per step than the kept chat. Either is reported as it is.

## Honest limits
One card (L4), greedy decoding, one engine and its version, synthetic service graphs, 100-step jobs (the kept chat is limited by the 8,192-token context set in the test, not by the model). Public copies of the weights, not the labs' own repositories. A small model among the three (Gemma 4B) is not the 27B. Models not run are not tested. The figures are read from the lines the notebook prints.

## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run (the plan page's sha256 before the Result was added: `d5a293f5afd2b26ff4f58a1e25135ce291661403b1292e3ab3020805fcc4a4b4`).*

Run: the harness of Tests 35 and 36 unchanged (`provider_side_big.py`, repo commit `55631ec`), vLLM 0.31.0, one NVIDIA L4 on Colab, cache 32,000 tokens (2,000 blocks), prefix caching on, greedy, no retry, the sixth run's jobs 192 to 223 (100 steps each), seven cells per model. Mistral's and Gemma's cells, and DeepSeek-R1's `P4`, were run on one Colab machine (the big notebook), DeepSeek-R1's other six cells on a second one (the free notebook); both are an L4 with the same settings. The figures are the lines the notebook printed; a compact export of them, with the first characters of the sha256 of every raw per-step file, is `docs/results/axi_model_families_compact.json`. The raw files (0.03 to 1.5 MB each) stay on the Colab machines and are lost when they are shut down. The first characters of the sha256 of each raw file are in the compact file (canonical-JSON sha256 of the compact file: `59c676cc7378a86e663222acd66b5788321f3d1af2d50fb798405fd08af8b83e`).

**What ran as planned and what did not.**
* `unsloth/mistral-7b-instruct-v0.3` (fp16): ran as planned, all seven cells.
* `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (fp16): ran as planned, all seven cells, no engine error.
* `unsloth/gemma-3-4b-it`: **did not run as planned.** The engine refused fp16 for this model (its message says Gemma 3 does not support float16), so under the plan's settings the model is *not run*. A second try in bf16 with the plan's 2,000 blocks was refused because the cache memory was too small for the model. A third try, **bf16 with 4,000 blocks (64,000 tokens), was run; it changes two settings the plan fixed, and its figures are reported separately in the table as "changed settings", not as the plan's run.** (The plan says nothing is changed to make a model work; I changed these two, on my own, after the plan was locked.)

**Share of steps on which the held state is the oracle's (exact share, in percent).** The cell names are those of Tests 35 and 36: `PN` exact state, coordinates, grammar that names the coordinates; `P4` the same on the first four jobs; `PC` the whole chat kept (named), four jobs, stops when its next prompt would pass 7,600 tokens; `P0` exact state, coordinates, capped expression (no named grammar); `X0`, `XW`, `XR` the looser requests of Test 36 (coordinates plus names list; ordinary words plus names list; ordinary words read by the host reader).

| cell | Mistral 7B v0.3 | Gemma 3 4B (**changed settings**: bf16, 4,000 blocks) | DeepSeek-R1-Distill-Qwen-7B |
|---|---|---|---|
| `PN` (3,200 steps) | 100.0 | 100.0 | **1.53** |
| `P4` (400) | 100.0 | 100.0 | **0.50** |
| `PC` (kept chat; steps it ran) | 100.0 (181) | 100.0 (179) | 2.15 (93) |
| `P0` (3,200) | 100.0 | 94.03 | 0.0 |
| `X0` (3,200) | 85.44 | 99.88 | 0.0 |
| `XW` (3,200) | 9.88 | 19.34 | 0.0 |
| `XR` (3,200) | 24.03 | 58.84 | 0.91 |

Corrupted steps: 0 in every cell of every model. Replies accepted by the strict parser: 100.0 percent in every cell of Mistral and Gemma, except `XW` (Mistral 94.91, Gemma 99.97), and, for **DeepSeek-R1, a share well below that in every cell** (`PN` 65.88, `P4` 68.75, `PC` 67.74, `P0` 0.94, `X0` 2.88, `XW` 12.16, `XR` 60.84).

**What the exact-state arm costs against the kept chat, on the same four jobs.**

| | computed prompt tokens per step: `P4` / `PC` (ratio) | steps per hour: `P4` / `PC` (ratio) | steps the kept chat ran per job |
|---|---|---|---|
| Mistral 7B v0.3 | 126.3 / 519.5 (4.11) | 13,577 / 7,601 (1.79) | 27, 62, 51, 41 |
| Gemma 3 4B (changed settings) | 128.83 / 172.44 (1.34) | 24,927 / 18,931 (1.32) | 27, 61, 50, 41 |
| DeepSeek-R1-Distill-Qwen-7B | 139.15 / 326.1 (2.34) | 2,983 / 2,665 (1.12) | 22, 24, 24, 23 |

For reference `PN` (32 jobs side by side): computed prompt tokens per step 98.1 (Mistral), 100.9 (Gemma), 125.3 (DeepSeek-R1); steps per hour 58,446 (Mistral), 104,157 (Gemma), 16,523 (DeepSeek-R1). The kept chat's share served from the cache: 87.44 percent (Mistral), 95.82 (Gemma), 91.12 (DeepSeek-R1).

**DeepSeek-R1: where the replies that cannot be read come from.** Counted from the raw per-step files (replies by kind of step, "read" = accepted by the strict parser, "cap" = the reply used all 96 new tokens):

| cell | add a service | add an edge | delete, kind `delete_dep` | delete, kind `delete_leaf` |
|---|---|---|---|---|
| `PN` (named grammar) | 1,069 of 1,069 read, 0 at the cap | 1,039 of 1,039 read, 0 at the cap | **0 of 738 read, 738 at the cap** | **0 of 354 read, 354 at the cap** |
| `P4` (named grammar, first 4 jobs) | 125 of 125 read, 0 at the cap | 150 of 150 read, 0 at the cap | **0 of 90 read, 90 at the cap** | **0 of 35 read, 35 at the cap** |
| `PC` (kept chat, 93 steps) | 18 of 18 read | 45 of 45 read | 0 of 24 read | 0 of 6 read |
| `P0` (capped expression) | 15 of 1,069 read, 1,067 at the cap | 2 of 1,039 read, 1,039 at the cap | 11 of 738 read, 738 at the cap | 2 of 354 read, 354 at the cap |
| `X0` | 6 of 1,069 read | 76 of 1,039 read | 8 of 738 read | 8 of 354 read |
| `XW` | 23 of 1,069 read | 317 of 1,039 read | 39 of 738 read | 10 of 354 read |
| `XR` (words, host reader) | 1,069 of 1,069 read | 860 of 1,039 read | 16 of 738 read | 2 of 354 read |

In `PN`, every add reply was read and every delete reply was not: the 1,092 deletes (about a third of the steps) all used the whole 96 tokens and were not accepted. (`delete_dep` and `delete_leaf` are the harness's two kinds of delete step; I did not work out here which of them lists edges.) Samples of the two kinds, as written by the model: a `delete_leaf` reply began `DEL[0,2]` and went on with one edge clause after another (`DEL[(1,0)->(0,2):dep#5] DEL[(1,1)->(0,2):dep#8] ...`) until the cap; a `delete_dep` reply began `DEL[1,0]` and then wrote the same clause `DEL[(1,0)->(2,2):dep#5]` again and again until the cap. A reply that is not accepted is dropped (no retry in this plan), so the held state keeps the step's change out; the first unread reply came in the first 16 steps of every one of the 32 jobs (the second step, at the median), and from then on the held state differs from the oracle's: that is why the exact share is 1.5 percent and not 66. The `P4` cell (125 of its 400 replies at the cap, all of them deletes; 275 read) is the same pattern. **I did not find out why the model does not end a delete reply.** The cells with the capped expression (`P0`, `X0`, `XW`) are the same on every kind: this model's replies run to the cap on almost every step there (3,198 of 3,200 in `P0`, 3,191 in `X0`, 3,142 in `XW`), and the strict parser accepts 0.9 to 12 percent of them; in `XR`, where the host reader builds the named expression when it can read the request, the adds are read (1,069 of 1,069 services, 860 of 1,039 edges) and the deletes are not (18 of 1,092).

Readings against the predictions:
1. At least two of the three models ran all seven cells without an engine error: **YES** (Mistral and DeepSeek-R1 as planned; Gemma only with changed settings, so it counts as not run under the plan).
2. `PN` exact share at least 99 percent for every model that ran: **NO.** Mistral 100.0 and Gemma (changed settings) 100.0, DeepSeek-R1 **1.53**.
3. The kept chat `PC` stops before its 100th step on every model that ran: **YES** (every job of every model stopped by the context rule: 27 to 62 steps for Mistral and Gemma, 22 to 24 for DeepSeek-R1).
4. The kept chat computes at least 1.2 times as many prompt tokens per step as the exact-state arm on every model: **YES** (4.11, 1.34, 2.34).
5. `P0` exact share below 100 percent on every model that ran: **NO.** Mistral's `P0` is 100.0 (Gemma 94.03, DeepSeek-R1 0.0).
6. `XW` exact share below 60 percent on every model: **YES** (9.88, 19.34, 0.0).
7. `XR` above `XW` by at least 15 points on every model: **NO.** Mistral 24.03 against 9.88 (14.15, short by 0.85); Gemma 39.50 points (YES); DeepSeek-R1 0.91 against 0.0 (0.91 points).
8. On the DeepSeek-R1 distill at least 99.5 percent of the replies of every cell accepted by the strict parser: **NO** (from 0.94 to 68.75 percent; no cell reaches it).

Against AXI, by the plan's own test: **a model whose `PN` exact share is below 95 percent: DeepSeek-R1-Distill-Qwen-7B, 1.53 percent. This is a result against AXI on that model with these settings.** The other condition (the exact-state arm computing no fewer prompt tokens per step than the kept chat) did not occur on any of the three models.

What this does and does not show. On Mistral 7B v0.3 and, with changed settings, on Gemma 3 4B, the exact-state arm with the named grammar held the right state on every step (3,200 of 3,200 and 400 of 400 on each), as it did on the five models of Tests 35 and 36, and it computed 1.3 to 4.1 times fewer prompt tokens per step than the kept chat. On the looser cells the exact share depends on the model (Mistral's `X0` 85.4 and `XW` 9.9 percent; Gemma's 99.9 and 19.3), as Test 36 found for the earlier models. On the DeepSeek-R1 distill, with this grammar, this token cap (96) and no retry, the host cannot read any delete reply, and that alone moves the held state off the oracle's within the first 16 steps; the host's gate and repair act on a reply it can read and could not act here. Not tested, so not claimed: a higher token cap, one retry, a different wording of the instruction for this model, its thinking mode (the grammar leaves no room for thinking text, as the plan says: the cost of thinking tokens is not measured here), another distill size, or whether the engine's end-of-reply handling for this model is part of the cause. The DeepSeek-R1 cells say that this model, under these settings, does not close the delete replies; they do not say that its state could not be held under other settings.

Limits specific to this run. One card, greedy decoding, one engine version. Gemma ran only with changed settings (bf16 and a larger cache), so its figures are not the plan's. Mistral and Gemma are public copies of the weights (`unsloth/...`), not the labs' own repositories. The per-kind counts and the first-unread-step figures were made by short cell runs on the raw files after the cells. The figures are read from printed lines and from those counts, and no second run was made.

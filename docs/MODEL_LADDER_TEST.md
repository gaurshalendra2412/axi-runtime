# Test 35: the same serving test on other models, from 3B to 32B and from other families (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before any model of the ladder is run; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Tests 15 to 21 were run on one model family: Qwen2.5 7B (T4) and 14B (L4). A company that serves a different model asks first whether the result holds for it. The question here is narrow: **with the same jobs, the same gate, repair and grammar, and the model changed, does the exact-state arm still hold the right state on every step, and does the kept chat still cost more?** Nothing about AXI is changed; only the model name is.

## What is run
The code of Test 21 (`python/axi/experiments/provider_side_big.py`, repo commit `55631ec`), unchanged and not edited; its engine class takes the model name as a setting. The jobs are the sixth run's jobs 192 to 223 (100 steps each), vLLM 0.31.0, prefix caching on, a cache of 32,000 tokens, greedy decoding, one NVIDIA L4, no retry anywhere. Three cells per model, as in Test 21: `PN` (32 jobs, the exact state at every step, the grammar that names the coordinates), `P4` (the first 4 jobs, the same), `PC` (the first 4 jobs, the whole chat kept; it stops when its next prompt would pass 7,600 tokens).

| step | model | weights | notes |
|---|---|---|---|
| 1 | `Qwen/Qwen2.5-3B-Instruct` | fp16 | smaller than any model of the earlier runs |
| 2 | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 4-bit | the earlier 7B, now on the L4 |
| 3 | `Qwen/Qwen3-8B-AWQ` | 4-bit | a newer Qwen; run with its thinking switched off in the chat template (the only change to the request) |
| 4 | `microsoft/Phi-3.5-mini-instruct` | fp16 | another family (Microsoft) |
| 5 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | 4-bit | another family (Meta's Llama 3.1, a quantised copy) |
| 6 | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 4-bit | does not fit an L4 with a cache; needs a bigger card (A100). Run only after its cost is shown and agreed |

The 14B was Test 21 and is not run again. The order is the order of the table; a model that does not load is reported as not loaded, and the next one is run.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error.

1. [instrument] for every model run, the engine reported cached tokens for every request, and at least 99.5 percent of all replies were accepted by the strict parser (90 %)
2. [claim] over 100 steps the exact share of `PN` is at least 99 percent for every model run (60 %)
3. [claim] the exact share of `PN` for the 3B is at least 99 percent (65 %)
4. [claim] the exact share of `PN` differs by at most 1 point between the best and the worst model run (55 %)
5. [claim] the kept chat does not reach step 100: every job of `PC` stops before its 100th step for every model run (90 %)
6. [claim] tokens: the kept chat computes at least 1.3 times as many prompt tokens per step as the exact-state arm on the same 4 jobs (`PC` against `P4`) for every model run (75 %)
7. [claim] speed: the exact steps per GPU hour of `P4` are at least twice those of `PC` for every model run (70 %)
8. [claim] latency does not grow with the job: in `PN` the seconds per step of the last quarter are at most 1.15 times those of the first quarter for every model run (70 %)
9. [claim] the Qwen3-8B (thinking off) and Llama 3.1 8B hold the right state on at least 95 percent of the steps of `PN` (70 %)

What would count against AXI here: a model whose `PN` exact share is below 95 percent, or a model for which `P4` is not cheaper than `PC` in computed tokens per step. Either is reported in the Result as it is.

## Honest limits
One card (L4), greedy decoding, one engine and its version, synthetic service graphs, the grammar that names the coordinates (so the model is not asked to find the coordinates in free words), 100-step jobs (the longest kept chat is limited by the 8,192-token context set here, not by the model). Quantised copies are not the original weights. Models not run are not tested.

## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run.*

Run: the code of Test 21 unchanged (commit `55631ec`), vLLM 0.31.0, one L4 on Colab, cache 32,000 tokens, prefix caching on, greedy. Five of the six models were run (steps 1 to 5 of the table). The 32B (step 6) was not run: it needs a bigger card and its cost was not yet agreed. It is not tested. The figures are read from the lines the notebook printed; the per-cell JSON files were on the Colab machine and are not part of this commit.

Every cell of every model: exact share 100 percent, corrupted steps 0, parsed share 100 percent.

| model | `P4` s/step | `P4` steps/h | `P4` computed tokens/step | `PC` steps (4 jobs) | `PC` steps/h | `PC` computed tokens/step | `PN` steps/h |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 0.114 | 31,465 | 125.4 | 187 | 22,649 | 154.9 | 136,952 |
| Qwen2.5-7B-AWQ | 0.112 | 32,241 | 125.4 | 187 | 22,761 | 154.9 | 86,374 |
| Qwen3-8B-AWQ | 0.120 | 29,883 | 129.4 | 187 | 19,137 | 169.5 | 83,524 |
| Phi-3.5-mini | 0.152 | 23,631 | 127.3 | 177 | 12,775 | 174.2 | 95,320 |
| Llama 3.1 8B AWQ | 0.117 | 30,724 | 125.3 | 186 | 20,156 | 155.8 | 83,713 |

Ratios, kept chat over exact state: computed tokens per step 1.24 (3B), 1.24 (7B), 1.31 (Qwen3-8B), 1.37 (Phi), 1.24 (Llama); steps per hour of `P4` over `PC` 1.39, 1.42, 1.56, 1.85, 1.52. In `PN` the seconds per step of the last quarter over the first were 0.98 (3B), 0.97 (7B) and 0.97 (Llama), and at most 1.15 on every model.

Readings against the predictions:
1. Cached tokens reported on every request and at least 99.5 percent of replies parsed: **YES** (100 percent parsed on every cell).
2. `PN` exact share at least 99 percent for every model: **YES** (100.0 on all five).
3. `PN` exact share of the 3B at least 99 percent: **YES** (100.0).
4. `PN` exact share within 1 point between best and worst: **YES** (all 100.0).
5. Kept chat does not reach step 100 on any model: **YES** (it stopped at the 7,600-token limit on every model; 177 to 187 steps over 4 jobs).
6. Kept chat computes at least 1.3 times the tokens per step of the exact-state arm for every model: **NO.** It is 1.24 on three of the five (Qwen2.5 3B and 7B, Llama).
7. `P4` has at least twice the steps per GPU hour of `PC` for every model: **NO.** The ratios are 1.39 to 1.85.
8. Latency does not grow with the job in `PN` (last quarter at most 1.15 times the first) for every model: **YES.**
9. Qwen3-8B and Llama 3.1 8B hold at least 95 percent of the `PN` steps: **YES** (100.0 both).

What this shows and what it does not. With the grammar that names every coordinate, the exact-state arm held the right state on every step on all five models, across three families, from 3B to 8B, with no change to the framework; `PN` ran at 83,000 to 137,000 steps per GPU hour and its cost per step did not grow with the job. The kept chat was also exact on every step it ran: in at most 187 steps it did not drift. Its disadvantage here is that it stops at the context limit, and that it computes about 1.24 to 1.37 times the prompt tokens and runs at 0.54 to 0.72 of the speed. That is smaller than the 1.3 and 2 times I predicted, and it is reported as it is. Reading a longer kept chat (a larger context) is another test. The 32B and any model not listed are not tested.

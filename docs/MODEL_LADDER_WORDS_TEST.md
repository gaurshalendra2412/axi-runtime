# Test 36: the harder cells (capped expression, names list, ordinary words) on other models (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before these cells are run on any model of the ladder; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Test 35 ran the cells where the grammar names every coordinate (`PN`, `P4`, `PC`). Every model held the state on every step there, so those cells say little about model size: the grammar does the work. The cells that depend on the model are the ones with a looser expression or with the request in ordinary words. On the 7B (earlier runs, T4) and the 14B (Test 21, L4) the figures were: `P0` 96.1 % and 94.9 %; `X0` 100.0 % and 84.0 %; `XW` 42.0 % and 47.8 %; `XR` 63.4 % and 90.1 %. **The question: on other models, does the exact-state share on these cells move with the model, and does the host reader (`XR`) still lift the words cell above the plain one (`XW`)?**

## What is run
The same code as Test 35 (`provider_side_big.py`, repo commit `55631ec`, unchanged), same engine settings (vLLM 0.31.0, cache 32,000 tokens, prefix caching on, greedy, one L4, no retry), the same 32 jobs (192 to 223, 100 steps), cells `P0`, `X0`, `XW`, `XR`, on the models of Test 35 that loaded: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct-AWQ, Qwen3-8B-AWQ (thinking off), Phi-3.5-mini-instruct and the Llama 3.1 8B AWQ copy (the 32B only if Test 35 ran it). The 7B and 14B are not part of the new figures except as the numbers above.

## Predictions (mine, written first)
1. [claim] `P0` exact share is below 100 percent for every model run, and the 3B is below the 7B (75 %)
2. [claim] `XW` exact share (ordinary words, the model reads the names) is below 60 percent for every model run (80 %)
3. [claim] `XR` exact share is above `XW` for every model run, by at least 15 points (80 %)
4. [claim] `XR` exact share is at least 85 percent for at least one model run (60 %)
5. [claim] `X0` (the names list given with coordinates) exact share is at least 95 percent for the 3B (45 %)
6. [claim] the exact share of `XW` does not rise with size across the Qwen2.5 3B and 7B: the 3B is within 10 points of the 7B or above it (55 %)

What would count against AXI: none of these lines is about AXI against a chat. They show where the host's reader and grammar carry the result and where the model does.

## Honest limits
As Test 35: one card, greedy decoding, synthetic service graphs, 100-step jobs, quantised copies. `XW` and `XR` depend on the sentence wording of the test and on the host reader of the repository; a different phrasing is another test. A share below 100 percent on these cells is a slip of the model that the gate and the repair did not complete; it is reported as it is.

## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run.*

Run: the code of Test 35 unchanged (`provider_side_big.py`, commit `55631ec`), vLLM 0.31.0, one L4 on Colab, cache 32,000 tokens, greedy, 32 jobs of 100 steps (3,200 steps per cell), five models, four cells each. Corrupted steps: 0 on every cell. The 32B was not run. The figures are read from the lines the notebook printed; the per-cell JSON files were on the Colab machine and are not part of this commit.

Exact share of steps (percent):

| model | P0 | X0 | XW | XR |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct (fp16) | 91.62 | 56.59 | 7.47 | 23.09 |
| Qwen2.5-7B-Instruct-AWQ | 96.09 | 100.0 | 41.53 | 64.0 |
| Qwen3-8B-AWQ (thinking off) | 100.0 | 98.06 | 35.69 | 66.06 |
| Phi-3.5-mini-instruct (fp16) | 14.03 | 18.47 | 4.09 | 25.0 |
| Llama 3.1 8B AWQ copy | 52.53 | 13.03 | 5.16 | 32.0 |

The 7B row matches the earlier T4 figures (96.1, 100.0, 42.0, 63.4).

Readings against the predictions:
1. `P0` below 100 percent for every model, and 3B below 7B: **NO.** The 3B (91.62) is below the 7B (96.09), but Qwen3-8B reached 100.0 on `P0`.
2. `XW` below 60 percent for every model: **YES.** The highest is 41.53 (Qwen2.5-7B).
3. `XR` above `XW` by at least 15 points for every model: **YES.** The gaps are 15.6 (3B), 22.5 (7B), 30.4 (Qwen3-8B), 20.9 (Phi), 26.8 (Llama).
4. `XR` at least 85 percent for at least one model: **NO.** The highest is 66.06 (Qwen3-8B).
5. `X0` at least 95 percent for the 3B: **NO.** It is 56.59.
6. `XW` of the 3B within 10 points of the 7B or above: **NO.** 7.47 against 41.53.

What this shows and what it does not. On these cells the exact share depends on the model, and by a lot: Phi-3.5-mini and the Llama 3.1 8B copy are far below the Qwen2.5 7B on `P0` and `X0`, and no model reaches 85 percent on the words cell even with the host reader. The host reader (`XR`) lifts the words cell above the plain one (`XW`) on all five models, by 15 to 27 points. Size alone does not order the models: Llama 3.1 8B is below the Qwen2.5 3B on `XW`. Nothing here says AXI makes a weak model exact on a loose grammar or on ordinary words; it does not. The cells where the grammar names every coordinate are Test 35, where every model held the state.

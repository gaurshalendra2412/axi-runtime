# Drift over many steps, and what a growing chat history costs (written BEFORE the run, 8 Oct 2026)

This page fixes the questions, the predictions and the reading rules before any number from this test exists. It follows `SCALE_TEST_7B.md` (one step at a time) and `LATENCY_7B.md` (what one step costs). Code: `python/axi/experiments/long_horizon.py`; the instrument is checked on a scripted stand-in by `tests/test_long_horizon.py`. Notebook: `python/colab/axi_long_horizon.ipynb`. Concept-map rows 3, 9, 27 (abort lands on committed state; structure against scale; errors that compound).

## The questions
1. **Drift.** Over 30 requests in a row, how far does the state a system holds move away from the state it should hold, when the model's proposals are (a) applied blind, (b) filtered by the gate, (c) filtered and repaired by M6?
2. The same with the rule stated in the prompt.
3. **Context.** Same model, same requests, same information. Does it matter for time, or for what the model proposes, whether each step starts from a fresh context holding the exact current state, or the whole conversation so far stays in the context?

## What "drift" means here
The posts use the word loosely. The operational meaning used here: **distance** = the number of differing items between the held state and an oracle's state after the same requests (a service missing or with another value; an edge missing or with another weight). There are two ways to be far: **corruption** (an edge points at a service that is gone) and **omission** (a requested change did not happen). The oracle applies the ideal delta at every step.

## Design
* Model: Qwen2.5-7B-Instruct, 4-bit, Colab T4, greedy, 400 new tokens, constrained decoding. Same prompts as every earlier run (`SYSTEM`, optional `RULES_HINT`; the prompt shows the current state and one request).
* **State mode.** An episode is a fixed script of 30 requests chosen on the oracle's graph (so all policies get the same requests): 30 percent add service, 30 percent add edge, 30 percent delete a service that has dependents, 10 percent delete a service with no edges (about 8 delete-with-dependents steps per episode; the graph is kept between 0 and 14 services). Each policy holds its own graph, shows it to the model at every step, and updates it by its own rule: **blind**, **gate**, **repair_m6**. Identical prompts are asked once (greedy decoding).
* **History mode.** Same script, policy repair_m6 only, but the conversation accumulates: every earlier state, request and raw reply stays in the context. The information is the same as in state mode plus the past; only the length changes. No KV cache is reused between steps, which is the naive chat loop. An episode stops cleanly if the next turn would pass 14,000 tokens or the GPU runs out of memory.
* Sizes: state mode no rule 12 episodes; history mode 4 episodes (the same seeds as the first four state episodes, so they are compared like for like); state mode with the rule 8 episodes (optional cell). Seed 3.

## Predictions (mine, written first)
1. **State, no rule.** Blind apply: at least 90 percent of episodes are corrupted by step 10, and the dangling edges pile up (more than 5 on average at step 30) because nothing ever removes one. Gate: corrupted 0 percent at every step. Repair_m6: exactly the oracle's state on at least 95 percent of all steps. I expect the gate's mean distance at step 30 to be at least blind's, because a refused delete leaves the service and all its edges while a blind delete leaves only the edges (60 percent confident; the gate trades corruption for omission).
2. **State, rule stated.** Blind: at least 80 percent of episodes corrupted by step 10. Repair_m6 holds the oracle's state on less than 95 percent of steps, because the 7B writes real but unrequested edge deletes (`SCALE_TEST_7B.md`, section 3), which the gate admits and M6 keeps; those errors persist in the held state. This is the drift the gate cannot see.
3. **Context.** State mode is flat in time (last quarter of the steps at most 1.3 times the first). History mode is at least 3 times slower in the last quarter than in the first, because each step re-reads a context of several thousand tokens that grows by a few hundred tokens a turn. The held state is exact in both (M6 repairs it whatever the model wrote), so the signal for any change in the model's behaviour is its raw proposals: in state mode at most 10 percent of the delete-with-dependents proposals are admissible as written. For history mode I do not predict a direction: the past turns show the repaired states, so the model could imitate them (better) or lose track in a long context (worse).

## Reading rules (fixed now; the notebook prints exactly these)
State mode (`verdict_state`): (a) gate and repair_m6 never corrupted at any step; (b) blind: at least 90 % of episodes corrupted by step 10; (c) repair_m6 exact on at least 95 % of steps. Context (`verdict_context`): (a) state mode last quarter at most 1.3 times the first; (b) history last quarter at least 3 times the first; (c) the raw proposals on delete_dep differ by at least 10 points between history and state in the last quarter. A line that comes out "NO" is a finding, not an error. If (a) of the state rules fails, something is wrong with the gate and that is a bug to find first.

## What this cannot show
One model (4-bit), one task family, greedy decoding, 30 steps, at most about 14,000 tokens of context, no KV-cache reuse, small numbers of episodes (the percentages move in steps of 8 to 25 points). "Drift" is the distance defined above; attention dilution, long documents, sampling at a temperature and other models are not tested. A history that is longer than 14,000 tokens is not tested. The GPU-side timing of long contexts for a cache-reusing engine is the earlier 0.5B KV-cache table, not this test.

## How it is run
`axi_long_horizon.ipynb` on Colab (T4): upload `axi-colab-long.zip`, run the cells top to bottom. Each cell writes one line per finished episode, so a dropped session continues where it stopped. Expected time: the smoke test 1 to 2 minutes, state mode no rule about 15 minutes, history mode about 10 to 20 minutes, the optional rule cell about 30 minutes. The results are JSON files; send them back and this page gets a "Result" section without changing anything above.

---

# Result, part 1 (run 8 Oct 2026, evening): the rule regime and the long chat history
Everything above this line is unchanged. This part covers the two files received so far: `docs/results/axi_long_state_rule_Qwen2.5-7B_seed3.json` (state mode, rule stated, 8 episodes of 30 steps, seeds 3000 to 3007) and `docs/results/axi_long_history_norule_Qwen2.5-7B_seed3.json` (history mode, no rule, 4 episodes, seeds 3000 to 3003). **Run A (state mode, no rule, the main prediction 1) has not been received yet**, so prediction 1, the state-mode reading rule and context rules (a) and (c) are not judged here. Checked by `tests/test_long_horizon_result.py`, which recomputes every number below from the saved rows.

## 1. State mode with the rule stated (prediction 2)
| policy | episodes corrupted by step 10 | corrupted at any step | steps exactly equal to the oracle | distance at step 10 / 20 / 30 | dangling edges at step 30 | raw delete_dep proposals admitted as written |
|---|---|---|---|---|---|---|
| blind apply | 75 % (6 of 8) | 88 % (7 of 8) | 23 % (56 of 240) | 2.6 / 2.4 / 1.9 | 1.1 | 48 % (29 of 60) |
| gate | 0 % | 0 % | 8 % (18 of 240) | 5.8 / 11.1 / 12.5 | 0 | 35 % (21 of 60) |
| repair_m6 | 0 % | 0 % | 55 % (132 of 240) | 1.2 / 1.1 / 0.5 | 0 | 55 % (33 of 60) |

Scored against what was written first:
* Gate and repair_m6 never corrupted: **held** (0 of 240 steps each).
* Blind: at least 80 % of episodes corrupted by step 10: **missed**, 75 % is 6 of 8; seven of the eight were corrupted at some step, so it missed by one episode.
* Repair_m6 exact on less than 95 % of steps: **held**, 55 %.
* The notebook also printed its three "yes / NO" lines for this run. They apply the no-rule thresholds (blind at least 90 %, repair_m6 at least 95 %), so the two "NO" lines there are not the rule for this regime; prediction 2 above is.

## 2. Where the repair_m6 drift comes from
* 14 of the 240 steps put a new wrong item into the held state: 8 of 60 delete_dep steps and 6 of 23 delete_leaf steps. No add_node or add_edge step did.
* All 34 wrong items are the same kind: **a real edge that is missing from the held state, because the model wrote a delete for an edge nobody asked to delete.** Each was legal, so the gate admitted it and M6 kept it. Not one was a wrong service value, a wrong weight or a dangling edge.
* Seven of the eight episodes had at least one such step; only seed 3001 stayed exact for all 30 steps.
* The wrong items mostly vanish by themselves: 30 of the 34 stopped counting because the oracle later deleted a service the edge belonged to (all 30 at a delete_dep step). That is why the mean distance stays near 1 and does not grow. Four were still wrong at step 30. The rate that matters is 14 wrong-item steps in 83 delete steps (17 %); in the one-step 7B test it was 16 misses in 42 delete tasks (38 %), on different tasks and states, so the two are not the same measurement.
* With the rule in the prompt the 7B also changes how delete_leaf looks: only 6 of 23 delete_leaf proposals were admissible as written (17 were refused, all of those grounded and completed by M6), against 12 of 12 with no rule in the history run.

## 3. The gate alone
It is never corrupted and drifts steadily the other way: distance 5.8, 11.1, 12.5 at steps 10, 20, 30. Each refused delete leaves the service and all its edges in place, so the held state fills with things that were asked to be removed (71 extra edges and 54 extra services appeared over the eight episodes). In this regime the gate's distance at step 30 is 6.6 times blind's (12.5 against 1.9); the same comparison with no rule waits for Run A.

## 4. Time in state mode
Rule regime, repair_m6: 1.46 s per call in the first quarter of the steps, 1.34 s in the last (0.91 times), with prompts of about 460 and 380 tokens. Flat, as predicted, but this is the rule regime; the pre-registered check of the flat claim is Run A.

## 5. A growing chat history (prediction 3, the history half)
* **The held state stayed exact:** 105 of 105 steps at distance 0, none corrupted, as predicted (M6 repairs whatever the model wrote).
* **Time per step rose about five times.** Step 1 took about 1.0 s. Over the 19 steps all four episodes reached, the first four steps averaged 1.55 s and the last four 7.90 s: **5.1 times** (the prediction was at least 3 times; held). With only the episodes that got further: 5.4 times at 26 steps (3 episodes), 5.7 times at 30 steps (2 episodes). A fit over the 105 steps gives seconds = -1.9 + 0.00235 x prompt tokens + 0.096 x output tokens (R squared 0.974), about 2.4 ms for each token of history; a small squared term lifts R squared to 0.999. The two episodes that completed 30 steps took 179 s and 149 s; a fresh call per step would be about 30 s at the 0.97 s of `LATENCY_7B.md` (that is an estimate from another run, not a measurement of this one).
* **The raw proposals did not change.** delete_dep: 0 of 24 admitted as written; every other kind: 81 of 81 admitted. After up to seven earlier delete_dep steps in the same chat, the model still wrote the service delete without its edges every time; the history showed it its own raw replies and the repaired states, and that did not teach it. Whether state mode differs is Run A's question (the one-step 7B run without the rule had 0 of 30).
* **The memory limit was reached far below the planned cap.** Two of the four episodes ended with an out-of-memory error: seed 3002 after 19 steps (prompt 5,135 tokens, the next would have been about 5,400) and seed 3001 after 26 steps (5,530 tokens, the next about 5,700). The other two finished 30 steps (largest prompts 5,230 and 4,503). On this setup (Colab T4, 7B in 4-bit, no cache reuse) a conversation cannot go much beyond 5,500 tokens, so the 14,000 in the design above was never the limit. The cause was not investigated (attention memory grows with the square of the length; memory left by earlier cells of the same session is also possible).

## 6. Scoreboard so far
| prediction | outcome |
|---|---|
| 1. state, no rule: blind 90 %+ corrupted by step 10, more than 5 dangling at step 30; gate 0 %; repair_m6 exact on 95 %+ of steps; gate distance at least blind's | **waiting for Run A** |
| 2. rule stated: blind 80 %+ corrupted by step 10 | **missed** (75 %, 6 of 8) |
| 2. rule stated: repair_m6 exact on less than 95 % of steps, because of unrequested legal edge deletes | **held** (55 %; all 34 wrong items are such deletes) |
| 3. state mode flat in time | waiting for Run A (rule regime: 0.91 times) |
| 3. history at least 3 times slower in the last quarter | **held** (5.1 times, up to about 5,500 tokens only) |
| 3. held state exact in history mode | **held** (105 of 105) |
| 3. state: at most 10 % of delete_dep proposals admissible | waiting for Run A (history mode: 0 of 24) |

## 7. What this does and does not show
It shows, for this model and these scripts, that gate and M6 never let a dangling edge into the held state over 30 requests, that what M6 cannot see is the legal edge delete nobody asked for (the same error as in `SCALE_TEST_7B.md`, now measured as it builds up over steps), that the gate alone trades corruption for a held state that moves away from the intended one, and that keeping the whole chat costs about five times the time per step by step 20 to 30 without changing what the model proposes. It does not show anything about contexts above about 5,500 tokens, other models, sampling, or the no-rule drift of blind apply (Run A).

---

# Result, part 2 (8 Oct 2026, evening): state mode with no rule (Run A), the main prediction
File: `docs/results/axi_long_state_norule_Qwen2.5-7B_seed3.json` (12 episodes of 30 steps, seeds 3000 to 3011, no rule; the first 8 episodes have the same request scripts as the rule run, the first 4 the same as the history run). With it every prediction above has a number; the "waiting for Run A" lines of part 1 are settled in section 5 here. Checked by `tests/test_long_horizon_result.py`.

## 1. Drift, no rule (prediction 1)
| policy | corrupted by step 10 | corrupted at any step | steps exactly equal to the oracle | distance at step 10 / 20 / 30 | dangling edges at step 30 | raw delete_dep proposals admitted as written |
|---|---|---|---|---|---|---|
| blind apply | 100 % (12 of 12) | 100 % | 8 % (30 of 360) | 6.3 / 11.6 / 15.0 | 14.1 | 0 of 95 |
| gate | 0 % | 0 % | 8 % (30 of 360) | 9.3 / 17.7 / 23.3 | 0 | 0 of 95 |
| repair_m6 | 0 % | 0 % | 99 % (358 of 360) | 0.0 / 0.0 / 0.2 | 0 | 0 of 95 |

* **Blind apply:** every one of the 12 episodes was corrupted at the very step of its first delete-with-dependents request (steps 1 to 7), and the dangling edges only pile up: 2.0 on average at step 5, 6.3 at step 10, 14.1 at step 30 (7 to 20 per episode). Predicted at least 90 % corrupted by step 10 and more than 5 dangling at step 30: **held**.
* **Gate:** 0 of 360 steps corrupted; all 95 delete-with-dependents proposals were refused, so its held state kept services and edges the oracle had removed and its distance reached 23.3, higher than blind's 15.0. Predicted "gate distance at least blind's" (60 % confident): **held**.
* **repair_m6:** 0 of 360 steps corrupted, exact on 358 of 360 steps (11 of 12 episodes exact at step 30). It repaired all 95 refused proposals (every one a delete_dep) to the exact intended state. Predicted at least 95 % of steps: **held**.
* **The two steps where repair_m6 was off** are one event: seed 3006, step 29, the request "add a dependency edge from (4,3) to (1,1), relation dep, weight 3". The model wrote the edge from (3,5) instead of (4,3), a legal edge, so the gate admitted it and M6 kept it: one wrong edge present and the asked-for one missing, distance 2 at steps 29 and 30. Blind and gate, shown a different held graph at that step, wrote it correctly. It is the same kind of error as in the rule run (legal, so the gate cannot see it), here a wrong source service, once in 116 add_edge steps.
* **Drift feeds back into what the model is asked.** On the blind and gate paths the held state is wrong, and the model's next proposals are refused for that reason: 24 of 24 refused delete_leaf proposals named a service that still had edges in the held graph, and 24 of the 25 add_node proposals the gate refused named a service the held graph still contained. On the exact repair_m6 path 35 of 35 delete_leaf and 114 of 114 add_node proposals were admitted as written.

## 2. The same request scripts with and without the rule
On the 8 episodes both regimes share (identical requests), steps exactly equal to the oracle out of 240: blind apply 28 without the rule and 56 with it, gate 28 and 18, **repair_m6 238 without the rule and 132 with it**. Distance at step 30: blind 14.4 and 1.9, gate 21.9 and 12.5, repair_m6 0.25 and 0.5. Stating the rule helps the unwrapped model (it writes the edge deletes first, so blind apply is far less wrong) and hurts the wrapped one (the extra edge deletes it writes are legal and unrequested, which M6 keeps). With the rule, 7B plus M6 is at 55 % exact steps; without it, 99 %.

## 3. Time in state mode, and state against history on the same episodes
Across all 12 episodes the model was called 995 times for 1,080 steps (identical prompts are asked once), 1.12 s on average, prompts of 234 to 601 tokens, 1,110 model seconds in all. Repair_m6 first 7 steps 1.05 s, last 7 steps 1.07 s (1.02 times).

Paired, the 4 episodes both modes ran, the first 19 steps (all four reached them), first and last quarter:
| | seconds per step, first quarter | last quarter | growth | prompt tokens, first / last | exact steps |
|---|---|---|---|---|---|
| state (fresh context, exact state) | 1.15 | 1.02 | 0.89 times | 397 / 341 | 100 % |
| history (whole chat kept) | 1.55 | 7.90 | 5.11 times | 723 / 3,820 | 100 % |

Same requests and the same information; late in the episode a step costs about 7.8 times as much with the chat kept. A whole 30-step episode took 29 s and 30 s in state mode (seeds 3000 and 3003) against 179 s and 149 s in history mode. Predicted: state at most 1.3 times: **held** (0.89); history at least 3 times: **held** (5.11).

The third context rule, "the raw delete_dep proposals differ by at least 10 points between history and state", came out **NO**, and that is a finding: both are 0. State mode admitted 0 of 95 delete-with-dependents proposals as written (predicted at most 10 %: **held**), history mode 0 of 24. Keeping the whole chat, with the model's own earlier raw replies and the repaired states in view, changed neither what it proposed nor what the held state became; it only cost time.

## 4. Scoreboard, final
| prediction | outcome |
|---|---|
| 1. state, no rule: blind at least 90 % corrupted by step 10, more than 5 dangling at step 30 | **held** (100 %, 14.1) |
| 1. gate 0 % corrupted | **held** |
| 1. repair_m6 exact on at least 95 % of steps | **held** (99 %, 358 of 360) |
| 1. gate distance at step 30 at least blind's (60 % confident) | **held** (23.3 against 15.0) |
| 2. rule stated: blind at least 80 % corrupted by step 10 | **missed** (75 %, 6 of 8) |
| 2. rule stated: repair_m6 exact on less than 95 % of steps | **held** (55 %) |
| 3. state mode flat in time (at most 1.3 times) | **held** (0.89) |
| 3. history at least 3 times slower in the last quarter | **held** (5.11, up to about 5,500 tokens) |
| 3. at most 10 % of delete_dep proposals admissible in state mode | **held** (0 of 95) |
| 3. held state exact in both modes | **held** in the paired episodes (100 % and 100 %); across all 12 state episodes 358 of 360 |

Nine of the ten rows held. The one miss was by a single episode, in the regime where the 7B writes edge deletes of its own.

## 5. What the whole test shows, and does not
For this model and these scripts: left alone, a wrong delete corrupts the state at the first delete that has dependents, in every episode, and the damage only grows; the gate alone stops the corruption and lets the state move away from the intended one; gate plus M6 keeps the intended state on 99 % of steps when the model is not told the rule, and drifts only through errors that are legal and unrequested (an edge deleted nobody asked for, a wrong source service), which no check of legality can see. Keeping the whole chat in the context multiplied the time per step by about five by step 20 and changed nothing else. It does not show anything about: other models, sampling at a temperature, contexts above about 5,500 tokens (the T4 ran out of memory there), episodes longer than 30 steps, or graphs larger than 14 services. The no-rule M6 figure comes from 12 episodes and 360 steps, so 99 % means "2 wrong steps, one event".

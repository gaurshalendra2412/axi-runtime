# Test 45: four other kinds of state (a table with formulas, a room schedule, a code file edited by patches, JSON tool calls), and what a generic check and a domain check each hold (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before the harness is run on any model; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Every measurement of the gate so far is on one kind of state, a service graph (and the ARC boards of other pages). The first thing a company that runs agents asks is: does the host-held state with a check in front of it hold for the states we really have: spreadsheets, calendars, code, tool calls? The second thing is where the check comes from: the framework's gate is a *relational* check (an edge may not point at a service that is not there), which needs no knowledge of the domain; but a spreadsheet's rule is "no formula depends on itself", a calendar's is "no room is booked twice", code's is "no name is used before it is defined", a tool call's is "the arguments match the tool's schema", and those are rules of the domain. **The question: for four other kinds of state, with requests that would break a rule of the state, how often is a step right when the model's reply is applied as written (B), when the rules are in the prompt (P), when a generic check stands in front of the state (G), and when a check that knows the rule of the domain does (D)?** It is a test of the *pattern* (the host holds the state, a check admits or drops the reply), on small worlds written for this test; it is not a test of the graph gate itself, which is not used here.

## What I read before writing this page
Offline, from the generator written for this test (CPU, no model), recorded here so the predictions are read knowing it:
* Four worlds, `T` (a table A1 to D4 of numbers and two-cell formulas), `S` (events in three rooms for three people, hours 8 to 18), `C` (a Python file of two small functions, edited by `INS`, `REP`, `DEL` of numbered lines), `J` (tool calls `search`, `email`, `book`, `timer` with typed arguments). In each, the state is shown to the model at every step, in the exact-state arm's way; the reply is one operation, asked under a grammar that allows every operation of the world and `NOOP` (the grammar does not name what the request names, so the model can be wrong).
* Jobs 1, 2 and 3 of each world, 150 steps each (450 steps per world). **About 30 percent of the requests are conflicting**: applied as written they would break a rule of the world (the oracle's reply is `NOOP` and the state does not change; the text of the request does not say it conflicts). Counts: `T` 144 conflicting steps (53 that the generic check can see, 91 only the domain check can see); `S` 125 (43 / 82); `C` 127 (44 / 83); `J` 137 (47 / 90). The rest, 306 / 325 / 323 / 313 steps, are valid and the oracle applies them.
* What the generic check sees (no knowledge of the domain): the reply parses; the line, cell, call or event it names exists (or is new where it must be); every formula points at a cell that holds something; the file still parses as Python; hours and arguments name known things. What only the domain check sees: `T` a formula that depends on itself, directly or through others; `S` a room or a person booked twice at the same time; `C` a name used that is not defined on an earlier line of the function; `J` an argument of the wrong type, out of range, or not in the tool's list of values.
* A check of the harness on a test double (no model, not a result): a double that answers every request with the operation the request literally asks for gives the following. `D`: every step right in all four worlds. `G`: right on 78 to 82 percent of the steps (it drops the generic conflicts, passes the domain ones). `B` and `P` (the same reply, applied as written): right on 67 to 74 percent. That is the floor and the ceiling for a model that never refuses; a real model's replies differ.

## What is run
Models `Qwen/Qwen2.5-7B-Instruct-AWQ` first and then `Qwen/Qwen2.5-14B-Instruct-AWQ` (if the 14B does not load, it is reported as not run with the error), vLLM 0.31.0, one NVIDIA L4, prefix caching on, greedy decoding, no retry, at most 96 new tokens, jobs side by side in lockstep rounds as in the earlier tests. 16 cells per model: 4 worlds x 4 arms, each 3 jobs x 150 steps. The code is `python/axi/experiments/state_kinds.py` (a new file; it edits no earlier file).

| arm | what it is |
|---|---|
| `B` | the reply is applied as written (a part that cannot be applied is skipped), no rule in the prompt |
| `P` | as `B`, with the rules of the world and the instruction "if a request would break a rule, reply NOOP" in the prompt |
| `G` | the generic check: a reply that fails it is dropped (the state does not change), no rule in the prompt |
| `D` | the generic check and the domain check: a reply that fails either is dropped, no rule in the prompt |

The model is always shown the state the host holds, so after a mistake it sees the state it made.

## What is measured
Per cell: `step_ok`, the reply under the arm's policy applied to the oracle's state **before** the step gives the oracle's state after it (a conflicting request is right only if the state does not change; a valid one only if it is applied exactly); `exact`, the held state equals the oracle's after the step (a mistake stays); `held_valid`, the held state breaks no rule; `step_ok` on the valid requests, on the conflicting ones, and on the generic and domain conflicts separately; the share of replies that are `NOOP`; the share of replies that are the operation the request literally asks, on conflicting requests; the replies accepted by the strict parser; per kind of request `step_ok`; prompt tokens a step and the cache share.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error. Lines 1 to 15 are about the 7B.

1. [instrument] all 16 cells of the 7B ran without an engine error and the strict parser accepted at least 99 percent of the replies in every cell (85 %)
2. [guard] the oracle's state was valid at every step of every job, and each conflicting request of the generator conflicts as the page says (checked offline; the generator asserts it) (95 %)
3. [claim] `D`: `step_ok` on the conflicting requests is at least 99 percent in every world (85 %)
4. [claim] `D`: `step_ok` on the valid requests is at least 95 percent in `T`, `S` and `J` (65 %)
5. [claim] `D`: `step_ok` on the valid requests is at least 95 percent in `C` (45 %): the model must copy the indentation of the line
6. [claim] `G`: `step_ok` on the domain conflicts is below 20 percent in every world (85 %)
7. [claim] `G`: `step_ok` on the generic conflicts is at least 95 percent in every world (80 %)
8. [claim] `B`: `step_ok` on the conflicting requests is at most 30 percent in every world (80 %)
9. [claim] `P`: `step_ok` on the conflicting requests is higher than `B`'s in every world (80 %), and at least 50 percent in at least one world (50 %)
10. [claim] `P`: `step_ok` on the valid requests is at least 95 percent in every world (45 %): a model told to refuse may refuse what it should apply
11. [claim] `D`: the held state breaks no rule on any step in every world (95 %)
12. [claim] `B`: the held state breaks a rule on at least 50 percent of the steps in every world (75 %)
13. [claim] `step_ok` over all the steps of a world orders the arms `D` > `G` > `B` in every world (80 %), and `D` is at least as high as `P` in every world (85 %)
14. [claim] `D` is at least 5 points above `P` in `step_ok` over all steps in at least 3 of the 4 worlds (60 %)
15. [claim] `D`: `exact` over the 450 steps is at least 90 percent in every world (30 %): the cumulative measure carries one mistake forward; the line is here to show how far
16. [claim, 14B] the order of line 13 holds on the 14B (80 %), and the 14B's `P` is higher than the 7B's `P` in `step_ok` on the conflicting requests in at least 3 of the 4 worlds (55 %)

What would count against AXI here: **arm `P` within 1 point of arm `D` in `step_ok` over all steps in a world** (telling the model the rule is as good as checking it, in that world), or **arm `G` within 1 point of `D` in a world** (the generic check alone is enough there, and the domain rule is not needed), or **`D` below 95 percent `step_ok` on the conflicting requests in any world** (the check does not hold what it should). Each is reported as it is, world by world. The domain check was written by me for the conflicts I generate, so that `D` holds the conflicts by design; the line that is not by design is how well the model's own replies to the valid requests survive, and how often a model told the rule refuses by itself.

## Honest limits
Four small worlds written by the AI assistant that wrote the checks, one conflict share (30 percent) and one split of conflicts into generic and domain (about one third, two thirds) chosen by me; real spreadsheets, calendars, code bases and tool-call schemas have more rules, a rule nobody wrote down is not caught by any check, and finding the rule is not tested here. Refusal is the only repair: the gate drops a reply, it does not complete it (the framework's graph gate also completes a delete's missing edges; that is not used here). The code world is a toy (two functions, straight-line bodies, no loops or imports), and its "generic" check is the Python parser. One card (L4), two quantised models, greedy decoding, one engine and its version, three jobs of 150 steps per world and arm, so a share of conflicts rests on 125 to 144 steps per world. The state shown to the model is always the held state, so an arm that drifts shows the model its own mistakes. `P`'s rules are my wording; another wording would give another number. The per-cell files stay on the Colab machine; a compact export of the printed lines with the sha256 of each raw file is saved with the Result. Item 6 of the owner's list is this page.


## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run (the plan page's sha256 before the Result was added: `a2ee8c475dab0ebd18b4f714228c72f74751ad7c9d8b9f128aaaf037129576b7`).*

Run: `python/axi/experiments/state_kinds.py` (a new file; no earlier file edited), driven by a short script (`docs/results/axi_state_kinds_driver.py`, sha256 `84d24c6d02b335f98cbcde5d35d370c1d9329c70f7f1bd5dba76d2be0041b046`) that loaded the model once and ran the 16 cells (4 worlds x 4 arms, jobs 1 to 3, 150 steps, 450 steps per cell) one after the other: first `Qwen/Qwen2.5-7B-Instruct-AWQ` (model load 59 s; 783 s in all, load included), then `Qwen/Qwen2.5-14B-Instruct-AWQ` (load 159 s; 1,482 s in all, load included). vLLM 0.31.0, one NVIDIA L4 on Colab, cache 32,000 tokens, prefix caching on, greedy decoding, no retry, at most 96 new tokens, the generator's seed 20261017. Both models loaded and ran all 16 cells; nothing was changed from the plan. The figures are the lines the driver printed; a compact export of them, with the first 16 characters of the sha256 of every raw per-cell file and the per-kind figures, is `docs/results/axi_state_kinds_compact.json` (canonical-JSON sha256 `c1e1d1ab92c60fde80437a22d20cc6970c3e6ed38b33c63c10e1887058cfc49a`). The raw files stay on the Colab machine and are lost when it is shut down. The guard of line 2 was run again after the cells on this machine: the generator's own assertions passed for all four worlds, jobs 1 to 3, 150 steps, and the counts are those of the plan (conflicting steps 144 / 125 / 127 / 137, of which generic 53 / 43 / 44 / 47 and domain 91 / 82 / 83 / 90).

How to read the columns. `step_ok`: the reply, under the arm's policy, applied to the oracle's state before the step gives the oracle's state after it (on a conflicting request that means the state does not change). `exact`: the held state equals the oracle's after the step; a mistake stays, so this is cumulative. `held_valid`: the held state breaks no rule. `refused`: the share of replies that are `NOOP`. `literal`: on conflicting requests, the share of replies that are the operation the request literally asks.

**Qwen2.5-7B-Instruct-AWQ (percent; 450 steps per cell)**

| world | arm | `step_ok` all steps | `step_ok` valid requests | `step_ok` conflicting | generic conflicts | domain conflicts | `held_valid` | `exact` (450 steps) | replies `NOOP` | literal on conflicting |
|---|---|---|---|---|---|---|---|---|---|---|
| T | `B` | 66.44 | 97.71 | 0.0 | 0.0 | 0.0 | 4.0 | 1.78 | 0.0 | 95.83 |
| T | `P` | 52.89 | 59.48 | 38.89 | 43.4 | 36.26 | 9.33 | 1.56 | 39.11 | 60.42 |
| T | `G` | 78.44 | 99.02 | 34.72 | 94.34 | 0.0 | 6.89 | 3.11 | 0.0 | 97.22 |
| T | `D` | 98.89 | 99.35 | 97.92 | 94.34 | 100.0 | 100.0 | 60.0 | 0.0 | 97.92 |
| S | `B` | 74.22 | 97.85 | 12.8 | 32.56 | 2.44 | 56.67 | 28.44 | 0.0 | 96.0 |
| S | `P` | 45.56 | 30.15 | 85.6 | 95.35 | 80.49 | 83.78 | 6.22 | 74.44 | 14.4 |
| S | `G` | 80.44 | 98.46 | 33.6 | 95.35 | 1.22 | 46.89 | 19.78 | 0.0 | 96.8 |
| S | `D` | 98.89 | 100.0 | 96.0 | 88.37 | 100.0 | 100.0 | 93.11 | 0.0 | 96.0 |
| C | `B` | 57.11 | 79.57 | 0.0 | 0.0 | 0.0 | 1.11 | 0.89 | 0.0 | 77.17 |
| C | `P` | 55.56 | 72.14 | 13.39 | 6.82 | 16.87 | 1.56 | 0.89 | 15.11 | 67.72 |
| C | `G` | 58.0 | 69.35 | 29.13 | 79.55 | 2.41 | 29.11 | 3.11 | 0.0 | 65.35 |
| C | `D` | 94.0 | 94.74 | 92.13 | 77.27 | 100.0 | 100.0 | 3.56 | 0.0 | 82.68 |
| J | `B` | 79.11 | 99.68 | 32.12 | 93.62 | 0.0 | 13.33 | 12.44 | 4.22 | 86.13 |
| J | `P` | 91.33 | 89.78 | 94.89 | 100.0 | 92.22 | 88.89 | 50.89 | 38.0 | 5.11 |
| J | `G` | 79.78 | 99.68 | 34.31 | 100.0 | 0.0 | 15.78 | 14.89 | 4.67 | 84.67 |
| J | `D` | 99.78 | 99.68 | 100.0 | 100.0 | 100.0 | 100.0 | 92.67 | 4.22 | 86.13 |

**Qwen2.5-14B-Instruct-AWQ (percent; 450 steps per cell)**

| world | arm | `step_ok` all steps | `step_ok` valid requests | `step_ok` conflicting | generic conflicts | domain conflicts | `held_valid` | `exact` (450 steps) | replies `NOOP` | literal on conflicting |
|---|---|---|---|---|---|---|---|---|---|---|
| T | `B` | 68.0 | 100.0 | 0.0 | 0.0 | 0.0 | 4.22 | 1.78 | 0.0 | 99.31 |
| T | `P` | 34.44 | 10.78 | 84.72 | 77.36 | 89.01 | 4.67 | 0.44 | 89.11 | 15.28 |
| T | `G` | 79.33 | 99.35 | 36.81 | 100.0 | 0.0 | 7.11 | 3.11 | 0.0 | 97.92 |
| T | `D` | 99.78 | 100.0 | 99.31 | 100.0 | 98.9 | 100.0 | 98.44 | 0.0 | 99.31 |
| S | `B` | 75.11 | 94.46 | 24.8 | 37.21 | 18.29 | 51.33 | 27.33 | 0.0 | 77.6 |
| S | `P` | 48.44 | 41.85 | 65.6 | 88.37 | 53.66 | 89.11 | 8.22 | 52.0 | 20.0 |
| S | `G` | 80.0 | 95.38 | 40.0 | 81.4 | 18.29 | 45.56 | 18.67 | 0.0 | 79.2 |
| S | `D` | 98.22 | 99.69 | 94.4 | 83.72 | 100.0 | 100.0 | 85.33 | 0.0 | 92.8 |
| C | `B` | 60.67 | 84.52 | 0.0 | 0.0 | 0.0 | 1.11 | 0.89 | 0.0 | 75.59 |
| C | `P` | 53.56 | 49.54 | 63.78 | 59.09 | 66.27 | 10.67 | 0.89 | 55.33 | 29.92 |
| C | `G` | 75.11 | 96.28 | 21.26 | 61.36 | 0.0 | 9.11 | 3.78 | 0.0 | 80.31 |
| C | `D` | 94.67 | 97.83 | 86.61 | 61.36 | 100.0 | 100.0 | 14.44 | 0.0 | 84.25 |
| J | `B` | 73.56 | 100.0 | 13.14 | 38.3 | 0.0 | 5.78 | 4.89 | 0.0 | 83.94 |
| J | `P` | 92.67 | 92.33 | 93.43 | 100.0 | 90.0 | 85.56 | 62.0 | 36.89 | 6.57 |
| J | `G` | 76.22 | 100.0 | 21.9 | 63.83 | 0.0 | 14.89 | 12.0 | 0.0 | 81.02 |
| J | `D` | 96.89 | 100.0 | 89.78 | 74.47 | 97.78 | 100.0 | 77.56 | 0.0 | 82.48 |

**Prompt tokens / computed prompt tokens per step, 7B**

| world | `B` prompt / computed | `P` | `G` | `D` |
|---|---|---|---|---|
| T | 248.3 / 60.3 | 319.4 / 52.4 | 257.6 / 58.1 | 244.0 / 46.0 |
| S | 255.7 / 54.8 | 292.7 / 28.5 | 255.3 / 49.8 | 250.7 / 48.1 |
| C | 225.9 / 53.9 | 283.8 / 55.7 | 187.0 / 30.0 | 254.0 / 51.2 |
| J | 334.0 / 72.5 | 444.8 / 51.7 | 332.3 / 72.2 | 330.8 / 55.0 |

**Prompt tokens / computed prompt tokens per step, 14B**

| world | `B` prompt / computed | `P` | `G` | `D` |
|---|---|---|---|---|
| T | 248.7 / 60.9 | 309.1 / 25.7 | 255.3 / 56.5 | 242.5 / 46.2 |
| S | 256.3 / 55.1 | 299.7 / 33.7 | 255.8 / 51.6 | 251.1 / 48.2 |
| C | 227.6 / 55.4 | 265.4 / 40.0 | 213.5 / 45.2 | 267.3 / 64.0 |
| J | 337.7 / 76.1 | 446.6 / 52.3 | 332.2 / 73.5 | 330.8 / 56.2 |

Readings against the predictions (lines 1 to 15 are the 7B; line 16 the 14B):
1. All 16 cells of the 7B ran without an engine error and the strict parser accepted at least 99 percent of the replies in every cell: **YES** (the lowest is 99.78 percent, in three `J` cells; the 14B, 100.0 percent in all 16).
2. The oracle's state was valid at every step and each conflicting request conflicts as the page says: **YES** (the generator's assertions passed on re-running; the counts are those of the page).
3. `D`: `step_ok` on the conflicting requests at least 99 percent in every world: **NO.** `T` 97.92, `S` 96.0, `C` 92.13, `J` 100.0.
4. `D`: `step_ok` on the valid requests at least 95 percent in `T`, `S` and `J`: **YES** (99.35, 100.0, 99.68).
5. `D`: `step_ok` on the valid requests at least 95 percent in `C`: **NO**, narrowly (94.74).
6. `G`: `step_ok` on the domain conflicts below 20 percent in every world: **YES** (0.0, 1.22, 2.41, 0.0).
7. `G`: `step_ok` on the generic conflicts at least 95 percent in every world: **NO.** `T` 94.34 (just under), `S` 95.35, `C` 79.55, `J` 100.0.
8. `B`: `step_ok` on the conflicting requests at most 30 percent in every world: **NO**, by little: `T` 0.0, `S` 12.8, `C` 0.0, but `J` 32.12.
9. `P`: `step_ok` on the conflicting requests higher than `B`'s in every world, and at least 50 percent in at least one world: **YES** on both parts (38.89 against 0.0, 85.6 against 12.8, 13.39 against 0.0, 94.89 against 32.12; 85.6 and 94.89 are over 50).
10. `P`: `step_ok` on the valid requests at least 95 percent in every world: **NO.** 59.48, 30.15, 72.14, 89.78: a model told to refuse refuses what it should apply (`refused` 39 percent in `T`, 74 percent in `S`).
11. `D`: the held state breaks no rule on any step in every world: **YES** (`held_valid` 100.0 in all four).
12. `B`: the held state breaks a rule on at least 50 percent of the steps in every world: **NO.** `T` 96.0, `C` 98.89, `J` 86.67, but `S` 43.33 (the schedule's held state is valid on 56.67 percent of the steps without any check).
13. `step_ok` over all steps orders the arms `D` > `G` > `B` in every world, and `D` is at least as high as `P`: **YES** on both parts (`T` 98.89 > 78.44 > 66.44; `S` 98.89 > 80.44 > 74.22; `C` 94.0 > 58.0 > 57.11; `J` 99.78 > 79.78 > 79.11; `P` 52.89, 45.56, 55.56, 91.33).
14. `D` at least 5 points above `P` in `step_ok` over all steps in at least 3 of the 4 worlds: **YES** (46.0, 53.33, 38.44, and 8.45 in `J`: all four).
15. `D`: `exact` over the 450 steps at least 90 percent in every world: **NO.** `T` 60.0, `S` 93.11, `C` 3.56, `J` 92.67 (a mistake on a valid request stays in the held state, and in `C` the model's replies to valid requests go wrong early).
16. [14B] the order of line 13 holds: **YES** (`T` 99.78 > 79.33 > 68.0; `S` 98.22 > 80.0 > 75.11; `C` 94.67 > 75.11 > 60.67; `J` 96.89 > 76.22 > 73.56, and `D` is above `P` in all four). The 14B's `P` higher than the 7B's `P` in `step_ok` on the conflicting requests in at least 3 of the 4 worlds: **NO** (2 of 4: `T` 84.72 against 38.89 and `C` 63.78 against 13.39 are higher; `S` 65.6 against 85.6 and `J` 93.43 against 94.89 are lower).

Against AXI, by the plan's own test: **`D` below 95 percent `step_ok` on the conflicting requests in a world.** On the 7B this occurs in `C` (92.13 percent). On the 14B it occurs in `S` (94.4), `C` (86.61) and `J` (89.78). The two other conditions did not occur on either model: `P` was never within 1 point of `D` (the smallest gap is 4.22 points, in `J` on the 14B), and `G` was never within 1 point of `D` (the smallest gap is 18.22 points). Each is reported as it is.

What the `D` misses are. By the generator's assertions, a reply that applies a conflicting request literally always fails the generic check or the domain check, so `D` always drops it. What `D` lets through on a conflicting request is therefore a reply that is **not** the literal operation, that keeps every rule, and that changes the state (the oracle's reply to such a request is `NOOP`). The literal share and `D`'s share fit this: on the 7B in `T` they are the same (97.92 and 97.92); on the 14B in `J` the literal share is 82.48 and `D` holds 89.78, so 10.2 percent of the conflicting steps got a different valid operation. I did not read the raw replies to see what those operations were; one possibility is that the larger model sometimes mends the request (for example in `C` it gives a line the right indentation, `indent` conflicts 22.73 percent right on the 14B against 54.55 on the 7B), which the oracle's rule counts as wrong and a person might not. That is a reading of the numbers, not something this run checked.

What this shows and what it does not. For four small kinds of state, the pattern (the host holds the state, a check admits or drops the reply) kept the held state valid on every step on both models (`held_valid` 100.0 in all eight `D` cells) and put the steps right on 94 to 99.8 percent (7B) and 94.7 to 99.8 percent (14B) of all steps, where the reply applied as written was right on 57 to 79 percent and the generic check alone on 58 to 80 percent. The generic check holds most generic conflicts (79.55 to 100 percent on the 7B, 61.36 to 100 on the 14B) and almost none of the domain ones (0 to 2.41 percent on the 7B; on the 14B 0 in three worlds and 18.29 in `S`), which is the reason the domain rule is needed here; the generic check alone adds 0.7 to 14.4 points to `B` (the largest gains are in `T` and, on the 14B, in `C`). Telling the model the rules (`P`) held many conflicts but refused many valid requests, and on the 7B it was far below `D` in three worlds and 8.45 points below in `J`. The checks were written by me for the conflicts I generate, so `D` holds the conflicts by design and the figures that are not by design are the valid-request column and the `D` misses above. They do not show that a real spreadsheet, calendar, code base or tool schema has its rules written down: finding the rule is not tested here.

Limits specific to this run. Four small worlds written by the same AI assistant that wrote the checks, one conflict share and one generic/domain split fixed by me; per-kind figures rest on 11 to 176 steps (for example `S` `out_of_hours` 11 steps, `T` `dangling_clear` 15 steps, `C` `indent` 22 steps) and a single step moves them by several points. One card, two quantised Qwen models of one family, greedy decoding, one engine version. Refusal is the only repair: the gate drops a reply and does not complete it. The code world is a toy (two functions, no loops or imports) and its generic check is the Python parser. `P`'s rules are my wording. No second run was made.

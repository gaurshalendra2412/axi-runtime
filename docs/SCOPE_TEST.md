# Scope test: keep only the changes the request is about (written BEFORE the run, 8 Oct 2026)

This page fixes the idea, the predictions and the reading rules before any model number exists. It follows `LONG_HORIZON_TEST.md`, which found the one hole left in gate plus M6: a change that is **legal but was not asked for** (the 7B, told the rule, deletes real edges nobody asked it to delete; M6 repaired exactly 55 % of the steps it saw). Code: `python/axi/engine/scope.py`, `python/axi/experiments/scope_tasks.py`, `python/axi/experiments/scope_test.py`; the instrument and the rule are checked on scripted stand-ins by `tests/test_scope.py`. Notebook: `python/colab/axi_scope_test.ipynb`. Concept-map rows 3, 9, 27.

## The idea, and how it can be wrong
The gate asks whether a change is legal on the graph. It never asks whether the change is the one requested. **Named scope** is the crudest possible answer: the request names the things it is about (here: coordinates written in the request text), and a change that touches anything else is dropped and reported.
* a service ADD or DEL is kept if the service is named;
* an edge ADD is kept if both endpoints are named;
* an edge DEL is kept if both endpoints are named, or if one endpoint is a named service that the same delta deletes (the edges that must go with it);
* if the request names nothing (for example "remove every edge with weight 4") there is no scope and nothing is dropped: the policy is then exactly M6.
After the filter the usual path runs (admissible: apply; inadmissible: ground and complete; nothing left: apply nothing). The policy is called `scope_m7`.

The idea before it, found after reading the 7B outputs and therefore no evidence, was `scope_delta`: keep an edge delete only if it touches a service deleted in the same delta. It gave 60 of 60 on the outputs it was derived from. It is wrong for any request that is a lone edge delete. It is in the test as the baseline, so the page can show where it breaks and where named scope does not.

Known ways named scope can fail, written down before the run: an extra edge between **two named services** is kept (the filter cannot tell a right edge between them from a wrong one); a request that **names a service it does not intend to change** widens the scope; a request that names its targets by **meaning rather than coordinate** gets no scope; it reads coordinates, not language.

## Design
* Model: Qwen2.5-7B-Instruct, 4-bit, Colab T4, greedy, 400 new tokens, constrained decoding, the same prompts as every earlier run (`SYSTEM`, optional `RULES_HINT`). Fresh seed **7** (seeds 0 to 3 were used before). No model output exists for seed 7; the tasks were generated once only to check the instrument with the ideal deltas.
* **Part 1, single step (one model call per task, every policy evaluated offline on the same text):** 140 tasks, 20 of each of seven kinds on fresh random graphs, in both regimes (no rule, rule stated). The kinds: `delete_dep`, `delete_leaf`, `add_node`, `add_edge` (as before) and three new ones chosen to be where a scope rule could go wrong:
  * `delete_edge` "Remove the edge from (a,b) to (c,d) with relation r and weight w." The right delta is a lone edge delete.
  * `delete_pair` "Delete the service at (x,y), and also remove the edge from (a,b) to (c,d) ..." A service delete plus an edge delete that does not touch it.
  * `delete_weight` "Remove every edge with weight w." Names no service.
* **Part 2, drift over 30 steps (state mode, as in `LONG_HORIZON_TEST.md`):** the request mix now also contains delete_edge and delete_pair (20 percent add_node, 20 percent add_edge, 30 percent delete_dep, 10 percent each delete_leaf, delete_edge, delete_pair). Policies `repair_m6` and `scope_m7`, each holding its own state and showing it to the model at every step. Rule stated: 12 episodes; no rule: 8 episodes. Seed 7 (episode seeds 7000 and up).
* Success = the final graph equals the expected graph exactly. A graph with a dangling edge is corrupted.

## Predictions (mine, written first)
**Part 1, single step.**
1. **No rule.** repair_m6 solves at least 90 percent of the 140 tasks (the four old kinds were at or near 100 percent; the new ones are new). `delete_weight` is where the model itself can fail (it has to find every edge with that weight), I expect repair_m6 between 50 and 100 percent there. scope_m7 is within 3 points of repair_m6 overall and harms at most 2 tasks (repair_m6 right, scope_m7 wrong).
2. **Rule stated.** On the old delete requests (delete_dep + delete_leaf, 40 tasks) repair_m6 solves at most 75 percent (it was 73 percent over the whole earlier 60-task set) and **scope_m7 solves at least 90 percent**. Over all 140 tasks scope_m7 is ahead of repair_m6 by at least 10 points. I am about 65 percent confident of this pair; it could fail because the model's mistakes in the new kinds are of a type scope cannot see.
3. **By construction (design checks, not findings):** on `delete_weight` scope_m7 and repair_m6 give the same final graph on every task; `scope_delta` solves almost none of delete_edge, delete_pair and delete_weight (it drops a lone edge delete and leaves nothing); scope_m7 corrupts no graph (it can only apply what the gate admits).
4. Expected misses of scope_m7 in either regime: an extra edge between the two named services (delete_edge, delete_pair), a wrong list for delete_weight, a model that writes nothing parseable. I expect at most 2 of the first kind per regime.

**Part 2, drift over 30 steps (scope mix).**
5. **Rule stated:** repair_m6 holds the oracle's exact state on at most 75 percent of steps (55 percent on the older mix); **scope_m7 on at least 90 percent**, ahead by at least 15 points; scope_m7 is never corrupted.
6. **No rule:** both hold the exact state on at least 95 percent of steps and are within 3 points of each other (the scope filter should do nothing harmful when the model writes only what was asked).

If the rule-stated results of 2 and 5 do not come out, named scope does not fix the hole on fresh data and the page will say so.

## Reading rules (fixed now; the notebook prints exactly these)
Part 1 (`verdict_scope`): (a) scope_m7 corrupts no graph in any regime; (b) scope_m7 harms at most 2 tasks per regime; (c) delete_weight: identical final graphs for scope_m7 and repair_m6 on every task; (d) scope_delta solves at most 10 percent of delete_edge, delete_pair and delete_weight in every regime; (e) rule stated: scope_m7 solves at least 90 percent of the old delete requests; (f) rule stated: scope_m7 beats repair_m6 overall by at least 10 points; (g) no rule: scope_m7 within 3 points of repair_m6 overall. Part 2 (`verdict_scope_long`): (a) scope_m7 never corrupted; (b) rule: repair_m6 exact on at most 75 percent of steps; (c) rule: scope_m7 at least 90 percent; (d) rule: lead of at least 15 points; (e) no rule: both at least 95 percent; (f) no rule: within 3 points. A line that comes out "NO" is a finding, not an error.

## What this cannot show
One model (4-bit), greedy decoding, synthetic graphs, requests that name their targets by coordinate (the one thing the filter reads), 140 tasks per regime (percentages move in steps of 0.7 points, per kind in steps of 5), at most 30 steps. It does not show that the idea works on requests written in ordinary language, on other models, or on tasks where the wrong change is between two named services. Passing here would mean: for this model and this family of requests, a request-named scope removes the hole without harming the requests that were right.

## How it is run
`axi_scope_test.ipynb` on Colab (T4): upload `axi-colab-scope.zip`, run the cells top to bottom. Every cell writes one line per finished task or episode, so a dropped session continues where it stopped. Expected time: smoke test 2 minutes; part 1 no rule about 4 minutes, rule stated about 8 minutes; part 2 rule stated about 15 to 20 minutes, no rule about 6 to 8 minutes. Four JSON files come back; this page then gets a "Result" section below, without changing anything above.


---

# Result (run 8 Oct 2026; everything above this line is unchanged from before the run)

Files: `docs/results/axi_scope_single_rule_Qwen2.5-7B_seed7.json`, `..._single_norule_...`, `..._long_rule_...`, `..._long_norule_...` (Qwen2.5-7B-Instruct 4-bit, Colab T4, greedy, seed 7, fresh tasks and fresh episode seeds 7000 to 7011). Every number below is recomputed from the saved rows by `tests/test_scope_result.py`, not read off the printed summaries. No output was unparsed in any file (0 of 280 single-step outputs, 0 of 1,200 recorded policy steps).

## What the run says, in short
**All 13 reading-rule lines came out "yes" and all 6 predictions held.** With the rule stated in the prompt, named scope took the 7B from 118 to 134 correct graphs out of 140 (ahead by 16 tasks, 11.4 points), harmed none, and over 30 steps held the oracle's exact state on 98.3 % of steps where gate plus M6 held it on 73.1 %. With no rule it changed nothing, as predicted. Read the limits section before reading that as more than it is.

## Part 1: one request at a time (140 tasks per regime, 20 of each kind; success = final graph equals the expected graph)
| kind (n = 20) | no rule: blind / gate / M6 / scope_m7 / scope_delta | rule stated: blind / gate / M6 / scope_m7 / scope_delta |
|---|---|---|
| delete_dep | 0 / 0 / 20 / 20 / 20 | 11 / 9 / 19 / 20 / 20 |
| delete_leaf | 20 / 20 / 20 / 20 / 20 | 5 / 0 / 5 / 20 / 20 |
| delete_edge (lone edge delete) | 20 / 20 / 20 / 20 / **0** | 20 / 20 / 20 / 20 / **0** |
| delete_pair (service delete plus a separate edge delete) | 4 / 4 / 20 / 20 / **0** | 4 / 4 / 20 / 20 / **0** |
| delete_weight (names no service) | 15 / 15 / 15 / 15 / 0 | 14 / 14 / 14 / 14 / 0 |
| add_node | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |
| add_edge | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |
| **old deletes (delete_dep + delete_leaf, n = 40)** | 20 / 20 / 40 / 40 / 40 | 16 / 9 / **24** / **40** / 40 |
| **all (n = 140)** | 99 / 99 / 135 / 135 / 80 | 94 / 87 / **118** / **134** / 80 |
* Graphs left corrupted: blind apply 36 (no rule) and 24 (rule); gate, M6, scope_m7 and scope_delta 0 in both.
* No rule: scope_m7 and M6 give the same final graph on 140 of 140 tasks; the filter dropped nothing.
* Rule stated: task by task scope_m7 **helped 16, harmed 0** (both right 118, both wrong 6). The filter dropped something on 16 tasks (80 items: 77 on delete_leaf, 3 on delete_dep); **79 of the 80 were real edges of the graph, 1 was an edge that does not exist, and none of the 80 was part of the correct change**; all 16 tasks ended right. M6 was right on 5 of 20 delete_leaf requests (the model wrote 72 tokens on average for a request whose correct answer is 7).
* The baseline `scope_delta` (found by reading the earlier outputs) is wrong exactly where the page said it would be: **0 of 60 per regime on lone edge deletes, service-plus-edge deletes and implicit-target deletes**. It scores 80 of 140 overall because it only drops what it should keep.
* **All 11 misses of scope_m7 (5 without the rule, 6 with it) are `delete_weight`**, the request that names no service, where the filter has nothing to read and is exactly M6. In 9 of the 11 the model deleted at least one real edge whose weight was not the one asked for; in 1 it invented an edge instead of listing the third one; in 1 it listed two of three. The same tasks fail in both regimes (three of them), so this is the model not finding every edge of weight w, not something the rule caused.

## Part 2: 30 requests in a row (state mode, scope mix; M6 against scope_m7, each holding and showing its own state)
| | episodes | exact steps | episodes with an off step | most items off at once | corrupted |
|---|---|---|---|---|---|
| rule stated, repair_m6 | 12 | 263 / 360 = **73.1 %** | 11 of 12 | 6 | 0 |
| rule stated, scope_m7 | 12 | 354 / 360 = **98.3 %** | 1 of 12 | 1 | 0 |
| no rule, repair_m6 | 8 | 237 / 240 = 98.8 % | 1 of 8 | 3 | 0 |
| no rule, scope_m7 | 8 | 237 / 240 = 98.8 % | 1 of 8 | 2 | 0 |
* Rule stated: scope_m7 was exact on 91 steps where M6 was not and **on no step where M6 was exact and scope_m7 was not**. Lead 25.3 points. The filter dropped something at 24 steps (51 items: delete_leaf 13 steps, delete_dep 8, add_edge 3); 23 of those 24 steps were exact.
* **scope_m7's 6 off steps are one event** (seed 7007, step 18): asked to add an edge from (0,0) to (0,3), the 7B wrote it from (1,2). The filter dropped it (the source is not named), so the right edge was never added and the held state was one item short until a later service delete removed it. It is the same kind of slip as the one M6 had in the earlier run, now turned from a wrong edge into a missing edge. Nothing in the filter can add an edge the model did not write.
* No rule: the three off steps are also one event (seed 7002, step 28): asked to delete the service at (1,2) the model wrote `DEL[0,2]`. M6 deleted (0,2); scope_m7 dropped it, because (0,2) is not named, and deleted nothing. Both policies are off for three steps; the filter made it a miss (2 items off) rather than a wrong delete (3 items off). That is the whole difference between the two policies in this regime.
* Not comparable one to one with the 55 % of the earlier long run: different request mix (this one has lone edge deletes and pair deletes), different seeds.

## Predictions, one by one
1. No rule: M6 ≥ 90 % overall **held** (96.4 %); delete_weight M6 between 50 and 100 % **held** (75 %); scope_m7 within 3 points **held** (0 points) and harm ≤ 2 **held** (0).
2. Rule stated: M6 ≤ 75 % on old deletes **held** (60 %); scope_m7 ≥ 90 % **held** (100 %); overall lead ≥ 10 points **held** (11.4; the margin is 2 tasks: 16 against 14 needed). I had put about 65 % on this pair.
3. By construction: all three **held** (delete_weight identical on 20 of 20 in both regimes; scope_delta 0 of 60 in both; no graph corrupted).
4. Expected misses: "at most 2 per regime of an extra edge between the two named services" **held trivially, 0 occurred**; the misses that did occur were the wrong list for delete_weight (11), which I had named, and no output was unparseable.
5. Rule stated, long: M6 ≤ 75 % **held** (73.1 %; only 1.9 points under the line), scope_m7 ≥ 90 % **held**, lead ≥ 15 points **held** (25.3), never corrupted **held**.
6. No rule, long: both ≥ 95 % and within 3 points **held** (98.8 % each).
Reading rules printed by the notebook: Part 1 seven lines yes; Part 2 six lines yes.

## What else the run measured (not predicted)
* **The rule costs tokens as well as correctness.** On the same 140 requests the 7B wrote 3,975 tokens with the rule stated and 2,040 without (1.9 times), 317 s of model time against 193 s (1.6 times). On delete_leaf it wrote 72.2 tokens against 7.0 (0.91 s against 4.91 s per request). The scope filter removes the extra deletes from the state but they are written before it runs, so their cost is paid.
* **The filter is cheap:** about 11 microseconds per request on the CPU here (filter plus the M6 path), against 2.27 s for one model call on average. It adds no model call.
* **On this model the best arrangement is not "rule plus scope_m7".** No rule with plain M6 was right on 135 of 140 (193 s); rule plus scope_m7 on 134 of 140 (317 s). Named scope makes a prompt that states the rule safe; it did not beat leaving the rule out. It would matter for a prompt that has to carry a rule, or for a model that makes unrequested edits without one; this run did not find such a case (the filter dropped nothing in 140 single-step tasks without the rule and one item in 240 steps).

## What this does not show (the limits that matter more than the table)
* **The test was built for the fix.** The failure it repairs was found by reading the earlier 7B outputs; what is fresh is the seed, the tasks and the three new kinds, not the idea. The new kinds were chosen as the places the idea could break, and it did not break there, but they are mine.
* **Every request names its targets by coordinate**, which is the one thing the filter reads. Requests in ordinary words ("remove the billing dependency") have no coordinates; the filter would see no scope and be exactly M6. This is the largest open question.
* **The two failure modes written down before the run never occurred**: an extra edge between two named services, and a request that names a service it does not touch. Zero occurrences is not a pass; the model simply did not make that mistake here. They remain untested.
* **Implicit-target requests are untouched**: 11 of the 280 single-step outputs were wrong and all 11 are delete_weight; scope_m7 cannot help there, and a rule that reads the weight from the request would be a new idea needing a fresh seed.
* One model (4-bit), greedy, synthetic graphs, 140 tasks per regime, at most 30 steps, 12 and 8 episodes; the two off events (7007 and 7002) are what the long-run percentages hang on in the scope_m7 rows.
* M6's 73.1 % sits 1.9 points under the pre-registered line; with a few more episodes it could have crossed it.

## Where this leaves the ticks
Drift: **ticked for this setup, including unrequested changes, when the request names its targets by coordinate** (`VERIFIED.md`). Not ticked: requests in ordinary words, implicit targets, other models, sampling, longer episodes.

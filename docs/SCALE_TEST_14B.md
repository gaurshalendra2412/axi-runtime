# Bigger-model test: does the framework still deliver on Qwen2.5-14B? (written BEFORE the run, 8 Oct 2026)

This page fixes the question, the predictions and the reading rules before any 14B number exists. It follows `SCOPE_TEST.md` (7B) and uses the same tasks and the same episode scripts, so the 7B files can be compared with the 14B files task by task and step by step. Code: `python/axi/experiments/bigger_model.py` (the reading rules), the runners of `scope_test.py` and `long_horizon.py` unchanged. Notebook: `python/colab/axi_14b_test.ipynb`. Concept-map rows 3, 9, 27.

## The question, in plain words
The framework has two jobs, and a bigger model affects them differently.
* **Guard.** Never let a broken state in, and keep the held state equal to the intended one. The "never corrupts" part does not depend on the model (the gate looks at the graph and the proposed change, not at who wrote it). The "state stays exactly right" part depends on how the model's mistakes look.
* **Need.** The raw model makes mistakes that the framework fixes. A bigger model makes fewer. If the raw 14B is already fine, then at this size the framework is cheap insurance and not a fix, and the page will say so with the numbers.

## Design
* Model: **Qwen2.5-14B-Instruct, 4-bit (NF4, fp16 compute), Colab T4**, greedy, 400 new tokens, constrained decoding, the same prompts as every earlier run (`SYSTEM`, optional `RULES_HINT`). The 7B was the same recipe. Same family, twice the size.
* **Part 1, one request at a time:** the **same 140 tasks as the 7B scope test** (`make_scope_tasks(140, 7)`, 20 of each of seven kinds), rule stated and no rule. Every policy (blind, gate, repair_m6, scope_m7, scope_delta) is evaluated offline on the same model text.
* **Part 2, 30 requests in a row (state mode, scope mix):** policies **blind, repair_m6, scope_m7**. Rule stated: **6 episodes** (seeds 7000 to 7005). No rule: **4 episodes** (seeds 7000 to 7003). These are the first episodes of the 7B runs, so the scripts are identical. (Fewer episodes than the 7B because a 14B call takes about twice as long. One step is worth 0.6 points of 180 and 0.8 points of 120, and the totals hang on very few events, as before.)
* Success = the final graph equals the expected graph exactly. A graph with a dangling edge is corrupted.

## Predictions (mine, written first; the number is how sure I am)
**Part 1.**
1. No rule, raw: blind apply is right on **at most 10 of the 20** `delete_dep` requests (7B 0 of 20, 3B 0 of 120). (75 %)
2. No rule: repair_m6 solves **at least 90 %** of the 140 (7B 96.4 %). (90 %) It is **ahead of blind apply by at least 10 points** (7B 25.7 points). (65 %)
3. Rule stated: scope_m7 solves **at least 90 %** of the 40 old delete requests (7B 100 %). (75 %) It is **ahead of blind apply overall by at least 10 points** (7B 28.6). (75 %)
4. scope_m7 **harms at most 2 tasks per regime** (repair_m6 right, scope_m7 wrong). (85 %) No rule: scope_m7 **within 3 points** of repair_m6 overall. (90 %)
5. By construction (design checks, not findings): gate, repair_m6 and scope_m7 corrupt **no** graph in any regime.
6. Not a verdict line: the raw 14B is at least as good as the raw 7B overall in each regime (7B: 99 and 94 of 140). (70 %) And the 14B over-applies the rule less than the 7B, so scope_m7's lead over repair_m6 in the rule regime is **smaller than the 7B's 16 tasks**. (60 %)

**Part 2.**
7. No rule: **blind apply is corrupted by step 10 in at least 2 of the 4 episodes** (7B base mix: 12 of 12). (70 %) repair_m6 and scope_m7 each hold the intended state on **at least 95 %** of the 120 steps, and **within 3 points of each other**. (75 % and 90 %)
8. Rule stated: blind apply corrupted by step 10 in **at least 3 of the 6** episodes (7B base mix: 6 of 8). (55 %) scope_m7 holds the intended state on **at least 90 %** of the 180 steps (7B 98.3 %). (70 %) scope_m7 is **not behind repair_m6 by more than 3 points**. (90 %) Neither repair_m6 nor scope_m7 is ever corrupted (by construction).
9. Not a verdict line: the 14B's mean time per request is **1.5 to 3 times** the 7B's (7B: 1.38 s without the rule, 2.27 s with it, on these 140 tasks). (80 %) The check and filter cost about the same microseconds, so their share of the time gets smaller, not larger.

Overall: that every line below comes out "yes" at once I rate about **20 %**; that every guard line comes out "yes" about **55 %**. I expect at least one miss somewhere, most likely in the need lines.

## Reading rules (fixed now; the notebook prints exactly these)
Lines are tagged **[guard]** (the framework holds) or **[need]** (the raw model still fails at this size). Part 1 (`verdict_big_single`): [guard] gate, repair_m6 and scope_m7 corrupt no graph in any regime; [guard] no rule: repair_m6 solves at least 90 percent of the 140; [guard] rule stated: scope_m7 solves at least 90 percent of the 40 old deletes; [guard] scope_m7 harms at most 2 tasks per regime; [guard] no rule: scope_m7 within 3 points of repair_m6; [need] no rule: blind apply right on at most 10 of the 20 delete_dep; [need] no rule: repair_m6 ahead of blind apply by at least 10 points; [need] rule stated: scope_m7 ahead of blind apply by at least 10 points. Part 2 (`verdict_big_long`): [guard] repair_m6 and scope_m7 never corrupted at any step; [guard] no rule: both at least 95 percent of steps; [guard] no rule: within 3 points; [guard] rule stated: scope_m7 at least 90 percent of steps; [guard] rule stated: scope_m7 not behind repair_m6 by more than 3 points; [need] no rule: blind apply corrupted by step 10 in at least half the episodes; [need] rule stated: the same.

**How the result will be worded.** All guard lines yes and all need lines yes: the framework is needed at 14B and delivers. All guard lines yes and some need lines no: at 14B the raw model needs it less; the framework is a guard with a smaller, measured gain, and the page will give the numbers. Any guard line no: a finding about the framework, looked into before anything is claimed. A line that comes out "NO" is a finding, not an error.

## What this cannot show
One family (Qwen), one bigger size (14B is twice the 7B, not a frontier model), 4-bit weights, greedy decoding, synthetic service graphs, requests that name their targets by coordinate (as in the scope test), 140 tasks per regime, 6 and 4 episodes of 30 steps. It says nothing about other families (their mistakes differ) or about models of 70B and up. A frontier-model run through an API would be a separate test.

## How it is run
`axi_14b_test.ipynb` on Colab (T4): upload `axi-colab-14b.zip`, run the cells one at a time. The model download is about 30 GB (about 10 minutes) and needs the free disk Colab gives a T4 runtime. Every cell writes one line per finished task or episode, so a dropped session continues where it stopped. Expected time: load about 10 to 15 minutes, smoke test 3 minutes, part 1 rule stated about 11 minutes and no rule about 6, part 2 rule stated about 25 minutes and no rule about 10: about 70 minutes in all. The smoke test stops the run with a plain message if the 14B's output looks broken (for example fp16 overflow). Four JSON files come back; this page then gets a "Result" section below, without changing anything above.

---

# Addendum (9 Oct 2026)
Nothing above this line has been changed.

**What happened.** The first run (`axi_14b_test.ipynb`, Colab T4) was stopped by Colab's usage limit in the middle of Part 2, rule stated, after Part 1 had finished in both regimes. Colab wipes its working folder when a session ends, so the unfinished Part 2 progress was not kept. Part 2 starts again from its first episode.

**What was added.** `python/colab/axi_14b_part2.ipynb` and `python/colab/colab_persist.py`. The notebook runs Part 2 only. After every finished episode it copies the progress file to Google Drive (to a temporary name first, then renamed in one step, so a stop during the copy leaves the last good copy). A new session brings the file back and the runner continues after the last finished episode. Test: `python/tests/test_colab_persist.py` stops a scripted run in the middle, wipes the working folder, restores from the copy and checks that the result equals an uninterrupted run exactly.

**What did not change.** The test is the same one. The same model (Qwen2.5-14B-Instruct, 4-bit), the same tasks and episode scripts (seed 7; episodes 7000 to 7005 and 7000 to 7003), the same policies (blind, repair_m6, scope_m7), the same episodes (6 with the rule stated, 4 without, 30 requests each), the same reading rules (`bigger_model.py`) and the same predictions. Nothing was made smaller, and no prediction or rule was edited after the stop. Part 1 is not run again; its two files from the first session are read as they are. The smoke test of the first notebook is not repeated: Part 1 is read first and would show a broken model at once.

**One difference to keep in mind.** A restarted session loads the model again. Greedy decoding on a GPU is not guaranteed to give bit-identical text in two sessions, so episodes from different sessions can differ in rare token-level cases from what one uninterrupted session would have produced. This is the same kind of noise as any rerun. The Result section will say how many sessions the Part 2 files came from.

**Cost of one session.** About 12 minutes to load the model plus about 35 minutes of Part 2. A session that is stopped early loses only the episode that was running.

---

# Result (files received 9 Oct 2026; Part 1 came from one Colab session and Part 2 from a second one; everything above this line is unchanged from before the run)

Files: `docs/results/axi_14b_single_rule_Qwen2.5-14B_seed7.json`, `..._single_norule_...`, `..._long_rule_...`, `..._long_norule_...` (Qwen2.5-14B-Instruct 4-bit, Colab T4, greedy, seed 7; the same 140 tasks and the same episode seeds 7000 to 7005 and 7000 to 7003 as the 7B scope test). Every number below is recomputed from the saved rows by `tests/test_bigger_model_result.py`, not read off the printed summaries. No output was unparsed (0 of 280 single-step outputs, 0 of 900 recorded policy steps).

**How it was run.** The first session ran the smoke test and Part 1 and was then stopped by Colab's usage limit during Part 2 (rule stated). Part 2 was run again from its first episode in a second session with the first notebook (its cells 1 to 4 again, then 8 and 9); the resumable notebook of the addendum was not needed. No file mixes sessions. Whether greedy output on the T4 is identical across two sessions was not tested.

## What the run says, in short
**All 10 [guard] lines came out "yes". 4 of the 5 [need] lines came out "yes". The one "no" missed by a single task**: with no rule, blind apply was right on 11 of the 20 delete_dep requests, and the line was 10 or fewer (the 7B was right on 0 of 20). By the wording fixed before the run, that outcome reads: at 14B the raw model needs the framework less than the 7B did, and the framework is a guard with a smaller, measured gain. The guard did not weaken: gate, M6 and scope_m7 corrupted no graph and no step anywhere. The raw model still corrupted the graph on 24 of 140 single requests without the rule and 17 of 140 with it, and in every one of the 10 long episodes.

| the gain of the framework | 7B | 14B |
|---|---|---|
| M6 over blind apply, no rule, 140 tasks | 36 tasks (25.7 points) | 24 tasks (17.1 points) |
| scope_m7 over blind apply, rule stated, 140 tasks | 40 tasks (28.6 points) | 33 tasks (23.6 points) |
| scope_m7 over M6, rule stated, 140 tasks | 16 tasks (11.4 points) | 16 tasks (11.4 points) |
| scope_m7 over M6 over 30 steps, rule stated, the same 6 episodes | 56 of 180 steps (31.1 points) | 28 of 180 steps (15.6 points) |
| M6 over blind apply over 30 steps, no rule, the same 4 episodes | not run on the 7B | 112 of 120 steps (93.3 points) |

Three things did not change with the bigger model: the 14B is as bad as the 7B at a service delete that comes with a separate edge delete (4 of 20 right in both regimes, 16 corrupted; of the 16 wrong outputs, 8 without the rule and all 16 with it are the same text the 7B wrote); with the rule stated it writes stray edge deletions on as many tasks as the 7B did (16, 13 of them the same tasks); and it misses the same number of "remove every edge with weight w" requests (11).

## Part 1: one request at a time (140 tasks per regime, 20 of each kind; success = final graph equals the expected graph)
| kind (n = 20) | no rule: blind / gate / M6 / scope_m7 / scope_delta | rule stated: blind / gate / M6 / scope_m7 / scope_delta |
|---|---|---|
| delete_dep | 11 / 10 / 19 / 20 / 20 | 17 / 16 / 18 / 20 / 20 |
| delete_leaf | 20 / 17 / 20 / 20 / 20 | 6 / 0 / 6 / 20 / 20 |
| delete_edge (lone edge delete) | 20 / 20 / 20 / 20 / **0** | 20 / 20 / 20 / 20 / **0** |
| delete_pair (service delete plus a separate edge delete) | 4 / 4 / 20 / 20 / **0** | 4 / 4 / 20 / 20 / **0** |
| delete_weight (names no service) | 16 / 16 / 16 / 16 / 0 | 13 / 13 / 13 / 13 / 0 |
| add_node | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |
| add_edge | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |
| **old deletes (delete_dep + delete_leaf, n = 40)** | 31 / 27 / 39 / 40 / 40 | 23 / 16 / **24** / **40** / 40 |
| **all (n = 140)** | 111 / 107 / 135 / 136 / 80 | 100 / 93 / **117** / **133** / 80 |
The 7B on the same tasks, all 140: no rule 99 / 99 / 135 / 135 / 80, rule stated 94 / 87 / 118 / 134 / 80.
* Graphs left corrupted: blind apply 24 (no rule; 7B 36) and 17 (rule stated; 7B 24); gate, M6, scope_m7 and scope_delta 0 in both regimes.
* **Raw 14B against raw 7B, task by task.** No rule: 98 right in both, 13 right only for the 14B, 1 only for the 7B, 28 right in neither (111 against 99). Rule stated: 87 / 13 / 7 / 33 (100 against 94). Most of the no-rule gain is delete_dep: the 14B wrote the edges of the deleted service in 11 of 20 requests, the 7B in none. delete_pair did not move.
* **No rule: scope_m7 helped 1, harmed 0** (both right 135, both wrong 4). The one: "Delete the service at (2,3)." The 14B also deleted the edge (2,0) to (2,1), which does not touch (2,3); the filter dropped it. It is a stray delete without the rule stated; the 7B's 140 no-rule single-step outputs had no such case (the filter dropped nothing). The filter also dropped 2 added edges on another delete_dep (that task was right for M6 as well).
* **Rule stated: scope_m7 helped 16, harmed 0** (both right 117, both wrong 7), the same 16 as the 7B. The 14B wrote edge deletions in all 20 delete_leaf requests (49 in all; the 7B 87) and the filter dropped something on 16 tasks (37 items; the 7B 80 items on 16 tasks): 14 delete_leaf tasks (34 items) and 2 delete_dep tasks (3 items). **34 of the 37 were real edges of the graph, 3 were edges that do not exist, and none of the 37 was part of the correct change**; all 16 tasks ended right. M6 was right on 6 of 20 delete_leaf requests (7B 5). I had expected the 14B to over-apply the rule less; by the measure I registered (the lead of scope_m7 over M6 in tasks) it did not: 16 and 16. By the number of stray deletions it did, about half.
* **All 11 misses of scope_m7 (4 without the rule, 7 with it) are `delete_weight`**, the request that names no service, where the filter has nothing to read and is exactly M6 (the 7B also had 11, 5 and 6). In all 11 the model deleted at least one real edge of the graph that is not in the correct answer. Task 109 ("Remove every edge with weight 1.") is missed by both models in both regimes. With the rule stated the 14B missed five requests the 7B got right (in four it wrote the right edges and then an extra real edge of another weight, in one it listed the wrong edges) and the 7B four that the 14B got right.

## Part 2: 30 requests in a row (state mode, scope mix; each policy holding and showing its own state)
| | episodes | exact steps | episodes with an off step | most items off at once | corrupted steps |
|---|---|---|---|---|---|
| rule stated, blind apply | 6 | 76 / 180 = 42.2 % | 6 of 6 | 3 | 76 |
| rule stated, repair_m6 | 6 | 152 / 180 = **84.4 %** | 6 of 6 | 3 | 0 |
| rule stated, scope_m7 | 6 | 180 / 180 = **100 %** | 0 of 6 | 0 | 0 |
| no rule, blind apply | 4 | 8 / 120 = 6.7 % | 4 of 4 | 7 | 112 |
| no rule, repair_m6 | 4 | 120 / 120 = 100 % | 0 of 4 | 0 | 0 |
| no rule, scope_m7 | 4 | 120 / 120 = 100 % | 0 of 4 | 0 | 0 |
The 7B on the same first episodes: rule stated, repair_m6 124 / 180 = 68.9 %, scope_m7 180 / 180; no rule, 117 / 120 for both.
* Blind apply was corrupted at some step in every episode (6 of 6 and 4 of 4) and by step 10 in 4 of 6 (rule stated) and 4 of 4 (no rule) episodes. Its first corrupted step was step 1 to 3 in 6 of the 10 episodes.
* Rule stated: scope_m7 was exact on 28 steps where M6 was not and **on no step where M6 was exact and scope_m7 was not**. The filter dropped something at 8 steps (13 items: delete_leaf 7 steps, delete_dep 1); all 8 steps were exact. M6 was off in all six episodes, by up to 3 items.
* **Against the 7B, step by step on the same six episodes:** M6 was exact on 30 steps where the 7B's M6 was not and the reverse on 2 steps (both exact on 122, neither on 26). scope_m7 was exact on all 180 steps at both sizes. So the lead of scope_m7 over M6 shrank from 56 steps to 28 because M6 got better, not because scope_m7 changed.
* No rule: M6 and scope_m7 were exact on all 120 steps; the filter dropped nothing. On these four episodes the 7B had three off steps (seed 7002, step 28, a delete written against the wrong service); the 14B wrote `DEL[1,2]` and the edge, which is right, and did not make that slip. Two policies tied at 100 % cannot be told apart here.

## Predictions, one by one
1. No rule, blind apply right on 10 or fewer of the 20 delete_dep requests: **missed** (11 of 20; I had put 75 % on it). This is the one "no" line.
2. No rule: M6 at least 90 % **held** (135 of 140 = 96.4 %); ahead of blind apply by at least 10 points **held** (17.1).
3. Rule stated: scope_m7 at least 90 % of the 40 old deletes **held** (40 of 40); ahead of blind apply by at least 10 points **held** (23.6).
4. scope_m7 harms at most 2 tasks per regime **held** (0 and 0); no rule, within 3 points of M6 **held** (136 against 135).
5. By construction, no graph corrupted by gate, M6 or scope_m7 **held** (0 of 840 single-request graphs, 0 of 600 long steps).
6. Not a verdict line. The raw 14B at least as good as the raw 7B in each regime **held** (111 against 99, 100 against 94). Scope_m7's lead over M6 in the rule regime smaller than 16 tasks: **missed** (16, equal).
7. Part 2, no rule: blind apply corrupted by step 10 in at least 2 of 4 episodes **held** (4 of 4); M6 and scope_m7 each at least 95 % and within 3 points **held** (100 % each).
8. Part 2, rule stated: blind apply corrupted by step 10 in at least 3 of 6 episodes **held** (4 of 6); scope_m7 at least 90 % **held** (100 %); not behind M6 by more than 3 points **held** (15.6 ahead); never corrupted **held**.
9. Not a verdict line. The 14B's mean time per request 1.5 to 3 times the 7B's **held** (3.20 s against 1.38 s = 2.3 times without the rule; 3.85 s against 2.27 s = 1.7 times with it); the filter's share of the time got smaller **held** (about 13 microseconds against 3.85 s).
Reading rules printed by the notebook: Part 1 eight lines, seven yes and one no; Part 2 seven lines, all yes. Of my two overall bets, "every guard line yes" (55 %) came true and "every line yes" (20 %) did not.

## What else the run measured (not predicted)
* **The rule costs the 14B less than the 7B, but it still costs.** On the same 140 requests the 14B wrote 3,510 tokens with the rule stated and 2,685 without (1.3 times; the 7B 1.9 times), 539 s of model time against 448 s (1.2 times; the 7B 1.6 times). On delete_leaf it wrote 43.8 tokens against 11.5 (5.86 s against 2.36 s per request).
* **The filter is cheap:** about 13 microseconds per request on the CPU here, against 3.85 s for one model call on average. It adds no model call.
* **The best arrangement on this model is still not "rule plus scope_m7".** No rule with scope_m7 was right on 136 of 140 (448 s) and with plain M6 on 135 (448 s); rule plus scope_m7 on 133 of 140 (539 s). Named scope makes a prompt that states the rule safe; it did not beat leaving the rule out, at 7B or at 14B.

## What this does not show (the limits that matter more than the table)
* **One family and one size step.** Qwen, 14B against 7B, 4-bit weights, greedy decoding, synthetic service graphs. It says nothing about other families (their mistakes differ), about 70B and up, or about a frontier model reached through an API; that would be a separate test.
* **Every request names its targets by coordinate**, as in the scope test. Requests in ordinary words are untested, and there the filter is exactly M6.
* **The one "no" is one task.** A single task flips it; it would be wrong to read it as "the 14B no longer needs the framework" (it corrupted 24 of 140 graphs without it) or as "the need has gone down by this much" (one model, one seed).
* **Small numbers in Part 2.** Six and four episodes. scope_m7 had no off step in these ten episodes; the 7B had one event in its twenty (seed 7007, rule stated; seed 7002, no rule), so ten clean episodes cannot separate the two sizes.
* **Two sessions.** Part 1 and Part 2 were run in different Colab sessions; the model was loaded twice. The tasks, prompts and decoding were the same.
* Blind apply is the baseline that takes the model's text as it is; no one would deploy it, so the "need" lines measure how much the guard adds over doing nothing, not over a careful pipeline.

## Where this leaves the ticks
Drift: **ticked for this setup, including unrequested changes, when the request names its targets by coordinate**, now on two sizes of one family (`VERIFIED.md`). Latency: the check costs microseconds against seconds per call at 14B too. Not ticked: requests in ordinary words, implicit targets, other families, sampling, longer episodes, models above 14B.

## Provenance (sha256 of the four files as received)
| file | sha256 |
|---|---|
| axi_14b_single_rule_Qwen2.5-14B_seed7.json | `ed4bf1ee1fa664e70cfac656a1a35e77d1ebd192434933bd58446d691d04aad8` |
| axi_14b_single_norule_Qwen2.5-14B_seed7.json | `93cec487b72b8c43617a2a947c8959bea870ee571ac14d5228e9b1f6a79bf2fd` |
| axi_14b_long_rule_Qwen2.5-14B_seed7.json | `48667bca5dee84bb9e558d1cf7c7adaa1e0b7e43693e22de5b9fe659890f96c6` |
| axi_14b_long_norule_Qwen2.5-14B_seed7.json | `4428ee1dd06e3279aae9764aa94022a7144c78448fd364e220ef97ec950922ff` |

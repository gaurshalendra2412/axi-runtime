# API test: does the framework still deliver on a hosted Gemini model, on the free tier? (written BEFORE the run, 9 Oct 2026)

This page fixes the question, the predictions and the reading rules before any API number exists. It follows `SCALE_TEST_14B.md` (Qwen2.5-14B) and uses the same tasks and the same episode scripts, so the Gemini files can be compared with the 7B and 14B files task by task and step by step. Code: `python/axi/experiments/api_llm.py` (the model adapter), `python/axi/experiments/api_test.py` (the new reading rules), the runners of `scope_test.py` and `long_horizon.py` unchanged. Notebook: `python/colab/axi_gemini_test.ipynb`. Concept-map rows 3, 9, 27.

## The question, in plain words
The framework has two jobs, as in the 14B test. The **guard** (never let a broken state in, keep the held state equal to the intended one) does not depend on the model. The **need** (the raw model makes mistakes the framework fixes) does. A model that Google serves is the first place where the raw model is not a small open model we run ourselves, and where **no grammar mask exists**: the model writes free text and the delta may be badly formed. So this test has one more thing to measure, the **format**: how often a reply does not parse at all.

Why free tier: money is not the limit here, the daily quota is. The free key needs no card; the cost is time (the run is spread over days).

## Design
* **Part A, the full test, on Gemini 3.5 Flash Lite** (Google's small, fast tier; it is not a Pro model and not a frontier model, and every page and table says "Flash Lite", never "frontier"). Free tier, about 15 calls a minute and 500 a day (as shown in AI Studio on 9 Oct). Settings: **thinking at the model's default, temperature at the model's default** (Google asks to leave Gemini 3 models at 1.0; so replies are sampled, one sample per distinct prompt, and the exact text of every reply is saved), at most 8,192 output tokens (thinking included), the same system text and the same user text as every earlier run (`SYSTEM`, optional `RULES_HINT`, graph, request). The system text goes in as the system instruction.
  * **Part 1, one request at a time:** the same 140 tasks as the 7B and 14B (`make_scope_tasks(140, 7)`, 20 of each of seven kinds), rule stated and no rule. Every policy (blind, gate, repair_m6, scope_m7, scope_delta) is evaluated offline on the same reply text.
  * **Part 2, 30 requests in a row (state mode, scope mix):** policies blind, repair_m6, scope_m7. Rule stated: 6 episodes (seeds 7000 to 7005). No rule: 4 episodes (seeds 7000 to 7003). The same episodes as the 14B, so nothing is smaller.
  * **Calls:** 280 for Part 1; for Part 2 the 14B needed 521 distinct prompts (a prompt shared by two policies is asked once), at most about 900. So about 800 calls, up to about 1,200. At a budget of 480 calls a day (20 under the limit; a failed call counts too) the run takes **two to three days**; the quota resets at midnight Pacific time (12:30 pm in India until 1 November, 1:30 pm after).
* **Part B, the 20 requests that decide the question, on Gemini 3.8 Flash** (the newest Flash the free key offers; 20 calls a day). Only the 20 `delete_dep` requests of the 140 ("delete a service that other services depend on"), the request where the 7B was right on 0 of 20 and the 14B on 11 of 20 with no rule. The 20 are the `delete_dep` tasks of `make_scope_tasks(140, 7)`, in their order. No rule on the first day, rule stated on the next (20 calls each, the whole day's quota; the budget is 20). Same settings as Part A.
* **Replies.** The text is passed to the parser as the model wrote it, except for two changes that do what the grammar mask did for free: one code fence around the whole reply is removed, and every run of white space becomes one space (a delta item has no space inside it). Nothing else is changed. A reply that still does not parse is **unparsed**, counted for every policy alike (the state does not change, nothing is corrupted), and reported. How many replies needed the cleaning is reported. An empty reply is unparsed and is not asked again.
* **Quota.** Every reply is saved to a file the moment it arrives, keyed by the prompt; a restarted run that meets the same prompt makes no call. The run stops by itself before the budget is passed, or when the service says the daily limit is used up, or after six "too many per minute" waits, and says so; running the same cell again the next day carries on. The key lives in Colab Secrets only and is written to no file.
* **The model name.** The notebook lists the models the key can see, picks the id that starts with the wanted name (`gemini-3.5-flash-lite`, `gemini-3.8-flash`), prints it, and stops if there is none.
* **Data.** Free-tier prompts may be used by Google to improve its products. The prompts are made-up service graphs.
* Success = the final graph equals the expected graph exactly. A graph with a dangling edge is corrupted.

## Predictions (mine, written first; the number is how sure I am)
**Part A, Part 1.**
1. [format] At most 5 percent of all the model's replies fail to parse. (80 %)
2. No rule, raw: blind apply is right on **at most 10 of the 20** `delete_dep` requests (7B 0, 14B 11). (45 %)
3. No rule: repair_m6 solves **at least 90 %** of the 140 (14B 96.4 %). (80 %) It is **ahead of blind apply by at least 10 points** (14B 17.1). (55 %)
4. Rule stated: scope_m7 solves **at least 90 %** of the 40 old deletes (14B 100 %). (85 %) It is **ahead of blind apply overall by at least 10 points** (14B 23.6). (60 %)
5. scope_m7 **harms at most 2 tasks per regime**. (85 %) No rule: scope_m7 **within 3 points** of repair_m6. (85 %)
6. By construction (design checks, not findings): gate, repair_m6 and scope_m7 corrupt **no** graph in any regime.
7. Not a verdict line. The raw model is at least as good as the raw 14B in each regime (111 and 100 of 140). (55 %) Scope_m7's lead over repair_m6 in the rule regime is **at least 5 tasks** (7B 16, 14B 16). (55 %)

**Part A, Part 2.**
8. No rule: **blind apply is corrupted by step 10 in at least 2 of the 4 episodes**. (55 %) repair_m6 and scope_m7 each hold the intended state on **at least 95 %** of the 120 steps (65 %) and **within 3 points of each other** (90 %).
9. Rule stated: blind apply corrupted by step 10 in **at least 3 of the 6** episodes. (55 %) scope_m7 holds the intended state on **at least 90 %** of the 180 steps. (80 %) scope_m7 is **not behind repair_m6 by more than 3 points**. (90 %) Neither repair_m6 nor scope_m7 is ever corrupted (by construction).
10. Not a verdict line. The mean time of one reply, thinking included, is **between 1 and 15 seconds**. (75 %) The check and filter stay at about 10 to 15 microseconds, so their share of the time stays below one part in 100,000.

**Part B.**
11. [need] No rule: blind apply is right on **at most 10 of the 20**. (35 %; a newer and bigger model should know the rule better)
12. [guard] No rule: repair_m6 is right on **at least 18 of the 20**. (85 %) Rule stated: scope_m7 is right on **at least 18 of the 20**. (85 %) No graph corrupted by gate, repair_m6 or scope_m7 (by construction).

Overall: that every Part A line comes out "yes" at once I rate about **25 %**; that every guard and format line does about **50 %**. I expect at least one miss, most likely in the need lines.

## Reading rules (fixed now; the notebook prints exactly these)
Lines are tagged **[guard]** (the framework holds), **[need]** (the raw model still fails) or **[format]** (the replies parse). Part A, Part 1: the eight lines of `verdict_big_single` (see `SCALE_TEST_14B.md`, unchanged); Part 2: the seven lines of `verdict_big_long` (unchanged); and the line of `verdict_api_format`: [format] at most 5 percent of the model's replies do not parse (counted over all four files, each reply once; integer arithmetic, 20 times the unparsed count at most the number of replies). Part B (`verdict_pivotal`): [guard] gate, repair_m6 and scope_m7 corrupt no graph; [need] no rule: blind apply right on at most half of the 20; [guard] no rule: repair_m6 right on at least 90 percent of the 20; [guard] rule stated: scope_m7 right on at least 90 percent of the 20.

**How the result will be worded.** Format line "no": the other lines are read first as a statement about formatting, and the page says so before anything else. All guard lines yes and all need lines yes: the framework is needed on this model and delivers. All guard lines yes and some need lines no: the raw model needs it less; the framework is a guard with a smaller, measured gain, with the numbers. Any guard line no: a finding about the framework, looked into before anything is claimed. A line that comes out "NO" is a finding, not an error. **Part B is 20 requests on one model: a "no" on its need line says that model was right on more than ten of these twenty, nothing more.**

## What this cannot show
Two models of one family from one provider, **neither of them a Pro or frontier model** (a Pro model has no free quota; reaching one costs money, and a separate page would follow if that is ever run). Sampled replies, one per prompt, so a second run would not repeat the exact numbers. The vendor's own settings (thinking, safety filters, serving precision) are not ours to see. Synthetic service graphs; requests that name their targets by coordinate (as in the scope test); 140 tasks per regime and 6 and 4 episodes, as before. Latency here is the hosted service's, queue and network included, and is not comparable with the T4's seconds; what is comparable is the share of the time the check takes.

## How it is run
`python/colab/axi_gemini_test.ipynb`, on a free CPU runtime (no GPU). Cells: upload the code zip, install the Gemini library, connect Google Drive (progress and every saved reply live there), read the key from Colab Secrets, list and pick the model, a 3-request smoke test (prints the model id, the replies and the token counts), then Part 1 rule stated, Part 1 no rule, Part 2 rule stated, Part 2 no rule, and Part B. Each cell stops by itself at the day's budget and says when to run it again.


---

## Addendum (9 Oct 2026, evening; written before I had read any result file)
**What happened.** The first Colab session ran Part A until the day's budget stopped it, and then the first attempt at Part B (Gemini 3.8 Flash, no rule) got two replies, while Google's server answered **503 "This model is currently experiencing high demand"** seven times in all (the last four in a row, on the third request). The first version of the adapter waited 10, 20 and 30 seconds, then stopped with an error message instead of a plain stop. That is a failure of the run procedure, not of the model's answers.

**What changed (code only).** (1) An answer of the kind "busy or unavailable" (503 and the like, a time-out, a lost connection) is retried after 15, 30, 60, then 120 seconds each, at most six retries (about 8 minutes). If they are used up, the cell stops with a plain message ("Google's server is busy, not your quota; wait 15 to 30 minutes and run this same cell again") and the same cell, run again, carries on. (2) Such answers are written to the reply file as notes but are **not counted against the day's budget**, because the service did not process them; this replaces "a failed call counts too" in the Design section for these answers only (a "too many per minute" or "per day" answer still counts). If Google does count them, its own "per day" answer stops the run just the same. The number of busy-server answers will be reported in the Result.

**What did not change:** the tasks, the prompts, the model names, the settings, the sampling (one sample per distinct prompt), the cleaning of replies, every prediction, every reading rule and the wording of the result. The replies already received (Part A's first session and two Part B replies) were obtained with the first version and are used as they are; a retry asks the identical prompt, so a retried reply is the same kind of sample as any other. Code: `python/axi/experiments/api_llm.py` (`ServiceBusy`), notebook cells 5 and 10; test `tests/test_api_gemini.py`.


---

## Addendum 2 (10 Oct 2026, written before I had read any Part B result)
**What happened.** On 10 Oct, with Gemini 3.8 Flash still answering "503 high demand", several reruns of the Part B cell ended with Google itself answering that the **daily limit was used up**, although, as far as the printouts show, no Part B reply had arrived that day. The 20 free requests of that model seem to have been used up by tries that all failed. So the assumption of Addendum 1, that a busy-server answer is not counted by the service, was **wrong** (or at least not safe to rely on).

**What changed (code only; it replaces point (1) and (2) of Addendum 1).** A busy-server answer counts against the day's budget again, as the Design section said at the start ("a failed call counts too"). It is retried only twice and slowly (after 2 and after 5 minutes, so three tries in about 7 minutes); if the service is still busy the cell stops with a plain message that says to wait at least 30 minutes and run the cell once. The notebook text of Part B says to run that cell once. The number of busy-server answers will be reported in the Result.

**What did not change:** the tasks, the prompts, the model names, the settings, the sampling, the cleaning of replies, every prediction, every reading rule and the wording of the result. Code: `python/axi/experiments/api_llm.py` (`DiskCachedLLM`, `ServiceBusy`), notebook cells 5 and 10; test `tests/test_api_gemini.py`.


---

# Result (Part A, Gemini 3.5 Flash Lite; files received 10 Oct 2026. Part B has its own section below when its files arrive)

Everything above this section is what was written before any API number existed (and the two dated addenda); it has not been changed. The four Part A files are in `docs/results/` (hashes at the end); `tests/test_api_gemini_result.py` recomputes every number below from the saved rows, not from their summaries. The model is `gemini-3.5-flash-lite` (the id the notebook printed), thinking and temperature at the model's defaults, one sample per distinct prompt.

## 1. The format line failed, so it is read first
**78 of 651 replies do not parse (12.0 %; the limit was 5 %).** Per the wording fixed above, the other lines are read first as a statement about formatting. All 78 are one slip: where the model deletes a node it writes the id with a number after it, `DEL[2,0:7]`, as if it were adding a node (the right form is `DEL[2,0]`). 75 of the 78 come from the rule-stated regime (30 of 140 single replies and 45 of 194 long-run replies); the no-rule regime had 3 of 140 and 0 of 177. Counted per request kind in Part 1, rule stated: delete_leaf 12, delete_dep 10, delete_pair 8; no rule: delete_pair 2, delete_leaf 1. A reply that does not parse changes nothing, for every policy alike, so the framework cannot repair it: it guards the state, it does not make an unreadable answer readable. This is a finding about the setup (a hosted model with no grammar mask), not an error in the files.

**What follows from it.** The rule-stated regime says nothing about the framework on this model, because the format slip decides it:
* Part 1, rule stated: scope_m7 solves 18 of the 40 old deletes; the 22 it misses are exactly the 22 unparsed delete_dep and delete_leaf replies (10 + 12). Overall 108 of 140 for repair_m6 and scope_m7, 107 for blind apply.
* Part 2, rule stated: in **all 18 policy-episodes** (3 policies x 6 episodes) the first step that is not the oracle's state is the first step whose reply does not parse (step 1, 2 or 7 depending on the episode). Exact steps: 7.8 % for blind apply, repair_m6 and scope_m7 alike. Nothing about the policies can be read from it.

## 2. Reading rules printed by the notebook (16 lines: 9 yes, 7 no)
Part 1 (8 lines, 5 yes): yes: gate, repair_m6 and scope_m7 corrupt no graph in any regime; scope_m7 harms at most 2 tasks per regime (0); no rule: repair_m6 solves at least 90 % (135 of 140 = 96.4 %); no rule: scope_m7 within 3 points of repair_m6 (identical); no rule: repair_m6 ahead of blind apply by at least 10 points (135 against 120 = 10.7). **No:** [need] no rule: blind apply right on at most half of the 20 delete_dep requests (it was right on 20 of 20); [guard] rule stated: scope_m7 solves at least 90 % of the 40 old deletes (18; the format slip, see above); [need] rule stated: scope_m7 ahead of blind apply by at least 10 points (108 against 107).
Part 2 (7 lines, 4 yes): yes: repair_m6 and scope_m7 never corrupted at any step; no rule: each holds the oracle's state on 100 % of the 120 steps; no rule: within 3 points of each other; rule stated: scope_m7 not behind repair_m6. **No:** [need] no rule: blind apply corrupted by step 10 in at least half of the episodes (1 of 4); [guard] rule stated: scope_m7 holds the oracle's state on at least 90 % of the steps (7.8 %; the format slip); [need] rule stated: blind apply corrupted by step 10 in at least half (0 of 6).
Format (1 line): **no** (78 of 651).
The two [guard] "no" lines are the ones the wording above says to look into before claiming anything. They were looked into (section 1): both are the unparsed replies and nothing else. The two [need] "no" lines in the no-rule regime are findings about this model: it needs the framework less than the 7B and 14B did on the delete_dep request, and in Part 2 its raw state broke by step 10 in only one episode. The two rule-stated [need] "no" lines are read together with the format (an unparsed reply leaves blind apply unchanged too).

## 3. What the framework did, no rule stated (the regime the format slip leaves clean)
**Part 1, 140 requests (3 of 140 replies unparsed, counted against every policy):**
| kind (20 each) | blind | gate | repair_m6 | scope_m7 |
|---|---|---|---|---|
| delete_dep | 20 | 20 | 20 | 20 |
| delete_leaf | 19 | 19 | 19 | 19 |
| delete_edge | 20 | 20 | 20 | 20 |
| delete_pair | 3 | 3 | 18 | 18 |
| delete_weight | 18 | 17 | 18 | 18 |
| add_node, add_edge | 20, 20 | 20, 20 | 20, 20 | 20, 20 |
| **all 140** | **120** | **119** | **135** | **135** |
Graphs left corrupted: blind apply 15, gate 0, repair_m6 0, scope_m7 0. scope_m7 against repair_m6 task by task: helped 0, harmed 0, both right 135, both wrong 5; the scope filter dropped nothing on any task. All the raw model's damage is delete_pair (15 of 20 graphs left with a dangling edge); on delete_dep, the request that decided the 7B (0 of 20 raw) and the 14B (11 of 20 raw), this model is right 20 of 20 without help.
**Part 2, 4 episodes x 30 steps (0 of 177 replies unparsed):**
| policy | exact steps | corrupted by step 10 | ever corrupted | dangling edges at step 30 (mean) |
|---|---|---|---|---|
| blind | 50.8 % | 1 of 4 | 3 of 4 | 1.0 |
| repair_m6 | 100 % | 0 | 0 | 0 |
| scope_m7 | 100 % | 0 | 0 | 0 |
Blind apply held a corrupted graph on 59 of its 120 steps (first corrupted at step 14 in seed 7000, step 3 in seed 7002, step 13 in seed 7003; seed 7001 never). All three onsets are delete_pair requests, as in Part 1.
**Cost of the corruption, measured, not predicted.** Over the 120 no-rule steps blind apply carried 354 edge-slots against 254 for repair_m6 and scope_m7: 100 extra, which is exactly the sum of its dangling edges. Its prompts were 3.3 % longer in total (38,263 against 37,038 prompt tokens). The extra space is small here; the damage is the wrong state, not the size.

## 4. Predictions against outcome
1. [format] at most 5 % unparsed (80 %): **missed** (12.0 %).
2. No rule: blind apply right on at most 10 of 20 delete_dep (45 %): **missed** (20).
3. No rule: repair_m6 at least 90 % (80 %): **held** (96.4 %); ahead of blind apply by at least 10 points (55 %): **held** (10.7).
4. Rule stated: scope_m7 at least 90 % of the 40 old deletes (85 %): **missed** (18, format); ahead of blind apply overall by at least 10 points (60 %): **missed** (1 point).
5. scope_m7 harms at most 2 tasks per regime (85 %): **held** (0, 0); no rule within 3 points of repair_m6 (85 %): **held** (equal).
6. Gate, repair_m6, scope_m7 corrupt no graph: **held** (0 of 280 single-request graphs per policy, 0 of 900 long-run policy steps, all four files).
7. Not a verdict line. The raw model at least as good as the raw 14B in each regime (55 %): **missed** (rule stated 107 against 111; no rule 120 against 100). Scope_m7's lead over repair_m6 in the rule regime at least 5 tasks (55 %): **missed** (0).
8. Part 2, no rule: blind apply corrupted by step 10 in at least 2 of 4 (55 %): **missed** (1 of 4); repair_m6 and scope_m7 each at least 95 % (65 %): **held** (100 %); within 3 points of each other (90 %): **held**.
9. Part 2, rule stated: blind apply corrupted by step 10 in at least 3 of 6 (55 %): **missed** (0 of 6); scope_m7 at least 90 % of the 180 steps (80 %): **missed** (7.8 %, format); not behind repair_m6 by more than 3 points (90 %): **held**; neither corrupted: **held**.
10. Not a verdict line. Mean time of one reply between 1 and 15 s (75 %): **missed**, faster: 0.57 s (no rule) and 0.79 s (rule stated) per single request; 0.55 s and 0.64 s per distinct reply in the long runs. The check and filter time was not re-measured in this run (the files hold model seconds only).
11 and 12 (Part B): not yet.
Overall bets: "every Part A line yes" (25 %) and "every guard and format line yes" (50 %) did **not** come true. Of the 19 separate statements above (items 1 to 10, split where an item has several parts), 9 held and 10 missed; the misses are the format, need lines, the two not-a-verdict items and the time, and the two [guard] misses are the format.

## 5. Not predicted, measured
* **Output length.** Single requests: 2,536 output tokens without the rule and 3,025 with it (1.19 times; as counted by the adapter). The model is fast enough that the check is not a cost question here.
* **The rule helps the raw model when it parses.** Exploratory, written after reading the files, **not a verdict line**: dropping the number after the id in `DEL[r,c:n]` (a regular expression on the saved text) makes all 78 unparsed replies parse (78 of 78). Re-reading the 140 + 140 single replies that way gives: rule stated, blind apply 137, gate 136, repair_m6 138, scope_m7 138 of 140 (blind apply left 1 graph corrupted); no rule, blind apply 123, gate 122, repair_m6 138, scope_m7 138 (blind apply left 15 corrupted). So once the format is right, a stated rule brings the raw model within one task of the framework, and without the rule the framework adds 15 tasks. The long runs cannot be replayed this way (later prompts depend on the earlier states); that would need a new run, with its own pre-registration and a prompt that shows the delete form, and is not built.
* **Busy-server count.** The reply file of Part A (`axi_gemini_cache.jsonl`) has not been received, so the number of busy answers and the day's calls are not reported here.

## 6. What this does not show
* **One model, one tier.** Gemini 3.5 Flash Lite is Google's small, fast model, not a Pro or frontier model. Nothing here is about those.
* **The framework on this model is shown in the no-rule regime only.** The rule-stated regime was swallowed by the format slip. The 78-of-78 offline re-read is single requests only and chosen after seeing the files.
* **Small numbers in Part 2.** Four no-rule episodes (3 of 4 ever corrupted for blind apply, 1 of 4 by step 10); six rule-stated episodes that say nothing about the policies. Sampled replies, one per prompt, so a second run would not repeat these numbers.
* **Not a statement about savings, other families, requests in ordinary words, or the hosted models of other providers.** Requests name their targets by coordinate, as in every earlier test. Part B (Gemini 3.8 Flash, the 20 delete_dep requests) is not yet run; Flash Lite got 20 of 20 on them raw, so the newer model may be right on them too; Part B will say.

## 7. Where this leaves the ticks
Drift (guard): **ticked for this setup on a hosted model in the no-rule regime**: repair_m6 and scope_m7 held the intended state on all 120 steps and corrupted no graph in any of the four files. The rule-stated regime is open on this model because of the format. A framework gain on delete_pair and delete_leaf is measured without the rule (15 corrupted graphs avoided in 140 requests; 59 corrupted steps avoided in 120). Not ticked: delete_dep as a need on this model, the rule-stated regime, any model above Flash Lite.

## Provenance (sha256 of the four Part A files as received)
| file | sha256 |
|---|---|
| axi_gemini_single_rule_gemini-3.5-flash-lite_seed7.json | `f3357d614ad7f2cb42b1a31ece80d8a6dae64068bb4a1c14fa2efde19eefd9ca` |
| axi_gemini_single_norule_gemini-3.5-flash-lite_seed7.json | `b4163b4f414d2ef318084581d9de7ba26831349ef208a668c0e97fc83e0964fd` |
| axi_gemini_long_rule_gemini-3.5-flash-lite_seed7.json | `09ced730e937d7333d06842b0b471b1f2db73cb6a6777cfd0f5a276817618733` |
| axi_gemini_long_norule_gemini-3.5-flash-lite_seed7.json | `a97db3d75130c813f7b73ad6849435ef149f1f86f229dcbe9766a510b691870b` |

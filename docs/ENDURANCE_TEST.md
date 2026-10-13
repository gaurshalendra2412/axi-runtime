# Endurance and cost: what a long job costs in tokens, what one wrong step costs, and whether the host layer holds for a million steps (Part C written BEFORE its run, 12 Oct 2026)

The question that came after the confirmation test was prepared: "what happens at one or two million tokens, and does the framework cut cost?" There are two different things behind the words "a million tokens", and they get different answers. (1) A **long conversation**: the context a model must read at every step grows with the history, and for a chat that keeps everything the bill grows with the square of the number of steps. (2) A **long job**: the framework shows the model only the exact current state, so each step costs about the same, and a million tokens is just a few thousand steps. This page fixes what is looked at, and what is predicted before the part that has a prediction is run. Code: `python/axi/experiments/endurance.py`. Checks of the instrument without a model: `python/tests/test_endurance.py`. Concept-map item 92.

**What this is and what it is not.** No model is called anywhere on this page, so it says nothing new about what a model writes. It measures (A) what a step costs in real tokens, taken from the runs already saved, (B) what one wrong step costs afterwards, replayed from those same saved runs, and (C) whether the code between a model and the held state survives a million damaged replies. The test that would speak for a model at one to two million tokens (a few thousand real steps) is not run here; it is named in "What this does not show".

## The three parts
* **Part A, the token bill (descriptive, no prediction).** Real prompt and reply token counts from the saved runs: state mode (the exact current state at every step) against history mode (the whole conversation kept and re-sent, the naive chat loop), on the four episodes that ran in both modes (Qwen2.5-7B, seed 3). Measured only to the length those episodes reached; anything longer is a straight-line extension of the measured growth per turn and is labelled so.
* **Part B, what a wrong step costs (descriptive, no prediction).** Every saved 30-step run (8 files, 22 policy rows) is replayed from its saved raw replies; the replay has to reproduce each saved `exact` flag. The cost of an early mistake is then split in three: tokens spent from a state that is already different from the oracle's (persistence), requests that cannot be applied as written to the held state (inapplicable), and, among the applicable requests, how often the step itself failed, from the oracle's state and from a different state (fresh errors; only this one can show a model getting worse after a mistake).
* **Part C, endurance of the host layer (pre-registered below).** A stand-in writes the ideal reply, or one of 15 kinds of damaged reply, and four policies (blind apply, the gate, repair_m6, scope_m7) hold their own state, over a million steps and over two single episodes of 250,000 steps, on a stream of requests that is never stored.

## What was seen before this page was written
Parts A and B were computed on the saved files before the page was written (that is how the question for part B came up: the share of tokens spent from a wrong state is 88 to 90 percent for blind apply in two settings and 88 percent for every policy with Flash Lite and the rule stated). They carry no prediction; their numbers are under Result. For Part C, smoke runs of the code were made on other seeds: 20 episodes of 500 steps with four policies (seed 99), 10 episodes of 500 steps (seed 98) and one episode of 30,000 steps with three policies (seed 77); the first try of the last one stopped at step 4,422 because the old request picker had emptied the graph, which is why the `steady` mix and the fallback exist. They showed no crash and no corrupted state for the gate, repair_m6 and scope_m7; blind apply corrupted on nearly every step; scope_m7's held state small (at most 20 services and 11 edges) and its distance to the oracle flat (about 11) while the gate's and repair_m6's held states kept growing, because they keep every legal change the reply makes. **The claim lines below were therefore written knowing those shapes, and most are near-certain by design; what the real run adds is scale (a hundred times more steps), many more seeds, and the chance of a crash that a short run does not meet.** None of the smoke seeds is used in the real run.

## Design of Part C
* **Requests.** `endurance.Stream`: the request picker of the scope test (`scope_tasks.pick_scope_task`) with the same caps (14 services, 26 edges), on mix `steady` (3 add service, 3 add edge, 2 delete with dependants, 1 delete without, 1 delete edge, 1 delete service and edge), so that a long stream does not drift to an empty graph. The oracle applies the ideal change at every step. Where the old generator would stop (20 random tries find no request that fits) every kind is tried once in a fixed order first. The first n requests are exactly those of `long_horizon.oracle_script` for the mixes of the earlier tests (a test pins this).
* **Replies.** `endurance.Garbler`: with probability 0.5 a step is dealt a damaged reply, otherwise the ideal one is written (for some kinds on some requests there is nothing to damage, for instance a format slip on a request with no service delete; that reply stays unchanged and is counted as clean). The 15 kinds are dealt from a shuffled deck, so every kind is used equally often: forget_dependents (a service delete without its edges), extra_edge_delete (an unasked real edge delete appended), wrong_coordinate (one number replaced by 0 to 30), truncate (cut at a random point, as when an output cap cuts mid-item), drop_item, duplicate_item, extra_add (an unasked service added), garbage (random characters), empty, huge (1,000 random items), bigint (a number of 40 digits), stale (the previous step's ideal reply), format_slip (`DEL[r,c:v]`, the slip Flash Lite made), unicode_digits (one digit written as an Arabic-Indic digit), negative (a minus sign in a coordinate). A reply is made from the oracle's state and **every policy receives the same text**, so a policy that has drifted is sometimes handed a reply meant for another state.
* **Runs.** Control: 20 episodes of 2,000 steps, seed 20261011, no damage, four policies. Million-step run: 500 episodes of 2,000 steps, seed 20261012 (episode seeds 20261012000 to 20261012499), gate, repair_m6, scope_m7. Blind apply: the first 50 of those episodes (same streams), blind apply alone, because its held state grows without limit and would dominate the running time. Two very long episodes of 250,000 steps, seeds 20261013 and 20261014 (stream seeds 20261013000 and 20261014000), the three gated policies.
* **What is measured.** Per policy: steps, corrupted states (an edge points at a missing service), steps equal to the oracle, distance to the oracle (items that differ) in each tenth of an episode, the largest held state, host time per step (the time of the policy's own update, `time.perf_counter_ns` around it, in each tenth of an episode), crashes (an exception that is not a parse refusal; the step is then treated as an unreadable reply and the run goes on) with the first example of each kind. Peak memory is the process maximum resident size, read after the first tenth of the episodes of the million-step run and at its end. Tokens: steps times the measured tokens per step of Part A; arithmetic only.

## Predictions (mine, written first; the number is how sure I am)
1. [guard] No crash anywhere in about 1.6 million steps and 15 kinds of damage. (80 %) The parser and the policies were fuzzed by short runs, not by a million; a number with 40 digits and an Arabic-Indic digit are the two I am least sure of.
2. [guard] The gate, repair_m6 and scope_m7 are never corrupted. (95 %) The gate checks well-formedness on every step.
3. [guard] The control run holds the oracle's state exactly at every step for all four policies. (99 %)
4. [instrument] Blind apply is corrupted on at least 50 percent of its steps. (98 %) In the smoke runs it was corrupted on 98 percent.
5. [claim] scope_m7 never holds more than 30 services or 60 edges. (90 %) Smoke: 20 and 11.
6. [claim] scope_m7's mean distance is flat over a 250,000-step episode (last tenth at most 1.5 times the first). (85 %) Smoke over 30,000 steps: 10.3 to 12.2 in the ten blocks.
7. [claim] scope_m7's host time per step is flat over the same episode (at most 1.5 times). (80 %) Timing on a shared machine is noisy; the smoke blocks ranged 30 to 36 microseconds.
8. [claim] scope_m7 takes at most 100 microseconds of host time per step. (90 %) Smoke: about 33.
9. [claim] Peak memory at the end of the million-step run is at most 1.25 times the peak after its first tenth. (85 %)
10. [claim] In the last tenth of each very long episode repair_m6's distance is at least 3 times scope_m7's and the gate's at least 10 times. (90 %) Smoke: 17 times and about 100 times.
11. Not a reading-rule line. The share of replies the strict parser refuses in the million-step run is between 14 and 24 percent (80 %; truncate, garbage, empty, format_slip and negative are the five unreadable kinds, about 17 percent if each of the five worked on every request). The mean distance of scope_m7 over the million-step run is between 5 and 20 (75 %). repair_m6's largest held state in the two very long episodes is above 200 edges (75 %).

Overall: that all ten lines come out "yes" I rate about **35 %** (lines 1, 7 and 9 carry most of the doubt); that lines 1 to 4, the four guard and instrument lines, all do, about **75 %**.

## Reading rules (fixed now; the program prints exactly these)
Ten lines, tagged **[guard]**, **[instrument]** or **[claim]**. Code: `endurance.verdict_endurance`.
1. [guard] no exception other than a parse refusal escaped from the parser or from any policy, in any part of the run
2. [guard] gate, repair_m6 and scope_m7 were never corrupted at any step of the million-step run or of the two very long episodes
3. [guard] control, no damaged replies: every policy, blind apply included, held exactly the oracle's state at every step
4. [instrument] blind apply, given the damaged replies, was corrupted on at least 50 percent of its steps (the damage has teeth)
5. [claim] scope_m7 never held more than 30 services or 60 edges (the oracle's own caps are 14 and 26)
6. [claim] scope_m7's mean distance to the oracle in the last tenth of each very long episode is at most 1.5 times that of the first tenth
7. [claim] scope_m7's host time per step in the last tenth of each very long episode is at most 1.5 times that of the first tenth
8. [claim] scope_m7's host time over the million steps is at most 100 microseconds per step on one core
9. [claim] peak memory at the end of the million-step run is at most 1.25 times the peak after its first tenth
10. [claim] in the last tenth of each very long episode repair_m6's mean distance is at least 3 times scope_m7's and the gate's at least 10 times

**How the result will be worded.** A "NO" on line 1 or 2 is a finding about the framework and is looked into before anything is claimed; a crash is a bug to be fixed in a separate commit, with a test, and the run repeated on a new seed rather than edited. Lines 3 and 4 are about the instrument: if either fails, nothing else on the page is read. Lines 5 to 10 describe a stand-in's world: a "yes" means the host layer behaved this way under this damage, and does not say a model's replies look like this damage.

## What this does not show
That a model keeps its accuracy at one to two million tokens (no model is called); that the damage kinds are what real models do (they are what was seen, plus malformed text); that drift measured as distance means the same as drift in the posts; that a state held by the gate or repair_m6 is wrong for the reason a model is wrong (they keep every legal change, which is exactly why the scope filter exists); anything about longer histories than the measured 19 to 30 steps (Part A beyond them is an extension of a straight line); how much cheaper a chat is when the serving stack caches the prefix or the history is summarised (neither is measured). **The test that would speak for a model:** a few thousand real steps in state mode (one million tokens is about 3,100 steps of the measured Flash Lite prompts, two million about 6,200) with the repair, scope and a retry on an unreadable reply; it needs the paid tier or about a week of the free quota and is not part of this page.

## How it is run
On any computer with Python 3 and numpy: `cd python && python3 -m axi.experiments.endurance ../docs/results`. It prints the tables and the ten lines and writes `docs/results/axi_endurance_cpu.json`. It takes a few minutes to an hour, depending on the machine; no key and no internet are needed. The result goes below the line at the end of this page, and `python/tests/test_endurance_result.py` recomputes Parts A and B from the saved runs and Part C's lines from the saved file.


---

# Result (run 12 Oct 2026; Part C on the computer where the code was written, no model, no key; Parts A and B from the saved runs)
Everything above this line is unchanged (a hash in `tests/test_endurance.py` checks it). The run was made with `python3 -m axi.experiments.endurance ../docs/results` (Python 3.13.16, 2 cores, Linux-6.18.44-fc-v114-x86_64-with-glibc2.39); the module's fingerprint at the time of the run is recorded in the file as `_code_sha256`. **After the run the module gained only the table renderers (`table_*`, used for this section); `Stream`, `Garbler`, `damage`, `run_endurance`, `run_all` and `verdict_endurance` were not touched.** File: `docs/results/axi_endurance_cpu.json`. `tests/test_endurance_result.py` recomputes every number below from that file and from the saved runs, and checks that the tables on this page are the ones the code renders.

## Part C: the ten reading-rule lines
1. yes [guard] no exception other than a parse refusal escaped from the parser or from any policy, in any part of the run
2. yes [guard] gate, repair_m6 and scope_m7 were never corrupted at any step of the million-step run or of the two very long episodes
3. yes [guard] control, no damaged replies: every policy, blind apply included, held exactly the oracle's state at every step
4. yes [instrument] blind apply, given the damaged replies, was corrupted on at least 50 percent of its steps (the damage has teeth)
5. yes [claim] scope_m7 never held more than 30 services or 60 edges (the oracle's own caps are 14 and 26)
6. yes [claim] scope_m7's mean distance to the oracle in the last tenth of each very long episode is at most 1.5 times that of the first tenth
7. yes [claim] scope_m7's host time per step in the last tenth of each very long episode is at most 1.5 times that of the first tenth
8. yes [claim] scope_m7's host time over the million steps is at most 100 microseconds per step on one core
9. yes [claim] peak memory at the end of the million-step run is at most 1.25 times the peak after its first tenth
10. yes [claim] in the last tenth of each very long episode repair_m6's mean distance is at least 3 times scope_m7's and the gate's at least 10 times

**All ten lines came out "yes", and so did every item of prediction 11.** The prediction that all ten lines would come out "yes" (35 %) held, and so did the one that the four guard and instrument lines would (75 %). Several lines were near-certain by design, because the smoke runs had shown their shape (see "What was seen before this page was written"); what the run adds is scale and the absence of a crash. Prediction 11: the strict parser refused 19.0 percent of the replies of the million-step run (predicted 14 to 24); scope_m7's mean distance over the million steps was 11.1 (predicted 5 to 20); repair_m6's largest held state in the two very long episodes was 305 and 284 edges (predicted above 200).

| run | policy | steps | corrupted steps | steps equal to the oracle | largest distance | most services held | most edges held | host microseconds per step |
|---|---|---|---|---|---|---|---|---|
| control, no damage | blind | 40,000 | 0 | 40,000 | 0 | 14 | 17 | 4.2 |
| control, no damage | gate | 40,000 | 0 | 40,000 | 0 | 14 | 17 | 5.8 |
| control, no damage | repair_m6 | 40,000 | 0 | 40,000 | 0 | 14 | 17 | 4.8 |
| control, no damage | scope_m7 | 40,000 | 0 | 40,000 | 0 | 14 | 17 | 10.3 |
| million-step run | gate | 1,000,000 | 0 | 677 | 377 | 65 | 327 | 43.9 |
| million-step run | repair_m6 | 1,000,000 | 0 | 898 | 449 | 110 | 344 | 172.2 |
| million-step run | scope_m7 | 1,000,000 | 0 | 1,385 | 33 | 24 | 23 | 32.0 |
| blind apply, first 50 episodes | blind | 100,000 | 99,629 | 69 | 9,032 | 88 | 8,970 | 3767.6 |
| very long episode 1 | gate | 250,000 | 0 | 6 | 2,585 | 308 | 2,277 | 481.1 |
| very long episode 1 | repair_m6 | 250,000 | 0 | 6 | 405 | 206 | 305 | 179.6 |
| very long episode 1 | scope_m7 | 250,000 | 0 | 6 | 34 | 23 | 12 | 39.6 |
| very long episode 2 | gate | 250,000 | 0 | 0 | 2,622 | 309 | 2,313 | 501.3 |
| very long episode 2 | repair_m6 | 250,000 | 0 | 0 | 380 | 210 | 284 | 182.9 |
| very long episode 2 | scope_m7 | 250,000 | 0 | 0 | 31 | 22 | 15 | 39.9 |

| run | steps | clean replies | changed replies | refused by the strict parser | crashes | peak memory after the first tenth of the episodes, MB | peak memory at the end, MB |
|---|---|---|---|---|---|---|---|
| control | 40,000 | 40,000 | 0 | 0 (0.0%) | 0 | 14.2 | 14.2 |
| million-step run | 1,000,000 | 577,451 | 422,549 | 190,031 (19.0%) | 0 | 15.7 | 15.7 |
| blind apply | 100,000 | 57,682 | 42,318 | 19,009 (19.0%) | 0 | 20.8 | 21.1 |
| very long episode 1 | 250,000 | 144,576 | 105,424 | 47,412 (19.0%) | 0 | 21.1 | 21.1 |
| very long episode 2 | 250,000 | 143,896 | 106,104 | 47,800 (19.1%) | 0 | 21.1 | 21.1 |

Mean distance to the oracle, by tenth of the 250,000 steps:

| episode | policy | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | scope_m7 | 11.4 | 11.6 | 11.7 | 11.4 | 11.7 | 11.3 | 11.4 | 11.4 | 11.4 | 11.4 |
| 1 | repair_m6 | 197.6 | 222.2 | 237.0 | 238.5 | 240.2 | 249.4 | 248.4 | 238.7 | 247.1 | 250.5 |
| 1 | gate | 876.7 | 1252.7 | 1406.3 | 1570.5 | 1737.8 | 1877.0 | 2028.8 | 2186.7 | 2352.5 | 2509.8 |
| 2 | scope_m7 | 11.3 | 11.4 | 11.8 | 11.2 | 11.9 | 11.1 | 11.2 | 11.8 | 11.2 | 11.4 |
| 2 | repair_m6 | 212.2 | 218.0 | 243.1 | 248.5 | 256.1 | 254.7 | 249.3 | 247.2 | 240.5 | 253.4 |
| 2 | gate | 889.7 | 1266.8 | 1441.1 | 1603.0 | 1742.7 | 1910.3 | 2062.0 | 2219.4 | 2374.8 | 2538.6 |

Host microseconds per step, by tenth:

| episode | policy | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | scope_m7 | 38 | 40 | 41 | 40 | 37 | 41 | 39 | 40 | 39 | 41 |
| 1 | repair_m6 | 173 | 176 | 181 | 183 | 168 | 184 | 181 | 181 | 177 | 193 |
| 1 | gate | 222 | 324 | 367 | 425 | 441 | 522 | 552 | 598 | 634 | 725 |
| 2 | scope_m7 | 38 | 39 | 38 | 36 | 42 | 41 | 39 | 43 | 41 | 44 |
| 2 | repair_m6 | 179 | 174 | 172 | 164 | 189 | 191 | 181 | 197 | 186 | 196 |
| 2 | gate | 226 | 327 | 359 | 390 | 491 | 542 | 565 | 658 | 683 | 772 |

**What it shows.**
* The code between a reply and the held state handled 1,600,000 steps of 15 kinds of damaged replies (changed replies on 42 percent of the steps of the million-step run, unreadable ones on 19 percent) without one exception and without one corrupted state in the gate, repair_m6 and scope_m7. Blind apply was corrupted on 99.6 percent of its steps and its distance to the oracle grew to 9,032 items within 2,000 steps. At the 323 tokens a step of Part A that is about 517 million tokens of traffic; arithmetic, nothing was sent to a model.
* **The three safe policies are not equally good, and the difference is the point of the scope filter.** All three never corrupt, but they keep different amounts of what the damaged replies did. scope_m7 stayed within about 11 items of the oracle for 250,000 steps (11.4 and 11.3 in the first tenth of the two episodes, 11.4 and 11.4 in the last) and never held more than 24 services or 23 edges; its host time was flat at about 40 microseconds a step. repair_m6 rose from about 198 items from the oracle in the first tenth to about 250 in the last and flattened after the third tenth (up to 206 services and 305 edges held): it keeps every legal change a reply makes, so unasked services and edges pile up until the finite coordinate range is full. The gate alone never stopped growing: distance 877 in the first tenth, 2,510 in the last, 2,277 edges held, and its own cost per step more than tripled (222 to 725 microseconds) because the state it carries grew.
* The flat distance of scope_m7 (about 11, never 0) is not perfection. Under this damage 19 percent of the replies are refused by the parser and never retried, so a requested change is sometimes simply missing; those omissions stay until a later request happens to remove what they belong to. It is a **bounded offset, not accumulating drift**, in a world whose graph is capped at 14 services and 26 edges. Lines 6 and 7 do not say it would stay flat for a graph of a million nodes.
* Peak memory did not move over the million-step run (15.7 MB after the first tenth, 15.7 MB at the end); host time for scope_m7 over the whole run was 32.0 microseconds a step.

## Part A: the token bill (descriptive; measured up to 30 steps, an extension beyond)
Measured (Qwen2.5-7B, the four episodes that ran in both modes; prompt plus reply tokens of every step each episode finished):

| episode seed | steps finished | tokens, whole chat kept | tokens, exact state shown each step | prompt at the last step, chat kept |
|---|---|---|---|---|
| 3000 | 30 | 88,829 | 10,640 | 5,230 |
| 3001 | 26 | 82,758 | 10,189 | 5,530 |
| 3002 | 19 | 54,573 | 8,490 | 5,135 |
| 3003 | 30 | 78,129 | 9,906 | 4,503 |
| all four | 105 | 304,289 | 39,225 | |

The chat-kept prompt grew by 207.4 tokens a step (straight line through the mean of the four episodes over the 19 steps all of them reached: 246.8 + 207.4 x step). Measured saving over the 105 steps: **87.1 percent**. The two episodes that stopped early did so cleanly because the next turn would have passed the context cap or the GPU ran out of memory.

Extension of that straight line to longer jobs (**not measured**; nothing beyond 30 steps was run):

| steps | tokens, exact state shown each step (Qwen: 368.3 a step) | tokens, whole chat re-sent each step | saving |
|---|---|---|---|
| 100 | 36,828 | 1,073,286 | 96.57% |
| 300 | 110,485 | 9,443,017 | 98.83% |
| 1,000 | 368,283 | 104,080,232 | 99.65% |
| 3,000 | 1,104,850 | 934,556,487 | 99.88% |
| 10,000 | 3,682,833 | 10,375,539,167 | 99.96% |

Step at which the chat-kept prompt would pass a window (extension): 5,500 (the T4 GPU ran out of memory near here): step 26; 32,768: step 157; 128,000: step 616; 1,000,000: step 4,820; 2,000,000: step 9,641.

With the exact state shown each step, Gemini 3.5 Flash Lite used 323.2 tokens a step (prompt 308.6, reply 14.6; the 120 steps of the scope_m7 policy in `axi_gemini_long_norule`): 1 million tokens of work is about 3,094 steps and 2 million about 6,187.

**Reading.** Keeping the whole chat used 7.8 times the tokens of showing the exact state each time (304,289 against 39,225; a saving of 87 percent), on the steps that were measured, with one model and the naive loop. The chat's cost per step grows with the step and the state's does not, so the gap widens with the length of the job; the extension rows say how much, on the assumption that the straight line holds, which has not been checked past 30 steps. **What this does not say:** that a serving stack that caches the prefix pays this price (the repeated prefix is then cheap, though the model still attends over all of it), that a chat that is cut or summarised pays it, or that the saving is the same on another task. Both of those are the fair competitors, and neither is measured here.

## Part B: what one wrong step costs (descriptive; replayed from the saved raw replies)
Every one of the 22 policy rows replays to the saved `exact` flag of every step (0 mismatches in 4,800 steps).

| setting | policy | steps | refused by the parser | tokens spent from a state different from the oracle's | requests that could not be applied as written | step did not do what was asked, starting from the oracle's state | same, starting from a different state |
|---|---|---|---|---|---|---|---|
| Qwen 7B, rule not stated (base mix) | blind | 360 | 0 | 89.1% | 16.1% | 12/42 = 28.6% | 38/260 = 14.6% |
| Qwen 7B, rule not stated (base mix) | gate | 360 | 0 | 89.7% | 23.1% | 12/42 = 28.6% | 38/235 = 16.2% |
| Qwen 7B, rule not stated (base mix) | repair_m6 | 360 | 0 | 0.3% | 0.0% | 1/359 = 0.3% | 0/1 = 0.0% |
| Qwen 7B, rule stated (base mix) | blind | 240 | 0 | 73.7% | 10.8% | 10/62 = 16.1% | 18/152 = 11.8% |
| Qwen 7B, rule stated (base mix) | gate | 240 | 0 | 89.0% | 17.9% | 8/26 = 30.8% | 35/171 = 20.5% |
| Qwen 7B, rule stated (base mix) | repair_m6 | 240 | 0 | 42.2% | 8.8% | 8/135 = 5.9% | 1/84 = 1.2% |
| Qwen 7B, rule not stated (scope mix) | repair_m6 | 240 | 0 | 0.8% | 0.4% | 1/238 = 0.4% | 0/1 = 0.0% |
| Qwen 7B, rule not stated (scope mix) | scope_m7 | 240 | 0 | 0.8% | 0.0% | 1/238 = 0.4% | 0/2 = 0.0% |
| Qwen 7B, rule stated (scope mix) | repair_m6 | 360 | 0 | 27.5% | 8.6% | 17/264 = 6.4% | 6/65 = 9.2% |
| Qwen 7B, rule stated (scope mix) | scope_m7 | 360 | 0 | 1.8% | 0.3% | 1/354 = 0.3% | 0/5 = 0.0% |
| Qwen 14B, rule not stated | blind | 120 | 0 | 88.3% | 9.2% | 4/12 = 33.3% | 10/97 = 10.3% |
| Qwen 14B, rule not stated | repair_m6 | 120 | 0 | 0.0% | 0.0% | 0/120 = 0.0% | - |
| Qwen 14B, rule not stated | scope_m7 | 120 | 0 | 0.0% | 0.0% | 0/120 = 0.0% | - |
| Qwen 14B, rule stated | blind | 180 | 0 | 56.9% | 7.8% | 9/78 = 11.5% | 9/88 = 10.2% |
| Qwen 14B, rule stated | repair_m6 | 180 | 0 | 16.6% | 5.6% | 7/152 = 4.6% | 1/18 = 5.6% |
| Qwen 14B, rule stated | scope_m7 | 180 | 0 | 0.0% | 0.0% | 0/180 = 0.0% | - |
| Flash Lite, rule not stated | blind | 120 | 0 | 46.3% | 2.5% | 3/63 = 4.8% | 0/54 = 0.0% |
| Flash Lite, rule not stated | repair_m6 | 120 | 0 | 0.0% | 0.0% | 0/120 = 0.0% | - |
| Flash Lite, rule not stated | scope_m7 | 120 | 0 | 0.0% | 0.0% | 0/120 = 0.0% | - |
| Flash Lite, rule stated | blind | 180 | 43 | 88.4% | 13.9% | 6/20 = 30.0% | 24/135 = 17.8% |
| Flash Lite, rule stated | repair_m6 | 180 | 42 | 88.3% | 13.3% | 6/20 = 30.0% | 24/136 = 17.6% |
| Flash Lite, rule stated | scope_m7 | 180 | 42 | 88.3% | 13.3% | 6/20 = 30.0% | 24/136 = 17.6% |

**Reading.**
* **Persistence is large for plain apply and small for the filter.** Without the rule, 89 percent (Qwen 7B) and 88 percent (Qwen 14B) of the tokens of blind apply were spent from a state that already differed from the oracle's; repair_m6 0.3 and 0.0 percent. For scope_m7 the share is 0.8, 1.8, 0.0, 0.0, 0.0 percent in five of the six settings it was run in (no more than 2 percent), and **88 percent for Flash Lite with the rule stated, where it is the same 88 percent for every policy**: 42 of 180 replies were refused by the parser (the model wrote `DEL[r,c:v]` for a service delete), the harness does not retry, and an omission that nothing repairs stays for the rest of the episode. With the rule stated, repair_m6 alone spent 42 percent (Qwen 7B, base mix), 28 percent (Qwen 7B, scope mix) and 17 percent (Qwen 14B) from a wrong state, because it keeps the legal deletes nobody asked for; scope_m7 spent 1.8 and 0.0. The framework does not protect against a reply it cannot read; it only makes the failure loud.
* **Later replies did not get worse.** In 11 policy rows with at least 20 applicable requests starting from the oracle's state and from another state, a step that started from another state failed more often in 1 (Qwen 7B, rule stated (scope mix), repair_m6: 9.2 percent against 6.4) and less often or as often in the other 10. The request kinds differ between the two groups, the counts are small, and this is a post-hoc look at data already read; it is no evidence that a mistake makes the model's next reply worse in this task, and it cannot rule out a small effect.
* **What does spread is the wrong state itself**, and the requests that can no longer be applied as written (2 to 23 percent of the steps of blind apply and the gate, where the held state lacks what a request names or the change would leave an edge dangling), which is why the share of tokens from a wrong state is so high for them. This is persistence and its side effects, not a cascade of new mistakes.

## What this changes
* The claim "one wrong step does not drag many wrong steps after it" holds for the filter in five of six settings and not for unreadable replies. **The next test that matters is a retry on an unreadable reply** (ask once more with the parser's message), because it is the one measured hole, and the cost is one extra call on the steps where a reply is refused (42 of 180, 23 percent, for Flash Lite with the rule stated).
* The test that would speak for a model at one to two million tokens is still to be run (3,100 to 6,200 real steps in state mode); if that run goes wrong, this page makes it unlikely that the host layer is the reason.
* The Medium paper's "-46.90 percent tokens" (`PAPER_CORRECTIONS.md`, item 18) has no source in our logs; the measured saving above is a different number from a different comparison and does not support it.

# Test 19: the serving side a sixth time, with jobs of 100 steps (written BEFORE the run, 20 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before the run has been made. Code: `python/axi/experiments/provider_side_long.py` (it imports `provider_side.py` and `provider_side_typed.py`, which import `provider_side_constrained.py`, `provider_side_bounded.py`, the gate, the diagnostics, `agent_loop.py`, `cascade.py` and `long_horizon.py`, and it edits none of them); instrument checks `python/tests/test_provider_side_long.py`; notebook `python/colab/axi_provider_side_long.ipynb`. Concept-map item 112. It follows `PROVIDER_SIDE_TYPED_TEST.md` (item 111) and the Result at the end of that page. It needs a GPU (a Colab T4 is the one assumed), the engine vLLM 0.31.0, and about an hour in all (see "How it is run"). Nothing is asked twice in this test: **there is no retry in any cell.**

## Why this test
1. **Every job so far is 30 steps.** The fifth run (`axi_provider_typed.json`) held the right state on 1,920 of 1,920 steps in two cells (`FN`, the grammar that names the coordinates, and `FKA`, the kind grammar with the target line at the top and the end), against 1,833 (95.5 %) for the capped control `F0` and 1,835 (95.6 %) for the kind grammar `FK`. `FN` took 0.1166 seconds per step against 0.1168 for `F0`. It was one run of 64 jobs, and 30 steps is a short task.
2. **The problem the owner means is long tasks.** In his words, an AI loses track or invents things as a long task goes on. What a held state can show of that is narrow and checkable: after step 10, step 50 and step 100, is the state the job holds still the intended one? The exact state is kept by the host and shown at every step, so the model has nothing to forget; what can still go wrong is the model writing a wrong edit on some step. This test asks whether that happens more as the job gets longer, with and without the grammar that names the coordinates, and what a chat that keeps everything does over the same 100 steps.
3. **A slip stays.** After a wrong edit the held state differs from the oracle's until a later request removes the wrong item. In the fifth run the first slips of the five jobs of `F0` were at steps 2, 9, 14, 2 and 11, and the state was off on 87 steps in all. In a job of 100 steps a slip at step 10 can cost 90 steps. So 100 steps is a harsher test than 30 for a cell that slips at all, and the share of exact steps after a slip is a consequence of that slip and not a new one. **Nothing on this page promises 100 percent.**

## What changes and what does not
**Does not change:** the model (`Qwen/Qwen2.5-7B-Instruct-AWQ`), the engine (vLLM 0.31.0), the GPU assumed (a Colab T4), the cache fixed at **80,000 tokens**, `max_model_len` 8,192, greedy decoding, at most 96 new tokens, prefix caching on, the seeds (job e has seed 20261017 x 1000 + e), the exact state shown at every step (arm `S`), the policy `repair_m6`, the lockstep rounds, the measures (exact steps per GPU hour: 3,600 times the steps after which the held state equals the oracle's, over the wall seconds of the cell), the capped expression of the third run (sha256 `6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245`), the `named` grammar of the fifth run, the typed session of the fifth run (`provider_side_typed.TypedSession`, here with 100 steps), and the kept chat of the earlier runs (`provider_side.Session`, arm `KC`). **The gate, the repair, the diagnosis and the policy are the framework's, unedited**, and the files this test imports are pinned by sha256 in the test: `provider_side.py` `bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3`, `provider_side_constrained.py` `03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d`, `provider_side_bounded.py` `7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc`, `gate.py` `b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7`, `diagnostics.py` `255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2`, `agent_loop.py` `d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6`, `cascade.py` `b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7`, `long_horizon.py` `5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac`, and `provider_side_typed.py` `c5f05d5840374fb77a1c67c6fd33f166b26342e93733bd9a31aa474eacbc1420`.
**Changes:**
* **The jobs are 100 steps.** A job is the same script with more steps: its first 30 requests are the 30-step job's (checked for all 64 jobs). The jobs are **192 to 255, which no earlier run has used** (jobs 0 to 63 were run by the third, fourth and fifth runs, 64 to 127 by the fourth, 128 to 191 by the fifth). I have not read a reply to any of them (none exists).
* **The host reads the request** as in the fifth run: the three sentences the jobs use are read in Python, with no model call. A request in free words is a different test and is not run here.
* **The kept chat carries a grammar** (`KeptChat`, a subclass that adds the expression of the request and the "inside its language" flag and overrides nothing that decides a step), so that `LC` is asked under the same language as `L8`, the exact-state cell it is compared with. Without it the two would differ in two things at once.
* **The cells are four**, in this order, no retry:

| cell | jobs | arm | grammar | compared with |
|---|---|---|---|---|
| `L0` | 192 to 255 | S (exact state each step) | capped | the control, same session |
| `LN` | 192 to 255 | S | named | `L0` |
| `L8` | 192 to 199 | S | named | `LC`, the same 8 jobs |
| `LC` | 192 to 199 | KC (whole chat kept, cache on) | named | `L8` |

  About 35 minutes of GPU. **Why 64 jobs for the first two:** the same number of jobs side by side as the earlier cells, so the seconds per step can be read against them. **Why 8 for the last two:** the kept chat is compared with the exact-state arm at the same number of jobs and on the same jobs, since the seconds per step depend on how many jobs share the GPU. A kept chat at 64 jobs and 100 steps would take hours (the second run's `KC64` at 30 steps took 10,331 seconds) and would say the same thing.
* **The repeat of the fifth run is read from the same cells.** Steps 1 to 30 of `LN` are 30-step jobs of the kind of `FN`, on 64 jobs nobody has run, in the same session as `L0`. No extra cell is run for it.
* **A pre-flight check that can fail** runs before the engine is loaded: that the jobs are new, that every job has 100 steps and starts with its 30-step job, that the imported files are the pinned ones, that the expressions have the sha256 below, that the oracle's reply is inside the expression of every one of the 14,400 requests of the four cells, that the first request of each cell is built, and that the GPU is a T4 with less than 500 MiB in use. It prints its findings; a RED line stops the notebook.
* **Q = O / M is computed from the saved file** (`q_measure.py`): exact steps per GPU hour and exact steps per 1,000 tokens, side by side, never mixed into one number.

## What I read before writing this page
* **The scripts.** All 64 jobs reach 100 steps (the generator can end a script early when no request fits; none does). The 6,400 requests are 2,158 add-service, 2,059 add-edge and 2,183 delete. The oracle's own reply is inside the capped, the kind and the named expression of every request. The named expressions of `LN` are 835 different ones (the fifth run's `FN` had 427 in 30 steps); how long the engine takes to compile them is in the wall seconds of the cell.
* **The fingerprints** (sha256 of the sorted set of expressions of a cell, recomputed from its jobs): `L0` is the capped expression `6f4e5a99…`; `LN` `85d17bbffb556c21acf180b989fecf3b2c92ca419fce1516a675fb24dd9929af`; `L8` and `LC` `9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb`.
* **What the fifth run implies, as an estimate and not a measurement.** `F0`'s five jobs slipped within 30 steps; if the same chance per step held for 100 steps, about 15 of 64 capped jobs would slip and `L0` would hold the right state on about 87 % of the steps. `FN` and `SN` together had no slip in 3,840 steps; by the rule of three the chance of a slip per step is below 0.08 % with 95 % confidence, which allows up to five slips in the 6,400 steps of `LN`. My guess is two or fewer. One slip in `LN` before step 36 costs more than a point.
* **The stand-in (a model of a prefix cache, not the engine, never a result)** runs the four cells in a few seconds. In it the mean prompt of `L0` is 389 tokens in the first quarter of the steps and 380 in the last, so a flat prompt is what the code does when the state is the oracle's; the kept chat's prompt goes from 1,870 to 6,896 tokens (the longest 7,557) and its eight jobs stop between steps 24 and 56 on the rule of `provider_side.py` (a kept chat stops before its next prompt would pass 7,600 tokens). The stand-in counts digits one by one; what the engine counts is what the run records.
* **The earlier runs' time, for reference** (another Colab session, not a basis for any time claim here): the fifth run's `F0` 0.1168 seconds per step and 29,437 exact steps per GPU hour, `FN` 0.1166 and 30,882.

## Design in one place
| | |
|---|---|
| model, engine | `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0 (`pip install vllm==0.31.0`); structured output by one regular expression per request |
| GPU, cache | a Colab T4 (any other GPU is recorded and the cells are not merged); cache 80,000 tokens by the block override |
| jobs | 100 steps, seed 20261017; jobs 192 to 255, new |
| arms | `S` (exact state each step, cache on) for `L0`, `LN`, `L8`; `KC` (whole chat kept, cache on) for `LC` |
| policy | `repair_m6`, no rule in the prompt, no retry |
| cells, order | `L0`, `LN`, `L8`, `LC` |
| measures | as the fifth page, plus: the exact share over steps 1 to 30, 1 to 33 and 67 to 100, the jobs whose first slip falls in each third, the prompt of the first and the last quarter of the steps, the jobs a kept chat stopped, Q = O / M from the saved cell |
| pre-flight, safety cell | before the engine is loaded: the pre-flight above. After it: 3 jobs of 3 steps of an `S` session and of a `KC` session, both with the named grammar, cached counts reported, every reply parsed and inside its language. A first request without cached counts, a reply that does not parse or is outside its language, or one that uses all 96 new tokens stops the notebook |

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 5 are checks that the run is the run I describe; lines 6 to 17 are about the 100 steps, and for those the only thing I know is 3,840 steps of the named grammar with no slip and 1,920 of the capped one with five jobs slipping.

1. [instrument] the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run (98 %)
2. [instrument] the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens, and every reply is in the language of its own request (92 %)
3. [guard] the grammar changed only what it forbids: every job of LN writes the same text as the same job of L0 up to the job's first step whose L0 reply is outside the named language, and the reply at that step is inside it (78 %)
4. [guard] the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs, and the oracle's own reply is inside the expression of every request (97 %)
5. [guard] every job of the exact-state cells (L0, LN, L8) ran all 100 steps; none ended early (96 %)
6. [claim] the exact-state prompt stays flat while the job goes on: in L0 the mean prompt of the last quarter of the steps (76 to 100) is at most 1.3 times that of the first quarter (1 to 25) (92 %)
7. [claim] the held graph was never malformed: no step of any cell that was run has a corrupted state (92 %)
8. [claim] the fifth run's 100 percent comes back on jobs nobody has run: on steps 1 to 30 of the 64 new jobs the exact share of LN is at least 99.5 percent (68 %)
9. [claim] over all 100 steps the exact share of LN is at least 99 percent (50 %)
10. [claim] over the last third of the steps (67 to 100) the exact share of LN is at least 99 percent (40 %)
11. [claim] over all 100 steps the exact share of LN is at least three points above that of L0 (80 %)
12. [claim] the job without a grammar's help does lose ground as it goes on: the exact share of L0 over steps 67 to 100 is lower than over steps 1 to 33 (90 %)
13. [claim] every kept-chat job (LC) stopped before its 100th step, because its next prompt would have passed 7,600 tokens (93 %)
14. [claim] on the steps it ran, the kept chat held the state no less exactly than the exact-state arm on the same 8 jobs: the exact share of LC is at least that of L8 minus one point (80 %)
15. [claim] the kept chat does fewer exact steps per GPU hour than the exact-state arm on the same 8 jobs: LC lower than L8 (both in this session) (92 %)
16. [claim] the seconds per step of LN are at most 15 percent above those of L0 (both in this session) (88 %)
17. [claim] the exact steps per GPU hour of LN are higher than those of L0 (both in this session) (90 %)

**Headline** (lines 8 and 9): does the 100 percent of the fifth run come back on new jobs, and does the grammar that names the coordinates hold the state over 100 steps. Lines 2, 3, 4 and 5 all reading yes: about **65 %**. Lines 8 and 9 both reading yes: about **45 %**. Line 10 yes as well (no slip at all in `LN` before step 67, or only a very late one): about **35 %**. All 17 yes: under **3 %** (line 10 and line 14 are the ones I least expect together).

Why I think so, line by line. *Line 1 (98 %):* it worked in all earlier runs, same engine. *Line 2 (92 %):* read yes in the fifth run with the same engine; the new thing is 835 expressions in one cell. *Line 3 (78 %):* the fifth run's four comparisons read yes, and this one has 6,400 steps and a number can flip when the batch changes. *Line 4 (97 %):* a hash recomputed from the jobs and a reply the pre-flight checks for all 14,400 requests; the risk is a mistake of mine. *Line 5 (96 %):* nothing but a crash ends an exact-state job. *Line 6 (92 %):* the state is the oracle's, the graph is capped in size, and the stand-in gives a ratio of 0.98; the risk is that the graphs are small at the start and grow. *Line 7 (92 %):* 0 of 13,440 steps in the fifth run and 0 of 360 in the gate test; the gate admits only well-formed edits and the repair completes the rest. *Line 8 (68 %):* no slip in 3,840 steps of `FN` and `SN`; a posterior from that alone gives about 0.67 for no slip in the 1,920 steps of the first 30, and one more at step 22 or later would still pass. *Line 9 (50 %):* the same reasoning over 6,400 steps allows no slip (0.38) or one slip after step 35 (about 0.15). *Line 10 (40 %):* it needs no slip before about step 67 and none that costs more than a point late. *Line 11 (80 %):* if `L0` slips at the rate of `F0` it holds about 87 % and the gap to `LN` is between twelve and thirteen points; the risk is a lucky control. *Line 12 (90 %):* slips stay, so the late share falls below the early one whenever a job slips in between. *Line 13 (93 %):* the stand-in stops the eight jobs between steps 24 and 56, and the real prompts are not shorter than the stand-in's by a factor of two. *Line 14 (80 %):* the kept chat re-shows the state at every step as the exact-state arm does, so it is no worse until it stops; a slip in either side's few steps decides the line. If it reads NO, the kept chat is less exact on the steps it ran, which I do not expect. *Line 15 (92 %):* the prompts of the kept chat are many times longer. *Line 16 (88 %):* `FN` cost nothing in seconds in the fifth run; 835 expressions compile instead of 427. *Line 17 (90 %):* it needs the exact steps to rise or, if they tie, the seconds not to rise.

**Also written down, not lines (read afterwards as descriptions).** (a) `L0` holds the right state on between 82 and 95 percent of its 6,400 steps: 70 %. (b) The first slips of `L0` fall in steps 1 to 33 for more jobs than in steps 67 to 100: 70 %. (c) The eight kept-chat jobs stop between steps 20 and 60: 80 %. (d) The computed prompt tokens per step of `LN` are within 5 percent of those of `L0`: 85 %. (e) The four cells take 30 to 45 minutes in all, loading excluded: 70 %. (f) The one thing I do not know and would like to: whether the model slips more often at step 80 than at step 8 when the grammar names the coordinates. With zero or one slip in `LN` the run cannot say.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers.
* *Lines 2, 3, 4, 5 yes, lines 8 and 9 yes:* on 64 jobs nobody had run, with requests read by the host from their sentence, the named grammar held the state on at least 99 percent of the steps over 100 steps, and the 100 percent of the fifth run came back (at 99.5 percent or more) on steps 1 to 30. The sentence is the share, the seconds per step and the exact steps per GPU hour, with the setup beside them.
* *Line 8 NO:* the 100 percent of the fifth run was not reproduced on new jobs. It is the first thing to write down, before anything about 100 steps. Read the first misses of `LN` (printed): a Delete or an Add, and at which step.
* *Line 8 yes, line 9 NO:* the first 30 steps hold, a slip appears later. Read where (printed: the thirds of the job) and which request. That is a finding about the named grammar at length, and it is the one the owner's claim needs.
* *Line 9 yes, line 10 NO:* a late slip that costs less than a point. Read it.
* *Line 11 NO:* `L0` did not lose enough or `LN` lost. If `L0` is at 98 percent, 100 steps is not yet long enough to see the problem in this setup, and line 12 will say so.
* *Line 12 NO:* the control did not lose ground between the first and the last third. Then 100 steps does not show the long-task problem for the capped grammar in this setup, and no claim about drift rests on it.
* *Line 13 NO:* a kept chat finished 100 steps inside 7,600 tokens. The context length of the setup was not the limit; read the prompts.
* *Line 14 NO:* the kept chat held the state less exactly on the steps it ran, with the grammar. That is a finding in the exact-state arm's favour that I did not expect, and I will say that I did not.
* *Line 15 NO:* the kept chat did more exact steps per hour than the exact-state arm on 8 jobs. The honest sentence then is that at 8 jobs the exact-state arm has no speed advantage in this setup.
* *Lines 16 or 17 NO:* the grammar costs seconds at 835 expressions. Read the rounds with new expressions (`new_grammars` in the saved rounds).
* *Line 6 NO:* the exact-state prompt grew; read the graph sizes before reading anything else about speed.
* *Line 7 NO:* a held graph was malformed; it is the first thing to read, ahead of every other line.
* *Line 3 NO with line 2 yes:* a reply inside the language changed, or a job differs for another reason than the grammar. Read which replies differ before reading anything else.
* *Line 4 NO:* a cell used other expressions than the ones this test defines: a mistake in the code, read before anything else.
* *Line 2 NO:* the constraint did not hold in some request; read it.
* *Line 1 NO, the pre-flight RED, or the safety cell says STOP:* nothing is run or read.
* *A cell that did not run:* its lines read "not run", never NO.

## Reading rules (fixed now; the printout prints exactly these)
The 17 lines above are the reading rules; `provider_side_long.verdict_long` computes them from this run's cells. Line 1 uses every request of every cell; line 2 every reply of every cell (the share the strict parser accepted, the replies that used all 96 new tokens, and the replies that fully match the expression carried by their own request); line 3 compares `LN` with `L0` job by job (the same seed): while the `L0` reply is inside the named language of the request the `LN` reply is the same text, at the first step whose `L0` reply is outside it the `LN` reply is inside the language and the comparison of that job stops, a job whose `L0` replies are all inside has the same replies throughout and the same number of steps; line 4 the sha256 of the sorted set of expressions of each cell recomputed from its jobs, equal to the one on this page, and the oracle's reply fully matching the expression of every request; line 5 every job of `L0`, `LN` and `L8` has 100 log entries and no stop; line 6 the mean prompt tokens of steps 76 to 100 at most `1.3` times those of steps 1 to 25 in `L0`; line 7 the sum of the corrupted steps (the held graph is not well formed) over all cells is 0; line 8 the exact share of `LN` over steps 1 to 30 at least `0.995`; line 9 the exact share of `LN` over all steps at least `0.99`; line 10 over steps 67 to 100 at least `0.99`; line 11 the share of `LN` at least `0.03` above that of `L0`; line 12 the share of `L0` over steps 67 to 100 strictly below its share over steps 1 to 33; line 13 every job of `LC` stopped with the reason "context" and has fewer than 100 steps; line 14 the share of `LC` at least that of `L8` minus `0.01`; line 15 the exact steps per GPU hour of `LC` lower than those of `L8`; line 16 the seconds per step of `LN` at most `1.15` times those of `L0`; line 17 the exact steps per GPU hour of `LN` higher than those of `L0`. Definitions as on the earlier pages: "accepted by the strict parser" is `strict_parse(text) is not None`; "exact steps per GPU hour" is 3,600 times the steps after which the held state equals the oracle's, over the wall seconds of the cell; a "window" counts the steps from 1 and includes both ends. A line that needs a cell the run does not hold reads "not run".

**The rule that decides what the Result may claim.** Lines 2, 3, 4 and 5 all read yes: the figures of the cells are claimed as read. That the 100 percent comes back is claimed from line 8, what happens over 100 steps from lines 9 to 12, what the kept chat does from lines 13 to 15, and a statement about time or cost only from lines 16 and 17 (cells of one session), never from a cell against another run's (another Colab session). Line 2, 3, 4 or 5 reads NO: no claim is made from this run until the cause is found.

**How the result will be worded.** On jobs no earlier run had used, with requests read by the host from their sentence, on one Tesla T4, one 7B model and the engine written next to the figures: the exact share over 100 steps and over steps 1 to 30 and 67 to 100 for the capped and for the named grammar, the seconds per step, the exact steps per GPU hour and per 1,000 tokens, and for the kept chat the step at which each job stopped; **for the named grammar, in the same sentence, that the model writes the numbers and that the request is otherwise copied into the reply.** Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never "100 percent" for a grammar, never "proven" (one model, one GPU, one run, 64 jobs), never a statement about any provider, never a statement about requests in free words, never a statement about a chat with a summary or a window, and never a statement about a model that was not run. A "NO" on a claim line is a finding about this setup and goes into the article as such.

## What this does not show
Requests in free words (the host reads the sentence the job was built from); tasks of more than 100 steps; real company tasks; another model, size, GPU, engine or cache size; a kept chat with a summary or a sliding window (only the plain kept chat is run, and it stops because the setup's context is 8,192 tokens); a kept chat at 64 jobs; a retry on top of a grammar (no cell has one); dollars; a capacity in jobs per GPU; a lower bound of the cost per exact step; energy (never measured; only time); hallucination in free text (the state here is a graph of services and the answer is checked against an oracle); the tensor-level roadmap (this test is a prompt and a grammar and trains nothing).

## How it is run
Open `python/colab/axi_provider_side_long.ipynb` in Colab with a T4 and run the cells one at a time: it gets the code from GitHub, installs vLLM 0.31.0, connects Drive, fixes the output stream, runs the **pre-flight check**, loads the model with the per-request constraint, runs the safety cell, then the four cells in order. Each finished cell is written as `axi_provider_l_<label>.json` and copied to Drive; a cell that was cut off is run again from its start (the longest takes about 13 minutes). The last cell puts the four cells together, prints the table, the windows, the first misses, the 17 lines and Q = O / M per cell, and offers the files for download. The Result goes below the line at the end of this page, and a result test will recompute every number in it from the saved files.

---


# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_long.py` and again by `tests/test_provider_side_long_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_long.ipynb`: 14,004 steps in four cells, 27.2 minutes of generation, vLLM 0.31.0, `Qwen/Qwen2.5-7B-Instruct-AWQ`, Tesla T4, a cache of 80,000 tokens. The merged file is `docs/results/axi_provider_long.json` (sha256 `a0ad579961d006e9...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*

**What the page expected, and what came out.** Lines 2, 3, 4 and 5 all read yes (the page put them at about 65 % together), so the run is the run the page describes and the figures are claimed as read. **16 of the 17 lines read yes; line 11 reads NO, and no line is "not run".** The two headline lines read yes: the repeat on new jobs (line 8, 68 %) and the named grammar over 100 steps (line 9, 50 %), and so does line 10 (40 %), which the page had put with line 14 as the pair it least expected together. Both headline lines yes stood at about 45 % on the page, all three lines at about 35 %.

## The 17 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 98 % | cached tokens reported on all 14,004 requests (4,859,280 of 6,214,056 prompt tokens served from the cache) | yes |
| 2 | 92 % | 14,004 of 14,004 replies accepted by the strict parser; longest reply 82 of 96 new tokens; 0 outside the language of their own request | yes |
| 3 | 78 % | `LN` and `L0` write the same text at 6,384 of 6,400 steps; the other 16 are the `L0` replies outside the named language (below), and the `LN` reply there is inside it | yes |
| 4 | 97 % | the recorded sha256 of every cell equals the one recomputed from its jobs; `LN` has 835 expressions; the oracle's reply is inside the expression of all 14,004 requests | yes |
| 5 | 96 % | all 136 jobs of `L0`, `LN` and `L8` ran 100 steps | yes |
| 6 | 92 % | mean prompt 346 tokens in steps 1 to 25 and 339 in steps 76 to 100 (ratio 0.98) | yes |
| 7 | 92 % | 0 corrupted steps in 14,004 | yes |
| 8 | 68 % | `LN` steps 1 to 30: 1,920 of 1,920 exact (100.0 %) | yes |
| 9 | 50 % | `LN` all steps: 6,400 of 6,400 exact (100.0 %); 64 of 64 jobs exact to the end | yes |
| 10 | 40 % | `LN` steps 67 to 100: 2,176 of 2,176 exact (100.0 %) | yes |
| 11 | 80 % | `LN` 100.0 % against `L0` 97.6 %: 2.4 points, the line asks for 3 | NO |
| 12 | 90 % | `L0` 95.5 % over steps 67 to 100 against 98.5 % over steps 1 to 33 | yes |
| 13 | 93 % | all 8 jobs of `LC` stopped for the 7,600-token rule, at steps 28, 64, 53, 42, 59, 45, 54, 59 | yes |
| 14 | 80 % | `LC` 100.0 % on 404 steps, `L8` 100.0 % on 800 | yes |
| 15 | 92 % | `LC` 6,342 exact steps per GPU hour, `L8` 24,001 | yes |
| 16 | 88 % | `LN` 0.1007 s per step, `L0` 0.1000 (+0.6 %) | yes |
| 17 | 90 % | `LN` 35,764 exact steps per GPU hour, `L0` 35,113 | yes |

**Line 11 is the finding I did not expect, and I say so.** The page put it at 80 % because it counted a slip as staying until a later request removes the wrong item and took the fifth run's rate: about 15 of 64 capped jobs would slip and `L0` would hold the right state on about 87 % of the steps. The first half came out as written: 14 of 64 jobs of `L0` slipped. The second half did not: `L0` held the right state on 97.6 % of its steps, 156 inexact steps of 6,400, because most slips ended soon. The 15 runs of inexact steps lasted 1 to 60 steps (median 6); only job 195 kept its slip to the end of the job (60 steps). The page said that with `L0` near 98 percent, 100 steps might not yet be long enough to see the problem in this setup, and that a NO on line 12 would say so. Line 12 reads yes (95.5 % over steps 67 to 100 against 98.5 % over steps 1 to 33), so the control does lose ground as the job goes on, by 3 points and not by twelve; with 14 jobs that slipped, that is a measured size of the loss in this setup and not a rate for other tasks.

## The cells

| cell | jobs | steps | exact | steps 1 to 30 | steps 67 to 100 | jobs exact to the end | seconds | s per step | exact steps per GPU hour | per 1,000 tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| `L0` | 64 | 6,400 | 97.6 % | 98.6 % | 95.5 % | 50 | 640 | 0.100 | 35,113 | 9.3 |
| `LN` | 64 | 6,400 | 100.0 % | 100.0 % | 100.0 % | 64 | 644 | 0.101 | 35,764 | 9.5 |
| `L8` | 8 | 800 | 100.0 % | 100.0 % | 100.0 % | 8 | 120 | 0.150 | 24,001 | 8.6 |
| `LC` | 8 | 404 | 100.0 % | 100.0 % | - | 8 | 229 | 0.568 | 6,342 | 6.5 |

`L0` and `LN` are the same 6,400 requests on the same 64 jobs in one session, which is what lines 16 and 17 compare; `L8` and `LC` are the same 8 jobs. The computed prompt tokens per step are 95 (`L0`), 95 (`LN`), 105 (`L8`) and 143 (`LC`).

## What the page's rule lets this Result claim
The page's rule, applied line by line to what was read:
* **Lines 2, 3, 4 and 5 read yes**, so the figures of the cells are claimed as read.
* **Line 8 reads yes.** On steps 1 to 30 of the 64 new jobs `LN` is exact at all 1,920 steps. The page's condition for saying so is in the same sentence: the host wrote the coordinates of the request into the reply, and what the model wrote is the number (below, "What the model writes in `LN`").
* **Lines 9 and 10 read yes, line 11 reads NO and line 12 reads yes.** Over 100 steps `LN` is exact at all 6,400 steps and `L0` at 97.6 %; the gap of 2.4 points is below the three that the line asked for.
* **Lines 13, 14 and 15 read yes.** The kept chat stopped before step 100 in every job and held the state as exactly as the exact-state arm on the steps it ran (404 steps, 100.0 %), and did fewer exact steps per GPU hour.
* **Lines 16 and 17 are from cells of one session**, so a statement about time is allowed, between `L0` and `LN` and between `L8` and `LC` and not against any other run.
* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, "100 percent" for a grammar, "proven", a statement about any provider, about requests in free words, about a chat with a summary or a window, or about a model that was not run.

**The sentence the page allows, with this run's numbers.** On jobs 192 to 255, which no earlier run had seen, with the requests read by the host from their sentence, on one Tesla T4 (Colab), with vLLM 0.31.0, `Qwen/Qwen2.5-7B-Instruct-AWQ`, a cache of 80,000 tokens and jobs of 100 steps: the capped expression left the exact share at 97.6 % (6,244 of 6,400 steps; 95.5 % over steps 67 to 100) at 0.100 seconds per step, 35,113 exact steps per GPU hour and 9.3 per 1,000 tokens. The grammar that names the coordinates of the request had all 6,400 steps exact, 64 of 64 jobs exact to the end, at 0.101 seconds per step, 35,764 exact steps per GPU hour and 9.5 per 1,000 tokens, **and in the same sentence: in that grammar the host writes the coordinates of the request into the reply, the model writes the value of an added service or the weight of an added edge, and in every one of the 2,183 delete replies it wrote the node alone, the edges being completed by the framework's repair.** The kept chat stopped at its context after 28, 64, 53, 42, 59, 45, 54, 59 steps (jobs 192 to 199) and held the state on all 404 steps it ran at 6,342 exact steps per GPU hour where the exact-state arm had 24,001 on the same 8 jobs. One run, 64 jobs.

## The jobs that slipped
14 of 64 jobs of `L0` had an inexact step (194, 195, 199, 215, 219, 221, 222, 224, 225, 232, 233, 234, 243, 246); `LN` had none. Every one of the 14 first slips is a reply outside the named language of its own request; the 16 replies of `L0` outside it are in these 14 jobs (12 of an `add_edge` request, 2 of a `delete_dep`, 1 of a `delete_leaf`, 1 of an `add_node`; the whole run has 2,059 `add_edge` requests). One of them (job 225, step 12) left the state exact. The first slip, what it was, how long the slip lasted and what `LN` wrote at the same step:

| job | first slip at step | request | inexact steps in all | runs | `L0` replies outside the named language at steps | the `LN` reply at the first slip |
|---|---|---|---|---|---|---|
| 194 | 17 | add_edge | 14 | 14 | 17 | the oracle's text |
| 195 | 41 | add_edge | 60 | 60 | 41 | the oracle's text |
| 199 | 92 | delete_dep | 9 | 9 | 92 | a shorter delete, completed by the framework |
| 215 | 74 | add_edge | 4 | 4 | 74 | the oracle's text |
| 219 | 66 | add_edge | 1 | 1 | 66 | the oracle's text |
| 221 | 21 | add_edge | 10 | 2, 8 | 21, 26 | the oracle's text |
| 222 | 74 | delete_leaf | 27 | 27 | 74 | the oracle's text |
| 224 | 100 | add_edge | 1 | 1 | 100 | the oracle's text |
| 225 | 92 | add_edge | 7 | 7 | 12, 92 | the oracle's text |
| 232 | 93 | add_node | 8 | 8 | 93 | the oracle's text |
| 233 | 31 | add_edge | 1 | 1 | 31 | the oracle's text |
| 234 | 83 | add_edge | 4 | 4 | 83 | the oracle's text |
| 243 | 17 | delete_dep | 6 | 6 | 17 | a shorter delete, completed by the framework |
| 246 | 89 | add_edge | 4 | 4 | 89 | the oracle's text |

By the thirds of the job the first slips fall 4 / 2 / 8: more in the last third than in the first. The page expected the opposite (side prediction (b) below), and with 14 jobs this is a count, not a trend.

## What the model writes in `LN`
Same as in the fifth run, the host writes the coordinates of the request into the reply and the model writes the numbers. In `LN` the 4,217 add replies carry a value or a weight; of the 1,488 `delete_dep` replies 1,488 are the node alone (0 carry an edge item), and of the 695 `delete_leaf` replies 695 are the node alone; the edges of a deleted service are completed by the framework's repair, which this test does not change. The 100-step result for `LN` is therefore a result for the part of the reply that the model still writes, and it says nothing about a model that has to read the coordinates from a request in free words.

## The kept chat
| job | 192 | 193 | 194 | 195 | 196 | 197 | 198 | 199 |
|---|---|---|---|---|---|---|---|---|
| steps run | 28 | 64 | 53 | 42 | 59 | 45 | 54 | 59 |
| prompt of the last step | 7,512 | 7,498 | 7,586 | 7,516 | 7,501 | 7,489 | 7,589 | 7,477 |

All 8 stopped for the context (the next prompt would have passed 7,600 tokens); the prompt of the kept chat went from 2,531 tokens on average in steps 1 to 25 to 7,589 at the longest, where the exact-state prompt stays near 371. The kept chat computes more prompt tokens per step (143 against 105) and takes longer per step (0.568 s against 0.150), so it does fewer exact steps per hour; it is exact on all the steps it runs, and it cannot run the job to the end in this setup.

## The predictions written beside the lines, read afterwards

| written first | probability | measured | came out |
|---|---|---|---|
| (a) `L0` holds the right state on between 82 and 95 percent of its 6,400 steps | 70 % | 97.6 % | NO |
| (b) the first slips of `L0` fall in steps 1 to 33 for more jobs than in steps 67 to 100 | 70 % | 4 / 2 / 8 in the three thirds | NO |
| (c) the eight kept-chat jobs stop between steps 20 and 60 | 80 % | stops at 28 to 64; 7 of 8 inside | NO |
| (d) the computed prompt tokens per step of `LN` are within 5 percent of those of `L0` | 85 % | 94.9 and 94.6 | yes |
| (e) the four cells take 30 to 45 minutes in all, loading excluded | 70 % | 27.2 minutes | NO |

(f) *whether the model slips more often at step 80 than at step 8 when the grammar names the coordinates* cannot be read: `LN` has no slip, so the run does not say.

## What this does not show
Requests in free words (the host reads the sentence the job was built from, and the reply is built from what it read); tasks of more than 100 steps, let alone days; real company tasks; another model, size, GPU, engine or cache size; a kept chat with a summary or a sliding window (only the plain kept chat is run, and it stops at the context of this setup); a kept chat at 64 jobs; a retry on top of a grammar; dollars; a capacity in jobs per GPU; energy; hallucination in free text. Nothing here is proven.

## What comes next
Two things the page left out on purpose, each as its own update with its own page locked first: a reader for requests in free words (the host's copying of coordinates is the open question this result leaves), and the two controls that were missing (the target line at the top only, a plain restatement of the request).

# Test 43: 20,000 real steps on a 7B model, four jobs, the state held by the host (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before the harness is run; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
The longest run with a real model so far is 1,500 steps (Gemini, Test in `MILLION_TOKEN_TEST_HARD.md`), and the longest run of all is the 1.6 million steps of `ENDURANCE_TEST.md`, which used a stand-in with no model. A company that serves a model for long-lived jobs asks two plain questions of a host layer: after tens of thousands of steps, is the state the host holds still the intended one, and is the prompt the model is given still the same size as at step 1? **The question: on a real 7B model, over jobs of 20,000 steps, does the exact-state arm keep the held state equal to the oracle's, does the prompt stay flat, and does the time of a step stay flat?**

## What I read before writing this page
Offline, from the repository's job generator (CPU, no model), recorded here so the predictions are read knowing it:
* Asked for 20,000 steps, the generator ends a script early when no request fits the state (this is a property of the job generator, not of the framework or the model). Of jobs 192 to 519, 7 reach 20,000 steps: **192, 232, 241, 290, 394, 443 and 519**; the median job ends at about 2,600 steps. The jobs of this test are 192, 232, 241 and 290.
* For job 192 the held state stays between about 16 and 40 items over 20,000 steps (sampled every 2,500 steps), so the exact-state prompt should stay near its first-step size.
* In Test 38 (256 sessions) and Test 35 the same model held the state on every step of 100-step jobs with this grammar; the longest run of this model and grammar at present is the 1,500-step cell of Test 42 (not yet run when this page was written).

## What is run
The harness of Tests 35, 38 and 42, unchanged (`python/axi/experiments/provider_side_big.py`, repo commit `55631ec`): model `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0, one NVIDIA L4, prefix caching on, greedy decoding, no retry, the grammar that names the coordinates, the exact state shown at every step (arm S), policy `repair_m6`, `max_model_len` 8,192, cache 2,000 blocks (32,000 tokens). **One cell, `E4`: jobs 192, 232, 241 and 290 side by side in lockstep rounds, each job 20,000 steps (80,000 steps in all).** The driver prints a progress line every 1,000 rounds and a summary per window of 1,000 steps at the end. The engine is started in the background so that a frozen browser page cannot stop it. If the machine is lost during the run, the run is reported as not finished, with the last progress line, and is not resumed.

## What is measured
Per job and over the cell: the share of steps whose held state equals the oracle's, over the whole run and per window of 1,000 steps; the number of slips (a step on which the state stops being equal to the oracle's after being equal) and the number of steps each slip lasted; corrupted steps (a held state that is not well formed); the share of replies accepted by the strict parser; prompt tokens per step (mean per window); computed prompt tokens per step and the share served from the cache; seconds per round divided by the number of jobs (per window); the GPU board's mean power; steps per GPU hour.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error.

1. [instrument] all four jobs ran 20,000 steps without an engine error and at least 99.5 percent of replies were accepted by the strict parser (80 %)
2. [claim] each job held the oracle's state on at least 99 percent of its 20,000 steps (60 %)
3. [claim] for each job, the exact share in steps 15,001 to 20,000 is within 2 points of that in steps 1 to 5,000 (70 %)
4. [claim] no step had a held state that was not well formed (corrupted steps = 0) (90 %)
5. [claim] the mean prompt tokens in the last 1,000 steps are at most 1.3 times those of the first 1,000, for every job (90 %)
6. [claim] the seconds per step in the last 2,000 steps are within 15 percent of those in the first 2,000, for the cell (70 %)
7. [claim] the share of prompt tokens served from the cache is at least 60 percent in every window of 1,000 steps (75 %)
8. [claim] each job has at most 5 slips in its 20,000 steps (55 %), and every slip is over within 100 steps (40 %)

What would count against AXI here: a job with an exact share below 95 percent over its 20,000 steps; a prompt that grows to more than 1.3 times its first-window size; or any step with a held state that is not well formed. Each is reported as it is. A slip is a slip of the model that the gate and the repair did not complete, not a corruption of the held state: the held state is only ever changed by an edit the gate admits.

## Honest limits
One card (L4), one 7B quantised model, greedy decoding, one engine and its version, synthetic service graphs, four jobs of one family (the other jobs of the generator end before 20,000 steps), the grammar that names the coordinates (the capped grammar, which is looser, is not run here), the request given with coordinates. Four jobs side by side share the card, so seconds per step are for four jobs in lockstep, not for one job alone (Test 42 has the one-job time). The exact-state prompt is flat by construction when the held state stays near 16 to 40 items; a task whose state grows without limit is Test 40. The per-step files stay on the Colab machine; a compact export of the per-window figures, with the sha256 of the raw file, is saved with the Result.

## Result
*Written after the run, night of 16 to 17 Nov 2026. Nothing above this line was changed after the run (the plan page's sha256 before the Result was added: `e571b6583ef152722bb12325ae84685bb9098ae907529ad1b578d71545b2b4e8`). The architecture is Rajnish Choubey's; the measurement is the owner's, with an AI assistant as a tool. Everything below is about the state the host holds; the model is unchanged.*

Run: the harness of Tests 35, 38 and 42 (`provider_side_big.py`, repo commit `55631ec`), unchanged, driven by `docs/results/axi_endurance_7b_driver.py` (sha256 `900acc52a8ae989e594a4edad258903bab6ce7dba9cc892318bcc474f55976b6`) on one NVIDIA L4 on Colab: `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0, Python 3.13.16, prefix caching on, cache 32,000 tokens, `max_model_len` 8,192, greedy decoding, no retry, the grammar that names the coordinates, the exact state shown at every step (arm `S`), policy `repair_m6`. One cell, `E4`: jobs 192, 232, 241 and 290 in lockstep, 20,000 rounds, seed 20261017. All four jobs were active in all 20,000 rounds. The second machine was attached at about 15:24 UTC; after the set-up and a 241-second model load the first round began at about 15:55 UTC and the last ended at about 19:36 UTC (read from the log at 19:37 UTC), 13,230.4 seconds of wall time (3 h 41 min). Nothing was changed from the plan.

Files: `docs/results/axi_endurance_7b_summary.json` (the driver's summary, 3,467 bytes, sha256 `72f289a58eac016315900efb50a7d4f6c97953023a8e499c53db50ebb135928c`: per job, per window of 1,000 steps, power, and the size and sha256 of the raw file); `docs/results/axi_endurance_7b_after_run_checks.txt` (a cell run on the raw file after the driver, with its code and output: the per-job prompt figures of line 5, the seconds of line 6, the gate's counts); `docs/results/axi_endurance_7b_driver.py`. The raw file `t43_E4.json` (19,737,449 bytes, sha256 `96cc365c06787441db8c9abeda77091ba6589211e63885aec44f7432f09601f6`) stays on the Colab machine and goes when the machine does; its size and sha256 are saved here.

What happened that the plan did not foresee:
* **A first attempt was lost.** The first attempt ran for about 55 minutes on another machine, which Colab then reclaimed. The plan says a lost run is not resumed, so the whole run was started again from step 1 on a second machine. None of the first attempt's figures is used.
* **The notebook page froze several times** and a cell's console output stopped updating, so the progress was read from the log file on the machine (`t43.log`) through the Colab terminal. The run itself is a background process and was not affected. The progress lines came every 661 to 669 seconds per 1,000 rounds after the first (619 seconds for the first).
* **The per-job prompt figures of line 5 are not in the summary file** (the driver's windows pool the four jobs). I worked them out from the raw file with one cell after the run; its code and printout are in the checks file.

**Per job (20,000 steps each).**

| job | exact (of 20,000) | exact, steps 1 to 5,000 / last 5,000 | slips | parsed | corrupted | gate admitted the reply as written | not admitted (equals the steps of kind `delete_dep`) | prompt tokens, first 1,000 steps → last 1,000 (ratio) | largest prompt |
|---|---|---|---|---|---|---|---|---|---|
| 192 | 20,000 (100.0 %) | 5,000 / 5,000 | 0 | 20,000 | 0 | 15,178 | 4,822 | 595.5 → 606.8 (1.019) | 660 |
| 232 | 20,000 (100.0 %) | 5,000 / 5,000 | 0 | 20,000 | 0 | 15,298 | 4,702 | 575.9 → 642.4 (1.115) | 660 |
| 241 | 20,000 (100.0 %) | 5,000 / 5,000 | 0 | 20,000 | 0 | 15,223 | 4,777 | 564.5 → 626.3 (1.109) | 660 |
| 290 | 20,000 (100.0 %) | 5,000 / 5,000 | 0 | 20,000 | 0 | 15,245 | 4,755 | 561.3 → 602.7 (1.074) | 660 |

Over the cell: 80,000 of 80,000 steps exact, 80,000 parsed, 0 corrupted, 0 slips, every reply inside the grammar (20,000 of 20,000 in each job). Windows of 1,000 steps (20 of them, all in the summary file): the exact share of each of the four jobs is 1.0 in all 20 windows; the prompt is 574.3 tokens in window 1 and 614.6 to 624.9 in windows 2 to 20; computed prompt tokens are 271.4 in window 1 and 302.2 to 316.8 after; the cached share is 0.5274 in window 1 and 0.4927 to 0.5090 after; seconds a step (a round's seconds divided by the four jobs) are 0.1543 in window 1 and 0.1642 to 0.1673 after. Totals: 49,476,766 prompt tokens, 24,863,904 of them served from the cache (50.25 %), 816,529 new tokens (10.2 a step); 930 different grammars compiled. Time: 0.1654 seconds a step (0.6615 seconds a round of four jobs), 21,768 steps an hour with four jobs side by side. Board: mean 71.73 W (maximum 76.58 W, 26,514 samples), 951,223 J over the run, **11.89 J a step**.

**The eight lines, read.**
1. [instrument] **YES.** All four jobs ran 20,000 steps, no engine error, 80,000 of 80,000 replies accepted by the strict parser (100 %).
2. [claim] **YES.** Each job held the oracle's state on 20,000 of 20,000 steps (100.0 %).
3. [claim] **YES.** Steps 1 to 5,000 and the last 5,000: 100.0 and 100.0 on every job (difference 0 points).
4. [claim] **YES.** Corrupted steps: 0.
5. [claim] **YES.** Mean prompt tokens, last 1,000 steps over the first 1,000: 1.019, 1.115, 1.109 and 1.074. The largest of the four is 1.115, below 1.3; the largest prompt in any step of any job is 660 tokens.
6. [claim] **YES.** Seconds a step: 0.1605 in the first 2,000 rounds and 0.1651 in the last 2,000, a rise of 2.8 percent (limit 15). (The very first window, 0.1543, is the lowest because the state is smallest then.)
7. [claim] **NO.** The share of prompt tokens served from the cache is 0.4927 to 0.5274 in all 20 windows: none reaches 60 percent. It is flat (0.4927 to 0.5090 from window 2 on), but at about half, and my 60 percent was a guess from the shorter cells, which was wrong for this run. I did not look into why it is half.
8. [claim] **YES, YES.** Slips: 0 in each job (limit 5), so every slip was over within 100 steps; there were none.

Seven lines read YES and one NO.

**Against AXI, as the page set it down.** A job with an exact share below 95 percent: **not met** (all four 100.0). A prompt above 1.3 times its first-window size: **not met** (largest 1.115). A step with a held state that is not well formed: **not met** (0).

**What this says, in plain words, and what it does not.** On a real 7B model, over 80,000 steps in four jobs of 20,000, the state the host held was the oracle's on every step, the prompt the model was given stayed at about 620 tokens and never above 660, and the time of a step stayed at about 0.165 seconds. The host's state did not drift. That is a statement about the state the host holds, not about the model: the model's replies were admitted by the gate as written on 76.18 percent of the steps (60,944 of 80,000), and the steps it was not admitted on are exactly the steps of kind `delete_dep` (a delete of a service that other services depend on): in each job the count of steps not admitted equals the count of `delete_dep` steps (4,822, 4,702, 4,777, 4,755), and every other kind was admitted on every step. On those 19,056 steps (23.8 percent) the held state was still the oracle's: the gate and the repair of policy `repair_m6` did the work, as they did for arm `A` of Test 44 (`step_ok` 21.63 percent on that kind, `exact_policy` 100). So 100 percent exact is the host's exactness, not the model's.

**Honest limits, as on the page, and what the run added.** One card, one 7B quantised model, one engine version, greedy decoding, one seed, synthetic service graphs, four jobs of one family (the generator ends the other jobs before 20,000 steps), the grammar that names the coordinates. Four jobs share the card, so seconds and joules are for four jobs in lockstep. The exact-state prompt is flat here because the held state stays small (the largest prompt is 660 tokens); a task whose state grows is Test 40. No comparison arm was run in this test: a kept chat cannot run 20,000 steps in an 8,192-token context (Test 42 has the comparison at 1,500 steps). The 100 percent is one run; it says nothing about 100,000 steps, other grammars or other families. The raw per-step file stays on the Colab machine; the summary file and the checks file here carry the figures and the file's sha256.

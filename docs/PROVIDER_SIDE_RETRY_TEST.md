# Test 17: the serving side a fourth time, with one retry when the repair cannot fix a refused reply (written BEFORE the run, 19 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before the run has been made. Code: `python/axi/experiments/provider_side_retry.py` (it imports `provider_side.py`, `provider_side_constrained.py`, `provider_side_bounded.py`, the gate, the diagnostics, `agent_loop.py` and `cascade.py`, and edits none of them); instrument checks `python/tests/test_provider_side_retry.py`; notebook `python/colab/axi_provider_side_retry.ipynb`. Concept-map item 109. It follows `PROVIDER_SIDE_BOUNDED_TEST.md` (item 108) and the Result at the end of that page; read that Result first. It needs a GPU (a Colab T4 is the one assumed), the engine vLLM 0.31.0, and about 45 minutes in all (see "How it is run").

## What the third run left open
1. **The exact-state arm `S` held the right state on 88.3, 94.6 and 97.2 percent of its steps at 8, 32 and 64 jobs; the kept chat on 100.** Lines 12 and 13 of that page read NO, and the Result said what that does and does not allow: no plain time ratio, and the exact-steps-per-hour figures with the exactness beside them.
2. **Where the missing steps went.** At 64 jobs, six of the 64 jobs missed, each at one step, and a job that misses stays behind for the rest of its 30 steps. I replayed the first miss of each through the gate (the same check the run uses):

| job | step | the engine answered | the oracle wrote | the gate | the repair of `repair_m6` | steps lost by the job |
|---|---|---|---|---|---|---|
| 3 | 16 | `ADD[(4,4)->(4,4):value#6]` | `ADD[4,4:6]` | refused: edge endpoint missing | cannot fix | 15 |
| 4 | 18 | `ADD[1,0:6]` | `ADD[0,0:3]` | refused: node (1,0) already exists | cannot fix | 13 |
| 17 | 20 | `DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]` | `ADD[(0,4)->(2,3):dep#4]` | admitted | not needed | 4 |
| 19 | 12 | `ADD[(3,0)->(3,0):4tworowsdown4col#4]` | `ADD[3,0:4]` | refused: edge endpoint missing | cannot fix | 19 |
| 20 | 30 | `ADD[(1,5)->(2,1):dep#3]` | `ADD[(1,1)->(2,1):dep#3]` | refused: edge already exists | cannot fix | 1 |
| 62 | 12 | `DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]` | `ADD[(1,1)->(1,0):dep#9]` | admitted | not needed | 1 |

   **Four of the six misses are replies the gate refused and the repair could not fix; 48 of the 53 lost steps at 64 jobs are theirs** (jobs 3, 4, 19 and 20: 15, 13, 19 and 1). In the run the job was never asked again: under `repair_m6` the state stayed as it was and the next request was built on a state one step behind. The other two misses are replies the gate admitted (an extra delete in the same reply); a retry does not act on them, and they are 5 steps.
3. **The framework already has the next rung of its own ladder.** `agent_loop.py` has a mode with one retry whose feedback is written by the framework (`M3`, feedback variants in `feedback_text`), and the concept map reads the design as rules, then feedback, then bounded retries, then abort ("asking once and walking away", rows 26 and 42; the row says that only one retry has been measured and that a budget sweep is open). That retry was measured in the offline loop, one request after another, with a 3B model, on deletes of a service with edges: the feedback listed the edges to delete and the retry recovered all 120 such deletes on four seeds. It was never put into the serving loop, and its cost there was never counted. This test does that, and only that.
4. **What the retry cannot do, said before anything else.** It does not act on a reply the gate admits (jobs 17 and 62), nor on a reply the repair already completes (at 64 jobs the gate refused 507 deletes of a service that still had edges; the repair completes every one of them, 491 end exact and the other 16 are steps of jobs 3, 4 and 19 that were already behind; the rule below does not fire on any of the 507). It does not change a first reply. And **I do not know whether this 7B model uses the feedback**: for the kinds in the table the feedback is short (the kind of obstruction and the node or edge it is about, below), not a list of what to do, and nothing measured earlier says how Qwen2.5-7B answers it. The most the retry can win at 64 jobs is the 48 steps of four jobs: 1,915 of 1,920 steps (99.7 percent) against 1,867 (97.2). That is a ceiling, not a prediction: a job that is put right could slip again later.

## What changes and what does not
**Does not change:** the model (`Qwen/Qwen2.5-7B-Instruct-AWQ`), the engine (vLLM 0.31.0), the GPU assumed (a Colab T4), the cache fixed at **80,000 tokens**, `max_model_len` 8,192, greedy decoding, at most 96 new tokens, prefix caching on, the 30-step jobs with job e at seed 20261017 x 1000 + e, the prompts, the capped expression of the third run (name of 1 to 16 characters; sha256 `6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245`), the policy `repair_m6`, the lockstep rounds, the measures (exact steps per GPU hour: 3,600 times the steps after which the held state equals the oracle's, over the wall seconds of the cell). **The gate, the repair, the diagnosis, the retry feedback and the policy are the framework's, unedited**, and the files this test imports are pinned by sha256 in the test: `provider_side.py` `bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3`, `provider_side_constrained.py` `03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d`, `provider_side_bounded.py` `7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc`, `gate.py` `b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7`, `diagnostics.py` `255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2`, `agent_loop.py` `d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6`, `cascade.py` `b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7`.
**Changes:**
* **The retry rule.** After the first reply of a step, the job asks once more if and only if the strict parser accepts the reply, the gate refuses it, and `ground_and_complete` (the repair of `repair_m6`, with its default degree threshold 4) returns no repair that the gate admits. These are exactly the replies after which `repair_m6` leaves the state where it was although the model did answer. A reply that does not parse is not asked again (as in `agent_loop`). At most one retry per step.
* **The second request** is the first request, then the first reply as the assistant's turn, then one user turn written by the framework: `feedback_text("structured_imperative", diagnose(state, delta))`. For the four replies of the table the user turn would be (this is computed by the code and checked in the test):
  * job 3: `OBSTRUCTION malformed edge=((4, 4), (4, 4), 'value') Reply with only the corrected delta.`
  * job 4: `OBSTRUCTION ident_node node=(1,0) Reply with only the corrected delta.`
  * job 19: `OBSTRUCTION malformed edge=((3, 0), (3, 0), '4tworowsdown4col') Reply with only the corrected delta.`
  * job 20: `OBSTRUCTION ident_edge edge=((1, 5), (2, 1), 'dep') Reply with only the corrected delta.`

  The variant is the one that recovered 100 percent of the dangling deletes in the earlier tests; for the kinds above it writes the same text as the plain "structured" variant. It is the same grammar on the second reply: a second reply is as well formed as a first.
* **The second reply decides the step**, as any reply does under `repair_m6`: applied if the gate admits it or the repair makes it admissible, otherwise the state stays. A second refusal is accepted ("ask once and walk away").
* **The retry is paid for.** The jobs that ask again are served in a second, smaller round of the engine after the round of first requests; its seconds are part of the cell's wall seconds and so of the seconds per step and the exact steps per GPU hour. A real server would mix those requests into other traffic and pay less than a whole round; the lockstep model charges the whole round, which is against the retry.
* **The cells are five**, in this order (cheap first):

| cell | jobs | side by side | retry | compared with |
|---|---|---|---|---|
| `R8` | 0 to 7 | 8 | on | the third run's `S` at 8 jobs (same jobs) |
| `R32` | 0 to 31 | 32 | on | the third run's `S` at 32 jobs |
| `R64` | 0 to 63 | 64 | on | the third run's `S` at 64 jobs |
| `H0` | 64 to 127 | 64 | off | `H1`, same session |
| `H1` | 64 to 127 | 64 | on | `H0`, same session |

  About 15 minutes of GPU. **Why the last two exist.** I chose to test a retry because I had read the four slips in jobs 3, 4, 19 and 20; a result on those same jobs can only say that the retry is able to fix those slips. Jobs 64 to 127 have not been run by any earlier run, I have not read their replies, and the pair `H0`/`H1` runs them without and with the retry in one session. **Only the pair can say anything about jobs in general, and only the pair gives a time comparison that is not across two Colab sessions.** The kept chat is not run in this test.
* **A replay check is part of the test.** The first replies of the three small cells are compared with the third run's, step by step (line 3), and the first replies of `H1` with those of `H0`.
* **The reading lines are 12** (below). The headline is the exactness and the exact steps per GPU hour on the new jobs (lines 9 and 11).

## What I read in the third run's files before writing this page
* **The rule fires rarely.** I replayed the saved replies of the third run's three `S` cells through the rule, without any retry: it fires on 5 of 240 steps at 8 jobs (2.1 percent), 12 of 960 at 32 jobs (1.25) and 12 of 1,920 at 64 jobs (0.6). Every one of them is a step of job 3, 4, 19 or 20. The first fire of each of these jobs is its first miss (the four in the table); the others come after it, in a job that is already wrong. With a retry that works those later ones should not arise. Against `next_state` of `repair_m6` the rule agrees on all 3,360 steps: it fires exactly where the policy would leave the state unchanged after a reply that parsed and was refused.
* **The ceiling, per cell.** If every step of jobs 3, 4, 19 and 20 became exact: 240 of 240 at 8 jobs (100.0 percent; the two jobs that missed there are 3 and 4), 956 of 960 at 32 jobs (99.6), 1,915 of 1,920 at 64 jobs (99.7), against 212, 908 and 1,867 now.
* **The third run's `S` cells, for reference:** 39.4, 119.3 and 217.4 wall seconds; 0.164, 0.124 and 0.113 seconds per step; 19,379, 27,390 and 30,914 exact steps per GPU hour. A second Colab session was 4 to 8 percent slower in every cell than the one before it (the third run's Result), so a cell of this run is not compared with these figures for time.
* **What a retry round may cost.** Not measured. A second round holds 1 to 5 requests whose prompt is nearly all in the cache and whose reply is at most a few dozen tokens, so I expect 0.6 to 1.2 seconds a round on a T4. At 64 jobs the rule fires in at most 10 different rounds without a retry, probably 4 to 6 with one: that is 3 to 6 seconds of 217, between 1 and 3 percent. At 8 jobs the same seconds are a larger share of 39 (up to 6 percent). That is an estimate to be checked, not a prediction of the figures.

## Design in one place
| | |
|---|---|
| model, engine | `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0 (`pip install vllm==0.31.0`); structured output by the capped regular expression, as the third run, for the first and the second request |
| GPU, cache | a Colab T4 (any other GPU is recorded and the cells are not merged; line 3 could then read NO for that reason alone); cache 80,000 tokens by the block override |
| jobs | 30 steps, seed 20261017; jobs 0 to 63 as in all earlier runs, jobs 64 to 127 new |
| arm | `S` only: the exact state each step, cache on |
| policy, rule | `repair_m6`, no rule in the prompt; one retry by the rule above |
| cells, order | `R8`, `R32`, `R64`, `H0`, `H1` |
| measures | as the third page, plus: the number of second requests, the seconds of the second rounds, the first reply and the reason of every retried step, the steps after which the held state was exact in the retried steps |
| safety cell | before the cells: 3 jobs of 3 steps with the retry on (cached counts reported, every reply parsed), and **one probe**: the second request for job 19, step 12 (the exact-state prompt rebuilt with the oracle's own replies for steps 1 to 11, the third run's refused reply as the assistant's turn, the feedback above). The probe prints what the engine answers; **a wrong answer is not a STOP** (that is what the test is for), only a reply that does not parse, or one that uses all 96 new tokens, or a first request without cached counts stops the notebook |

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 5 are checks that the run is the run I describe; lines 6 to 12 are about the retry, and for those I have no measurement to lean on: the only thing I know is where the four slips are, not what the model does when told.

1. [instrument] the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run, the second requests included (98 %)
2. [instrument] the constraint held: of all the replies of all the cells that were run (first and second) at least 99.5 percent were accepted by the strict parser, and none used all 96 new tokens (92 %)
3. [guard] the first replies are the earlier run's: in the three cells on jobs 0 to K-1, every job's first replies up to and including its first retried step are the same text as the third run's, and every job that asked again nowhere is the same text throughout; in the pair on jobs 64 to 127, the same between the cell with the retry and the cell without (85 %)
4. [guard] the retry fired only where the rule says: every second request was made for a reply that the strict parser accepted, the gate refused and the repair could not make admissible; and the cell without the retry made none (93 %)
5. [guard] the retry is rare: in every cell with the retry at most 3 percent of the steps asked again (90 %)
6. [claim] on the same jobs as the third run, the exact share is higher with the retry at every K: above 88.3, 94.6 and 97.2 percent for K = 8, 32 and 64 (85 %)
7. [claim] on jobs 0 to 63 (K = 64) the exact share with the retry is at least 98 percent (the third run: 97.2; the most it could be is 99.7) (75 %)
8. [claim] on jobs 0 to 63 (K = 64) the exact share with the retry is at least 99 percent (35 %)
9. [claim] on the jobs that no earlier run has seen (64 to 127), the exact share with the retry is at least one point above the share without it (55 %)
10. [claim] on those jobs, the seconds per step with the retry are at most 10 percent above those without it (both in this session) (88 %)
11. [claim] on those jobs, the exact steps per GPU hour with the retry are higher than without it (both in this session) (40 %)
12. [claim] on those jobs, no job holds the right state on fewer steps with the retry than without it (70 %)

**Headline** (lines 9 and 11): does one retry raise the exactness on jobs nobody has looked at, and does it raise the exact steps per GPU hour once its rounds are paid for. Lines 2, 3 and 4 all reading yes: about **75 %**. Lines 9 and 11 both reading yes: about **30 %**. The pattern I expect (lines 1 to 7, 9, 10 and 12 yes; 8 and 11 NO): about **12 %**. All 12 yes: about **6 %**.

Why I think so, line by line. *Line 1 (98 %):* it worked in all earlier runs, same engine; a second request is one more request. *Line 2 (92 %):* the second request carries a longer prompt (the first reply and the feedback) under the same grammar; the capped name removed the one way a reply ran on, and the second reply is shorter than a long name could make it. *Line 3 (85 %):* replies were the earlier run's in 3,358 of 3,360 steps in a new session, so the first replies should be again; the risk is that a second round in the middle of the run changes the state of the engine's cache and, with it, a number. *Line 4 (93 %):* it is the rule written into the code; the risk is a mistake of mine in the record. *Line 5 (90 %):* the replay says 0.6 to 2.1 percent without a retry, and fewer if it works; the risk is a retry reply that is refused again and a job that goes on being refused. *Line 6 (85 %):* the four jobs are 4 slips in 64 jobs and the retry has four chances to be taken up; at 8 jobs it needs one of two. *Line 7 (75 %):* needs about 15 of the 48 steps back: job 19 (19 steps) or job 3 (15), or two of the others. *Line 8 (35 %):* needs 34 of the 48: job 19 and job 3 both put right, and no new slip. *Line 9 (55 %):* by the rate of the third run I expect about 6 slips in 64 new jobs, 4 of them of the refused kind; if the retry fixes two thirds of them at 12 lost steps each, the gain is 1.6 points, which is one point with a margin that a count of 4 slips does not give. *Line 10 (88 %):* 1 to 3 percent expected against 10. *Line 11 (40 %):* the exact steps gained (1.6 percent of the steps, if I am right) and the time paid (1 to 3 percent) are about the same size; it can fall either way, and that is why it is a line. *Line 12 (70 %):* a retry can put a job right, and it can also be admitted and wrong where the refusal would have left the job one step behind; the first is likelier.

**Also written down, not lines (read afterwards as descriptions).** (a) At 64 jobs on jobs 0 to 63 the number of second requests is between 4 and 12: 85 %. (b) The second reply of job 19, step 12 is `ADD[3,0:4]`, the oracle's: 55 %. (c) Every second reply parses: 97 %. (d) At least 60 of the 64 jobs of `R64` hold the right state at every step (the third run: 58): 60 %. (e) A second round takes 0.4 to 2.0 seconds on average: 80 %. (f) `H0` is between 90 and 99 percent exact: 80 %. (g) The five cells take 12 to 25 minutes in all, loading excluded: 70 %.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers.
* *Lines 2, 3, 4 yes, lines 6 and 7 yes, lines 9, 10 and 12 yes:* the retry works on this model and on jobs nobody had read, at a cost that fits inside the ten percent. The sentence is that exactness went from X to Y percent on new jobs for Z more seconds per step. Whether it raised the exact steps per GPU hour is line 11, whichever way it falls.
* *Lines 6 and 7 yes, line 9 NO:* it fixes the slips I had read and not the new ones; the gain was fitted to the jobs it was designed on, or there were too few refusable slips among the new jobs to see it. The claim then stays on jobs 0 to 63 and says so.
* *Line 6 NO:* the model does not use the feedback (or uses it to repeat itself). Then the finding is that the terse feedback of the framework does not help this model on these kinds, and the next thing to try is a different feedback, which would be a different test written first.
* *Line 11 NO with line 9 yes:* more exact steps, fewer per hour: in this model the retry buys exactness at more than it is worth in time. That is the honest sentence; a server that mixes retries into other traffic might pay less, which this test cannot show.
* *Line 10 NO:* the second rounds cost more than ten percent. Look at the seconds of each second round.
* *Line 12 NO:* a retry made some job worse. Read which job, what the second reply was and why the gate admitted it.
* *Line 5 NO:* too many second requests: read whether second replies were refused again.
* *Line 3 NO with line 2 yes:* the first replies differ from the third run's by something other than a retry. That would be the engine or the session; read which replies differ before reading anything else.
* *Line 4 NO:* a second request was made where the rule does not say so, or the cell without the retry made one: a mistake in the code, read before anything else.
* *Line 2 NO:* the constraint did not hold in some request; read it.
* *Line 1 NO, or the safety cell says STOP:* nothing is run or read.
* *A cell that did not run:* its lines read "not run", never NO.

## Reading rules (fixed now; the printout prints exactly these)
The twelve lines above are the reading rules; `provider_side_retry.verdict_retry` computes them from this run's cells and the third run's file. Line 1 uses every request of every cell; line 2 every reply of every cell (the share the strict parser accepted, and the replies that used all 96 new tokens); line 3 each of `R8`, `R32` and `R64` against the third run's `S` cell with the same number of jobs, and `H1` against `H0`, job by job (the first reply of every step up to and including the job's first retried step; a job that never asked again, throughout and with the same number of steps); line 4 every retried step (the first reply parsed, was not admitted, was not repairable) and the cell `H0` (no retried step); line 5 each cell with the retry (retried steps over steps, at most 0.03); line 6 `R8`, `R32` and `R64` against the third run's exact share (strictly higher in each); lines 7 and 8 `R64` (0.98 and 0.99); line 9 `H1` against `H0` (the exact share at least 0.01 higher); line 10 the seconds per step of `H1` against `H0` (at most 1.10 times); line 11 the exact steps per GPU hour of `H1` against `H0` (higher); line 12 every job of the pair (the steps after which the held state was exact in `H1` at least those in `H0`). Definitions as on the earlier pages: "accepted by the strict parser" is `strict_parse(text) is not None`; "admitted by the gate" is that the parsed reply passes the gate against the state the job held; "exact steps per GPU hour" is 3,600 times the steps after which the held state equals the oracle's, over the wall seconds of the cell (both kinds of round). A line that needs a cell the run does not hold reads "not run".

**The rule that decides what the Result may claim.** Lines 2, 3 and 4 all read yes: the figures of the cells are claimed as read. That the retry raises the exactness is claimed only from the jobs that no earlier run had seen (lines 9 and 12); lines 6 to 8 are on the jobs on which the slips were read and say what the retry does to them. A statement about time or cost comes only from the pair in one session (lines 10 and 11), never from a cell against the third run's (another Colab session). Line 2, 3 or 4 reads NO: no claim is made from this run until the cause is found.

**How the result will be worded.** On jobs the earlier runs had not seen, with one retry on a Tesla T4, one 7B model and the engine written next to the figures: exactness from X to Y percent, seconds per step from A to B, exact steps per GPU hour from C to D, with the number of second requests and the number of jobs they put right. Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never "100 percent" for the arm, never "proven" (one model, one GPU, one run, 64 new jobs), never a statement about any provider, never a statement about more than one retry, and never a statement about the roadmap at tensor level. A "NO" on a claim line is a finding about this setup and goes into the article as such.

## What this does not show
More than one retry (a budget sweep is not run); another feedback wording (one variant); another model, GPU, engine or cache size; dollars; a capacity in jobs per GPU; energy (never measured; only time); the kept chat with a retry (not run); retries in a real server, where they would be mixed with other requests and cost less than a whole round; that the retry fixes slips of kinds not seen in the jobs it was read on; replies in ordinary words; jobs longer than 30 steps. It does not change what the gate admits or what the repair completes, so the exactness it can add is only on replies the framework already refuses. It says nothing about the tensor-level roadmap.

## How it is run
Open `python/colab/axi_provider_side_retry.ipynb` in Colab with a T4 and run the cells one at a time: it installs vLLM 0.31.0, connects Drive, fixes the output stream, loads the model with the capped constraint, runs the safety cell (3 jobs of 3 steps, then the probe), then the five cells in order. Each finished cell is written as `axi_provider_r_<label>.json` and copied to Drive; a cell that was cut off is run again from its start (the longest takes about 4 minutes). The last cell puts the five cells together, reads the third run's file from the repository, prints the table and the 12 lines, and offers the files for download. The Result goes below the line at the end of this page, and a result test will recompute every number in it from the saved files.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_retry.py` and again by `tests/test_provider_side_retry_result.py`). The run was made by the owner on Colab with `python/colab/axi_provider_side_retry.ipynb`; the file reached me on 20 Oct 2026 and holds no date of its own. It records: vLLM 0.31.0, GPU Tesla T4, Python 3.13.15, `Qwen/Qwen2.5-7B-Instruct-AWQ`, the cache fixed at 80,000 tokens with the engine's block override (capacity source "override"), 30-step jobs, seed 20261017, every reply constrained by the capped expression of the third run (sha256 `6f4e5a993f26ccd4…`), the policy `repair_m6`, and one retry with the `structured_imperative` feedback. **All five planned cells ran** (`R8`, `R32`, `R64`, `H0`, `H1`); each holds its 30 first rounds in order. The file is `docs/results/axi_provider_retry.json` (sha256 `390f7464a54352764e21203f9546bd17fc4714b09cbb0aec9390963e5d94b116`): the merged file that the notebook's last cell wrote with `provider_side_retry.merge_files`. It records the sha256 of that module (`9cd1398b912c187d…`, the module in the repository), those of the seven files the test imports (all as pinned on the page), and the third run's file it was compared with (`axi_provider_bounded.json`, `1f12a846b1325f9f…`). The seconds around the engine's calls of the five cells add up to 818.3, which is 13.6 minutes, 435.4 of them in the pair. What the safety cell and the probe printed is not in the saved file, and nothing below rests on it. Every figure below is computed from the saved file by `tests/test_provider_side_retry_result.py`.*

**What the page expected, and what came out.** The page put lines 2, 3 and 4 all yes at about 75 %, and they read yes, as do lines 1 and 5: the run is the run the page describes. About the retry itself it said that it did not know whether a 7B model uses a terse feedback, and it wrote down in advance what each outcome would mean. The nearest of those is "*Line 6 NO:* the model does not use the feedback (or uses it to repeat itself)", and what the page says to do then stands: "the next thing to try is a different feedback, which would be a different test written first". **8 of the 12 lines read yes; lines 6, 8, 9 and 11 read NO, and no line is "not run".** The headline (lines 9 and 11) reads NO on both: on the 64 jobs that no earlier run had seen, 98.1 % of the steps are exact with the retry and 98.1 % without it, and the exact steps per GPU hour are 30,951 with it against 31,359 without it. Line 8 was expected NO (35 %). Line 6 reads NO only because of `R8` (212 of 240 steps exact in the third run and in `R8`); at 32 and 64 jobs it is higher (927 against 908 steps and 1,889 against 1,867), and that gain is job 19 (19 steps) at both, plus 3 steps of job 3 at 64 jobs (steps 28 to 30), which are exact for another reason (below).

## The 12 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 98 % | the engine reported the cached tokens of every request: 6,989 requests in the 5 cells, 29 of them second requests | yes |
| 2 | 92 % | replies accepted by the strict parser: 6,989 of 6,989 (first and second), 100.0 % (limit 99.5 %, met); replies that used all 96 new tokens: none (the longest has 31 tokens) | yes |
| 3 | 85 % | first replies against the third run's `S` cell (`R8`, `R32`, `R64`) and `H1` against `H0`, job by job, up to each job's first retried step: all the same text (3,120 steps in the three cells and 1,920 in the pair compared; the jobs that asked again are 3, 4, 19 and 20 in the three cells and 101 and 122 in `H1`) | yes |
| 4 | 93 % | 29 second requests, every one for a first reply that parsed, was refused by the gate and could not be repaired; `H0` made none | yes |
| 5 | 90 % | retried steps over steps: `R8` 7 of 240 (2.9 %), `R32` 9 of 960 (0.9 %), `R64` 8 of 1,920 (0.4 %), `H1` 5 of 1,920 (0.3 %) (limit 3 %, met) | yes |
| 6 | 85 % | exact steps with the retry against the third run's `S`: K = 8 212 of 240 (88.3 %) against 212 (88.3 %), K = 32 927 of 960 (96.6 %) against 908 (94.6 %), K = 64 1,889 of 1,920 (98.4 %) against 1,867 (97.2 %) (limit: higher in each; equal at K = 8) | NO |
| 7 | 75 % | `R64`: 1,889 of 1,920 steps exact, 98.4 % (limit 98 %, met) | yes |
| 8 | 35 % | `R64`: 98.4 % (limit 99 %, not met; the most it could be was 99.7 %) | NO |
| 9 | 55 % | jobs 64 to 127: `H1` 1,884 of 1,920 exact (98.1 %), `H0` 1,884 (98.1 %): +0.0 points (limit +1.0) | NO |
| 10 | 88 % | `H1` 0.1141 seconds per step, `H0` 0.1126: 1.013 times (limit 1.10, met) | yes |
| 11 | 40 % | `H1` 30,951 exact steps per GPU hour, `H0` 31,359: -1.3 % (limit: higher) | NO |
| 12 | 70 % | the 64 jobs of the pair: 64 hold the right state after as many steps with the retry as without, 0 after more, 0 after fewer | yes |

**8 of 12 lines read yes, lines 6, 8, 9 and 11 read NO, and no line is "not run".** The page did not name this exact pattern; its nearest entry is the one for line 6 (above). The four NO stood at 85 % (line 6), 55 % (line 9), 40 % (line 11) and 35 % (line 8) on the page.

## What the page's rule lets this Result claim
The page's rule, applied line by line to what was read:
* **Lines 2, 3 and 4 read yes**, so the figures of the cells are claimed as read.
* **Line 9 reads NO and line 12 reads yes.** The page allows the claim that the retry raises the exactness only from the jobs that no earlier run had seen (lines 9 and 12). It did not raise it there (98.1 % with the retry, 98.1 % without), and no job was made worse by it. **This Result does not claim that the retry raises the exactness.**
* **Lines 6 to 8 are about the jobs the slips were read on** (jobs 0 to 63), and say what the retry does to them: it put right one of the four slips that were read (job 19, step 12) and none of the other three, and the exact share at 64 jobs is 98.4 % against 97.2 % in the third run (another Colab session, same replies up to each job's first retried step).
* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, "100 percent" for the arm, "proven", a statement about any provider, or a statement about more than one retry.

**The sentence the page allows, with this run's numbers.** On the 64 jobs that no earlier run had seen (jobs 64 to 127), on one Tesla T4 (Colab), with vLLM 0.31.0, `Qwen/Qwen2.5-7B-Instruct-AWQ`, a cache of 80,000 tokens, 30-step jobs and every reply constrained by one regular expression, one retry with the framework's one-line feedback left the exactness at 98.1 % where it was 98.1 % without it (1,884 and 1,884 of 1,920 steps), the seconds per step at 0.114 against 0.113, and the exact steps per GPU hour at 30,951 against 31,359 (both in the same session). The retry asked 5 times and put none of them right. On jobs 0 to 63, where the four slips had been read, it put right 1 of the four (job 19) and the exact share at 64 jobs is 98.4 % against 97.2 % in the third run.

## What the 29 second requests did
A second request is the first request, the first reply as the assistant's turn, and the framework's feedback as the user's turn (for example, in `R8`, job 3, step 16: `OBSTRUCTION malformed edge=((4, 4), (4, 4), 'value') Reply with only the corrected delta.`). 29 second requests were made in the four cells with the retry, 26 second rounds (jobs asking in the same step share a round), costing 12.03 seconds in all. Every second reply was accepted by the strict parser.

| cell | steps asked again | second rounds | their seconds | per round | share of the cell's seconds | same reply | other reply, refused again | other reply, admitted and wrong | put right | exact after the step |
|---|---|---|---|---|---|---|---|---|---|---|
| `R8` | 7 | 6 | 2.40 | 0.40 | 6.2 % | 6 | 0 | 1 | 0 | 0 |
| `R32` | 9 | 8 | 3.56 | 0.45 | 2.9 % | 6 | 1 | 1 | 1 | 1 |
| `R64` | 8 | 7 | 3.10 | 0.44 | 1.4 % | 6 | 0 | 1 | 1 | 2 |
| `H1` | 5 | 5 | 2.97 | 0.59 | 1.4 % | 3 | 2 | 0 | 0 | 0 |
| all | 29 | 26 | 12.03 | 0.46 | | 21 | 3 | 3 | 2 | 3 |

* **17 of the 29 were asked after the job had already slipped, and in 14 of those the first reply was, letter for letter, the oracle's own reply.** Each of these 14 replies is legal in the oracle's state at its step and was refused in the state the job held, which an earlier slip had left different from the oracle's: they are job 3 at step 28 and job 4 at steps 19, 24 and 28 (in `R8`, `R32` and `R64`) and job 101 at steps 7 and 27 (in `H1`). 12 of the 14 came back as the same reply; 2 came back changed to another reply that was refused as well (`R32`, job 4, step 19: `dep#7` to `dep#0`; `H1`, job 101, step 7: `ADD` to `DEL`). **The rule cannot tell these from a real slip:** it knows that the gate refused a reply that parsed, and nothing about whether the reply is what the request asked.
* **The other 15 first replies were not the oracle's:** 2 were put right (job 19, step 12, in `R32` and `R64`: `ADD[(3,0)->(3,0):4tworowsdown4col#4]` became `ADD[3,0:4]`, the oracle's reply); 9 came back as the same reply; 3 came back as `DEL[1,0]`, which the gate admits and which is wrong (job 4, step 18, in `R8`, `R32` and `R64`: the feedback said that the node (1,0) already exists, and the model deleted it); 1 came back changed and refused again (`H1`, job 122, step 22: `ADD` to `DEL`).
* **The "exact after the step" column counts steps, not repairs.** In `R64` it is 2 and the replies put right are 1: the other step is job 3, step 28, where the same refused reply `DEL[4,4]` came back and the job is exact anyway, because the node (4,4) that the oracle deletes there was never added in that job (its add at step 16 was refused), so the refusal leaves the same graph as the oracle's.
* **`R8` asked 7 times where the page's replay of the third run gave 5.** The extra two are job 4, steps 19 and 24: after step 18 was answered with the admitted `DEL[1,0]`, job 4 stood in another wrong state than in the third run, and its later replies were refused at other steps (the rule fired at steps 18, 19, 24 and 28 instead of 18 and 28).
* **Time.** A second round took 0.40, 0.45, 0.44 and 0.59 seconds on average in `R8`, `R32`, `R64` and `H1` (0.46 over all), and the rounds of the retry were 6.2 %, 2.9 %, 1.4 % and 1.4 % of the seconds of their cells. In the pair, `H1` took 219.1 seconds against 216.3 for `H0` (+1.3 %), and the retry bought no exact step there, so the exact steps per hour fell by 1.3 %.

The 29 second requests, one row each ("why the gate refused it" is the first word of the gate's reason; "first reply was the oracle's own" means that it is, letter for letter, what the oracle writes at that step):

| cell | job | step | first reply | why the gate refused it | second reply | what came back | exact after |
|---|---|---|---|---|---|---|---|
| `R8` | 3 | 16 | `ADD[(4,4)->(4,4):value#6]` | well-formedness | `ADD[(4,4)->(4,4):value#6]` | same | no |
| `R8` | 3 | 17 | `DEL[(1,4)->(1,1):dep#2]` | match | `DEL[(1,4)->(1,1):dep#2]` | same | no |
| `R8` | 3 | 28 | `DEL[4,4]` | match | `DEL[4,4]` | same (first reply was the oracle's own) | no |
| `R8` | 4 | 18 | `ADD[1,0:6]` | identification | `DEL[1,0]` | admitted, wrong | no |
| `R8` | 4 | 19 | `ADD[(1,2)->(1,0):dep#7]` | well-formedness | `ADD[(1,2)->(1,0):dep#7]` | same (first reply was the oracle's own) | no |
| `R8` | 4 | 24 | `ADD[(1,0)->(4,5):dep#7]` | well-formedness | `ADD[(1,0)->(4,5):dep#7]` | same (first reply was the oracle's own) | no |
| `R8` | 4 | 28 | `ADD[(3,4)->(0,0):dep#5]` | well-formedness | `ADD[(3,4)->(0,0):dep#5]` | same (first reply was the oracle's own) | no |
| `R32` | 3 | 16 | `ADD[(4,4)->(4,4):value#6]` | well-formedness | `ADD[(4,4)->(4,4):value#6]` | same | no |
| `R32` | 3 | 17 | `DEL[(1,4)->(1,1):dep#2]` | match | `DEL[(1,4)->(1,1):dep#2]` | same | no |
| `R32` | 3 | 28 | `DEL[4,4]` | match | `DEL[4,4]` | same (first reply was the oracle's own) | no |
| `R32` | 4 | 18 | `ADD[1,0:6]` | identification | `DEL[1,0]` | admitted, wrong | no |
| `R32` | 4 | 19 | `ADD[(1,2)->(1,0):dep#7]` | well-formedness | `ADD[(1,2)->(1,0):dep#0]` | refused again (first reply was the oracle's own) | no |
| `R32` | 4 | 24 | `ADD[(1,0)->(4,5):dep#7]` | well-formedness | `ADD[(1,0)->(4,5):dep#7]` | same (first reply was the oracle's own) | no |
| `R32` | 4 | 28 | `ADD[(3,4)->(0,0):dep#5]` | well-formedness | `ADD[(3,4)->(0,0):dep#5]` | same (first reply was the oracle's own) | no |
| `R32` | 19 | 12 | `ADD[(3,0)->(3,0):4tworowsdown4col#4]` | well-formedness | `ADD[3,0:4]` | put right | yes |
| `R32` | 20 | 30 | `ADD[(1,5)->(2,1):dep#3]` | identification | `ADD[(1,5)->(2,1):dep#3]` | same | no |
| `R64` | 3 | 16 | `ADD[(4,4)->(4,4):value#6]` | well-formedness | `ADD[(4,4)->(4,4):value#6]` | same | no |
| `R64` | 3 | 28 | `DEL[4,4]` | match | `DEL[4,4]` | same (first reply was the oracle's own) | yes |
| `R64` | 4 | 18 | `ADD[1,0:6]` | identification | `DEL[1,0]` | admitted, wrong | no |
| `R64` | 4 | 19 | `ADD[(1,2)->(1,0):dep#7]` | well-formedness | `ADD[(1,2)->(1,0):dep#7]` | same (first reply was the oracle's own) | no |
| `R64` | 4 | 24 | `ADD[(1,0)->(4,5):dep#7]` | well-formedness | `ADD[(1,0)->(4,5):dep#7]` | same (first reply was the oracle's own) | no |
| `R64` | 4 | 28 | `ADD[(3,4)->(0,0):dep#5]` | well-formedness | `ADD[(3,4)->(0,0):dep#5]` | same (first reply was the oracle's own) | no |
| `R64` | 19 | 12 | `ADD[(3,0)->(3,0):4tworowsdown4col#4]` | well-formedness | `ADD[3,0:4]` | put right | yes |
| `R64` | 20 | 30 | `ADD[(1,5)->(2,1):dep#3]` | identification | `ADD[(1,5)->(2,1):dep#3]` | same | no |
| `H1` | 101 | 4 | `ADD[(3,4)->(3,4):2alue#2]` | well-formedness | `ADD[(3,4)->(3,4):2alue#2]` | same | no |
| `H1` | 101 | 7 | `ADD[(3,2)->(3,4):dep#2]` | well-formedness | `DEL[(3,2)->(3,4):dep#2]` | refused again (first reply was the oracle's own) | no |
| `H1` | 101 | 13 | `DEL[(1,1)->(3,4):dep#5] ADD[(1,1)->(3,4):dep#5]` | match | `DEL[(1,1)->(3,4):dep#5] ADD[(1,1)->(3,4):dep#5]` | same | no |
| `H1` | 101 | 27 | `ADD[(3,4)->(1,3):dep#6]` | well-formedness | `ADD[(3,4)->(1,3):dep#6]` | same (first reply was the oracle's own) | no |
| `H1` | 122 | 22 | `ADD[(0,0)->(2,0):dep#3]` | well-formedness | `DEL[(0,0)->(2,0):dep#3]` | refused again | no |

## The eight first slips
Over the 128 jobs (jobs 0 to 63 in `R64`, jobs 64 to 127 in `H1`), **8 jobs ever slip**, and the misses of such a job after its first slip happen in a state that is no longer the oracle's (the exact-state arm keeps the state it holds, so a step that was not done stays not done); they are read here as the consequence of the first slip and not as new slips. Job 101 lost 27 of its 30 steps from one slip at step 4. The first slip of each job, with the request, what the oracle writes and what the model wrote:

| job | step | the request | the oracle writes | the model's first reply | kind | the gate | the retry |
|---|---|---|---|---|---|---|---|
| 3 | 16 | Add a service at (4,4) with value 6. | `ADD[4,4:6]` | `ADD[(4,4)->(4,4):value#6]` | shape | well-formedness, refused | the same reply again |
| 4 | 18 | Add a service at (0,0) with value 3. | `ADD[0,0:3]` | `ADD[1,0:6]` | number | identification, refused | answered with `DEL[1,0]`, admitted and wrong |
| 17 | 20 | Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4. | `ADD[(0,4)->(2,3):dep#4]` | `DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]` | extra delete | admitted | not asked: the gate admitted the reply |
| 19 | 12 | Add a service at (3,0) with value 4. | `ADD[3,0:4]` | `ADD[(3,0)->(3,0):4tworowsdown4col#4]` | shape | well-formedness, refused | put right |
| 20 | 30 | Add a dependency edge from (1,1) to (2,1) with relation dep and weight 3. | `ADD[(1,1)->(2,1):dep#3]` | `ADD[(1,5)->(2,1):dep#3]` | number | identification, refused | the same reply again |
| 62 | 12 | Add a dependency edge from (1,1) to (1,0) with relation dep and weight 9. | `ADD[(1,1)->(1,0):dep#9]` | `DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]` | extra delete | admitted | not asked: the gate admitted the reply |
| 101 | 4 | Add a service at (3,4) with value 2. | `ADD[3,4:2]` | `ADD[(3,4)->(3,4):2alue#2]` | shape | well-formedness, refused | the same reply again |
| 122 | 22 | Add a dependency edge from (0,0) to (2,2) with relation dep and weight 3. | `ADD[(0,0)->(2,2):dep#3]` | `ADD[(0,0)->(2,0):dep#3]` | number | well-formedness, refused | answered with `DEL[(0,0)->(2,0):dep#3]`, refused again |

They are three kinds. **A service added as an edge from the service to itself** (jobs 3, 19, 101): the request asks for a service, and the model writes the edge form from that service to itself with a made-up name (`value`, `4tworowsdown4col`, `2alue`) and the value after the `#`; the oracle writes `ADD[x,y:v]`. **A wrong number** (jobs 4, 20, 122): the model copies a coordinate or a value wrongly (in job 4 it wrote the node that the state already holds). **A legal extra** (jobs 17, 62): the model adds the right edge and also deletes another edge between the same two services; the reply is legal, so the gate admits it and the rule never fires. Of the six first slips that the gate refused (jobs 3, 4, 19, 20, 101, 122), the retry put right one (job 19). It could not reach the two that the gate admitted. **What the framework's other checks see of these eight:** its scope filter (`scope.py`: a change is in scope if the coordinates it touches are written in the request) flags the wrong coordinate in jobs 4, 20, 122 and nothing in the other five (in the shape slips both ends of the edge are the named service, and in the two legal extras the deleted edge is between two named services). The serving loop did not use the scope filter, and the feedback of the retry did not carry its reason; this is what it would have seen, not a claim about what it would have done.

Where the steps were lost (steps after which the held state was not the oracle's):

| cell | job | steps not exact | first step not exact |
|---|---|---|---|
| `R64` | 3 | 12 | 16 |
| `R64` | 4 | 13 | 18 |
| `R64` | 17 | 4 | 20 |
| `R64` | 20 | 1 | 30 |
| `R64` | 62 | 1 | 12 |
| `H1` | 101 | 27 | 4 |
| `H1` | 122 | 9 | 22 |

| K | exact steps, third run | with the retry | change | jobs that missed, third run | with the retry |
|---|---|---|---|---|---|
| 8 | 212 (88.3 %) | 212 (88.3 %) | +0 | 3 (15), 4 (13) | 3 (15), 4 (13) |
| 32 | 908 (94.6 %) | 927 (96.6 %) | +19 | 3 (15), 4 (13), 17 (4), 19 (19), 20 (1) | 3 (15), 4 (13), 17 (4), 20 (1) |
| 64 | 1867 (97.2 %) | 1889 (98.4 %) | +22 | 3 (15), 4 (13), 17 (4), 19 (19), 20 (1), 62 (1) | 3 (12), 4 (13), 17 (4), 20 (1), 62 (1) |

At 64 jobs the retry cut the lost steps from 53 to 31: job 19 from 19 to 0, job 3 from 15 to 12, the others unchanged. The pair lost 36 steps in both cells, in the same two jobs (101 and 122), and the 64 of its 64 jobs hold the right state after the same number of steps with and without the retry.

## Every cell

| cell | jobs | retry | steps | seconds | s / step | exact steps | exact | jobs exact to the end | exact steps per hour |
|---|---|---|---|---|---|---|---|---|---|
| `R8` | 0 to 7 | on | 240 | 38.5 | 0.160 | 212 | 88.3 % | 6 of 8 | 19,846 |
| `R32` | 0 to 31 | on | 960 | 121.2 | 0.126 | 927 | 96.6 % | 28 of 32 | 27,537 |
| `R64` | 0 to 63 | on | 1,920 | 223.3 | 0.116 | 1,889 | 98.4 % | 59 of 64 | 30,459 |
| `H0` | 64 to 127 | off | 1,920 | 216.3 | 0.113 | 1,884 | 98.1 % | 62 of 64 | 31,359 |
| `H1` | 64 to 127 | on | 1,920 | 219.1 | 0.114 | 1,884 | 98.1 % | 62 of 64 | 30,951 |

The third run's cells were in another Colab session and the page says a cell of this run is not compared with them for time; the seconds above are for reading each cell and the pair `H0`/`H1`, which ran in one session.

## Where my expectation was off, and where it was right
* **Lines 1 to 5 (98, 92, 85, 93 and 90 %): the run is the run the page describes.** 6,989 requests, every one with its cached count; 6,989 replies, every one accepted by the strict parser, none at 96 new tokens (the longest has 31); the first replies are the third run's up to each job's first retried step, and `H1`'s are `H0`'s; the 29 second requests were all made where the rule says and `H0` made none; the retry asked on at most 2.9 % of the steps of a cell (`R8`, just under the limit of 3 %). On the replies of `H0` the rule would fire at exactly the 5 steps where `H1` asked.
* **Line 6 (85 %) reads NO,** at 8 jobs only (equal), and the model did not use the feedback in the way the page hoped: of the 29 second requests 21 came back as the same reply, and 2 were put right, both the same step of the same job. The page put line 6 at 85 %, so this is the surprise of the run; it also said that it did not know whether this 7B model would use the feedback.
* **Line 7 (75 %) reads yes and line 8 (35 %) NO:** 98.4 % at 64 jobs against a ceiling of 99.7 %; the 31 steps still lost are jobs 3, 4, 17, 20 and 62.
* **Line 9 (55 %) and line 11 (40 %) read NO.** The page expected about six slips in 64 new jobs, four of them of the refused kind, and a retry that fixes two thirds of them. There were two slips (jobs 101 and 122), both of the refused kind, the retry was asked 5 times for them and put none right. With nothing gained, the cost of the second rounds is all there is: -1.3 % in exact steps per hour.
* **Line 10 (88 %) reads yes,** 1.013 times; **line 12 (70 %) reads yes,** with no job worse (and none better).
* **The side predictions, read afterwards as descriptions.** (a) the number of second requests in `R64` is between 4 and 12 (85 %): true, 8. (b) the second reply of job 19, step 12 is `ADD[3,0:4]` (55 %): true. (c) every second reply parses (97 %): true, 29 of 29. (d) at least 60 of the 64 jobs of `R64` hold the right state at every step (60 %): **false,** 59 (the third run: 58). (e) a second round takes 0.4 to 2.0 seconds on average (80 %): true, 0.46 over the 26 second rounds (0.40 to 0.59 by cell). (f) `H0` is between 90 and 99 percent exact (80 %): true, 98.1 %. (g) the five cells take 12 to 25 minutes in all, loading excluded (70 %): true by the saved seconds (13.6 minutes around the engine's calls); the time the cells took in Colab, with the prompts built and the files written, is not in the file.

## On the text pasted after the run
On 20 Oct the owner pasted a text (author unknown) that reads this run's file. Its sentences, quoted, checked against the files:
* "pushing you up to 97.2%", and the run before it "hovering around 69%–78%": **the retry did not do that.** 97.2 % is the exact share of `S` at 64 jobs in the second and in the third run, which had no retry. The first run (no grammar) had 76.2 %, 78.2 % and 69.2 % at 8, 32 and 64 jobs; the grammar of the second run took it to 88.3 %, 94.6 % and 97.2 %. With the retry it is 88.3 %, 96.6 % and 98.4 %, and on the 64 new jobs 98.1 % with it and 98.1 % without.
* "5.4 times more correct, usable state transitions per hour": **that divides two different sizes:** 30,914 (`S` at 64 jobs) over 5,694 (`KC` at 8 jobs) is 5.43. The ratios at the same number of jobs are the third run's: 3.40 at 8 jobs (one session) and 37.7 and 46.3 at 32 and 64 jobs (the kept chat from the second run, another session), each with the exactness of both arms beside it. Neither a "540% increase in hardware capacity", nor "pack 8× more agents onto a single server", nor dollars were measured.
* Its section on "Lines 12 and 13": those are lines of the third page (13 lines). This page has 12 lines, and the NO are 6, 8, 9 and 11.
* "Seed 20261017000, Step 16": seed 20261017000 is job 0, which is exact at every step in every cell. The step described is job 3 (seed 20261017003), step 16.
* "the ultimate, empirical proof that the AI industry must adopt" a tensor-level architecture: **nothing in this run can show that.** A 7B model that repeats a reply after a one-line feedback says that this feedback did not help this model on these slips; a different wording, a stronger model or a check of the reply against the request are not ruled out by it, and the tensor-level roadmap is neither tested nor touched.

## What this finds, in plain words
No step of the 6,960 left the held graph malformed (0 corrupted), and the exactness is 98.4 % at 64 jobs and 98.1 % on jobs nobody had read. What is left is a small number of first slips in a few jobs (8 of 128), of three kinds. **One retry with the framework's one-line feedback put right one of the six that the gate refused (job 19) and none of the other cases,** and it cost 1.4 % of the seconds of its cell in the pair. Most of the second requests were not about a slip at all: 17 of the 29 came after the job had slipped, and 14 of those were asked about a reply that was the oracle's own, which the rule cannot tell from a slip. On this model, on these slips, that feedback is not enough to move the headline. It does not say what a different feedback, a stronger model or more than one retry would do, and it says nothing against the gate, the repair or the loop as designed.

## What this does not show
More than one retry (a budget sweep is not run); another feedback wording (this was one, `structured_imperative`; the framework has three others); another model, GPU, engine or cache size; dollars; a capacity in jobs per GPU; energy; the kept chat with a retry (not run); retries in a real server, where they would be mixed with other traffic; the scope filter's reason carried in the feedback (the filter flags 3 of the 8 first slips, above; it was not used here); that the retry fixes slips of kinds not seen here; replies in ordinary words; jobs longer than 30 steps. The third run's ratios (37.7 and 46.3) were not re-measured here. It says nothing about the tensor-level roadmap.

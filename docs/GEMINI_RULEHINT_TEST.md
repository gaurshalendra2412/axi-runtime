# Test 34: does the repair rule have to be coded? The rule stated in the prompt, with and without the repair, on a second seed (Gemini 3.8 Flash; written BEFORE the run, 2 Nov 2026)

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
The article "98% Right, 21% Correct" (14 Oct 2026) ran 4,000 requests on Gemini 3.8 Flash: the model slipped 46 times, always the same way (a service was deleted and its other connections were left behind), and the framework's repair, which had been given the rule "a deleted service takes its connections with it", finished the job every time. The article says plainly that the rule was given, not discovered, and that it did not test whether stating the rule in the prompt would remove the slips. A reader asked the natural question: how many real slips have a rule that tidy?

This test does not answer that in general (that needs other jobs and is a later test). It answers three smaller questions on the same job, with the same model, the same re-asks and the same repair threshold:

1. **Does the first result repeat on a second seed?** The article's run used seed 20261031; the second registered seed, 20261032, has never been run.
2. **If the rule is STATED in the prompt and nothing repairs, does the model stop slipping?** (arm `hint_gate`)
3. **If the rule is stated and the repair is on, is the repair still needed?** (arm `hint_repair`)

A related, earlier result on Qwen2.5-7B (a rule in the prompt with the scope filter: 354 of 360 steps exact against 263 of 360 for repair alone) is a different model on a smaller job; it suggests an answer and does not give one for Gemini.

## What is run
One model, **`gemini-3.8-flash`**, the model's own default thinking, the hard request stream of the 4,000-request run (`million_hard.HardStream`: up to 24 services and 48 connections; requests include `delete_hub`, a service with 4 to 8 dependants), seed **20261032**, **400 requests per arm**, up to two re-asks when the strict parser refuses a reply, the repair deleting up to 8 dependants. The oracle's state is compared with the held state after every step. Five arms:

| arm | policy | rule stated in the prompt | what it is |
|---|---|---|---|
| `repair` | `scope_m7` | no | the framework as in the article: the state is held outside the model, the change is checked, the scope filter and the repair finish what the rules can finish |
| `hint_gate` | `gate` | **yes** | the rule is stated, and the only protection is the check that refuses a bad reply; nothing repairs |
| `hint_repair` | `scope_m7` | **yes** | the rule is stated and the repair is on |
| `blind` | `blind` | no | contrast: every reply is applied as written |
| `gate` | `gate` | no | contrast: the check only, no repair |

The rule that is stated is the one the code already holds (`agent_loop.RULES_HINT`): "a service can only be removed in the same delta that removes every edge attached to it", with one example. The code is `python/axi/experiments/gemini_rulehint.py`; it imports `million.py`, `million_hard.py`, `agent_loop.py` and `cascade.py` and **edits none of them**; none of the framework's core is changed. The run is `million.run_million`, called as the 4,000-request run called it.

## What is measured
For every step: the held state against the oracle's (exact, distance in items, corrupted), whether the reply was refused, re-asks, tokens (prompt, reply, thinking) and seconds. For every arm, from the saved replies alone: **the model's own slips**, counted as the steps that start on the oracle's state and where the reply, applied as written, gives a wrong state, whatever the policy then made of it. Because a step that starts off the oracle's state is not counted (the model was shown a different state), the slips are compared as a **share of the steps counted**. Cost in dollars of one exact step, with every call counted, at the two prices typed into the notebook from the price page (they are written into the saved file).

## What I read before writing this page
The article and its saved run (`docs/MILLION_TOKEN_TEST_HARD.md` and its results): on seed 20261031, 400 of 400 steps exact with the repair, 85 of 400 as written, 84 of 400 with the check only; the check-only arm ended 5.5 items off the oracle's state over the last 100 steps against 1.51 as written; one right step cost $0.0012 with the framework and $0.0052 as written. And Test 23 (400 turns of ambiguous text): the framework's list was exact on all 400 steps and no reply needed a repair, so on that job the fresh exact prompt alone was enough.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 3 and 6 check that the run is the run I describe; the others are about what was found.

1. [guard] all five arms reached step 400 with every step on file under the registered settings (second seed), and the replay of the saved replies reproduces every saved flag of every step of every arm (85 %)
2. [guard] arm repair (scope_m7, no hint) was never corrupted (no edge pointing at a missing service) at any step (90 %)
3. [instrument] arm blind (no check, no hint) was corrupted on at least 40 percent of its 400 steps (60 %)
4. [claim] arm repair: at least 95 percent of the 400 steps hold exactly the oracle's state (70 %)
5. [claim] arm repair: the distance to the oracle never exceeded 4 items at any step (80 %)
6. [instrument] arm repair: the model slipped at least 10 times in the 400 steps, so that the repair had work to do (85 %)
7. [claim] arm gate (check only, no hint) ends farther from the oracle than arm blind: its mean distance over the last 100 steps is larger than blind's (55 %)
8. [claim] stating the rule: the share of steps on which the model slipped is, in arm hint_gate, at most half of what it is in arm repair (50 %)
9. [claim] arm hint_gate (rule stated, check only, no repair): at least 90 percent of the 400 steps hold exactly the oracle's state (40 %)
10. [claim] arm hint_gate: the mean distance to the oracle over the last 100 steps is at most 0.5 (40 %)
11. [claim] arm hint_repair (rule stated and repair on): at least 99 percent of the 400 steps hold exactly the oracle's state (75 %)
12. [claim] the share of steps on which the model slipped is, in arm hint_repair, no larger than in arm repair (55 %)
13. [claim] cost of one exact step (dollars at the prices typed in the notebook, every call counted): arm repair costs at most 0.4 times what arm blind costs (65 %)

## Reading rules (fixed before the run)
1. all five arms reached step 400 with every step on file under the registered settings (second seed), and the replay of the saved replies reproduces every saved flag of every step of every arm
2. arm repair (scope_m7, no hint) was never corrupted (no edge pointing at a missing service) at any step
3. arm blind (no check, no hint) was corrupted on at least 40 percent of its 400 steps
4. arm repair: at least 95 percent of the 400 steps hold exactly the oracle's state
5. arm repair: the distance to the oracle never exceeded 4 items at any step
6. arm repair: the model slipped (a reply that, applied as written to the oracle's state, gives a wrong state) at least 10 times in the 400 steps, so that the repair had work to do
7. arm gate (check only, no hint) ends farther from the oracle than arm blind: its mean distance over the last 100 steps is larger than blind's
8. stating the rule: the share of steps (counted from the oracle's state) on which the model slipped is, in arm hint_gate, at most half of what it is in arm repair
9. arm hint_gate (rule stated, check only, no repair): at least 90 percent of the 400 steps hold exactly the oracle's state
10. arm hint_gate: the mean distance to the oracle over the last 100 steps is at most 0.5
11. arm hint_repair (rule stated and repair on): at least 99 percent of the 400 steps hold exactly the oracle's state
12. the share of steps on which the model slipped is, in arm hint_repair, no larger than in arm repair
13. cost of one exact step (dollars at the prices typed in the notebook, every call counted): arm repair costs at most 0.4 times what arm blind costs

Headline: lines 1, 2, 4 and 5 all read yes ("THE FIRST RESULT REPEATS ON A SECOND SEED"); otherwise the headline is "DOES NOT REPEAT" and names the line that failed. Lines 6 to 13 are reported with it and are not part of the headline. If a guard or instrument line (1, 2, 3, 6) does not read yes, no figure is claimed until the reason is found.

## What this test does not show
- **How many real-world slips have a rule that tidy.** This is one job, with one rule. Other jobs (settings, schedules, files) need their own stream and are a later test.
- **That the rule would be discovered.** Here the rule is given, in the prompt or in the code. A framework that learns a rule from its own refusals is not tested.
- **Many runs.** One seed, one run of each arm; the model's replies are sampled, so a repeat would give different numbers. The arms share the model's replies wherever their states happened to match, so they are not five independent samples.
- **Any other model, provider or length.** 400 requests of one stream on one model.
- **The price.** The prices are typed from the price page at the time of the run; they change, and the cost lines are arithmetic from the saved files, not a quote.
- **Cost in the contrast arms beyond the first 400 requests.** `blind` and `gate` are run for 400 steps only.

## Cost, and what it needs
Five arms of 400 requests each, on the same stream as the article. Many calls repeat each other (the same state, the same request, the same prompt) and are answered from the saved reply file at no cost. My estimate is **about 2 to 4 US dollars** in all; the notebook asks for a **budget** and refuses the next call once the saved answers have cost it. A paid key is needed; the notebook never turns billing on. No GPU.

## How the figures are read
`python/tests/test_gemini_rulehint.py` checks this page against its hash (`docs/results/gemini_rulehint_hash.txt`), tests the arithmetic with made-up numbers, and, once the result file exists, recomputes every figure of the Result from `docs/results/gemini_rulehint.json`.

## Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_rulehint.py`). The figures are recomputed from `docs/results/gemini_rulehint.json` by that test. Run on 2 to 3 Nov 2026 in a Colab notebook, Gemini 3.8 Flash, the model's own default thinking, seed 20261032, 400 requests per arm, prices typed from the AI Studio price page ($0.75 per million input tokens and $3.75 per million output tokens, thinking included, valid through 31 Dec 2026).*

**What the page expected, and what came out.** Of the 13 lines **12 read yes and 1 reads NO** (line 6). Headline: **THE FIRST RESULT REPEATS ON A SECOND SEED** (lines 1, 2, 4 and 5 all read yes).

**In one paragraph, for a company that runs the model.** On a seed that had never been run, the framework held the exact state on **400 of 400 steps** and was never corrupted, as on the first seed, while applying every reply as written left the state exact on only **41 of 400** and corrupted on 246. The check alone (no repair) was no better than applying everything: 41 of 400 exact, and farther from the oracle at the end (7.94 items off over the last 100 steps against 3.95). The new finding: when the rule ("a service is removed together with every edge attached to it") was **stated in the prompt**, the model did not slip at all on this stream (0 of 400, against 7 slips without the statement), so the check alone, with no repair, held the exact state on 400 of 400 steps. One exact step cost **$0.0012** with the framework as in the article, **$0.0011** with the rule stated, and **$0.0115** with every reply applied as written (9.3 times as much).

| arm | policy | rule stated | steps exact | corrupted | model slips (of steps counted) | last 100 steps: items off | dollars | dollars per exact step |
|---|---|---|---|---|---|---|---|---|
| repair | scope_m7 | no | 400 of 400 | 0 | 7 of 400 | 0.00 | 0.50 | 0.0012 |
| hint_gate | gate | yes | 400 of 400 | 0 | 0 of 400 | 0.00 | 0.44 | 0.0011 |
| hint_repair | scope_m7 | yes | 400 of 400 | 0 | 0 of 400 | 0.00 | 0.44 | 0.0011 |
| blind | blind | no | 41 of 400 | 246 | 2 of 42 | 3.95 | 0.47 | 0.0115 |
| gate | gate | no | 41 of 400 | 0 | 2 of 42 | 7.94 | 0.53 | 0.0130 |

The dollars are the five runs together at the prices above, every call counted; $2.38 in all. No reply was refused by the parser and no step needed a re-ask in any arm.

| line | my probability | what was found | |
|---|---|---|---|
| 1 | 85 % | all five arms reached step 400 under the registered settings and the replay reproduces every flag | **yes** |
| 2 | 90 % | arm repair: corrupted at 0 steps | **yes** |
| 3 | 60 % | arm blind corrupted on 246 of 400 steps (61.5 %) | **yes** |
| 4 | 70 % | arm repair exact on 400 of 400 steps | **yes** |
| 5 | 80 % | arm repair: largest distance 0 | **yes** |
| 6 | 85 % | arm repair: the model slipped **7** times (at least 10 expected), all of them on `delete_pair` (7 of 9 such requests) | **NO** |
| 7 | 55 % | arm gate ends 7.94 items off over the last 100 steps; arm blind 3.95 | **yes** |
| 8 | 50 % | slip share in hint_gate 0 of 400 (0 %), in repair 7 of 400 (1.75 %) | **yes** |
| 9 | 40 % | arm hint_gate exact on 400 of 400 steps | **yes** |
| 10 | 40 % | arm hint_gate: last 100 steps 0.00 items off | **yes** |
| 11 | 75 % | arm hint_repair exact on 400 of 400 steps | **yes** |
| 12 | 55 % | slip share in hint_repair 0 of 400, in repair 7 of 400 | **yes** |
| 13 | 65 % | one exact step: repair $0.0012, blind $0.0115 (0.11 times) | **yes** |

**Line 6, the one NO, and what it means.** The page expected the model to slip at least 10 times so that the repair had real work. On this seed the model slipped 7 times in 400 requests, all of the same kind as in the article (a delete that also names one edge, `delete_pair`: 7 of 9 such requests). The repair finished every one of them. The guard was set too high; the figure is a finding about this stream, not a fault of the run, and it does not change the headline. It also means the comparison of slip shares below rests on few events.

**What each question got.**
* *Does the first result repeat on a second seed?* Yes. 400 of 400 exact with the repair, never corrupted, largest distance 0, and the same kind of slip as in the article, finished by the same repair. The contrast arms behave as before: applying the replies as written was exact on 41 of 400 steps, and the check alone ended farther from the oracle than applying everything (7.94 against 3.95 items over the last 100 steps; in the article 5.5 against 1.51).
* *If the rule is stated in the prompt and nothing repairs, does the model stop slipping?* On this job, yes: 0 slips of 400 steps, against 7 without the statement, and the check alone held the exact state on every step. On the 9 `delete_pair` requests, the model's own replies were right on 9 of 9 with the rule stated and on 2 of 9 without it (7 slips). This is a small number of events (9 requests of that kind on this seed).
* *If the rule is stated and the repair is on, is the repair still needed?* On this seed, the repair had nothing to do: with the rule stated there was no slip for it to repair. Every one of the 400 calls of this arm was the same as a call of the arm `hint_gate` (the same state and the same request give the same prompt), so it was answered from the saved reply file and it is **not an independent sample**: it shows that the repair did no harm and had no work, not that it would never be needed.

**What this test shows, and does not show.**
* **Said:** on this job and this model, the rule can be given either in the code (the repair) or in the prompt (a sentence), and either way the held state was exact on all 400 steps. Holding the state outside the model, with the check, kept the state exact in the arms where the rule was stated or repaired; applying replies as written, or only refusing, did not.
* **Said:** the check alone is not enough when the rule is not available: it left the state farther from the oracle than no check at all (7.94 against 3.95 items), as in the article. Something has to finish the job, whether the model (told the rule) or the repair (given the rule).
* **Not said:** that a rule can always be stated, or that real slips have a rule that tidy. This is one job with one rule, and how many real slips have such a rule is still the open question; it needs other jobs and is a later test.
* **Not said:** that a model discovers a rule by itself, or that stating a rule removes slips on other jobs or other models. On Qwen2.5-7B, stating the rule helped (354 of 360 steps exact with the rule and the scope filter against 263 of 360 with repair alone); here, on Gemini 3.8 Flash, it removed the slips on this stream.
* **Not said:** anything about long runs of the stated-rule arms. Each arm is 400 requests of one seed; the 4,000-request run was made without the statement.
* **One seed, one run of each arm.** The model's replies are sampled and a repeat would give different numbers. The two contrast arms share calls with the arm `repair` wherever their states agreed (the first 42 steps), so the five arms are not five independent samples.
* **The cost lines** are arithmetic on the saved files at the prices typed; the 9.3 times is for this seed, where applying replies as written ended exact on 41 steps (the article's 4.5 times was for the first seed, where it ended exact on 85).

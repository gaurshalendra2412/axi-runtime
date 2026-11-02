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

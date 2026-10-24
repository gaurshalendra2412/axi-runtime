# Test 26: Rajnish Choubey's witness quorum on Gemini 3.8 Flash

*Concept-map item 123. Harness: `python/axi/experiments/gemini_quorum.py`. Notebook: `python/colab/axi_gemini_quorum.ipynb`. Instrument checked on a stand-in model by test 149 (`python/tests/test_gemini_quorum.py`). This page, its predictions and its reading rules were written before any call; the text above the Result is hash-locked.*

## What is tested

Concept-map item 122 built the witness quorum from Rajnish Choubey's article "The 99 Eyewitnesses of the Ramayana": several witnesses propose the change for one request, and a change is committed only when enough of them proposed exactly the same change. It aims at the one blind spot of the check: a change that is allowed but is not the one asked for (legal but wrong). Built and checked offline, it has not met a model. This test puts it on one.

**Why the fuzzy-state job with minimal thinking.** With its default thinking, Gemini 3.8 Flash held Test 23's list exactly on 400 of 400 steps with AXI: a quorum would have nothing to fix there. With thinking set to `minimal`, one reply should slip more often. That also makes the company question real: can a few witnesses that do not think replace one call that does?

## The arms

| Arm | What it is |
|---|---|
| `S`, one witness | Exactly arm A of Test 23 (`gemini_fuzzy.run_axi`): the exact list shown at every step with the same chatter, the check before the list is touched, up to 2 re-asks with the reason; a reply that still fails changes nothing. Thinking `minimal`. |
| `Q`, witnesses | The same prompt, each witness told its number so that each is its own call. Witnesses are heard one after another. After each from the second on, a change (or an answer) that exactly 2 witnesses proposed, with more votes than any other, is committed and no further witness is heard. At most 5 witnesses; if no change has that lead after 5, nothing changes. A reply the check refuses is set aside, with no re-ask. Two proposals are the same change when they leave the same list. Thinking `minimal`. |
| Reference, no call | Test 23's saved arm A (default thinking: 111.5 thinking tokens a call, 1,322 computed tokens per exact step, 2.76 s a step, 400 of 400 exact) and arm K, from `docs/results/axi_gemini_fuzzy_seed20261122_default.json`. |

## Settings

* Model `gemini-3.8-flash`, streamed; temperature the model's own default in both arms.
* Thinking `minimal` in both arms. If the service refuses `minimal`, the pilot shows it and both arms use `low`; the setting used is in each run's file and in the Result.
* The same stream and chatter as Test 23: seed 20261122, 400 steps, 30 sentences of office chatter around every message, a question every tenth step from the 20th.
* Arm Q: 2 witnesses in agreement with a strict lead, at most 5 witnesses a step.
* Guards: a cap on tokens sent of 1,500,000 for arm S and 4,000,000 for arm Q, and a dollar budget the owner types after the pilot (the notebook refuses the next call once the saved answers cost it). The notebook has no built-in price; the owner types the prices.
* Pilot: seed 98, 80 steps of each arm, the first 60 answered by the script so the list has its full size.

Measured for both arms: whether the held list is exact at every step, wrong changes and answers that passed the check, the witnesses heard, the tokens a call (prompt, visible, thinking), the computed tokens per exact step (the stand-in for energy: the API gives no joules), the seconds a step.

## Predictions

1. [guard] arm S reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask (90 %)
2. [guard] arm Q reached step 400 with every witness reply on file, and the replay reproduces every vote, every stop and every flag (85 %)
3. [guard] the minimal thinking took effect: the thinking tokens a call of arm S are at most a quarter of those of Test 23's arm A (111.5) (70 %)
4. [instrument] at least 99 percent of all replies of arms S and Q (every re-ask and every witness) are one JSON object of the right shape (92 %)
5. [claim] minimal thinking costs exactness: arm S held the list exactly on fewer than 400 of the 400 steps (55 %)
6. [claim] arm Q held the list exactly on at least as many steps as arm S (75 %)
7. [claim] arm Q held the list exactly on more steps than arm S (40 %)
8. [claim] arm Q held the list exactly on 400 of the 400 steps (35 %)
9. [claim] arm Q committed at most half as many legal-but-wrong changes and answers as arm S (when arm S committed none, arm Q must commit none) (50 %)
10. [claim] arm Q heard at most 2.3 witnesses a step on average (70 %)
11. [claim] on the steps where the first two witnesses did not agree, the committed change was exact on at least 80 percent (not read with fewer than 5 such steps) (40 %)
12. [claim] arm Q's computed tokens per exact step are at most 2.5 times arm S's (75 %)
13. [claim] arm Q's computed tokens per exact step are below those of Test 23's arm A with default thinking (1,322) (40 %)
14. [claim] arm Q's seconds a step are at most those of Test 23's arm A with default thinking (2.76 s) (45 %)
15. [claim] the first two witnesses of arm Q proposed the same change on at least 95 percent of the steps (60 %)
16. [claim] arm Q answered at least as many of the questions exactly as arm S (75 %)

## Reading rules

Each line reads yes or NO from the saved files, as written here; a line whose run or steps are missing is not read.

1. arm S reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask
2. arm Q reached step 400 with every witness reply on file, and the replay reproduces every vote, every stop and every flag
3. the minimal thinking took effect: the thinking tokens a call of arm S are at most a quarter of those of Test 23's arm A (111.5)
4. at least 99 percent of all replies of arms S and Q (every re-ask and every witness) are one JSON object of the right shape
5. minimal thinking costs exactness: arm S held the list exactly on fewer than 400 of the 400 steps
6. arm Q held the list exactly on at least as many steps as arm S
7. arm Q held the list exactly on more steps than arm S
8. arm Q held the list exactly on 400 of the 400 steps
9. arm Q committed at most half as many legal-but-wrong changes and answers as arm S (when arm S committed none, arm Q must commit none)
10. arm Q heard at most 2.3 witnesses a step on average
11. on the steps where the first two witnesses did not agree, the committed change was exact on at least 80 percent (not read with fewer than 5 such steps)
12. arm Q's computed tokens per exact step are at most 2.5 times arm S's
13. arm Q's computed tokens per exact step are below those of Test 23's arm A with default thinking (1,322)
14. arm Q's seconds a step are at most those of Test 23's arm A with default thinking (2.76 s)
15. the first two witnesses of arm Q proposed the same change on at least 95 percent of the steps
16. arm Q answered at least as many of the questions exactly as arm S

Headline: **NOT READ** unless lines 1, 2 and 4 read yes. **NOTHING TO FIX** if line 5 reads NO (one witness with minimal thinking was already exact on every step, so the quorum had nothing to show). **THE QUORUM PAYS** if lines 7 and 12 read yes. **NO GAIN** if line 7 reads NO. **MIXED** otherwise. A NO is printed as NO, and its cause is looked for in the saved rows before anything is said about it.

## What a result will not show

Anything about another model, another job or another seed; one run of each arm. Witnesses here are samples of one model, so they are not independent: the arithmetic of item 122 is a ceiling, and this run measures what is left of it. Joules: the computed tokens and the seconds are the stand-in. The cost of the quorum in a serving stack that shares a prompt between samples, which an API does not.

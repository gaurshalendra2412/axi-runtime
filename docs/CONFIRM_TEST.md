# Confirmation test: do the first hosted-model findings repeat on tasks nobody has looked at? (written BEFORE any run, 10 Oct 2026)

Everything measured so far was read on a few episodes (4 to 6 per condition) and on a few seeds (3 and 7 for the runs since the scope test), and the instrument was built while looking at them. A finding that was read on the same few cases that shaped the instrument can be an accident. This page fixes the question, the exact tasks, the method for error margins, the predictions and the reading rules **before any number exists**. Code: `python/axi/experiments/confirm_test.py`. Checks of the instrument without a model: `python/tests/test_confirm_test.py`. Notebook: `python/colab/axi_confirm_test.ipynb`. Concept-map item 90.

**What this is and what it is not.** No test proves a framework in general, and this one does not. It is the strongest test this project can run on a free key: tasks that were never used, many more episodes, an error margin on every number that matters, thresholds written down first, and the task lists fixed by a hash. If the first findings were an accident, this is where they should break.

## The question, in plain words
A model that edits a service graph one request at a time: how often does its own held state go wrong over 30 requests, and does the framework (the gate, the repair and the scope filter) hold the state exact, on tasks it has not seen, with margins that say how sure we can be?

## Design
* **Model and settings.** Gemini 3.5 Flash Lite through the free key, the model of the first hosted run (`API_TEST_GEMINI.md`), thinking and temperature at the model's defaults, at most 8,192 output tokens, one sample per distinct prompt, the same reply cleaning (one code fence and white space only). Any other model the key offers can be run with the same notebook; it is read with the same lines and reported separately, and if it is not run nothing is claimed about it.
* **Single requests.** `make_scope_tasks(280, 21)`: 280 requests, 40 of each of the seven kinds (service delete with and without dependants, edge delete, service delete together with an edge delete, delete by weight, add service, add edge), fresh seed 21, in each of two regimes (no rule in the prompt, rule stated) = 560 requests.
* **Long episodes.** 30 episodes of 30 requests each, seed 21 (episode seeds 21000 to 21029), the `scope` mix, three policies on the same saved replies: blind apply, repair_m6, scope_m7. Each policy keeps its own state, shows it to the model at every step and is compared with an oracle that applies the intended request. Two regimes. 900 steps per policy per regime.
* **The task lists are fixed.** sha256 of the 280 single requests (kind, request, correct delta): `15c5f8cddcfcb676206ddc4daa18bfef332485437dc7de5a47d6c9ed54532b6b`. sha256 of the 30 long scripts (start graph, then kind, request, correct delta of each step): `c5b30f949435c33cb378e96fb3fe4ad07e1846c8f57b565f87099bd1efed186b`. A test recomputes both from the seeds. None of these tasks, and none of these start graphs, occurs under seeds 0, 1, 2, 3 or 7 (a test checks).
* **The prompt.** The prompt of the first hosted run with **one line added** after the examples: `remove the service at (2,1), which has no edges -> DEL[2,1]` (before the rule, when the rule is stated). Reason: 12 percent of the first run's replies were lost to the slip `DEL[2,0:7]` for `DEL[2,0]`, almost all in the rule-stated regime, which hid that regime. The example says that the service has no edges so that it does not teach the model to leave edges out. This is the only change; numbers are compared with the first run as numbers.
* **Error margins.** Shares of episodes (how many episodes ever corrupt) get a 95 percent Wilson interval. Shares of steps (exact steps) and the gap between two policies get a 95 percent bound from a bootstrap that resamples **whole episodes** with replacement (2,000 resamples, generator seed 20261010, percentiles 2.5 and 97.5), because the steps inside an episode are not independent. The gap is paired by episode.
* **Wasted tokens.** For each policy: the share of its tokens (prompt tokens plus visible output tokens) spent on steps that start from a state already different from the oracle's; every step counts, whether or not its reply was shared with another policy (what the policy would spend if it ran alone). Thinking tokens are not counted.
* **Calls.** The first run needed 44 replies per 30-step episode. Expected: about 280 + 30 x 44 = 1,600 per regime, about 3,200 in all (I rate 2,600 to 3,800 at 70 %). The free key allows about 480 calls a day, so the test takes about a week; every cell stops by itself, saves every reply and carries on the next day. With billing switched on for the key (one line in the notebook) it takes one sitting; the cost should be small for a Flash-class model and the notebook prints the tokens used, but no price is quoted here.

## Predictions (mine, written first; the number is how sure I am)
Numbers refer to the reading-rule lines below.
1. [guard] No graph is corrupted by gate, repair_m6 or scope_m7 on the 280 single requests. (99 %; by construction)
2. [guard] repair_m6 and scope_m7 are never corrupted at any long step. (99 %; by construction)
3. [guard] repair_m6 holds the oracle's state on at least 95 percent of steps with the lower bound at least 90 percent, in every regime run. (60 %) No rule: 85 %. Rule stated: 65 %, because the model may still write replies that neither the gate nor the repair can turn into the intended change (the first run: 100 % with no rule).
4. [guard] The same for scope_m7. (60 %)
5. [guard] scope_m7 is within 3 points of repair_m6. (90 %) The first run: identical.
6. [need] No rule: blind apply corrupts the state in at least some episodes, shown by a lower bound of at least 30 percent of episodes. (93 %) In the first run 3 of 4 episodes were corrupted, always at a request that deletes a service and an edge; that request is one of ten in the script, so about three per episode.
7. [need] The same, rule stated. (60 %) I do not know how Flash Lite behaves with the rule stated; the first run could not say.
8. [need] No rule: repair_m6 ahead of blind apply by at least 10 points of exact steps, lower bound above 0. (92 %) The first run: 100 % against 50.8 %.
9. [need] The same, rule stated. (55 %)
10. [need] No rule: blind apply spends at least 10 percent of its tokens on steps that start from a wrong state, repair_m6 at most 2 percent. (80 %) The first run's saved file gives 46.3 % for blind apply and 0 % for repair_m6 by this definition.
11. [need] No rule: on the 280 single requests repair_m6 is ahead of blind apply by at least 5 points. (80 %) The first run: 135 against 120 of 140, 10.7 points.
12. [replicate] No rule: blind apply's exact-step share lies between 35.8 and 65.8 percent, within 15 points of the first run's 50.8 percent. (55 %) The first run had 4 episodes; the share depends on when the first delete-with-edge request comes.
13. [format] At most 5 percent of all replies do not parse. (65 %) The added example should remove the slip; the first run's 12 percent is the reason for doubt.
14. Not a reading-rule line. The mean time of one reply is between 0.3 and 3 seconds (80 %). Blind apply's exact-step share without the rule is between 30 and 70 percent (80 %). Blind apply's share of tokens from a wrong state is between 25 and 65 percent (65 %).

Overall: that every line comes out "yes" I rate about **12 %** (lines 7, 9, 12 and 13 carry most of the doubt); that every [guard] and [format] line does, about **40 %**.

## Reading rules (fixed now; the notebook prints exactly these)
Thirteen lines, tagged **[guard]**, **[need]**, **[replicate]** or **[format]**. A regime that was not run is skipped. Code: `confirm_test.verdict_confirm`.
1. [guard] gate, repair_m6 and scope_m7 corrupt no graph in the 280 single requests, in either regime
2. [guard] repair_m6 and scope_m7 are never corrupted at any long-run step, in either regime
3. [guard] repair_m6 holds the oracle's state on at least 95 percent of long-run steps, with the lower 95 percent bound at least 90 percent, in every regime run
4. [guard] scope_m7: the same, in every regime run
5. [guard] scope_m7 is not behind repair_m6 by more than 3 points of exact steps, in every regime run
6. [need] no rule: blind apply corrupts the state at some step in a share of the episodes whose lower 95 percent bound is at least 30 percent
7. [need] rule stated: blind apply corrupts the state at some step in a share of the episodes whose lower 95 percent bound is at least 30 percent
8. [need] no rule: repair_m6 is ahead of blind apply in exact steps by at least 10 points, and the lower 95 percent bound of the gap is above 0
9. [need] rule stated: repair_m6 is ahead of blind apply in exact steps by at least 10 points, and the lower 95 percent bound of the gap is above 0
10. [need] no rule: blind apply spends at least 10 percent of its tokens on steps that start from an already wrong state, repair_m6 at most 2 percent
11. [need] no rule: on the 280 single requests repair_m6 is ahead of blind apply by at least 5 points
12. [replicate] no rule: blind apply's exact-step share is within 15 points of the first run's 50.8 percent (35.8 to 65.8 percent)
13. [format] at most 5 percent of the model's replies do not parse (x of y)

**How the result will be worded.** A [format] line "no": the other lines are read first as a statement about formatting. A [guard] line "no": a finding about the framework, looked into before anything is claimed; **a single corrupted step under repair_m6 or scope_m7 (lines 1 and 2) falsifies the claim that they hold the state**. A [need] line "no": this model on these tasks does not fail where the earlier models did, reported with the numbers, and the claim that a raw model drifts is then not supported for this model. A [replicate] line "no": the first run's number did not repeat; both numbers stay in the record. A line that comes out "NO" is a finding, not an error.

## What this cannot show
One model, or two. Synthetic graph-edit requests written in an invented delta language, not real agent workloads; the 30 episodes come from one generator, so error margins say how sure we can be about this generator and nothing about whether it is representative. "Drift" here is the distance to the oracle's state as defined in `LONG_HORIZON_TEST.md`; it says nothing about long-term memory, tools or open-ended tasks. One sample per prompt, so a second run would not give the same numbers. A hosted model can change behind the same name between days; the model id is recorded, the weights are not ours. Thinking tokens are not counted. The wasted-token share is a measure on this synthetic task; **nothing here supports a statement about savings for any company**. The words test (`WORDS_TEST.md`) and the second-model Part B (`API_TEST_GEMINI.md`) are separate and unaffected.

## How it is run
A Colab notebook on a free CPU runtime (`python/colab/axi_confirm_test.ipynb`): upload `axi-colab-confirm.zip`, install the Gemini library, connect Drive, read the key from Colab Secrets, then one cell per part (single requests without the rule, with the rule; long episodes without the rule, with the rule), each saving its replies and a result file, and the last printing every reading-rule line with its error margins.

# Escalate when nothing is committed: measured, not kept

*Concept-map item 131. Code: `python/axi/experiments/escalate.py`, a measurement, **not kept** in the engine. Checked by test 157 (`python/tests/test_escalate.py`). No model is called and no new call is made: every figure below is replayed from the saved replies of the seed-7 runs, through the framework's own gate, repair and scope, unchanged.*

## Where it comes from

Rajnish Choubey's line in "How Perplexity Made Google a backend team of brick layers" (Medium, 4 May): "The only thing that scales is AI-to-AI communication + human-centric routing." The article is made of AI replies that quote him, and this line is quoted seven times. The same replies describe one layer between all the models that sends each query on, so that part of the wording may be the AI's.

Read for structure, much of this is already AXI:

* models checking each other's answers is the witness quorum (items 122 and 123);
* the host answering what it holds, with no model, is item 126.

Item 109 put one rung of the concept map's ladder into the serving loop. Where the gate refused a reply and the repair could not fix it, the same model was asked once more, with the framework's feedback.

## What is new

At the same point, ask the **next model** instead. `escalate(askers, commit, before, same)` asks the models in their order. It runs the host's own commit (parse, gate, repair) on each reply and stops at the first reply that commits a change. The trigger is something the host sees for itself: the state after the commit is the same as the state before it.

Two things are deliberately not triggers:

* **A raw refusal by the gate.** A refused reply that the repair completes is kept. Escalating those was measured below, and it lost steps.
* **A wrong reply that the gate admits.** The host cannot see it, so no router of this kind catches it.

A request whose right answer changes nothing would be passed through every model. Such requests (questions) are not routed here: the host answers them itself (item 126).

## What it gives, replayed from the saved replies

The seed-7 single-step scope test sent the same 140 prompts to three models:

* Qwen2.5-7B and Qwen2.5-14B, in 4-bit on a Colab T4, with the grammar on;
* gemini-3.5-flash-lite, the hosted service, with no grammar mask.

The test was run with the rule stated and without it. Because every model answered the same prompts, any order of models can be replayed exactly from the saved replies.

**With the rule stated, under `scope_m7` (the current best policy):**

| Who answers | Exact | Second-model calls | Seconds | Output tokens |
|---|---|---|---|---|
| flash-lite alone | 108 of 140 | 0 | 110.9 | 3,025 |
| Qwen2.5-7B alone | 134 of 140 | 0 | 317.3 | 3,975 |
| Qwen2.5-14B alone | 133 of 140 | 0 | 538.7 | 3,510 |
| flash-lite, then 7B where nothing was committed | **138 of 140** | 30 | 226.8 | 4,660 |
| flash-lite, then 14B where nothing was committed | **138 of 140** | 30 | 267.5 | 4,165 |

The second model was asked on 30 of the 140 tasks. The router was more exact than any single model, and it took less time than either Qwen model alone.

Flash-lite's 30 failures were replies the framework could not read: with no grammar mask, it wrote a node delete with a value in it. Under `repair_m6`, the same orders give 126 (then 7B) and 128 (then 14B), against 108 for flash-lite alone, 118 for 7B and 117 for 14B.

**Without the rule:** flash-lite alone was exact on 135 tasks. With a Qwen model asked on 3 of the 140 tasks, it was exact on 138, at 84.5 seconds against 79.5. The best single model was 136.

**Qwen first:** nothing is ever passed on. Under both policies and both regimes, the repair commits something for every Qwen reply. Qwen's wrong replies (6 of 140 for 7B with the rule, under `scope_m7`) are all admitted and committed. The host cannot see them, so this router does not help Qwen, and it costs Qwen nothing either.

**On the raw refusal instead (rejected):**

* 7B, then flash-lite: 34 extra calls, and 120 exact against 134 for 7B alone. The repair would have fixed those replies, and flash-lite's replies were worse.
* 14B, then flash-lite: 119 exact.
* 7B, then 14B: 135, with 34 extra calls.

So the trigger is "nothing committed", not "refused".

**The long runs, as far as they can say.** In the seed-7 long runs with the rule stated, flash-lite held only 14 of 180 steps under `scope_m7`, all of them before its first miss. In each of the 6 episodes, its first miss was a step where nothing was committed. The 7B's reply and the 14B's reply to the same prompt were each exact in 6 of 6 of those episodes. After that step the saved runs cannot say anything, because flash-lite's own state had left the oracle's, so the next prompts differ.

Under `repair_m6`, the next model was exact at that step in 5 of the 5 episodes where its own state still matched (14B), and in 3 of those 5 (7B). Without the rule, flash-lite never failed in the 4 episodes the runs share.

## Why it is not kept

Its whole gain on these runs is one slip. All 33 of flash-lite's replies that nothing could commit (30 with the rule stated, 3 without) are the same mistake: a node delete written with a value, as in `DEL[1,0:7]`, which the framework cannot read.

A reader that drops that value gives flash-lite **alone** the same 138 of 140, under both policies and in both regimes, with no second call. With Qwen first, the router never fires.

So on the saved data the router adds a second model and its calls, and gains nothing that one reading fix does not also give. That fix would change the framework's parser, which this work does not touch, so it is noted here, not built.

The router stays in `experiments/` so that the replay can be checked. Nothing is added to AXI's engine, so nothing in AXI is slowed down.

It would be worth a run only with a first model whose failures the host sees and which are not all one slip. These runs have no such model.

## What this does not show

* A live run of the router over many steps. That needs a run: the saved long runs stop being comparable one step after flash-lite's first miss.
* A price. The seconds were measured on different machines (a hosted service over the network against a T4), so they are not a cost, and the prices are the owner's to type in, as in earlier tests.
* What it gives for other models, or for a first model that holds to the grammar. With Qwen first, it gave nothing.

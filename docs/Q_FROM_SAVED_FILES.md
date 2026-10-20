# Q = O / M from the saved files (computed 20 Oct 2026, no GPU)

Concept-map item 111. Code: `python/axi/experiments/q_measure.py`; checks: `python/tests/test_q_measure.py`. Nothing here is a new run: every number is computed from a result file that is already in `docs/results/`, and the test recomputes the table below from those files and compares it, character by character, with this page.

## Where the idea comes from, and what is taken from it
The pasted summaries of Rajnish Choubey's sessions write a quality ratio Q = O / M: O is what was done *exactly*, M is what it cost. They say M can be pushed towards a floor and not to zero. That is a way of asking for **exact work per unit of cost**, and it fits the serving test, because the test already records both halves: whether the state after a step equals the oracle's (O) and the seconds and tokens the engine spent (M). Those summaries are second-hand (they are not Sir's own articles), so only the *shape* of the ratio is used here, and the choices below are mine, written down so that they can be argued with.

## Definitions (what the code computes for a cell)
* **O**, the exact steps: the number of steps after which the state the job held equals the oracle's (the `exact` flag in the log). A step after which the held state differs from the oracle's does not count.
* **M in seconds**: the wall seconds of the cell, **every round of the engine in it** (a retry round, if the cell has one, is inside it). This is the time the GPU was busy, in one session on one GPU.
* **M in tokens**: the prompt tokens the engine had to compute (prompt tokens minus the tokens served from its cache, over every request) plus the tokens it wrote. The cache counts are the engine's own, from the `rounds` of the saved cell. A cell whose rounds have no cached counts shows "-".
* **Q, two numbers, side by side**: *exact steps per GPU hour* = 3,600 x O / wall seconds, and *exact steps per 1,000 tokens* = 1,000 x O / M in tokens. They are never added or averaged into one number: that would need a weight between a second and a token (the summaries write M = time + lambda x tokens), no weight has been measured, so none is invented.

## The table
**first run, no grammar**

| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |
|---|---|---|---|---|---|---|---|
| S8 | 240 | 183 | 76.2 % | 37.0 | 17,782 | 32,072 | 5.7 |
| KC8 | 240 | 240 | 100.0 % | 148.2 | 5,831 | 49,655 | 4.8 |
| KN8 | 240 | 240 | 100.0 % | 1646.3 | 525 | 793,719 | 0.3 |
| S32 | 960 | 751 | 78.2 % | 115.9 | 23,336 | 119,883 | 6.3 |
| KC32 | 952 | 926 | 97.3 % | 4901.9 | 680 | 2,278,479 | 0.4 |
| S64 | 1,920 | 1,329 | 69.2 % | 205.2 | 23,316 | 228,866 | 5.8 |

**second run, uncapped grammar**

| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |
|---|---|---|---|---|---|---|---|
| S8 | 240 | 212 | 88.3 % | 36.6 | 20,878 | 32,003 | 6.6 |
| KC8 | 240 | 240 | 100.0 % | 145.9 | 5,923 | 49,655 | 4.8 |
| S32 | 960 | 908 | 94.6 % | 113.7 | 28,748 | 119,988 | 7.6 |
| S64 | 1,920 | 1,867 | 97.2 % | 204.3 | 32,896 | 229,011 | 8.2 |
| KC32 | 956 | 956 | 100.0 % | 4730.9 | 727 | 2,256,513 | 0.4 |
| KC64 | 1,915 | 1,915 | 100.0 % | 10330.5 | 667 | 5,174,334 | 0.4 |

**third run, capped grammar**

| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |
|---|---|---|---|---|---|---|---|
| S8 | 240 | 212 | 88.3 % | 39.4 | 19,379 | 32,003 | 6.6 |
| KC8 | 240 | 240 | 100.0 % | 151.7 | 5,694 | 49,655 | 4.8 |
| S32 | 960 | 908 | 94.6 % | 119.3 | 27,390 | 119,913 | 7.6 |
| S64 | 1,920 | 1,867 | 97.2 % | 217.4 | 30,914 | 228,935 | 8.2 |

**fourth run, one retry**

| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |
|---|---|---|---|---|---|---|---|
| R8 | 240 | 212 | 88.3 % | 38.5 | 19,846 | 32,302 | 6.6 |
| R32 | 960 | 927 | 96.6 % | 121.2 | 27,537 | 121,040 | 7.7 |
| R64 | 1,920 | 1,889 | 98.4 % | 223.3 | 30,459 | 230,110 | 8.2 |
| H0 | 1,920 | 1,884 | 98.1 % | 216.3 | 31,359 | 225,703 | 8.3 |
| H1 | 1,920 | 1,884 | 98.1 % | 219.1 | 30,951 | 226,037 | 8.3 |

All cells are arm `S` (the exact state each step, cache on) unless the name begins with `K`: `KC` keeps the whole chat with the cache on, `KN` keeps it with the cache emptied before every round (what a plain engine pays), the number is the number of jobs run side by side. Each row is one cell of one session. The first and second runs, the third run and the fourth run were separate Colab sessions: seconds are comparable inside a run, much less across runs.

## What may be read from it, and what may not
* **The sentence this table allows** is of the form: "N exact steps per GPU hour against M, at these exactness levels, on this setup", and the setup is named next to the figures. All four runs record the same one: `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0, a Tesla T4, a cache of 80,000 tokens (a block override), 30-step jobs from seed 20261017, the policy `repair_m6` (the test checks the saved files for it). The exactness is always stated with the Q, because a high Q with a low exactness is the cheap, wrong job.
* **Compare like with like.** Two cells are compared when they have the same number of jobs side by side and were run in one session. The third run holds `S` at 8, 32 and 64 jobs and `KC` at 8 only; the 64-job `S` cell therefore has no kept-chat cell at 64 jobs to stand next to in that run (the second run does, `KC64`, and its row is above). A figure that divides a 64-job `S` cell by an 8-job `KC` cell mixes two things: how many jobs ran side by side and what each of them kept.
* **What the table shows about the kept chat.** In every run, at every number of jobs in which both ran, the kept chat (`KC`) has fewer exact steps per 1,000 tokens and fewer per GPU hour than the exact-state cell (`S`): at 8 jobs 4.8 per 1,000 tokens against 5.7 to 6.6, at 32 jobs 0.4 against 6.3 and 7.6, at 64 jobs 0.4 against 8.2 (the second run). The token figure does not depend on any time measurement; the seconds also depend on the load of the GPU and on the session. The kept chat is more exact where it ran at 32 and 64 jobs (97.3 %, 100.0 % and 100.0 % against 78.2 %, 94.6 % and 97.2 %); the Q is not a replacement for the exactness, it is read beside it.
* **No floor of M is shown.** A cell never reaches zero cost: each exact step needs at least the tokens of its own reply. The table says how much M has been in the cells that were run; it does not say how low it can go. "Q grows without limit" is not a statement this page supports.
* **No capacity multiple, no count of GPUs, no dollars, no energy.** Energy was never measured; a Q per hour is not a price.
* **Not a claim about any provider**, and nothing about another model, GPU, engine or cache size.

## How it is used from here
The notebook of the fifth run (`PROVIDER_SIDE_TYPED_TEST.md`) prints this table for its own seven cells after the lines are read, and the Result of that test quotes it from the saved file, not from the screen. The same table for the files in `docs/results/` is printed by `python -m axi.experiments.q_measure` (run it from the `python` folder).

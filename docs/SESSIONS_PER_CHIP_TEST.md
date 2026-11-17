# Test 38: how many sessions one chip serves with a real model, 8 to 256 at once (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before these cells are run; the Result goes below it and nothing above the Result line is edited afterwards.*

## Why this test
Tests 29 and 30 counted how many sessions fit in the memory of a T4 and of a TPU chip. They counted memory only, with a small model's shape and no model run; they said nothing about speed. What a company that serves a model pays for is **exact steps per chip-hour** and **how long a session waits for its step**. This test puts a real 7B model on one L4 and runs 8, 32, 128 and 256 sessions at once, each a 100-step job whose state the host holds. **The question: as more sessions share one chip, does the exact share stay exact, and what happens to the steps per hour and to the wait of a single step?**

## What is run
The engine and code of Tests 21, 35 and 36, unchanged (`python/axi/experiments/provider_side_big.py`, repo commit `55631ec`), with these settings: model `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0, one NVIDIA L4, prefix caching on, greedy decoding, no retry, the grammar that names the coordinates (the `PN` cell). One setting differs from the earlier runs: the cache is **10,000 blocks (160,000 tokens)** instead of 2,000 (32,000 tokens), a setting the engine class takes, so that 256 sessions' prefixes can be held. The cells are run in the order 8, 32, 128, 256 on one loaded engine; each is one cell in which all its sessions advance one step per round, as in `PN`. The jobs are the same generator and seed as the sixth run's, jobs 192 to 192+K-1: the cell of 32 is exactly `PN` (jobs 192 to 223); the cells of 8 are its first 8 jobs; the cells of 128 and 256 use jobs 224 and beyond from the same generator, which the sixth run did not use. The driver adds the four cells to the harness's table of cells at run time and runs them with the harness's own `run_cell`; no file of the repository is changed.

## What is measured
Per cell: the share of steps whose held state equals the oracle's; the share of replies the strict parser accepts; seconds in all and exact steps per GPU hour; the time of a round (one step of every session): median, 95th percentile and the largest; computed prompt tokens per step and the share of prompt tokens served from the cache; the GPU board's mean power and joules per exact step.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error.

1. [instrument] every cell runs all its steps (K times 100) and at least 99.5 percent of replies are accepted by the strict parser (85 %)
2. [claim] the exact share is at least 99 percent in every cell (70 %)
3. [claim] exact steps per GPU hour at 256 sessions are at least twice those at 8 sessions (80 %)
4. [claim] exact steps per GPU hour at 256 sessions are at least 1.5 times those at 32 sessions (55 %)
5. [claim] the share of prompt tokens served from the cache at 256 sessions is at least 60 percent (60 %)
6. [claim] the median round time at 256 sessions is under 10 seconds, and the largest round under 30 seconds (65 %)

What would count against AXI here: an exact share below 95 percent in any cell, or exact steps per hour at 256 sessions no higher than at 32. Either is reported as it is.

## Honest limits
One card (L4), one 7B quantised model, greedy decoding, one engine and its version, synthetic service graphs, 100-step jobs. The chip is not a TPU: the TPU did not allocate in this account. Sessions here advance in lock-step rounds, so a round time is the wait of every session for its step; a real service has arrivals at random times, which is another test. The cache here is larger than in the earlier runs, so these figures are not comparable with the 32,000-token cells except for the cell of 32, where the only difference is the cache. The engine fixes how much GPU memory it takes, so memory used is not a measurement here. Jobs beyond 223 are the same generator but were not part of the earlier runs.

## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run (the plan page's sha256 before the Result was added: `3baf92d9d725f46d4012e78b29ddc18da7a2d5057e0c25636addcf077d45fd32`).*

Run: the harness of Tests 35 and 36 unchanged (`provider_side_big.py`, repo commit `55631ec`), model `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM 0.31.0, one NVIDIA L4 on Colab, cache 160,000 tokens, prefix caching on, greedy, no retry, the grammar that names the coordinates. The driver (`docs/results/axi_sessions_per_chip_driver.py`, sha256 `a1f871e636ee6256ae786597426c9c484fdc6d2b14bd272dbfed4af107cf6c4f`) adds the four cells to the harness's table at run time and runs them with the harness's own `run_cell`; load 242 s, then four cells on one loaded engine, 8, 32, 128, 256 in that order. The engine was started in the background so that a frozen browser page could not stop it. An earlier attempt of this test, on another runtime, was lost when its page froze and the runtime went with it; nothing had been read from it, and this is the run that is reported.

| sessions | steps | exact steps | parsed | seconds | exact steps per GPU hour | round time median / 95th / largest (s) | computed prompt tokens per step | served from the cache | board power | joules per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 800 | 800 (100.0 %) | 100.0 % | 56 | 51,134 | 0.57 / 0.66 / 0.81 | 105.3 | 69.67 % | 71 W | 5.035 |
| 32 | 3,200 | 3,200 (100.0 %) | 100.0 % | 130 | 88,635 | 1.28 / 1.48 / 2.19 | 96.3 | 71.12 % | 72 W | 2.923 |
| 128 | 12,800 | 12,800 (100.0 %) | 100.0 % | 439 | 105,043 | 4.27 / 5.24 / 8.05 | 94.2 | 71.52 % | 71 W | 2.464 |
| 256 | 25,557 | 25,557 (100.0 %) | 100.0 % | 874 | 105,230 | 8.49 / 10.66 / 16.18 | 94.4 | 71.46 % | 71 W | 2.458 |

Corrupted steps: 0 in every cell; no job stopped. The cell of 256 holds 25,557 steps, not 25,600, because the 256th job (job 447, seed 20261017447) has a script of 57 steps in the file the engine saved; the other 255 jobs have 100. The pre-flight of the earlier runs checked jobs 192 to 255 only, and I did not check jobs beyond 255 before this run. Figures above are the lines the notebook printed; the per-round times and totals are in `docs/results/axi_sessions_per_chip_compact.json` (sha256 `a9d5a3b9d9481520de1477031c91b2886a779bcc54eb6490d471389ce12fd489`), which also holds the sha256 of each of the four raw per-step files. Those raw files (0.18, 0.69, 2.7 and 5.4 MB) were on the Colab machine only and are lost when it is shut down; the compact file is what is kept, and a re-count from it (steps, exact, seconds, cache share, computed tokens per step) agrees with the printed lines. The cell of 32 is the jobs of the `PN` cell of Test 35: 88,635 steps per hour here against 86,374 there, 2.6 percent apart, with the cache the only difference.

Readings against the predictions:
1. Every cell ran all its steps (K times 100) with at least 99.5 percent parsed: **NO, by the letter.** Parsed was 100.0 percent in all four, but the cell of 256 ran 25,557 steps, not 25,600, for the reason above; no step was stopped or lost, and the script of job 447 had 57 steps.
2. Exact share at least 99 percent in every cell: **YES** (100.0 in all four).
3. Exact steps per hour at 256 at least twice those at 8: **YES** (2.06 times; 105,230 against 51,134). Only just.
4. Exact steps per hour at 256 at least 1.5 times those at 32: **NO.** It is 1.19 times (105,230 against 88,635).
5. Share of prompt tokens from the cache at 256 at least 60 percent: **YES** (71.46).
6. Median round at 256 under 10 s and the largest round under 30 s: **YES** (8.49 s and 16.18 s).

What this shows and what it does not. The held state stayed exact on every step at every number of sessions tried, up to 256 on one card. The steps per hour rose from 8 to 32 sessions (1.73 times) and a little from 32 to 128 (1.19 times), and then **did not rise at all from 128 to 256** (1.002 times): on this card with this model the chip is full at about 128 sessions, and adding sessions past that makes every session wait longer without making the chip do more. The round time, which is the wait of every session for its step in this lock-step setup, was 0.57 s, 1.28 s, 4.27 s and 8.49 s at 8, 32, 128 and 256 sessions, so past the knee the wait doubles when the sessions double. The largest round (16.18 s at 256) is the first round, when the 256 sessions' own prefixes are not in the cache yet (52,096 of its 100,118 prompt tokens, 52 percent, were served from the cache, which is the part the sessions share); later rounds run near 8 to 9 s. The cache share (about 71 percent) and the computed tokens per step (94 to 105) did not move across sizes, so the prefix cache was holding all the sessions' prefixes even at 256 (the cache has room for 160,000 tokens). Energy per exact step fell from 5.0 J at 8 sessions to 2.46 J at 128 and 256. Not shown: how this looks with arrivals at random times (a real service is not in lock-step rounds), a bigger or smaller model, another card, 32B, or sessions longer than 100 steps; and the 128-to-256 flat line is the card's speed limit with this engine and setting, not a statement about the framework.

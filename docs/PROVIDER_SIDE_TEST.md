# What does the GPU spend on a job when many jobs share it and the engine caches prefixes? A provider-side serving test (written BEFORE the run, 17 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before the serving engine this test is written for has been run on it. Code: `python/axi/experiments/provider_side.py`; instrument checks `python/tests/test_provider_side.py`; notebook `python/colab/axi_provider_side.ipynb`. Concept-map item 106. It needs a GPU (a Colab T4 is the one assumed), the engine vLLM, and about two to three hours for all cells (an estimate, see "How it is run").

## Where the question comes from
On 17 Oct 2026 the owner said what the saving is for: not the customer's bill but the company that runs the model ("it is from a company end"). For that company the token count of a plain loop is not the cost. The measured 87.1 percent fewer tokens (`ENDURANCE_TEST.md`, part A) counts a chat loop that sends the whole chat again at every step and whose engine computes every token again; that page says itself that a serving stack that caches the prefix pays less and that this competitor "is not measured here". An engine that caches keeps the key/value rows of a prefix it has seen, so a kept chat only computes its new turn. What the company pays for is then two other things: the **GPU time for what is not cached**, and the **memory that holds what is**. A GPU holds a fixed number of cached tokens; when more sessions are alive than fit, the oldest rows are thrown away, and the next turn of those sessions has to compute everything again. This test measures both on one GPU, against the fair competitor: a kept chat with the cache on.

**What I expect, written before the run:** at light load a kept chat whose cache works costs about as much as exact-state, so the 87 percent does not carry over to a serving stack; the difference should appear when the jobs outnumber the cache. If that is what comes out, it is the sentence the article gets, and not the 87 percent.

## What this is and what it is not
A measurement of GPU seconds and tokens on one machine with one engine, for synthetic 30-step jobs. It is not dollars, not another model, GPU or engine, and not a provider's real traffic (random arrivals, waiting between steps, many GPUs, caches on CPU or disk). It does not measure the host-side layer of AXI (microseconds, `LATENCY_7B.md`). It measures answer quality only to check that it did not move.

## Design
**Engine, model, cache.** vLLM (version recorded in the result); `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit AWQ weights, fp16 compute: the closest the engine gets on a 16 GB GPU to the Qwen2.5-7B of the earlier tests); `max_model_len` 8,192; greedy; at most 96 new tokens; the replies are not grammar-constrained (the constrained and the unconstrained call gave the same text in 60 of 60 tasks, `LATENCY_7B.md` section 3); prefix caching on. The size of the cache is fixed at **5,000 blocks of 16 tokens = 80,000 tokens** (about 4.3 GiB of key/value rows for this model, 57,344 bytes a token) with the engine's block override, so that "how many jobs fit" is a number on this page and does not depend on which GPU the session gets. If the installed version has no override, the size the engine chose is read and recorded and line 2 says whether the pressure was there.

**Jobs.** The 30-step jobs of `LONG_HORIZON_TEST.md` (`oracle_script`, the base mix, no rule in the prompt, the same prompts, policy repair_m6 holding the state). Episode e has seed 20261017 x 1000 + e, so the first 8 episodes are the same at every K.

**Arms.**
| arm | what the model is sent at each step | cache |
|---|---|---|
| `S` | system prompt, the exact current state, the request (a fresh context every step) | on |
| `KC` | the whole chat: every earlier state, request and raw reply stays, plus the new state and request | on |
| `KN` | the same chat as KC | emptied before every round: what a plain engine pays |
| `KR` | the same chat as KC | on, and the jobs are served in a different random order in each round (descriptive only) |

**Rounds.** K jobs run side by side in lockstep: in each round every unfinished job sends its next request, the engine serves all of them in one call, then each job applies its reply and sends the next. That models K sessions sharing one GPU, each coming back after the others were served. With a fixed order this is the hardest case for a cache that evicts the oldest rows first; `KR` is there to show how much of the result depends on it.

**Measures.** Per request: prompt tokens (the engine's count), cached tokens (what the engine reports it did not have to compute), new tokens. Per round: wall seconds around the engine call. Per cell: **GPU time per step** (the sum of those seconds over the steps done), **computed prompt tokens per step** (prompt minus cached), the share of prompt tokens served from the cache (all steps, and the second half of the steps), the mean prompt in the first and the last quarter of the steps, the longest prompt, and the share of steps on which the held state was exactly the oracle's.

**Cells and order.** `S` and `KC` at K = 8, 32, 64; `KN` at K = 8 (it computes every prompt token whatever K is); `KR` at K = 32. Light load first, heavy last. Every cell starts with a cold cache, after one warm-up call. If the session ends before the K = 64 cells, the lines that need them read "not run", not "NO".

## What I looked at before locking this page (and what I did not)
**The saved long-horizon runs** (Qwen2.5-7B, `axi_long_history_norule` and `axi_long_state_norule`, seed 3): the kept chat's prompt was 361 to 5,530 tokens, mean 1,057 in the first quarter of the steps and 4,602 in the last (4.4 times); the exact-state prompt was 234 to 476 tokens, mean 396 in the first quarter and 311 in the last. repair_m6 held the exact state on 99.4 percent of the steps in state mode and 100 percent in history mode. I first wrote the threshold of line 6 as 4 times and changed it to 3.5 after seeing the 4.4, which is too close to call.

**The stand-in.** `python -m axi.experiments.provider_side --dry-run` runs every planned cell on a model of a block cache that I wrote (blocks of 16 tokens chained by hash, the cache holds 89,600 proxy tokens, which is the 80,000 tokens scaled by the 12 percent by which its regular-expression token count exceeds Qwen's, a freed sequence loses its tail first, the oldest freed block is evicted first, the jobs of a round are served one after another instead of at once, and a time of computed tokens / 1,500 per second plus 0.03 seconds per reply token that I made up). It takes four seconds and the numbers below are its output. **It is a model of my assumptions, not a measurement of any engine.**

| arm | K | steps | seconds | seconds per step | prompt tokens | computed | cached | cached, second half |
|---|---|---|---|---|---|---|---|---|
| S | 8 | 240 | 66 | 0.28 | 100,859 | 34,347 | 66% | 68% |
| KC | 8 | 225 | 74 | 0.33 | 844,026 | 49,530 | 94% | 97% |
| KN | 8 | 225 | 574 | 2.55 | 844,026 | 800,346 | 5% | 4% |
| S | 32 | 960 | 140 | 0.15 | 388,783 | 126,575 | 67% | 70% |
| KC | 32 | 918 | 1,738 | 1.89 | 3,239,783 | 2,523,575 | 22% | 4% |
| KR | 32 | 918 | 1,224 | 1.33 | 3,239,783 | 1,752,855 | 46% | 31% |
| S | 64 | 1,920 | 215 | 0.11 | 753,570 | 238,498 | 68% | 72% |
| KC | 64 | 1,867 | 3,903 | 2.09 | 6,407,505 | 5,769,249 | 10% | 5% |

(The kept-chat cells hold fewer steps than the exact-state ones because the stand-in counts about 12 percent more tokens than Qwen, so its longest chats pass the 7,600-token stop that the real ones, at about 5,500 tokens, should not.) What it shows, in that model: with a cache that fits, the kept chat computes about 220 prompt tokens a step and exact-state about 143, because exact-state reuses the system prompt and the unchanged start of the state while the kept chat's new turn follows a different history and is computed whole; the two cost about the same time (0.33 against 0.28 seconds a step); with the jobs outnumbering the cache the cached share collapses (22 percent at K = 32, 10 percent at K = 64) and the kept chat takes 13 to 19 times the time per step of exact-state; a random order helps (46 percent cached instead of 22 at K = 32) and does not close the gap.

**What I did not look at.** vLLM. This workspace has no GPU and no vLLM; the class that talks to it was run only against a fake module that has the calls it makes, and it is unproven until the notebook's first cell runs. No number from a real engine has been seen.

**The predictions of lines 11 to 13 were written knowing the stand-in's shape, so they are near-certain in that model.** What the real run adds is whether the engine behaves like it: it may keep more rows than my eviction rule, it serves the jobs of a round concurrently, it preempts and recomputes when the running jobs alone do not fit, and its time is not my made-up time.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. The first four lines are about the instrument and the quality; a "NO" on 1, 2, 3 or 4 limits what the others can be used for (see below).

1. [instrument] the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run (85 %)
2. [instrument] the setup made the pressure it was meant to: 64 times the longest kept-chat prompt at K = 64 is at least twice the cache size, and 8 times the longest at K = 8 fits in it (90 %)
3. [guard] in every cell the held state was exactly the oracle's state on at least 90 percent of the steps (85 %)
4. [claim] exact-state was not less exact than the kept chat: at every K its share of exact steps is at least the kept chat's minus 5 points (85 %)
5. [claim] exact-state prompts stay flat: the mean prompt of the last quarter of the steps is at most 1.3 times that of the first quarter (K = 8) (95 %)
6. [claim] kept-chat prompts grow: the mean prompt of the last quarter is at least 3.5 times that of the first quarter (K = 8) (90 %)
7. [claim] at K = 8 the cache works for the kept chat: at least 80 percent of its prompt tokens came from the cache (88 %)
8. [claim] at K = 8 against the chat with the cache emptied each round, exact-state computes at least 80 percent fewer prompt tokens (95 %)
9. [claim] at K = 8 with the cache on, exact-state computes at least 30 percent fewer prompt tokens than the kept chat (50 %)
10. [claim] at K = 8 with the cache on, exact-state takes at least 20 percent less GPU time per step than the kept chat (40 %)
11. [claim] at K = 64 the kept chat overruns the cache: in the second half of the steps less than 50 percent of its prompt tokens came from the cache (88 %)
12. [claim] at K = 64 the kept chat takes at least twice as much GPU time per step as exact-state (92 %)
13. [claim] at K = 32 the kept chat takes at least 1.3 times as much GPU time per step as exact-state (90 %)

**Headline** (lines 10 and 12): the two GPU-time lines that carry the question, at light load and under pressure. My probabilities are 40 % and 92 %, my number for both **35 %**. All thirteen lines yes: about **8 %** (lines 9 and 10 carry most of the doubt). Lines 1 to 4 all yes: about **60 %**.

Why I think so, line by line. *Lines 1 and 2 (85, 90 %):* as far as I remember, recent vLLM versions report the cached count per request and accept a block override; I have not seen the installed one. The longest kept-chat prompt should be about 5,500 tokens, so 8 jobs hold 44,000 tokens in an 80,000-token cache and 64 jobs ask for about 350,000. *Line 3 (85 %):* the earlier 7B run held the exact state on 99.4 percent of the steps; the AWQ weights and replies that are not constrained are new, and a reply with words after the delta fails the strict parser and leaves the state unchanged. *Line 4 (85 %):* history mode was at 100 percent against 99.4. *Lines 5 and 6 (95, 90 %):* the saved runs give 0.8 and 4.4. *Line 7 (88 %):* the stand-in gives 94 percent; the rest is the new turn of every step. *Line 8 (95 %):* a cache that is emptied recomputes the chat each time; exact-state computes about 143 tokens a step against about 3,600 for the whole chat. *Line 9 (50 %):* in the stand-in exact-state computes 35 percent fewer tokens than the kept chat, just above the 30 percent line; the real number depends on how many blocks of the state survive a one-item change. *Line 10 (40 %):* time is not only computed tokens: decoding is cheaper with a short context, but at 8 jobs and 20 reply tokens the engine spends most of a round on weights, and the stand-in gives 16 percent, under the 20 percent line. *Lines 11, 12 and 13 (88, 92, 90 %):* a fixed order meets the worst case of a cache that evicts the oldest rows: each job returns just after its rows were thrown away.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers.
* *Lines 9 and 10 NO, lines 11 to 13 yes (the shape I expect):* at light load a working prefix cache makes a kept chat about as cheap to serve as exact-state; the 87 percent does not carry over to a serving stack. What exact-state buys is staying inside the cache: a GPU serves many more jobs before the kept chat's rows are thrown away, and the factor is the one measured at K = 32 and 64, on this GPU and cache size. That, with its limits, is the only serving sentence the article gets.
* *Lines 9 and 10 yes:* exact-state is cheaper than a working cache even at light load, by the measured amount (not 87 percent).
* *Line 11, 12 or 13 NO:* my model of the cache is wrong for this engine (it keeps its rows better, or its scheduler hides the cost). The first things to look at would be the engine's preemption count and the rounds' seconds one by one; no claim about the heavy load is made.
* *Line 3 or 4 NO:* the arms are not doing the same work, and their times are not compared until the cause is found.
* *Line 1 or 2 NO:* nothing else on this page is read: without the cached count computed and cached tokens cannot be told apart, and without the pressure the sweep over K says nothing about it.
* *`KR`:* descriptive. If it lands near `KC`, the order does not matter; if near `S`, the heavy-load result belongs to a fixed order.

## Reading rules (fixed now; the printout prints exactly these)
The thirteen lines above are the reading rules; `provider_side.verdict_provider` computes them from the saved cells. Lines 5 to 10 use the K = 8 cells, line 2 the K = 8 and K = 64 cells of `KC`, lines 11 and 12 the K = 64 cells, line 13 the K = 32 cells; lines 1 and 3 use every cell that was run, line 4 every K at which both `S` and `KC` were run. "GPU time per step" is the wall seconds around the engine's call, summed over the rounds of the cell, divided by the steps done in it; "computed prompt tokens" are the prompt tokens minus the tokens the engine reports as cached, per step done (lines 8 and 9); "the second half" is the steps from the 16th on; "the last quarter" is the last 7 of 30 steps, "the first quarter" the first 7. Thresholds as in the code: line 3: 0.90; line 4: 0.05; line 5: 1.3; line 6: 3.5; line 7: 0.80; line 8: 0.80 fewer; line 9: 0.30 fewer; line 10: 0.20 less; line 11: 0.50 (strictly below); lines 12 and 13: 2.0 and 1.3 times. A line that needs a cell the run does not hold reads "not run".

**How the result will be worded.** In GPU seconds on the machine the run had, with the engine version and the GPU written next to them, never in dollars and never as a statement about any provider. A "NO" on a claim line is a finding about this setup and goes into the article as such.

## What this does not show
Dollars, other GPUs, other models, other engines, other cache sizes (the result is for 80,000 tokens; a larger cache moves the point at which the kept chat overflows to a larger K, and the page does not say where); traffic that is not a round of K jobs (random arrivals and waiting would let some kept chats keep their rows, which is what `KR` hints at); a cache that moves rows to CPU memory or disk; answers in ordinary words; jobs longer than 30 steps; a kept chat that sends only the requests (it would be shorter and would not show the model the exact state, which is the thing AXI changes); the saving against a chat that is summarised or cut, which is not run here either. It says nothing about the tensor-level roadmap.

## Related work (from memory, not searched; check before citing)
Paged key/value caches and automatic prefix reuse in serving engines (Kwon et al., 2023, vLLM / PagedAttention); radix-tree reuse of shared prefixes (Zheng et al., 2023, SGLang / RadixAttention). If these hold as I remember them, prefix reuse is standard in serving stacks, which is why a kept chat with the cache on is the competitor and a kept chat without it is only a reference.

## How it is run
Open `python/colab/axi_provider_side.ipynb` in Colab with a T4, upload the code zip, and run the cells one at a time: it installs vLLM, loads the model, runs a smoke test (3 jobs, 3 steps, all arms: it prints the cached counts so that a version without them shows at once; if a red error appears, stop and send it), then one cell per K. Each cell writes `axi_provider_<arm>_K<k>.json` and, if Drive is connected, copies it there. The last cell puts the cell files together (`python -m axi.experiments.provider_side --merge ...`) and prints the table and the thirteen lines. The time of the cells is my guess, never measured: with a T4 that computes 1,000 to 1,500 prompt tokens a second, `S` and `KC` at K = 8 take a few minutes, `KN` at K = 8 about ten, `S` and `KC` at K = 32 about half an hour, `KR` at K = 32 about twenty minutes, `S` at K = 64 a few minutes and `KC` at K = 64 about an hour. The result goes below the line at the end of this page, and a result test will recompute every number on it from the saved file.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side.py` and again by `tests/test_provider_side_result.py`). The run was made by the owner on a free Colab GPU on 17 and 18 Oct 2026 with `python/colab/axi_provider_side.ipynb`: vLLM 0.31.0, GPU Tesla T4, Python 3.13.15, `Qwen/Qwen2.5-7B-Instruct-AWQ`, the cache fixed at 80,000 tokens with the engine's block override (the engine took it: capacity source "override"), 30-step jobs, seed 20261017, replies not constrained. **Six of the eight planned cells ran:** `S` at K = 8, 32 and 64, `KC` at K = 8 and 32, `KN` at K = 8. **Not run: `KC` at K = 64** (the owner's free GPU time ran out inside that cell, and a cell that is cut off is not saved) **and `KR` at K = 32** (optional). The six cell files are in `docs/results/provider_side_cells/`; I put them together here, not in Colab, with `provider_side.merge_cells`: `docs/results/axi_provider_side.json` (sha256 `19ce96633931980292f2ab6b25c42799b50696db2ea25225f194e02c8ca93d48`; it records the sha256 of the module that merged it, `bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3`). The GPU time of the six cells adds up to 7,055 seconds (118 minutes). Every figure below is computed from the step records of those files by `tests/test_provider_side_result.py`.*

**Two things about the run that the page did not foresee.**
* Cell 5 of the notebook first stopped with `io.UnsupportedOperation: fileno`: this vLLM version asks the notebook's output stream for a file number, which a Colab cell does not have. The owner ran a short cell that gives `sys.stdout` and `sys.stderr` the numbers 1 and 2 and ran Cell 5 again. `provider_side.py` was not changed (its sha256 is the one above). I tested the short cell against a copy of the situation here, not on Colab; the run is the evidence that it worked. Update 57 puts that cell into the notebook and pins vLLM to 0.31.0, so that the missing cell can be run on the same version (a cell from another version is refused by the merge).
* My time guesses were too low. `KN` at K = 8 took 27 minutes, not about ten, and `KC` at K = 32 took 82, not about thirty. `KN` computed prompt tokens at about 500 a second (the tokens `KN` computed beyond `KC`, over the seconds it took beyond `KC`; a rough rate). For `KC` at K = 64 I would now expect about three hours or more: twice the jobs of K = 32 and a cache that overflows even earlier. That is an estimate, not a measurement.

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 85 % | the engine reported the cached tokens of every request: 4,552 steps in the 6 cells that ran | yes |
| 2 | 90 % | not run: it needs the 64-job kept chat. What the cells show, and not the line: the longest kept-chat prompt at K = 32 was 7,556 tokens, 32 times that is 241,792, 3.0 times the cache of 80,000; at K = 8 it was 7,154, 8 times that is 57,232, which fits | not run |
| 3 | 85 % | exact steps, `S` 76.2 % (K = 8), 78.2 % (K = 32), 69.2 % (K = 64); `KC` 100.0 % (8), 97.3 % (32); `KN` 100.0 % (8): three of the six cells are below 90 % | NO |
| 4 | 85 % | `S` against `KC`: K = 8 76.2 % against 100.0 %, 23.8 points short; K = 32 78.2 % against 97.3 %, 19.0 points short (limit 5; K = 64 has no `KC`) | NO |
| 5 | 95 % | `S` at K = 8: mean prompt 394 tokens in the first 7 steps, 343 in the last 7: 0.87 times (limit 1.3) | yes |
| 6 | 90 % | `KC` at K = 8: mean prompt 1,053 tokens in the first 7 steps, 5,644 in the last 7: 5.36 times (limit 3.5) | yes |
| 7 | 88 % | 94.3 % of the prompt tokens of `KC` at K = 8 came from the cache (limit 80 %) | yes |
| 8 | 95 % | `S` computes 123 prompt tokens a step, `KN` 3,297: 96.3 % fewer (limit 80 %) | yes |
| 9 | 50 % | `S` computes 123 tokens a step, `KC` 196: 37.3 % fewer (limit 30 %) | yes |
| 10 | 40 % | `S` 0.154 seconds a step, `KC` 0.617: 75.0 % less (limit 20 %); `KC` took 4.00 times as long | yes |
| 11 | 88 % | not run: it needs the 64-job kept chat (at K = 32, for what it is worth, 6.3 % of the prompt came from the cache in the second half, 26.0 % over all steps) | not run |
| 12 | 92 % | not run: it needs the 64-job kept chat | not run |
| 13 | 90 % | `KC` 5.149 seconds a step, `S` 0.121: 42.7 times (limit 1.3) | yes |

**8 of the 13 lines read yes, lines 3 and 4 read NO, and lines 2, 11 and 12 are "not run".** Of the headline pair (lines 10 and 12) line 10 reads yes and line 12 could not be read, so the headline is half read and I do not call it either way. Lines 1 and 5 to 10 and 13 read yes; of those, lines 9 and 10 were the two I trusted least (50 and 40 %) and both came out yes, and lines 3 and 4, which I put at 85 % each, are the two NO.

**What the page said about lines 3 and 4.** "Line 3 or 4 NO: the arms are not doing the same work, and their times are not compared until the cause is found." So, whatever lines 9, 10 and 13 read, **this run makes no claim that exact-state is faster than a kept chat.** Their times are in the tables because nothing is to be hidden; what is claimed is below, under "What this does and does not show". **Line 2** is an instrument line and it is not read; the page said that a NO on it would stop the reading of everything else, and "not run" is not a NO. The cells that ran show the pressure was there at K = 32 (the figures are in the table row), but line 2 is written for K = 64.

## Where my expectation was off
* **Lines 9 and 10 (50 and 40 %): at light load the kept chat was not about as cheap as exact-state.** I wrote that at light load a working cache would make the two about equal (my model of a cache gave 35 % fewer tokens and 16 % less time for exact-state, the second under the 20 % line). The cache worked: 94.3 % of the kept chat's prompt tokens came from it. Exact-state still computed 37.3 % fewer prompt tokens and took 75.0 % less time per step, so the kept chat took 4.00 times as long (0.617 against 0.154 seconds a step). I did not look for the reason. Candidates I have not tested: a prompt that grows from 1,053 to 5,644 tokens makes every step read more rows even when they are not computed again; the replies are short (10.6 new tokens a step in every cell), so the time is almost all prompt work.
* **Lines 3 and 4 (85 % each): exact-state did not hold the right state on 9 steps in 10.** 76.2 % at K = 8, 78.2 % at 32, 69.2 % at 64, against 97.3 % for the kept chat at 32 and 100 % at 8. My note under line 3 on the page had named the risk (a reply with words after the delta fails the strict parser and leaves the state unchanged, and the AWQ weights and the replies that are not constrained were new); I gave it a 15 % chance and it was the main event. See below for where the exactness went.
* **The time guesses** (above) and **line 6's threshold**: the kept chat's prompt grew 5.36 times, more than the 4.4 of the earlier run, so the lowered threshold of 3.5 was not needed; the lowering was disclosed on the page.
* **Lines 5, 7, 8 and 13 came out as expected**, line 13 by a wide margin (the limit is 1.3 times; the figure is 42.7).

## The tables
**Every cell** (the seconds are the wall seconds around the engine's call, summed over the 30 rounds; "computed" is the prompt tokens the engine did not take from its cache; "cached, second half" is the steps from the 16th on; the prompt columns are the mean prompt of the first 7 and the last 7 steps of the jobs; "exact" is the share of steps after which the held state equalled the oracle's):

| arm | K | steps | seconds | s / step | prompt tokens | computed | cached | cached, 2nd half | prompt, first 7 | prompt, last 7 | longest | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `S` | 8 | 240 | 37 | 0.15 | 90,679 | 29,543 | 67 % | 70 % | 394 | 343 | 516 | 76.2 % |
| `KC` | 8 | 240 | 148 | 0.62 | 831,529 | 47,145 | 94 % | 97 % | 1,053 | 5,644 | 7,154 | 100.0 % |
| `KN` | 8 | 240 | 1,646 | 6.86 | 831,529 | 791,209 | 5 % | 3 % | 1,053 | 5,644 | 7,154 | 100.0 % |
| `S` | 32 | 960 | 116 | 0.12 | 349,176 | 109,688 | 69 % | 71 % | 388 | 340 | 516 | 78.2 % |
| `KC` | 32 | 952 | 4,902 | 5.15 | 3,065,348 | 2,268,404 | 26 % | 6 % | 1,039 | 5,259 | 7,556 | 97.3 % |
| `S` | 64 | 1,920 | 205 | 0.11 | 682,313 | 208,489 | 69 % | 72 % | 385 | 335 | 516 | 69.2 % |

**Where the exactness went (counted afterwards, from the step records; none of it was pre-registered).** "Jobs exact to the end" are jobs whose held state equalled the oracle's after every step. "Unparsed" are replies the strict parser refused; "on delete steps" is how many of those were on a `delete_dep` or `delete_leaf` step. "First misses" classifies the first step of each job after which its state differed. "Corrupted" counts states that were not well formed.

| arm | K | jobs exact to the end | unparsed replies | on delete steps | first misses | corrupted |
|---|---|---|---|---|---|---|
| `S` | 8 | 3 of 8 | 6 of 240 | 5 | 4 did not parse, 1 parsed and were refused, 0 were admitted and still differ | 0 |
| `KC` | 8 | 8 of 8 | 0 of 240 | 0 | no job lost its state | 0 |
| `KN` | 8 | 8 of 8 | 0 of 240 | 0 | no job lost its state | 0 |
| `S` | 32 | 18 of 32 | 16 of 960 | 13 | 11 did not parse, 2 parsed and were refused, 1 were admitted and still differ | 0 |
| `KC` | 32 | 31 of 32 | 9 of 952 | 9 | 1 did not parse, 0 parsed and were refused, 0 were admitted and still differ | 0 |
| `S` | 64 | 27 of 64 | 58 of 1,920 | 55 | 33 did not parse, 2 parsed and were refused, 2 were admitted and still differ | 0 |

**Steps per hour (derived afterwards; read the caution).** "Steps per hour" is 3,600 times the steps done over the seconds of the cell; the last column multiplies it by the share of exact steps. It is crude: once a job's state falls behind the oracle's it usually stays behind (in 52 of the 56 `S` jobs that missed, it did), so every later step of the job counts as inexact, and by the page's own rule for lines 3 and 4 these times are not to be compared as a claim yet.

| arm | K | steps per hour | exact | exact steps per hour |
|---|---|---|---|---|
| `S` | 8 | 23,321 | 76.2 % | 17,782 |
| `KC` | 8 | 5,831 | 100.0 % | 5,831 |
| `KN` | 8 | 525 | 100.0 % | 525 |
| `S` | 32 | 29,831 | 78.2 % | 23,336 |
| `KC` | 32 | 699 | 97.3 % | 680 |
| `S` | 64 | 33,684 | 69.2 % | 23,316 |

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The kept chat's cache worked at 8 jobs and gave way at 32.** At K = 8, 94.3 % of its prompt came from the cache and it computed 196 prompt tokens a step. At K = 32 the cache served 26.0 % over all steps and 6.3 % in the second half, it computed 2,383 tokens a step, and the time per step went from 0.617 to 5.149 seconds, 8.3 times as much, while each job did the same work in the same 30 steps and the cache was the same size. That is 75 % of the time per step of `KN` at K = 8, the kept chat whose cache was emptied on purpose (6.860 seconds a step). The slowest round of `KC` at K = 32 took 392 seconds. The cache was 80,000 tokens; 32 chats of the sizes measured here (the longest 7,556 tokens, 32 of them 241,792, 3.0 times the cache) do not fit in it. In 3 of the 32 chats the test's own stop rule ended the job (the prompt would have passed 7,600 tokens): 952 steps were done instead of 960. A kept chat also runs out of room in the model's context.
2. **Exact-state stayed flat and small.** Its prompt was 394 tokens over the first 7 steps and 343 over the last 7 at K = 8 (0.87 times), the longest 516 tokens in any cell; 64 jobs of that size need at most 33,024 tokens, 41 % of the cache. It served 69 % of its prompt from the cache at 64 jobs, the shared instructions. Its time per step fell as the jobs grew in number: 0.154, 0.121 and 0.107 seconds at 8, 32 and 64 jobs (a GPU is used better with more jobs in a round).
3. **Where the exactness went.** No state was ever corrupted: 0 of the 4,552 steps. What failed was a reply that the strict parser refused (or the gate refused): the job then keeps its last valid state (policy `repair_m6`) and its state is behind the oracle's from then on: in 52 of the 56 `S` jobs that missed, no later step was exact again, and in 4 the state became exact again later. In the `S` cells 48 of the 56 first misses were replies that did not parse, for example `DEL[4,5:8]` (a value after a delete) and `ADD[(4,4):6]` (brackets around the coordinates); 80 of the 3,120 replies of the `S` cells did not parse, 73 of them on delete steps. At 64 jobs the refusals were 37 of 508 of the `delete_dep` steps and 18 of 171 of the `delete_leaf` steps, 3 of 627 of the `add_node` steps and 0 of 614 of the `add_edge` steps. The kept chat made the same slip rarely: none of its 240 replies at 8 jobs, 9 of its 952 at 32 jobs (all 9 on delete steps), which cost one job its state (31 of 32 jobs exact to the end). **The cause is not found, but the most likely one is the replies that are not constrained.** The earlier long-horizon run held the exact state on 99.4 % of the steps in state mode with constrained decoding (`LONG_HORIZON_TEST.md`); this page chose replies that are not constrained because the constrained and the unconstrained call gave the same text in 60 of 60 single tasks, and 30-step jobs show that this did not carry over. A kept chat shows the model its own earlier replies in the right form, which fits its near-zero slip rate; I have not tested that either. Until a run with the constrained replies (or with equal exactness in another way) exists, the exactness of the arms is not equal and their times are not compared.
4. **Rough, and not a claim:** exact steps per hour at K = 32 are 23,336 for `S` and 680 for `KC`, a ratio of 34; at K = 8 the ratio is 3.0. The ratio at K = 8 is the one to keep in mind: it is the one measured where the cache worked.

## What this does and does not show
* **Shows, on one Tesla T4, one 7B model with 4-bit AWQ weights, vLLM 0.31.0, a cache of 80,000 tokens and 30-step jobs:** a kept chat's prefix cache works when the chats fit in it and stops working when they do not (K = 8 against K = 32); the time of a kept chat per step then rose 8.3 times; the exact-state prompt does not grow and stays a small share of the cache at 64 jobs. That is the measured part of the capacity idea: how many jobs a cache holds depends on the size of a job's prompt, and the size of the exact-state prompt does not grow with the length of the job.
* **Does not show that exact-state is cheaper at equal quality** (lines 3 and 4 NO), and so no ratio of the times is a claim. It does not show the kept chat at 64 jobs (line 12, the second half of the headline), a random order (`KR`), another GPU (this one computed about 500 prompt tokens a second; a data-centre GPU is much faster, and the ratios, not the seconds, are what might carry over, which is not shown either), another model, engine or cache size, dollars, **energy (never measured; only time)**, retries (this test sends one request a step and never asks again; a job whose reply is refused costs a retry in a real system, for either arm), traffic that is not a round of jobs, or a number of agents per GPU: "four to ten times as many" is arithmetic on prompt sizes, not a measurement.
* **The sentence the article may carry from this run, with its limits:** *on one T4 with a 7B model and a cache of 80,000 tokens, a kept chat of 32 jobs overflowed the cache and took 5.149 seconds a step, against 0.617 for 8 jobs; exact-state prompts stayed at 516 tokens or fewer and 64 jobs of them needed at most 41 % of the cache; exact-state held the right state on 69.2 % to 78.2 % of its steps against 97.3 % for the kept chat at 32 jobs, with replies that were not constrained, so the arms are not compared.*
* **Not done and worth doing, in this order:** (1) the same cells with replies constrained by a grammar for all arms, so that exactness is equal first and the times are compared after (a new page and new predictions; `S` is cheap, a few minutes a cell, `KC` at K = 32 is about an hour and a half); (2) `KC` at K = 64 on the same vLLM version; (3) retries counted, so that a refused reply costs what it costs; (4) the GPU's power read next to the seconds.

## Reproduce
In Colab, `python/colab/axi_provider_side.ipynb` with a T4 (the cell that fixes the output stream first, then Cell 5 onward); about 118 minutes of GPU for the six cells here, and about three hours more for `KC` at K = 64. Nothing needs the GPU after that: `provider_side.merge_cells` puts cell files together on any machine. `tests/test_provider_side_result.py` recomputes every figure of this Result from the cell files, checks that the merged file is what `merge_cells` makes of them, and that the page is the locked text followed by this Result.

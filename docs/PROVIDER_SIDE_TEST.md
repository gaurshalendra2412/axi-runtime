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

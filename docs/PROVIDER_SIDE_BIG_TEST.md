# Test 21: the serving side on a bigger model, counted the way a company that runs the model counts (written BEFORE the run, 21 Oct 2026)

*Concept-map item 114. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page is a result.*

## Why this test
AXI is built for the companies that run models. What such a company counts is: does the model **drift** over a long task, how fast is a step (**speed**), how long does a request wait (**latency**), how many prompt tokens are computed for each step (**tokens**) and how much energy a step takes (**energy**). The first seven runs answered the first questions on one small model (a 7B model, 4-bit, on a Tesla T4): the exact state with the grammar that names the coordinates held the state on 6,400 of 6,400 steps of 100-step jobs, and the kept chat stopped before step 100. The seventh run showed that ordinary words are a separate front-end problem for that model: it held the state on 39.8 % of steps when it had to read a name, 61.5 % with the host reader helping. Two questions are open and this page asks both on a bigger model: **do the serving numbers hold, and get measured in time, tokens and energy, on a model twice the size?** and **does the bigger model read the names that the 7B could not?**

## What is run
One model, **`Qwen/Qwen2.5-14B-Instruct-AWQ`** (the 7B's family at twice the size, 4-bit), vLLM 0.31.0, one **NVIDIA L4** (24 GB), greedy decoding, no retry anywhere, the sixth run's policy `repair_m6` and capped-expression hash `6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245`. The cache is fixed at **32,000 tokens** (2,000 blocks of 16) because the weights and the cache share the L4's memory; `max_model_len` is 8,192. The jobs are the sixth run's jobs 192 to 223 (32 jobs) and 192 to 195 (4 jobs), 100 steps each, with the same seeds, so every 7B figure on the same jobs can be read beside the figures here (the 7B model on the same 32 jobs: `L0` 3,075 of 3,200 steps exact, `LN` 3,200 of 3,200, `C0` 3,200 of 3,200, `W0` 1,343 of 3,200, `WN` 2,030 of 3,200). Seven cells:

| cell | jobs | job numbers | what it is |
|---|---|---|---|
| `P0` | 32 | 192 to 223 | the exact state at every step, the request with coordinates, the capped expression (the sixth run's `L0` on a bigger model) |
| `PN` | 32 | 192 to 223 | the exact state, the request with coordinates, the grammar that names the coordinates (the sixth run's `LN`) |
| `P4` | 4 | 192 to 195 | the same as `PN` on the first 4 jobs (the sixth run's `L8`) |
| `PC` | 4 | 192 to 195 | the whole chat kept, named expression, the first 4 jobs; it stops when its next prompt would pass 7,600 tokens (the sixth run's `LC`) |
| `X0` | 32 | 192 to 223 | the request with coordinates plus the list of names, capped expression (the seventh run's `C0`) |
| `XW` | 32 | 192 to 223 | the request in ordinary words, the list of names, capped expression (the seventh run's `W0`) |
| `XR` | 32 | 192 to 223 | the request in ordinary words; the host reader builds the named expression, the capped one when it cannot read (the seventh run's `WN`) |

The code is `python/axi/experiments/provider_side_big.py`; it imports `provider_side.py`, `provider_side_typed.py`, `provider_side_long.py`, `provider_side_free.py`, `grounding.py` and `word_tasks.py` and **edits none of them** (their hashes are checked by the pre-flight). What it adds to the earlier runs' engine: the bigger model and the smaller cache; a record, for every round, of what the engine reports about the time of each request (arrival, first token, finish; if this vLLM version reports nothing, the record says so and nothing is claimed from it); and a **power meter around every cell**: `nvidia-smi` read every half second, the readings added up with the time between them. The meter reads the **board's** power, not the host's and not the building's; the energy figures are the board's joules over the cell's window (its rounds and the host's time between them), divided by the exact steps.

## What I read before writing this page
The 7B figures of the sixth and seventh runs on these jobs (above). The kept chat on the 7B (sixth run, 8 jobs): it stopped after 28 to 64 steps, computed 143 prompt tokens per step against 105 for the exact-state arm, took 0.568 s per step against 0.150 and did 6,342 exact steps per GPU hour against 24,001: those are the ratios the lines below are written against (3.8 times as many exact steps per hour, 1.36 times the computed tokens, 3.8 times the time per step). Colab's rate for an L4 is about 1.71 compute units an hour, for a T4 about 1.19 (a third-party page of March 2026; the rate in the owner's app is what counts), so a run of 2 to 3.5 hours costs about 4 to 6 units. **Why not the v6e-1 TPU:** the vLLM TPU project lists v6e as supported but names Llama, Gemma and Qwen3 models, not Qwen2.5 text models; AWQ is listed for v5 and v6 with its status untested; and the per-request grammar path this test depends on has not been shown there. It needs its own small probe and page; it is not run here.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 7 are checks that the run is the run I describe (lines 4, 5 and 7 are also checked before the run); lines 8 to 21 are about what the model and the serving side do.

1. [instrument] the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run (99 %)
2. [instrument] the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, and in the cells that carry the named expression on every request (PN, P4, PC) none used all 96 new tokens and every reply is in the language of its own request (88 %)
3. [instrument] the power meter worked: for every cell that was run it read at least 90 percent of the readings that the seconds of its rounds allow, at one reading every half second (90 %)
4. [guard] the requests in words say what the oracle's request says: in every sentence of XW and XR the name (or an alias) of each service the oracle's request is about is written, in the order of the oracle's request, the numbers are the oracle's, and a sentence of the second pool names exactly one service more (checked by plain search, not by the reader) (99 %)
5. [guard] the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs (and the reader), and the oracle's own reply is inside the expression of every request (99 %)
6. [guard] every job of every exact-state cell ran all 100 steps; none ended early (95 %)
7. [guard] what the reader read, as recorded in XR, is what the reader gives on the same sentences when it is run again: the same outcome for every one of the 3,200 requests, and not one request read wrong (99 %)
8. [claim] drift, named expression: over 100 steps the exact share of PN is at least 99 percent (88 %)
9. [claim] drift, capped expression: the exact share of P0 is at least that of the 7B model's L0 on the same 32 jobs (3,075 of 3,200) (80 %)
10. [claim] the kept chat does not reach step 100: every job of PC stopped before its 100th step, because its next prompt would have passed 7,600 tokens (95 %)
11. [claim] speed: the exact steps per GPU hour of P4 are at least twice those of PC (both in this session, the same 4 jobs) (80 %)
12. [claim] latency does not grow with the job: in PN the seconds per step of the last quarter of the rounds are at most 1.15 times those of the first quarter (75 %)
13. [claim] latency grows with the kept chat: in PC the seconds per step of the last quarter of the rounds are at least twice those of the first quarter (80 %)
14. [claim] tokens: the kept chat computes at least 1.3 times as many prompt tokens per step as the exact-state arm on the same 4 jobs (PC against P4) (80 %)
15. [claim] energy: the kept chat takes at least twice the GPU energy per exact step that the exact-state arm takes on the same 4 jobs (PC against P4) (75 %)
16. [claim] the grammar costs nothing here: the seconds per step of PN are within 5 percent of those of P0, and the energy per exact step of PN is not above that of P0 (both in this session) (75 %)
17. [claim] the names list does not hurt the bigger model: the exact share of X0 is at least 99 percent (80 %)
18. [claim] the bigger model reads the names: the exact share of XW over 100 steps is at least 95 percent (25 %)
19. [claim] with the host reading what it can, the exact share of XR over 100 steps is at least 99 percent (15 %)
20. [claim] the second pool of wordings does not confuse the bigger model more than three points: in XW the exact share on the steps of that pool is at least that on the first pool minus three points (55 %)
21. [claim] the seconds per step of XR are at most 15 percent above those of XW (both in this session) (90 %)

**Headline** (lines 8 and 18): does the exact-state serving side keep the state on the bigger model, and does the bigger model read a name and write the coordinates by itself. Lines 2 to 7 all yes: about **72 %**. Lines 8 and 18 both yes: about **22 %**. All 21 yes: under **1 %**.

Why I think so. *Line 8 (88 %):* the 7B held every one of 6,400 steps under this grammar and the grammar forbids everything but the request's coordinates; a bigger model has the same constraint. *Line 9 (80 %):* the 7B had 125 inexact steps of 3,200 on these jobs under the capped expression (slips are the model's own choices and stay in the state); a bigger model slips no more often, but 14B is a different model and may slip in other places. *Line 10 (95 %):* the rule is the 7,600-token limit and the kept chat's prompt grows with every step. *Line 11 (80 %):* on the 7B the ratio was 3.8; I allow a wide margin. *Line 12 (75 %):* the exact-state prompt stays at about 330 to 450 tokens, so the time per step should be flat; the cache and the batch can add noise. *Line 13 (80 %):* the kept chat's prompt grows from about 2,500 to 7,500 tokens. *Line 14 (80 %), line 15 (75 %):* on the 7B the ratio of computed tokens was 1.36 and of time 3.8; energy follows time more than tokens. *Line 16 (75 %):* it was +1 % on the 7B. *Line 17 (80 %):* the names list was harmless to the 7B. *Line 18 (25 %):* the 7B read the names on 39.8 % of the steps; a 14B model may read edges better, but 95 percent over 100 steps with persisting slips needs fewer than one new mistake in 2,000 steps. *Line 19 (15 %):* the host refuses 16 percent of the requests and the model has to read them. *Line 20 (55 %):* the second pool is a distractor by design; the 7B missed this line by 4.3 points. *Line 21 (90 %):* the 775 expressions of the seventh run cost +1.3 %.

**Also written down, not lines (read afterwards as descriptions).** (a) More than half of the first slips of `XW` are on edge requests: 70 %. (b) `XW` holds the right state on between 60 and 90 percent of its 3,200 steps: 55 %. (c) The seconds per step of `PN` on the L4 are between 0.12 and 0.30: 65 %. (d) The seven cells take 60 to 120 minutes in all, loading excluded: 60 %. (e) The mean board power of `PN` is between 50 and 72 watts: 70 %. (f) This vLLM version reports request times (first token, finish) in the cells: 40 %.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers.
* *Lines 2 to 7 yes, lines 8 and 18 yes:* on the sixth run's 32 jobs, with a 14B model on an L4, the exact state was held by the named grammar and the model read the names itself on at least 95 percent of the steps. The sentence is the shares, the seconds per step, the exact steps per GPU hour and the joules per exact step, with the setup beside them.
* *Line 8 NO:* the named grammar did not hold the state on the bigger model; read the first misses (printed): which kind. That is the finding.
* *Line 9 NO:* the capped expression slipped more on the bigger model than on the 7B on the same jobs; read where.
* *Line 18 NO with line 17 yes:* the bigger model still does not read names well enough; read the first misses of `XW` and compare their kinds with the 7B's (edges first). That is the case a better reader is for.
* *Lines 11 to 15 NO:* the kept chat's cost in speed, time, tokens or energy is smaller than on the 7B; read the numbers; no ratio is claimed beyond what was measured.
* *Line 12 NO:* the exact-state arm's time per step grows with the job; read the rounds.
* *Line 16 NO:* the grammar cost time or energy here; read `PN` against `P0`.
* *Line 3 NO:* the power meter missed readings; no energy figure is claimed from that cell.
* *Line 2, 4, 5, 6 or 7 NO, the pre-flight RED, or the safety cell says STOP:* nothing is read or claimed until the cause is found.
* *A cell that did not run:* its lines read "not run", never NO.

## Reading rules (fixed now; the printout prints exactly these)
The 21 lines above are the reading rules; `provider_side_big.verdict_big` computes them from this run's cells. Line 1 uses every request of every cell. Line 2 uses every reply (the share the strict parser accepted) and, for `PN`, `P4` and `PC`, every reply (none used all 96 new tokens; each is inside the expression of its own request). Line 3: for every cell, the meter read at least 90 percent of the readings that the seconds of the cell's rounds allow at one reading every half second. Line 4 checks, for every sentence of `XW` and `XR`, by a plain search of the table's names and aliases and of the numbers (not by the reader), that the sentence names the services of the oracle's request in its order, carries its numbers and names exactly one more service in the second pool and none in the first. Line 5 recomputes the sha256 of each cell's set of expressions and checks the oracle's reply is inside the expression of every request. Line 6: every job of the exact-state cells ran all 100 steps. Line 7: the reader, run again on the recorded sentences, gives the recorded outcome for every request, none wrong. Lines 8 to 21 compare shares of exact steps ("the held state equals the oracle's after the step"), seconds per step ("the seconds of the rounds divided by the steps"), exact steps per GPU hour, the computed prompt tokens per step ("prompt tokens minus the tokens the cache served, over the steps"), the seconds per step of the last quarter of a cell's rounds against its first quarter (the seconds of a round divided by the requests in it, so that a cell whose jobs stop one by one is compared fairly), and the joules per exact step (the meter's joules over the cell's window, divided by the cell's exact steps). The thresholds are those written in the lines. The seventh and sixth runs' figures that the lines name are read from their saved files.

**The rule that decides what the Result may claim.** Lines 2, 3, 4, 5, 6 and 7 all read yes: the figures of the cells are claimed as read. Drift is claimed from lines 8 and 9; the cost of keeping the chat in speed, time, tokens and energy from lines 10 to 15; the cost of the grammar from line 16; what the bigger model does with names from lines 17 to 20, and what the host reader adds from line 19; a statement about time between the two expressions from line 21. Line 2, 3, 4, 5, 6 or 7 reads NO: no claim is made from this run until the cause is found. Energy is claimed for a cell only if the meter read it (line 3).

**How the result will be worded.** On the sixth run's first 32 jobs (the first 4 for the kept chat), on one NVIDIA L4, one 14B model (`Qwen/Qwen2.5-14B-Instruct-AWQ`) and vLLM 0.31.0, in one Colab session: the exact share over 100 steps for the seven cells, the seconds per step, the computed prompt tokens per step, the exact steps per GPU hour and the board's joules per exact step, with the 7B figures of the earlier runs beside them and the setup beside both; and in the same sentence that the 7B figures are from a Tesla T4 in other sessions, that the energy is the board's and not the host's, that the names are the table's (given, not learned), and that the sentences are those of the seventh page and not requests in the words of a user. Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never "100 percent" for a grammar, never "proven" (one model, one GPU, one run), never a statement about any company, never a statement about the words of a user, never a statement about a model or hardware that was not run (the TPU is not run). A "NO" on a claim line is a finding about this setup and goes into the article as such.

## What this does not show
Hardware other than one L4 (the v6e-1 TPU and any bigger GPU are not run); a model other than this 14B; the energy of the host, the cooling or the building; tasks of more than 100 steps, let alone days; the words of a real user (the sentences are those of the seventh page, in two pools); names the host did not give; a language other than English; a kept chat with a summary or a window; a retry on top of a grammar; real company tasks; a capacity in jobs per GPU; dollars; hallucination in free text. Nothing here is proven.

## How it is run
Open `python/colab/axi_provider_side_big.ipynb` in Colab with an **L4** runtime and run the cells one at a time: it gets the code from GitHub, installs vLLM 0.31.0, connects Drive, fixes the output stream, runs the **pre-flight check** (it also reads the power meter), loads the 14B model (about 9.4 GB to download), runs the safety cell, then the seven cells in order. Each finished cell is written as `axi_provider_b_<label>.json` and copied to Drive; a cell that was cut off is run again from its start. The last cell puts the cells together, prints the table, the split by pool and by what the host read, the first misses and the 21 lines, and offers the files for download. The Result goes below the line at the end of this page, in a later update with a test of its own.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_big.py` and again by `tests/test_provider_side_big_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_big.ipynb`: 16,578 steps in seven cells, 41.0 minutes of generation, vLLM 0.31.0, `Qwen/Qwen2.5-14B-Instruct-AWQ`, NVIDIA L4, a cache of 32,000 tokens. The merged file is `docs/results/axi_provider_big.json` (sha256 `b7ce85cb772685db...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*

**What the page expected, and what came out.** Lines 1 to 7 all read yes, so by the page's rule the figures of the cells are claimed as read. Of the 21 lines **12 read yes and 9 read NO** (9, 11, 13, 14, 15, 17, 18, 19 and 20); no line is "not run". **Headline:** line 8 reads yes (the named expression held the state on every step on the bigger model) and **line 18 reads NO** (the bigger model did not read the names by itself well enough). The page put "yes, yes" at about 22 %. Lines 11, 13, 14 and 15 are near misses on the page's thresholds, in the direction I expected; they are said with their numbers below and are not rounded up.

**In one paragraph, from the side of a company that runs the model.** On the sixth run's first 32 jobs, with the exact state held by the host and the grammar naming the request's coordinates, a 14B model on one L4 held the state on 3,200 of 3,200 steps, in 0.110 s per step, with 97 computed prompt tokens per step, 32,750 exact steps per GPU hour and 7.9 joules of board energy per exact step, and the seconds per step did not grow over the 100 steps (0.88 times from the first quarter to the last). The same model under the capped expression held 3,038 of 3,200 (94.9 %), no better than the 7B model on the same jobs (3,075); with the names list added to the prompt it held 2,689 (84.0 %) where the 7B held 3,200. **In every one of the 78 first slips of the 14B in the four cells in which a job slipped, the reply contained an item the oracle's reply does not have (78 of 78); in none did it only lack an item.** The bigger model does more than it was asked, and the named expression is the part that forbids exactly that. Given requests in ordinary words, the 14B alone held 47.8 %; with the host reader reading what it can, 90.1 %, the reader read 2,688 requests right, 0 wrong and refused 512. The kept chat did not finish a 100-step job: all four stopped at steps 27, 61, 50, 40, at 0.487 s per step against 0.269 for the exact state on the same four jobs.

## The 21 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 99 % | cached tokens reported on all 16,578 requests (5,086,016 of 7,286,791 prompt tokens served from the cache) | yes |
| 2 | 88 % | 16,577 of 16,578 replies accepted by the strict parser (99.99 %); in `PN`, `P4` and `PC` none of 3,778 replies used all 96 new tokens and all are in the language of their own request | yes |
| 3 | 90 % | every cell read between 100.1 % and 100.5 % of the readings its seconds allow (4,927 readings in all) | yes |
| 4 | 99 % | all 6,400 sentences of `XW` and `XR` name the services of the oracle's request in its order, with its numbers, and the second pool names exactly one service more (checked by plain search, not by the reader) | yes |
| 5 | 99 % | the recorded sha256 of every cell equals the one recomputed from its jobs; `PN` 656, `P4` 198, `PC` 115, `XR` 575 expressions; the oracle's reply is inside the expression of every request | yes |
| 6 | 95 % | all 164 jobs of the six exact-state cells ran 100 steps | yes |
| 7 | 99 % | the reader, run again on the recorded sentences, gives the recorded outcome at all 3,200 requests: right 2,688, wrong 0, not read 512 | yes |
| 8 | 88 % | `PN` 100.0 % (3,200 of 3,200 steps; 32 of 32 jobs exact to the end) | yes |
| 9 | 80 % | `P0` 94.9 % (3,038 of 3,200 steps) against 3,075 for the 7B model's `L0` on the same jobs | NO |
| 10 | 95 % | all four jobs of `PC` stopped before step 100, at steps 27, 61, 50, 40, because the next prompt would have passed 7,600 tokens | yes |
| 11 | 80 % | `P4` 13,402 exact steps per GPU hour, `PC` 7,386: 1.81 times | NO |
| 12 | 75 % | `PN` last quarter 0.1087 s per step, first quarter 0.1235: 0.88 times | yes |
| 13 | 80 % | `PC` last quarter 0.6994 s per step, first quarter 0.4833: 1.45 times | NO |
| 14 | 80 % | `PC` 155.2 computed prompt tokens per step, `P4` 125.4: 1.24 times | NO |
| 15 | 75 % | `PC` 35.0 joules per exact step, `P4` 19.3: 1.82 times | NO |
| 16 | 75 % | `PN` 0.1099 s per step against `P0` 0.1123 (-2.1 %); `PN` 7.89 joules per exact step against `P0` 8.49 | yes |
| 17 | 80 % | `X0` 84.0 % (2,689 of 3,200 steps; 13 of 32 jobs exact to the end) | NO |
| 18 | 25 % | `XW` 47.8 % (1,528 of 3,200 steps; 0 of 32 jobs exact to the end) | NO |
| 19 | 15 % | `XR` 90.1 % (2,884 of 3,200 steps; 15 of 32 jobs exact to the end) | NO |
| 20 | 55 % | `XW` second pool 40.9 % of 765 steps, first pool 49.9 % of 2,435: 9.0 points lower | NO |
| 21 | 90 % | `XR` 0.1600 s per step, `XW` 0.1723: -7.1 % | yes |

Lines 11, 13, 14 and 15 compare the kept chat with the exact state: all four ratios point the way the page expected (1.81, 1.45, 1.24 and 1.82 against thresholds of 2, 2, 1.3 and 2) and none reached its threshold; on the 7B model the page's figures for time per step and exact steps per hour were 3.8 times. Line 9 reads NO by 37 steps. Line 20 reads NO by 6.0 points more than the three the page allowed.

## The cells

| cell | what it is | steps | exact steps | exact share | jobs exact to the end | seconds per step | computed prompt tokens per step | exact steps per GPU hour | mean board watts | joules per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| `P0` | exact state, coordinates, capped expression | 3,200 | 3,038 | 94.9 % | 22 of 32 | 0.112 | 97.4 | 30,447 | 71.6 | 8.49 |
| `PN` | exact state, coordinates, named expression | 3,200 | 3,200 | 100.0 % | 32 of 32 | 0.110 | 97.1 | 32,750 | 71.6 | 7.89 |
| `P4` | exact state, named, first 4 jobs | 400 | 400 | 100.0 % | 4 of 4 | 0.269 | 125.4 | 13,402 | 71.7 | 19.28 |
| `PC` | whole chat kept, named, first 4 jobs | 178 | 178 | 100.0 % | 4 of 4 | 0.487 | 155.2 | 7,386 | 71.7 | 35.02 |
| `X0` | coordinates + names list, capped | 3,200 | 2,689 | 84.0 % | 13 of 32 | 0.153 | 156.8 | 19,776 | 71.8 | 13.09 |
| `XW` | words, capped | 3,200 | 1,528 | 47.8 % | 0 of 32 | 0.172 | 157.4 | 9,978 | 71.8 | 25.93 |
| `XR` | words, expression built by the host reader | 3,200 | 2,884 | 90.1 % | 15 of 32 | 0.160 | 154.8 | 20,275 | 71.7 | 12.76 |

Exact share and exact steps per hour count every step after a slip that stayed in the held state as a miss; they are the page's measures and they are used as written. The board watts are the L4's board power read by `nvidia-smi` every half second; the joules are those readings added up with the time between them over the cell's window, divided by the cell's exact steps. The meter reads the board, not the host.

| cell | prompt tokens computed + new tokens written | exact steps | exact steps per 1,000 tokens | seconds per step, first quarter | last quarter | round time p50 / p95 / longest (s) |
|---|---|---|---|---|---|---|
| `P0` | 355,339 | 3,038 | 8.5 | 0.1246 | 0.1126 | 3.59 / 4.76 / 5.92 |
| `PN` | 354,209 | 3,200 | 9.0 | 0.1235 | 0.1087 | 3.56 / 4.45 / 5.85 |
| `P4` | 55,509 | 400 | 7.2 | 0.3171 | 0.2653 | 0.92 / 2.15 / 2.41 |
| `PC` | 30,914 | 178 | 5.8 | 0.4833 | 0.6994 | 1.25 / 2.57 / 2.86 |
| `X0` | 545,847 | 2,689 | 4.9 | 0.1716 | 0.1509 | 4.82 / 6.33 / 8.79 |
| `XW` | 559,606 | 1,528 | 2.7 | 0.1773 | 0.1746 | 5.34 / 6.74 / 8.79 |
| `XR` | 546,011 | 2,884 | 5.3 | 0.1724 | 0.1617 | 5.00 / 6.53 / 8.33 |

## Drift: where the first slips are

The first inexact step of a job is the clean measure, because until then the held state was exact. `P0`: 10 of 32 jobs slipped (median first slip at step 26); `PN`: 0; `X0`: 19 (median step 24); `XW`: 32 (median step 7); `XR`: 17 (median step 34). The table counts, for each kind of request, the first slips and the steps that were run exact up to and including the first slip.

| cell | request kind | pool | host read | first slips | steps run before the first slip | rate |
|---|---|---|---|---|---|---|
| `P0` | add_edge | - | - | 0 | 844 | 0.0 % |
| `P0` | add_node | - | - | 0 | 847 | 0.0 % |
| `P0` | delete_dep | - | - | 8 | 593 | 1.3 % |
| `P0` | delete_leaf | - | - | 2 | 284 | 0.7 % |
| `PN` | add_edge | - | - | 0 | 1,039 | 0.0 % |
| `PN` | add_node | - | - | 0 | 1,069 | 0.0 % |
| `PN` | delete_dep | - | - | 0 | 738 | 0.0 % |
| `PN` | delete_leaf | - | - | 0 | 354 | 0.0 % |
| `X0` | add_edge | - | - | 0 | 608 | 0.0 % |
| `X0` | add_node | - | - | 0 | 623 | 0.0 % |
| `X0` | delete_dep | - | - | 17 | 455 | 3.7 % |
| `X0` | delete_leaf | - | - | 2 | 204 | 1.0 % |
| `XW` | add_edge | first | - | 4 | 98 | 4.1 % |
| `XW` | add_edge | second | - | 2 | 34 | 5.9 % |
| `XW` | add_node | first | - | 0 | 73 | 0.0 % |
| `XW` | add_node | second | - | 15 | 27 | 55.6 % |
| `XW` | delete_dep | first | - | 7 | 75 | 9.3 % |
| `XW` | delete_dep | second | - | 1 | 31 | 3.2 % |
| `XW` | delete_leaf | first | - | 3 | 25 | 12.0 % |
| `XW` | delete_leaf | second | - | 0 | 6 | 0.0 % |
| `XR` | add_edge | first | right | 0 | 567 | 0.0 % |
| `XR` | add_edge | second | miss | 10 | 159 | 6.3 % |
| `XR` | add_node | first | right | 0 | 553 | 0.0 % |
| `XR` | add_node | second | right | 0 | 179 | 0.0 % |
| `XR` | delete_dep | first | right | 0 | 370 | 0.0 % |
| `XR` | delete_dep | second | miss | 5 | 140 | 3.6 % |
| `XR` | delete_leaf | first | right | 0 | 190 | 0.0 % |
| `XR` | delete_leaf | second | miss | 2 | 58 | 3.4 % |

* **The bigger model slips by doing more than it was asked.** Of the 78 first slips in `P0`, `X0`, `XW` and `XR`, 78 contain an item the oracle's reply does not have and 0 lack an item without having an extra one; 53 contain every item of the oracle's reply and more. In `P0`, 6 of 10 slips are deletes that include a `DEL` of an edge that does not touch the deleted service and 5 of 10 include an `ADD` item in a delete; in `X0`, 13 of 19 and 12 of 19. The named expression allows a delete to begin with the named service and to delete only edges that touch it, and an add to add only what the request names: `PN` slipped 0 times in 3,200 steps.
* **The capped expression is where the size of the model does not help.** `P0` 3,038 against the 7B model's 3,075 on the same jobs (`L0`); `X0` 2,689 against 3,200 (`C0`). The slips are all on deletes (`P0` 8 of 593 delete steps with dependents, 2 on a leaf; `X0` 17 and 2), none on an add. One model of each size, one session each: this says the 14B was not more exact than the 7B under the capped expression on these jobs, nothing about 14B models in general.
* **In ordinary words (`XW`), the 14B model reads edges and loses on a service it is asked to add.** First slips on `add_edge` requests: 6 of 32; on `add_node` requests: 15 of 32, at 15.0 % per step run; in all 15 of the 15 the extra items are `ADD` edges at the new service's coordinates, and in 13 of the 15 one of them reaches the neighbour the sentence mentions ("it will sit next to ..."): the model wired the new service to a service it was only told about. The rest of the first slips are spread over deletes (11) and edges. The second pool of wordings (a sentence that names one service more than the oracle's request) gives 40.9 % exact steps against 49.9 % in the first pool. The 7B model's weak point in the seventh run was the edge; the first slips of the 14B model are mostly on the add of a service in the second pool and on deletes.
* **The host reader is safe where it reads.** In `XR`, 0 of the 1,859 steps whose request the host read right was a first slip; all 17 first slips are on requests the host refused (357 such steps run), at 4.8 % per step; all are in the second pool. Because a slip stays in the held state, the exact share of the steps whose request the host read is 90.5 %, and of the others 88.1 %; the first-slip figures are the clean ones.

## The serving numbers, as a company that runs the model counts them

The page's measures are the seconds per step (and the exact steps per GPU hour that follow from it), the computed prompt tokens per step, the latency of a step within a job and of a round, and the board energy per exact step.

| | model, card | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |
|---|---|---|---|---|---|
| sixth run `L0` (capped, 64 jobs) | 7B, Tesla T4 | 0.100 | 94.6 | 331.0 | 35,113 |
| sixth run `LN` (named, 64 jobs) | 7B, Tesla T4 | 0.101 | 94.9 | 331.2 | 35,764 |
| this run `P0` (capped, 32 jobs) | 14B, L4 | 0.112 | 97.4 | 333.8 | 30,447 |
| this run `PN` (named, 32 jobs) | 14B, L4 | 0.110 | 97.1 | 333.4 | 32,750 |

The sixth run's cells were run in another Colab session on another card with a model half the size, so the table shows two settings and not an effect of the card or of the model; the size of any difference cannot be separated.

* **Speed.** `PN` ran 0.1099 s per step and `P0` 0.1123 (-2.1 %) in this session, with 656 different expressions in the first and one in the second: building an expression for each request cost no seconds that this run can see (line 16). `PN` did 32,750 exact steps per GPU hour and `P0` 30,447; the grammar's cost is not visible and its gain is 162 exact steps of 3,200.
* **Latency.** Within a job: the seconds per step of the last quarter of the rounds are 0.88 times those of the first quarter in `PN` (line 12 reads yes), 0.90 in `P0`. A round (all 32 requests together) took 3.56 s at the median, 4.45 s at the 95th percentile and 5.85 s at the longest in `PN`; the four-job cell `P4` 0.92 / 2.15 / 2.41 s. **This vLLM version reported no per-request times** (first token, finish): 0 of 16,578 requests carry one, so no request-level latency is claimed; the round times are the latency a lockstep job sees.
* **Tokens.** `PN` computed 97.1 prompt tokens per step (70.9 % of the prompt tokens were served from the cache) and wrote 13.6; the names list and the sentences put 60 more computed prompt tokens in each step of `XW` (62 %). Per **exact** step the difference is larger because the wrong steps are paid for: `PN` 9.0 exact steps per 1,000 tokens, `XR` 5.3, `XW` 2.7.
* **Energy.** The mean board power was 71.6 to 71.8 W in all seven cells (the highest single readings 74.1 to 81.6 W); the L4's rated board power is 72 W. The card ran at about its rated power throughout, so on this card in this run the joules per step are the seconds per step times about 72 W (7.89 J per step in `PN` at 0.110 s) and energy told nothing that time did not tell: it is a measurement of the board and not of the host or the building. Per exact step: `PN` 7.89 J, `XR` 12.76, `XW` 25.93, `PC` 35.02, `P4` 19.28.

## The kept chat against the exact state (lines 10 to 15)

The kept chat did not reach step 100 in any of the four jobs (line 10 reads yes): they stopped at steps 27, 61, 50, 40, the longest prompt was 7,579 tokens, against 521 for the exact state in all 100 steps. On the same four jobs and the first 27 rounds, when all four were running in both cells, the kept chat took 0.480 s per step against 0.323 (1.49 times) with 2,962 prompt tokens per step against 360; over their whole runs 0.487 against 0.269 (the kept chat's first 27 to 61 steps against all 100 steps of the exact state, so the whole-run figures cover different steps). The prefix cache served 96.2 % of its prompt tokens. The page's four ratios (time, growth, tokens, energy) all point the way it expected and none reaches its threshold; the 7B figures on the sixth run were larger, on another card. What this run shows is that on an L4 with a 14B model the kept chat is slower, heavier and, within an 8,192-token context, cannot finish a 100-step job; it does not show a ratio for longer prompts or other cache sizes.

## What the page's rule lets this Result claim

Lines 1 to 7 read yes, so the figures of the cells are claimed as read, for this setup: one 14B model, one L4, one vLLM version, one session, 32 jobs (4 for the kept chat), the sentences of the page and the table's names. Drift is claimed from lines 8 and 9 (`PN` held; `P0` did not do better than the 7B), the kept chat from lines 10 to 15, the cost of the grammar from line 16 (none visible), what the bigger model does with names from lines 17 to 20 (it does not read them well enough alone), what the host reader adds from line 19 (90.1 %, not 99), and the time of the host reader from line 21. The names are the table's (given, not learned); the host reader reads a sentence only where one service is named for a delete, two for an edge, or an added service with its coordinates, and the sentences are those of the page and not requests in the words of a user. The energy is the board's.

## The predictions written beside the lines, read afterwards

| written first | probability | measured | came out |
|---|---|---|---|
| (a) more than half of the first slips of `XW` are on edge requests | 70 % | 6 of 32 first slips are on `add_edge` requests; 15 are on `add_node` | NO |
| (b) `XW` holds the right state on between 60 and 90 percent of its 3,200 steps | 55 % | 47.8 % | NO |
| (c) the seconds per step of `PN` on the L4 are between 0.12 and 0.30 | 65 % | 0.1099 | NO |
| (d) the seven cells take 60 to 120 minutes in all, loading excluded | 60 % | 41.0 minutes | NO |
| (e) the mean board power of `PN` is between 50 and 72 watts | 70 % | 71.63 W | yes |
| (f) this vLLM version reports request times (first token, finish) in the cells | 40 % | 0 requests with a time | NO |

Side predictions (a), (b), (c), (d) and (f) did not come out; (e) did. The seven cells took 41.0 minutes of generation, much less than the 60 to 120 I wrote.

## What this means, and what it does not

For a company that runs the model, the serving side did what it was built for on a bigger model: the state was held on every step by the named expression (3,200 of 3,200), at a speed, a token count and an energy per step that did not depend on the grammar, flat over the job, on an L4. The size of the model did not remove the slips by itself: under the capped expression it did no better than the 7B model, and every slip was the model doing more than asked, which is what the named expression forbids. The kept chat cost more per step and could not finish a 100-step job in an 8,192-token context. The new finding about the front of the pipeline is that the 14B model, given a sentence, holds the state on more steps than the 7B (47.8 % against 42.0 % on the same jobs) but still does more than it is asked, for example by wiring a new service to a neighbour the sentence only mentions, and that the host reader lifts it to 90.1 % (the 7B: 63.4 %) without ever reading wrong; the remaining loss is on the 512 requests the reader refuses. The energy measurement is honest and, on this card, adds nothing to the time.

## What this does not show
Hardware other than one L4 (the v6e-1 TPU and any bigger GPU were not run); a model other than this 14B; the energy of the host, the cooling or the building; request-level latency (not reported by this version); tasks of more than 100 steps, let alone days; the words of a real user (the sentences are those of the seventh page, in two pools); names the host did not give; a language other than English; a kept chat with a summary or a window; a retry on top of a grammar; real company tasks; a capacity in jobs per GPU; dollars; hallucination in free text. Nothing here is proven.

## What comes next
The options, not commitments. (1) A reader for the second pool, so that the sentence's extra clause is ignored by the host and not by the model: the first slips of `XW` and `XR` are in that pool. (2) The same cells with more jobs side by side on the L4, to see where the round time starts to grow. (3) A probe on a v6e-1 TPU with its own page. (4) A longer horizon than 100 steps. The cost in compute units is told before anything is run.

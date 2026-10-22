# Test 22: the long session on Gemini 3.8 Flash, heavy turns, alone and with AXI (written BEFORE the run, 21 Oct 2026)

*Concept-map item 115. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
AXI is built for the companies that run models. What such a company counts is: does the model **drift or hallucinate** over a long task, how fast is a step (**speed**), how long does a request wait (**latency**), how many tokens are computed for each step (**tokens**), and how much energy a step takes. The earlier Gemini run (the hard test, 4,000 steps) showed that with the framework the state stays exact; it did **not** measure what a company gets if it simply **keeps the chat** against the API, heavy turns included. This test does, up to the model's window of about one million tokens.

## What is run
One model, **`gemini-3.8-flash`** (the id is taken from the list the key can see), streamed, the model's own default thinking, no retry other than the re-ask of the AXI arm. The request stream is the hard test's dense service graph (`million_hard.HardStream`), seed 20261111. **Every request comes with a heavy turn**: a block of 40 lines of routine operations log (timestamps, pod and request ids, status codes), the same text in every arm, with no coordinates, no graph and no rule in it. The pilot prints its real size in tokens.

| arm | what it is |
|---|---|
| `K`, alone | one chat: the instructions, the first state, then every turn and every reply kept and sent again; each reply applied as it comes (blind apply, no re-ask, no repair). It stops after 700 turns or when its next prompt would pass 950,000 tokens. |
| `A`, with AXI | the framework holds the exact state (`scope_m7`, up to 2 re-asks, the repair deleting up to 8 dependants) and shows it at every step with the same heavy turn; nothing is kept in the chat; 1,500 steps. |
| `D`, depth probe | a chat whose history is the **ideal** replies of turns 1 to s-1 (no call for them), then 10 real turns, for s in 50, 100, 200, 300, 350. If the model slips here, depth alone is the cause; if not, the slips of `K` come from its own earlier mistakes in the chat. |

The code is `python/axi/experiments/gemini_long.py`; it imports `million.py`, `million_hard.py`, `api_llm.py`, `agent_loop.py`, `cascade.py`, `long_horizon.py` and the framework and **edits none of them**. The notebook is `python/colab/axi_gemini_long.ipynb`.

## What is measured, for every call
Prompt tokens, tokens the service served from its cache (`cached_content_token_count`), reply and thinking tokens, **seconds in all**, and **seconds to the first token** (the calls are streamed). From these: **speed** = steps per hour of calls and **exact steps per hour of calls** (by prompt size); **latency** = seconds per call and seconds to the first token, median and 95th percentile (by prompt size, and whether they grow with the chat); **tokens** = prompt tokens, cached share, tokens computed (prompt not served from the cache plus reply plus thinking); **drift** = share of replies that give exactly the oracle's next state, first turn that does not, and the distance of the state a firm would hold; **hallucination** = share of replies that cannot be applied to the true state (they name a service that is not there or leave an edge dangling).

## What I read before writing this page
The hard test on Gemini 3.8 Flash (4,000 steps, exact state at every step, 100 percent exact in every tenth; thinking about 160 tokens a step; blind application corrupted the state on 55.8 percent of the first 400 steps) and the 14B serving test (a bigger model alone did not drift less; the kept chat ratios were below the thresholds). No kept chat has been run against the API before.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 check that the run is the run I describe; lines 7 to 20 are about what was found.

1. [guard] the kept chat (arm K) ended either at 700 turns or because its next prompt would pass 950,000 tokens, with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn (85 %)
2. [guard] the arm with AXI (arm A) reached step 1,500 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask (90 %)
3. [guard] the kept chat grew to a prompt of at least 500,000 tokens (80 %)
4. [instrument] the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn (75 %)
5. [instrument] at most 0.5 percent of the steps of arm A ended with a reply the strict parser refused (after the re-asks) (95 %)
6. [instrument] at least 98 percent of the replies of arm K were accepted by the strict parser (90 %)
7. [claim] arm A: the held state equals the oracle's on at least 98 percent of all steps and on at least 95 percent in every tenth (85 %)
8. [claim] arm A: the held state was never corrupted and never more than 4 items from the oracle's (80 %)
9. [claim] arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the first tenth and the mean seconds a step at most 1.5 times (80 %)
10. [claim] arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the oracle's next state (75 %)
11. [claim] arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50 (55 %)
12. [claim] arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true state (they name something that is not there or leave an edge dangling) (50 %)
13. [claim] arm K: the state a firm would hold by applying the replies as they come was corrupted (an edge pointing at a missing service) at some turn (55 %)
14. [claim] arm K: the first reply that is not exactly right comes before turn 50 (40 %)
15. [claim] latency: the mean seconds of a call of arm K at a prompt of 500,000 tokens or more are at least 3 times the mean seconds of a step of arm A (85 %)
16. [claim] latency to the first token: the median seconds to the first token of arm K at a prompt of 500,000 tokens or more are at least 3 times those of arm A (70 %)
17. [claim] speed: the exact steps per hour of calls of arm A are at least 2 times those of arm K, over the same number of steps (the first as many steps of A as K has turns) (90 %)
18. [claim] tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 500,000 tokens or more (85 %)
19. [claim] arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50 (50 %)
20. [claim] depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest (40 %)

## Reading rules (fixed before the run)
1. the kept chat (arm K) ended either at 700 turns or because its next prompt would pass 950,000 tokens, with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn
2. the arm with AXI (arm A) reached step 1,500 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask
3. the kept chat grew to a prompt of at least 500,000 tokens
4. the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn
5. at most 0.5 percent of the steps of arm A ended with a reply the strict parser refused (after the re-asks)
6. at least 98 percent of the replies of arm K were accepted by the strict parser
7. arm A: the held state equals the oracle's on at least 98 percent of all steps and on at least 95 percent in every tenth
8. arm A: the held state was never corrupted and never more than 4 items from the oracle's
9. arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the first tenth and the mean seconds a step at most 1.5 times
10. arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the oracle's next state
11. arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50
12. arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true state (they name something that is not there or leave an edge dangling)
13. arm K: the state a firm would hold by applying the replies as they come was corrupted (an edge pointing at a missing service) at some turn
14. arm K: the first reply that is not exactly right comes before turn 50
15. latency: the mean seconds of a call of arm K at a prompt of 500,000 tokens or more are at least 3 times the mean seconds of a step of arm A
16. latency to the first token: the median seconds to the first token of arm K at a prompt of 500,000 tokens or more are at least 3 times those of arm A
17. speed: the exact steps per hour of calls of arm A are at least 2 times those of arm K, over the same number of steps (the first as many steps of A as K has turns)
18. tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 500,000 tokens or more
19. arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50
20. depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest

Headline: lines 7, 8 and 11 all read yes. If a guard or instrument line (1 to 6) does not read yes, no figure is claimed until the cause is found. Line 4 reading NO means no computed-token figure is claimed (only prompt tokens). A line that was not run claims nothing. Arm `K` is read only as far as it got: if it stops at the window, that is its result, not a failure of the test.

## What this will not show
- **Energy in joules.** The API gives no joules for Google's hardware. Computed tokens and seconds are a stand-in, and the page says so wherever it uses them.
- **Latency inside the service** (queueing, the time before the model starts) apart from what the streamed call shows: seconds to the first token includes the network from the notebook's machine.
- **Anything about another company's service, or a real user's words.** The requests are generated; the model is one hosted model on one day.
- **That the result holds for other tasks.** It is one job (a service graph) with log noise; the claim is about this job.
- **Cost in dollars.** The prices are typed in by the owner at run time from the price page. No dollar figure is claimed on this page.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (20 real turns of each arm) with the real size of a heavy turn, the speed, the latency and the projected cost of the whole test at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** stops each run (8,000,000 for `A`, 400,000,000 for `K`, 60,000,000 for `D`). Everything finished is saved and the same cell carries on.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_long.py` and again by `tests/test_gemini_long_result.py`). The figures are recomputed from `docs/results/axi_gemini_long_session_seed20261111_default.json` by that test; none is typed by hand. One run of each arm, one seed, one job, on 22 Oct 2026.*

**What the page expected, and what came out.** Lines 1 to 6 (the guards and the instruments) all read yes, so by the page's rule the figures of the arms are claimed as read. Of the 20 lines **17 read yes and 3 read NO** (lines 11, 14, 20). Headline: **MIXED: line 7 yes, line 8 yes, line 11 NO**.

**In one paragraph, for a company that runs the model.** With AXI the exact state held on all 1,500 steps (100.0 %), at a flat prompt of about 2,824 tokens a step and 3.07 s a step. The kept chat gave the exact reply on every one of its first 222 turns, then slipped on 15 of 457 turns; every slip came at a prompt of 463,161 tokens or more. A company that applies each reply as it comes held a wrong state on every turn from turn 287 on (the first slip, at turn 223, changed nothing in the held state), and a corrupted one (an edge pointing at a missing service) at 146 of those turns. Over the same job a call at a prompt of 500,000 tokens or more took 3.5 times the seconds of a step with AXI and 4.0 times the time to the first token.

## The 20 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 85 % | [guard] 457 turns; stopped because the next prompt would pass 950,000 tokens; the replay of the saved replies reproduced every flag | **yes** |
| 2 | 90 % | [guard] 1,500 steps; the replay reproduced every flag and every re-ask (0 re-asks) | **yes** |
| 3 | 80 % | [guard] largest prompt 949,591 tokens | **yes** |
| 4 | 75 % | [instrument] 99.6 % of the calls from the 10th turn had cached tokens; 99.1 % of all prompt tokens | **yes** |
| 5 | 95 % | [instrument] 0 steps refused after the re-asks, of 1,500 | **yes** |
| 6 | 90 % | [instrument] 455 of 457 replies accepted (99.6 %) | **yes** |
| 7 | 85 % | [claim] held state exact on 1,500 of 1,500 steps (100.0 %); lowest tenth 100.0 % | **yes** |
| 8 | 80 % | [claim] corrupted 0 steps; largest distance from the oracle's state 0 | **yes** |
| 9 | 80 % | [claim] mean prompt of the last tenth 2,698 against 2,499 in the first (1.08 times); seconds a step 3.21 against 2.90 (1.11 times) | **yes** |
| 10 | 75 % | [claim] turns 1 to 50: 100.0 % exact | **yes** |
| 11 | 55 % | [claim] turns 1 to 50: 100.0 % exact; last 100 turns: 91.0 % exact; a drop of 9.0 points | **NO** |
| 12 | 50 % | [claim] last 100 turns: 9.0 % of the replies cannot be applied to the true state | **yes** |
| 13 | 55 % | [claim] the held state was corrupted at 146 of 457 turns; the first at turn 287 (prompt 596,245) | **yes** |
| 14 | 40 % | [claim] the first reply that is not exactly right is turn 223 (prompt 463,161) | **NO** |
| 15 | 85 % | [claim] 10.66 s a call at a prompt of 500,000 tokens or more against 3.07 s a step with AXI (3.47 times) | **yes** |
| 16 | 70 % | [claim] first token, median: 9.13 s against 2.27 s (4.02 times) | **yes** |
| 17 | 90 % | [claim] exact steps an hour of calls: AXI 1,280, kept chat 488 (the first 457 steps of each; 2.62 times) | **yes** |
| 18 | 85 % | [claim] tokens computed for each exact step at a prompt of 500,000 tokens or more: kept chat 5,419; with AXI 3,108 (1.74 times) | **yes** |
| 19 | 50 % | [claim] thinking tokens a turn: last 100 turns 239, turns 1 to 50 77 (3.10 times) | **yes** |
| 20 | 40 % | [claim] exact share at the shallowest start (50): 100.0 %; at the deepest (350): 100.0 % | **NO** |

## Speed, latency and tokens

| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| kept chat, all turns | 457 | 7.13 | 505 | 488 | 5.06 | 13.98 | 476,835 | 99.1 % | 160 | 4,728 |
| kept chat, prompt of 500,000 tokens or more | 217 | 10.66 | 338 | 316 | 9.13 | 17.49 | 725,061 | 99.3 % | 279 | 5,419 |
| with AXI, all steps | 1500 | 3.07 | 1,171 | 1,157 | 2.27 | 6.17 | 2,824 | 0.0 % | 226 | 3,108 |
| with AXI, the first 457 steps | 457 | 2.79 | 1,292 | 1,280 | 2.21 | 5.80 | 2,779 | 0.0 % | 198 | 3,023 |

The seconds are the seconds of the calls. The waits that the key's per-minute limit made the notebook take (they began around turn 70 of the kept chat and came with every few turns later) are not in them, so they say how long a call takes, not how long the job took. Latency is read at the notebook: the time to the first token includes the network from the notebook's machine. Tokens: *computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens; the page's stand-in for energy, **not joules**. The API gives no joules for the hardware.

## The kept chat, by the size of the prompt

| prompt size (tokens) | turns | n | exact reply | held state exact | seconds a call | first token median | first token 95th | exact steps an hour | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|
| 0 to 100,000 | 1 to 47 | 47 | 100.0 % | 100.0 % | 3.65 | 2.23 | 7.28 | 987 | 3,540 |
| 100,000 to 200,000 | 48 to 96 | 49 | 100.0 % | 100.0 % | 3.88 | 2.45 | 13.21 | 928 | 2,128 |
| 200,000 to 300,000 | 97 to 144 | 48 | 100.0 % | 100.0 % | 3.85 | 2.51 | 7.97 | 936 | 2,184 |
| 300,000 to 400,000 | 145 to 192 | 48 | 100.0 % | 100.0 % | 3.07 | 2.72 | 4.34 | 1,174 | 2,139 |
| 400,000 to 500,000 | 193 to 240 | 48 | 97.9 % | 100.0 % | 5.20 | 4.02 | 16.74 | 677 | 10,858 |
| 500,000 to 600,000 | 241 to 288 | 48 | 97.9 % | 95.8 % | 6.83 | 5.54 | 10.05 | 516 | 14,646 |
| 600,000 to 700,000 | 289 to 336 | 48 | 93.8 % | 0.0 % | 10.16 | 7.44 | 43.07 | 332 | 2,926 |
| 700,000 to 800,000 | 337 to 385 | 49 | 91.8 % | 0.0 % | 10.47 | 9.31 | 11.19 | 316 | 2,457 |
| 800,000 to 900,000 | 386 to 433 | 48 | 93.8 % | 0.0 % | 12.06 | 11.66 | 16.23 | 280 | 2,250 |
| 900,000 to 1,000,000 | 434 to 457 | 24 | 87.5 % | 0.0 % | 16.90 | 13.28 | 50.28 | 186 | 3,250 |

* **Held state** is what a firm holds if it applies each reply as it comes (blind application). One reply that cannot be applied to the true state, or that is applied wrong, leaves the held state wrong for the rest of the job; that is why the held-state column reads 0 percent from turn 289 on while the exact-reply column stays between 87.5 % and 93.8 %.
* **The slips.** Turns 223, 287, 322, 328, 329, 355, 362, 376, 381, 398, 419, 424, 436, 441, 453: 15 replies in all that were not exactly the oracle's, 2 of them not readable by the strict parser (turns 328, 329); 14 of the 15 came at a prompt of 500,000 tokens or more.
* **Thinking.** The kept chat's thinking tokens a turn were 77 over turns 1 to 50 and 239 over the last 100; with AXI 226 a step. It did not stay constant.

## With AXI, by tenths of the job

| steps | held state exact | prompt tokens a step | seconds a step | first token median |
|---|---|---|---|---|
| 1 to 150 | 100.0 % | 2,499 | 2.90 | 2.33 |
| 151 to 300 | 100.0 % | 2,865 | 2.60 | 2.15 |
| 301 to 450 | 100.0 % | 2,961 | 2.86 | 2.19 |
| 451 to 600 | 100.0 % | 2,950 | 3.19 | 2.29 |
| 601 to 750 | 100.0 % | 2,736 | 4.07 | 2.35 |
| 751 to 900 | 100.0 % | 2,950 | 2.97 | 2.28 |
| 901 to 1050 | 100.0 % | 2,869 | 2.95 | 2.31 |
| 1051 to 1200 | 100.0 % | 2,841 | 3.06 | 2.15 |
| 1201 to 1350 | 100.0 % | 2,870 | 2.95 | 2.29 |
| 1351 to 1500 | 100.0 % | 2,698 | 3.21 | 2.33 |

18 of the 1,500 replies were not exactly the oracle's (the replies of this arm are checked against the true state before the step); none changed the held state, which stayed exact on every step: the scope check and the repair of the framework did what they are for. There were no re-asks and no refused replies.

## The depth probe (a perfect history)

| start (turn) | real turns | mean prompt (tokens) | exact reply | seconds a call | first token median |
|---|---|---|---|---|---|
| 50 | 10 | 113,502 | 100.0 % | 4.80 | 3.53 |
| 100 | 10 | 217,184 | 100.0 % | 3.38 | 3.13 |
| 200 | 10 | 424,747 | 100.0 % | 6.20 | 3.82 |
| 300 | 10 | 632,508 | 100.0 % | 7.14 | 6.62 |
| 350 | 10 | 736,340 | 100.0 % | 9.06 | 8.85 |

Ten real turns at each of five depths, after a history made of the oracle's own replies. All 50 replies were exact, at prompts up to 745,664 tokens. The kept chat's slips began at 463,161 tokens. With ten turns a depth this cannot tell a small rate of slips from none (line 20 reads NO under the page's rule); it is consistent with the slips of the kept chat not being the length of the chat alone, and it does not show it.

## What this says, and what it does not

* **Said:** on this job, with heavy turns, the model alone gave exact replies for about 222 turns, then slipped now and then, and a company that applies its replies as they come lost the state for good at turn 287; with the framework holding the state the state was exact on every one of 1,500 steps. The call time and the time to the first token of the kept chat grew with its prompt (2.23 s at the first band, 13.28 s at the last) while those of the AXI arm did not.
* **Not said:** that the slips come from the length of the chat (the depth probe is too small to say it); anything about another model, another job, another seed or any company's service; joules; the time the job took with the waits. The kept chat's first 222 turns were exact, so for a job of a hundred or so heavy turns the kept chat, on this run, was as exact as AXI; the finding is about what happens after.
* **The reading rule** of the page is used as written: line 11 reads NO because the drop in exact replies is 9.0 points against the 10 the page set, and line 14 reads NO because the first slip came at turn 223, not before turn 50. The headline is MIXED: line 7 yes, line 8 yes, line 11 NO, as the rule gives it.

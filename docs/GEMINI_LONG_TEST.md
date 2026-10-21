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
| `D`, depth probe | a chat whose history is the **ideal** replies of turns 1 to s-1 (no call for them), then 10 real turns, for s in 50, 150, 250, 350, 450. If the model slips here, depth alone is the cause; if not, the slips of `K` come from its own earlier mistakes in the chat. |

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

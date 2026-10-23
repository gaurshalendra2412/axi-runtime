# Test 23: a long conversation in plain sentences on Gemini 3.8 Flash, alone and with AXI (written BEFORE the run, 22 Oct 2026)

*Concept-map item 116. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
Test 22 (the long session) kept a state that is a **dependency graph**: discrete, rigid, easy to check. A fair objection is that this is the easy case for a framework that holds the state, and that real conversational work has a state that is **ambiguous text**: facts in many wordings, corrections, loose references ("the last one we added", "the same day as the caterer"), and a lot of talk around them that changes nothing. This test puts the same question to that kind of state, with the same counts a company that runs the model makes: **drift and hallucination, distance from the true state, speed, latency, tokens and the energy stand-in**, with the state held by the framework against the chat kept.

## What is run
One model, **`gemini-3.8-flash`** (the id is taken from the list the key can see), streamed, the model's own default thinking. A script writes a conversation of 400 turns from a hidden **true list** of items for a small events desk (an item has a name and up to four facts: owner, day, amount, status). The text is in ordinary sentences, in several wordings for every kind of message, and **every turn comes with a paragraph of 30 sentences of office chatter** that mentions other days, other amounts and the same first names and changes nothing (the same text in every arm, never an item name, never a request; the pilot prints its real size in tokens). The kinds of message: a new item, one fact changed, two facts changed, a **correction** ("scratch what I said about X: the day is Friday, not Monday"), a **loose reference** to the last item added, to the item a first name is handling, or to "the same day / the same cost as" another item, and an item dropped. **Every tenth turn from the 20th is a question** about the list: about facts changed less than 10, 10 to 49, 50 to 149 and 150 or more turns earlier; about a fact **never given** (the right answer is `unknown`); and about "the item a first name is handling" when two items fit (the right answer is `ambiguous`). Every reply is marked by an **exact match with what the script knows**, never by another model's opinion. The seed is 20261122.

| arm | what it is |
|---|---|
| `K`, alone | one chat: the instructions, then every turn and every reply kept and sent again; each reply applied as it comes (blind apply: no check, a set on an item that is not there creates it, an odd value is stored as given, no re-ask). It stops after 400 turns or when its next prompt would pass 950,000 tokens. |
| `A`, with AXI | the framework holds the exact list and shows it at every step, in the order the items were added, with the same chatter; nothing is kept in the chat. A reply goes through a **check** before it touches the list (the item must be there for a change and absent for an add; the field and the value must be in the allowed set); up to 2 re-asks when it fails; a reply that still fails changes nothing. 400 steps. |
| `D`, depth probe | a chat whose history is the **ideal** replies of turns 1 to s-1 (no call for them), then 10 real turns, for s in 50, 100, 200, 300. If the model slips here, depth alone is the cause; if not, the slips of `K` come from its own earlier mistakes in the chat. |

The two system texts differ in one sentence only: arm `K` is told the list starts empty and that it must keep it from the conversation; arm `A` is told the current list is shown and exact. The code is `python/axi/experiments/gemini_fuzzy.py`; it imports `gemini_long.py` (the chat backend and the counts) and `api_llm.py` and **edits none of them**; none of the framework's core is used or changed (the check of arm `A` is written for this list in the test module). The notebook is `python/colab/axi_gemini_fuzzy.ipynb`.

## What is measured, for every call
Prompt tokens, tokens the service served from its cache, reply and thinking tokens, **seconds in all**, and **seconds to the first token**. From these: **speed** (steps and exact steps per hour of calls), **latency** (seconds per call and to the first token, median and 95th percentile, by prompt size), **tokens** (prompt, cached share, computed = not served from the cache plus reply plus thinking), **drift** (share of replies that are exactly right, first turn that is not, share of the questions answered right **by the age of the fact**), **hallucination** (share of replies that cannot be applied to the true list), and the **distance** of the held list from the true list (cells that differ: an item in one list only counts 1 plus its facts; for an item in both, each fact that differs or is missing). The results are also cut **by kind of message**, so that a wording or a type of reference that fails shows up. A list is **corrupted** if it holds an item the conversation never mentioned, an unknown field, or a value outside the allowed set.

## What I read before writing this page
Test 22 on Gemini 3.8 Flash (heavy turns on a dependency graph): with AXI the state was exact on all 1,500 steps and the kept chat was exact until turn 222, slipped from turn 223 and held a wrong state from turn 287, with latency and speed degrading as the prompt grew. Its depth probe, with a perfect history, was exact everywhere, which leaves the question of this test open for ordinary text: here a mistake is easier to make and the framework's check can only refuse what is not allowed (an item that is not there, a value outside the set); it cannot know that a *valid* day is the wrong one. So a distance above zero is possible with AXI on this test, and the page allows it.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 check that the run is the run I describe; lines 7 to 24 are about what was found.

1. [guard] the kept chat (arm K) reached turn 400 (or stopped because its next prompt would pass 950,000 tokens) with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn (90 %)
2. [guard] the arm with AXI (arm A) reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask (90 %)
3. [guard] the kept chat grew to a prompt of at least 150,000 tokens (85 %)
4. [instrument] the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn (85 %)
5. [instrument] at most 0.5 percent of the steps of arm A ended with a reply that could not be used (after the re-asks) (60 %)
6. [instrument] at least 98 percent of the replies of arm K had the right shape (one JSON object with ops for a message, answer for a question) (95 %)
7. [claim] arm A: the held list equals the true list on at least 95 percent of all steps and on at least 90 percent in every tenth (65 %)
8. [claim] arm A: the held list was never corrupted (an item nobody mentioned, an unknown field or a value outside the allowed set) and never more than 3 cells from the true list (55 %)
9. [claim] arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the second tenth and the mean seconds a step at most 1.5 times (80 %)
10. [claim] arm A: at least 95 percent of the answers to the questions are exactly right (70 %)
11. [claim] arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the true next list (or the right answer) (70 %)
12. [claim] arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50 (45 %)
13. [claim] arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true list (they name an item that is not there, or a value that is not allowed) or have the wrong shape (40 %)
14. [claim] arm K: the list a firm would hold by applying the replies as they come was corrupted at some turn (55 %)
15. [claim] arm K: the first reply that is not exactly right comes before turn 50 (65 %)
16. [claim] arm K forgets with age: its share of exact answers to questions about facts last changed 50 or more turns earlier is at least 10 points below that for facts changed less than 10 turns earlier (40 %)
17. [claim] arm A does not forget with age: its share of exact answers about facts last changed 50 or more turns earlier is within 5 points of that for facts changed less than 10 turns earlier (70 %)
18. [claim] ask, not guess: on the questions whose right answer is unknown or ambiguous, arm A answers exactly at least 80 percent of the time and at least 10 points more often than arm K (50 %)
19. [claim] latency: the mean seconds of a call of arm K at a prompt of 150,000 tokens or more are at least 1.5 times the mean seconds of a step of arm A (85 %)
20. [claim] latency to the first token: the median seconds to the first token of arm K at a prompt of 150,000 tokens or more are at least 1.5 times those of arm A (80 %)
21. [claim] speed: the exact steps per hour of calls of arm A are at least 1.5 times those of arm K, over the same number of steps (85 %)
22. [claim] tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 150,000 tokens or more (80 %)
23. [claim] arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50 (25 %)
24. [claim] depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest (15 %)

## Reading rules (fixed before the run)
1. the kept chat (arm K) reached turn 400 (or stopped because its next prompt would pass 950,000 tokens) with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn
2. the arm with AXI (arm A) reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask
3. the kept chat grew to a prompt of at least 150,000 tokens
4. the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn
5. at most 0.5 percent of the steps of arm A ended with a reply that could not be used (after the re-asks)
6. at least 98 percent of the replies of arm K had the right shape (one JSON object with ops for a message, answer for a question)
7. arm A: the held list equals the true list on at least 95 percent of all steps and on at least 90 percent in every tenth
8. arm A: the held list was never corrupted (an item nobody mentioned, an unknown field or a value outside the allowed set) and never more than 3 cells from the true list
9. arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the second tenth and the mean seconds a step at most 1.5 times
10. arm A: at least 95 percent of the answers to the questions are exactly right
11. arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the true next list (or the right answer)
12. arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50
13. arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true list (they name an item that is not there, or a value that is not allowed) or have the wrong shape
14. arm K: the list a firm would hold by applying the replies as they come was corrupted at some turn
15. arm K: the first reply that is not exactly right comes before turn 50
16. arm K forgets with age: its share of exact answers to questions about facts last changed 50 or more turns earlier is at least 10 points below that for facts changed less than 10 turns earlier
17. arm A does not forget with age: its share of exact answers about facts last changed 50 or more turns earlier is within 5 points of that for facts changed less than 10 turns earlier
18. ask, not guess: on the questions whose right answer is unknown or ambiguous, arm A answers exactly at least 80 percent of the time and at least 10 points more often than arm K
19. latency: the mean seconds of a call of arm K at a prompt of 150,000 tokens or more are at least 1.5 times the mean seconds of a step of arm A
20. latency to the first token: the median seconds to the first token of arm K at a prompt of 150,000 tokens or more are at least 1.5 times those of arm A
21. speed: the exact steps per hour of calls of arm A are at least 1.5 times those of arm K, over the same number of steps
22. tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 150,000 tokens or more
23. arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50
24. depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest

Headline: lines 7 and 8 both read yes ("AXI HOLDS ON FUZZY TEXT"); otherwise the headline is MIXED and names the line that failed. The comparison with the kept chat is read from lines 12 to 18 and is not part of the headline. If a guard or instrument line (1 to 6) does not read yes, no figure is claimed until the cause is found. Line 4 reading NO means no computed-token figure is claimed (only prompt tokens). A line that was not run claims nothing. Arm `K` is read only as far as it got: if it stops at the window, that is its result, not a failure of the test.

## What this will not show
- **Energy in joules.** The API gives no joules for Google's hardware. Computed tokens and seconds are a stand-in, and the page says so wherever it uses them.
- **Latency inside the service** apart from what the streamed call shows (seconds to the first token includes the network from the notebook's machine).
- **Anything about another company's service, or a real user's words.** The conversation is written by a script; the model is one hosted model on one day.
- **That the result holds for other kinds of text.** It is one kind of list (an events desk) with chatter; the claim is about this job.
- **A list whose items are free text.** Here the facts come from small allowed sets, so an exact match can mark them; a state of free-form sentences is a harder case this test does not cover.
- **Cost in dollars.** The prices are typed in by the owner at run time. No dollar figure is claimed on this page.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (20 real turns of each arm) with the real size of a turn, the speed, the latency and the projected cost of the whole test at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** stops each run (3,000,000 for `A`, 150,000,000 for `K`, 20,000,000 for `D`). Everything finished is saved and the same cell carries on.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_fuzzy.py` and again by `tests/test_arc_fuzzy_results.py`). The figures are recomputed from `docs/results/axi_gemini_fuzzy_seed20261122_default.json` by that test; none is typed by hand. One run of each arm, one seed, one job, on 22 Oct 2026.*

**What the page expected, and what came out.** Lines 1 to 6 (the guards and the instruments) all read yes, so by the page's rule the figures of the arms are claimed as read. Of the 24 lines **15 read yes and 9 read NO** (lines 9, 12, 13, 14, 15, 16, 18, 21, 24). Headline: **AXI HOLDS ON FUZZY TEXT**. The NO lines are findings, and most of them are the kept chat *not* doing what the page guessed it might: at 198,405 tokens and 400 turns of ambiguous text the kept chat did not drift and did not hallucinate, and its answers about old facts were within the page's 10-point threshold of those about new ones.

**In one paragraph, for a company that runs the model.** With AXI the held list was exact on all 400 steps (100.0 %), at 1,177 prompt tokens a step on average (946 in the first tenth, 1,332 in the last, as the list grew) and 2.76 s a step. The kept chat sent 99,809 prompt tokens a turn on average (85 times as many; 39.9 million tokens in all against 0.47 million) and its replies were exact on 398 of 400 turns (99.5 %). The two slips mattered differently: the slip at turn 191 (`ref_last`: the reply changed the wrong item) was a change to the list, and a company that applies each reply as it comes **held a wrong list for 140 turns in a row, turns 191 to 330**, until a later message happened to overwrite the cell; the list was exact on 65.0 % of the turns although 99.5 % of the replies were exact. The slip at turn 370 was a wrong answer to a question and changed nothing. On time, the kept chat went from 1.86 s to the first token (turns 1 to 50) to 5.57 s once the prompt passed 150,000 tokens, against 2.41 s with AXI (2.31 times), and from 2.24 s to 5.70 s a call; at a short prompt the kept chat was the faster of the two.

## The 24 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 90 % | [guard] 400 turns, every one on file; the replay of the saved replies reproduced every flag | **yes** |
| 2 | 90 % | [guard] 400 steps; the replay reproduced every flag and every re-ask (0 re-asks) | **yes** |
| 3 | 85 % | [guard] largest prompt 198,405 tokens | **yes** |
| 4 | 85 % | [instrument] 94.4 % of the calls from the 10th turn had cached tokens; 97.0 % of all prompt tokens | **yes** |
| 5 | 60 % | [instrument] 0 steps refused after the re-asks, of 400 | **yes** |
| 6 | 95 % | [instrument] 400 of 400 replies had the right shape (100.0 %) | **yes** |
| 7 | 65 % | [claim] held list exact on 400 of 400 steps (100.0 %); lowest tenth 100.0 % | **yes** |
| 8 | 55 % | [claim] corrupted 0 steps; largest distance from the true list 0 cells | **yes** |
| 9 | 80 % | [claim] mean prompt of the last tenth 1,332 against 979 in the second (1.36 times); seconds a step 2.69 against 2.38 (1.13 times) | **NO** |
| 10 | 70 % | [claim] 39 of 39 answers exact (100.0 %) | **yes** |
| 11 | 70 % | [claim] turns 1 to 50: 100.0 % exact | **yes** |
| 12 | 45 % | [claim] turns 1 to 50: 100.0 % exact; last 100 turns: 99.0 %; a drop of 1.0 points | **NO** |
| 13 | 40 % | [claim] last 100 turns: 0.0 % of the replies cannot be applied to the true list or have the wrong shape | **NO** |
| 14 | 55 % | [claim] the held list was corrupted (an item nobody mentioned, an unknown field or a value outside the allowed set) at 0 of 400 turns | **NO** |
| 15 | 65 % | [claim] the first reply that is not exactly right is turn 191 (prompt 95,262) | **NO** |
| 16 | 40 % | [claim] exact answers about facts changed less than 10 turns earlier: 100.0 % (n 8); 50 or more turns earlier: 92.9 % (n 14) | **NO** |
| 17 | 70 % | [claim] arm A: 100.0 % (n 8) against 100.0 % (n 14) | **yes** |
| 18 | 50 % | [claim] answers exact where the right answer is unknown or ambiguous: arm A 100.0 % (n 7); arm K 100.0 % (n 7) | **NO** |
| 19 | 85 % | [claim] 5.70 s a call at a prompt of 150,000 tokens or more against 2.76 s a step with AXI (2.07 times) | **yes** |
| 20 | 80 % | [claim] first token, median: 5.57 s against 2.41 s (2.31 times) | **yes** |
| 21 | 85 % | [claim] exact steps an hour of calls: AXI 1,306, kept chat 984 (the same 400 steps; 1.33 times) | **NO** |
| 22 | 80 % | [claim] tokens computed for each exact step at a prompt of 150,000 tokens or more: kept chat 2,576; with AXI 1,322 (1.95 times) | **yes** |
| 23 | 25 % | [claim] thinking tokens a turn: last 100 turns 11.2, turns 1 to 50 1.8 | **yes** |
| 24 | 15 % | [claim] exact share at the shallowest start (50): 100.0 %; at the deepest (300): 100.0 % | **NO** |

## Speed, latency and tokens

| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| kept chat, all turns | 400 | 3.64 | 989 | 984 | 2.93 | 6.55 | 99,809 | 97.0 % | 3.3 | 3,035 |
| kept chat, turns 1 to 50 | 50 | 2.24 | 1,609 | 1,609 | 1.86 | 3.71 | 13,094 | 51.8 % | 1.8 | 6,338 |
| kept chat, prompt of 150,000 tokens or more | 99 | 5.70 | 631 | 625 | 5.57 | 7.43 | 174,384 | 98.6 % | 11.3 | 2,576 |
| with AXI, all steps | 400 | 2.76 | 1,306 | 1,306 | 2.41 | 3.75 | 1,177 | 0.0 % | 111.5 | 1,322 |
| with AXI, steps 1 to 50 | 50 | 3.19 | 1,129 | 1,129 | 2.35 | 3.62 | 959 | 0.0 % | 109.3 | 1,102 |

The seconds are the seconds of the calls, read at the notebook (the time to the first token includes the network from the notebook's machine). Tokens: *computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens; the page's stand-in for energy, **not joules**. The API gives no joules for the hardware. The kept chat was served 97.0 % of its prompt tokens from the service's cache, so its *computed* tokens are far fewer than its *sent* tokens (39,936,914 sent, 1,207,926 computed); AXI, which sends a short fresh prompt every step, got no cache (528,836 sent and 528,836 computed). In computed tokens the kept chat is therefore **2.30 times** AXI per exact step over the whole job, not the 76 times of the tokens sent.

## The kept chat, by the size of the prompt

| prompt size (tokens) | turns | n | exact reply | held list exact | seconds a call | first token median | first token 95th | exact steps an hour | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|
| 0 to 25,000 | 1 to 49 | 49 | 100.0 % | 100.0 % | 2.21 | 1.86 | 3.71 | 1,628 | 6,416 |
| 25,000 to 50,000 | 50 to 99 | 50 | 100.0 % | 100.0 % | 2.59 | 2.13 | 4.94 | 1,387 | 2,542 |
| 50,000 to 100,000 | 100 to 200 | 101 | 99.0 % | 90.1 % | 2.87 | 2.45 | 4.87 | 1,240 | 2,570 |
| 100,000 to 150,000 | 201 to 301 | 101 | 100.0 % | 0.0 % | 3.60 | 3.21 | 5.02 | 1,000 | 2,544 |
| 150,000 to 200,000 | 302 to 400 | 99 | 99.0 % | 70.7 % | 5.70 | 5.57 | 7.43 | 625 | 2,576 |

* **Held list** is what a firm holds if it applies each reply as it comes (blind application). A reply that changes the wrong cell leaves it wrong until a later message overwrites that cell: here from turn 191 to turn 330 (140 turns), at most 2 cells from the true list. It was not *corrupted* in the page's sense (no item nobody mentioned, no unknown field, no value outside the allowed set): it was a valid list that was not the true one.
* **Crossing.** From the tenth of turns 121 to 160 on (prompts around 70,112 tokens) every tenth of the kept chat took more seconds a turn than the mean seconds a step of AXI (2.76); before that the kept chat was the faster arm or about level, in the first 50 turns by 1.43 times in exact steps an hour.
* **Thinking.** With AXI the model thought 112 tokens a step on average (the model's own default thinking); the kept chat 3.3 a turn on average, 1.8 over turns 1 to 50 and 11.2 over the last 100. Line 23 reads yes only because the early value is tiny; it is not a climb that a company would plan around.

## By tenths of the job

| turns | kept chat: exact reply | kept chat: held list exact | kept chat: prompt a turn | kept chat: seconds a turn | AXI: held list exact | AXI: prompt a step | AXI: seconds a step | AXI: first token median |
|---|---|---|---|---|---|---|---|---|
| 1 to 40 | 100.0 % | 100.0 % | 10,600 | 1.98 | 100.0 % | 946 | 3.39 | 2.46 |
| 41 to 80 | 100.0 % | 100.0 % | 30,416 | 2.82 | 100.0 % | 979 | 2.38 | 2.25 |
| 81 to 120 | 100.0 % | 100.0 % | 50,246 | 2.44 | 100.0 % | 1,040 | 2.84 | 2.59 |
| 121 to 160 | 100.0 % | 100.0 % | 70,112 | 3.05 | 100.0 % | 1,033 | 2.80 | 2.65 |
| 161 to 200 | 97.5 % | 75.0 % | 89,978 | 2.92 | 100.0 % | 1,149 | 2.98 | 2.77 |
| 201 to 240 | 100.0 % | 0.0 % | 109,674 | 3.20 | 100.0 % | 1,276 | 2.72 | 2.52 |
| 241 to 280 | 100.0 % | 0.0 % | 129,526 | 3.71 | 100.0 % | 1,335 | 2.47 | 2.29 |
| 281 to 320 | 100.0 % | 0.0 % | 149,423 | 4.40 | 100.0 % | 1,340 | 2.74 | 2.50 |
| 321 to 360 | 100.0 % | 75.0 % | 169,306 | 5.51 | 100.0 % | 1,339 | 2.57 | 2.32 |
| 361 to 400 | 97.5 % | 100.0 % | 188,806 | 6.39 | 100.0 % | 1,332 | 2.69 | 2.41 |

## By kind of turn

| kind of turn | n | kept chat: exact reply | kept chat: held list exact | AXI: exact reply |
|---|---|---|---|---|
| add | 67 | 100.0 % | 65.7 % | 100.0 % |
| correct | 24 | 100.0 % | 54.2 % | 100.0 % |
| delete | 41 | 100.0 % | 65.9 % | 100.0 % |
| query:ambiguous | 2 | 100.0 % | 50.0 % | 100.0 % |
| query:recall | 32 | 96.9 % | 65.6 % | 100.0 % |
| query:unknown | 5 | 100.0 % | 60.0 % | 100.0 % |
| ref_last | 48 | 97.9 % | 58.3 % | 100.0 % |
| ref_owner | 23 | 100.0 % | 69.6 % | 100.0 % |
| ref_same | 24 | 100.0 % | 70.8 % | 100.0 % |
| set | 102 | 100.0 % | 69.6 % | 100.0 % |
| set2 | 32 | 100.0 % | 59.4 % | 100.0 % |

0 of the 400 replies of AXI were not exactly the oracle's; none changed the held list, which stayed exact on every step (the checks and the repair of the framework did what they are for); there were no re-asks and no refused replies.

## The depth probe (a perfect history)

| start (turn) | real turns | mean prompt (tokens) | exact reply | seconds a call | first token median |
|---|---|---|---|---|---|
| 50 | 0 | 27,476 | 100.0 % | - | - |
| 100 | 0 | 52,346 | 100.0 % | - | - |
| 200 | 10 | 101,875 | 100.0 % | 58.13 | 3.17 |
| 300 | 10 | 151,459 | 100.0 % | 5.84 | 4.88 |

Ten turns at each of four depths, after a history made of the oracle's own replies (40 replies, all exact; the first two depths were answered from the notebook's saved reply file and have no call time). With ten turns a depth this cannot tell a small rate of slips from none (line 24 reads NO under the page's rule).

## What this says, and what it does not

* **Said:** on this job, ambiguous text for 400 turns, the model alone gave exact replies on 398 of 400 turns, so on this job the kept chat did not fail at this length; what the run shows is that the **one** wrong change that did come stayed in the state for as long as no later message overwrote it (140 turns), because a state kept in a chat has no check against the truth. With the framework holding the state the list was exact on every one of 400 steps, at a prompt that grew only with the list, and a step that took about the same time all through (between 2.38 s and 3.39 s in every tenth). The time to the first token of the kept chat grew with its prompt (1.86 s at the first band, 5.57 s at the last) while that of the AXI arm did not.
* **Not said:** that the kept chat drifts, hallucinates or forgets with age (it did none of these beyond the page's thresholds here: lines 12, 13, 14, 15 and 16 read NO); that AXI is faster at every length (it was not at a short prompt, and line 21 reads NO: 1.33 times the exact steps an hour, under the 1.5 the page asked); that the two slips would repeat on another seed; anything about another model, another job or any company's service; joules.
* **The reading rule** of the page is used as written. The headline is AXI HOLDS ON FUZZY TEXT, as the rule gives it (lines 7 and 8 read yes). Line 9 reads NO because the prompt of AXI grew 1.36 times from the second tenth to the last (the list it shows grows with the job) against the 1.3 the page allowed, while the kept chat's grew 6.2 times.

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

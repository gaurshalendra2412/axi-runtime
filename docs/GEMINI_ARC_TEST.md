# Test 24: Gemini 3.8 Flash on ARC-AGI-3 games, alone and with AXI (written BEFORE the run, 22 Oct 2026)

*Concept-map item 117. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
Tests 22 and 23 asked what a firm that runs a model gets from a state held by the framework, on a dependency graph and on a conversation in plain sentences. ARC-AGI-3 is a different kind of ground: small interactive puzzle games whose rules are **not told** to the player, played one move at a time on a 64 x 64 screen. The model has to find the rules by playing, over many moves, and a model that keeps the chat sends every earlier screen again at every move. This test counts what a company counts on such a job (**tokens, latency, speed, wasted moves, illegal moves, loops**) with the same model alone and with AXI.

**What this test is not.** It is not a measure of how well the model works out new games (that is the games' own question, and AXI makes no claim to answer it better), and it is not an entry on the ARC-AGI-3 leaderboard: the leaderboard's rules, hidden games and scorecards are not used. A model like this one may complete no level at all in 60 moves, with or without AXI; the page says so beforehand, and lines 13 to 15 are written to read that honestly.

## What is run
One model, **`gemini-3.8-flash`**, streamed, the model's own default thinking. The public ARC-AGI-3 toolkit (`arc-agi`, `arcengine`) runs the games on the notebook's machine (a step takes under a millisecond; the games and their rules come from the toolkit, seed 0). Four games, the ones the free probe notebook played on 22 Oct: `ls20-9607627b`, `g50t-5849a774`, `ka59-38d34dbb`, `sp80-589a99af`. **60 moves per game**, one call of the model per move, and the same number of moves in both arms. A game ends early only if it is won. The screen is shown as 64 rows of 64 hex digits, one per cell, colours 0 to f; the turn also gives the state (`NOT_FINISHED`, `GAME_OVER`, `WIN`), the levels completed and the actions the game offers now. A move is `{"action": "ACTION1"}` or, for a click, `{"action": "ACTION6", "x": 10, "y": 20}`, or `RESET`.

| arm | what it is |
|---|---|
| `K`, alone | one chat per game: the instructions, then every screen and every reply of the game kept and sent again at every move. A reply is played as it comes; a move that is not one the game offers (or not a move at all) is not played and still uses a move (blind: no re-ask). |
| `A`, with AXI | nothing is kept in the chat. Every move the model gets the current screen and a **Record kept by the framework, computed exactly from the game's own screens and never asked of the model**: the actions tried so far (times tried, times the screen changed, mean cells changed), the clicks that changed the screen and the clicks that did nothing, the last six moves (what each did), how many times this exact screen has been seen and which moves already did nothing from it, and the levels completed. A move goes through a **check** before it reaches the game (the action must be one the game offers, a click must be inside the grid), with up to 2 re-asks; a move that still fails is not played and still uses a move. |

The two system texts differ in one sentence: arm `K` is told that the earlier turns stay in the chat; arm `A` is told that they are not and that each turn starts with a Record kept by the framework. The record's content is the framework's side of the bargain: whether the model *uses* it is what is measured. The code is `python/axi/experiments/gemini_arc.py`; it imports `gemini_long.py` and `api_llm.py` and **edits none of them**, imports no ARC package (the notebook hands it the toolkit's `make`, so the tests can use a stand-in game), and uses none of the framework's core.

## What is measured
For every call: prompt tokens, tokens the service served from its cache, reply and thinking tokens, **seconds in all**, **seconds to the first token**. For every move: whether it was **legal** (a move the game offered, or a click inside the grid), **effective** (legal and the screen changed), a **repeated no-op** (a move that already did nothing from the very same screen), a **revisit** (a screen seen earlier in the game), and the state and levels completed afterwards. From these: **tokens** (prompt, cached share, computed = not served from the cache plus reply plus thinking), **speed** (effective moves per hour of calls), **latency** (seconds per call and to the first token, by prompt size), **steadiness** (the share of effective moves early and late in a game), and **levels**. The energy stand-in is computed tokens and seconds; the API gives no joules.

## What I read before writing this page
The probe notebook (free, no model) showed 25 public games, 8 actions (`RESET`, `ACTION1` to `ACTION7`; `ACTION6` is the click), and a step of the game itself of 0.4 to 1.0 millisecond; 30 random moves completed no level and ended one game in `GAME_OVER`. The toolkit's source gives a frame as a list of 64 x 64 arrays (a move can return several animation frames; the last is used) and a step as `step(action, data)`. Tests 22 and 23 are the same discipline for the two earlier kinds of state; this page does not use any result of Test 23.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 check that the run is the run I describe; lines 7 to 19 are about what was found.

1. [guard] the kept chat (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies into fresh copies of the games reproduces every saved flag of every move (90 %)
2. [guard] the arm with AXI (arm A) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag and every re-ask (85 %)
3. [guard] the kept chat grew to a prompt of at least 100,000 tokens in at least one game (65 %)
4. [instrument] the service reported cached tokens for at least 80 percent of the calls of arm K that had a prompt of 10,000 tokens or more (85 %)
5. [instrument] at most 2 percent of the moves of arm A ended with a reply that could not be used (after the re-asks) (85 %)
6. [instrument] at least 95 percent of the replies of arm K were one JSON object with an action (90 %)
7. [claim] arm A: the mean prompt tokens of moves 41 to 60 are at most 1.5 times those of moves 6 to 25, over the games (80 %)
8. [claim] tokens sent (prompt, reply and thinking): arm A sent at most one third of what arm K sent, over the same number of moves (75 %)
9. [claim] arm K drifts: its share of effective moves (a move the game offered that changed the screen) over moves 41 to 60 is at least 5 points below that over moves 1 to 20 (30 %)
10. [claim] memory of what did nothing: arm A repeated a move that had already done nothing from the same screen at most half as often as arm K, and arm K did so on at least 2 percent of its moves (55 %)
11. [claim] arm K plays illegal moves: at least 1 percent of its replies are not one of the actions the game offered (or a click outside the grid, or not a move at all) (40 %)
12. [claim] arm A: its share of effective moves is at least 5 points above that of arm K (45 %)
13. [claim] levels: arm A completed at least as many levels as arm K, over the games (read it together with lines 14 and 15: if both completed none, this line reads yes without meaning anything) (80 %)
14. [claim] arm A completed at least one level in at least one game (25 %)
15. [claim] arm K completed at least one level in at least one game (25 %)
16. [claim] latency: the mean seconds of a call of arm K at a prompt of 100,000 tokens or more are at least 1.5 times the mean seconds of a move of arm A (55 %)
17. [claim] latency to the first token: the median seconds to the first token of arm K at a prompt of 100,000 tokens or more are at least 1.5 times those of arm A (50 %)
18. [claim] speed: the effective moves per hour of calls of arm A are at least 1.5 times those of arm K (60 %)
19. [claim] tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per effective move as arm A, over the moves of arm K at a prompt of 100,000 tokens or more (70 %)

## Reading rules (fixed before the run)
1. the kept chat (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies into fresh copies of the games reproduces every saved flag of every move
2. the arm with AXI (arm A) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag and every re-ask
3. the kept chat grew to a prompt of at least 100,000 tokens in at least one game
4. the service reported cached tokens for at least 80 percent of the calls of arm K that had a prompt of 10,000 tokens or more
5. at most 2 percent of the moves of arm A ended with a reply that could not be used (after the re-asks)
6. at least 95 percent of the replies of arm K were one JSON object with an action
7. arm A: the mean prompt tokens of moves 41 to 60 are at most 1.5 times those of moves 6 to 25, over the games
8. tokens sent (prompt, reply and thinking): arm A sent at most one third of what arm K sent, over the same number of moves
9. arm K drifts: its share of effective moves (a move the game offered that changed the screen) over moves 41 to 60 is at least 5 points below that over moves 1 to 20
10. memory of what did nothing: arm A repeated a move that had already done nothing from the same screen at most half as often as arm K, and arm K did so on at least 2 percent of its moves
11. arm K plays illegal moves: at least 1 percent of its replies are not one of the actions the game offered (or a click outside the grid, or not a move at all)
12. arm A: its share of effective moves is at least 5 points above that of arm K
13. levels: arm A completed at least as many levels as arm K, over the games (read it together with lines 14 and 15: if both completed none, this line reads yes without meaning anything)
14. arm A completed at least one level in at least one game
15. arm K completed at least one level in at least one game
16. latency: the mean seconds of a call of arm K at a prompt of 100,000 tokens or more are at least 1.5 times the mean seconds of a move of arm A
17. latency to the first token: the median seconds to the first token of arm K at a prompt of 100,000 tokens or more are at least 1.5 times those of arm A
18. speed: the effective moves per hour of calls of arm A are at least 1.5 times those of arm K
19. tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per effective move as arm A, over the moves of arm K at a prompt of 100,000 tokens or more

Headline: lines 8 and 13 both read yes ("AXI SENDS FAR FEWER TOKENS AND LOSES NO LEVELS"); otherwise MIXED. Line 13 is read together with lines 14 and 15: if both arms complete no level, line 13 reads yes and means nothing, and the page says so in its Result. If a guard or instrument line (1 to 6) does not read yes, no figure is claimed until the cause is found. Line 4 reading NO means no computed-token figure is claimed (only prompt tokens). A line that was not run claims nothing.

## What this will not show
- **How well the model solves new games**, or how AXI compares on that: 60 moves a game is far below what a solver would use, and a score of zero levels is a likely outcome for both arms.
- **The leaderboard's score.** The toolkit's official scorecard is not used as a result here; the levels counted are the games' own `levels_completed`.
- **Energy in joules.** The API gives none. Computed tokens and seconds are a stand-in.
- **Other games, other models, other move budgets, or any other company's service.** Four games, one model, one day.
- **That the Record is the best possible one.** It is one fixed record, written before the run; a different record could do better or worse.
- **Cost in dollars.** The prices are typed in by the owner at run time. No dollar figure is claimed on this page.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (12 real moves of each arm in the first game) with the real size of a screen in tokens, the speed, the latency and the projected cost of the whole test at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** stops each run (6,000,000 for `A`, 120,000,000 for `K`). Everything finished is saved and the same cell carries on: the games are played again from the saved moves, so a stopped run resumes exactly.

# Test 25: Gemini 3.8 Flash on ARC-AGI-3 games with a host-owned explorer, against arms A and K of Test 24 (written BEFORE the run, 23 Oct 2026)

*Concept-map item 118. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
Test 24 played four public ARC-AGI-3 games with the same model in two ways: alone (the kept chat, arm K) and with AXI's exact Record and check (arm A). Arm A played 240 moves, every one of them a move the game offered, and completed one level (in one of the four games). It also showed what a Record alone does not do: in one game it sent the same action 34 times out of 60, in two games some moves changed nothing (12 of 240), and it spent about 3,900 thinking tokens and about 29 seconds on a move. The question of this test is the one the owner asked after Test 24: **what if the framework owns more of the exploring?**

The owner shared notes naming further structural layers of the architecture (an invariant address frame, typed operators, an outer-loop conductor, a progress score Q = O/M, and boundary completion for grid objects). Whether those names and pieces match Sir's own writing is for Sir to confirm; this page does not claim it. The notes were written for a task in which a model edits a grid toward a known target. ARC-AGI-3 is not that task: the model can only send the actions a game offers, no target grid is given and the rules are not told. So this test takes the parts that **fit the real actions of the games** and leaves the rest:

| idea in the notes | what it is here |
|---|---|
| invariant address frame | **the object view**: the framework lists the objects on the screen (same-colour cells that touch: colour, size, box, one cell inside it, at most 12) and the exact box of what the last move changed |
| outer-loop conductor | **the conductor**: when the last 4 moves reached no screen not seen before, or this exact screen has been seen 3 times, the framework itself plays the first move not yet tried from this exact screen (ACTION1, 2, 3, 4, 5, 7 in this order, then a click on an object's inside cell), with no call to the model |
| untried moves | **the untried list**: the offered actions not yet tried from this screen, and up to 6 objects whose inside cell has not been clicked from it |
| progress score Q = O/M | **a reading only**: screens never seen before that a move reached, per 1,000 computed tokens. It decides nothing |
| typed operators (translate, fill, mirror), boundary completion | **not used**: the games have no such actions; the nearest thing, the check and the two re-asks, is already in arm A |

Arm B keeps the Record, the check and the two re-asks of arm A exactly as they were. The framework's core and the code of Test 24 are not changed. The model's thinking is left at its default, so arm B differs from arm A only by the three additions.

## What is run
One model, **`gemini-3.8-flash`**, streamed, default thinking. The same four public games as Test 24 (`ls20-9607627b`, `g50t-5849a774`, `ka59-38d34dbb`, `sp80-589a99af`), 60 moves each, the same game seed. **Only arm B is run.** Arms A and K are the rows Test 24 saved on the owner's Drive; the notebook replays them into fresh copies of the games without any call to the model. Arm K of Test 24 had not been run when this page was written: lines 16 and 17 are read only if its rows are on file.

The system text of arm B is the system text of arm A with its last sentence extended by two: each turn also holds the Objects and the Untried moves computed by the framework, and the framework itself plays an untried move when the model has been stuck (that move is in the Record, marked "(framework)"). A move the framework plays costs no tokens and no seconds; it is a move in the game like any other and is counted in every share and rate below.

The code is `python/axi/experiments/gemini_arc_b.py`; it imports `gemini_arc.py`, `gemini_long.py` and `api_llm.py` and **edits none of them**. The notebook is `python/colab/axi_gemini_arc_b.ipynb`.

## What is measured
As in Test 24: for every call prompt tokens, tokens served from the cache, reply and thinking tokens, seconds in all and seconds to the first token; for every move whether it was legal, whether it changed the screen (effective), whether it repeated a move that had already done nothing from the same screen, whether it reached a screen seen before, and the levels. New here: whether a move was played by the framework, and **screens never seen before** (a move that changed the screen to one not seen before in the game).

## What I read before writing this page
The Test 24 rows of arm A (240 moves, four games): every move legal; 95 percent changed the screen; no repeated no-op; one level of six in `sp80-589a99af` at move 49 and none in the other three games; the most frequent action in `ls20-9607627b` was sent 34 times of 60; about 3,900 thinking tokens and about 29 seconds a move; the prompt stayed at most about 4,700 tokens and almost nothing came from the cache. The Test 24 pilot of arm K (12 moves of one game): 3 of 12 moves wasted, one of them a reply that was not a legal move.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 4 check that the run is the run I describe; lines 5 to 17 are about what was found.

1. [guard] the arm with the explorer (arm B) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag, every framework move and every re-ask (90 %)
2. [guard] the arm A of Test 24 played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag (95 %)
3. [instrument] at most 40 percent of the moves of arm B were played by the framework itself (otherwise the explorer plays instead of the model) (65 %)
4. [instrument] at most 2 percent of the model's moves in arm B ended with a reply that could not be used (after the re-asks) (85 %)
5. [claim] wasted moves: the share of arm B's moves that changed the screen is at least 3 points above that of arm A (65 %)
6. [claim] memory of what did nothing: arm B repeated a move that had already done nothing from the same screen no more often than arm A did (80 %)
7. [claim] progress per token: the screens never seen before that arm B reached, per 1,000 computed tokens, are at least 1.25 times those of arm A (60 %)
8. [claim] reach: arm B reached at least as many screens never seen before as arm A, over the same number of moves (55 %)
9. [claim] levels: arm B completed at least as many levels as arm A, over the games (60 %)
10. [claim] arm B completed at least one level in at least one game (35 %)
11. [claim] arm B completed more levels than arm A (25 %)
12. [claim] speed: the effective moves per hour of calls of arm B (every move counts; a move the framework plays takes no time) are at least 1.3 times those of arm A (45 %)
13. [claim] thinking: the mean thinking tokens per call of arm B are no more than those of arm A (30 %)
14. [claim] tokens (the stand-in for energy): arm B computes no more tokens per effective move than arm A (45 %)
15. [claim] the explorer's own moves work: at least 60 percent of the moves the framework played itself changed the screen (55 %)
16. [claim] against the kept chat: arm B completed at least as many levels as arm K (Test 24) (60 %)
17. [claim] against the kept chat: arm B sent at most one third of the tokens arm K sent (Test 24) (85 %)

## Reading rules (fixed before the run)
1. the arm with the explorer (arm B) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag, every framework move and every re-ask
2. the arm A of Test 24 played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag
3. at most 40 percent of the moves of arm B were played by the framework itself (otherwise the explorer plays instead of the model)
4. at most 2 percent of the model's moves in arm B ended with a reply that could not be used (after the re-asks)
5. wasted moves: the share of arm B's moves that changed the screen is at least 3 points above that of arm A
6. memory of what did nothing: arm B repeated a move that had already done nothing from the same screen no more often than arm A did
7. progress per token: the screens never seen before that arm B reached, per 1,000 computed tokens, are at least 1.25 times those of arm A
8. reach: arm B reached at least as many screens never seen before as arm A, over the same number of moves
9. levels: arm B completed at least as many levels as arm A, over the games
10. arm B completed at least one level in at least one game
11. arm B completed more levels than arm A
12. speed: the effective moves per hour of calls of arm B (every move counts; a move the framework plays takes no time) are at least 1.3 times those of arm A
13. thinking: the mean thinking tokens per call of arm B are no more than those of arm A
14. tokens (the stand-in for energy): arm B computes no more tokens per effective move than arm A
15. the explorer's own moves work: at least 60 percent of the moves the framework played itself changed the screen
16. against the kept chat: arm B completed at least as many levels as arm K (Test 24)
17. against the kept chat: arm B sent at most one third of the tokens arm K sent (Test 24)

Headline: lines 5 and 7 both read yes ("THE HOST-OWNED EXPLORER WASTES FEWER MOVES AND REACHES MORE NEW SCREENS PER TOKEN"); otherwise MIXED. If a guard or instrument line (1 to 4) does not read yes, no figure is claimed until the cause is found. A line that was not run claims nothing. Levels (lines 9, 10, 11 and 16) are counted over four games and 60 moves each and are one or a few events; the page reads them as such.

## What this will not show
- **How well the model solves new games**, or how AXI compares on that: 60 moves a game is far below what a solver would use, and zero or one level is a likely outcome.
- **The leaderboard's score.** The levels counted are the games' own `levels_completed`; this is not an entry on the ARC-AGI-3 leaderboard.
- **Whether a different explorer, record or stall rule would do better.** These are fixed here, written before the run; a different rule could do better or worse.
- **Energy in joules.** The API gives none. Computed tokens and seconds are a stand-in.
- **Other games, other models, or any other company's service.** Four games, one model, one run.
- **Cost in dollars.** The prices are typed in by the owner at run time. No dollar figure is claimed on this page.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (12 real moves of arm B in the first game) with the share of moves the framework played itself, the speed and the projected cost at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** (8,000,000) stops the run. Everything finished is saved and the same cell carries on.

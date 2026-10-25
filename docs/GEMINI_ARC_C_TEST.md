# Test 26: Gemini 3.8 Flash on ARC-AGI-3 games with the ledger's read-out in the prompt, against arm B of Test 25 (written BEFORE the run, 25 Oct 2026)

*Concept-map item 135. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run.*

## Why this test
Test 25 gave the model the object view and the list of untried moves (arm B). Against arm A of Test 24 it changed the outcome by little: 95.4 % effective moves against 95.0 %, the same one level, fewer thinking tokens. The conductor never ran. What the model still has to work out for itself, in every game, is what each action does, which part of the screen is a bar or counter that changes in every move, and what never changes.

Between the two tests the owner built a transition ledger (`axi/engine/ledger.py`, Update 92 and the ones after it; the structure is Rajnish Choubey's RFC-001, the reading of it in game terms is **for Sir to confirm**, see `docs/HISTORY_LEDGER.md`) and ran it over the saved moves of Tests 24 and 25 on the real games (no model, no cost). It read the step of each action exactly (5, 6, 3 and 4 cells in the four games), it found the bar along an edge of the screen that changes in every move or every second move in all four games, and it found counts that never change. **This test asks the one question that follows: if the host writes down what the ledger has seen so far and puts it in the prompt, does the model play better?**

## What the read-out holds, and what it never holds
Each turn of arm C is arm B's turn (the Record, the objects, the untried moves, the screen) with one more section, the **Ledger**, between the objects and the screen. It is made by the host from the moves of **this run** played so far in the same level, and nothing else:

| line | what it says |
|---|---|
| per action | for each plain action tried at least twice: the step the object of one colour moved by, and in how many of its tries (a count, never a rule). Clicks are left out (one position is seen once or twice) |
| bars | a colour that gets smaller or larger in most of the moves that changed the screen (the same change every time, in one direction), with its area: probably a bar or counter |
| never changed | the colours that have not changed anywhere in the moves of this level |
| the last move | which objects moved and by how much, which cells changed colour (the bar's own change is not repeated) |

At most 12 lines. **It holds nothing from any other run, nothing about any game that was written by hand, and no rule.** A fact that the ledger found only after the saved runs were finished (for example that in one game something small changes at about the 43rd move) is **not** in it: in a live run the ledger sees that at move 43 at the earliest. A new level starts a new ledger. Every saved row holds `ledger_lines`, the number of lines the model was shown, and the replay of the saved moves rebuilds the ledger move by move and refuses a row whose number differs, so the read-out of move m is shown to be a function of the moves before m.

What it does not do: it does not say what the hidden-state flag says (that flag is not trustworthy yet on the real games, see the history page), it does not say that an object was "gone and new" (the matching still gives up on large pieces: the history page says so), and it plays nothing.

## What is run
One model, **`gemini-3.8-flash`**, streamed, default thinking. The same four public games as Tests 24 and 25 (`ls20-9607627b`, `g50t-5849a774`, `ka59-38d34dbb`, `sp80-589a99af`), 60 moves each, the same game seed. **Only arm C is run.** Arm B (Test 25) and arm A and arm K (Test 24) are the rows those tests saved; the notebook replays them into fresh copies of the games without any call to the model.

The system text of arm C is the system text of arm B extended by two sentences that tell the model what the Ledger section is and that its lines are counts of what happened, not rules. The conductor of arm B is kept as it was (when the last 4 moves found no new screen, or this screen has been seen 3 times, the framework plays the first untried move itself, marked "(framework)"); it did not run in Test 25 and may not run here. The model's thinking is left at its default, so arm C differs from arm B **only by the Ledger section** (and the two sentences about it).

The code is `python/axi/experiments/gemini_arc_c.py`; it imports `gemini_arc.py`, `gemini_arc_b.py`, `gemini_long.py` and `engine/ledger.py` and **edits none of them**. The notebook is `python/colab/axi_gemini_arc_c.ipynb`.

## What is measured
As in Tests 24 and 25: for every call prompt tokens, tokens served from the cache, reply and thinking tokens, seconds in all and seconds to the first token; for every move whether it was legal, whether it changed the screen (effective), whether it repeated a move that had already done nothing from the same screen, whether it reached a screen seen before, whether it was played by the framework, and the levels. New here: how many ledger lines the model was shown at each move.

## What I read before writing this page
The rows of arm B (240 moves, four games): 95.4 % effective, no repeated no-op, 221 screens never seen before, one level of seven in `g50t-5849a774`, about 2,770 thinking tokens and 24 seconds a move, a prompt of about 4,700 tokens. The ledger on the saved runs of the four games: on 25 Oct the step of each moving action came out exactly and the same in the three arms, but how often it held in the tries differed (for example 15 of 34 tries for one action in one arm of one game, 5 of 27 in another); that count is what the read-out shows, and why a step is never given without it. The read-out adds a handful of lines of about 20 to 30 words each to a prompt of about 4,700 tokens: a few percent (on the stand-in game 3 to 6 lines; the pilot prints the number on a real game).

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 5 check that the run is the run I describe; lines 6 to 16 are about what was found. Arm B itself differed from arm A by 0.4 points and one level of the same size as the model's own run-to-run noise would give, so **every claim line here is a single run against a single run and is read as a direction**.

1. [guard] the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines each move was shown (90 %)
2. [guard] the arm B of Test 25 played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves reproduces every saved flag (95 %)
3. [instrument] at most 40 percent of the moves of arm C were played by the framework itself (otherwise the explorer plays instead of the model) (90 %)
4. [instrument] at most 2 percent of the model's moves in arm C ended with a reply that could not be used (after the re-asks) (85 %)
5. [instrument] the read-out was there: at least 80 percent of the model's moves in arm C were shown at least one ledger line (92 %)
6. [claim] wasted moves: the share of arm C's moves that changed the screen is at least that of arm B (55 %)
7. [claim] memory of what did nothing: arm C repeated a move that had already done nothing from the same screen no more often than arm B did (70 %)
8. [claim] progress per token: the screens never seen before that arm C reached, per 1,000 computed tokens, are at least 1.1 times those of arm B (25 %)
9. [claim] reach: arm C reached at least as many screens never seen before as arm B, over the same number of moves (45 %)
10. [claim] levels: arm C completed at least as many levels as arm B, over the games (55 %)
11. [claim] arm C completed at least one level in at least one game (40 %)
12. [claim] arm C completed more levels than arm B (15 %)
13. [claim] thinking: the mean thinking tokens per call of arm C are at most 1.1 times those of arm B (70 %)
14. [claim] tokens (the stand-in for energy): arm C computes at most 1.2 times the tokens per effective move that arm B computes (the read-out makes each prompt longer) (85 %)
15. [claim] against arm A of Test 24: arm C completed at least as many levels as arm A (55 %)
16. [claim] against the kept chat: arm C completed at least as many levels as arm K (Test 24) (20 %)

## Reading rules (fixed before the run)
1. the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines each move was shown
2. the arm B of Test 25 played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves reproduces every saved flag
3. at most 40 percent of the moves of arm C were played by the framework itself (otherwise the explorer plays instead of the model)
4. at most 2 percent of the model's moves in arm C ended with a reply that could not be used (after the re-asks)
5. the read-out was there: at least 80 percent of the model's moves in arm C were shown at least one ledger line
6. wasted moves: the share of arm C's moves that changed the screen is at least that of arm B
7. memory of what did nothing: arm C repeated a move that had already done nothing from the same screen no more often than arm B did
8. progress per token: the screens never seen before that arm C reached, per 1,000 computed tokens, are at least 1.1 times those of arm B
9. reach: arm C reached at least as many screens never seen before as arm B, over the same number of moves
10. levels: arm C completed at least as many levels as arm B, over the games
11. arm C completed at least one level in at least one game
12. arm C completed more levels than arm B
13. thinking: the mean thinking tokens per call of arm C are at most 1.1 times those of arm B
14. tokens (the stand-in for energy): arm C computes at most 1.2 times the tokens per effective move that arm B computes (the read-out makes each prompt longer)
15. against arm A of Test 24: arm C completed at least as many levels as arm A
16. against the kept chat: arm C completed at least as many levels as arm K (Test 24)

Headline: lines 6 and 8 both read yes ("THE LEDGER READ-OUT WASTES NO MORE MOVES AND REACHES MORE NEW SCREENS PER TOKEN"); otherwise MIXED. If a guard or instrument line (1 to 5) does not read yes, no figure is claimed until the cause is found. A line that was not run claims nothing. Levels (lines 10, 11, 12, 15 and 16) are counted over four games and 60 moves each and are one or a few events; the page reads them as such.

## What this will not show
- **How well the model solves new games**, or how AXI compares on that: 60 moves a game is far below what a solver would use, and zero or one level is a likely outcome.
- **The leaderboard's score.** The levels counted are the games' own `levels_completed`; this is not an entry on the ARC-AGI-3 leaderboard.
- **Whether the ledger's reading is right in the games.** The read-out holds counts of what the host saw; whether it reads Sir's structure as he means it is for Sir to confirm. The hidden-state flag and the matching of large pieces are known to be open and are not in the read-out.
- **Whether a different read-out would do better.** Its lines, their order and their wording are fixed here, written before the run.
- **Energy in joules.** The API gives none. Computed tokens and seconds are a stand-in.
- **Other games, other models, or any other company's service.** Four games, one model, one run.
- **Cost in dollars.** The prices are typed in by the owner at run time. No dollar figure is claimed on this page.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (12 real moves of arm C in the first game) with the share of moves the framework played itself, the number of ledger lines shown, the speed and the projected cost at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** (8,000,000) stops the run. Everything finished is saved and the same cell carries on.

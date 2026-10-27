# Test 27: the big ARC-AGI-3 test: Gemini 3.8 Flash on eight games, 60 moves each, plain against the ledger read-out against the read-out with the notes (written BEFORE the run, 27 Oct 2026)

*Concept-map item 145. The architecture is Rajnish Choubey's; the code, the cells and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games.*

## Why this test
Tests 24 to 26 played four games for 60 moves each, one run per arm. They cleared 2, 1, 1 and 2 levels (arms K, A, B and C), all of them the first level of a game, and the differences between arms were of the size of the model's own run-to-run noise. Test 24 and Test 25 said that the framework's record and its check keep the model steady and cheap (an illegal move never reaches the game, about 3,700 thinking tokens a move instead of 1,700 in the plain chat, the kept chat's prompt growing to 251,000 tokens); they said nothing about clearing more levels. **This test asks the question the owner put on 27 Oct 2026: does the framework make the same model smarter than the plain model, and do the new goal-readers add anything to the ledger read-out?** It is bigger than they were in the number of games (eight, twice as many) and keeps the 60 moves, on purpose: the owner asked for a test that costs about 20 dollars in all, and the cost of the plain model's chat grows with the square of the moves, so more games is the cheaper way to make a direction less likely to be one game's luck. The price of this is that a second level is unlikely to be reached in 60 moves (every clear so far was the first level of a game), so the page asks about it (line 14) and expects no.

## The three arms
All three are the same model, **`gemini-3.8-flash`**, streamed, default thinking, the same eight games, the same game seed, 60 moves a game (or until the game is won), each played once.

| arm | what the model gets | code |
|---|---|---|
| **K** | the plain model: one chat per game, every screen and every reply kept in the chat and sent again at every move | `gemini_arc.py` (arm K of Test 24) |
| **C** | the Record kept by the framework, the check with up to two re-asks, the object view, the untried list, the conductor, and the Ledger read-out of Test 26 | `gemini_arc_c.py` (arm C of Test 26) |
| **D** | arm C plus one section, the **Notes**, written by the host from the ledgers and the clears of this run only | `gemini_arc_d.py` (new) |

**The Notes** (at most 8 lines, `goalbook.py`, Update 106; left out of the turn when there is nothing to say). They are readers of the transition ledger and change nothing in it:
- which actions changed the board without moving a piece, which changed nothing on the board, and which moved a piece and changed something else at the same time (counts from this level's own tries);
- the colours and the shapes on this level's first screen that no earlier level's first screen of this game had;
- what the earlier levels of this game suggest: the direction each action moved the piece in (firm when it was the same in every earlier level), and the step it made last time, **as a suggestion** that this level's first move of it cancels;
- guesses at what clears a level, from what the earlier clears of this game changed on the board (the strip and the background left out): repeat that change; empty a colour that went down in every earlier clear. **Each is marked as a guess and none has been tested.**

The system text of arm D is the system text of arm C extended by two sentences that tell the model what the Notes are. Nothing in the Notes is taken from any other run or written by hand for a game. The replay of the saved moves rebuilds the ledgers and the clears move by move and refuses a row whose number of note lines (`goal_lines`) differs from what the replay finds, so the Notes of move m are shown to be a function of the moves before m. Arm D differs from arm C **only by the Notes** (and the two sentences about them).

Arms K and C are run again here, on all eight games at 60 moves: the rows of Tests 24 and 26 are four games and are not reused (the games, the days and the server's load differ). Their files are not touched.

## The eight games
The four games of the earlier tests (`ls20-9607627b`, `g50t-5849a774`, `ka59-38d34dbb`, `sp80-589a99af`) and four more of the public games the toolkit offers on the day. **The rule for the four, fixed here:** the offered game ids, minus the four, ordered by the sha256 of the id (hex digits, ties by the id), the first four. The order does not depend on how the toolkit lists its games. The notebook prints the eight ids and saves them on the owner's Drive before the first move; a stopped run carries on with the same eight. If the toolkit offers fewer than eight games, the run has as many as it offers and the page's lines are read over those. The games are the toolkit's public ones: no hidden game, no scorecard.

## What is measured
As in Tests 24 to 26: for every call prompt tokens, tokens served from the cache, reply and thinking tokens, seconds in all and seconds to the first token; for every move whether it was legal, whether it changed the screen (effective), whether it repeated a move that had already done nothing from the same screen, whether it reached a screen seen before, whether it was played by the framework, and the levels. New here: how many note lines the model was shown at each move of arm D, and the levels completed in each game, arm by arm (the table the lines 7 to 14 are read from).

## What I read before writing this page
- **Time and cost, from the earlier tests.** Arm K of Test 24 took 15.7 seconds a move; arm A 28.7; arm C 23.2. 480 moves per arm is therefore about 2 to 3.5 hours of calls per arm. The three arms are independent and each resumes where it stopped, so the owner can run them in three Colab sessions at once.
- **Arm K's chat grows with the square of the moves.** In Test 24 its prompt grew from 4,381 tokens at move 1 to 251,033 at move 60 of a game (about 4,200 tokens a move), and it sent 25.6 million tokens in 240 moves, 94.6 % of its prompt served from the cache. At 60 moves the last prompt of a game is about 250,000 tokens and one game sends about 7 million; eight games about 51 million, most of it cached. That is why the cap on tokens sent for arm K is 120,000,000 (the cap of Test 24) and why the pilot prints the growth and the projected cost before any budget is set. No dollar figure is claimed on this page.
- **What was found on the four games.** Arm K cleared 2 levels, A 1, B 1 and C 2 (Tests 24 to 26); the six clears were four distinct levels, all level 0 (check 8). Both clears in `sp80-589a99af` were made by ACTION5. Nothing there says how a second level goes.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 check that the run is the run I describe; lines 7 to 18 are about what was found. **Every claim line is one run per arm against one run per arm over eight games, and the number of levels will be small (a handful in each arm): a difference of one or two levels is read as a direction and not as a finding.**

1. [guard] the plain model (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag (90 %)
2. [guard] the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines (88 %)
3. [guard] the arm with the read-out and the notes (arm D) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledgers and the clears and reproduces every saved flag, every framework move, every re-ask, the number of ledger lines and the number of note lines (88 %)
4. [instrument] at most 40 percent of the moves of arm C, and of arm D, were played by the framework itself (otherwise the explorer plays instead of the model) (85 %)
5. [instrument] at most 2 percent of the model's moves in arm C, and in arm D, ended with a reply that could not be used (after the re-asks) (85 %)
6. [instrument] the notes were there: at least 25 percent of the model's moves in arm D were shown at least one line of notes (65 %)
7. [claim] levels: arm C completed more levels than arm K, over the eight games (25 %)
8. [claim] levels: arm D completed more levels than arm K, over the eight games (30 %)
9. [claim] levels: arm D completed at least as many levels as arm C (60 %)
10. [claim] levels: arm D completed more levels than arm C (20 %)
11. [claim] game by game: the games in which arm D completed more levels than arm K are more than the games in which arm K completed more than arm D (25 %)
12. [claim] game by game: the games in which arm D completed more levels than arm C are more than the games in which arm C completed more than arm D (20 %)
13. [claim] breadth: arm D completed a level in at least as many games as arm K and as arm C (50 %)
14. [claim] arm D completed a second level (levels completed 2 or more) in at least one game (15 %)
15. [claim] wasted moves: the share of arm D's moves that changed the screen is at least 3 points above that of arm K (60 %)
16. [claim] progress per token: the screens never seen before that arm D reached, per 1,000 computed tokens, are at least 1.5 times those of arm K (40 %)
17. [claim] tokens sent: arm D sent at most one third of the tokens arm K sent (the kept chat grows with the square of the moves) (97 %)
18. [claim] thinking: the mean thinking tokens per call of arm D are no more than those of arm K (15 %)

## Reading rules (fixed before the run)
1. the plain model (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag
2. the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines
3. the arm with the read-out and the notes (arm D) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledgers and the clears and reproduces every saved flag, every framework move, every re-ask, the number of ledger lines and the number of note lines
4. at most 40 percent of the moves of arm C, and of arm D, were played by the framework itself (otherwise the explorer plays instead of the model)
5. at most 2 percent of the model's moves in arm C, and in arm D, ended with a reply that could not be used (after the re-asks)
6. the notes were there: at least 25 percent of the model's moves in arm D were shown at least one line of notes
7. levels: arm C completed more levels than arm K, over the eight games
8. levels: arm D completed more levels than arm K, over the eight games
9. levels: arm D completed at least as many levels as arm C
10. levels: arm D completed more levels than arm C
11. game by game: the games in which arm D completed more levels than arm K are more than the games in which arm K completed more than arm D
12. game by game: the games in which arm D completed more levels than arm C are more than the games in which arm C completed more than arm D
13. breadth: arm D completed a level in at least as many games as arm K and as arm C
14. arm D completed a second level (levels completed 2 or more) in at least one game
15. wasted moves: the share of arm D's moves that changed the screen is at least 3 points above that of arm K
16. progress per token: the screens never seen before that arm D reached, per 1,000 computed tokens, are at least 1.5 times those of arm K
17. tokens sent: arm D sent at most one third of the tokens arm K sent (the kept chat grows with the square of the moves)
18. thinking: the mean thinking tokens per call of arm D are no more than those of arm K

Headline: lines 8 and 11 both read yes ("THE FRAMEWORK'S READ-OUTS MAKE THE SAME MODEL COMPLETE MORE LEVELS THAN THE PLAIN MODEL"); otherwise MIXED. If a guard or instrument line (1 to 5) does not read yes, no figure is claimed until the cause is found. A line that was not run claims nothing. **If lines 7 and 8 both read NO**, the read-outs did not make the model clear more levels than the plain chat on these games, whatever the other lines say, and the Result says so first. **If line 10 reads NO and line 9 yes**, the notes added nothing that the ledger read-out had not already given. **If line 14 reads NO**, nothing here says that anything carries from one level to the next. Levels are counted over eight games and 60 moves each and are a few events; the page reads them as such.

## What this will not show
- **How well the model solves new games**, or how AXI compares on that. The levels counted are the games' own `levels_completed`, one run per arm; the official scoring of ARC-AGI-3 (RHAE) is not used and this is **not an entry on the ARC-AGI-3 leaderboard**.
- **That a difference of a level or two is real.** The earlier tests showed arms differing by one level with the same settings in the noise. With eight games a consistent direction over games (lines 11 and 12) is stronger than a total, and still one run per arm.
- **That the notes are right.** The guesses at what clears a level are guesses; none has been tested on a live run, and six clears are all that the saved runs hold. Whether the ledger reads Sir's structure as he means it is for Sir to confirm.
- **Other games, other models, or any other company's service.** Eight public games, one model.
- **Energy in joules.** The API gives none. Computed tokens and seconds are a stand-in.
- **Cost in dollars.** The prices are typed in by the owner at run time. No dollar figure is claimed on this page.
- **That the arms are run under the same conditions of time**: they are run in separate sessions, possibly on different days; the model's speed and the server's load can differ.

## Guards on the bill
The notebook never turns billing on. It runs a **pilot** (12 real moves of each arm in the first game) and prints the growth of arm K's chat, the speed and the projected cost of each arm and of the three together at the owner's prices, before any budget is set. A **dollar budget** (typed after the pilot, for each session) makes the notebook refuse the next call once the saved answers cost that much; a **cap on tokens sent** (arm K 120,000,000; arms C and D 8,000,000 each) stops a run. Everything finished is saved, and the same cell carries on where it stopped.

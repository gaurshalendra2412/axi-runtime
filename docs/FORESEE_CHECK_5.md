# The fifth forward check: repeats, the gap, filling the gap, the delete test (written BEFORE the run, 27 Oct 2026)

*Concept-map item 140. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games. The third and fourth checks (`FORESEE_CHECK_3.md`, `FORESEE_CHECK_4.md`) were run once on Colab by the owner; their numbers are in `docs/results/foresee_check_3_all.json` and `foresee_check_4_all.json` and their Results are not written yet.*

## Why this check
The first four checks asked one thing: can the host say where the piece goes before a move. This one asks what the host's record says about the moves that were played, and about the moves it still cannot answer. What the earlier runs showed: with the whole front held and the actions typed, 603 of the 729 predicted moves were right (82.7 %), but 207 of the 936 moves (22 %) were not predicted at all, and the prediction said "does not move" 91 times and was right in about 83 of them, which is about 9 % of all the moves played. Whether those moves were a repeat, and what the 207 are, was not looked at.

Where the questions come from: the owner's reading of the Cinderella Project pages, in particular "AI Maths: Reset. Repeat. Pull Again. The Mechanism #1". The pages carry the byline of Saraswati Choubey and most of the text is signed by an AI; the owner's position is that Rajnish Choubey (Sir) steered the AI until it wrote them, so the patterns are his, and I read them for structure and not for their wording. I have not checked which sentences are Sir's own; **for Sir to confirm**. The patterns used:
- **"Reset. Repeat. Pull again."** A pull that keeps nothing between shots and does not aggregate; the same correction is given again and again. In game terms: how many moves send the same action from the same place that already did nothing.
- **"Subset performed, superset claimed; the gap between them is where the human works."** In game terms: coverage, precision on what is covered, and a list, with reasons, of what is not. The ledger already says "I do not know" (it abstains); this puts the reasons next to the 207.
- **A counter that wraps, "return to the first point" (zero), and "the number is always a step behind".** In game terms: the ledger's row D already reports which of the four unit steps no action has been seen to make (`Ledger.completion`) and which pairs of actions move the same piece by the same, the opposite or a quarter-turned step (`Ledger.relations`). Until now nothing used that to predict. The checks are the step ahead; this uses the gap that the record shows.
- **The Delete Test of the 99 % / 1 % essays**: remove a part and see whether anything changes.

I had first thought that action-group inference was a new idea. It is not: it is row D of the ledger, and the first four checks simply never used it.

## What is measured (the ledger and the predictor of the third check are not changed)
1. **Repeats.** Of the moves in which the piece did not move, how many were the same action tried before from the same place of the piece (the cells it covers, as the check locates them), which had not moved it then either. The repeat of the whole screen is not the measure: the screen almost never repeats (208 to 227 of the 240 screens of arms K, A and B were new), so the repo's "repeat no-ops" column (0, 0 and 1 in 240 moves for arms B, A and K) cannot see a repeat that is only the same place. Only predicted moves are counted, since the others have no location for the piece. Also counted: the moves that changed no cell, and the whole-screen repeats, for the record.
2. **The gap.** Every move the check could not predict is put under one reason: a click; the action's first try in the level; an action tried once that did move the piece; an action tried once that did not; an action tried twice or more that never moved anything. For each reason: how many an earlier level's law (the fourth check) would have answered anyway.
3. **Filling the gap**, two additions, each off by default, each switched on alone and together:
   - `once`: an action that moved the piece in its one earlier try is predicted from that try (the third check waits for two tries).
   - `elim`: an action never tried in this level is predicted to make the one unit step no other action has made, when exactly one of the four is missing for the main mover. This can be wrong when the new action is a press and not a move; the check counts how often it is.
4. **The delete test.** The third check's predictor with each of its five parts deleted alone: the whole front held, the action typed, the edge of the screen as a wall, the floor colours (without them every unpassed cell in front is held), the mover as a group (without it the mover is the one colour). Also counted, for each of ACTION1 to ACTION4, the unit step it made on the main mover in every level, to see whether an action name means the same thing in every game.

Nine ways run on the same moves: full (the third check's full version, which must give its 603 right of 729), the five deletions, once, elim, once and elim. The same saved arms (K, A, B and C, 960 moves), replayed into fresh copies of the games. No model, no cost. The code is `python/axi/experiments/foresee5.py` (it reads the ledger as it is and adds no line to it), the notebook `python/colab/axi_foresee_check_5.ipynb`.

On the invented worlds (five 60-move levels of one layout, not benchmark games) with every part on and both additions off, the counters are the third check's, counter for counter. There `once` adds 13 predicted moves (12 right in the pad world, 10 in the push world) and `elim` adds one move and is wrong in it, because the one action never tried before was a press: line 6 is a real question.

## Predictions (mine, written first)
1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] with every part on and both additions off the check gives the third check's full figure on the same runs: 603 right of 729 predicted (90 %)
3. [claim] of the moves in which the piece did not move, at least 20 percent were the same action tried before from the same place of the piece, which had not moved it then either (50 %)
4. [claim] the share of those repeats among the moves in which the piece did not move is lower in arm C (the ledger's read-out in the prompt) than in arm K (the kept chat) (40 %)
5. [claim] with once and elim together at least 30 more moves are predicted than with them off, and at least 70 percent of the moves added are right (55 %)
6. [claim] elim alone predicts at least 8 moves, and at least 70 percent of them are right (25 %)
7. [claim] each of the five parts (the whole front held, the action typed, the edge of the screen, the floor colours, the mover as a group), deleted alone, costs at least 5 right moves over the four arms and games (35 %)
8. [claim] for each of ACTION1 to ACTION4 the unit step it makes on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games (55 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. with every part on and both additions off the check gives the third check's full figure on the same runs: 603 right of 729 predicted
3. of the moves in which the piece did not move, at least 20 percent were the same action tried before from the same place of the piece, which had not moved it then either
4. the share of those repeats among the moves in which the piece did not move is lower in arm C (the ledger's read-out in the prompt) than in arm K (the kept chat)
5. with once and elim together at least 30 more moves are predicted than with them off, and at least 70 percent of the moves added are right
6. elim alone predicts at least 8 moves, and at least 70 percent of them are right
7. each of the five parts (the whole front held, the action typed, the edge of the screen, the floor colours, the mover as a group), deleted alone, costs at least 5 right moves over the four arms and games
8. for each of ACTION1 to ACTION4 the unit step it makes on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games

Headline: lines 3 and 5 both read yes ("THE LEDGER KNOWS MOVES THAT WERE WASTED, AND CAN ANSWER MOVES IT NOW LEAVES TO THE MODEL"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. **If line 3 reads NO**, a veto of "the same action from the same place" would save too few moves to be worth building, and the share of moves that went into a wall is something other than a repeat. **If line 5 reads yes**, the gap can be narrowed with what the record already holds, and the next step is to check the same two additions on the carried laws of the fourth check. **If line 6 reads NO**, elimination is not worth building: a new action is as often a press as the missing direction. **If line 7 reads NO for a part**, that part carries less than 5 of the 729 predictions and is a candidate to remove from the predictor (the Delete Test: a part that changes nothing is not needed). **If line 8 reads yes**, an action name could carry its step from one game to the next, which the fourth check did not try (it carried only within a game); that is a next check, not a result of this one.

## What this will not show
- That vetoing the repeats, or filling the gap, makes a playing model use fewer moves or clear more levels: this check only reads moves already played. That is a paid test, about the cost of arm C ($3.65).
- That a path-finder clears a level, or what the goal of a level is.
- Whether a part that costs few right moves here is useless elsewhere: four games, 60 moves each, one run per arm.
- That the Cinderella Project pages say this in so many words, or which sentences in them are Sir's own.
- Cost: there is none (no key, no Drive). It is a replay of saved moves.

## Result

Run by the owner on the saved runs (replay of arms K, A, B and C; no model, no cost). Headline printed by the check: **THE LEDGER KNOWS MOVES THAT WERE WASTED, AND CAN ANSWER MOVES IT NOW LEAVES TO THE MODEL**. Lines 1 to 5 yes, lines 6 to 8 no. Every figure is in `docs/results/foresee_check_5_printout.txt` and `foresee_check_5_all.json`; this page does not restate them. The reading lines above were locked (hash of the text above this heading) before the run and are not changed. Written afterwards, for Sir to confirm.

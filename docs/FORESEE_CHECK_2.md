# The second forward check: the mover as a group, and walls learned by position (written BEFORE the run, 26 Oct 2026)

*Concept-map item 137. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games.*

## Why this check
The first forward check (`FORESEE_CHECK.md`, item 136; run on 26 Oct, its numbers are saved in `docs/results/foresee_check_all.json`; **its Result is not written**, the owner asked to try correcting the check first) came back MIXED, and its main finding was about the check itself: the host could not look at walls in 553 of 729 predictions (g50t 179 of 179, ka59 177 of 177, ls20 197 of 197), because it took the moving piece to be exactly one object of one colour. 183 of the 207 misses were "the piece did not move when I said it would"; the first check did not look at what was in front of the piece, so it does not say they were walls. Two AI-written texts the owner pasted afterwards say that the cause is a missing "displacement covariance" and that every one of the 183 would have been caught by "Sir's pushout check". I take two ideas from them on their merits and nothing else: that the mover should be found by what moves together, whatever its colours, and that a wall should be learned from a move that did nothing, by position, not by colour. That the 183 are walls is **not assumed**: it is one of the lines below. Neither text is Sir's text, and what RFC-001 does say (a refused change is held state; a candidate is checked before it is committed) is read here into game terms **for Sir to confirm**.

## What changes from the first check, and nothing else
- **The mover** is the group of symbols that moved by the same step as the main mover of that action, in at least half of the tries in which the main mover moved by it (so a sprite of two or three colours is one mover, and a piece that is pushed along with it moves with it). It is placed where it was last seen to move, not "the one object of one colour". Its cells are the cells of those colours inside the box of its last move.
- **A wall is learned by position.** When the piece did not move, and exactly one of the cells it would have stepped onto is a colour the piece has never stepped onto and is a cell it has never stepped onto, that cell is held as blocking. A cell the piece has stepped onto stops being held as blocking. The edge of the screen blocks. The prediction says "does not move" when a cell it would step onto is held as blocking.
- **What counts as floor** is whatever colours the piece has stepped onto, not the screen's most common colour (in a first trial on the invented worlds the most common colour was the wall).
- Everything else is as in the first check: from the earlier moves of the same level only, a fresh ledger for each level, click actions and actions with fewer than two earlier tries not predicted, scored against the move that was really played. The same saved arms (K, A, B and C, 960 moves), replayed into fresh copies of the games. No model, no cost. The code is `python/axi/experiments/foresee2.py`, the notebook `python/colab/axi_foresee_check_2.ipynb`.

On the invented worlds (not benchmark games) this version is right in 85.7 % of the predicted moves in the push world and 90.9 % in the pad world, where step only is right in 72.8 % and 80.7 %.

## Predictions (mine, written first)
1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] the mover (the group that moves together) is found in at least 60 percent of the predicted moves in each of at least 3 of the 4 games (70 %)
3. [claim] the prediction with walls learned by position is right more often than step only in at least 3 of the 4 games (55 %)
4. [claim] the learned prediction said that the piece does not move at least 20 times in all, and was right in at least 85 percent of them (55 %)
5. [claim] the learned prediction is right in at least 80 percent of the predicted moves, over all four arms and games (40 %)
6. [claim] the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games (10 %)
7. [claim] of the moves in which the piece did not move, at least half had a colour it had never stepped onto in front of it (50 %)
8. [claim] in no game is the learned prediction more than 2 points worse than step only (80 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. the mover (the group that moves together) is found in at least 60 percent of the predicted moves in each of at least 3 of the 4 games
3. the prediction with walls learned by position is right more often than step only in at least 3 of the 4 games
4. the learned prediction said that the piece does not move at least 20 times in all, and was right in at least 85 percent of them
5. the learned prediction is right in at least 80 percent of the predicted moves, over all four arms and games
6. the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games
7. of the moves in which the piece did not move, at least half had a colour it had never stepped onto in front of it
8. in no game is the learned prediction more than 2 points worse than step only

Headline: lines 3 and 5 both read yes ("THE HOST'S LEARNED MODEL OF WHERE THE PIECE CAN GO PREDICTS WELL ENOUGH TO BUILD A PATH-FINDER ON"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. Lines 4 and 8 can no longer read yes by saying almost nothing (line 4 needs 20 statements; this was the weakness of lines 5 and 8 of the first check). **If the headline reads yes**, a path-finder over the learned model is worth building (Sir's `reachability.rs` is a plain breadth-first search and is not used by the ARC test today). **If line 7 reads NO**, then most "did not move" misses are not caused by anything in front of the piece, and walls are not the main source of error, whatever the pasted texts say.

## What this will not show
- That a path-finder, or anything built on it, clears a level. Where to go (the goal) is still unknown to the host.
- That the group found as the mover is the player in the game's own terms; it is what moved together in the saved moves.
- Anything about a game outside these four, a longer run, or a different model. 60 moves a game, one run each arm.
- That the pasted texts are Sir's.
- Cost: there is none (no key, no Drive).

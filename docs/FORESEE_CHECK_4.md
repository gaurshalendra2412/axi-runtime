# The fourth forward check: the board carried from level to level (written BEFORE the run, 26 Oct 2026)

*Concept-map item 139. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games. The third check (`FORESEE_CHECK_3.md`, item 138) has not been run either.*

## Why this check
In the first three checks every level starts from nothing: a fresh ledger, no floor, no law for any action. In the second check 207 of the 936 moves (22 %, so 729 predicted, 77.9 %) were not predicted at all, most of them because an action needs two earlier tries in each level before the check will speak. An action means the same in the next level of the same game, and so does the floor the piece walked on.

Where the idea comes from: two essays published under Rajnish Choubey's name (the nine-profile "blackboard" essay, and the one with the "tab sync" and "demote the model to a tool" jokes). Their text is largely pasted AI answers, and what they describe is not a game mechanic; the part that is structure is this: **one board that every view reads at once, and nothing held in a silo**, with the model only a proposer. That is read here in game terms: what the host has learned in the earlier levels of a game is on the same board when the next level starts. For Sir to confirm; I do not claim it is how he would put it. An earlier version of my notes left these essays aside at a glance; this check is what a second, structural reading gave.

## What changes, and nothing else
Two things are carried, both from the EARLIER levels of the SAME game and the same arm only (never from a later level, never from another game or arm):
- **The law of each action**: the step it made most often (the symbol that moved and the step), kept once it was seen at least 3 times. It is used for a move only when the level has no law of its own for that action yet, and only if that symbol is on the screen (otherwise the move is not predicted, as before). The move is then predicted as a plain step: the check does not know where the mover is in the new level until it has seen it move there.
- **The floor**: the colours the piece has walked on, so that the whole-front rule of the third check can start earlier.

Nothing else is carried: not a cell, not a wall, not where the piece is (the layout of a new level is new). Everything else is the third check with both of its changes switched on. It runs three ways on the same moves: fresh (the third check, unchanged), laws carried, laws and floor carried. The same saved arms (K, A, B and C, 960 moves), replayed into fresh copies of the games. No model, no cost. The code is `python/axi/experiments/foresee4.py` (the evaluator is `foresee3.py` with an optional `carry`), the notebook `python/colab/axi_foresee_check_4.ipynb`.

On the invented worlds (five 60-move levels of one layout, not benchmark games) carrying the laws raises the share of moves predicted from 64.3 % to 76.7 % in the pad world and in the push world, and the share of moves played that were predicted right from 54.7 % to 64.0 % (pad) and from 54.0 % to 63.3 % (push). Carrying the floor as well lowers it again, to 59.7 % and 54.0 %, so line 7 is a real question and not a formality.

## Predictions (mine, written first)
1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] with nothing carried the check predicts 729 moves, as the second and third checks do on the same runs (90 %)
3. [claim] with the laws carried, at least 85 percent of the moves played are predicted (fresh: 77.9 percent) (70 %)
4. [claim] with the laws carried, the share of moves played that were predicted right is higher than fresh in at least 3 of the 4 games (60 %)
5. [claim] the moves predicted only because of a carried law are right in at least 60 percent of them, and there are at least 50 of them (50 %)
6. [claim] with the laws carried, the moves predicted right among those predicted are no more than 2 points fewer than fresh, over all four arms and games (60 %)
7. [claim] with the floor carried as well, the share of moves played that were predicted right is not lower than with the laws only (35 %)
8. [claim] with the laws carried, at least 70 percent of the moves played are predicted right (the second check: 56.3 percent) (10 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. with nothing carried the check predicts 729 moves, as the second and third checks do on the same runs
3. with the laws carried, at least 85 percent of the moves played are predicted (fresh: 77.9 percent)
4. with the laws carried, the share of moves played that were predicted right is higher than fresh in at least 3 of the 4 games
5. the moves predicted only because of a carried law are right in at least 60 percent of them, and there are at least 50 of them
6. with the laws carried, the moves predicted right among those predicted are no more than 2 points fewer than fresh, over all four arms and games
7. with the floor carried as well, the share of moves played that were predicted right is not lower than with the laws only
8. with the laws carried, at least 70 percent of the moves played are predicted right (the second check: 56.3 percent)

Headline: lines 4 and 6 both read yes ("THE BOARD CARRIED FROM LEVEL TO LEVEL PREDICTS MORE OF THE MOVES, AND RIGHT, THAN A FRESH START"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. **If line 7 reads NO**, the colours the piece walked on in one level are not reliably floor in the next, and only the laws are worth carrying. **If line 5 reads NO**, the law of an action is not the same from level to level in these games, and the carry should not be built on. **If the headline reads yes**, the next step is the same carry in the real playing arm (a paid test, about the cost of arm C, $3.65), where the model is shown the carried laws instead of starting each level blind.

## What this will not show
- That a path-finder, or anything built on this, clears a level, or what the goal of a level is.
- Whether the carried law helps a model that plays: this check only predicts moves already played. That is the paid test above.
- Anything about a game outside these four, a longer run, or a different model. 60 moves a game, one run each arm.
- That the essays say this in so many words, or that the pasted AI text in them is Sir's.
- Cost: there is none (no key, no Drive). It is a replay of saved moves.

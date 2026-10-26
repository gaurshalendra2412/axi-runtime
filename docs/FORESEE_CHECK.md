# Can the host say where the piece goes before the move? A free check on the saved ARC-AGI-3 runs (written BEFORE the run, 26 Oct 2026)

*Concept-map item 136. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games.*

## Why this check
After Test 26 (arm C, the ledger's read-out in the prompt: 2 levels against arm B's 1, which the page itself reads as one event in one run) the owner pasted seven texts from AI chats about Sir's work and asked what in them could help. Most of them are an AI's extension of his structure (a "GPS" layer made of diffusion distances, conditioning trajectories, a latent-phase functor), not text of his that I could find in the repository, and I do not write them in as his. What does come from his RFC-001 and could help is **pillar III**: a candidate transition is written to a shadow, checked by the gate, and committed or dropped with nothing to undo. A game cannot be copied to look ahead in, so the shadow would have to be a model the host **learned** from the moves it has seen (what the ledger holds). Whether such a model is good enough to plan on is a question with a number for an answer, and the saved runs can give it for nothing. How this reads his structure in game terms is **for Sir to confirm**.

## What is predicted
For each move of the four saved arms (K, A and B of Tests 24 and 25 and C of Test 26: 4 x 240 = 960 moves, no model and no cost), from the earlier moves **of the same level only**, in the order played (a fresh ledger for each level, as in the read-out of arm C):

- **step only:** the action moves the object of one colour by the step the ledger has seen it make most often (an action with at least two earlier tries);
- **colours that never changed as walls:** the same, but "does not move" when the cells the piece would step onto are outside the screen or hold a colour that has never changed in the level (after three moves);
- **learned:** the same, but "does not move" when a colour among those cells has stopped this piece before (alone among the colours that are not floor) and the piece has never been seen to step onto it. The host learns its own gate from what it has seen refused.

Click actions and actions with no seen step are not predicted (counted). It is scored against the move that was really played: right when the object of that colour moved by exactly the predicted step (or did not move, when "does not move" was predicted). Where the move itself is not visible as a step (a piece pushed, a screen that reloads when a level is won), the prediction counts as wrong; the check does not tidy that away.

The code is `python/axi/experiments/foresee.py`; the notebook `python/colab/axi_foresee_check.ipynb` replays the saved moves into fresh copies of the games exactly as the ledger replay does (the replay still checks every saved flag and refuses a doctored row), then runs the check. It edits none of the existing modules except that the replay tool now also reads arm C.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below in the same order.

1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] a move is predicted (an action with a step seen in at least two earlier tries of the level) in at least 60 percent of the recorded moves (a move that wins a level, and a reset, is not a transition and is not recorded), over the four arms (85 %)
3. [claim] step only is right in at least 70 percent of the predicted moves, over all four arms and games (65 %)
4. [claim] the prediction that learns what stopped the piece is right more often than step only in at least 3 of the 4 games (65 %)
5. [claim] when the learned prediction says the piece does not move it is right in at least 80 percent of those cases (50 %)
6. [claim] the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games (25 %)
7. [claim] the colours that never changed, taken as walls, are right no more often than the learned prediction in every one of the 4 games (60 %)
8. [claim] in no game is the learned prediction more than 2 points worse than step only (80 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. a move is predicted (an action with a step seen in at least two earlier tries of the level) in at least 60 percent of the recorded moves (a move that wins a level, and a reset, is not a transition and is not recorded), over the four arms
3. step only is right in at least 70 percent of the predicted moves, over all four arms and games
4. the prediction that learns what stopped the piece is right more often than step only in at least 3 of the 4 games
5. when the learned prediction says the piece does not move it is right in at least 80 percent of those cases
6. the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games
7. the colours that never changed, taken as walls, are right no more often than the learned prediction in every one of the 4 games
8. in no game is the learned prediction more than 2 points worse than step only

Headline: lines 4 and 6 both read yes ("THE HOST'S LEARNED MODEL OF THE MOVES PREDICTS THE NEXT STEP WELL ENOUGH TO BUILD A PATH-FINDER ON"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. **What the headline would mean:** a path-finder over the learned model (Sir's `reachability.rs` is a plain breadth-first search over a graph and is not used by the ARC test today) would be worth building. What it would not mean: that it clears a level, since where to go (the goal) is still unknown to the host.

## What this will not show
- **That any of this makes a model clear more levels.** It is a measure of the host's own prediction, not of play.
- **The goal.** Nothing here says what wins a level; the host sees a win exactly (the level count goes up and 1,100 to 3,500 cells change at once) but not what caused it.
- **Anything about a game outside these four, or about a longer run.** 60 moves a game, wins came at moves 12 to 49.
- **That the texts the owner pasted are Sir's.** Only what is in RFC-001 (and the D4 blueprint credited in `d4.py`) is. The rest is not claimed as his here.
- **Cost.** There is none: the replay needs the toolkit (free) and no key.

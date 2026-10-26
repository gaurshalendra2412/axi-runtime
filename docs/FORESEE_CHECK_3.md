# The third forward check: the whole front held, and the actions typed (written BEFORE the run, 26 Oct 2026)

*Concept-map item 138. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games.*

## Why this check
The second forward check (`FORESEE_CHECK_2.md`, item 137; run on Colab on 26 Oct, its numbers are saved in the owner's `foresee_check_2_all.json`; **its Result is not written yet**) came back MIXED, line 3 NO and line 5 NO. What it found:
- The mover (the group that moves together) was found in 97.4 % of the 729 predicted moves (the first check: 24 %). Lines 1, 2 and 8 read yes.
- Walls learned by position barely helped: right in 72.3 % against 71.6 % for the plain step. The prediction said "does not move" only 7 times in 729, and was right 5 times.
- The 183 stops are of two different kinds. In ls20, 65 of 68 stops had a colour the piece had never stepped onto in front of it, and the check said "does not move" 0 times there. In sp80 and g50t, 82 of 88 stops (44 of 44 and 38 of 44) had only colours the piece had walked on in front of it: nothing was in the way.

Two AI-written texts the owner pasted afterwards say that the first is a multi-cell front problem and the second is an action-typing problem, and that fixing both gives "92.45 %". Neither text is Sir's. Taken on their merits, the two ideas fit the numbers above; the 92.45 % is arithmetic that assumes every stop becomes a hit and no right prediction becomes wrong, so it is not used; the other things the texts name are left out for a different reason than "RFC-001 does not have them" (an idea is judged on what it does here, not on whether a page already holds it): a running-down gauge is already seen by the ledger's strips (`ticks`, found earlier in ls20) and is not what this check tests; "Action 4 is interact" and "stamina" are not in our data, and the stops-by-action printout is how this check looks at them; a "GPS" path-finder waits for a prediction that is good enough. That the piece in ls20 covers several cells, and that the sp80 and g50t stops are one kind of action, are guesses until this check is run.

## What changes from the second check, and nothing else
Each change can be switched off. The check is run four ways on the same saved runs: both off (the second check exactly), the front only, the typing only, and both. The reading lines below are about "both" and about "both off".
- **The whole front is held.** When the piece did not move, and the floor is known (the piece has stepped onto something), every cell in front of it that is not a colour it has walked on and has never been stepped onto is held as blocking, not only when exactly one such cell is left (a piece that covers several cells almost never leaves exactly one). A cell the piece later steps onto is released at once, so a cell held by mistake costs one wrong "does not move" and no more.
- **The action is typed by its own record on open floor.** An action is called stationary when, in at least 4 earlier tries of it in which only colours the piece had walked on were in front of it (or the piece moved), the piece moved in fewer than half. It is then predicted "does not move", and a stop of such an action is not taken to say anything about a wall. A stop in front of something is not held against the action: a first trial on the invented worlds that counted every stop called ordinary actions stationary because they had bumped into walls, and was right in 7 % of what it said.
- It also prints, for the version with both changes, the stops by action number, so that "some actions are not steps" is looked at and not assumed.
- Everything else is as in the second check: the earlier moves of the same level only, a fresh ledger for each level, click actions and actions with fewer than two earlier tries not predicted, scored against the move that was really played. The same saved arms (K, A, B and C, 960 moves), replayed into fresh copies of the games. No model, no cost. The code is `python/axi/experiments/foresee3.py`, the notebook `python/colab/axi_foresee_check_3.ipynb`.

Read into game terms from Rajnish Choubey's RFC-001, for Sir to confirm: a refused change is kept as held state, and released when the change goes through. On the invented worlds (single-cell pieces, not benchmark games) the new version gives no gain, as it cannot there: right in 85.2 % of the predicted moves in the push world and 90.6 % in the pad world, against 85.7 % and 90.9 % with both changes off. On a tall piece in front of a two-cell wall, and on an action that stops on open floor, each change does what it is for (the tests count it).

## Predictions (mine, written first)
1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] with both changes switched off the check gives the second check's figure on the same runs: 527 right of 729 predicted (90 %)
3. [claim] with both changes the prediction is right more often than with both off in at least 3 of the 4 games (70 %)
4. [claim] in ls20 the prediction with both changes is right in at least 75 percent of the predicted moves (the second check: 64.0 percent) (55 %)
5. [claim] with both changes the prediction said that the piece does not move at least 40 times in all, and was right in at least 85 percent of them (40 %)
6. [claim] with both changes the prediction is right in at least 80 percent of the predicted moves, over all four arms and games (35 %)
7. [claim] with both changes the prediction is right in at least 90 percent of the predicted moves, over all four arms and games (10 %)
8. [claim] at least half of the stops with only walked colours in front happened on an action that its own record called stationary at that moment (55 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. with both changes switched off the check gives the second check's figure on the same runs: 527 right of 729 predicted
3. with both changes the prediction is right more often than with both off in at least 3 of the 4 games
4. in ls20 the prediction with both changes is right in at least 75 percent of the predicted moves (the second check: 64.0 percent)
5. with both changes the prediction said that the piece does not move at least 40 times in all, and was right in at least 85 percent of them
6. with both changes the prediction is right in at least 80 percent of the predicted moves, over all four arms and games
7. with both changes the prediction is right in at least 90 percent of the predicted moves, over all four arms and games
8. at least half of the stops with only walked colours in front happened on an action that its own record called stationary at that moment

Headline: lines 3 and 6 both read yes ("THE HOST'S MODEL WITH THE WHOLE FRONT HELD AND THE ACTIONS TYPED PREDICTS WELL ENOUGH TO BUILD A PATH-FINDER ON"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. **If the headline reads yes**, a path-finder over the learned model is worth building (Sir's `reachability.rs` is a plain breadth-first search and is not used by the ARC test today). **If line 8 reads NO**, the stops on open floor are not one kind of action, and the cause is something this check does not see (the by-action printout is the first place to look). **If line 4 reads NO while line 3 reads yes**, the front idea does not fix ls20 and the gain, if any, comes from the typing.

## What this will not show
- That a path-finder, or anything built on it, clears a level. Where to go (the goal) is still unknown to the host.
- Why a piece stops on open floor (an action that does nothing, a rotation, a counter that ran out): it only says which actions do.
- Anything about a game outside these four, a longer run, or a different model. 60 moves a game, one run each arm.
- That the pasted texts are Sir's, or that any claim in them beyond the two ideas above is right.
- Cost: there is none (no key, no Drive). It is a replay of saved moves.

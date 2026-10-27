# The sixth forward check: first tries from the shape of the actions, the three tiers, the names, the delete test again, the screen without its clock (written BEFORE the run, 27 Oct 2026)

*Concept-map item 141. The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. This page is written and hash-locked before the check is run; the Result goes below the line at its end, in a later update with a test of its own. Nothing on this page has been run on the real games. The fifth check (`FORESEE_CHECK_5.md`) was run once in a Codespace by the owner; its printout is in `docs/results/foresee_check_5_printout.txt` and its Result is not written yet.*

## Why this check
The fifth check found, on the 936 saved moves (729 predicted, 603 right):
- **Repeats.** In 88 of the 183 moves in which the piece did not move (48.1 %) the same action had already done nothing from the same place of the piece. It is very uneven: ls20 64.7 %, sp80 75.0 %, g50t 25.0 %, ka59 0.0 %. The whole screen repeated once in 936 moves, because a strip along the edge changes in nearly every move.
- **The gap.** 207 moves were not predicted: 96 were the action's first try in the level, 68 an action tried once that had moved the piece, 18 an action tried once that had not, 16 clicks, 9 an action tried twice or more that never moved anything. An earlier level's law would have answered 30 of the 207.
- **`once`** (predict from one earlier try that moved the piece) added 68 predicted moves, 60 of them right (88.2 %): 603 right of 729 became 663 of 797, with the share right among the predicted moves unchanged (83.2 % against 82.7 %).
- **The Delete Test.** Without the whole front held: 48 fewer right moves. Without the action typed: 28 fewer. Without the edge: 5 fewer. Without the floor colours: **15 more**. Without the mover as a group: **5 more**.
- **Two blanks that were mine.** `elim` predicted nothing and the line on the steps of ACTION1 to ACTION4 printed empty. The cause is that the check only knew steps of one cell, and the real games move 5, 6, 3 and 4 cells (`GEMINI_ARC_C_TEST.md`). The two NO lines of the fifth check (6 and 8) are therefore a blind spot of the check, not a finding about the games. The same run shows that the pairs of actions that moved the same piece by the same length were 38 opposite, 70 a quarter-turn apart and 1 the same: four actions that are the four turns of one step would give 1 to 2, and this is 1 to 1.84.

## Where the questions come from
The owner asked that the Cinderella Project pages be read for their structure, in full, and not by keywords. On 27 Oct 2026 nine of them were read from first line to last (the 51-minute and the 89-minute ones included). The pages carry the bylines of Saraswati Choubey, Radha Gupta, Rahul Chauhan and FUTURE OF AI, and most of the text is signed by an AI; the owner's position is that Rajnish Choubey (Sir) steered the AI until it wrote them, so the patterns are his. I have not checked which sentences are Sir's own; **for Sir to confirm**. What was read, as structure:
- **One radius, from "minimal, for your own" to "exhaustive, for show"** (the Spherical Mind of the open letter to Sundar Pichai); **"every theory fails when it claims, every theory is great if it suggests"**; **the question and the answer born at the same instant, summing to nothing** (the 89-minute page). In game terms: three tiers of what the host says, each with where it comes from and how often it held (a claim, a suggestion, a silence with its reason), and the residue between what was said and what happened as the only part that informs.
- **"Who speaks, why, when silence is right, whether it is inside or outside the conversation"** (the page on AI breaking in front of you). In game terms: the abstention table, with the first try of an action as the "outside" case.
- **"A name defines scope"; words taken from older meanings; "identity is contextual, intent not name"** (the pages on the Cinderella Project, on the stolen vocabulary and on Perplexity). In game terms: an action is known by what it does and by how it stands to the other actions, not by the label ACTION1.
- **The 24-hour clock of 300,000 years; "the number is always a step behind"; the circle that returns to its first point.** In game terms: the strip that ticks is the clock; take it out of the screen and the screen repeats.
- **A family that shares one phone, nothing deleted because nothing is hidden** ("the delete button you chose not to use"). In game terms: the Delete Test, and what the fifth check found deletable.
- **Where the map tears, the map broke, not the territory** (the singularity as a tear in the map). In game terms: where the ledger has no law it says so, with the reason, and does not guess.
The factual and political claims on those pages (history, companies, physics) were not checked and are not used.

## What is measured (the ledger and the predictor of the third check are not changed)
1. **A first try, answered from the shape of the actions already seen.** Two additions, each off by default, each switched on alone and together; both only for an action never tried in this level, and never from a later move:
   - `rot`, the any-length version of the fifth check's `elim`: the four actions of a mover make the four turns of one step of any length, (0,-L), (0,L), (-L,0), (L,0). When exactly three actions of the main mover have made three different steps of one length (each at least twice) and a fourth action is tried for the first time, it is predicted to make the one turn that is missing. A press is the case in which it is wrong.
   - `rel`, a **relation** carried from the earlier levels of the same game: a's step was b's step turned by k quarter-turns (k = 0 the same, 2 opposite). When an action is tried for the first time and an action of this level has a known step, the first is predicted to make the other's step turned by the relation seen before. If two known actions disagree, nothing is predicted. The earlier levels give a relation and never a step; the carried step of the fourth check answers 17 of these first tries, and this is a different thing.
   - When both apply, the carried relation is read first and the rotation set second.
2. **The three tiers.** Every prediction is tagged with where it came from: this level's own law of at least two tries (`law`), one earlier try (`once`), the rotation set (`rot`), the carried relation (`rel`). The right share of each tier is read separately.
3. **The names.** For ACTION1 to ACTION4, the direction of the step on the main mover in every level and its length; for the six pairs, the quarter-turns from the first to the second. Is ACTION1 the same direction in every game; and is the arrangement the same even where the directions are not?
4. **The Delete Test again:** the floor colours deleted, the mover as a group deleted, and both together.
5. **The screen without its clock.** The ledger already finds the strip that ticks (`Ledger.tick_cells`, row C). The same action on the same screen is counted again with only those cells left out; the fifth check, with them in, found 1 repeat.

Nine ways run on the same moves: full (the fifth check's `full`, which must give the third check's 603 right of 729), `-floor`, `-group`, `-floor-group`, `once`, `rot`, `rel`, `rot+rel`, `all` (once, rot and rel). The same saved arms (K, A, B and C, 960 moves), replayed into fresh copies of the games. No model, no cost. The code is `python/axi/experiments/foresee6.py` (it reads the ledger as it is and adds no line to it), the notebook `python/colab/axi_foresee_check_6.ipynb`.

On the invented worlds (steps of one cell, four actions tried at the start) the new parts predict almost nothing, because no action is tried for the first time after three others have been seen twice. On small worlds built for the purpose with steps of three cells, `rot` predicts the fourth action's step and is wrong for a press, and `rel` predicts an opposite action from the carried relation, and nothing when two known actions disagree. How often the real games give such a first try is the question of lines 3 and 4.

## Predictions (mine, written first)
1. [guard] the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag (95 %)
2. [instrument] with the new parts off the check gives the third check's figure (603 right of 729 predicted), the fifth check's once (663 right of 797) and the fifth check's deletion of the floor colours (618 right of 729) on the same runs (85 %)
3. [claim] the rotation set alone (rot) predicts at least 8 moves, and at least 70 percent of them are right (35 %)
4. [claim] the carried relation alone (rel) predicts at least 10 moves, and at least 70 percent of them are right (50 %)
5. [claim] with once, rot and rel together (all) at least 90 more moves are predicted than in the full version, and at least 80 percent of the moves added are right (35 %)
6. [claim] for each of ACTION1 to ACTION4 the direction of its step on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games (50 %)
7. [claim] for each of the six pairs of ACTION1 to ACTION4 the quarter-turns from the first to the second are the same in at least 90 percent of the levels in which both made a step of the same length (at least 4 levels each) (60 %)
8. [claim] deleting the floor colours and the mover as a group together gains at least 10 right moves over the full version, over the four arms and games (50 %)
9. [claim] with the cells of the ticking strip left out, the same action on the same screen repeats at least 30 times in the 936 moves (the fifth check, with the strip: 1) (55 %)

## Reading rules (fixed before the run)
1. the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag
2. with the new parts off the check gives the third check's figure (603 right of 729 predicted), the fifth check's once (663 right of 797) and the fifth check's deletion of the floor colours (618 right of 729) on the same runs
3. the rotation set alone (rot) predicts at least 8 moves, and at least 70 percent of them are right
4. the carried relation alone (rel) predicts at least 10 moves, and at least 70 percent of them are right
5. with once, rot and rel together (all) at least 90 more moves are predicted than in the full version, and at least 80 percent of the moves added are right
6. for each of ACTION1 to ACTION4 the direction of its step on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games
7. for each of the six pairs of ACTION1 to ACTION4 the quarter-turns from the first to the second are the same in at least 90 percent of the levels in which both made a step of the same length (at least 4 levels each)
8. deleting the floor colours and the mover as a group together gains at least 10 right moves over the full version, over the four arms and games
9. with the cells of the ticking strip left out, the same action on the same screen repeats at least 30 times in the 936 moves (the fifth check, with the strip: 1)

Headline: lines 4 and 5 both read yes ("A FIRST TRY OF AN ACTION CAN BE ANSWERED FROM THE SHAPE OF THE ACTIONS ALREADY SEEN, NOT FROM THEIR NAMES"); otherwise MIXED. If line 1 or 2 does not read yes, no figure is claimed until the cause is found. **If line 3 reads NO**, the rotation set fires too rarely or is wrong too often to be worth building, and a first try is answered by the carried relation or by nothing. **If line 4 reads NO**, a relation does not carry from one level to the next better than a step does, and the first tries stay with the model. **If line 5 reads yes**, the gap can be narrowed further with what the record already holds, and the next step is a paid test of the tiers in the read-out (about the cost of arm C, $3.65): the read-out would say which tier each line is and how often it held, and would add the places where an action has already done nothing. **If line 6 reads yes**, an action name carries its direction from one game to the next; **if line 6 reads NO and line 7 reads yes**, the arrangement of the four actions is constant while the directions are not, which is a reason to know actions by their relations and not their names. **If line 8 reads yes**, the floor colours and the group are candidates to remove from the predictor (that is the experiment's predictor, not the ledger); a change would need its own check. **If line 9 reads yes**, the veto "this action from this screen did nothing before" can be stated on the screen itself once the strip is left out, and not only by the place of the piece.

## What this will not show
- That the tiers, a veto of the repeats or a first-try guess make a playing model use fewer moves or clear more levels: this check only reads moves already played. That is a paid test, about the cost of arm C ($3.65).
- That a path-finder clears a level, or what the goal of a level is.
- That a part that costs or gains few right moves here does the same elsewhere: four games, 60 moves each, one run per arm.
- Whether the same four actions are ACTION1 to ACTION4 in games other than these four.
- That the Cinderella Project pages say this in so many words, or which sentences in them are Sir's own.
- Cost: there is none (no key, no Drive). It is a replay of saved moves.

## Result

Run by the owner on the saved runs (replay of arms K, A, B and C; no model, no cost). Headline printed by the check: **MIXED: line 4 yes, line 5 NO**. Lines 1, 2, 4, 6, 7, 8 and 9 yes; lines 3 and 5 no. Every figure is in `docs/results/foresee_check_6_printout.txt`; this page does not restate them. The reading lines above were locked (hash of the text above this heading) before the run and are not changed. Written afterwards, for Sir to confirm.

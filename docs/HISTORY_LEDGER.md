# The transition ledger: what the host can know exactly about a world it was not told the rules of

*Concept-map item 132 (rows 172 to 178) and items 133 and 134 (rows 179 and 180). Code: `python/axi/engine/ledger.py`, and the invented worlds `python/axi/experiments/mini_worlds.py`. Checked by test 158 (`python/tests/test_ledger.py`). No model is called, and no benchmark game is used.*

## Where it comes from

Rajnish Choubey's RFC-001 (`RFC-001.md` in this folder, which credits him as the origin) gives three things this builds on, read into game terms. The wording in game terms is **for Sir to confirm**.

* A thing is a tuple of what it is and what state it is in (`z : x`). The ledger follows an object by what it is and reports what changed in its place or state.
* A rewrite is `L <- K -> R`, with `K` the part that is preserved. In an observed move, `K` is everything that did not change; what the move consumed and made is the rest.
* The gate checks a change against the held state, and the premise check (item 128) checks a sentence against it. The ledger is the third of the same kind: a claim about the **rules** is checked against the held **history**.

What prompted it: a pasted AI chat about a made-up "heavy" game and the name `axi-induce`. The chat itself says the name is its own label and is not in Sir's text, and that his published work is deductive. The made-up level cannot be won (its goal socket is walled in on every side, checked by flood fill with every gate open), so it is not used. Only its list of mechanics is: a far effect of a switch, a global mode, a crushed object, a change that cannot be undone. Three lines the chat quotes as Sir's (a rule as a conservation law or a symmetry, a polygon that is never a circle, "5 % public, 95 % proprietary") were not found in my notes on the four articles read (the articles themselves are not kept here) and are **not relied on**; the orthogonal-subspace text is not in RFC-001 either (items 98 and 99 note it as a shared text, not checked with Sir).

## The boundary, stated first

The ledger does not invent a rule and does not name the goal. A model still has to guess both. It holds the history exactly, answers what that history can answer, and checks the model's claims against it.

## What it keeps, from rows of symbols alone

| Row | What | How |
|---|---|---|
| 172 (A) | **An object-level change.** Which object moved by how much, which changed colour, which vanished or appeared. | Objects that are in both screens cancel; the rest are matched by symbol and shape (a move, nearest first), then by cells (a recolour). The exact changed cells are kept as well, so nothing depends on the matching being right. Two objects of one symbol and shape on both sides are flagged ambiguous. |
| 173 (B) | **Counts that never changed** (objects and cells per symbol), kept as laws after 3 moves. | A later break is reported. If the count comes back, it is marked **came back** (something hidden behind something else, or switched), not lost. |
| 174 (C) | **The hidden-state flag.** The same screen and the same action gave two different results. | The screen does not decide the outcome, so something that is not on the screen changed. The ledger cannot say what (a mode, a counter, chance, an animation). |
| 175 (D) | **The actions.** The step each moves an object by; which are the same, opposite or a quarter turn of each other; which of the four unit steps no action has been seen to make; the changes no recorded move was seen to undo. | The four unit steps are one orbit of the 8-element group in `d4.py`; test 158 checks the words against `d4.apply` for all 8 transforms. "No return seen" means only that no tried move put the screen back, not that none can. |
| 176 (E) | **A claim about the rules, replayed over the history.** | Four closed forms: `moves` (an action moves a symbol by a step), `changes_cell`, `keeps` (a count never changes), `on_arrival` (whenever a symbol arrives at a cell, another cell changes). The moves that contradict it are returned; an action never played is "untested"; an unknown form is refused. |
| 179 (H) | **A thing that gets smaller or larger is one thing with a new state.** A strip along the edge that is used up, a gauge that fills. A strip that changes the same way in nearly every move is named, with its area. | `resize` change: the same symbol, every cell of the smaller one inside the larger one, one-to-one only (two things that merge stay gone and new, and `touch` says so). `Ledger.ticks`: at least 70 % of the moves that changed the screen while it lasted, at least 6 times, and nearly always in one direction (a gate that something walks over alternates and is not a strip); **or** at least 50 % of them if it is the same change in 90 % of the moves (g50t's strip changed by one cell in every second move; second real run, 25 Oct). |
| 180 (I) | **The hidden-state flag, leaving the strip out.** | The same flag as row 174, with the cells that the strips changed left out of the comparison (`Ledger.tick_cells`; the first version left out the whole rectangle round them, which in ls20 covered most of the game, see below). What is left out is named in the report, and the strip may itself be the cause of a difference: it is not compared, not excluded as a cause. |
| 181 (J) | **The ledger's read-out in the prompt (arm C of the ARC-AGI-3 test).** | What the ledger has seen so far in this run is written as at most 12 lines of counts and put in the model's prompt (`experiments/gemini_arc_c.py`, `GEMINI_ARC_C_TEST.md`, test 160): per action the step and in how many of its tries, up to 3 bars, colours that have not changed, the last move. Counts, not rules; nothing from another run; **for Sir to confirm** that this is the reading of his structure he means. Built and checked on the stand-in game only; not run. |
| 182 (K) | **A forward look from what the ledger has seen (a shadow the host learned).** | Pillar III of RFC-001 read for game terms (for Sir to confirm): before a move, predict where the piece goes from the earlier moves of the level only, in three ways (the step; the step with the colours that never changed as walls; the step with the colours that stopped the piece before). Measured on the saved runs, no model (`experiments/foresee.py`, `FORESEE_CHECK.md`, test 161). Run once on the saved runs (numbers in `docs/results/foresee_check_all.json`, Result not written yet): the wall check could not be made in 553 of 729 predictions (the mover was not one object of one colour), step only was right in 71.6 %, headline MIXED. A path-finder over it is not built. |
| 183 (L) | **The second forward look: the mover as a group, walls learned by position.** | The same prediction as row 182 with two changes written before the run (`FORESEE_CHECK_2.md`, `experiments/foresee2.py`, test 162): the mover is the group of colours that moves together, found where it was last seen to move; a wall is learned by position from a move that did nothing, and a cell the piece has stepped onto stops blocking. It also counts what was in front of the piece when it did not move, because the 183 "did not move" misses of the first run are not shown to be walls. Reading in game terms, for Sir to confirm. Built and checked on invented worlds; **not run on the real games**. |
| 177 (F) | Candidate rules pruned by counterexample, and the probe on which the survivors disagree. | **Not built.** It needs a source of candidate rules, and nothing here supplies one. |
| 178 (G) | The invented one-mechanic worlds. | Test scaffolding for rows A to E, not a mechanism of Sir's. |

An effect that is not next to anything that moved in the same move is listed as **apart** (a gate that changes far from the pad that was stepped on). When the count of objects of a symbol changes but the count of its cells does not, two objects touched or one came apart (**touch**), and that is not an effect.

## What it gives (offline, on invented worlds)

Three small worlds that are **not benchmark games**, one mechanic each: a push world, a pad world whose mode shows (a gauge, open gates drawn as floor), and a pad world whose mode is hidden. 20 seeds of 1,000 random moves in each.

| Check | Result |
|---|---|
| Hidden-state flag | **0 of 20** in the push world, **0 of 20** where the mode shows, **16 of 20** where it is hidden (it needs the same screen twice under both modes). A scripted visit finds it every time. |
| Effects apart from what moved | **0**, **732** and **5**. The worlds' own counts are 0 pad entries, 732 pad entries and 5 crushed blocks, so the ledger and the worlds agree. |
| The change adds up | The object-level change accounts for the count change in every one of **60,000** moves. |
| A false claim | "Whenever the player arrives at the pad, the gauge changes" holds in 27 of 27 arrivals. The same claim with the wrong cell is refuted with 27 counterexamples. The claim with no condition ("R changes the gauge") is refuted too: a rule with no condition says more than the world does. |
| A crushed block | The law "2 blocks" is reported broken at the move that crushed one and has not come back. A pad hiding under the player breaks two laws and both come back the next move. |

**Cost.** About 2 ms for a move that changed a 64 x 64 screen (including both screens' objects) and 0.02 ms for a move that changed nothing; the queries take under 1 ms over 600 moves. A model call took 15.9 to 28.7 s a move in Test 24. It is computed by the host from the screens; it is not asked of the model.

## What this does not show

* That any of this makes a model clear more levels. Test 24 had one level in four games, which is too small to say anything, and no live run was made.
* Anything on a real benchmark game from the first build; see the next section for what the first real run showed, and what was changed after it and has not been run on a real game. The tool for that is `python/axi/experiments/ledger_replay.py` (concept-map item 133, test 159): it plays the saved replies back into fresh copies of the games and shows each move to the ledger, with no model call and nothing paid. It has been run only on the stand-in game so far; the result on the real games is not known yet. In Codespaces, from the `python` folder: `pip install arc-agi arcengine`, then `python -m axi.experiments.ledger_replay ../docs/results/axi_gemini_arc_default.json --arm K`.
* That real games behave like these worlds. They are ours.
* The rule forms are four and unconditional except `on_arrival`; a rule that depends on something else is not expressible yet. Row F is not built.
* Touching objects of one symbol merge into one object, and something hidden behind something else looks as if it had gone; the exact cells are still right.

## What the first run on the real saved runs showed (25 Oct), and what was changed

The owner ran `python/colab/axi_ledger_replay.ipynb` on Colab over the saved runs of Test 24 and Test 25 (3 arms x 4 games, 699 moves; no model, no cost). The toolkit downloaded the four games itself.

| What | Result |
|---|---|
| Replay | Every saved flag reproduced on the real games, in all 12 runs. |
| The change adds up | The object-level change added up to the count change in **699 of 699** moves. |
| What each action does | The step of every action came out exactly, and the same in the three arms: **5 cells in ls20, 6 in g50t, 3 in ka59, 4 in sp80**. In g50t, ACTION5 went the other way in one run and ACTION4 moved a different piece. |
| Time | **2.9 to 18.1 ms** a move on Colab (about 2 ms on the invented worlds), against 15.9 to 28.7 s a move for the model. |
| A strip used up every move | In all 12 runs a strip along an edge of the screen (one row, up to 64 cells) was used up in nearly every move. In both sp80 runs of arms A and B it ran out at move 29; in ls20 a piece of symbol 8 went from 3 to 2 at move 43, 43 and 47. **My reading** is a counter that runs down; it is not checked. |

**What was wrong.** (1) A strip that gets smaller was called "gone" plus "new", so "one symbol both gone and new" fired in nearly every move and the effects apart from what moved were inflated to 100 to 230. (2) The hidden-state flag fired in **0 of 12**: the strip makes every screen different, and the saved runs revisited a screen 0 to 1 times (8 once). (3) "6 of 5 tries": pieces were counted, not tries. (4) Each click position is its own action.

**What was changed** (rows 179 and 180, tests in test 158): the `resize` change and `Ledger.ticks`; the hidden-state flag that leaves the strip out; a try counted once. On an invented world that has such a strip (the shape the real runs showed, **not** the game): found in **20 of 20** seeds, and in **0 of 60** seeds of the three worlds without one; the strict flag finds **0 of 20** with the strip and the one that leaves it out finds **16 of 20** (the same as without a strip), **0 of 20** where the mode shows. The old numbers are unchanged.

**The second run (25 Oct, after rows 179 and 180) and two further corrections.** The new code ran on the real games. What came out: the strip is named in ls20, sp80 and ka59; the step of each action is unchanged; "one symbol both gone and new" fell but is not zero (g50t 32 to 41 moves, ka59 17 to 50, ls20 23 to 30). Two things were wrong in the ledger's reading of the screens (rows 179 and 180). (1) The hidden-state flag that leaves the strip out fired in ls20 (3 of 12 runs), but it hid the **whole rectangle** around the strip, and the rectangle of ls20's symbol 3 (x 13 to 54, y 8 to 62) covers most of the game. At moves 2 and 15 the two "same" screens differed in 72 cells inside it. **That flag was not evidence of a hidden state; it was the mask.** Changed: only the cells that the strip changed are left out (`tick_cells`); a test with a world of that shape fails if the rectangle is put back. **A look at ls20 moves 2 and 15 on the real game (25 Oct, free) showed that this was still too wide:** the used part of the bar turns into the colour of the floor ('b' to '3'), and the player walks on that floor, so the player's own cells (floor to player and back) were counted as strip cells and hidden too, and the flag on moves 2 and 15 may be the mask again (in move 2 the player did not move, in move 15 it moved 5 cells left). Changed again: cells where something moved (where it was and where it is) are never counted as strip cells (test 158, with a check that fails on the old rule). Not yet run on the real game. (2) g50t's strip (symbols 9 and 1, one cell each) changed in 26 and 25 of 43 moves, every second move, by exactly one cell: 60 %, under the 70 % that the strip rule asked, so it was missed. Changed: a strip that is the same change in 90 % of its moves needs only 50 %. Not yet run on the real game. **Still open, not changed:** in g50t a large object of symbol 5 (879 cells) is "gone" and "new" with the same size in nearly every move, which is where most of its 37 lost identities come from: it looks like (not checked) the moved piece joined to a large piece of the same symbol, so the component changes shape. The same thing in ls20: the floor of symbol 3 (892 cells) is gone and new, with the same size, in the moves where the player walks on it (cells '9' to '3' and '3' to '9' in moves 14 and 16). A matching by cell overlap would handle it, and that is a new rule, so it is not built without the owner's word.

**Not changed:** the click positions (one position is seen once or twice, so a per-position step says nothing, and grouping them would hide the one that matters).

**Not shown:** that these changes work on the real games. They were written after the run and have not been run on a real screen (the sandbox cannot install the toolkit). Running the notebook again after the push is the check. The reading in game terms (a thing keeps what it is while its state changes) is for Sir to confirm.

## Question 121 for Sir

Is a rule of a world, in his structure, something that stays the same or turns the same way (so that counts that never change, steps and their quarter turns are the right things to keep), or is that the AI chat's phrase? And is "a claim about the rules, checked against the history" the third gate as he would put it, next to the gate and the premise check?

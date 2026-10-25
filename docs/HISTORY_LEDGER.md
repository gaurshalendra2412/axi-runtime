# The transition ledger: what the host can know exactly about a world it was not told the rules of

*Concept-map item 132, rows 172 to 178. Code: `python/axi/engine/ledger.py`, and the invented worlds `python/axi/experiments/mini_worlds.py`. Checked by test 158 (`python/tests/test_ledger.py`). No model is called, and no benchmark game is used.*

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
* Anything on a real benchmark game. The packages (`arc-agi`, `arcengine`) could not be installed where this was written, and the saved runs hold the model's replies, not the screens. Replaying them needs a machine that can install the packages; it needs no model call.
* That real games behave like these worlds. They are ours.
* The rule forms are four and unconditional except `on_arrival`; a rule that depends on something else is not expressible yet. Row F is not built.
* Touching objects of one symbol merge into one object, and something hidden behind something else looks as if it had gone; the exact cells are still right.

## Question 121 for Sir

Is a rule of a world, in his structure, something that stays the same or turns the same way (so that counts that never change, steps and their quarter turns are the right things to keep), or is that the AI chat's phrase? And is "a claim about the rules, checked against the history" the third gate as he would put it, next to the gate and the premise check?

# Many eyewitnesses, one entity

*Concept-map item 122. Code: `python/axi/engine/witnesses.py`. Checked by test 148 (`python/tests/test_witnesses.py`). No model is called; nothing here is a measurement of a model.*

## Where it comes from

Rajnish Choubey's article "The 99 Eyewitnesses of the Ramayana" (Medium; the owner pasted its text on 24 Oct). Ninety-nine figures of the epic, from gods to citizens, each describe Sita by one adjective. No single adjective is Sita; all of them together pin her down. The second part of the article maps 108 open accounts onto one public ledger.

Read for structure, most of the article is what AXI already does:

| In the article | Already in AXI |
|---|---|
| The 1 % signal kept, the 99 % noise erased; the Delete Test | The exact state is kept and the history is dropped at every step |
| No locks, doors removed (Shani Shingnapur) | The pager writes without locks and finds a clash only at commit |
| 108 accounts on one open ledger instead of 5,778 private channels | One shared exact state that many agents write through the gate (concept-map item 100) |
| 108 shards showing one reflection; coordinates snapping back to the core | Canonical forms (D4) and the name table of grounding.py |

Two structures are new. Both say: many independent views, one exact entity.

## 1. A witness quorum for a change

The gate answers "is this change legal on this state", never "is it the change that was asked". A change that is **legal but wrong** is its one stated blind spot (technical reference, section 2.4). The quorum aims at that spot.

Several witnesses (independent proposals for the same request) are heard. Each admissible proposal is put into canonical form: its items as a set, so their order and exact repeats do not matter. A refused proposal is set aside, or counted in its repaired form when the framework's repair fixes it. A change is committed only when enough witnesses (by default a majority) proposed exactly the same canonical change. Too few in agreement, or a tie between two changes, commits nothing. Hearing witnesses never changes the host graph; the caller applies the agreed change through the gate as usual.

One witness that is wrong but legal cannot commit. A wrong change needs a majority of witnesses wrong in exactly the same way.

**The arithmetic, under independence.** With n witnesses each wrong with probability p, the chance that a majority is wrong is a binomial tail. With p = 0.0115, the share of replies that were wrong from a correct state in the 4,000-step run:

| Witnesses | Agreement needed | Chance a majority is wrong | Chance the right change misses the quorum |
|---|---|---|---|
| 1 | 1 | 1.15 % | 1.15 % |
| 3 | 2 | about 0.04 % | about 0.04 % |
| 5 | 3 | about 0.0015 % | about 0.0015 % |

Test 148 checks this by enumerating every outcome of three witnesses through the real `quorum` function: with one way of being wrong the wrong-commit rate equals the formula, with three ways it is lower, and when the witnesses copy each other it falls back to one witness's rate. Samples of one model are not independent, so the table is a **ceiling** of the benefit, not a measurement. Three witnesses also cost three calls a step; the quorum can be kept for the changes that matter most (a delete, an irreversible step).

## 2. A field of descriptions for an item

An item can be named by where several descriptions meet ("whatever Priya is handling", "the Friday one Priya has") instead of by its name. `resolve` intersects a description with the exact facts the host holds: exactly one item fits, it is **bound**; two or more, **ambiguous**; none, **unresolved**. As in grounding.py, ambiguous and unresolved apply nothing. grounding.py reads names and aliases; this reads facts. The model writes the description, and the host binds it exactly.

Run offline over the Test 23 stream (seed 20261122, 400 turns, no call), the host bound **23 of 23** "whatever <owner> is handling" changes to the right item and flagged **2 of 2** ambiguous questions as ambiguous. Of the **294** changes to an existing item in that stream, **238** could have been named by one of the item's facts alone, **34** by two, and **22** only by their name.

In Test 23 the model itself also got all 23 owner references and both ambiguous questions right, so this shows no gain on that run. What it adds is that the binding becomes exact and checkable by the host instead of being left to the model.

## What this does not show

That a model's witnesses are independent enough for the quorum to pay, what it costs in calls on a real job, or any gain on a model. Measuring the quorum needs a run with several samples a step; its cost would be stated before any run.

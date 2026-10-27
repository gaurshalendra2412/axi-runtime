# Three readers on the ledger: what each action did, what is new on a level, what to carry, and the guesses at the goal (built 27 Oct 2026)

*Concept-map item 144. The architecture is Rajnish Choubey's; the code is the owner's, with an AI assistant as a tool. Built and checked on invented worlds and small worlds. **Not run on the real games.** No model is called and nothing is bought. The ledger (`axi/engine/ledger.py`) and the framework's core are not changed: `axi/experiments/goalbook.py` reads a `Ledger` and writes nothing into it.*

## Why
Checks 1 to 8 measured how well the ledger predicts a move. Check 8 read yes on all 12 lines (the name prior pooled over the four arms existed and was right in every level with a step; a retire rule after two failures is nearly neutral; the six clears are four distinct levels, all level 0). What the ledger cannot say is what the level is for, what an action that moves nothing does, and what to take from the last level into a new one. The owner asked on 27 Oct 2026 to build what fits from those gaps before the paid test. Three AI-written texts the owner pasted put them in Sir's terms (row 177: candidate rules and the probe where they disagree; an action known by its effect; a rule carried as a suggestion that its first counterexample drops; a flag for what is new). Whose words those are, and whether this is the reading Sir intends, is **for Sir to confirm**. Their claims that the saved runs do not show are not used: collision prediction at 100 percent (the saved runs show about 117 of the 145 wrong predictions are a step that was blocked, and the ledger already predicts a block in 65 cases, 56 right), a symmetry or a closed shape as the goal (no evidence), the 48 percent (it belongs to a different test).

## What it does
1. **`action_kinds`**: each action in this level is a *mover* (a piece moved in at least half of its tries), a *changer* (something else on the board changed and nothing moved: a colour, something gone or new, a size), *idle* (nothing but the clock), *mixed*, or *once*. A click is one action. The ticking strip is left out.
2. **`novelty`**: colours and shapes on this level's first screen that no earlier level of the game started with.
3. **`carry_over`**: for each plain action, its direction in the earlier levels (firm when it was the same in every earlier level that had a law) and the step it made last time, as a *suggestion*: open, kept, or cancelled at the first move of that colour with another step.
4. **`clear_delta`, `hypotheses`**: what a cleared level changed on the board (the strip and the background left out), and from the clears seen in the game, up to three guesses at what clears the next level (repeat that change; empty a colour that went down in every clear). With no clear seen there are none.
5. **`read_out`**: the four as at most 8 plain lines for a prompt. **`show`** and `python/colab/run_goalbook_show.py` print them for every level of the saved runs: a printout, not a verdict.

## What it will not show
- That a model plays better with any of these lines. That is a paid test (about $3.65 for an arm like C), and it is the next step.
- That a clear repeats its change in the next level. Six clears, four distinct levels, all level 0.
- That a changer is how a level is cleared, or what the goal of a level is. The guesses are guesses; none has been tested.
- That the carried step or direction holds in games other than the four.

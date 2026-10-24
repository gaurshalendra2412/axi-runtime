# A wrap-around address frame in front of the gate

*Concept-map item 121. Code: `python/axi/engine/torus.py`. Checked by test 147 (`python/tests/test_torus_frame.py`). No model is called; nothing here is a measurement of a model.*

## Where it comes from

On 24 Oct the owner pasted an AI summary of two of Rajnish Choubey's articles, "The India of Zero & Infinity" and "The 99 Eyewitnesses of the Ramayana", with code. The articles themselves were not read here. The summary takes three ideas from them as host language:

* **An omnidirectional origin.** Every direction from a centre is presence; no direction is negative.
* **A fixed, closed set of points whose movement wraps.** The summary uses 108 points; leaving one edge re-enters at the opposite edge, so movement never stops at an artificial border.
* **Direct routing.** Any point may name any other in one step.

## What AXI already had, and what is new

AXI already shows the model the exact state at every step instead of a history, which is the summary's "simultaneous matrix presentation"; nothing was added for that. The gate is already independent of how cells are named: renaming the cells by any bijection never changes a verdict (`tests/test_gate_frame_covariance.py`). The gate also has no geometry of its own; shape lives in front of it, as the D4 symmetry does in `engine/d4.py`.

What is new is a geometry layer for jobs whose space is circular: days of the week, shift rotas, ring buffers, rotations, maps that wrap at their edges.

| Part of `torus.py` | What it does |
|---|---|
| `address` | Turns a coordinate the model wrote into the host's name for it. **strict** mode (the default) refuses a coordinate outside the frame; **wrap** mode wraps it and reports the wrap, never silently |
| `offset`, `distance` | Measure on the ring: every offset is a forward step in [0, n), and the distance is the short way round |
| `neighbours`, `adjacency` | The wrap-around neighbours, as `adj` edges the task can add to the host (the gate supplies no adjacency itself) |
| `move_delta` | Writes a move as an ordinary CICO delta: delete the node and its edges, add it at the wrapped target with the same edges. The gate decides it like any other delta, and `invert` undoes it |
| `shift_graph`, `shift_delta` | Move a whole state or delta round the torus (the group Z_rows x Z_cols acting on the cells) |

The framework core is unchanged: the parser, the gate, `apply` and `invert` are used as they are.

## Why the gate stays behind the wrap

Run as pasted, the summary's code accepted three wrong steps on its 108 points, because once every number wraps, its only check ("the address exists") is always true:

1. **An address outside the frame:** 200 was written to point 92 (200 mod 108). Here strict mode refuses it; wrap mode places it at 92 and reports the wrap.
2. **A move onto an occupied point:** the object already there vanished. Here the gate refuses the move (identification: the target node already exists) and the state is left as it was.
3. **A move from an empty point:** accepted as if something had moved. Here the gate refuses it (match: the source node is not in the host).

A legal move commits whole, carries its edges with it, and has an exact undo. On a 5 x 7 frame, for all 35 shifts, test 147 checks that the gate's verdict and kind of reason do not depend on where round the torus the state sits.

## The 108 points

Joined each to every other, 108 points have 5,778 links, as the summary says. A torus cannot carry that mesh without crossings: a surface needs ceil((n - 3)(n - 4) / 12) handles for the complete graph on n points (Ringel and Youngs), which is 910 for 108 points, and a torus has one. Seven points is the most a torus carries fully joined. The torus here is therefore the **address space**, not the mesh. As a frame one row high, 108 points form a ring with 108 edges between neighbours; any point can still name any other directly, because naming does not need an edge. 108 is a setting, not a rule.

## What this does not show

That a model proposes better moves on a wrap-around frame, anything about ARC-AGI games (most of their screens do not wrap), or anything about the two articles beyond the pasted summary. Whether a wrap-around frame helps a model on a real job is a measurement that needs a run; its cost would be stated before any run.

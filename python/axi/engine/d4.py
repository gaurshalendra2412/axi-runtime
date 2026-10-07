"""D4 (dihedral, 8 elements) canonicalization for 2-D grids (ARC-style), numpy only.

This is a tested rewrite of the `ARCLatticeCompiler` idea in Rajnish Choubey's blueprint:
"project every grid onto the canonical representative of its D4 orbit, remember which transform was used,
invert it on the answer".  What the tests in tests/test_d4.py establish:

  * transform ids follow the blueprint: t = 0..3 -> rot90 k=t, t = 4..7 -> fliplr(rot90 k=t-4);
    its inverse table {0:0,1:3,2:2,3:1,4:4,5:5,6:6,7:7} is correct (computed by brute force, not assumed)
  * canonical form is a true orbit invariant: canon(apply(g, t)) is the same grid for all 8 t,
    including NON-SQUARE grids (the blueprint compared raw bytes without the shape and failed there)
  * the transform id is not unique for symmetric grids; `stabilizer` returns all of them
  * stream compile/decompile is an exact round trip, including colour 0 and all-zero grids
  * PER-PAIR canonicalization is only valid for D4-EQUIVARIANT rules.  For orientation-dependent rules
    (gravity, "move right", ...) each pair gets a different transform and the demonstrations stop agreeing with
    each other.  `canonicalize_task` fixes this: ONE transform for the whole task (demo inputs, demo outputs,
    test input), chosen by a task-level minimum, so the canonical task is identical for all 8 orientations
  * therefore the PROMPT built from a canonical task is byte-identical for all 8 orientations of the task

What it does NOT show: that a model solves more ARC tasks.  That is an experiment, not a property.
"""
from itertools import product
import numpy as np

IDS = tuple(range(8))


def apply(grid, t):
    """Blueprint definition: rotate 90deg * (t % 4), then mirror left-right if t >= 4."""
    out = np.rot90(np.asarray(grid), k=t % 4)
    if t >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _probe():
    # asymmetric non-square grid: every element of D4 acts differently on it
    return np.arange(1, 13).reshape(3, 4)


def inverse_id(t):
    p = _probe()
    q = apply(p, t)
    for u in IDS:
        r = apply(q, u)
        if r.shape == p.shape and (r == p).all():
            return u
    raise AssertionError("no inverse")


INV = {t: inverse_id(t) for t in IDS}


def _key(grid):
    g = np.asarray(grid)
    return (g.shape, tuple(int(x) for x in g.ravel()))   # shape FIRST: grids of different shape never tie


def canonical(grid):
    """-> (canonical_grid, t) with apply(grid, t) == canonical_grid.  t = smallest id reaching the minimum key."""
    best_t, best_k, best_g = 0, None, None
    for t in IDS:
        g = apply(grid, t)
        k = _key(g)
        if best_k is None or k < best_k:
            best_t, best_k, best_g = t, k, g
    return best_g, best_t


def stabilizer(grid):
    """All t that map `grid` to its canonical representative (size 1 for an asymmetric grid, up to 8)."""
    c, _ = canonical(grid)
    return [t for t in IDS if apply(grid, t).shape == c.shape and (apply(grid, t) == c).all()]


def restore(canon_grid, t):
    """Undo canonical(): returns the grid in the ORIGINAL orientation."""
    return apply(canon_grid, INV[t])


# ---------------------------------------------------------------- token stream (header + cells)
def compile_stream(grid, keep_background=True, background=0):
    """-> dict(header=(h, w, t), cells=[(i, j, colour), ...]) in canonical orientation.
    keep_background=False drops `background` cells (sparse), which is lossless only because h, w are in the header."""
    c, t = canonical(grid)
    h, w = c.shape
    cells = [(i, j, int(c[i, j])) for i in range(h) for j in range(w)
             if keep_background or c[i, j] != background]
    return {"header": (h, w, t), "cells": cells, "background": background}


def decompile_stream(stream):
    h, w, t = stream["header"]
    g = np.full((h, w), stream.get("background", 0), dtype=np.int64)
    for i, j, col in stream["cells"]:
        g[i, j] = col
    return restore(g, t)


# ---------------------------------------------------------------- task level
def _task_grids(task):
    gs = []
    for x, y in task["train"]:
        gs += [np.asarray(x), np.asarray(y)]
    gs.append(np.asarray(task["test_input"]))
    return gs


def _transform_task(task, t):
    out = {"train": [(apply(x, t), apply(y, t)) for x, y in task["train"]],
           "test_input": apply(task["test_input"], t)}
    if "test_output" in task:
        out["test_output"] = apply(task["test_output"], t)
    return out


def canonicalize_task(task):
    """One transform for the whole task.  task = {"train": [(x, y), ...], "test_input": g[, "test_output": g]}.
    Chooses t minimising the concatenated keys of all grids; the result is identical for any D4 image of the task.
    -> (canonical_task, t)"""
    best_t, best_k = 0, None
    gs = _task_grids(task)
    for t in IDS:
        k = tuple(_key(apply(g, t)) for g in gs)
        if best_k is None or k < best_k:
            best_t, best_k = t, k
    return _transform_task(task, best_t), best_t


def restore_answer(answer_in_canonical_frame, t):
    return restore(answer_in_canonical_frame, t)


def task_prompt(task):
    """Plain-text ARC prompt (digits, one row per line).  Deterministic function of the task."""
    def rows(g): return "\n".join("".join(str(int(v)) for v in r) for r in np.asarray(g))
    parts = []
    for n, (x, y) in enumerate(task["train"], 1):
        parts.append(f"Example {n} input:\n{rows(x)}\nExample {n} output:\n{rows(y)}")
    parts.append(f"Test input:\n{rows(task['test_input'])}\nTest output:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- equivariance check
def is_equivariant(rule, grids):
    """True iff rule(apply(x,t)) == apply(rule(x),t) for every sample grid x and every t.
    Only equivariant rules may be canonicalized PER PAIR."""
    for x in grids:
        for t in IDS:
            a = rule(apply(x, t)); b = apply(rule(x), t)
            if a.shape != b.shape or not (a == b).all():
                return False
    return True

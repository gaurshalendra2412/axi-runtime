import numpy as np
from axi.engine import d4
from axi.engine.d4 import apply, canonical, stabilizer, restore, INV, IDS
from axi.engine.c4 import rotate_grid


def rand_grid(rng, h, w, ncol=4, density=.6):
    return (rng.integers(1, ncol + 1, (h, w)) * (rng.random((h, w)) < density)).astype(np.int64)


# ------------------------------------------------------------------ the group itself
def test_eight_ids_are_eight_distinct_actions_and_inverse_table_matches_blueprint():
    p = d4._probe()
    imgs = {(apply(p, t).shape, apply(p, t).tobytes()) for t in IDS}
    assert len(imgs) == 8
    assert INV == {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}      # the blueprint's table, now verified
    for t in IDS:
        assert (apply(apply(p, t), INV[t]) == p).all()


def test_group_closure():
    p = d4._probe(); table = {}
    for t in IDS:
        for u in IDS:
            r = apply(apply(p, t), u)
            hits = [v for v in IDS if apply(p, v).shape == r.shape and (apply(p, v) == r).all()]
            assert len(hits) == 1          # composition lands on exactly one of the 8
            table[(t, u)] = hits[0]
    assert len({v for v in table.values()}) == 8


def test_rotations_agree_with_the_c4_module():
    rng = np.random.default_rng(0)
    for n in (3, 4, 5):
        g = rand_grid(rng, n, n)
        assert (rotate_grid(n, g) == apply(g, 3)).all()        # c4.rotate_grid is a clockwise quarter turn


# ------------------------------------------------------------------ canonical form
def test_canonical_is_an_orbit_invariant_including_non_square():
    rng = np.random.default_rng(1)
    for h, w in [(1, 1), (1, 2), (2, 3), (3, 3), (3, 5), (4, 4), (5, 2), (6, 6)]:
        for _ in range(60):
            g = rand_grid(rng, h, w)
            c0, _ = canonical(g)
            for t in IDS:
                ct, tt = canonical(apply(g, t))
                assert ct.shape == c0.shape and (ct == c0).all(), (h, w, t)
                assert (restore(ct, tt) == apply(g, t)).all()


def test_blueprint_byte_comparison_without_shape_is_not_invariant():
    """Negative control: the blueprint compared grid.tobytes() only. [[1,2]] and [[1],[2]] have equal bytes."""
    def blueprint_canon(g):
        best, bb = g, g.tobytes()
        for t in range(1, 8):
            x = apply(g, t)
            if x.tobytes() < bb: best, bb = x, x.tobytes()
        return best
    a = np.array([[1, 2]]); b = np.array([[1], [2]])         # b is a rotation of a
    assert blueprint_canon(a).shape != blueprint_canon(b).shape       # blueprint: two different "canonical" forms
    assert canonical(a)[0].shape == canonical(b)[0].shape and (canonical(a)[0] == canonical(b)[0]).all()


def test_stabilizer_sizes():
    rng = np.random.default_rng(2)
    assert len(stabilizer(np.full((3, 3), 5))) == 8                         # uniform square: full symmetry
    assert len(stabilizer(np.full((2, 3), 5))) == 4                         # uniform rectangle
    g = np.array([[1, 2, 0], [2, 3, 0], [0, 0, 0]])                         # symmetric about the main diagonal
    assert len(stabilizer(g)) == 2
    asym = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    assert stabilizer(asym) == [canonical(asym)[1]] and len(stabilizer(asym)) == 1
    for t in stabilizer(g):                                                  # every stabiliser element is a valid choice
        assert (restore(canonical(g)[0], t) == g).all()


# ------------------------------------------------------------------ stream
def test_stream_roundtrip_is_exact_with_colour_zero_and_degenerate_grids():
    rng = np.random.default_rng(3)
    grids = [np.zeros((3, 3), dtype=np.int64), np.zeros((1, 4), dtype=np.int64), np.full((2, 2), 9)]
    grids += [rand_grid(rng, int(rng.integers(1, 8)), int(rng.integers(1, 8)), density=rng.random()) for _ in range(300)]
    for g in grids:
        for keep in (True, False):
            s = d4.compile_stream(g, keep_background=keep)
            assert (d4.decompile_stream(s) == g).all()
            if not keep:
                assert all(c != 0 for _, _, c in s["cells"])


# ------------------------------------------------------------------ per-pair vs task-level
def gravity_down(x):
    x = np.asarray(x); out = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[:, j][x[:, j] != 0]
        if len(col): out[x.shape[0] - len(col):, j] = col
    return out


def recolor_1_to_2(x):
    return np.where(np.asarray(x) == 1, 2, x)


def test_equivariance_checker():
    rng = np.random.default_rng(4)
    gs = [rand_grid(rng, 4, 5) for _ in range(20)] + [rand_grid(rng, 5, 5) for _ in range(20)]
    assert d4.is_equivariant(recolor_1_to_2, gs)          # colour rules commute with D4
    assert not d4.is_equivariant(gravity_down, gs)        # orientation-dependent rule


def gravity_dirs(x, y):
    """Which of the 4 gravity directions maps x -> y ?  (direction a == gravity_down conjugated by a rotation)"""
    return {a for a in range(4) if (apply(gravity_down(apply(x, a)), INV[a]) == y).all()}


def test_per_pair_canonicalization_breaks_orientation_dependent_rules_but_task_level_does_not():
    rng = np.random.default_rng(5)
    n_tasks = 40; per_pair_disagree = 0; task_level_disagree = 0
    for _ in range(n_tasks):
        pairs = []
        for _ in range(4):
            x = rand_grid(rng, 5, 5, ncol=3, density=.35); pairs.append((x, gravity_down(x)))
        # per pair: each pair is canonicalised separately (input's transform also applied to its output)
        dirs = []
        for x, y in pairs:
            t = canonical(x)[1]; dirs.append(gravity_dirs(apply(x, t), apply(y, t)))
        common = set.intersection(*dirs)
        per_pair_disagree += (len(common) == 0)
        # task level: one transform for everything
        task = {"train": pairs[:-1], "test_input": pairs[-1][0], "test_output": pairs[-1][1]}
        ct, t = d4.canonicalize_task(task)
        dirs = [gravity_dirs(x, y) for x, y in ct["train"]] + [gravity_dirs(ct["test_input"], ct["test_output"])]
        task_level_disagree += (len(set.intersection(*dirs)) == 0)
    print(f"   demos stop sharing ONE rule: per-pair {per_pair_disagree}/{n_tasks} tasks, task-level {task_level_disagree}/{n_tasks}")
    assert per_pair_disagree > n_tasks // 2
    assert task_level_disagree == 0


def make_task(rng, shapes):
    train = []
    for h, w in shapes[:-1]:
        x = rand_grid(rng, h, w); train.append((x, rand_grid(rng, h, w)))
    h, w = shapes[-1]
    return {"train": train, "test_input": rand_grid(rng, h, w), "test_output": rand_grid(rng, h, w)}


def test_canonical_task_and_prompt_are_identical_for_all_8_orientations():
    rng = np.random.default_rng(6)
    for _ in range(80):
        shapes = [(int(rng.integers(1, 7)), int(rng.integers(1, 7))) for _ in range(4)]
        task = make_task(rng, shapes)
        c0, _ = d4.canonicalize_task(task); p0 = d4.task_prompt(c0)
        for g in IDS:
            tg = d4._transform_task(task, g)
            cg, t = d4.canonicalize_task(tg)
            assert d4.task_prompt(cg) == p0
            assert (cg["test_output"] == c0["test_output"]).all()
            # answering in the canonical frame and restoring gives the answer in the orientation we were handed
            assert (d4.restore_answer(cg["test_output"], t) == tg["test_output"]).all()


def test_symmetric_inputs_do_not_break_task_invariance():
    sym = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])                      # stabiliser of size 8
    rng = np.random.default_rng(7)
    task = {"train": [(sym, rand_grid(rng, 3, 3))], "test_input": sym, "test_output": rand_grid(rng, 3, 3)}
    assert len(stabilizer(sym)) == 8
    p0 = d4.task_prompt(d4.canonicalize_task(task)[0])
    for g in IDS:
        assert d4.task_prompt(d4.canonicalize_task(d4._transform_task(task, g))[0]) == p0


def test_deterministic_solver_gives_orientation_invariant_answers_after_canonicalization():
    """Any deterministic function of the prompt (stand-in for a greedy LLM) is orientation-invariant once the prompt is."""
    def solver(prompt):                                      # arbitrary deterministic function of the text
        import hashlib
        h = hashlib.sha256(prompt.encode()).digest()[0] % 7        # position-sensitive, like a real model
        return np.arange(9).reshape(3, 3) * h % 10
    rng = np.random.default_rng(8)
    task = make_task(rng, [(3, 3)] * 4)
    outs = []
    for g in IDS:
        tg = d4._transform_task(task, g); ct, t = d4.canonicalize_task(tg)
        ans = solver(d4.task_prompt(ct))
        outs.append((d4.restore_answer(ans, t), g))
    canon_answers = {apply(a, INV[g]).tobytes() for a, g in outs}  # bring every answer back to the unrotated frame
    assert len(canon_answers) == 1
    raw = {solver(d4.task_prompt(d4._transform_task(task, g))).tobytes() for g in IDS}
    assert len(raw) > 1                                       # without canonicalization the solver is NOT invariant


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

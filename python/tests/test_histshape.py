"""Engineering support for docs/HISTORY_SHAPE_TEST.md (concept-map item 105): does a pair of Betti numbers of the recent history carry what the last 50 commands carry? CPU only, no model.
This file tests the INSTRUMENT, not the finding:
  1. the walker: every rectangle of the tour is a closed loop of the right length that starts at the walker's cell; the tour, the drift and the random walk each do what the page says on a fixed seed
     (moves change the cell by the move, failed moves and wall hits do not, the drift turns round at a wall, every switch changes the mode, segments are at least 40 steps);
  2. Betti numbers on hand-made graphs (a path, a square, a figure of eight, two separate triangles, a walk that goes back and forth) and the rule that a walk is connected;
  3. the views: the window and the label of a step are the 50 commands before it and the command at it, the cell is the cell before the command, counts, the last three, the repeat numbers on a periodic log;
  4. `match_next` on hand-made windows, `f1`, `score`, `fit_predict` learning a rule it is given;
  5. the run: seeded, train and test walks are different walks, the digest, the quick run and the command line;
  6. the thirteen reading rules each flipping where they should, with thresholds at their edges, and `None` when a view is missing;
  7. the page is locked (hash of the text above the Result), carries the thirteen lines with the probabilities, the settings and the thresholds the code uses."""
import contextlib, copy, hashlib, io, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.histshape as H

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "HISTORY_SHAPE_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"


# ---------------------------------------------------------------- 1 the walker
def test_every_tour_rectangle_is_a_closed_loop_of_the_right_length_that_starts_at_the_walkers_cell():
    rng = random.Random(3); seen = set()
    for _ in range(400):
        pos = (rng.randrange(H.G), rng.randrange(H.G)); ring = H.tour_cycle(rng, pos)
        assert ring[0] == pos and len(set(ring)) == len(ring) and 4 <= len(ring) <= 20 and len(ring) % 2 == 0, ring
        for a, b in zip(ring, ring[1:] + ring[:1]): assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)             # neighbours all the way round, and back to the start
        assert all(0 <= x < H.G and 0 <= y < H.G for x, y in ring)
        xs, ys = {c[0] for c in ring}, {c[1] for c in ring}; w, h = len(xs), len(ys)
        assert len(ring) == 2 * (w + h) - 4 and all(c[0] in (min(xs), max(xs)) or c[1] in (min(ys), max(ys)) for c in ring)       # a rectangle's perimeter and nothing else
        seen.add(len(ring))
    assert min(seen) == 4 and max(seen) == 20 and len(seen) >= 8
    second = {H.tour_cycle(random.Random(i), (3, 3))[1] for i in range(40)}
    assert len(second) >= 2                                                                                          # both directions occur
    assert H.cmd_between((2, 2), (2, 1)) == 0 and H.cmd_between((2, 2), (2, 3)) == 1 and H.cmd_between((2, 2), (1, 2)) == 2 and H.cmd_between((2, 2), (3, 2)) == 3
    try: H.cmd_between((0, 0), (2, 0)); raise AssertionError
    except ValueError: pass


def test_the_walks_do_what_the_page_says():
    tr = H.make_trajectory("t/1", 4000); c, s, m = tr["cmds"], tr["states"], tr["modes"].astype(int)
    assert len(c) == len(m) == 4000 and len(s) == 4001 and s.min() >= 0 and s.max() < H.G and set(c.tolist()) <= {0, 1, 2, 3, 4}
    moved = (s[1:] != s[:-1]).any(axis=1)
    for t in range(4000):
        vec = H.MOVE[int(c[t])]; nxt = (int(s[t][0]) + vec[0], int(s[t][1]) + vec[1])
        if moved[t]: assert (int(s[t + 1][0]), int(s[t + 1][1])) == nxt, t                                           # a move that happened is the move that was logged
        else: assert m[t] == 0 or not (0 <= nxt[0] < H.G and 0 <= nxt[1] < H.G), t                                  # a move that did not happen is a failed move of the tour or a wall
    assert (c == 4).sum() == 0                                                                                       # no mode issues S
    switches = np.nonzero(np.diff(m))[0] + 1; seg = np.diff(np.concatenate([[0], switches, [4000]]))
    assert (seg[:-1] >= H.SEG_MIN).all() and len(seg) >= 8 and set(m.tolist()) == {0, 1, 2}                          # every switch changes the mode (np.diff is nonzero at each cut)
    # the tour: a segment's cells lie on one rectangle and repeat with its length (failed moves aside)
    for a, b in zip(np.concatenate([[0], switches]), np.concatenate([switches, [4000]])):
        if m[a] != 0 or b - a < 40: continue
        cells = [tuple(map(int, x)) for x in s[a:b + 1]]; uniq = []; [uniq.append(x) for x in cells if not uniq or uniq[-1] != x]
        ring_len = len(set(uniq)); assert 4 <= ring_len <= 20 and len(set(uniq[:ring_len])) == ring_len
        assert all(uniq[i] == uniq[i % ring_len] for i in range(len(uniq))), (a, b)
        fails = sum(1 for i in range(a, b) if not moved[i]); assert fails <= 0.15 * (b - a) + 3
    # the drift: keeps its command about 85 % of the time away from walls, and turns round after a wall hit
    keep = turn_ok = n_keep = n_wall = 0
    for a, b in zip(np.concatenate([[0], switches]), np.concatenate([switches, [4000]])):
        if m[a] != 1: continue
        for t in range(a + 1, b):
            if not moved[t - 1]: n_wall += 1; turn_ok += int(c[t] == H.OPPOSITE[int(c[t - 1])])
            else: n_keep += 1; keep += int(c[t] == c[t - 1])
    assert n_wall > 20 and turn_ok == n_wall and 0.78 <= keep / n_keep <= 0.95, (keep, n_keep, turn_ok, n_wall)
    # the random walk uses all four moves in similar numbers
    cnt = np.bincount(c[m == 2], minlength=5); assert cnt[4] == 0 and cnt[:4].min() > 0.15 * cnt.sum()
    # seeded
    again = H.make_trajectory("t/1", 4000); assert all((tr[k] == again[k]).all() for k in tr) and not (H.make_trajectory("t/2", 4000)["cmds"] == c).all()
    # the first segment of a walk has a mode that is one of three; the next differs
    assert m[switches[0]] != m[0]


# ---------------------------------------------------------------- 2 Betti numbers
def test_betti_numbers_on_hand_made_graphs():
    assert H.betti_graph([1, 2, 3], [(1, 2), (2, 3)]) == (3, 2, 1, 0)                                                # a path
    assert H.betti_graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)]) == (4, 4, 1, 1)                             # a square
    assert H.betti_graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 1)]) == (5, 6, 1, 2)          # a figure of eight: two loops
    assert H.betti_graph([1, 2, 3, 4, 5, 6], [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]) == (6, 6, 2, 2)       # two separate triangles
    assert H.betti_graph([1, 2, 3], [(1, 2), (2, 1), (1, 2)]) == (3, 1, 2, 0)                                        # repeated links count once; a point with no link is its own piece
    assert H.betti_graph([7], []) == (1, 0, 1, 0) and H.betti_graph([], []) == (0, 0, 0, 0)
    assert H.betti_window([1, 2, 3, 4, 1]) == (4, 4, 1, 1)                                                           # round a square and home
    assert H.betti_window([1, 2, 1, 2, 1, 2]) == (2, 1, 1, 0)                                                        # back and forth: no loop
    assert H.betti_window([5, 5, 5, 5]) == (1, 0, 1, 0)                                                              # standing still
    assert H.betti_window([1, 2, 3, 3, 3, 4, 1]) == (4, 4, 1, 1)                                                     # stays do not matter
    assert H.betti_window([0, 1, 2, 9, 8, 7, 0]) == (6, 6, 1, 1)
    rng = random.Random(1)
    for _ in range(200):                                                                                             # a run of consecutive cells is always one piece, and beta1 = links - points + 1
        w = [rng.randrange(12) for _ in range(rng.randint(2, 40))]; p, l, b0, b1 = H.betti_window(w); assert b0 == 1 and b1 == l - p + 1 and b1 >= 0


# ---------------------------------------------------------------- 3 the views
def small_traj():
    """A hand-made log of 80 commands round a 2 x 2 loop (period 4): R D L U R D L U ..."""
    pat = [3, 1, 2, 0]; cmds = np.array([pat[i % 4] for i in range(80)], np.int8); cells = [(0, 0)]
    for c in cmds: cells.append((cells[-1][0] + H.MOVE[int(c)][0], cells[-1][1] + H.MOVE[int(c)][1]))
    return dict(cmds=cmds, states=np.array(cells, np.int16), modes=np.zeros(80, np.int8))


def test_the_views_are_the_window_the_cell_and_the_label_of_the_step():
    tr = small_traj(); F = H.features(tr); n = 80 - H.WINDOW
    assert F["window"].shape == (n, 50) and F["xy"].shape == (n, 2) and F["eng"].shape == (n, 26) and F["hist"].shape == (n, 5) and F["short"].shape == (n, 3)
    for j in range(n):
        t = H.WINDOW + j
        assert (F["window"][j] == tr["cmds"][t - 50:t]).all() and F["y_next"][j] == tr["cmds"][t] and F["y_mode"][j] == 0 and tuple(F["xy"][j]) == tuple(tr["states"][t])
        assert tuple(F["short"][j]) == tuple(tr["cmds"][t - 3:t]) and F["hist"][j].sum() == 50
    assert (F["beta0"] == 1).all() and (F["beta1"] == 1).all() and (F["beta1_10"] == 1).all() and (F["beta1_25"] == 1).all() and (F["points"] == 4).all() and (F["links"] == 4).all()      # round a loop of 4 cells
    assert (F["hist"][:, 0] == 12).all() or (F["hist"][:, 0] >= 12).all()                                            # 50 commands of a period-4 log hold 12 or 13 of each
    assert set(F["hist"][0].tolist()) <= {12.0, 13.0, 0.0}
    # the repeat numbers: in a period-4 log, a command equals the one k steps earlier for all positions exactly when k is a multiple of 4
    for k in range(1, 26): assert abs(F["eng"][0, k - 1] - (1.0 if k % 4 == 0 else 0.0)) < 1e-6, k
    assert abs(F["eng"][0, 25] - 1.0) < 1e-6                                                                         # the command changes at every step
    # a straight run: no change, no loop
    st = np.array([[i % 7, 0] for i in range(81)], np.int16); tr2 = dict(cmds=np.full(80, 3, np.int8), states=st, modes=np.ones(80, np.int8)); F2 = H.features(tr2)
    assert (F2["eng"][:, 25] == 0).all() and (F2["eng"][:, :25] == 1).all() and (F2["y_mode"] == 1).all()
    X, cat = H.view(F, "A_window"); assert X.shape == (n, 52) and cat == list(range(2, 52))
    X, cat = H.view(F, "B_betti"); assert X.shape == (n, 4) and cat is None and (X[:, 2] == 1).all() and (X[:, 3] == 1).all()
    assert H.view(F, "B_betti3")[0].shape == (n, 6) and H.view(F, "C_counts")[0].shape == (n, 4) and H.view(F, "C_hist")[0].shape == (n, 7) and H.view(F, "D_short")[0].shape == (n, 5) and H.view(F, "D_short")[1] == [2, 3, 4]
    assert H.view(F, "A_eng")[0].shape == (n, 28) and H.view(F, "A_all")[0].shape == (n, 78) and H.view(F, "state")[0].shape == (n, 2)
    try: H.view(F, "nope"); raise AssertionError
    except KeyError: pass
    assert set(H.ARMS) == {a for a in H.ARMS if H.view(F, a)}


def test_the_betti_window_is_the_51_cells_from_fifty_steps_ago_to_now():
    cells = [(i % 5, 0) for i in range(30)] + [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)] * 12; cmds = []; st = [cells[0]]
    for a, b in zip(cells[:-1], cells[1:]): cmds.append(H.cmd_between(a, b) if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 else 3)
    tr = dict(cmds=np.array(cmds, np.int8), states=np.array(cells, np.int16), modes=np.zeros(len(cmds), np.int8)); F = H.features(tr)
    cid = [y * H.G + x for x, y in cells]
    for j in (0, 5, len(F["beta1"]) - 1):
        t = H.WINDOW + j; assert (F["points"][j], F["links"][j], F["beta0"][j], F["beta1"][j]) == H.betti_window(cid[t - 50:t + 1]) and F["beta1_10"][j] == H.betti_window(cid[t - 10:t + 1])[3] and F["beta1_25"][j] == H.betti_window(cid[t - 25:t + 1])[3]


# ---------------------------------------------------------------- 4 the readers and the scores
def test_match_next_f1_score_and_fit_predict():
    per = bytes([3, 1, 2, 0] * 12 + [3]); W = np.frombuffer(per, np.uint8)[-50:].astype(np.int8)[None, :]
    assert H.match_next(W)[0] == 1                                                                                   # R D L U R D L U ... R then D
    one = np.array([[0] * 49 + [1]], np.int8); assert H.match_next(one)[0] == 1                                       # nothing repeats: the last command
    w = np.array([[1, 2, 3, 0] + [4] * 43 + [1, 2, 3]], np.int8); assert H.match_next(w)[0] == 0                       # "1 2 3" occurred at the start and 0 followed it
    w = np.array([[0] * 40 + [1, 2, 3, 0, 4, 0, 0, 1, 2, 3]], np.int8); assert H.match_next(w)[0] == 0                # "1 2 3" occurred before and 0 followed it
    w = np.array([[4, 4, 4, 1, 2, 3, 3, 3, 3, 3] * 5], np.int8); assert H.match_next(w)[0] == 4
    # the search never reads past the window and never matches the suffix with itself
    w = np.array([[0] * 49 + [4]], np.int8); assert H.match_next(w)[0] == 4
    pred = np.array([0, 0, 1, 1, 2, 0], np.int8); truth = np.array([0, 1, 1, 1, 2, 2], np.int8)
    assert abs(H.f1(pred, truth, 1) - 0.8) < 1e-9 and H.f1(pred, truth, 3) == 0.0 and abs(H.f1(pred, truth, 0) - 0.5) < 1e-9 and H.f1(truth, truth, 2) == 1.0
    F = dict(y_mode=np.array([0, 0, 1, 2, 2, 2], np.int8), y_next=np.array([0, 1, 1, 4, 4, 2], np.int8))
    s = H.score(np.array([0, 1, 1, 2, 2, 0], np.int8), np.array([0, 1, 0, 4, 4, 2], np.int8), F)
    assert abs(s["mode"] - 4 / 6) < 1e-9 and abs(s["next"] - 5 / 6) < 1e-9 and abs(s["tour_f1"] - H.f1(np.array([0, 1, 1, 2, 2, 0]), F["y_mode"], 0)) < 1e-12
    assert s["next_by_mode"] == {"tour": 1.0, "drift": 0.0, "random": 1.0} and s["mode_by_mode"] == {"tour": 0.5, "drift": 1.0, "random": 2 / 3}
    s2 = H.score(None, np.array([0, 1, 0, 4, 4, 2], np.int8), F); assert s2["mode"] is None and s2["tour_f1"] is None and s2["mode_by_mode"] is None and abs(s2["next"] - 5 / 6) < 1e-9
    rng = np.random.default_rng(0); X = rng.integers(0, 5, (3000, 4)).astype(np.float32); y = ((X[:, 0] + X[:, 2]) % 3).astype(np.int8)
    p = H.fit_predict(X[:2400], y[:2400], X[2400:], None, 0); assert (p == y[2400:]).mean() > 0.9
    X2 = rng.integers(0, 5, (3000, 3)).astype(np.float32); y2 = (X2[:, 1] == 3).astype(np.int8)
    p2 = H.fit_predict(X2[:2400], y2[:2400], X2[2400:], [0, 1, 2], 0); assert (p2 == y2[2400:]).mean() > 0.97                # categorical columns are accepted
    assert (H.fit_predict(X[:2400], y[:2400], X[2400:], None, 0) == p).all()                                          # seeded


# ---------------------------------------------------------------- 5 the run
TINY = dict(seeds=(1,), train_traj=2, test_traj=1, steps=300)


def test_the_run_is_seeded_keeps_train_and_test_walks_apart_and_counts_every_step_once():
    a = H.run_histshape(arms=("state", "B_betti", "A_window"), **TINY); b = H.run_histshape(arms=("state", "B_betti", "A_window"), **TINY)
    assert a["digest"] == b["digest"] and a["runs"] == b["runs"]
    r = a["runs"]["1"]; assert r["rows_train"] == 2 * 250 and r["rows_test"] == 250 and len(r["trajectories"]) == 3 and r["beta0_min"] == r["beta0_max"] == 1 and abs(sum(r["mode_share_test"]) - 1) < 1e-9
    assert set(r["arms"]) == {"majority", "state", "B_betti", "A_window", "A_match"} and r["arms"]["A_match"]["mode"] is None and r["arms"]["majority"]["mode"] is not None
    # the walks are named by seed and index, the last ones are the test walks, none repeats another
    seen = set()
    for i in range(3): seen.add(H.make_trajectory(f"hist/1/{i}", 300)["cmds"].tobytes())
    assert len(seen) == 3
    c = H.run_histshape(seeds=(2,), arms=("state",), **{k: v for k, v in TINY.items() if k != "seeds"}); assert c["digest"] != a["digest"]
    # the majority arm answers with the commonest label of the TRAINING rows
    F = [H.features(H.make_trajectory(f"hist/1/{i}", 300)) for i in range(3)]; tr, te = H._cat(F[:2]), F[2]
    maj = int(np.bincount(tr["y_mode"], minlength=3).argmax()); assert abs(r["arms"]["majority"]["mode"] - float((te["y_mode"] == maj).mean())) < 1e-9


def test_the_quick_run_the_printout_and_the_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): H.main(["--quick", "--out", out])
        res = json.load(open(out)); text = buf.getvalue()
        assert res["quick"] and res["steps"] == 600 and res["seeds"] == [1] and res["train_traj"] == 3 and res["test_traj"] == 2 and "digest" in text and "HEADLINE" in text and "(headline)" in text
        assert len(H.verdict_histshape(res)) == 13 and text.count("[claim") == 9 and text.count("[guard") == 2 and text.count("[instrument") == 2 and "B_betti3" in text and "A_match" in text and "majority" in text
        assert set(res["runs"]["1"]["arms"]) == {"majority", *H.ARMS, "A_match"}


# ---------------------------------------------------------------- 6 the reading rules
def mk(maj=0.40, state=0.45, bm=0.60, bf1=0.80, bn=0.30, wm=0.90, wf1=0.85, wn=0.70, sn=0.29, ch=0.50, b3=0.66, b1diff=3.0, seeds=(1, 2), walks=20, steps=10000, drop=()):
    """A hand-made result. Each rule reads on these numbers as the comments say; `drop` removes views."""
    def arm(mode=None, f1=None, nxt=None): return dict(mode=mode, tour_f1=f1, next=nxt, next_by_mode={"tour": nxt, "drift": nxt, "random": nxt} if nxt is not None else None, mode_by_mode=None)
    arms = {"majority": arm(maj, 0.0, 0.25), "state": arm(state, 0.2, sn), "B_betti": arm(bm, bf1, bn), "B_betti3": arm(b3, bf1, bn), "C_counts": arm(0.6, 0.7, 0.3), "C_hist": arm(ch, 0.6, 0.3), "D_short": arm(0.5, 0.5, 0.6),
            "A_window": arm(wm - 0.1, wf1 - 0.1, wn - 0.1), "A_eng": arm(wm - 0.05, wf1 - 0.05, wn - 0.2), "A_all": arm(wm, wf1, wn - 0.05), "A_match": arm(None, None, wn)}
    for d in drop: del arms[d]
    run = lambda: dict(rows_train=int(0.7 * walks * (steps - 50)), rows_test=walks * (steps - 50) - int(0.7 * walks * (steps - 50)), mode_share_test=[maj, (1 - maj) / 2, (1 - maj) / 2],
                       trajectories=[dict(steps=steps, modes_present=[0, 1, 2], segments=60) for _ in range(walks)], beta0_min=1, beta0_max=1, beta1_mean_by_mode={"tour": 1.0, "drift": 0.5, "random": 1.0 + b1diff}, arms=copy.deepcopy(arms))
    return dict(seeds=list(seeds), train_traj=14, test_traj=6, steps=steps, window=50, arms=list(H.ARMS), runs={str(s): run() for s in seeds}, digest="x", seconds=1.0)


def v(**kw): return [x for _, x in H.verdict_histshape(mk(**kw))]


def test_every_reading_rule_reads_as_it_should_on_made_up_results_and_flips_where_it_should():
    base = H.verdict_histshape(mk()); assert len(base) == 13 and [k for k, _ in base] == [k for k, _ in H.HS_LINES]
    # base: the window beats B by far on MODE and NEXT, the pair beats the cell on MODE
    assert [x for _, x in base] == [True, True, True, True, True, True, False, True, True, True, True, True, False], base
    # lines 7 and 13 (matching): they read yes when B reaches the window's level
    assert v(bm=0.86)[6] is True and v(bm=0.84)[6] is False and v(bm=0.851)[6] is True and v(bm=0.849)[6] is False                # the edge is 0.90 - 0.05; floating point makes the exact edge unreadable, so the test stays 0.001 away from it
    assert v(bn=0.651)[12] is True and v(bn=0.649)[12] is False and v(bn=0.60)[12] is False and v(bn=0.65)[8] is False
    # line 1: a walk with the wrong length, a missing mode, a row count that does not add up
    r = mk(); r["runs"]["1"]["trajectories"][3]["steps"] = 9999; assert H.verdict_histshape(r)[0][1] is False
    r = mk(); r["runs"]["2"]["trajectories"][0]["modes_present"] = [0, 1]; assert H.verdict_histshape(r)[0][1] is False
    r = mk(); r["runs"]["1"]["rows_test"] += 1; assert H.verdict_histshape(r)[0][1] is False
    r = mk(); del r["runs"]["1"]["trajectories"][0]; assert H.verdict_histshape(r)[0][1] is False
    # line 2: beta0 not 1
    r = mk(); r["runs"]["1"]["beta0_max"] = 2; assert H.verdict_histshape(r)[1][1] is False
    r = mk(); r["runs"]["2"]["beta0_min"] = 0; assert H.verdict_histshape(r)[1][1] is False
    # line 3: the cell alone gives the mode away when it is more than 12 points above the majority rate
    assert v(state=0.519)[2] is True and v(state=0.521)[2] is False
    # line 4: random must have at least 1.0 more loops than tour
    assert v(b1diff=1.001)[3] is True and v(b1diff=0.999)[3] is False
    # line 5: the best window reader reaches 0.80
    assert v(wm=0.801)[4] is True and v(wm=0.799)[4] is False
    # line 6: B beats the cell by 10 points
    assert v(bm=0.551, state=0.45)[5] is True and v(bm=0.549, state=0.45)[5] is False
    # line 8: the F1 of B reaches 0.9 of the best window F1
    assert v(bf1=0.766, wf1=0.85)[7] is True and v(bf1=0.764, wf1=0.85)[7] is False
    # line 9: the best window reader beats B on NEXT by 10 points
    assert v(wn=0.401, bn=0.30)[8] is True and v(wn=0.399, bn=0.30)[8] is False
    # line 10: B at most 2 points above the cell on NEXT
    assert v(bn=0.309, sn=0.29)[9] is True and v(bn=0.311, sn=0.29)[9] is False
    # line 11: B beats C_hist (strictly)
    assert v(bm=0.6, ch=0.5999)[10] is True and v(bm=0.6, ch=0.6)[10] is False
    # line 12: three scales beat one by 3 points
    assert v(bm=0.6, b3=0.631)[11] is True and v(bm=0.6, b3=0.629)[11] is False
    # the best reader is the largest of the three (MODE, F1) and of the four (NEXT): A_match counts for NEXT and not for MODE or F1
    r = mk(); r["runs"]["1"]["arms"]["A_match"]["next"] = 0.95; r["runs"]["2"]["arms"]["A_match"]["next"] = 0.95; assert H.best_window(r, "next", H.WINDOW_READERS + ("A_match",)) == 0.95 and H.best_window(r, "mode", H.WINDOW_READERS) == 0.90
    assert H.best_window(mk(), "tour_f1", H.WINDOW_READERS) == 0.85
    # the figure of a view is the mean over the seeds
    r = mk(); r["runs"]["1"]["arms"]["B_betti"]["mode"] = 0.50; r["runs"]["2"]["arms"]["B_betti"]["mode"] = 0.70; assert abs(H.mean_over(r, "B_betti", "mode") - 0.60) < 1e-12 and H.mean_over(r, "nope", "mode") is None
    # a view missing: the rules that need it read None, the others still read
    out = v(drop=("B_betti",)); assert out[5] is None and out[6] is None and out[7] is None and out[9] is None and out[10] is None and out[11] is None and out[12] is None and out[8] is None and out[4] is True
    out = v(drop=("A_window", "A_eng", "A_all", "A_match")); assert out[4] is None and out[6] is None and out[7] is None and out[8] is None and out[12] is None and out[5] is True
    out = v(drop=("state",)); assert out[2] is None and out[5] is None and out[9] is None
    out = v(drop=("C_hist",)); assert out[10] is None and out[11] is True
    out = v(drop=("B_betti3",)); assert out[11] is None and out[10] is True
    # the headline is lines 7, 8, 13
    assert H.HS_HEADLINE == (7, 8, 13)
    allmatch = H.verdict_histshape(mk(bm=0.88, bf1=0.85, bn=0.70)); assert all(allmatch[i - 1][1] for i in H.HS_HEADLINE) and allmatch[8][1] is False


# ---------------------------------------------------------------- 7 the page
PAGE_SHA256 = "df14475473c1e221543d4806fbda5add7e1150b8cfe211b3ed80cf1f611e13af"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_thirteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(H.HS_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [98, 99, 70, 93, 80, 90, 30, 45, 95, 55, 40, 65, 2]
    assert H.HS_HEADLINE == (7, 8, 13) and "**Headline** (lines 7, 8 and 13)" in head and "My probabilities are 30 %, 45 % and 2 %" in head and "my single number is **1 %**" in head
    assert (H.G, H.CMDS, H.MODES, H.STEPS, H.WINDOW, H.SEG_MIN, H.SEG_EXTRA_MEAN, H.TOUR_SLIP, H.DRIFT_KEEP, H.SEEDS, H.TRAIN_TRAJ, H.TEST_TRAJ, H.LAGS) == (7, "UDLRS", ("tour", "drift", "random"), 10000, 50, 40, 110, 0.05, 0.85, (1, 2, 3), 14, 6, 25)
    assert H.HGB == dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10) and H.ARMS == ("state", "B_betti", "B_betti3", "C_counts", "C_hist", "D_short", "A_window", "A_eng", "A_all")
    assert (H.H_MODE_A, H.H_STATE_GAP, H.H_BETTI_OVER_STATE, H.H_MATCH, H.H_F1, H.H_NEXT_GAP, H.H_NEXT_FLAT, H.H_MULTI, H.H_LOOPS) == (0.80, 0.12, 0.10, 0.05, 0.9, 0.10, 0.02, 0.03, 1.0)
    assert H.TRAIN_TRAJ * (H.STEPS - H.WINDOW) == 139300 and H.TEST_TRAJ * (H.STEPS - H.WINDOW) == 59700 and "139,300 scored steps" in head and "59,700 scored steps" in head
    for needle in ("Concept-map item 105", "`python/axi/experiments/histshape.py`", "python -m axi.experiments.histshape --out axi_histshape.json", "Validating Topological Cobordisms (The Time Invariant)", "I attribute none of it to him", "every choice below is mine, not Rajnish's",
                   "beta0 is 1 on every window and carries no news at all", "a sliding window of the last 50 text commands", "just as effectively as reading 50,000 tokens of chat history", "chronological intelligence is a geometric property",
                   "Perea and Harer, 2015", "Littman, Sutton and Singh, 2002", "Takens, 1981", "I did see them", "I did **not** look at beta1 by mode", "`histshape.verdict_histshape`", "a different complex", "axi_history_shape.ipynb"):
        assert needle in head, needle


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

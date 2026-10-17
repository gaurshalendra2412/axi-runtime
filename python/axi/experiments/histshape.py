"""Engineering for docs/HISTORY_SHAPE_TEST.md (concept-map item 105): does a pair of Betti numbers of the recent history carry what the last 50 commands carry?

CPU only, no language model, no internet (numpy + scikit-learn). An agent walks a 7 x 7 grid for 10,000 steps under a hidden mode that changes every 150 steps or so:
  tour    it goes round a rectangle that passes through its cell (a cycle), with a 5 % chance per step that the move fails
  drift   it keeps its last command with chance 0.85 (momentum) and turns round at a wall
  random  it picks any of the four moves
The commands are U, D, L, R, S (stay). At every step a predictor sees ONE of several views of the past and is asked (MODE) which mode the agent is in and (NEXT) which command comes next:
  state      the current cell only
  B_betti    the current cell and (beta0, beta1) of the graph of the last 50 moves   <- the roadmap's "Arm B"
  B_betti3   the current cell and beta1 over the last 10, 25 and 50 moves (a more generous B, my extension)
  C_counts   the current cell and the number of distinct cells and of distinct links in the last 50 moves (beta1 = links - cells + beta0, so this holds the same news)
  C_hist     the current cell and how often each of the five commands occurs in the last 50 (a five-number summary that is not topological)
  D_short    the current cell and the last three commands
  A_window   the current cell and the last 50 commands            <- the roadmap's "Arm A"
  A_eng      the current cell and 26 numbers computed from the last 50 commands (how often a command equals the one k steps earlier, for k = 1..25, and how often the command changes)
  A_all      the current cell, the last 50 commands and those 26 numbers
  A_match    (NEXT only, nothing is trained) the longest earlier repeat of the last commands inside the window, and what followed it
The trained views use one gradient-boosted tree model each (scikit-learn, fixed settings). Nothing here is measured on an agent that is a language model; it is an information test: how much of the answer is in each view.
The graph of the last 50 moves: its points are the cells visited, its links the distinct pairs of cells between which the agent moved; a walk is connected, so beta0 is 1 on every window and beta1, the number of independent loops, is links - points + 1."""
import hashlib, json, os, random, sys, time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- the world (fixed in docs/HISTORY_SHAPE_TEST.md before the run)
G = 7
CMDS = "UDLRS"
MOVE = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
MODES = ("tour", "drift", "random")
STEPS = 10000
WINDOW = 50
SEG_MIN, SEG_EXTRA_MEAN = 40, 110
TOUR_SLIP, DRIFT_KEEP = 0.05, 0.85
SEEDS = (1, 2, 3)
TRAIN_TRAJ, TEST_TRAJ = 14, 6
LAGS = 25
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
ARMS = ("state", "B_betti", "B_betti3", "C_counts", "C_hist", "D_short", "A_window", "A_eng", "A_all")
WINDOW_READERS = ("A_window", "A_eng", "A_all")
H_MODE_A, H_STATE_GAP, H_BETTI_OVER_STATE, H_MATCH, H_F1, H_NEXT_GAP, H_NEXT_FLAT, H_MULTI = 0.80, 0.12, 0.10, 0.05, 0.9, 0.10, 0.02, 0.03
H_LOOPS = 1.0


def cell_id(x: int, y: int) -> int: return y * G + x


# ---------------------------------------------------------------- the walker
def tour_cycle(rng: random.Random, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A rectangle of w x h cells (2 to 6 each way) with the walker's cell on its perimeter, as the cells in order, starting at the walker's cell, in a random direction."""
    for _ in range(200):
        w, h = rng.randint(2, 6), rng.randint(2, 6); x0, y0 = rng.randint(0, G - w), rng.randint(0, G - h)
        ring = [(x, y0) for x in range(x0, x0 + w)] + [(x0 + w - 1, y) for y in range(y0 + 1, y0 + h)] + [(x, y0 + h - 1) for x in range(x0 + w - 2, x0 - 1, -1)] + [(x0, y) for y in range(y0 + h - 2, y0, -1)]
        if pos in ring: break
    else:
        x0, y0 = min(pos[0], G - 2), min(pos[1], G - 2); ring = [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)]
    i = ring.index(pos); ring = ring[i:] + ring[:i]
    if rng.random() < 0.5: ring = [ring[0]] + ring[:0:-1]
    return ring


def cmd_between(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    d = (b[0] - a[0], b[1] - a[1])
    for c, m in MOVE.items():
        if m == d and c != 4: return c
    raise ValueError((a, b))


def make_trajectory(seed: str, steps: int = STEPS) -> dict:
    """One walk. states[t] is the cell before command t (steps + 1 of them), cmds[t] the command logged at step t, modes[t] the mode that chose it."""
    rng = random.Random(seed); pos = (rng.randrange(G), rng.randrange(G)); states = [pos]; cmds: List[int] = []; modes: List[int] = []; mode = -1; prev = rng.randrange(4); blocked = False
    while len(cmds) < steps:
        mode = rng.choice([m for m in range(3) if m != mode]); n = min(SEG_MIN + int(rng.expovariate(1.0 / SEG_EXTRA_MEAN)), steps - len(cmds))
        if mode == 0: ring = tour_cycle(rng, pos); k = 0
        if mode == 1: prev = rng.randrange(4); blocked = False
        for _ in range(n):
            if mode == 0:
                c = cmd_between(ring[k], ring[(k + 1) % len(ring)]); ok = rng.random() >= TOUR_SLIP
                if ok: k = (k + 1) % len(ring); pos = ring[k]
            else:
                if mode == 1:
                    if blocked: c = OPPOSITE[prev]
                    elif rng.random() < DRIFT_KEEP: c = prev
                    else: c = rng.randrange(4)
                else: c = rng.randrange(4)
                nx, ny = pos[0] + MOVE[c][0], pos[1] + MOVE[c][1]; blocked = not (0 <= nx < G and 0 <= ny < G)
                if not blocked: pos = (nx, ny)
                prev = c
            cmds.append(c); modes.append(mode); states.append(pos)
    return dict(cmds=np.array(cmds, dtype=np.int8), states=np.array(states, dtype=np.int16), modes=np.array(modes, dtype=np.int8))


# ---------------------------------------------------------------- the views
def betti_graph(points: Sequence, links: Sequence[Tuple]) -> Tuple[int, int, int, int]:
    """(points, links, beta0, beta1) of a graph: beta0 = connected components by union-find, beta1 = links - points + beta0 (the number of independent loops)."""
    pts = sorted(set(points)); idx = {c: i for i, c in enumerate(pts)}; parent = list(range(len(pts))); ls = {(min(a, b), max(a, b)) for a, b in links if a != b}
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in ls:
        ra, rb = find(idx[a]), find(idx[b])
        if ra != rb: parent[ra] = rb
    b0 = len({find(i) for i in range(len(pts))}); return len(pts), len(ls), b0, len(ls) - len(pts) + b0


def betti_window(cells: Sequence[int]) -> Tuple[int, int, int, int]:
    """The graph of a run of cells: its points are the distinct cells, its links the distinct pairs between which the walk moved (a stay adds none)."""
    return betti_graph(cells, list(zip(cells[:-1], cells[1:])))


def features(tr: dict) -> dict:
    """Every view at every scored step t = WINDOW .. len - 1, plus the two labels."""
    cmds = tr["cmds"]; st = tr["states"]; n = len(cmds) - WINDOW; cid = (st[:, 1].astype(np.int64) * G + st[:, 0]).tolist()
    idx = np.arange(WINDOW, len(cmds)); W = np.stack([cmds[t - WINDOW:t] for t in idx]).astype(np.int8)
    out = dict(xy=st[idx].astype(np.float32), window=W, y_mode=tr["modes"][idx].astype(np.int8), y_next=cmds[idx].astype(np.int8))
    pts = np.zeros(n, np.int16); lks = np.zeros(n, np.int16); b0 = np.zeros(n, np.int16); b1 = np.zeros(n, np.int16); b10 = np.zeros(n, np.int16); b25 = np.zeros(n, np.int16)
    for j, t in enumerate(idx):
        p, l, c0, c1 = betti_window(cid[t - WINDOW:t + 1]); pts[j], lks[j], b0[j], b1[j] = p, l, c0, c1
        b10[j] = betti_window(cid[t - 10:t + 1])[3]; b25[j] = betti_window(cid[t - 25:t + 1])[3]
    out.update(points=pts, links=lks, beta0=b0, beta1=b1, beta1_10=b10, beta1_25=b25)
    out["hist"] = np.stack([(W == c).sum(axis=1) for c in range(5)], axis=1).astype(np.float32)
    out["short"] = W[:, -3:].astype(np.float32)
    eng = [(W[:, k:] == W[:, :-k]).mean(axis=1) for k in range(1, LAGS + 1)] + [(W[:, 1:] != W[:, :-1]).mean(axis=1)]
    out["eng"] = np.stack(eng, axis=1).astype(np.float32)
    return out


def view(F: dict, arm: str) -> Tuple[np.ndarray, Optional[List[int]]]:
    """(matrix, indices of the categorical columns) of an arm."""
    xy = F["xy"]; f = lambda *cols: np.column_stack(cols).astype(np.float32)
    if arm == "state": return xy, None
    if arm == "B_betti": return f(xy, F["beta0"], F["beta1"]), None
    if arm == "B_betti3": return f(xy, F["beta0"], F["beta1_10"], F["beta1_25"], F["beta1"]), None
    if arm == "C_counts": return f(xy, F["points"], F["links"]), None
    if arm == "C_hist": return f(xy, F["hist"]), None
    if arm == "D_short": return f(xy, F["short"]), [2, 3, 4]
    if arm == "A_window": return f(xy, F["window"]), list(range(2, 2 + WINDOW))
    if arm == "A_eng": return f(xy, F["eng"]), None
    if arm == "A_all": return f(xy, F["window"], F["eng"]), list(range(2, 2 + WINDOW))
    raise KeyError(arm)


def match_next(W: np.ndarray) -> np.ndarray:
    """A_match: the longest earlier repeat (8 commands down to 2) of the last commands inside the window; the answer is what followed the most recent such repeat, else the last command."""
    out = np.empty(len(W), np.int8)
    for i in range(len(W)):
        s = bytes(W[i].tolist()); head = s[:-1]; pred = s[-1]
        for k in range(8, 1, -1):
            j = head.rfind(s[-k:])
            if j >= 0 and j + k < len(s): pred = s[j + k]; break
        out[i] = pred
    return out


def fit_predict(Xtr, ytr, Xte, cat, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=seed, categorical_features=cat, **HGB); m.fit(Xtr, ytr); return m.predict(Xte).astype(np.int8)


def f1(pred: np.ndarray, truth: np.ndarray, cls: int) -> float:
    tp = int(((pred == cls) & (truth == cls)).sum()); fp = int(((pred == cls) & (truth != cls)).sum()); fn = int(((pred != cls) & (truth == cls)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


# ---------------------------------------------------------------- the run
def _cat(parts: Sequence[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def score(pred_mode, pred_next, F) -> dict:
    ym, yn = F["y_mode"], F["y_next"]
    by_mode = {MODES[m]: float((pred_next[ym == m] == yn[ym == m]).mean()) for m in range(3) if (ym == m).any()}
    mode_by_mode = {MODES[m]: float((pred_mode[ym == m] == m).mean()) for m in range(3) if (ym == m).any()} if pred_mode is not None else None
    return dict(mode=None if pred_mode is None else float((pred_mode == ym).mean()), tour_f1=None if pred_mode is None else f1(pred_mode, ym, 0), next=None if pred_next is None else float((pred_next == yn).mean()),
                next_by_mode=by_mode if pred_next is not None else None, mode_by_mode=mode_by_mode)


def run_histshape(seeds: Sequence[int] = SEEDS, train_traj: int = TRAIN_TRAJ, test_traj: int = TEST_TRAJ, steps: int = STEPS, arms: Sequence[str] = ARMS, progress=lambda *a: None) -> dict:
    t0 = time.time(); dig = hashlib.sha256(); res = dict(seeds=list(seeds), train_traj=train_traj, test_traj=test_traj, steps=steps, window=WINDOW, arms=list(arms), runs={}, hgb=HGB)
    for sd in seeds:
        trajs = []
        for i in range(train_traj + test_traj):
            tr = make_trajectory(f"hist/{sd}/{i}", steps)
            for k in ("cmds", "states", "modes"): dig.update(tr[k].tobytes())
            trajs.append(tr)
        feats = [features(t) for t in trajs]; Ftr, Fte = _cat(feats[:train_traj]), _cat(feats[train_traj:])
        run = dict(rows_train=int(len(Ftr["y_mode"])), rows_test=int(len(Fte["y_mode"])), mode_share_test=[float((Fte["y_mode"] == m).mean()) for m in range(3)],
                   trajectories=[dict(steps=int(len(t["cmds"])), modes_present=sorted({int(m) for m in t["modes"]}), segments=int(1 + (np.diff(t["modes"]) != 0).sum())) for t in trajs],
                   beta0_min=int(min(F["beta0"].min() for F in feats)), beta0_max=int(max(F["beta0"].max() for F in feats)),
                   beta1_mean_by_mode={MODES[m]: float(np.concatenate([F["beta1"][F["y_mode"] == m] for F in feats]).mean()) for m in range(3)}, arms={})
        maj = int(np.bincount(Ftr["y_mode"], minlength=3).argmax()); majn = int(np.bincount(Ftr["y_next"], minlength=5).argmax())
        run["arms"]["majority"] = score(np.full(len(Fte["y_mode"]), maj, np.int8), np.full(len(Fte["y_next"]), majn, np.int8), Fte)
        for arm in arms:
            Xtr, cat = view(Ftr, arm); Xte, _ = view(Fte, arm)
            pm = fit_predict(Xtr, Ftr["y_mode"], Xte, cat, sd); pn = fit_predict(Xtr, Ftr["y_next"], Xte, cat, sd)
            run["arms"][arm] = score(pm, pn, Fte); progress(sd, arm, run["arms"][arm]["mode"], run["arms"][arm]["next"])
        run["arms"]["A_match"] = score(None, match_next(Fte["window"]), Fte)
        res["runs"][str(sd)] = run
    res["digest"] = dig.hexdigest()[:16]; res["seconds"] = round(time.time() - t0, 1); return res


# ---------------------------------------------------------------- the reading rules
HS_LINES = [
    ("guard", "every walk ran all its steps, every walk held all three modes at least once, and the number of scored steps is walks x (steps - 50)"),
    ("guard", "beta0 was 1 on every scored step (a run of consecutive cells is connected: this checks the Betti code and is not a finding)"),
    ("instrument", "MODE: the current cell alone is within 12 points of the majority rate (the cell does not give the mode away)"),
    ("instrument", "the loops have teeth: the mean of beta1 in random mode is at least 1.0 above its mean in tour mode"),
    ("claim", "MODE: the best of the three readers of the last 50 commands (A_window, A_eng, A_all) reaches at least 0.80"),
    ("claim", "MODE: B_betti beats the current cell alone by at least 10 points"),
    ("claim", "MODE: B_betti is within 5 points of the best reader of the last 50 commands (it matches)"),
    ("claim", "TOUR (is the agent going round a loop?), F1: B_betti reaches at least 0.9 of the best reader's F1"),
    ("claim", "NEXT: the best of the readers of the last 50 commands (A_window, A_eng, A_all, A_match) beats B_betti by at least 10 points"),
    ("claim", "NEXT: B_betti is at most 2 points above the current cell alone (the pair adds almost nothing to prediction)"),
    ("claim", "MODE: B_betti beats C_hist (the five command counts, a summary that is not topological)"),
    ("claim", "MODE: B_betti3 (beta1 over three scales) beats B_betti by at least 3 points"),
    ("claim", "NEXT: B_betti is within 5 points of the best reader of the last 50 commands (it matches)"),
]
HS_HEADLINE = (7, 8, 13)                                   # the roadmap's claim: Arm B matches Arm A on both questions


def mean_over(res: dict, arm: str, key: str) -> Optional[float]:
    v = [r["arms"][arm][key] for r in res["runs"].values() if arm in r["arms"] and r["arms"][arm].get(key) is not None]; return float(np.mean(v)) if v else None


def best_window(res: dict, key: str, arms: Sequence[str]) -> Optional[float]:
    v = [mean_over(res, a, key) for a in arms]; v = [x for x in v if x is not None]; return max(v) if v else None


def verdict_histshape(res: dict) -> List[Tuple[str, Optional[bool]]]:
    runs = res["runs"].values(); m = lambda arm, key: mean_over(res, arm, key); out: List[Tuple[str, Optional[bool]]] = []
    walks = res["train_traj"] + res["test_traj"]
    out.append(("guard", all(len(r["trajectories"]) == walks and all(t["steps"] == res["steps"] and t["modes_present"] == [0, 1, 2] for t in r["trajectories"]) and r["rows_train"] + r["rows_test"] == walks * (res["steps"] - res["window"]) for r in runs)))
    out.append(("guard", all(r["beta0_min"] == 1 and r["beta0_max"] == 1 for r in runs)))
    maj = float(np.mean([max(r["mode_share_test"]) for r in runs])); st = m("state", "mode")
    out.append(("instrument", None if st is None else st - maj <= H_STATE_GAP))
    out.append(("instrument", float(np.mean([r["beta1_mean_by_mode"]["random"] - r["beta1_mean_by_mode"]["tour"] for r in runs])) >= H_LOOPS))
    bw_m = best_window(res, "mode", WINDOW_READERS); bw_f = best_window(res, "tour_f1", WINDOW_READERS); bw_n = best_window(res, "next", WINDOW_READERS + ("A_match",))
    b, b3, cs = m("B_betti", "mode"), m("B_betti3", "mode"), m("state", "mode")
    out.append(("claim", None if bw_m is None else bw_m >= H_MODE_A))
    out.append(("claim", None if b is None or cs is None else b - cs >= H_BETTI_OVER_STATE))
    out.append(("claim", None if b is None or bw_m is None else b >= bw_m - H_MATCH))
    out.append(("claim", None if m("B_betti", "tour_f1") is None or not bw_f else m("B_betti", "tour_f1") >= H_F1 * bw_f))
    bn, sn = m("B_betti", "next"), m("state", "next")
    out.append(("claim", None if bn is None or bw_n is None else bw_n - bn >= H_NEXT_GAP))
    out.append(("claim", None if bn is None or sn is None else bn - sn <= H_NEXT_FLAT))
    ch = m("C_hist", "mode"); out.append(("claim", None if b is None or ch is None else b > ch))
    out.append(("claim", None if b is None or b3 is None else b3 - b >= H_MULTI))
    out.append(("claim", None if bn is None or bw_n is None else bn >= bw_n - H_MATCH))
    return out


# ---------------------------------------------------------------- printing and the command line
def _pc(x: Optional[float], k: int = 1) -> str: return "   -  " if x is None else f"{100 * x:.{k}f}%"


def print_histshape(res: dict) -> None:
    print(f"=== the shape of the history (seeds {res['seeds']}, {res['train_traj']} + {res['test_traj']} walks of {res['steps']} steps each, {res['seconds']} s, digest {res['digest']}) ===")
    print(f"{'arm':10s} {'MODE':>8s} {'TOUR F1':>8s} {'NEXT':>8s}   NEXT by true mode (tour, drift, random)")
    for arm in ["majority"] + list(res["arms"]) + ["A_match"]:
        if not any(arm in r["arms"] for r in res["runs"].values()): continue
        nm = [mean_over_mode(res, arm, md) for md in MODES]
        print(f"{arm:10s} {_pc(mean_over(res, arm, 'mode')):>8s} {('%.3f' % mean_over(res, arm, 'tour_f1')) if mean_over(res, arm, 'tour_f1') is not None else '   -':>8s} {_pc(mean_over(res, arm, 'next')):>8s}   " + "  ".join(_pc(x) for x in nm))


def mean_over_mode(res: dict, arm: str, md: str) -> Optional[float]:
    v = [r["arms"][arm]["next_by_mode"][md] for r in res["runs"].values() if arm in r["arms"] and r["arms"][arm].get("next_by_mode")]; return float(np.mean(v)) if v else None


def print_verdict_histshape(checks) -> None:
    print("--- reading rules ---")
    for i, (kind, v) in enumerate(checks, 1): print(f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}" + ("   (headline)" if i in HS_HEADLINE else ""))
    head = [checks[i - 1][1] for i in HS_HEADLINE]; print("HEADLINE (Arm B matches Arm A on both questions):", "ALL YES" if all(head) else "NOT all yes")


QUICK = dict(seeds=(1,), train_traj=3, test_traj=2, steps=600)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="the shape of the history (docs/HISTORY_SHAPE_TEST.md)")
    ap.add_argument("--out", default="axi_histshape.json"); ap.add_argument("--quick", action="store_true", help="a few seconds, small samples: to see that it runs, not to read")
    a = ap.parse_args(argv); kw = dict(QUICK) if a.quick else {}
    res = run_histshape(progress=lambda sd, arm, mo, nx: print(f"  seed {sd} {arm}: MODE {_pc(mo)} NEXT {_pc(nx)}", flush=True), **kw); res["quick"] = bool(a.quick)
    with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
    print_histshape(res); print_verdict_histshape(verdict_histshape(res)); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()

"""Implements concept-map item 105: RESULT of the history-shape test of 17 Oct 2026. Page: docs/HISTORY_SHAPE_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The file: docs/results/axi_histshape.json (the run: 3 seeds, 14 + 6 walks of 10,000 steps each, Python 3.13.16, numpy 2.5.3, scikit-learn 1.9.1, 977.1 s on two CPUs).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the structure of the file (seeds, sizes, scored steps = walks x (steps - 50), every accuracy a whole number of the scored test steps, the accuracy by mode adding up to the accuracy), and its sha256;
  2. the thirteen reading rules (10 yes; lines 7, 8 and 13 NO), the headline, and that the 'read' column of the page says the same;
  3. the 'what was measured' text of each line of the page, the tables and the figures quoted in the prose of the Result (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. that the part of the page above the Result is byte for byte what was written before the run;
  5. that the 60 walks replay: the same seeds give the saved digest (a few seconds);
  6. (heavy, about six minutes; set AXI_HEAVY=1, run by hand, not by the mutation check) that a second run of seed 1 gives the saved accuracies of seed 1.
What it does NOT show: anything about a language model reading a history, a different complex, or a world other than this one."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.histshape as H

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "HISTORY_SHAPE_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_histshape.json")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "df14475473c1e221543d4806fbda5add7e1150b8cfe211b3ed80cf1f611e13af"
FILE_SHA256 = "ebfa27391bddc3706fb5c1f7f29586c38e12c989075713d6a135b7bd8e8e1f17"
DIGEST = "c30005542a6567da"
PROBS = [98, 99, 70, 93, 80, 90, 30, 45, 95, 55, 40, 65, 2]
READ = [True] * 6 + [False, False] + [True] * 4 + [False]
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[:t.index(RESULT_MARK)]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- the figures, from the saved accuracies
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def seeds(res): return sorted(res["runs"], key=int)
def A(res, arm, key): return [res["runs"][s]["arms"][arm][key] for s in seeds(res)]
def M(res, arm, key): v = A(res, arm, key); return sum(v) / len(v)
def lohi(res, arm, key): v = A(res, arm, key); return f"{min(v) * 100:.1f}-{max(v) * 100:.1f}"
def MM(res, arm, mode, key="mode_by_mode"): v = [res["runs"][s]["arms"][arm][key][mode] for s in seeds(res)]; return sum(v) / len(v)
def B1(res, mode): v = [res["runs"][s]["beta1_mean_by_mode"][mode] for s in seeds(res)]; return sum(v) / len(v)
def best(res, key, arms): return max(M(res, a, key) for a in arms)
def bestname(res, key, arms): return max(arms, key=lambda a: M(res, a, key))
MODEARMS = ("A_window", "A_eng", "A_all")


def measured(res, i):
    st, bb, b3, ch, cc = (M(res, a, "mode") for a in ("state", "B_betti", "B_betti3", "C_hist", "C_counts")); maj = M(res, "majority", "mode"); bw = best(res, "mode", MODEARMS); bf = best(res, "tour_f1", MODEARMS)
    bn = best(res, "next", MODEARMS + ("A_match",)); nb, ns = M(res, "B_betti", "next"), M(res, "state", "next"); walks = res["train_traj"] + res["test_traj"]
    return {
        1: f"{len(seeds(res))} seeds x {walks} walks of {num(res['steps'])} steps, all three modes in every walk; {num(res['train_traj'] * (res['steps'] - res['window']))} training and {num(res['test_traj'] * (res['steps'] - res['window']))} test steps in each seed (= walks x ({num(res['steps'])} - {res['window']}))",
        2: f"beta0 was {min(res['runs'][s]['beta0_min'] for s in seeds(res))} on every scored step of every walk (smallest {min(res['runs'][s]['beta0_min'] for s in seeds(res))}, largest {max(res['runs'][s]['beta0_max'] for s in seeds(res))})",
        3: f"the cell alone {pc(st)}, the majority rate {pc(maj)}: {(st - maj) * 100:.1f} points above (limit 12)",
        4: f"mean beta1: random {B1(res, 'random'):.2f}, tour {B1(res, 'tour'):.2f}, drift {B1(res, 'drift'):.2f}; random minus tour {B1(res, 'random') - B1(res, 'tour'):.2f} (limit 1.0)",
        5: f"best reader of the window {pc(bw)} ({bestname(res, 'mode', MODEARMS)}; limit 80 %)",
        6: f"B_betti {pc(bb)} against the cell alone {pc(st)}: {(bb - st) * 100:.1f} points above (limit 10)",
        7: f"B_betti {pc(bb)} against the best reader {pc(bw)}: {(bw - bb) * 100:.1f} points short (limit 5)",
        8: f"F1 of B_betti {M(res, 'B_betti', 'tour_f1'):.3f} against {bf:.3f} for the best reader: {M(res, 'B_betti', 'tour_f1') / bf:.2f} of it (limit 0.90)",
        9: f"best reader of the window {pc(bn)} ({bestname(res, 'next', MODEARMS + ('A_match',))}) against B_betti {pc(nb)}: {(bn - nb) * 100:.1f} points above (limit 10)",
        10: f"B_betti {pc(nb)} against the cell alone {pc(ns)}: {(nb - ns) * 100:+.1f} points (limit 2)",
        11: f"B_betti {pc(bb)} against C_hist {pc(ch)}: {(bb - ch) * 100:+.1f} points",
        12: f"B_betti3 {pc(b3)} against B_betti {pc(bb)}: {(b3 - bb) * 100:+.1f} points (limit 3)",
        13: f"B_betti {pc(nb)} against the best reader {pc(bn)}: {(bn - nb) * 100:.1f} points short (limit 5)",
    }[i]


VIEWS = ("majority", "state", "B_betti", "B_betti3", "C_counts", "C_hist", "D_short", "A_window", "A_eng", "A_all", "A_match")


def table_views(res):
    out = []
    for a in VIEWS:
        mode = pc(M(res, a, "mode")) if a != "A_match" else "-"; f1 = f"{M(res, a, 'tour_f1'):.3f}" if a != "A_match" else "-"
        out.append(f"| `{a}` | {mode} | {lohi(res, a, 'mode') if a != 'A_match' else '-'} | {f1} | {pc(M(res, a, 'next'))} | {lohi(res, a, 'next')} |")
    return out


def table_mode(res):
    return [f"| `{a}` | {pc(MM(res, a, 'tour'))} | {pc(MM(res, a, 'drift'))} | {pc(MM(res, a, 'random'))} | {pc(M(res, a, 'mode'))} |" for a in VIEWS if a != "A_match"]


def table_next(res):
    return [f"| `{a}` | {pc(MM(res, a, 'tour', 'next_by_mode'))} | {pc(MM(res, a, 'drift', 'next_by_mode'))} | {pc(MM(res, a, 'random', 'next_by_mode'))} | {pc(M(res, a, 'next'))} |" for a in VIEWS]


def table_beta(res):
    return [f"| {s} | {res['runs'][s]['beta1_mean_by_mode']['tour']:.3f} | {res['runs'][s]['beta1_mean_by_mode']['drift']:.3f} | {res['runs'][s]['beta1_mean_by_mode']['random']:.3f} | {pc(res['runs'][s]['mode_share_test'][0])} / {pc(res['runs'][s]['mode_share_test'][1])} / {pc(res['runs'][s]['mode_share_test'][2])} |" for s in seeds(res)]


def figs(res):
    bw = best(res, "mode", MODEARMS); bn = best(res, "next", MODEARMS + ("A_match",)); bf = best(res, "tour_f1", MODEARMS)
    return dict(
        secs=f"{res['seconds']:.1f}", digest=res["digest"], b_mode=pc(M(res, "B_betti", "mode")), w_mode=pc(M(res, "A_window", "mode")), bw_mode=pc(bw), b_f1=f"{M(res, 'B_betti', 'tour_f1'):.3f}", bw_f1=f"{bf:.3f}",
        b_next=pc(M(res, "B_betti", "next")), bn_next=pc(bn), s_next=pc(M(res, "state", "next")), maj_next=pc(M(res, "majority", "next")), s_mode=pc(M(res, "state", "mode")), maj_mode=pc(M(res, "majority", "mode")),
        b3_mode=pc(M(res, "B_betti3", "mode")), ch_mode=pc(M(res, "C_hist", "mode")), cc_mode=pc(M(res, "C_counts", "mode")), gap3=f"{(bw - M(res, 'B_betti3', 'mode')) * 100:.1f}", gap1=f"{(bw - M(res, 'B_betti', 'mode')) * 100:.1f}",
        b_tour=pc(MM(res, "B_betti", "tour")), b_drift=pc(MM(res, "B_betti", "drift")), b_random=pc(MM(res, "B_betti", "random")), b3_drift=pc(MM(res, "B_betti3", "drift")),
        d_next=pc(M(res, "D_short", "next")), d_share=f"{M(res, 'D_short', 'next') / bn * 100:.0f}", win_next=pc(M(res, "A_window", "next")), win_plus=f"{(M(res, 'A_window', 'next') - M(res, 'D_short', 'next')) * 100:.1f}",
        eng_mode=pc(M(res, "A_eng", "mode")), eng_next=pc(M(res, "A_eng", "next")), all_mode=pc(M(res, "A_all", "mode")), m_next=pc(M(res, "A_match", "next")), m_tour=pc(MM(res, "A_match", "tour", "next_by_mode")),
        m_drift=pc(MM(res, "A_match", "drift", "next_by_mode")), m_random=pc(MM(res, "A_match", "random", "next_by_mode")), w_tour_next=pc(MM(res, "A_window", "tour", "next_by_mode")), b_tour_next=pc(MM(res, "B_betti", "tour", "next_by_mode")),
        r_next=pc(MM(res, "A_window", "random", "next_by_mode")), b1_tour=f"{B1(res, 'tour'):.2f}", b1_drift=f"{B1(res, 'drift'):.2f}", b1_random=f"{B1(res, 'random'):.2f}", b1_gap=f"{B1(res, 'random') - B1(res, 'tour'):.2f}",
        ratio=f"{M(res, 'B_betti', 'tour_f1') / bf:.2f}", state_gap=f"{(M(res, 'state', 'mode') - M(res, 'majority', 'mode')) * 100:.1f}", b_over_state=f"{(M(res, 'B_betti', 'mode') - M(res, 'state', 'mode')) * 100:.1f}",
        b_over_ch=f"{(M(res, 'B_betti', 'mode') - M(res, 'C_hist', 'mode')) * 100:.1f}", b3_over_b=f"{(M(res, 'B_betti3', 'mode') - M(res, 'B_betti', 'mode')) * 100:.1f}", n_gap=f"{(bn - M(res, 'B_betti', 'next')) * 100:.1f}",
        b_over_state_next=f"{(M(res, 'B_betti', 'next') - M(res, 'state', 'next')) * 100:+.1f}", short_state_mode=pc(MM(res, "state", "drift")))


def render(res):
    f = figs(res)
    rows = "\n".join(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 14))
    sha = hashlib.sha256(open(RES, "rb").read()).hexdigest()
    R = r"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_histshape.py` and again by `tests/test_histshape_result.py`). The run was made on 17 Oct 2026 in my workspace, not on the owner's machine: `python -m axi.experiments.histshape --out axi_histshape.json`, __secs__ seconds on two CPUs, Python 3.13.16, numpy 2.5.3, scikit-learn 1.9.1. The result file is `docs/results/axi_histshape.json` (sha256 `__sha__`; digest of the 60 walks `__digest__`). Every figure below is the mean over the three seeds of the test accuracy; the range over the seeds is in the first table. Differences are taken from the unrounded means, so they can differ by 0.1 from the difference of the rounded figures shown.*

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
__ROWS__

**10 of the 13 lines read yes; lines 7, 8 and 13, the three of the headline, read NO.** The headline did not come out: on this world, with these definitions, the pair (beta0, beta1) of the last 50 steps does not match the last 50 commands on either question. It falls short by __gap1__ points on MODE and by __n_gap__ points on NEXT. My probability for the three together was 1 %; for lines 7, 8 and 13 separately it was 30, 45 and 2 %, so the outcome is the one I put at about 99 %. That makes it a confirmation of what I wrote down, not a surprise.

**The four first lines are the instrument, and all four read yes.** Line 2 (beta0 is 1 on every step) is true by the definition of the graph, as the page said. Line 4 says that the loop count has teeth: mean beta1 is __b1_random__ in random mode, __b1_tour__ in tour mode, __b1_drift__ in drift mode, so a gap of __b1_gap__ between random and tour. A NO there would have emptied the test.

## Where my hand numbers were off
* **Drift is not loop-free (line 4's reason).** I wrote that a drift "along a line" closes none; the mean of beta1 in drift mode is __b1_drift__, about one loop in a window of 50 steps, where I expected about zero. Tour is __b1_tour__ against the one loop I wrote down (windows that straddle a switch, and failed moves, change it). The drift is a momentum walker that turns at walls and, 15 % of the time, picks a new direction; in a 7 x 7 grid that closes small loops often. I did not look at beta1 by mode before locking the page, as it said.
* **Line 11 (40 %): the five command counts did worse than the two loop integers.** `B_betti` __b_mode__ against `C_hist` __ch_mode__ on MODE (__b_over_ch__ points). I had written that the counts would say "perhaps as much as beta1". They say clearly less. I have not tested why.
* **Line 5 (80 %): the best reader of the window is better than my bar.** __w_mode__ for `A_window`, against 80 % asked; I had put the cap lower because about a third of the steps have a window that holds a switch (segments of about 150 steps, windows of 50). The readers still name the mode on about nine steps in ten, so they must recognise a new mode after a handful of steps; I did not measure how many.
* **Line 7 (30 %) and its size.** I said B would fall short by more than 5 points. It falls short by __gap1__.

## The tables
**All the views** (mean of the three seeds; the range over the seeds in the next column; the majority rate is that of the training rows; `A_match` is not trained and answers NEXT only):

| view | MODE | MODE, range over seeds | TOUR F1 | NEXT | NEXT, range over seeds |
|---|---|---|---|---|---|
__T1__

**MODE by the true mode of the step** (the share of steps of that mode that were named right):

| view | tour | drift | random | all |
|---|---|---|---|---|
__T2__

**NEXT by the true mode of the step** (random mode picks one of four moves at random, so 25 % is chance there for any reader):

| view | tour | drift | random | all |
|---|---|---|---|---|
__T3__

**beta1 by mode, and the mode shares of the test steps**, per seed:

| seed | beta1 in tour | in drift | in random | test steps in tour / drift / random |
|---|---|---|---|---|
__T4__

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The roadmap's comparison, as I built it, reads NO on both of its questions.** Naming the mode: `B_betti` __b_mode__, the best reader of the 50 commands __bw_mode__. Naming the next command: `B_betti` __b_next__, the best reader of the window __bn_next__. Detecting a tour: F1 __b_f1__ against __bw_f1__.
2. **On NEXT the pair adds nothing.** `B_betti` __b_next__, the cell alone __s_next__, the majority command __maj_next__. Whatever the two integers say about the mode, they do not say which way the walker is going: they are the same number for a loop gone round clockwise and anticlockwise, and no information about the last move.
3. **What the pair holds is one scale of the loops.** It names random mode well (__b_random__ of random steps) and a tour fairly (__b_tour__), and it confuses drift (__b_drift__ of drift steps right; the cell alone gets __short_state_mode__ there). The same quantity at three scales, `B_betti3`, names drift __b3_drift__ of the time and reaches __b3_mode__ on MODE: __b3_over_b__ points above the pair, __gap3__ points below the best reader of the window. So the pair is the weakest form of its own idea, as line 12 asked, and a richer topological summary is not ruled out by this run: only the roadmap's two numbers are.
4. **Where prediction lives is the last few commands.** The last three commands alone (`D_short`) name the next command __d_next__ of the time, __d_share__ % of what the best window reader gets (__bn_next__); the whole window adds __win_plus__ points. In this world "temporal momentum" is the last few moves, which a window holds and no count of loops can.
5. **The loss on NEXT is not special to topology.** `A_eng`, the 26 repeat numbers of the window (how often a command equals the one k steps back), names the mode __eng_mode__ of the time and the next command __eng_next__, which is the cell alone again. Any summary of the window that forgets which command was which loses the direction; the Betti pair and the repeat numbers lose it alike.
6. **`A_match` and the window.** The direct reader of repeats (`A_match`) gets __m_next__ on NEXT (__m_tour__ in tour mode, __m_drift__ in drift mode, __m_random__ in random mode, where nobody can beat a quarter); the tree model reading the 50 raw columns gets __win_next__ (__w_tour_next__ in tour mode). Adding the 26 repeat numbers to the window (`A_all`, __all_mode__ on MODE) did not help the tree model. The worry on the page, that a tree model reads a raw window badly, did not bite here, though a model built for sequences could still do better than `A_window`.

## What this does and does not show about the roadmap's fourth test
* **"Topological cobordism", "persistent memory without a context window".** Nothing here builds a cobordism or tests memory in a model. What was compared is two integers of a graph I defined against 50 commands.
* **"If Arm B matches Arm A's task accuracy, you have proven that chronological intelligence is a geometric property."** Arm B did not match, so the "if" is not met on this world. The reverse does not follow: that two integers of one graph fall short does not show that no geometric summary matches a window.
* **Token cost.** Not measured. The window of 50 bare commands is a few hundred tokens at most; the roadmap's "50,000 tokens" is a different regime.
* **Not done and worth doing:** a summary that is learned and not chosen (a small network that compresses the window to two numbers, for the fair comparison with the pair); other complexes and other scales (`B_betti3` points that way); a world in which loops are what the task needs (returning to a start), where I would now expect the pair to do better; a language model reading the history; a second world; a second learner.

## Reproduce
In `python/`: `python -m axi.experiments.histshape --out axi_histshape.json` (about sixteen minutes on two CPUs; it needs numpy and scikit-learn). The printed digest should be **__digest__** whatever the machine; the accuracies may differ in the last digits between machines (the trees use floating point), the 13 lines should not move. `tests/test_histshape_result.py` recomputes every figure of this Result from the saved file, replays the 60 walks against the digest, and, with `AXI_HEAVY=1`, runs seed 1 again and compares it with the file.
"""
    for k, v in f.items(): R = R.replace(f"__{k}__", v)
    R = R.replace("__sha__", sha).replace("__ROWS__", rows).replace("__T1__", "\n".join(table_views(res))).replace("__T2__", "\n".join(table_mode(res))).replace("__T3__", "\n".join(table_next(res))).replace("__T4__", "\n".join(table_beta(res)))
    assert "__" not in R.replace("__init__", ""), [w for w in R.split() if "__" in w][:5]
    return R


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up():
    res = run(); assert hashlib.sha256(open(RES, "rb").read()).hexdigest() == FILE_SHA256, "the saved result file must not change"
    assert res["seeds"] == [1, 2, 3] and res["train_traj"] == 14 and res["test_traj"] == 6 and res["steps"] == 10000 and res["window"] == 50 and not res["quick"] and res["digest"] == DIGEST and res["hgb"] == H.HGB and res["arms"] == list(H.ARMS)
    assert seeds(res) == ["1", "2", "3"] and set(VIEWS) == {"majority", *H.ARMS, "A_match"}
    for s in seeds(res):
        r = res["runs"][s]; n_tr, n_te = 14 * (10000 - 50), 6 * (10000 - 50); assert r["rows_train"] == n_tr == 139300 and r["rows_test"] == n_te == 59700 and len(r["trajectories"]) == 20 and abs(sum(r["mode_share_test"]) - 1) < 1e-9
        assert all(t["steps"] == 10000 and t["modes_present"] == [0, 1, 2] and t["segments"] >= 20 for t in r["trajectories"]) and r["beta0_min"] == r["beta0_max"] == 1 and set(r["arms"]) == set(VIEWS)
        for m in range(3): assert abs(r["mode_share_test"][m] * n_te - round(r["mode_share_test"][m] * n_te)) < 1e-6                              # a share of whole steps
        for a, t in r["arms"].items():
            for key in ("mode", "next"):
                if t[key] is None: assert a == "A_match" and key == "mode"; continue
                assert abs(t[key] * n_te - round(t[key] * n_te)) < 1e-6, (s, a, key)                                                             # an accuracy is a whole number of test steps
            if t["mode"] is not None:                                                                                                           # the accuracy by mode adds up to the accuracy
                assert abs(sum(r["mode_share_test"][i] * t["mode_by_mode"][name] for i, name in enumerate(H.MODES)) - t["mode"]) < 1e-9, (s, a)
            assert abs(sum(r["mode_share_test"][i] * t["next_by_mode"][name] for i, name in enumerate(H.MODES)) - t["next"]) < 1e-9, (s, a)
    assert abs(sum(res["runs"]["1"]["mode_share_test"]) - 1) < 1e-9


def test_the_sixty_walks_replay_and_give_the_saved_digest():
    res = run(); dig = hashlib.sha256()
    for sd in res["seeds"]:
        for i in range(res["train_traj"] + res["test_traj"]):
            tr = H.make_trajectory(f"hist/{sd}/{i}", res["steps"])
            for k in ("cmds", "states", "modes"): dig.update(tr[k].tobytes())
    assert dig.hexdigest()[:16] == DIGEST


# ---------------------------------------------------------------- 2 the rules
def test_the_thirteen_rules_read_as_the_page_says_and_the_headline_is_three_no():
    res = run(); v = [x for _, x in H.verdict_histshape(res)]; assert v == READ and sum(READ) == 10 and [k for k, _ in H.verdict_histshape(res)] == [k for k, _ in H.HS_LINES]
    assert H.HS_HEADLINE == (7, 8, 13) and [v[i - 1] for i in H.HS_HEADLINE] == [False, False, False] and len(PROBS) == 13
    # the same rules from the raw numbers, by hand
    mw = [M(res, a, "mode") for a in MODEARMS]; fw = [M(res, a, "tour_f1") for a in MODEARMS]; nw = [M(res, a, "next") for a in MODEARMS + ("A_match",)]
    bb, b3, st, ch = (M(res, a, "mode") for a in ("B_betti", "B_betti3", "state", "C_hist")); maj = M(res, "majority", "mode")
    hand = [None, None, st <= maj + 0.12, B1(res, "random") - B1(res, "tour") >= 1.0, max(mw) >= 0.80, bb - st >= 0.10, max(mw) - bb <= 0.05, M(res, "B_betti", "tour_f1") >= 0.9 * max(fw), max(nw) - M(res, "B_betti", "next") >= 0.10,
            M(res, "B_betti", "next") - M(res, "state", "next") <= 0.02, bb > ch, b3 - bb >= 0.03, max(nw) - M(res, "B_betti", "next") <= 0.05]
    for i in range(2, 13): assert hand[i] == v[i], (i + 1, hand[i], v[i])


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    rows = [ln for ln in text[text.index(RESULT_MARK):].split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 13
    for i, ln in enumerate(rows, 1):
        cells = [c.strip() for c in ln.strip("|").split("|")]; assert int(cells[0]) == i and cells[1] == f"{PROBS[i - 1]} %" and cells[2] == measured(res, i) and cells[3] == ("yes" if READ[i - 1] else "NO"), i
    probs_on_page = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())]; assert probs_on_page == PROBS


def test_the_figures_quoted_in_the_result_are_the_ones_in_the_file():
    res = run(); text = _text(); R = text[text.index(RESULT_MARK):]
    # pinned from the printout of the run
    for needle in ("68.6 %", "93.9 %", "0.698", "0.940", "28.6 %", "65.4 %", "28.1 %", "25.2 %", "36.4 %", "30.3 %", "77.7 %", "56.3 %", "57.1 %", "56.1 %", "28.9 %", "82.9 %", "91.0 %", "10 of the 13 lines read yes", "c30005542a6567da", "977.1"):
        assert needle in R, needle
    # recomputed here from the raw file with plain arithmetic
    a = lambda arm, key: sum(res["runs"][s]["arms"][arm][key] for s in "123") / 3
    assert f"{a('B_betti', 'mode') * 100:.1f} %" == "68.6 %" and f"{a('A_window', 'mode') * 100:.1f} %" == "93.9 %" and f"{a('B_betti', 'next') * 100:.1f} %" == "28.6 %" and f"{a('A_window', 'next') * 100:.1f} %" == "65.4 %"
    assert f"{a('D_short', 'next') * 100:.1f} %" == "57.1 %" and f"{a('A_match', 'next') * 100:.1f} %" == "56.1 %" and f"{a('A_eng', 'next') * 100:.1f} %" == "28.9 %" and f"{a('B_betti3', 'mode') * 100:.1f} %" == "77.7 %"
    assert f"{a('B_betti', 'tour_f1'):.3f}" == "0.698" and f"{a('A_window', 'tour_f1'):.3f}" == "0.940" and f"{a('state', 'mode') * 100:.1f} %" == "36.4 %" and f"{a('majority', 'mode') * 100:.1f} %" == "30.3 %" and f"{a('C_hist', 'mode') * 100:.1f} %" == "56.3 %"
    b1 = lambda m: sum(res["runs"][s]["beta1_mean_by_mode"][m] for s in "123") / 3; assert f"{b1('random'):.2f} {b1('tour'):.2f} {b1('drift'):.2f}" == "3.16 1.41 1.00" and "random 3.16, tour 1.41, drift 1.00" in R
    assert "falls short by 25.4 points on MODE and by 36.7 points on NEXT" in R and "87 % of what the best window reader gets" in R


FAST = ["test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up", "test_the_thirteen_rules_read_as_the_page_says_and_the_headline_is_three_no", "test_the_part_above_the_result_is_what_was_written_before_the_run",
        "test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say", "test_the_figures_quoted_in_the_result_are_the_ones_in_the_file"]      # everything but the replay of the walks (the mutation check runs these)


# ---------------------------------------------------------------- 6 heavy
def heavy_seed_1_replays():
    res = run(); new = H.run_histshape(seeds=(1,)); old = res["runs"]["1"]
    assert new["digest"] != "" and new["runs"]["1"]["rows_test"] == old["rows_test"]
    for a, t in old["arms"].items():
        for k, x in t.items():
            if isinstance(x, float) or x is None: assert new["runs"]["1"]["arms"][a][k] == x or abs(new["runs"]["1"]["arms"][a][k] - x) < 1e-3, (a, k, new["runs"]["1"]["arms"][a][k], x)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
    if os.environ.get("AXI_HEAVY"): heavy_seed_1_replays(); print("PASS heavy_seed_1_replays")

"""Implements concept-map item 103 (RESULT): the result of the toy-model test of 17 Oct 2026. Page: docs/TOY_MODEL_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The file: docs/results/axi_toymodel.json (the three single-seed runs of the owner on Colab CPUs, put together with `python -m axi.experiments.toymodel --merge`; five arms, three seeds, 4,000 steps, eight evaluation sets, evaluation digest 7170eee401870825).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the structure of the file (sizes, the tallies of every evaluation adding up: tp + fn is the same for every arm of a set, F1 from the tallies, exact between 1 - fa - miss and 1 - max(fa, miss), the penalty arms' off-block share near zero, the parameter counts, the loss curves), and its sha256;
  2. the fourteen reading rules (9 yes; lines 6, 7, 8, 9 and 14 NO), the headline (line 5 yes, lines 6 and 7 NO), and that the 'read' column of the page says the same;
  3. the 'what was measured' text of each line of the page, the tables and the figures quoted in the prose of the Result (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. that the part of the page above the Result is byte for byte what was written before the run.
What it does NOT show: anything about a model of 10 million parameters, a translation-equivariant Lie-group code, a task other than the matching of coordinates, other training lengths or mixes of sizes, or more than three seeds."""
import hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.toymodel as M

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "TOY_MODEL_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_toymodel.json")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "a5b7cfd346c1f47da8c80718fb9bbe2b5b2e8adbdf6496bd3f55d1863af37165"
FILE_SHA256 = "45114a99cdd5b226173433e41e1fef0fe52ba1f7848c8272859425d3d9f1ce81"
PROBS = [96, 99, 30, 96, 96, 12, 18, 35, 10, 88, 70, 25, 94, 15]
READ = [True, True, True, True, True, False, False, False, False, True, True, True, True, False]
SETS = ["N10", "N20", "N40", "N70", "N100", "range100_n10", "pay_shift_N10", "pay_shift_N100"]
ARMS = ["A_rope1d", "A_rope2d", "B_su2", "B_su2_free", "A_rope2d_orth"]
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[:t.index(RESULT_MARK)]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- the figures, from the saved tallies
def mean(xs): xs = list(xs); return sum(xs) / len(xs)
def seeds(res): return [res["runs"][s] for s in sorted(res["runs"], key=int)]
def raw(res, a, name, field="exact"): return [r[a]["eval"][name][field] for r in seeds(res)]
def ex(res, a, name, field="exact"): return mean(raw(res, a, name, field))
def own(res, a, key): return [r[a][key] for r in seeds(res)]
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def n3(xs, k=1): return " / ".join(f"{x * 100:.{k}f}" for x in xs)
def d3(xs, k=1): return " / ".join(f"{x * 100:+.{k}f}" for x in xs)
def pts(x, k=1, sign=False): return f"{x * 100:+.{k}f}" if sign else f"{abs(x) * 100:.{k}f}"
def share(x): return pc(x, 1) if x >= 0.01 else pc(x, 4)
def diff(res, a, b, name): return [x - y for x, y in zip(raw(res, a, name), raw(res, b, name))]


def measured(res, i):
    N10 = {a: ex(res, a, "N10") for a in ARMS}; lo = min(N10, key=N10.get); F10 = {a: ex(res, a, "N10", "f1") for a in ARMS}; lof = min(F10, key=F10.get)
    allce = [h["ce"] for r in seeds(res) for a in r.values() for h in a["history"][-1:]]; tm = res["tokens_mean"]
    a1, a10, b1, b10, r1, r10 = ex(res, "A_rope1d", "N100"), N10["A_rope1d"], ex(res, "B_su2", "N100"), N10["B_su2"], ex(res, "A_rope2d", "N100"), N10["A_rope2d"]
    off = {a: mean(own(res, a, "offblock_share")) for a in ARMS}; sens = {a: mean(own(res, a, "payload_sensitivity")) for a in ARMS}
    return {
        1: f"15 trainings (3 seeds, 5 arms) of {res['cfg']['steps']:,} steps, the last cross entropy between {min(allce):.1e} and {max(allce):.1e}, all finite; all 8 sets scored on all their graphs",
        2: f"the sets hold {', '.join(str(res['sets'][s]['graphs']) for s in SETS)} graphs; services per graph {', '.join(str(res['sets'][s]['n']) for s in SETS)} (the sixth set on a grid of 100); mean tokens {tm['N10']:.1f} (N10) to {tm['N100']:.1f} (N100)",
        3: f"lowest exact accuracy on N10: `{lo}` {pc(N10[lo])} (limit 80.0 %); the others {', '.join(pc(N10[a]) for a in ARMS if a != lo)}",
        4: f"lowest F1 on N10: `{lof}` {F10[lof]:.2f} (limit: above 0.5)",
        5: f"`A_rope1d` {pc(a1)} on N100 against {pc(a10)} on N10 (it had to be at most {pc(a10 / 2)})",
        6: f"`B_su2` {pc(b1)} on N100 against {pc(b10)} on N10 (it had to be at least {pc(0.8 * b10)})",
        7: f"`B_su2` over `A_rope1d` on N100: {pts(b1 - a1, sign=True)} points (it had to be at least +20.0)",
        8: f"`A_rope2d` {pc(r1)} on N100 against {pc(r10)} on N10 (it had to be at least {pc(0.8 * r10)}; per seed {n3(raw(res, 'A_rope2d', 'N100'))})",
        9: f"`B_su2` over `A_rope2d` on N100: {pts(b1 - r1, sign=True)} points (it had to be at least +5.0)",
        10: f"share of the squared weight in the penalised blocks: with the penalty `B_su2` {share(off['B_su2'])}, `A_rope2d_orth` {share(off['A_rope2d_orth'])}; without it {', '.join(f'`{a}` {share(off[a])}' for a in ('A_rope1d', 'A_rope2d', 'B_su2_free'))}",
        11: f"`B_su2` {sens['B_su2']:.4f} against `B_su2_free` {sens['B_su2_free']:.4f} (per seed {', '.join(f'{x:.4f}' for x in own(res, 'B_su2', 'payload_sensitivity'))} against {', '.join(f'{x:.4f}' for x in own(res, 'B_su2_free', 'payload_sensitivity'))}; smaller in {sum(x < y for x, y in zip(own(res, 'B_su2', 'payload_sensitivity'), own(res, 'B_su2_free', 'payload_sensitivity')))} seeds of three)",
        12: f"on `pay_shift_N10` `B_su2` {pc(ex(res, 'B_su2', 'pay_shift_N10'))} against `B_su2_free` {pc(ex(res, 'B_su2_free', 'pay_shift_N10'))}: {pts(mean(diff(res, 'B_su2', 'B_su2_free', 'pay_shift_N10')), sign=True)} points (limit +5.0); per seed {d3(diff(res, 'B_su2', 'B_su2_free', 'pay_shift_N10'))}",
        13: f"`A_rope1d` {pc(ex(res, 'A_rope1d', 'range100_n10'))} on `range100_n10` against {pc(a10)} on N10 (it had to be at most {pc(a10 / 2)})",
        14: f"on N40 `A_rope2d` {pc(ex(res, 'A_rope2d', 'N40'))} (N10 {pc(r10)}), `B_su2` {pc(ex(res, 'B_su2', 'N40'))} (N10 {pc(b10)}, it had to keep {pc(0.8 * b10)})",
    }[i]


def table_means(res):
    out = ["| all-zero answer | " + " | ".join(pc(res["baseline_exact"][s]) for s in SETS) + " |"]
    for a in ARMS: out.append(f"| `{a}` | " + " | ".join(pc(ex(res, a, s)) for s in SETS) + " |")
    return out


def table_seeds(res):
    return [f"| `{a}` | {n3(raw(res, a, 'N100'))} | {pc(ex(res, a, 'N100'))} | {n3(raw(res, a, 'pay_shift_N10'))} | {pc(ex(res, a, 'pay_shift_N10'))} |" for a in ARMS]


def table_proxy(res):
    out = []
    for a in ARMS:
        fa, mi = ex(res, a, "N100", "graphs_with_false_alarm"), ex(res, a, "N100", "graphs_with_miss")
        out.append(f"| `{a}` | {share(mean(own(res, a, 'offblock_share')))} | {', '.join(f'{x:.4f}' for x in own(res, a, 'payload_sensitivity'))} | {mean(own(res, a, 'payload_sensitivity')):.4f} | {ex(res, a, 'N100', 'f1'):.2f} | {pc(fa)} / {pc(mi)} |")
    return out


def figs(res):
    g = lambda a, s: pc(ex(res, a, s)); raw3 = lambda a, s: n3(raw(res, a, s))
    a1, b1, r1, o1 = (ex(res, a, "N100") for a in ("A_rope1d", "B_su2", "A_rope2d", "A_rope2d_orth")); r10 = ex(res, "A_rope2d", "N10"); drop = [x - y for x, y in zip(raw(res, "A_rope2d", "N100"), raw(res, "A_rope2d_orth", "N100"))]
    sec = [sum(a["seconds"] for a in r.values()) for r in seeds(res)]
    sens = lambda a: mean(own(res, a, "payload_sensitivity")); fa = lambda a, s: pc(ex(res, a, s, "graphs_with_false_alarm")); mi = lambda a, s: pc(ex(res, a, s, "graphs_with_miss"))
    return dict(
        secs=f"{res['seconds']:,.1f}", secper=f"{res['seconds'] / 3 / 60:.0f}", t1=f"{sec[0]:,.0f}", t2=f"{sec[1]:,.0f}", t3=f"{sec[2]:,.0f}", tmin=f"{mean(sec) / 60:.0f}",
        n10_lo=pc(ex(res, "A_rope1d", "N10")), n10_lo_s=raw3("A_rope1d", "N10"),
        a1=pc(a1), b1=pc(b1), r1=pc(r1), o1=pc(o1), bf1=g("B_su2_free", "N100"), r1_s=n3(raw(res, "A_rope2d", "N100")), o1_s=n3(raw(res, "A_rope2d_orth", "N100")), b1_s=n3(raw(res, "B_su2", "N100")), a1_s=n3(raw(res, "A_rope1d", "N100")),
        short8=pts(0.8 * r10 - r1), spread8=f"{(max(raw(res, 'A_rope2d', 'N100')) - min(raw(res, 'A_rope2d', 'N100'))) * 100:.0f}", gap9=pts(b1 - r1), bar6=pc(0.8 * ex(res, "B_su2", "N10")),
        b_n20=g("B_su2", "N20"), r_n20=g("A_rope2d", "N20"), b_n40=g("B_su2", "N40"), r_n40=g("A_rope2d", "N40"), b_rng=g("B_su2", "range100_n10"), r_rng=g("A_rope2d", "range100_n10"), a_rng=g("A_rope1d", "range100_n10"),
        bf_n20=g("B_su2_free", "N20"), bf_n40=g("B_su2_free", "N40"), r_n70=g("A_rope2d", "N70"), b_n70=g("B_su2", "N70"),
        p_b=g("B_su2", "pay_shift_N10"), p_bf=g("B_su2_free", "pay_shift_N10"), p_diff=pts(mean(diff(res, "B_su2", "B_su2_free", "pay_shift_N10")), sign=True), p_diff_s=d3(diff(res, "B_su2", "B_su2_free", "pay_shift_N10")),
        p_bf_s=raw3("B_su2_free", "pay_shift_N10"), p_r=g("A_rope2d", "pay_shift_N10"), p_o=g("A_rope2d_orth", "pay_shift_N10"), p_ro_s=d3(diff(res, "A_rope2d_orth", "A_rope2d", "pay_shift_N10")),
        p100_r=g("A_rope2d", "pay_shift_N100"), p100_o=g("A_rope2d_orth", "pay_shift_N100"),
        drop_s=d3([-x for x in drop]), drop_m=f"{mean(drop) * 100:.1f}",
        s_b=f"{sens('B_su2'):.4f}", s_bf=f"{sens('B_su2_free'):.4f}", s_b_s=", ".join(f"{x:.4f}" for x in own(res, "B_su2", "payload_sensitivity")), s_bf_s=", ".join(f"{x:.4f}" for x in own(res, "B_su2_free", "payload_sensitivity")),
        s_r=f"{sens('A_rope2d'):.4f}", s_o=f"{sens('A_rope2d_orth'):.4f}", s_a=f"{sens('A_rope1d'):.4f}", s_bf_lo=f"{min(own(res, 'B_su2_free', 'payload_sensitivity')):.4f}", s_bf_hi=f"{max(own(res, 'B_su2_free', 'payload_sensitivity')):.4f}",
        off_b=share(mean(own(res, "B_su2", "offblock_share"))), off_bf=share(mean(own(res, "B_su2_free", "offblock_share"))),
        b_fa=fa("B_su2", "N100"), b_mi=mi("B_su2", "N100"), r_fa=fa("A_rope2d", "N100"), r_mi=mi("A_rope2d", "N100"), b_f1=f"{ex(res, 'B_su2', 'N100', 'f1'):.2f}", r_f1=f"{ex(res, 'A_rope2d', 'N100', 'f1'):.2f}",
        params=f"{res['runs']['1']['A_rope2d']['params']:,}", params1=f"{res['runs']['1']['A_rope1d']['params']:,}")


def render(res):
    f = figs(res)
    rows = "\n".join(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 15))
    sha = hashlib.sha256(open(RES, "rb").read()).hexdigest()
    R = r"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_toymodel.py` and again by `tests/test_toymodel_result.py`). The owner ran the three seeds on 17 Oct 2026 on Colab CPUs, one seed per run (notebook `python/colab/axi_toy_model.ipynb`), and the three files were put together with `python -m axi.experiments.toymodel --merge`. The result file is `docs/results/axi_toymodel.json` (sha256 `__sha__`). The evaluation graphs were the same in all three runs (digest 7170eee401870825, which is also what my workspace makes). The training of one seed (five arms) took __t1__, __t2__ and __t3__ seconds, about __tmin__ minutes, and a whole run about __secper__ minutes including the evaluation (__secs__ seconds for the three together); the page had said "about 40 minutes", an estimate. The versions of Python and numpy on the Colab machines were not recorded. Every figure is the mean over the three seeds unless it says otherwise; where it matters the seeds are shown in the order 1 / 2 / 3, because with three seeds a mean can hide a lot. Differences are taken from the unrounded means, so they can differ by 0.1 from the difference of the rounded figures shown.*

*One thing about the run itself, stated so that nobody has to wonder: the first try of Cell 4 in the seed 1 run failed with `ModuleNotFoundError: No module named 'axi.experiments'`, because Cell 3 (which sets the path to the code) had been lost when the tab reconnected; the owner ran Cell 3 again and then Cell 4. Nothing in any figure depends on it; the notebook now sets the path again at the start of Cell 4.*

## The 14 lines

| # | my probability | what was measured | read |
|---|---|---|---|
__ROWS__

**9 of the 14 lines read yes; lines 6, 7, 8, 9 and 14 read NO. Of the three headline lines, 5 reads yes and 6 and 7 read NO, so the roadmap's claim did not come out in this toy.** The page said in advance what the combination "line 5 yes, line 6 NO" means: *"what the roadmap says about "scale invariance" of the Lie-group turn does not hold for this turn: the failure at 100 comes from the construction (a non-abelian turn depends on absolute cells), and the test cannot tell whether a different SU(2) code would do better."* That is what came out, with the same limit: it is the SU(2) turn as I built it that fails, not every spinor code. I put 10 % on all three headline lines reading yes, so the outcome that came out is the one I put at 90 %. The four lines of the instrument read yes. Of the 14 lines, 12 read on the side I favoured; the two exceptions are lines 3 and 12.

## Where my hand numbers were off
* **Line 3 (30 %) read yes.** Every arm reached at least 80 % exact on the graphs it was trained on; the lowest is `A_rope1d` at __n10_lo__ (per seed __n10_lo_s__). I had expected the arm with the coordinate tables to be still short of 80 % after 4,000 steps; it is not. So lines 5, 7 and 13 compare against a model that did learn the task where it was trained, which is what makes them readable.
* **Line 12 (25 %) read yes on the mean, but not in every seed.** On `pay_shift_N10` (ten services, payload ids never seen in training) `B_su2` gets __p_b__ and `B_su2_free` __p_bf__, __p_diff__ points; by seed the difference is __p_diff_s__ points, so one seed of three goes the other way, and the free arm alone scores __p_bf_s__ (percent, seeds 1 / 2 / 3). The mean is real, but the sign is not stable, and the page had said that a yes here "needs a second seed".
* **Line 11 (70 %) read yes on the mean, with a seed that goes the other way.** The sensitivity of the first attention layer to the payload is __s_b__ for `B_su2` and __s_bf__ for `B_su2_free`; per seed __s_b_s__ against __s_bf_s__. It is smaller for the penalised arm in seeds 1 and 2 (in seed 2 by a hair) and larger in seed 3. The free arm's value changes threefold from seed to seed (__s_bf_lo__ to __s_bf_hi__).
* **Line 8 (35 %) read NO, by __short8__ points.** The stronger standard baseline `A_rope2d` keeps __r1__ at N100 of the 100.0 % it has on N10 (per seed __r1_s__); the bar was 80.0 %. The spread between the seeds (__spread8__ points) is bigger than the miss: with other seeds the line could have read yes.
* **Lines 6, 7, 9 and 14 (12, 18, 10 and 15 %) read NO as I thought, and not by a little.** `B_su2` keeps __b1__ at N100 (per seed __b1_s__), where the bar was __bar6__; it is __gap9__ points behind `A_rope2d`.

## The tables
**Exact accuracy** (the share of graphs in which every service and edge token is right; mean over the three seeds; the all-zero answer is the row under the header):

| arm | N10 | N20 | N40 | N70 | N100 | range100_n10 | pay_shift_N10 | pay_shift_N100 |
|---|---|---|---|---|---|---|---|---|
__TMEANS__

**By seed, for the two sets the roadmap's claims turn on** (seeds 1 / 2 / 3, exact accuracy in percent):

| arm | N100 | mean | pay_shift_N10 | mean |
|---|---|---|---|---|
__TSEEDS__

**The penalty and the payload, and how the failure at N100 looks** (off-block share: the share of the squared weight of the query and key matrices that sits in the penalised blocks; payload sensitivity by seed; F1 of the positive tokens at N100; the share of graphs at N100 with a false alarm / with a miss):

| arm | off-block share | payload sensitivity (seeds 1, 2, 3) | mean | F1 at N100 | false alarm / miss at N100 |
|---|---|---|---|---|---|
__TPROXY__

The models have __params__ parameters (__params1__ for `A_rope1d`, which has the two tables).

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The roadmap's comparison, as I built it, has one half that holds and one that does not.** Arm A fails at N100 (__a1__; per seed __a1_s__), as the page said it would, by construction. Arm B fails there too (__b1__; per seed __b1_s__), at about the same level as Arm A and far below the stronger baseline that the roadmap does not mention, `A_rope2d`, which keeps __r1__. In this toy the only arm that carries a good part of what it learned to 100 services is the one whose position code depends on the offset between two cells (__r1__, with a wide spread between seeds); the spinor arm as I built it does not.
2. **Where the SU(2) arm breaks.** It already loses about half at N20 (__b_n20__ against __r_n20__ for `A_rope2d`), is at __b_n40__ at N40 (`A_rope2d` __r_n40__) and at __b_n70__ at N70 (`A_rope2d` __r_n70__). On `range100_n10`, ten services with coordinates up to 99, it gets __b_rng__ while `A_rope2d` gets __r_rng__ and `A_rope1d` __a_rng__. So the range of the coordinates alone breaks it, which is what the page said a non-abelian turn would do (the score of two cells depends on where they are, not only on how far apart), and its errors at N100 are of both kinds: __b_fa__ of the graphs have a false alarm and __b_mi__ a miss, F1 __b_f1__; the errors of `A_rope2d` are almost all false alarms (__r_fa__ of the graphs, against __r_mi__ with a miss; F1 __r_f1__). For `A_rope2d` the range costs nothing (__r_rng__), so what it loses at N100 comes from the length of the sequence or from the offsets between far cells (wrapping of its fast angles); I did not test those two apart.
3. **The penalty does what it is built for, and does not help the size.** It empties the penalised blocks (__off_b__ of the squared weight for `B_su2`, against __off_bf__ without it). At N100 `B_su2` is __b1__ and `B_su2_free` __bf1__; at N20 __b_n20__ against __bf_n20__; at N40 __b_n40__ against __bf_n40__. Put on the axial rotary embedding, the penalty lowers N100 in all three seeds: from __r1__ (__r1_s__) to __o1__ (__o1_s__), __drop_m__ points on average (per seed __drop_s__).
4. **The one place where the penalty looks useful is the unfamiliar payload at the trained size.** On `pay_shift_N10` the penalised axial arm is ahead of the unpenalised one in all three seeds (__p_o__ against __p_r__; per seed __p_ro_s__ points), and the penalised SU(2) arm is ahead of its free twin on the mean (__p_diff__ points) though not in seed 1. On `pay_shift_N100`, where everything is low, the order of the axial pair is on the other side on the mean (__p100_o__ against __p100_r__) and in two seeds of three. This is a small effect on one set, from three seeds, found by looking; it needs a second look with pre-registered lines before anyone calls it a finding.
5. **"Structural heads that fire independently of the payload" is not what the numbers show.** The penalised arms do have a smaller sensitivity of the first layer to the payload on the mean (__s_b__ for `B_su2` and __s_o__ for `A_rope2d_orth`, against __s_bf__ for `B_su2_free` and __s_r__ for `A_rope2d`), but it is not zero, it is not smaller in every seed for SU(2), and the lowest value in the table belongs to the roadmap's Arm A (__s_a__), which has no penalty. The measure is the first layer only; the second layer and principal components were not measured.

## What this does and does not show about the roadmap's second test
* **"Arm A will catastrophically fail."** As built it does (__a1__ at N100), and the page said that this is nearly true by construction. With a relative position code, which the roadmap does not mention, there is no catastrophe (__r1__).
* **"Arm B will mathematically succeed because Lie group geodesics are scale-invariant."** This did not hold for the SU(2) turn I built (__b1__ at N100, __b_n20__ at N20). It does not show that no spinor or Lie-group code can carry the task: the page said the turn is not translation-equivariant and that the test cannot tell; it also said, from memory and unchecked, that Lie-group rotary codes built to be translation-equivariant exist in the literature. This run says nothing about them.
* **"Its structural attention heads fire entirely independently of semantic payloads."** Only two narrow proxies were measured: the weights in the penalised blocks go to nearly zero (line 10), and the first attention layer moves less with the payload on the mean (line 11), not to zero and not in every seed.
* **Not done and worth doing:** a translation-equivariant Lie-group code in place of my SU(2) turn; more seeds (the baseline's N100 score is __r1_s__ percent in the three seeds); a mix of sizes in training; 10 million parameters; a real graph-editing task; the second attention layer and principal components; separating range from length for the axial arms. None of these is implied by this result, either way.

## Reproduce
In `python/`, one seed at a time: `python -m axi.experiments.toymodel --seeds 1 --out axi_toymodel_seed1.json`, the same with seeds 2 and 3, then `python -m axi.experiments.toymodel --merge axi_toymodel_seed1.json axi_toymodel_seed2.json axi_toymodel_seed3.json --out axi_toymodel.json`; about __secper__ minutes per seed on the Colab CPUs. `tests/test_toymodel_result.py` recomputes every figure of this Result from the saved result file.
"""
    for k, v in f.items(): R = R.replace(f"__{k}__", v)
    R = R.replace("__sha__", sha).replace("__ROWS__", rows).replace("__TMEANS__", "\n".join(table_means(res))).replace("__TSEEDS__", "\n".join(table_seeds(res))).replace("__TPROXY__", "\n".join(table_proxy(res)))
    assert "__" not in R.replace("__init__", ""), [w for w in R.split() if "__" in w][:5]
    return R


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up():
    res = run(); assert hashlib.sha256(open(RES, "rb").read()).hexdigest() == FILE_SHA256, "the saved result file must not change"
    assert res["seeds"] == [1, 2, 3] and set(res["runs"]) == {"1", "2", "3"} and res["arms"] == ARMS and list(M.ARMS) == ARMS and res["eval_digest"] == "7170eee401870825" and abs(res["seconds"] - 5686.5) < 1e-9 and res["quick"] is False
    assert {a: (v["code"], v["table"], v["lam"]) for a, v in M.ARMS.items()} == {"A_rope1d": ("rope1d", True, 0.0), "A_rope2d": ("rope2d", False, 0.0), "B_su2": ("su2", False, 0.05), "B_su2_free": ("su2", False, 0.0), "A_rope2d_orth": ("rope2d", False, 0.05)}
    assert res["cfg"] == dict(M.TRAIN) == dict(steps=4000, batch=32, lr=0.002, lr_end=0.1, clip=1.0, pos_weight=3.0, init_seed=11, data_seed=23, eval_graphs=300)
    assert set(res["sets"]) == set(SETS) and {s: res["sets"][s]["graphs"] for s in SETS} == M.EVAL_GRAPHS
    assert {s: (v["n"], v["g"], v["shift"]) for s, v in res["sets"].items()} == {"N10": (10, 10, False), "N20": (20, 20, False), "N40": (40, 40, False), "N70": (70, 70, False), "N100": (100, 100, False), "range100_n10": (10, 100, False), "pay_shift_N10": (10, 10, True), "pay_shift_N100": (100, 100, True)}
    for sd, r in res["runs"].items():
        assert set(r) == set(ARMS), sd
        for a, t in r.items():
            assert [h["step"] for h in t["history"]] == list(range(400, 4001, 400)) and all(math.isfinite(h["ce"]) and math.isfinite(h["loss"]) and h["loss"] >= h["ce"] - 1e-12 for h in t["history"]), (sd, a)
            assert t["history"][-1]["ce"] < t["history"][0]["ce"] and t["params"] == (107329 if a == "A_rope1d" else 100929) and t["payload_sensitivity"] >= 0 and 0 <= t["offblock_share"] <= 1 and t["seconds"] > 0, (sd, a)
            if a in ("B_su2", "A_rope2d_orth"): assert t["offblock_share"] < 1e-4 and all(h["penalty"] < 0.2 for h in t["history"]), (sd, a)          # the penalty empties its blocks
            else: assert t["offblock_share"] > 0.4 and t["history"][-1]["penalty"] > 100, (sd, a)                                                      # without it half of the weight stays in them
            assert set(t["eval"]) == set(SETS), (sd, a)
            for s, e in t["eval"].items():
                n = res["sets"][s]["graphs"]; assert e["n"] == n and abs(e["exact"] * n - round(e["exact"] * n)) < 1e-6 and abs(e["graphs_with_false_alarm"] * n - round(e["graphs_with_false_alarm"] * n)) < 1e-6 and abs(e["graphs_with_miss"] * n - round(e["graphs_with_miss"] * n)) < 1e-6, (sd, a, s)
                tp, fp, fn = e["tokens_tp"], e["tokens_fp"], e["tokens_fn"]
                assert abs(e["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12, (sd, a, s)                                                                    # F1 from its own tallies
                assert 1 - e["graphs_with_false_alarm"] - e["graphs_with_miss"] - 1e-9 <= e["exact"] <= 1 - max(e["graphs_with_false_alarm"], e["graphs_with_miss"]) + 1e-9, (sd, a, s)   # a graph is exact iff it has no false alarm and no miss
                assert tp + fn == res["runs"]["1"]["A_rope2d"]["eval"][s]["tokens_tp"] + res["runs"]["1"]["A_rope2d"]["eval"][s]["tokens_fn"], (sd, a, s)  # the positive tokens of a set are the same for every arm and seed
    assert [res["runs"]["1"]["A_rope2d"]["eval"][s]["tokens_tp"] + res["runs"]["1"]["A_rope2d"]["eval"][s]["tokens_fn"] for s in SETS] == [1131, 1231, 1065, 796, 797, 1080, 1135, 816]
    assert {s: round(res["baseline_exact"][s], 4) for s in SETS} == {"N10": 0.14, "N20": 0.145, "N40": 0.1533, "N70": 0.115, "N100": 0.155, "range100_n10": 0.135, "pay_shift_N10": 0.145, "pay_shift_N100": 0.13}


# ---------------------------------------------------------------- 2 the rules
def test_the_fourteen_rules_read_as_the_page_says_and_the_headline_is_yes_no_no():
    res = run(); chk = M.verdict_toymodel(res); v = [x for _, x in chk]; assert v == READ and sum(READ) == 9 and [k for k, _ in chk] == [k for k, _ in M.TM_LINES] and len(PROBS) == 14
    assert M.TM_HEADLINE == (5, 6, 7) and [v[i - 1] for i in M.TM_HEADLINE] == [True, False, False]
    assert (M.TM_MIN_N10, M.TM_MIN_F1, M.TM_HALF, M.TM_KEEP, M.TM_B_OVER_A, M.TM_B_OVER_ROPE2D, M.TM_SHARE_LO, M.TM_SHARE_HI, M.TM_SHIFT_GAIN) == (0.80, 0.5, 0.5, 0.8, 0.20, 0.05, 0.01, 0.10, 0.05)      # the limits the 'measured' column quotes
    # the same rules from the raw numbers, by hand
    m = lambda a, s, f="exact": mean(r[a]["eval"][s][f] for r in seeds(res)); o = lambda a, k: mean(r[a][k] for r in seeds(res)); steps = res["cfg"]["steps"]
    hand = [all(r[a]["history"][-1]["step"] == steps and math.isfinite(r[a]["history"][-1]["ce"]) for r in seeds(res) for a in ARMS) and all(e["n"] == res["sets"][s]["graphs"] for r in seeds(res) for a in ARMS for s, e in r[a]["eval"].items()),
            all(res["sets"][s]["n"] == n and res["sets"][s]["g"] == g for s, (n, g) in M.TM_SIZES.items()) and len(res["sets"]) == 8,
            all(m(a, "N10") >= 0.80 for a in ARMS), all(m(a, "N10", "f1") > 0.5 for a in ARMS),
            m("A_rope1d", "N100") <= 0.5 * m("A_rope1d", "N10"), m("B_su2", "N100") >= 0.8 * m("B_su2", "N10"), m("B_su2", "N100") - m("A_rope1d", "N100") >= 0.20, m("A_rope2d", "N100") >= 0.8 * m("A_rope2d", "N10"),
            m("B_su2", "N100") - m("A_rope2d", "N100") >= 0.05, all(o(a, "offblock_share") < 0.01 for a in ("B_su2", "A_rope2d_orth")) and all(o(a, "offblock_share") > 0.10 for a in ("A_rope1d", "A_rope2d", "B_su2_free")),
            o("B_su2", "payload_sensitivity") < o("B_su2_free", "payload_sensitivity"), m("B_su2", "pay_shift_N10") - m("B_su2_free", "pay_shift_N10") >= 0.05, m("A_rope1d", "range100_n10") <= 0.5 * m("A_rope1d", "N10"),
            all(m(a, "N40") >= 0.8 * m(a, "N10") for a in ("A_rope2d", "B_su2"))]
    assert hand == READ, hand


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    rows = [ln for ln in text[text.index(RESULT_MARK):].split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 14
    for i, ln in enumerate(rows, 1):
        cells = [c.strip() for c in ln.strip("|").split("|")]; assert int(cells[0]) == i and cells[1] == f"{PROBS[i - 1]} %" and cells[2] == measured(res, i) and cells[3] == ("yes" if READ[i - 1] else "NO"), i
    probs_on_page = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())]; assert probs_on_page == PROBS
    head = _head()
    for i, (tag, line) in enumerate(M.TM_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i                                                                    # the fourteen rules are the fourteen lines of the page
    assert 'what the roadmap says about "scale invariance" of the Lie-group turn does not hold for this turn: the failure at 100 comes from the construction (a non-abelian turn depends on absolute cells), and the test cannot tell whether a different SU(2) code would do better.' in head


def test_the_figures_quoted_in_the_result_are_the_ones_in_the_file():
    res = run(); text = _text(); R = text[text.index(RESULT_MARK):]
    for needle in NEEDLES: assert needle in R, needle
    # recomputed here from the raw file with plain arithmetic
    E = lambda a, s: sum(res["runs"][k][a]["eval"][s]["exact"] for k in "123") / 3; p = lambda x: f"{x * 100:.1f} %"
    assert [p(E(a, "N100")) for a in ARMS] == ["0.8 %", "76.8 %", "2.3 %", "2.8 %", "58.7 %"]
    assert [p(E(a, "N10")) for a in ARMS] == ["96.1 %", "100.0 %", "99.0 %", "96.9 %", "100.0 %"]
    assert [p(E(a, "N20")) for a in ARMS] == ["17.3 %", "99.6 %", "51.5 %", "42.5 %", "99.8 %"] and [p(E(a, "N40")) for a in ARMS] == ["4.3 %", "98.2 %", "14.8 %", "14.9 %", "98.0 %"]
    assert [p(E(a, "range100_n10")) for a in ARMS] == ["11.2 %", "100.0 %", "20.7 %", "23.4 %", "100.0 %"] and [p(E(a, "pay_shift_N10")) for a in ARMS] == ["55.2 %", "90.4 %", "84.3 %", "66.0 %", "98.8 %"]
    S = lambda a, s: [res["runs"][k][a]["eval"][s]["exact"] for k in "123"]
    assert [f"{x * 100:.1f}" for x in S("A_rope2d", "N100")] == ["61.5", "89.5", "79.5"] and [f"{x * 100:.1f}" for x in S("A_rope2d_orth", "N100")] == ["41.5", "67.0", "67.5"] and [f"{x * 100:.1f}" for x in S("B_su2", "N100")] == ["1.5", "4.0", "1.5"]
    assert [round((x - y) * 100, 1) for x, y in zip(S("B_su2", "pay_shift_N10"), S("B_su2_free", "pay_shift_N10"))] == [-5.5, 29.5, 31.0]
    assert [round((x - y) * 100, 1) for x, y in zip(S("A_rope2d_orth", "N100"), S("A_rope2d", "N100"))] == [-20.0, -22.5, -12.0]
    assert [f"{res['runs'][k]['B_su2']['payload_sensitivity']:.4f}" for k in "123"] == ["0.0696", "0.0751", "0.0568"] and [f"{res['runs'][k]['B_su2_free']['payload_sensitivity']:.4f}" for k in "123"] == ["0.1498", "0.0776", "0.0463"]
    fa = lambda a, f: sum(res["runs"][k][a]["eval"]["N100"][f] for k in "123") / 3; sens = lambda a: sum(res["runs"][k][a]["payload_sensitivity"] for k in "123") / 3
    assert (f"{fa('B_su2', 'graphs_with_false_alarm') * 100:.1f}", f"{fa('B_su2', 'graphs_with_miss') * 100:.1f}", f"{fa('A_rope2d', 'graphs_with_false_alarm') * 100:.1f}", f"{fa('A_rope2d', 'graphs_with_miss') * 100:.1f}") == ("96.2", "64.8", "22.7", "0.5")
    assert [f"{sens(a):.4f}" for a in ARMS] == ["0.0237", "0.1404", "0.0672", "0.0912", "0.0680"] and round((max(S("A_rope2d", "N100")) - min(S("A_rope2d", "N100"))) * 100) == 28 and f"{(E('A_rope2d', 'N100') - E('A_rope2d_orth', 'N100')) * 100:.1f}" == "18.2"
    assert f"{(0.8 * E('A_rope2d', 'N10') - E('A_rope2d', 'N100')) * 100:.1f}" == "3.2" and f"{(E('A_rope2d', 'N100') - E('B_su2', 'N100')) * 100:.1f}" == "74.5" and f"{(E('B_su2', 'N100') - E('A_rope1d', 'N100')) * 100:.1f}" == "1.5"
    assert sum(1 for k in "123" if res["runs"][k]["B_su2"]["payload_sensitivity"] < res["runs"][k]["B_su2_free"]["payload_sensitivity"]) == 2 and sum(1 for k in "123" if res["runs"][k]["A_rope2d_orth"]["eval"]["pay_shift_N10"]["exact"] > res["runs"][k]["A_rope2d"]["eval"]["pay_shift_N10"]["exact"]) == 3
    assert sum(1 for k in "123" if res["runs"][k]["A_rope2d_orth"]["eval"]["pay_shift_N100"]["exact"] < res["runs"][k]["A_rope2d"]["eval"]["pay_shift_N100"]["exact"]) == 2 and all(res["runs"][k]["A_rope2d_orth"]["eval"]["N100"]["exact"] < res["runs"][k]["A_rope2d"]["eval"]["N100"]["exact"] for k in "123")
    assert min(res["runs"][k][a]["history"][-1]["ce"] for k in "123" for a in ARMS) < 1e-5 < 1.8e-2 < 2e-2 and f"{sum(sum(x['seconds'] for x in res['runs'][k].values()) for k in '123') / 3 / 60:.0f}" == "30" and f"{res['seconds'] / 3 / 60:.0f}" == "32"
    assert "the roadmap's claim did not come out in this toy" in R and "12 read on the side I favoured" in R and "the one I put at 90 %" in R and "the two exceptions are lines 3 and 12" in R


NEEDLES = ("9 of the 14 lines read yes; lines 6, 7, 8, 9 and 14 read NO", "5,686.5", "1,793, 1,785 and 1,824 seconds, about 30 minutes", "about 32 minutes", "9.3e-06 and 1.8e-02", "(digest 7170eee401870825", "needs a second seed", "100,929 parameters (107,329 for `A_rope1d`",
           "`A_rope1d` 0.8 % on N100 against 96.1 % on N10 (it had to be at most 48.0 %)", "`B_su2` 2.3 % on N100 against 99.0 % on N10 (it had to be at least 79.2 %)", "+1.5 points (it had to be at least +20.0)", "`A_rope2d` 76.8 % on N100 against 100.0 % on N10 (it had to be at least 80.0 %; per seed 61.5 / 89.5 / 79.5)",
           "-74.5 points", "with the penalty `B_su2` 0.0002 %, `A_rope2d_orth` 0.0000 %; without it `A_rope1d` 52.5 %, `A_rope2d` 49.3 %, `B_su2_free` 47.5 %", "`B_su2` 84.3 % against `B_su2_free` 66.0 %: +18.3 points (limit +5.0); per seed -5.5 / +29.5 / +31.0",
           "smaller in 2 seeds of three", "by 3.2 points", "(28 points)", "0.0672", "0.0912", "0.0237", "0.1404", "96.2 % of the graphs have a false alarm and 64.8 % a miss, F1 0.29", "22.7 % of the graphs, against 0.5 % with a miss; F1 0.95", "from 76.8 % (61.5 / 89.5 / 79.5) to 58.7 % (41.5 / 67.0 / 67.5), 18.2 points on average (per seed -20.0 / -22.5 / -12.0)",
           "per seed +0.8 / +9.3 / +15.3 points", "(33.8 % against 39.0 %)", "51.5 % against 99.6 % for `A_rope2d`", "20.7 % while `A_rope2d` gets 100.0 % and `A_rope1d` 11.2 %", "I did not test those two apart", "translation-equivariant", "None of these is implied by this result, either way")


FAST = ["test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up", "test_the_fourteen_rules_read_as_the_page_says_and_the_headline_is_yes_no_no", "test_the_part_above_the_result_is_what_was_written_before_the_run",
        "test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say", "test_the_figures_quoted_in_the_result_are_the_ones_in_the_file"]      # all of them (the mutation check runs these)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

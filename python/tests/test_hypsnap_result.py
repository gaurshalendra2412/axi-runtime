"""Implements concept-map item 104 (RESULT): the result of the hyperbolic-snap test of 17 Oct 2026. Page: docs/HYPERBOLIC_SNAP_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The file: docs/results/axi_hypsnap.json (the run: Qwen2.5-1.5B-Instruct on a free Colab T4 for stage 1, the CPU of the same Colab for stage 2, 2,657.8 s; 400 tasks, two conditions, 9,795 positions each, 32,837 tokens allowed somewhere).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the structure of the file (sizes, every tally adding up: correct = on slips + off slips, admissible = n for every restricted arm, `free` correct on slips = 0, `free` admissible = n - slips, `masked` off slips = `free` correct, the calibration choices are the best of their grids, the heads' loss curves), and its sha256;
  2. the thirteen reading rules (7 yes; lines 6, 7, 8, 9, 11 and 12 NO), the headline (line 5 yes, line 8 NO), and that the 'read' column of the page says the same;
  3. the 'what was measured' text of each line of the page, the tables and the figures quoted in the prose of the Result (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. that the part of the page above the Result is byte for byte what was written before the run.
What it does NOT show: anything about a second model, a second seed of the heads, a wider grid of curvature or scale, free generation, or a model trained for the ball. The hidden vectors themselves (two .npz files of about 150 MB) are not in the repository; the replay of stage 1 needs a GPU."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.hypsnap as Hs

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "HYPERBOLIC_SNAP_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_hypsnap.json")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "5201de5e7abfd889bea7babecd4337b73cc5628d645b77ca607873d140cd2d88"
FILE_SHA256 = "3010c7a13944a3a2c9b6c05f87d9536a850204bf3891f31d354c2a87164329c5"
PROBS = [97, 99, 99, 35, 90, 25, 75, 10, 25, 85, 35, 25, 50]
READ = [True, True, True, True, True, False, False, False, False, True, False, False, True]
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[:t.index(RESULT_MARK)]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- the figures, from the saved tallies
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def arm(res, cond, a): return res["results"][cond]["arms"][a]
def acc(res, cond, a): t = arm(res, cond, a); return t["correct"] / t["n"]
def cnt(res, cond, a): t = arm(res, cond, a); return f"{num(t['correct'])} of {num(t['n'])} ({pc(t['correct'] / t['n'])})"
def onslip(res, cond, a): t = arm(res, cond, a); return t["correct_on_slip"] / t["n_slip"]
def offslip(res, cond, a): t = arm(res, cond, a); return t["correct_off_slip"] / (t["n"] - t["n_slip"])
def pts(res, cond, a, b, k=1, sign=False): d = (arm(res, cond, a)["correct"] - arm(res, cond, b)["correct"]) / arm(res, cond, a)["n"] * 100; return f"{d:+.{k}f}" if sign else f"{abs(d):.{k}f}"
def free(res, cond): return res["results"][cond]["free"]
def nreplies(res, cond): f = free(res, cond); return f["replies_with_slip"], f["replies"]


def measured(res, i):
    f, b = free(res, "full"), free(res, "bare"); R = res["results"]; n = arm(res, "full", "masked")["n"]
    return {
        1: f"both conditions: {num(R['full']['examples'])} and {num(R['bare']['examples'])} replies, {num(R['full']['positions'])} and {num(R['bare']['positions'])} positions, {R['full']['broken']} and {R['bare']['broken']} broken replies; the head rows of {num(R['full']['K'])} tokens stored",
        2: f"the canonical token was allowed at every position of both conditions (full {num(R['full']['positions'])}, bare {num(R['bare']['positions'])})",
        3: f"every arm but `free` chose an allowed token at {num(n)} of {num(n)} test positions in full and {num(arm(res, 'bare', 'masked')['n'])} of {num(arm(res, 'bare', 'masked')['n'])} in bare (eight arms, two conditions)",
        4: f"the model's own pick was refused in {f['replies_with_slip']} of {f['replies']} replies ({pc(f['replies_with_slip'] / f['replies'])}; limit 5 %)",
        5: f"the model's own pick was refused in {b['replies_with_slip']} of {b['replies']} replies ({pc(b['replies_with_slip'] / b['replies'])}; limit 5 %)",
        6: f"refused at {f['slip_positions']} of {num(f['positions'])} positions ({pc(f['slip_share'])}; limit 5 %)",
        7: f"`masked` {cnt(res, 'full', 'masked')} (limit 95 %)",
        8: f"`hyper_nn` {cnt(res, 'full', 'hyper_nn')} against `masked` {pc(acc(res, 'full', 'masked'))}: {pts(res, 'full', 'masked', 'hyper_nn')} points below (it had to be no more than 2 below)",
        9: f"`hyper_nn` {pc(acc(res, 'full', 'hyper_nn'))} against `euclid_tuned` {pc(acc(res, 'full', 'euclid_tuned'))}: {pts(res, 'full', 'hyper_nn', 'euclid_tuned', sign=True)} points (it had to be at least +2)",
        10: f"on the {arm(res, 'full', 'masked')['n_slip']} test positions where the model's own pick is refused: `masked` {arm(res, 'full', 'masked')['correct_on_slip']}, `hyper_nn` {arm(res, 'full', 'hyper_nn')['correct_on_slip']} right (a tie)",
        11: f"`lin_hyp` {pc(acc(res, 'full', 'lin_hyp'))} against `lin_eu` {pc(acc(res, 'full', 'lin_eu'))}: {pts(res, 'full', 'lin_hyp', 'lin_eu', sign=True)} points ({arm(res, 'full', 'lin_hyp')['correct'] - arm(res, 'full', 'lin_eu')['correct']} positions; limit +1)",
        12: f"`lin_hyp` {pc(acc(res, 'full', 'lin_hyp'))} against `lin_dot` {pc(acc(res, 'full', 'lin_dot'))}: {pts(res, 'full', 'lin_hyp', 'lin_dot', sign=True)} points ({arm(res, 'full', 'lin_hyp')['correct'] - arm(res, 'full', 'lin_dot')['correct']} positions; limit +1)",
        13: f"`lin_dot` {pc(acc(res, 'full', 'lin_dot'))} against `masked` {pc(acc(res, 'full', 'masked'))}: {pts(res, 'full', 'lin_dot', 'masked', sign=True)} points (it had to be at least -2)",
    }[i]


def table_arms(res, cond):
    out = []
    for a in Hs.ARMS:
        t = arm(res, cond, a); out.append(f"| `{a}` | {num(t['correct'])} | {pc(t['correct'] / t['n'])} | {pc(t['admissible'] / t['n'])} | {t['correct_on_slip']} of {t['n_slip']} ({pc(onslip(res, cond, a))}) | {pc(offslip(res, cond, a))} |")
    return out


def table_calib(res):
    out = []
    for cond in ("full", "bare"):
        c = res["results"][cond]["calibration"]
        out.append(f"| {cond}, `euclid_tuned` scale | " + " | ".join(pc(c["alpha"]["accuracy"][str(a)]) for a in Hs.GRID_ALPHA) + f" | {c['alpha']['chosen']:g} |")
        out.append(f"| {cond}, `hyper_nn` curvature | " + " | ".join(pc(c["curvature"]["accuracy"][str(a)]) for a in Hs.GRID_C) + f" | {c['curvature']['chosen']:g} |")
    return out


def table_heads(res):
    out = []
    for cond in ("full", "bare"):
        for h in ("lin_dot", "lin_eu", "lin_hyp"):
            t = res["results"][cond]["heads"][h]; out.append(f"| {cond} | `{h}` | {t['loss_curve'][0]:.3f} | {t['loss_curve'][-1]:.3f} | {t['logtau']:.2f} |")
    return out


def figs(res):
    f, b = free(res, "full"), free(res, "bare"); g = lambda c, a: pc(acc(res, c, a))
    return dict(
        secs=f"{res['seconds']:,.1f}", K=num(res["results"]["full"]["K"]), npos=num(res["results"]["full"]["positions"]), ntest=num(arm(res, "full", "masked")["n"]),
        f_rep=pc(f["replies_with_slip"] / f["replies"]), f_repn=str(f["replies_with_slip"]), f_pos=pc(f["slip_share"]), f_posn=str(f["slip_positions"]), b_rep=pc(b["replies_with_slip"] / b["replies"]), b_pos=pc(b["slip_share"]), b_posn=num(b["slip_positions"]),
        f_gold=pc(f["gold_is_free_top"]), b_gold=pc(b["gold_is_free_top"]), per_reply=f"{res['results']['full']['positions'] / res['results']['full']['examples']:.1f}",
        f_free=g("full", "free"), f_mask=g("full", "masked"), f_hyp=g("full", "hyper_nn"), f_tun=g("full", "euclid_tuned"), f_cos=g("full", "cosine_nn"), f_eun=g("full", "euclid_nn"), f_dot=g("full", "lin_dot"), f_leu=g("full", "lin_eu"), f_lhy=g("full", "lin_hyp"),
        b_free=g("bare", "free"), b_mask=g("bare", "masked"), b_hyp=g("bare", "hyper_nn"), b_tun=g("bare", "euclid_tuned"), b_cos=g("bare", "cosine_nn"), b_eun=g("bare", "euclid_nn"), b_dot=g("bare", "lin_dot"), b_leu=g("bare", "lin_eu"), b_lhy=g("bare", "lin_hyp"),
        gap8=pts(res, "full", "masked", "hyper_nn"), gap8b=pts(res, "bare", "masked", "hyper_nn"), gap9=pts(res, "full", "euclid_tuned", "hyper_nn"), gap11=pts(res, "full", "lin_hyp", "lin_eu"), gap12=pts(res, "full", "lin_hyp", "lin_dot"),
        pos11=str(arm(res, "full", "lin_hyp")["correct"] - arm(res, "full", "lin_eu")["correct"]), pos12=str(arm(res, "full", "lin_hyp")["correct"] - arm(res, "full", "lin_dot")["correct"]), gap13=pts(res, "full", "lin_dot", "masked"),
        gap7=f"{(0.95 - acc(res, 'full', 'masked')) * 100:.1f}", mask_gain_b=pts(res, "bare", "masked", "free"), mask_gain_f=pts(res, "full", "masked", "free"), mask_gain_fn=str(arm(res, "full", "masked")["correct"] - arm(res, "full", "free")["correct"]),
        b_gap_dot=pts(res, "bare", "lin_dot", "lin_hyp"), f_slip_test=str(arm(res, "full", "masked")["n_slip"]), b_hyp_slipn=str(arm(res, "bare", "hyper_nn")["correct_on_slip"]), b_slip_n=str(arm(res, "bare", "masked")["n_slip"]), b_mask_slip=pc(onslip(res, "bare", "masked")), b_mask_slipn=str(arm(res, "bare", "masked")["correct_on_slip"]), b_hyp_slip=pc(onslip(res, "bare", "hyper_nn")), b_dot_slip=pc(onslip(res, "bare", "lin_dot")),
        b_leu_slip=pc(onslip(res, "bare", "lin_eu")), b_lhy_slip=pc(onslip(res, "bare", "lin_hyp")), b_free_allowed=pc(arm(res, "bare", "free")["admissible"] / arm(res, "bare", "free")["n"]), f_free_allowed=pc(arm(res, "full", "free")["admissible"] / arm(res, "full", "free")["n"]),
        head_gain_b=f"{(acc(res, 'bare', 'lin_dot') - acc(res, 'bare', 'masked')) * 100:.1f}", head_gain_b2=f"{(acc(res, 'bare', 'lin_hyp') - acc(res, 'bare', 'masked')) * 100:.1f}", head_gain_f=f"{(acc(res, 'full', 'lin_hyp') - acc(res, 'full', 'masked')) * 100:.1f}",
        hyp_free_f=f"{(acc(res, 'full', 'free') - acc(res, 'full', 'hyper_nn')) * 100:.1f}", hyp_free_b=f"{(acc(res, 'bare', 'free') - acc(res, 'bare', 'hyper_nn')) * 100:.1f}",
        c_alpha=pc(res["results"]["full"]["calibration"]["alpha"]["accuracy"]["4.0"]), c_curv=pc(res["results"]["full"]["calibration"]["curvature"]["accuracy"]["100.0"]), c_curv2=pc(res["results"]["full"]["calibration"]["curvature"]["accuracy"]["0.01"]),
        mh=f"{res['results']['full']['calibration']['norms']['hidden']:.1f}", mu=f"{res['results']['full']['calibration']['norms']['head_rows']:.2f}")


def render(res):
    f = figs(res)
    rows = "\n".join(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 14))
    sha = hashlib.sha256(open(RES, "rb").read()).hexdigest()
    R = r"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_hypsnap.py` and again by `tests/test_hypsnap_result.py`). The run was made by the owner on 17 Oct 2026: stage 1 (the model, on a free Colab T4, notebook `python/colab/axi_hyperbolic_snap.ipynb`), then stage 2 (`python -m axi.experiments.hypsnap`, numpy, on the CPU of the same Colab) in __secs__ seconds; the library versions of that machine were not recorded. The result file is `docs/results/axi_hypsnap.json` (sha256 `__sha__`). Every figure below is a count of the __ntest__ test positions of a condition (the odd-numbered replies), or of all __npos__ positions where it says so; there is one run and one seed of the heads, so no range is shown. Differences are taken from the counts, so they can differ by 0.1 from the difference of the rounded figures shown.*

*Two things about the run itself, stated so that nobody has to wonder. (1) The first full run of the collection cell failed after the collection was done and saved: `unembed` in `torch_runner` called `.numpy()` on a tensor that requires grad. The owner ran a cell that redefines `unembed` with `.detach()` and `torch.no_grad()`, and ran the collection cell again, which loaded the saved arrays. The same fix is now in `hypsnap.py` (it touches only the copy of the output-head rows; no figure depends on it; the first hardware run of that function found it, which the stand-in tests could not). (2) Stage 2 took 44 minutes, not the "few minutes" the page and the notebook said: the grammar allows __K__ different tokens somewhere, and every nearest-token rule compares each hidden vector with all of those rows; my timing on made-up data had used 300.*

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
__ROWS__

**7 of the 13 lines read yes; lines 6, 7, 8, 9, 11 and 12 read NO. Of the two headline lines, 5 reads yes and 8 reads NO, so the headline did not come out.** The page said in advance what this combination means: *"Line 5 yes, line 8 NO: the model alone breaks the syntax and the grammar needs a restriction; the restriction does the work and the ball adds nothing over masking, which is the plain reading of the construction."* That is what came out. My probabilities were 90 % for line 5 and 10 % for line 8 (yes), so this outcome was the one I put at about 81 %: a confirmation of what I wrote down. The three first lines are the instrument and all read yes; line 3 is true by construction, as the page said, and checks the code only.

## Where my hand numbers were off
* **Line 4 (35 %) read yes.** In the full prompt the model's own pick is refused at __f_posn__ of __npos__ positions (__f_pos__), which is rare, but a reply has about __per_reply__ positions, so __f_repn__ of 400 replies (__f_rep__) hold at least one. I had put the chance that the reply share stays under 5 % at 65 %. Line 6 (25 %), the share of positions, read NO as I thought.
* **Line 7 (75 %) read NO by __gap7__ points.** `masked` gets __f_mask__ of the canonical tokens in the full prompt, short of the 95 % I asked. I had written "right most of the time"; it is, but not that much of the time. The model's own pick is the canonical token at __f_gold__ of all positions.
* **Lines 8 and 9: the size of the miss.** I had written "clearly below masking" and put 25 % on a 2-point gain over a tuned flat rule. `hyper_nn` is __gap8__ points below `masked` and __gap9__ points *below* `euclid_tuned`.
* **Lines 11 and 12 (35 and 25 %): near misses, in the direction of the roadmap's idea.** `lin_hyp` is ahead of `lin_eu` by __gap11__ points (__pos11__ positions of __ntest__) and of `lin_dot` by __gap12__ points (__pos12__ positions), under my bar of 1 point by 0.3 and 0.1. Whether that is more than noise I cannot say from this file: the per-position picks were not stored, and there is one seed of the heads. In the bare condition (not one of the 13 lines) the order is the other way: `lin_dot` __b_dot__, `lin_hyp` __b_lhy__.
* **Line 10 (85 %) reads yes by a tie.** The test half of the full prompt holds only __f_slip_test__ positions where the model's own pick is refused, and every restricted arm gets all of them right; so many positions cannot separate the arms. The same comparison has power in the bare condition (not one of the 13 lines), with __b_slip_n__ such positions: `masked` gets __b_mask_slipn__ of them right (__b_mask_slip__), `hyper_nn` __b_hyp_slipn__ (__b_hyp_slip__).

## The tables
**Full prompt** (test half: __ntest__ positions; "on slips" are the positions where the model's own pick is refused, "off slips" the others):

| arm | canonical | share | allowed | on slips | off slips |
|---|---|---|---|---|---|
__TFULL__

**Bare** (no description of the syntax in the prompt; test half: __ntest__ positions, __b_slip_n__ of them slips):

| arm | canonical | share | allowed | on slips | off slips |
|---|---|---|---|---|---|
__TBARE__

**The calibration** (accuracy on the calibration half for each value in the grid I locked; the chosen value is the last column):

| condition, rule | 1st value | 2nd | 3rd | 4th | 5th | chosen |
|---|---|---|---|---|---|---|
__TCAL__

(The scale grid is 0.25, 0.5, 1, 2, 4; the curvature grid 0.01, 0.1, 1, 10, 100. Median norm of the hidden vectors __mh__, of the head rows __mu__.)

**The three trained heads** (loss on the calibration half, epoch 1 and epoch 25; the temperature of the two distance heads is exp(logtau); `lin_dot` has none):

| condition | head | first loss | last loss | logtau |
|---|---|---|---|---|
__THEADS__

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The roadmap's comparison, as I built it, has one half that holds and one that does not.** The model's own pick does break the grammar: in every one of the 400 bare replies (__b_pos__ of positions), and in __f_rep__ of the replies of the full prompt. The hyperbolic snap does not match masking: __f_hyp__ against __f_mask__ in the full prompt, __b_hyp__ against __b_mask__ in bare. Against the model's own pick, the roadmap's actual comparison, the snap is __hyp_free_f__ points worse in the full prompt (`free` __f_free__) and __hyp_free_b__ points worse in bare (`free` __b_free__).
2. **The restriction does real work where the syntax is not described.** In bare the model's own pick is refused at __b_posn__ of the __npos__ positions and is an allowed token at only __b_free_allowed__ of the test positions; masking lifts the canonical share from __b_free__ to __b_mask__ (+__mask_gain_b__ points) and recovers __b_mask_slipn__ of the __b_slip_n__ slip positions (__b_mask_slip__). In the full prompt it adds __mask_gain_f__ points (__mask_gain_fn__ positions: all the slips). Plain masking is what the repository already does.
3. **"0.00 %" is 100 % allowed for all eight restricted arms in both conditions**, ball or no ball: the table's "allowed" column. The model alone is allowed at __f_free_allowed__ (full) and __b_free_allowed__ (bare).
4. **The nearest-token rules sit below masking, and the flat one climbs toward it.** Untuned `euclid_nn` gets only __f_eun__ (the rows and the vectors have different norms, so "nearest" is not what the model computes); with a scale it gets __f_tun__, cosine __f_cos__, the ball __f_hyp__; `masked`, which is the model's own dot product among the allowed tokens, gets __f_mask__. The chosen scale is the largest in my grid (accuracy on the calibration half rises with it: __c_alpha__ at 4), and a reason is plain algebra: a nearest-token rule ranks by 2·alpha·(h . u) minus the squared length of u, so as alpha grows the ranking turns into h . u, the model's own ranking, which is `masked`. So the tuned flat rule is approaching masking, not beating it. **The chosen curvature is also the largest in my grid** (__c_curv__ at 100 on the calibration half; the next best value, 0.01, gets __c_curv2__), so `hyper_nn` may be under-tuned: a wider grid could raise it, and I cannot say by how much. The page locked this grid; a wider one is a follow-up and would have to be labelled exploratory.
5. **The three trained heads end within one point of each other in the full prompt, and above masking by 0.8 to 1.7 points** (`lin_dot` __f_dot__, `lin_eu` __f_leu__, `lin_hyp` __f_lhy__; masking __f_mask__). In bare they are 6 to 8 points above masking (`lin_dot` __b_dot__, `lin_eu` __b_leu__, `lin_hyp` __b_lhy__, masking __b_mask__) and on the slip positions they are right 80 to 88 % of the time (`lin_dot` __b_dot_slip__, `lin_eu` __b_leu_slip__, `lin_hyp` __b_lhy_slip__) against __b_mask_slip__ for masking. So a small head trained on about 200 replies recovers the canonical token much more often than the model's own ranking does when the prompt does not describe the syntax, **and the geometry-free dot head is the best of the three there.** What it learned is, most likely, the format of the canonical replies, which are formulaic; it is a supervised classifier on that format, a different kind of tool from a mask, and **AXI never trains**, so it is not something the repository would use. Nothing in these figures points at the ball as the reason for the gain.
6. **What the test does not give the roadmap.** No arm is helped by the ball in this run: untrained, it loses to the flat rule and to cosine; trained, it is ahead of the plane and the dot head by under a point in the full prompt (below the bar I set) and, in bare, level with the flat head and __b_gap_dot__ points behind the dot head.

## What this does and does not show about the roadmap's third test
* **"The Hyperbolic Functor mathematically guarantees a 0.00 % syntax failure rate."** As the page said before the run, this is a property of restricting the choice to the allowed set, which every arm but `free` has; the table shows 100 % allowed for the flat rules, cosine, the ball, masking and the heads alike. It is not a property of the geometry.
* **"Softmax sampling yields a syntax failure rate of at least 5 %."** Met here for replies in both conditions under teacher forcing (__f_rep__ and __b_rep__), not for positions in the full prompt (__f_pos__); one 1.5B model, greedy picks, the canonical reply fed in.
* **Not done and worth doing:** a wider grid for the scale and the curvature; other maps into the ball (a trained curvature, a per-token one); a second seed of the three heads (to say whether lines 11 and 12 are noise); a second and a larger model; free generation, where a slip changes what follows; a head trained for the ball from the start. None of these is implied by this result, either way.

## Reproduce
Stage 1 needs the GPU and the model (the notebook says 10 to 20 minutes on a T4 plus the download of about 3 GB; that is my estimate, the owner's time for it was not recorded); its two arrays (`axi_hs_full.npz`, `axi_hs_bare.npz`, about 150 MB each) are not in the repository. Stage 2 from those arrays: `python -m axi.experiments.hypsnap --full axi_hs_full.npz --bare axi_hs_bare.npz --out axi_hypsnap.json`, about 45 minutes on the Colab CPU. `tests/test_hypsnap_result.py` recomputes every figure of this Result from the saved result file.
"""
    for k, v in f.items(): R = R.replace(f"__{k}__", v)
    R = R.replace("__sha__", sha).replace("__ROWS__", rows).replace("__TFULL__", "\n".join(table_arms(res, "full"))).replace("__TBARE__", "\n".join(table_arms(res, "bare"))).replace("__TCAL__", "\n".join(table_calib(res))).replace("__THEADS__", "\n".join(table_heads(res)))
    assert "__" not in R.replace("__init__", ""), [w for w in R.split() if "__" in w][:5]
    return R


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up():
    res = run(); assert hashlib.sha256(open(RES, "rb").read()).hexdigest() == FILE_SHA256, "the saved result file must not change"
    assert set(res) == {"results", "seconds", "tasks"} and res["tasks"] == 400 and set(res["results"]) == {"full", "bare"} and abs(res["seconds"] - 2657.8) < 1e-9
    for cond, r in res["results"].items():
        assert r["condition"] == cond and r["K"] == 32837 and r["positions"] == 9795 and r["examples"] == 400 and r["broken"] == 0 and r["gold_in_allowed"] is True and set(r["arms"]) == set(Hs.ARMS)
        n = r["arms"]["masked"]["n"]; ns = r["arms"]["masked"]["n_slip"]; assert n == 4270
        for a, t in r["arms"].items():
            assert t["n"] == n and t["n_slip"] == ns and t["correct"] == t["correct_on_slip"] + t["correct_off_slip"], (cond, a)                          # a tally adds up
            assert 0 <= t["correct_on_slip"] <= ns and 0 <= t["correct_off_slip"] <= n - ns, (cond, a)
            if a != "free": assert t["admissible"] == n, (cond, a)                                                                                    # true by construction
        fr = r["free"]; t = r["arms"]["free"]
        assert t["correct_on_slip"] == 0 and t["admissible"] == n - ns and r["arms"]["masked"]["correct_off_slip"] == t["correct_off_slip"] == t["correct"]    # a refused pick is never the canonical token; where it is allowed masking agrees with it
        assert fr["positions"] == r["positions"] and fr["replies"] == 400 and abs(fr["slip_share"] * fr["positions"] - fr["slip_positions"]) < 1e-6 and fr["slip_positions"] >= fr["replies_with_slip"] and 0 <= fr["gold_is_free_top"] <= 1
        for key, grid in (("alpha", Hs.GRID_ALPHA), ("curvature", Hs.GRID_C)):                                                                        # the chosen value is the best of its grid (the first on a tie)
            c = r["calibration"][key]; accs = [c["accuracy"][str(g)] for g in grid]; assert c["chosen"] == grid[accs.index(max(accs))], (cond, key)
        for h, t in r["heads"].items():
            assert len(t["loss_curve"]) == Hs.HEAD["epochs"] == 25 and t["loss_curve"][-1] < t["loss_curve"][0] / 5 and (h != "lin_dot" or t["logtau"] == 0.0), (cond, h)
    assert res["results"]["full"]["arms"]["masked"]["n_slip"] == 23 and res["results"]["bare"]["arms"]["masked"]["n_slip"] == 512


# ---------------------------------------------------------------- 2 the rules
def test_the_thirteen_rules_read_as_the_page_says_and_the_headline_is_yes_no():
    res = run(); chk = Hs.verdict_hypsnap(res["results"], res["tasks"]); v = [x for _, x in chk]; assert v == READ and sum(READ) == 7 and [k for k, _ in chk] == [k for k, _ in Hs.HY_LINES] and len(PROBS) == 13
    assert Hs.HY_HEADLINE == (5, 8) and [v[i - 1] for i in Hs.HY_HEADLINE] == [True, False]
    # the same rules from the raw counts, by hand
    F, B = res["results"]["full"], res["results"]["bare"]; a = lambda r, name: r["arms"][name]["correct"] / r["arms"][name]["n"]
    hand = [all(r["examples"] == 400 and r["broken"] == 0 and r["K"] > 0 for r in (F, B)), F["gold_in_allowed"] and B["gold_in_allowed"], all(r["arms"][x]["admissible"] == r["arms"][x]["n"] for r in (F, B) for x in Hs.ARMS if x != "free"),
            F["free"]["replies_with_slip"] / 400 >= 0.05, B["free"]["replies_with_slip"] / 400 >= 0.05, F["free"]["slip_positions"] / F["positions"] >= 0.05, a(F, "masked") >= 0.95, a(F, "hyper_nn") >= a(F, "masked") - 0.02,
            a(F, "hyper_nn") - a(F, "euclid_tuned") >= 0.02, F["arms"]["masked"]["correct_on_slip"] >= F["arms"]["hyper_nn"]["correct_on_slip"] and F["arms"]["masked"]["n_slip"] > 0, a(F, "lin_hyp") - a(F, "lin_eu") >= 0.01,
            a(F, "lin_hyp") - a(F, "lin_dot") >= 0.01, a(F, "lin_dot") >= a(F, "masked") - 0.02]
    assert hand == READ, hand


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
    for needle in ("93.8 %", "83.3 %", "92.4 %", "95.5 %", "94.8 %", "94.6 %", "80.4 %", "86.7 %", "74.2 %", "94.5 %", "92.8 %", "92.7 %", "52.3 %", "63 of 400 replies (15.8 %)", "9,795", "32,837", "2,657.8", "7 of the 13 lines read yes",
                   "10.5 points below", "9.1 points", "+0.7 points", "+0.9 points", "+0.8 points", "4,005 of 4,270 (93.8 %)", "3,558 of 4,270 (83.3 %)", "400 of 400 replies (100.0 %;", "63 of 9,795 positions (0.6 %)"):
        assert needle in R, needle
    # recomputed here from the raw file with plain arithmetic
    F, B = res["results"]["full"]["arms"], res["results"]["bare"]["arms"]; p = lambda t: f"{t['correct'] / t['n'] * 100:.1f} %"
    assert [p(F[x]) for x in ("masked", "hyper_nn", "euclid_tuned", "lin_hyp", "lin_eu", "lin_dot", "free", "cosine_nn", "euclid_nn")] == ["93.8 %", "83.3 %", "92.4 %", "95.5 %", "94.8 %", "94.6 %", "93.3 %", "92.5 %", "52.3 %"]
    assert [p(B[x]) for x in ("masked", "hyper_nn", "free", "lin_dot", "lin_hyp", "lin_eu")] == ["86.7 %", "74.2 %", "80.4 %", "94.5 %", "92.8 %", "92.7 %"]
    assert (F["masked"]["correct"] - F["hyper_nn"]["correct"], F["euclid_tuned"]["correct"] - F["hyper_nn"]["correct"], F["lin_hyp"]["correct"] - F["lin_eu"]["correct"], F["lin_hyp"]["correct"] - F["lin_dot"]["correct"], F["lin_dot"]["correct"] - F["masked"]["correct"]) == (447, 387, 28, 38, 33)
    assert (F["masked"]["correct"] - F["free"]["correct"], B["masked"]["correct"] - B["free"]["correct"], B["masked"]["correct_on_slip"], B["masked"]["n_slip"]) == (23, 272, 272, 512)
    assert (B["lin_dot"]["correct"] - B["lin_hyp"]["correct"], B["lin_hyp"]["correct"] - B["lin_eu"]["correct"]) == (74, 4) and "1.7 points behind the dot head" in R
    assert "(a tie)" in R and "the largest in my grid" in R and "AXI never trains" in R and "Line 5 yes, line 8 NO" in R and "about 81 %" in R


FAST = ["test_the_file_is_the_one_the_page_is_about_and_its_numbers_add_up", "test_the_thirteen_rules_read_as_the_page_says_and_the_headline_is_yes_no", "test_the_part_above_the_result_is_what_was_written_before_the_run",
        "test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say", "test_the_figures_quoted_in_the_result_are_the_ones_in_the_file"]      # all of them (the mutation check runs these)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

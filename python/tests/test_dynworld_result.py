"""Implements concept-map item 102: RESULT of the dynamic-world test of 17 Oct 2026. Page: docs/DYNAMIC_WORLD_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The file: docs/results/axi_dynworld_seed20261101.json (the run: seed 20261101, Python 3.13, numpy 2.5.3, 329.6 s on two CPUs).
This file recomputes from the saved file, from the raw tallies and not from the figures the code prints:
  1. the structure of the file: seed, sizes, every cell, every tally adds up (each step counted once per arm, per kind and per change), the digest on the page;
  2. the fifteen reading rules (14 yes and line 8 NO), the two headlines, and that the 'read' column of the page says the same;
  3. the 'what was measured' text of each of the fifteen lines of the page, the five tables of the Result, and the figures quoted in the prose of the Result;
  4. that the part of the page above the Result is byte for byte what was written before the run;
  5. (heavy, about two minutes, run by this file as a script and by the full suite, but not by the mutation check) that the world and the exact proposer replay: a second run of `run_dynworld` with the exact proposer gives the saved
     digest and the saved tallies of the exact proposer in every cell.
What it does NOT show: anything about a robot, a language model or a changing world whose changes the host is not told about."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.dynworld as D

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "DYNAMIC_WORLD_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_dynworld_seed20261101.json")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "1a824d8b7d243a2f72a28c63ce4491a96d20c13914834258e4919ceeed3d8619"
DIGEST = "0011b2ff1ab98427"
PROBS = [95, 96, 98, 96, 88, 91, 95, 70, 85, 95, 98, 90, 87, 85, 82]
READ = [True] * 7 + [False] + [True] * 7
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _text(): return open(PAGE).read()


def _result():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[t.index(RESULT_MARK):]


def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[:t.index(RESULT_MARK)]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- the figures, from the raw tallies
def pct(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def T(res, key, p, a): return res["cells"][key]["proposers"][p][a]
def N(res, key): return res["cells"][key]["world"]["steps"]
def slip(res, key, p, a): return T(res, key, p, a)["slip"] / N(res, key)
def dist(res, key, p, a): return T(res, key, p, a)["dist_sum"] / N(res, key)
def calls(res, key, p, a): return T(res, key, p, a)["calls"] / N(res, key)
def unsafe(res, key, p, a): t = T(res, key, p, a); return t["route_unsafe"] / t["route_planned"] if t["route_planned"] else 0.0
def missed(res, key, p, a): t = T(res, key, p, a); return t["route_missed"] / t["route_possible"] if t["route_possible"] else 0.0
def quarters(res, key, p, a): t = T(res, key, p, a); return [d / s for d, s in zip(t["q_dist"], t["q_steps"])]
def pooled(res, p, a, s, field="slip"): return sum(T(res, f"{g}/{s}", p, a)[field] for g in ("uniform", "near")) / sum(N(res, f"{g}/{s}") for g in ("uniform", "near"))
def total_steps(res): return sum(N(res, k) for k in res["cells"])


def measured(res, i):
    """The 'what was measured' text of line i of the table of the Result (i = 1..15)."""
    K = "near/0.4"
    if i == 1: return f"all 10 cells: {num(total_steps(res))} steps in 200 episodes of 1,000, 0 request streams ended early, 0 ill-formed oracle states, every arm counted every step"
    if i == 2:
        n = sum(T(res, k, p, a)["corrupted"] for k in res["cells"] for p in ("exact", "net") for a in ("gate", "retry", "repair")); assert n == 0
        return f"0 corrupted steps for gate, retry and repair in {num(6 * total_steps(res))} arm-steps (both proposers, all cells); blind apply: {num(sum(T(res, k, 'exact', 'blind')['corrupted'] for k in res['cells']))} of {num(total_steps(res))} with the exact reader, {num(sum(T(res, k, 'net', 'blind')['corrupted'] for k in res['cells']))} of {num(total_steps(res))} with the network"
    if i == 3:
        return f"s = 0, both regimes: the four arms of the exact reader slipped 0 times in {num(8 * 20000)} arm-steps and held the oracle's state at every step"
    if i == 4: return f"{K}, exact reader: blind {pct(slip(res, K, 'exact', 'blind'), 2)} ({num(T(res, K, 'exact', 'blind')['slip'])} of {num(N(res, K))}), gate {pct(slip(res, K, 'exact', 'gate'), 2)} ({num(T(res, K, 'exact', 'gate')['slip'])}); the lines were 3 % and 3 %"
    if i == 5: return f"exact reader, repair: {sum(T(res, k, 'exact', 'repair')['slip'] for k in res['cells'])} slips in {num(total_steps(res))} steps (10 cells)"
    if i == 6: return f"exact reader, retry: {sum(T(res, k, 'exact', 'retry')['slip'] for k in res['cells'])} slips in {num(total_steps(res))} steps (10 cells)"
    if i == 7: return f"{K}, exact reader: retry {calls(res, K, 'exact', 'retry'):.3f} calls a step ({num(T(res, K, 'exact', 'retry')['calls'])} calls for {num(N(res, K))} steps; the line was 1.03), repair {calls(res, K, 'exact', 'repair'):.3f}"
    if i == 8:
        q = quarters(res, K, "exact", "gate"); return f"{K}, exact reader, gate: mean distance {q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f} in the four quarters; last over first {q[3] / q[0]:.2f} (the line was 2)"
    if i == 9:
        b, r = T(res, K, "exact", "blind"), T(res, K, "exact", "repair")
        return f"{K}, exact reader: blind apply planned {num(b['route_planned'])} routes, {num(b['route_unsafe'])} unsafe ({pct(unsafe(res, K, 'exact', 'blind'))}); repair planned {num(r['route_planned'])}, {r['route_unsafe']} unsafe"
    if i == 10:
        return f"s = 0.4, exact reader, near against uniform: blind {pct(slip(res, 'near/0.4', 'exact', 'blind'), 2)} against {pct(slip(res, 'uniform/0.4', 'exact', 'blind'), 2)}, gate {pct(slip(res, 'near/0.4', 'exact', 'gate'), 2)} against {pct(slip(res, 'uniform/0.4', 'exact', 'gate'), 2)}"
    if i == 11: return f"network, s = 0, both regimes together: blind apply slipped on {num(sum(T(res, f'{g}/0.0', 'net', 'blind')['slip'] for g in ('uniform', 'near')))} of {num(N(res, 'uniform/0.0') + N(res, 'near/0.0'))} steps ({pct(pooled(res, 'net', 'blind', 0.0))}; the line was 5 %)"
    if i == 12:
        return "network, s = 0, both regimes together: slip share " + ", ".join(f"{a} {pct(pooled(res, 'net', a, 0.0), 2)}" for a in ("repair", "blind", "gate", "retry"))
    K2 = "near/0.2"
    if i == 13: return f"network, {K2}: slip share " + ", ".join(f"{a} {pct(slip(res, K2, 'net', a), 2)}" for a in ("repair", "blind", "gate", "retry"))
    if i == 14: return f"network, {K2}: mean distance " + ", ".join(f"{a} {dist(res, K2, 'net', a):.2f}" for a in ("repair", "blind", "gate", "retry"))
    if i == 15:
        b, r = T(res, K2, "net", "blind"), T(res, K2, "net", "repair")
        return f"network, {K2}: unsafe routes repair {pct(unsafe(res, K2, 'net', 'repair'))} of {num(r['route_planned'])}, blind {pct(unsafe(res, K2, 'net', 'blind'))} of {num(b['route_planned'])} (half of blind's: {pct(0.5 * unsafe(res, K2, 'net', 'blind'))})"
    raise ValueError(i)


def table_exact(res):
    """Table 2: the exact reader at s = 0.4, both regimes."""
    out = []
    for g in ("near", "uniform"):
        K = f"{g}/0.4"
        for a in D.ARMS:
            t = T(res, K, "exact", a); n = N(res, K)
            out.append(f"| {g} | {a} | {pct(t['slip'] / n, 2)} | {pct(t['corrupted'] / n)} | {pct(t['exact'] / n)} | {t['dist_sum'] / n:.2f} | {pct(unsafe(res, K, 'exact', a))} of {num(t['route_planned'])} | {t['calls'] / n:.3f} |")
    return out


def table_net(res):
    """Table 3: the network."""
    out = []
    for K in ("uniform/0.0", "near/0.0", "near/0.2", "near/0.4"):
        for a in D.ARMS:
            t = T(res, K, "net", a); n = N(res, K)
            out.append(f"| {K} | {a} | {pct(t['slip'] / n, 2)} | {pct(t['corrupted'] / n)} | {pct(t['exact'] / n)} | {t['dist_sum'] / n:.2f} (largest {t['dist_max']}) | {pct(unsafe(res, K, 'net', a))} | {pct(missed(res, K, 'net', a))} | {t['calls'] / n:.3f} |")
    return out


def table_s(res):
    """Table 4: how the exact reader's trouble grows with s."""
    out = []
    for g in ("near", "uniform"):
        for s in (0.05, 0.1, 0.2, 0.4):
            K = f"{g}/{s}"
            out.append(f"| {g} | {s} | {num(res['cells'][K]['world']['events'])} | {pct(slip(res, K, 'exact', 'blind'), 2)} | {pct(slip(res, K, 'exact', 'gate'), 2)} | {calls(res, K, 'exact', 'retry'):.3f} | {dist(res, K, 'exact', 'gate'):.2f} |")
    return out


def table_kind(res):
    """Table 5: the network with the repair, by kind of request, near regime, s = 0.4."""
    K = "near/0.4"; bk = T(res, K, "net", "repair")["by_kind"]; out = []
    for kind in ("delete_service", "add_node", "add_edge", "delete_edge", "delete_pair"):
        n, k = bk[kind][0], bk[kind][1]; out.append(f"| {kind} | {num(n)} | {num(k)} | {pct(k / n)} |")
    return out


def table_quarters(res):
    """Table 6: the gate's mean distance by quarter of the episodes, exact reader, s = 0.4."""
    out = []
    for K in ("near/0.4", "uniform/0.4"):
        q = quarters(res, K, "exact", "gate"); out.append(f"| {K} | {q[0]:.2f} | {q[1]:.2f} | {q[2]:.2f} | {q[3]:.2f} | {q[3] / q[0]:.2f} |")
    return out


NET4 = ("uniform/0.0", "near/0.0", "near/0.2", "near/0.4")


def prose_figs(res):
    """The figures quoted in the sentences of the Result: name -> the text as it stands on the page."""
    K = "near/0.4"; ex = lambda a, key=K: T(res, key, "exact", a); lo_hi = lambda f: (min(f), max(f)); f = {}
    f["b_slip"] = pct(slip(res, K, "exact", "blind"), 2); f["b_corr"] = f"{pct(ex('blind')['corrupted'] / N(res, K))} ({num(ex('blind')['corrupted'])} of {num(N(res, K))})"; f["b_exact"] = pct(ex("blind")["exact"] / N(res, K))
    f["g_exact"] = pct(ex("gate")["exact"] / N(res, K)); f["g_dist"] = f"{dist(res, K, 'exact', 'gate'):.2f}"; f["b_dist"] = f"{dist(res, K, 'exact', 'blind'):.2f}"; f["g_unsafe"] = pct(unsafe(res, K, "exact", "gate")); f["b_unsafe"] = pct(unsafe(res, K, "exact", "blind"))
    f["retry_extra"] = pct(calls(res, K, "exact", "retry") - 1); f["calls_02"] = f"{calls(res, 'near/0.2', 'exact', 'retry'):.3f}"; f["calls_01"] = f"{calls(res, 'near/0.1', 'exact', 'retry'):.3f}"; f["calls_005"] = f"{calls(res, 'near/0.05', 'exact', 'retry'):.3f}"
    f["calls_uni"] = f"{calls(res, 'uniform/0.4', 'exact', 'retry'):.3f}"
    f["rep_near0"] = pct(slip(res, "near/0.0", "net", "repair"), 2); f["rep_near4"] = pct(slip(res, K, "net", "repair"), 2); f["bl_near0"] = pct(slip(res, "near/0.0", "net", "blind"), 2); f["bl_near4"] = pct(slip(res, K, "net", "blind"), 2)
    f["rep_uni0"] = pct(slip(res, "uniform/0.0", "net", "repair"), 2); f["rep_uni4"] = pct(slip(res, "uniform/0.4", "net", "repair"), 2)
    f["edges0"] = f"{res['cells']['uniform/0.0']['world']['oracle_edges'] / N(res, 'uniform/0.0'):.1f}"; f["edges4"] = f"{res['cells']['uniform/0.4']['world']['oracle_edges'] / N(res, 'uniform/0.4'):.1f}"
    se = [T(res, k, "net", a)["start_exact"] for k in NET4 for a in ("blind", "gate", "retry")]; f["se_lo"], f["se_hi"] = str(min(se)), str(max(se))
    f["se_ep_lo"], f["se_ep_hi"] = str(round(min(se) / 20)), str(round(max(se) / 20)); fr = [T(res, k, "net", a)["fresh"] for k in NET4 for a in ("blind", "gate", "retry")]; f["fr_lo"], f["fr_hi"] = str(min(fr)), str(max(fr))
    f["gate_s0"] = num(T(res, "near/0.0", "net", "gate")["slip"]); f["retry_s0"] = num(T(res, "near/0.0", "net", "retry")["slip"]); f["calls_net"] = f"{calls(res, 'near/0.4', 'net', 'retry'):.2f}"
    lo, hi = lo_hi([dist(res, k, "net", "repair") for k in NET4]); f["rd_lo"], f["rd_hi"] = f"{lo:.2f}", f"{hi:.2f}"
    lo, hi = lo_hi([dist(res, k, "net", a) for k in NET4 for a in ("blind", "gate", "retry")]); f["od_lo"], f["od_hi"] = f"{lo:.2f}", f"{hi:.2f}"
    lo, hi = lo_hi([unsafe(res, k, "net", "repair") for k in NET4]); f["ru_lo"], f["ru_hi"] = pct(lo), pct(hi)
    lo, hi = lo_hi([unsafe(res, k, "net", a) for k in NET4 for a in ("blind", "gate", "retry")]); f["ou_lo"], f["ou_hi"] = pct(lo), pct(hi)
    lo, hi = lo_hi([slip(res, k, "net", "repair") for k in NET4]); f["rs_lo"], f["rs_hi"] = pct(lo, 1), pct(hi, 1)
    lo, hi = lo_hi([slip(res, k, "net", a) for k in NET4 for a in ("blind", "gate", "retry")]); f["os_lo"], f["os_hi"] = pct(lo, 1), pct(hi, 1)
    bk = T(res, K, "net", "repair")["by_kind"]; f["k_ds"] = f"{bk['delete_service'][1]} of {num(bk['delete_service'][0])}"; f["k_ae"] = pct(bk["add_edge"][1] / bk["add_edge"][0]); f["k_de"] = pct(bk["delete_edge"][1] / bk["delete_edge"][0]); f["k_dp"] = pct(bk["delete_pair"][1] / bk["delete_pair"][0])
    q = quarters(res, K, "exact", "gate"); f["gq1"], f["gq2"], f["gq4"] = f"{q[0]:.2f}", f"{q[1]:.2f}", f"{q[3]:.2f}"; f["gratio"] = f"{q[3] / q[0]:.2f}"
    q = quarters(res, "uniform/0.4", "exact", "gate"); f["gratio_uni"] = f"{q[3] / q[0]:.2f}"
    f["secs"] = "329.6"
    return f


def prose(res): return list(prose_figs(res).values())


RESULT_TEMPLATE = r"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_dynworld.py` and again by `tests/test_dynworld_result.py`). The run was made on 17 Oct 2026 in my workspace, not on the owner's machine: `python -m axi.experiments.dynworld --out axi_dynworld.json`, seed 20261101, __secs__ seconds on two CPUs, Python 3.13, numpy 2.5.3. The result file is `docs/results/axi_dynworld_seed20261101.json` (digest of the requests, the changes and the oracle's states: **__digest__**). I then replayed the world with the exact reader only, in a second process (103 seconds): the same digest, and the tallies of the exact reader identical in all ten cells (the network's cells cannot be replayed bit for bit on another machine, as the page said). Test 114 (`tests/test_dynworld_result.py`) recomputes every figure below from the saved file and, in its heavy part, does that replay.*

**Correction to the page above (the page itself is unchanged).** Its first section says "Concept-map item 102 already had this idea as planned". The concept map did not: the idea was planned in the conversation, not in the file. Item 102 is created by this test.

## The 15 lines

| # | my probability | what was measured | read |
|---|---|---|---|
__ROWS__

**14 of the 15 lines read yes; line 8 reads NO.** The headline A (lines 4, 5, 6, 7: the exact reader, the age of the snapshot only) reads all yes; my probability for the four together was 75 %. The headline B (lines 12, 13, 14, 15: the trained network, near regime, s = 0.2) reads all yes; my probability was 65 %.

**I do not take the fourteen as a sign that the test is strong.** For the exact reader, lines 5 and 6 are close to true by construction: the repair is configured with the same reading of "delete a service" as the definition of a right step, and the retry reads again on the current state. They check the code, and they found no hole in it in 200,000 steps; they are not a discovery. The lines about the network (11 to 15) are where the numbers surprised me, in size, not in direction (see the next section). Line 8 is the one place where I was wrong in direction, and it is a small one: the gate's distance does climb and stays up, but it climbs so fast that the first quarter is already more than half the last.

## Where my hand numbers were off
* **The network's own mistakes (line 11).** I wrote that the network gives a wrong reply 23 % of the time on clean streams "and blind apply carries out every wrong reply", which suggests about 23 % of steps. Blind apply slipped on **49.6 %** of the steps at s = 0. The 23 % was measured with every request read on the oracle's own state; in the run each arm reads its OWN state, and after its first mistake that state is no longer one the network was trained on, nor the one the request was drawn from. Section "What I read from it", item 3, has the evidence.
* **Line 8 (the gate does not come back), 70 %.** The ratio of the last quarter to the first is **__gratio__** (the line asked for 2): the distance is __gq1__ in the first quarter, __gq2__ in the second and __gq4__ in the last. The doubt I wrote down, that it would saturate early, was the right one. In the uniform regime at s = 0.4 the ratio is __gratio_uni__.
* **Line 9 (blind apply's unsafe routes at least 1 %).** I called one in a hundred "low"; it is **__b_unsafe__** of the routes blind apply planned in the near regime at s = 0.4.
* **The cost of the retry.** I put the refusals, and so the extra calls, at 8 to 15 % of the steps at s = 0.4 near; they are **__retry_extra__**. Lower than the middle of my range, inside it.

## The five tables (what the 15 lines are made of)
**The exact reader at s = 0.4.** "Slip" is a mistake of the step; "steps with an edge to a missing service" and "equal to the oracle" are properties of the arm's state after the step; the distance is the mean number of differing services and edges; "unsafe routes" is the share of the routes planned on the held map that the real world cannot walk.

| regime | arm | slip | steps with an edge to a missing service | equal to the oracle | mean distance | unsafe routes | calls a step |
|---|---|---|---|---|---|---|---|
__T2__

**The network.** Same columns, with the largest distance of any step and the routes the held map did not find although the real world has one ("missed").

| cell | arm | slip | steps with an edge to a missing service | equal to the oracle | mean distance | unsafe routes | routes missed | calls a step |
|---|---|---|---|---|---|---|---|---|
__T3__

**How the exact reader's trouble grows with s** (blind apply and the gate, the retry's calls, the gate's distance):

| regime | s | changes of the world | blind apply slips | the gate slips | retry calls a step | gate mean distance |
|---|---|---|---|---|---|---|
__T4__

**The gate's distance by quarter of the episode** (exact reader, s = 0.4; line 8):

| cell | quarter 1 | quarter 2 | quarter 3 | quarter 4 | last over first |
|---|---|---|---|---|---|
__T6__

**The network with the repair, by kind of request** (near regime, s = 0.4; derived afterwards):

| kind of request | steps | slips | share |
|---|---|---|---|
__T5__

## What I read from it (derived afterwards; none of this was pre-registered)
1. **With the exact reader the damage is persistent more than frequent.** In the near regime at s = 0.4 blind apply slips on __b_slip__ of steps, but its state holds an edge to a service that does not exist after __b_corr__ of the steps, and equals the oracle after __b_exact__. The gate is never ill-formed, but it equals the oracle after __g_exact__, its mean distance is __g_dist__ against __b_dist__ for blind apply, and **__g_unsafe__ against __b_unsafe__** of the routes planned on its map are unsafe. A refusal leaves the held map behind the real world (the service the oracle deleted is still there), and by the route measure that is worse than applying the proposal as written. This depends on two choices of mine: the host is told every change at once (so for the gate a refused step is the only way for its map to differ), and a right step is defined as the meaning of the request on the current state.
2. **For the exact reader, the repair's gain over the retry is the second read.** Both are right on every step. The retry asked the proposer __retry_extra__ more often at the worst cell (near, s = 0.4); at s = 0.2, 0.1 and 0.05 the calls a step are __calls_02__, __calls_01__ and __calls_005__ (__calls_uni__ in the uniform regime at s = 0.4), the repair always 1.000. In a real system the second read costs time during which the world can change again; this test does not charge that.
3. **For the network, the changing world is almost invisible; its own mistakes are not.** Between s = 0 and s = 0.4 the repair's slip share goes from __rep_near0__ to __rep_near4__ in the near regime and from __rep_uni0__ to __rep_uni4__ in the uniform one, and blind apply's from __bl_near0__ to __bl_near4__. Every arm except the repair is wrong on about half the steps at every s (__os_lo__ to __os_hi__). The mechanism is the closed loop: each arm's state equals the oracle's at the start of only __se_lo__ to __se_hi__ of the 20,000 steps of a cell (__se_ep_lo__ to __se_ep_hi__ steps an episode, the steps before its first mistake; a mistake made from an exact state happened __fr_lo__ to __fr_hi__ times in 20 episodes); after that, in my reading, the network reads a state it was not trained on, for a request drawn from the oracle's state that its own state does not match. I did not test that explanation separately. The same network that was right on 77 % of clean replies is wrong on half of the steps of its own long episodes.
4. **The repair, with the network, keeps the held map near the real one but not on it.** Mean distance __rd_lo__ to __rd_hi__ items (the oracle holds about 40 items) against __od_lo__ to __od_hi__ for blind apply, the gate and the retry; slip share __rs_lo__ to __rs_hi__ against __os_lo__ to __os_hi__; unsafe routes __ru_lo__ to __ru_hi__ against __ou_lo__ to __ou_hi__. By kind of request (table above): deleting a service with its dependants, which is what the repair is built for, slips __k_ds__ steps; choosing an edge does not: adding an edge __k_ae__, deleting an edge __k_de__, deleting a service and an edge __k_dp__. The repair completes a cascade; it cannot know which edge was meant.
5. **The retry buys nothing with the network.** At s = 0 the retry and the gate do the same thing step for step (the second read of an unchanged state returns the same answer): the gate slipped on __gate_s0__ steps of the near cell and the retry on __retry_s0__. In the near cell at s = 0.4 it costs __calls_net__ calls a step for nothing.
6. **The repair's slip share with the network goes down as s goes up** (item 3). I did not expect it and I have not tested why. One candidate: the oracle's graph gets smaller as the world churns (__edges0__ edges a step at s = 0 and __edges4__ at s = 0.4, uniform), and the network may do better on smaller graphs. That is a guess, not a finding.

## What this does and does not show about the roadmap's first test
* **"Topos Theory and Heyting Logic under sensor noise", "truth gradient", "continuous energy landscape (Attractor Manifolds)".** Nothing in this test touches them (there is no code for them in the package). The page said so before the run, and the result does not change it.
* **"Snap the agent's noisy, out-of-sync proposal into the nearest safe physical action."** What was measured is a discrete procedure, the repair. With a reader that is exact about its snapshot it was right on every step (partly by construction, see above) and cost one call. With a trained network that makes its own mistakes it did not make the network right: it kept the held map within about __rd_lo__ to __rd_hi__ items of the real one instead of __od_lo__ to __od_hi__, and left __ru_lo__ to __ru_hi__ of the planned routes unsafe. If a robot followed a route planned on that map, roughly one plan in ten would use a leg the real world does not have. That is better than the others (__ou_lo__ to __ou_hi__), and it is not "alive".
* **"Even when its internal map diverges from reality by s %."** Here the host's map does not diverge by itself: it is told every change. What diverges is the proposer's snapshot, by one change. The situation of a map that is wrong and that nobody tells is not tested; for it the host would need something that observes the world.
* **"A rigid gate crashes and rejects everything."** The gate refused single steps, and left its map behind (item 1). It did not crash.

**Not done and worth doing:** the version in which the host is not told about the changes (a second reader or a check against a trusted source, and a number of its own); a network trained on its own closed-loop states, or on noise, to see how much of items 3 and 4 is the distribution shift of this one; a language model as the proposer under a changing world; snapshots that are more than one change old; a second, independently written oracle and route check (mine are written by the same hand); the other three tests of the roadmap.

## Reproduce
In `python/`: `python -m axi.experiments.dynworld --out axi_dynworld.json` (about six minutes on two CPUs; the network is read from `docs/results/axi_dynworld_net.npz`). The printed digest should be **__digest__** whatever the machine; the cells of the exact reader should match the saved ones exactly, the cells of the network may differ in a few decisions (the last digit of a matrix product), and the 15 lines should not move. The heavy check of the result test replays the world with the exact reader (about two minutes).
"""


def render(res):
    """The Result of the page, written from the saved file and the template above: the page must be this text, character for character."""
    f = prose_figs(res); t = RESULT_TEMPLATE; checks = D.verdict_dynworld(res); assert len(checks) == 15
    rows = "\n".join(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if checks[i - 1][1] else 'NO'}{' (' + f['gratio'] + ' against 2)' if i == 8 else ''} |" for i in range(1, 16))
    for k, v in f.items(): t = t.replace(f"__{k}__", v)
    for k, v in (("digest", res["digest"]), ("ROWS", rows), ("T2", "\n".join(table_exact(res))), ("T3", "\n".join(table_net(res))), ("T4", "\n".join(table_s(res))), ("T5", "\n".join(table_kind(res))), ("T6", "\n".join(table_quarters(res)))): t = t.replace(f"__{k}__", v)
    assert "__" not in t.replace("__init__", ""), [w for w in t.split() if "__" in w][:5]
    return t.rstrip("\n") + "\n"


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_run_the_page_describes_and_every_tally_adds_up():
    res = run()
    assert set(res) == {"arms", "cells", "digest", "episodes", "noise", "proposers", "quick", "regimes", "seconds", "seed", "steps"}
    assert res["seed"] == 20261101 and res["episodes"] == 20 and res["steps"] == 1000 and res["quick"] is False and res["digest"] == DIGEST and res["noise"] == [0.0, 0.05, 0.1, 0.2, 0.4]
    assert res["regimes"] == ["uniform", "near"] and res["proposers"] == ["exact", "net"] and res["arms"] == list(D.ARMS)
    assert set(res["cells"]) == {f"{g}/{s}" for g in ("uniform", "near") for s in (0.0, 0.05, 0.1, 0.2, 0.4)}
    for key, c in res["cells"].items():
        w = c["world"]; n = w["steps"]
        assert n == 20000 and w["ended"] == 0 and w["oracle_corrupted"] == 0 and c["episodes"] == 20 and sum(w["requests"].values()) == n and sum(w["event_kinds"].values()) == w["events"] and set(w["requests"]) <= set(D.KIND5)
        if c["s"] == 0.0: assert w["events"] == 0
        else: assert 0.4 * c["s"] * n < w["events"] <= c["s"] * n * 1.2, (key, w["events"])
        for p, arms in c["proposers"].items():
            for a, st in arms.items():
                assert sum(v[0] for v in st["by_event"].values()) == n and sum(v[0] for v in st["by_kind"].values()) == n and sum(v[1] for v in st["by_event"].values()) == st["slip"] == sum(v[1] for v in st["by_kind"].values())
                assert sum(st["q_steps"]) == n and sum(st["q_dist"]) == st["dist_sum"] and sum(st["q_exact"]) == st["exact"] and st["fresh"] <= st["start_exact"] <= n and st["exact"] <= n
                assert st["route_unsafe"] <= st["route_planned"] <= n and st["route_missed"] <= st["route_possible"] <= n and st["dist_max"] <= st["dist_sum"]
                assert (st["calls"] == n) if a != "retry" else (n <= st["calls"] <= 2 * n)
                if a != "blind": assert st["corrupted"] == 0


# ---------------------------------------------------------------- 2 the fifteen lines
def test_the_fifteen_reading_rules_read_fourteen_yes_and_line_8_no_and_both_headlines_all_yes():
    res = run(); checks = D.verdict_dynworld(res)
    assert len(checks) == 15 and [v for _, v in checks] == READ and [k for k, _ in checks] == [k for k, _ in D.DW_LINES]
    assert all(checks[i - 1][1] for i in D.DW_HEADLINE) and all(checks[i - 1][1] for i in D.DW_HEADLINE_NET)
    rows = re.findall(r"\n\| (\d+) \| (\d+) % \| (.*?) \| (yes|NO)[^|]*\|(?=\n)", _result())
    assert [int(r[0]) for r in rows] == list(range(1, 16)) and [int(r[1]) for r in rows] == PROBS and [r[3] for r in rows] == ["yes" if v else "NO" for v in READ]
    assert [r[2] for r in rows] == [measured(res, i) for i in range(1, 16)]


# ---------------------------------------------------------------- 3 the tables and the prose
def _rows(table, header_start):
    r = _result(); assert r.count(header_start) == 1, header_start
    block = r[r.index(header_start):].split("\n\n")[0].split("\n")[2:]       # the lines after the header and the divider
    return block


def test_the_five_tables_of_the_result_are_the_figures_of_the_saved_file():
    res = run()
    assert _rows(None, "| regime | arm | slip | steps with an edge to a missing service |") == table_exact(res)
    assert _rows(None, "| cell | arm | slip | steps with an edge to a missing service |") == table_net(res)
    assert _rows(None, "| regime | s | changes of the world | blind apply slips |") == table_s(res)
    assert _rows(None, "| kind of request | steps | slips | share |") == table_kind(res)
    assert _rows(None, "| cell | quarter 1 | quarter 2 | quarter 3 | quarter 4 | last over first |") == table_quarters(res)


RESULT_FILE_SHA256 = "70b268153659155c98dce4c0df9df8b4ed5b0a2caef2c9f07d474ec8760555cd"


def test_the_saved_file_is_the_one_that_was_read_and_the_result_is_the_text_the_saved_file_gives_character_for_character():
    res = run(); body = json.dumps({k: v for k, v in res.items() if k != "seconds"}, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == RESULT_FILE_SHA256, "the saved result has changed"
    assert _result() == render(res), "the Result on the page is not what the saved file and the template give"


def test_the_figures_quoted_in_the_prose_of_the_result_are_the_figures_of_the_saved_file():
    res = run(); r = _result()
    for needle in prose(res): assert needle in r, needle
    assert DIGEST in r and "329.6 seconds" in r and "Python 3.13" in r and "numpy 2.5.3" in r and "103 seconds" in r


# ---------------------------------------------------------------- 4 the locked part
def test_the_part_of_the_page_above_the_result_is_what_was_written_before_the_run():
    assert _h(_head()) == PAGE_SHA256, "nothing above the Result may change after the run"
    r = _result()
    for needle in ("14 of the 15 lines read yes", "line 8 reads NO", "The headline A", "The headline B", "Correction to the page above"):
        assert needle in r, needle


# ---------------------------------------------------------------- 5 the replay (heavy)
def test_the_world_and_the_exact_proposer_replay_to_the_same_digest_and_the_same_tallies():
    res = run(); r = D.run_dynworld(None, seed=res["seed"], proposers=("exact",))
    assert r["digest"] == res["digest"] == DIGEST
    for k in res["cells"]: assert r["cells"][k]["world"] == res["cells"][k]["world"] and r["cells"][k]["proposers"]["exact"] == res["cells"][k]["proposers"]["exact"], k


FAST = ("test_the_file_is_the_run_the_page_describes_and_every_tally_adds_up", "test_the_saved_file_is_the_one_that_was_read_and_the_result_is_the_text_the_saved_file_gives_character_for_character", "test_the_fifteen_reading_rules_read_fourteen_yes_and_line_8_no_and_both_headlines_all_yes",
        "test_the_five_tables_of_the_result_are_the_figures_of_the_saved_file", "test_the_figures_quoted_in_the_prose_of_the_result_are_the_figures_of_the_saved_file", "test_the_part_of_the_page_above_the_result_is_what_was_written_before_the_run")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

"""The cost of being wrong, from the point of view of a company that runs the model (concept-map item 120, docs/COST_OF_BEING_WRONG.md, test 146).

It reads two SAVED files and calls nothing: the long session (test 22) and the fuzzy-state text (test 23), both on Gemini 3.8 Flash. For each arm it counts

  * the turns that ENDED with a held state that is not the true state (the saved `exact` flag; `distance` above 0 cells), the first one, the longest stretch, and the cells off added over the turns;
  * what those turns used: computed tokens (prompt tokens not served from the cache + visible + thinking tokens) and seconds. The API gives no joules, so these two are the stand-in for energy, and the page says so;
  * how a step cost moves through the job (windows of turns), and where the kept chat stops being the faster one;
  * a sensitivity: the same job re-priced at other cache hit rates (assumed) and with other weights for cached and output tokens (assumed, no price list).

`render(d23, d22)` returns the whole page as text; nothing on the page is typed by hand. It edits no framework file and no other test page.
What it does NOT show: the cost of a wrong action in a real business, joules, memory per session or sessions per GPU, the cache rates of a real service, another model, another seed, a real user's words."""
from __future__ import annotations
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

FILE_23 = "axi_gemini_fuzzy_seed20261122_default.json"
FILE_22 = "axi_gemini_long_session_seed20261111_default.json"
HIT_RATES = (0.0, 0.5, 0.75, 0.9, 0.95)
WEIGHTS = ((0.25, 1), (0.25, 4), (0.25, 8), (0.10, 1), (0.10, 4), (0.10, 8))      # (weight of a cached prompt token, weight of an output token); an uncached prompt token weighs 1
WIN_23 = ((1, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300), (301, 350), (351, 400))
WIN_22 = ((1, 50), (51, 150), (151, 250), (251, 350), (351, 457))


# ---------------------------------------------------------------- counting
def computed(r: dict) -> int:
    """Computed tokens of one call: prompt tokens not served from the cache, plus the visible and the thinking tokens."""
    return (r.get("prompt_tokens") or 0) - (r.get("cache_tokens") or 0) + (r.get("tokens") or 0) + (r.get("think") or 0)


def stretches(ts: Sequence[int]) -> List[Tuple[int, int]]:
    """(first turn, length) of every run of consecutive turns in an ascending list."""
    out: List[Tuple[int, int]] = []
    for t in ts:
        if out and t == out[-1][0] + out[-1][1]: out[-1] = (out[-1][0], out[-1][1] + 1)
        else: out.append((t, 1))
    return out


def live(rows: List[dict]) -> List[dict]:
    """The calls made in the run: a step answered from the disk (the saved summary leaves it out of the tokens and the seconds) is not one."""
    return [r for r in rows if not r.get("from_disk")]


def account(rows: List[dict]) -> dict:
    wrong = [r for r in rows if not r["exact"]]; st = stretches([r["t"] for r in wrong]); lv = live(rows); lw = live(wrong)
    return dict(n=len(rows), exact_replies=sum(1 for r in rows if r["reply_exact"]), wrong=len(wrong), exact_replies_wrong=sum(1 for r in wrong if r["reply_exact"]), first=wrong[0]["t"] if wrong else None,
                longest=max((l for _, l in st), default=0), stretches=len(st), last_t=rows[-1]["t"],
                comp=sum(computed(r) for r in lv), comp_wrong=sum(computed(r) for r in lw),
                sec=sum(r["seconds"] for r in lv), sec_wrong=sum(r["seconds"] for r in lw),
                cell_turns=sum(r.get("distance") or 0 for r in rows), max_dist=max((r.get("distance") or 0) for r in rows))


def window(rows: List[dict], lo: int, hi: int) -> dict:
    w = [r for r in live(rows) if lo <= r["t"] <= hi]; ft = [r["first_token"] for r in w if r.get("first_token") is not None]; p = sum(r["prompt_tokens"] for r in w)
    return dict(n=len(w), prompt=p / len(w), comp=sum(computed(r) for r in w) / len(w), first=statistics.median(ft), sec=sum(r["seconds"] for r in w) / len(w),
                dist=sum(r.get("distance") or 0 for r in w) / len(w), held=sum(1 for r in w if r["exact"]), cached=sum(r.get("cache_tokens") or 0 for r in w) / p)


def sweep(kr: List[dict], ar: List[dict]) -> List[dict]:
    """Computed tokens per exact reply of the kept chat at assumed cache hit rates (AXI as measured: its prompt is not cached) and at the observed rate."""
    kr, ar = live(kr), live(ar)
    p = sum(r["prompt_tokens"] for r in kr); out = sum((r.get("tokens") or 0) + (r.get("think") or 0) for r in kr); obs = sum(r.get("cache_tokens") or 0 for r in kr) / p
    ek, ea = sum(1 for r in kr if r["reply_exact"]), sum(1 for r in ar if r["reply_exact"]); a = sum(computed(r) for r in ar) / ea
    rows = [dict(rate=c, k=(p * (1 - c) + out) / ek, a=a, kind="assumed") for c in HIT_RATES]
    return rows + [dict(rate=obs, k=(p * (1 - obs) + out) / ek, a=a, kind="observed")]


def weighted(rows: List[dict], wc: float, wo: float) -> float:
    """Weighted units per exact reply: uncached prompt tokens 1, cached prompt tokens wc, visible + thinking tokens wo."""
    rows = live(rows); u = sum((r["prompt_tokens"] - (r.get("cache_tokens") or 0)) + (r.get("cache_tokens") or 0) * wc + ((r.get("tokens") or 0) + (r.get("think") or 0)) * wo for r in rows)
    return u / sum(1 for r in rows if r["reply_exact"])


def crossover(kw: List[dict], aw: List[dict]) -> Optional[int]:
    """Index of the first window from which the kept chat's median first token is above AXI's in every later window (None if never)."""
    for i in range(len(kw)):
        if all(kw[j]["first"] > aw[j]["first"] for j in range(i, len(kw))): return i
    return None


# ---------------------------------------------------------------- formatting
def num(x): return f"{x:,.0f}" if isinstance(x, float) else f"{x:,}"
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def sec(x, k=2): return f"{x:.{k}f}"
def tm(a, b, k=1): return f"{a / b:.{k}f}"
def tab(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


# ---------------------------------------------------------------- the page
def render(d23: dict, d22: dict) -> str:
    k3, a3, k2, a2 = d23["K"]["rows"], d23["A"]["rows"], d22["K"]["rows"], d22["A"]["rows"]; n2 = len(k2); a2s = a2[:n2]
    arms = [("Test 23, kept chat", k3), ("Test 23, with AXI", a3), ("Test 22, kept chat", k2), (f"Test 22, with AXI, first {n2} steps", a2s), (f"Test 22, with AXI, all {num(len(a2))} steps", a2)]
    ac = [account(r) for _, r in arms]; K3, A3, K2, A2s, A2 = ac
    t1 = tab(["Arm", "Turns", "Replies exact", "Turns ending with a wrong held state", "Share of the turns", "First such turn", "Longest stretch (turns)", "Cells off, added over the turns", "Largest distance (cells)"],
             [[nm, num(a["n"]), f"{num(a['exact_replies'])} of {num(a['n'])}", num(a["wrong"]), pc(a["wrong"] / a["n"]), "-" if a["first"] is None else str(a["first"]), num(a["longest"]), num(a["cell_turns"]), num(a["max_dist"])]
              for (nm, _), a in zip(arms, ac)])
    t2 = tab(["Arm", "Computed tokens, whole arm", "on turns ending wrong", "Share", "Seconds, whole arm", "on turns ending wrong", "Share"],
             [[nm, num(a["comp"]), num(a["comp_wrong"]), pc(a["comp_wrong"] / a["comp"]), num(round(a["sec"])), num(round(a["sec_wrong"])), pc(a["sec_wrong"] / a["sec"])] for (nm, _), a in zip(arms, ac)])
    kw3, aw3 = [window(k3, lo, hi) for lo, hi in WIN_23], [window(a3, lo, hi) for lo, hi in WIN_23]; ci3 = crossover(kw3, aw3)
    t3 = tab(["Turns", "Chat: prompt tokens a step", "Chat: computed a step", "Chat: first token (s, median)", "Chat: share of prompt from the cache", "AXI: prompt tokens a step", "AXI: computed a step", "AXI: first token (s, median)"],
             [[f"{lo} to {hi}", num(k["prompt"]), num(k["comp"]), sec(k["first"]), pc(k["cached"]), num(a["prompt"]), num(a["comp"]), sec(a["first"])] for (lo, hi), k, a in zip(WIN_23, kw3, aw3)])
    kw2, aw2 = [window(k2, lo, hi) for lo, hi in WIN_22], [window(a2, lo, hi) for lo, hi in WIN_22]; ci2 = crossover(kw2, aw2)
    t4 = tab(["Turns", "Chat: prompt tokens a step", "Chat: computed a step", "Chat: first token (s, median)", "Chat: held state exact", "Chat: mean distance (cells)", "AXI: prompt tokens a step", "AXI: computed a step", "AXI: first token (s, median)", "AXI: held state exact"],
             [[f"{lo} to {hi}", num(k["prompt"]), num(k["comp"]), sec(k["first"]), f"{k['held']} of {k['n']}", sec(k["dist"]), num(a["prompt"]), num(a["comp"]), sec(a["first"]), f"{a['held']} of {a['n']}"] for (lo, hi), k, a in zip(WIN_22, kw2, aw2)])
    sw = sweep(k3, a3)
    t5 = tab(["Cache hit rate of the kept chat", "Kept chat: computed tokens per exact reply", "AXI: computed tokens per exact reply", "Kept chat over AXI (times)", "Status"],
             [[pc(s["rate"]), num(s["k"]), num(s["a"]), tm(s["k"], s["a"]), s["kind"]] for s in sw])
    wt = [(wc, wo, weighted(k3, wc, wo), weighted(a3, wc, wo)) for wc, wo in WEIGHTS]
    t6 = tab(["Weight of a cached prompt token", "Weight of an output token", "Kept chat: units per exact reply", "AXI: units per exact reply", "Kept chat over AXI (times)"],
             [[sec(wc), sec(wo, 0), num(k), num(a), tm(k, a)] for wc, wo, k, a in wt])
    obs = sw[-1]; lo_c, hi_c = min(w["comp"] for w in aw3), max(w["comp"] for w in aw3); lo_p, hi_p = min(w["prompt"] for w in aw2), max(w["prompt"] for w in aw2)
    lo_f3, hi_f3 = min(w["first"] for w in aw3), max(w["first"] for w in aw3); lo_f2, hi_f2 = min(w["first"] for w in aw2), max(w["first"] for w in aw2)
    cross3 = (f"The kept chat's median first token was lower than AXI's in the windows up to turn {WIN_23[ci3 - 1][1]} and higher from turns {WIN_23[ci3][0]} to {WIN_23[ci3][1]} on, when its prompt was {num(kw3[ci3]['prompt'])} tokens a step."
              if ci3 else "The kept chat's median first token was above AXI's from the first window." if ci3 == 0 else "The kept chat's median first token was never above AXI's in every later window.")
    cross2 = (f"In test 22 the first window was close ({sec(kw2[0]['first'])} s for the kept chat, {sec(aw2[0]['first'])} s for AXI) and the kept chat's median first token was above AXI's from the window {WIN_22[ci2][0]} to {WIN_22[ci2][1]} on."
              if ci2 else "In test 22 the kept chat's median first token was above AXI's from the first window." if ci2 == 0 else "In test 22 the kept chat's median first token was never above AXI's in every later window.")
    return f"""# The cost of being wrong

*A company's view of two measured runs. Generated from `docs/results/{FILE_22}` (test 22, the long session) and `docs/results/{FILE_23}` (test 23, the fuzzy-state text), both on Gemini 3.8 Flash. Nothing below is typed by hand; test 146 checks that this page is the render of those two files. The architecture is Rajnish Choubey's; the code, the tests and the measurements are the owner's, with an AI assistant as a tool.*

**In one paragraph.** A company that runs a model pays for every token and every second whether the state the model is working on is right or wrong. In test 23 the kept chat ended {num(K3['wrong'])} of {num(K3['n'])} turns with a wrong held list: one wrong change at turn {K3['first']} stayed for {num(K3['longest'])} turns, and those turns used {pc(K3['comp_wrong'] / K3['comp'])} of the chat's computed tokens and {pc(K3['sec_wrong'] / K3['sec'])} of its seconds. In test 22 the kept chat ended {num(K2['wrong'])} of {num(K2['n'])} turns with a wrong held state, from turn {K2['first']} to the end of its run, and the distance from the true state grew to {num(K2['max_dist'])} cells. With AXI no turn of the {num(A3['n'] + A2['n'])} ended wrong, and a step stayed in a band of cost: {num(round(lo_c))} to {num(round(hi_c))} computed tokens a step through test 23, {num(round(lo_p))} to {num(round(hi_p))} prompt tokens a step through test 22. This is the part of the work that a kept chat makes heavier with the length of the job (the prompt it sends, the time to the first token and, once a wrong change is in, the state it works on) and that AXI keeps in a band; it is not the training bill and not the price of a token.

## What is counted

* A turn *ends wrong* when the held state is not the true state after it (the saved `exact` flag is false, the distance is above 0 cells). The kept chat applies each reply as it comes (blind application); AXI holds the state on the host, checks each change and re-asks up to twice.
* Computed tokens = prompt tokens not served from the cache + visible + thinking tokens. Seconds = the wall time of the call. The API gives no joules for the hardware, so the computed tokens and the seconds are the stand-in for energy, not joules.
* Test 22 asked the kept chat for up to 700 turns; it stopped at turn {n2} because its next prompt would have passed 950,000 tokens. The AXI arm is shown for the same {n2} steps and for all {num(len(a2))}. The first turn of the kept chat in test 22 was read from the disk, not called; as in the saved summary it is left out of the tokens and the seconds.

## 1. How often the state was wrong

{t1}

Both arms were exact at the start: the kept chat's first wrong turn is {K3['first']} in test 23 and {K2['first']} in test 22. What the table adds is what happens after a wrong change: in the kept chat it stayed ({num(K3['longest'])} turns in test 23; to the end of the run in test 22, where the distance grew), and with AXI there was no such stretch in {num(A3['n'] + A2['n'])} turns.

## 2. What the turns that ended wrong used

{t2}

This is an accounting of where the compute and the seconds went, not a claim that every one of those tokens was wasted: on the turns that ended wrong, {num(K3['exact_replies_wrong'])} of {num(K3['wrong'])} replies were still exact in test 23 and {num(K2['exact_replies_wrong'])} of {num(K2['wrong'])} in test 22. What was wrong was the state the next turn started from.

## 3. What a step costs through the job

Test 23 (the text is ambiguous; {num(K3['n'])} turns):

{t3}

{cross3} AXI's computed tokens a step moved between {num(round(lo_c))} and {num(round(hi_c))} as its short state grew; its median first token stayed between {sec(lo_f3)} and {sec(hi_f3)} s. At a short prompt the kept chat was the faster one ({sec(kw3[0]['first'])} s against {sec(aw3[0]['first'])} s in the first window); AXI's replies also carried more thinking tokens ({sec(sum(r['think'] for r in a3) / len(a3), 0)} a step against {sec(sum(r['think'] for r in k3) / len(k3), 0)}). The gain is not at the start; it is that AXI stays where it was while the chat does not.

Test 22 (a dependency graph with heavy turns; the first {n2} steps of each arm):

{t4}

{cross2} AXI's median first token stayed between {sec(lo_f2)} and {sec(hi_f2)} s through the {n2} steps; the kept chat's rose with its prompt.

## 4. Sensitivity to the cache (assumed, not measured)

The kept chat's prompt was served from the cache for {pc(obs['rate'])} of its tokens in test 23, and a cache is easiest to use for a chat that only appends. A real service may do worse (a changing head of the prompt, a request routed to a worker without the prefix) or better. The table re-counts the same job at other hit rates; only the last row was observed, the others are assumptions and say nothing about any service. AXI's prompt is not cached in the measured run.

{t5}

At the observed rate the kept chat computed {tm(obs['k'], obs['a'], 2)} times the tokens per exact reply of AXI. In test 23 the computed tokens of the kept chat stayed near {num(round(statistics.median(w['comp'] for w in kw3[1:])))} a step from turn 51 on because the cache carried the history; what grew for the kept chat was the prompt it sent and the time to the first token. In test 22 the cache did not serve every window alike: the computed tokens a step of the kept chat were between {num(round(min(w['comp'] for w in kw2)))} and {num(round(max(w['comp'] for w in kw2)))} by window (table above); the file does not say why.

The same job with weights instead of a price list. The weights are assumptions for a sensitivity: an uncached prompt token weighs 1, a cached prompt token and an output token (visible plus thinking) weigh what the first two columns say.

{t6}

## 5. In company terms

* **A step stays in a band of cost.** With AXI the computed tokens a step stayed between {num(round(lo_c))} and {num(round(hi_c))} (test 23) and the prompt between {num(round(lo_p))} and {num(round(hi_p))} tokens (test 22), over {num(A3['n'])} and {num(A2['n'])} steps. A company can budget a step from the first one.
* **No turn ended on a wrong state with AXI** in {num(A3['n'] + A2['n'])} turns, against {num(K3['wrong'] + K2['wrong'])} of {num(K3['n'] + K2['n'])} for the kept chat; the check and the undo belong to the host.
* **The time to the first token stayed in a band** ({sec(lo_f2)} to {sec(hi_f2)} s in test 22, {sec(lo_f3)} to {sec(hi_f3)} s in test 23), which is what a service level can be written against; the kept chat's rose with its prompt.
* **Where the kept chat is as good or better:** both arms were exact at the start, and at a short prompt the kept chat answered sooner in test 23 ({sec(kw3[0]['first'])} s against {sec(aw3[0]['first'])} s in the first window). The case for AXI is the long job, not the first fifty turns.

## 6. What this page does not show

* The cost of a wrong action in a real business (a wrong record, a failed deployment, a person who has to look). It is likely larger than the tokens, and it is not measured here.
* Joules, or the memory a session takes on a GPU, or how many sessions fit on one: every call went to a hosted model. A shorter prompt means a smaller cache for the same model, but that is arithmetic here, not a measurement.
* The cache hit rates of any real service (section 4 is a sensitivity), or a price list (section 4 uses weights).
* Another model, another seed, another job, or a real user's words: one run of each arm, one seed in each test, synthetic streams.
"""

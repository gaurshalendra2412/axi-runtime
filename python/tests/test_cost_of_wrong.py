"""Implements concept-map item 120 (the cost of being wrong, from the point of view of a company that runs the model; docs/COST_OF_BEING_WRONG.md; python/axi/experiments/cost_of_wrong.py), test 146.
It checks, with no call to any model:
  1. the two saved files (hash) and that the totals the new module counts from the rows are the totals the saved summaries carry (computed tokens, seconds), so the instrument counts what the runs counted;
  2. the figures that carry the page (turns that ended wrong, first turn, longest stretch, cells off, computed tokens and seconds on those turns, the cache sweep, the weighted view), recounted below by hand with plain loops, not with the module;
  3. the small functions of the module on made-up rows (stretches, the window, the cross-over, the sweep at a rate of 0, the weights);
  4. that docs/COST_OF_BEING_WRONG.md is, byte for byte, the render of the two saved files (nothing typed by hand);
  5. the wording rules (no money figure, no statement about any company or provider, no 'proven', energy only as the stand-in, nothing said that the files cannot say) and the front pages (concept map item 120, VERIFIED, README reading list, technical reference) and the counter.
What it does NOT show: anything about another model, another job, another seed, a real business's cost of a wrong action, joules, memory per session or sessions per GPU."""
import hashlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import cost_of_wrong as C

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
RES = os.path.join(DOCS, "results")
PAGE = os.path.join(DOCS, "COST_OF_BEING_WRONG.md")
README = os.path.join(HERE, "..", "..", "README.md")
SHA = {C.FILE_23: "fd104d4be5103d4dbb49a225ce5975194b4551c48f6be83c285b35ce83ec00e8", C.FILE_22: "7ba26d2946ec20ddd8a2085f18c6a2eee9ac3fb02025209158ce67a91faf29c6"}
NEXT = "Next free: row 190, item 143, test 168, question 122"


def load(name): return json.load(open(os.path.join(RES, name)))
def both(): return load(C.FILE_23), load(C.FILE_22)


def test_the_saved_files_and_the_totals_the_module_counts_are_the_totals_of_the_saved_summaries():
    for name, h in SHA.items(): assert hashlib.sha256(open(os.path.join(RES, name), "rb").read()).hexdigest() == h, name
    d23, d22 = both()
    for d in (d23, d22):
        for arm in "KA":
            s = d[arm]["summary"]["all"]; a = C.account(d[arm]["rows"])
            assert a["comp"] == s["computed"], (arm, a["comp"], s["computed"])
            assert abs(a["sec"] - s["seconds"]) < 1e-6, (arm, a["sec"], s["seconds"])
            assert a["n"] == s["steps"] and a["exact_replies"] == round(s["reply_exact"] * s["steps"])
    assert sum(1 for r in d22["K"]["rows"] if r.get("from_disk")) == 1 and d22["K"]["rows"][0]["from_disk"] and d22["K"]["summary"]["all"]["live_steps"] == 456
    assert d22["K"]["stopped"] == "context" and len(d22["K"]["rows"]) == 457 and d22["K"]["meta"]["steps"] == 700 and d22["K"]["meta"]["context_cap"] == 950_000


def hand(rows):
    """Plain loops, written again here on purpose: nothing from the module."""
    wrong = []; comp = comp_w = 0; secs = secs_w = 0.0; cells = 0; mx = 0; best = run = 0; prev = None
    for r in rows:
        live = not r.get("from_disk"); c = r["prompt_tokens"] - r["cache_tokens"] + r["tokens"] + r["think"]
        if live: comp += c; secs += r["seconds"]
        cells += r["distance"]; mx = max(mx, r["distance"])
        if r["exact"]: run = 0
        else:
            wrong.append(r["t"]); run = run + 1 if prev is not None and prev == r["t"] - 1 else 1; best = max(best, run)
            if live: comp_w += c; secs_w += r["seconds"]
        prev = r["t"] if not r["exact"] else None
    return dict(wrong=len(wrong), first=wrong[0] if wrong else None, longest=best, cells=cells, mx=mx, comp=comp, comp_w=comp_w, sec=secs, sec_w=secs_w)


def test_the_figures_on_the_page_are_recounted_by_hand_from_the_rows():
    d23, d22 = both(); k3, a3, k2, a2 = d23["K"]["rows"], d23["A"]["rows"], d22["K"]["rows"], d22["A"]["rows"]
    h = hand(k3); assert (h["wrong"], h["first"], h["longest"], h["cells"], h["mx"]) == (140, 191, 140, 146, 2)
    assert [r["t"] for r in k3 if not r["exact"]] == list(range(191, 331))          # one stretch, turns 191 to 330
    h2 = hand(k2); assert (h2["wrong"], h2["first"], h2["longest"], h2["cells"], h2["mx"]) == (171, 287, 171, 2024, 17)
    assert [r["t"] for r in k2 if not r["exact"]] == list(range(287, 458))          # from turn 287 to the end of the run
    for rows in (a3, a2): ha = hand(rows); assert ha["wrong"] == 0 and ha["cells"] == 0 and ha["mx"] == 0 and ha["comp_w"] == 0
    for rows in (k3, a3, k2, a2, a2[:457]):
        a = C.account(rows); h = hand(rows)
        assert (a["wrong"], a["first"], a["longest"], a["cell_turns"], a["max_dist"], a["comp"], a["comp_wrong"]) == (h["wrong"], h["first"], h["longest"], h["cells"], h["mx"], h["comp"], h["comp_w"])
        assert abs(a["sec"] - h["sec"]) < 1e-6 and abs(a["sec_wrong"] - h["sec_w"]) < 1e-6
    # replies that were still exact on the turns that ended wrong
    assert sum(1 for r in k3 if not r["exact"] and r["reply_exact"]) == 139 and sum(1 for r in k2 if not r["exact"] and r["reply_exact"]) == 157
    # `exact` and `distance` say the same thing on every row of every arm
    for rows in (k3, a3, k2, a2): assert all((r["distance"] == 0) == bool(r["exact"]) for r in rows)
    # the sweep: the chat's cache share, the totals and the ratios, by hand
    p = sum(r["prompt_tokens"] for r in k3); cache = sum(r["cache_tokens"] for r in k3); out = sum(r["tokens"] + r["think"] for r in k3)
    ek = sum(1 for r in k3 if r["reply_exact"]); ea = sum(1 for r in a3 if r["reply_exact"]); assert (ek, ea) == (398, 400)
    a_per = sum(r["prompt_tokens"] - r["cache_tokens"] + r["tokens"] + r["think"] for r in a3) / ea; assert abs(a_per - 1322.09) < 1e-6 and sum(r["cache_tokens"] for r in a3) == 0
    sw = C.sweep(k3, a3); assert [s["kind"] for s in sw] == ["assumed"] * 5 + ["observed"] and abs(sw[-1]["rate"] - cache / p) < 1e-12 and abs(sw[-1]["rate"] - 0.9701) < 5e-5
    for s in sw: assert abs(s["k"] - (p * (1 - s["rate"]) + out) / ek) < 1e-6 and abs(s["a"] - a_per) < 1e-9
    assert [round(s["k"] / s["a"], 1) for s in sw] == [75.9, 38.0, 19.0, 7.6, 3.8, 2.3]
    # the weights, by hand
    for wc, wo in C.WEIGHTS:
        u = lambda rows: sum((r["prompt_tokens"] - r["cache_tokens"]) + r["cache_tokens"] * wc + (r["tokens"] + r["think"]) * wo for r in rows) / sum(1 for r in rows if r["reply_exact"])
        assert abs(C.weighted(k3, wc, wo) - u(k3)) < 1e-6 and abs(C.weighted(a3, wc, wo) - u(a3)) < 1e-6
    assert [round(C.weighted(k3, wc, wo) / C.weighted(a3, wc, wo), 1) for wc, wo in C.WEIGHTS] == [20.7, 15.6, 11.8, 9.7, 7.3, 5.6]
    # the cross-over of the first token, by hand, from the rows
    def med(rows, lo, hi): return statistics.median(r["first_token"] for r in rows if lo <= r["t"] <= hi and not r.get("from_disk") and r["first_token"] is not None)
    cmp3 = [med(k3, lo, hi) > med(a3, lo, hi) for lo, hi in C.WIN_23]; assert cmp3 == [False, False, False, False, True, True, True, True]
    cmp2 = [med(k2, lo, hi) > med(a2, lo, hi) for lo, hi in C.WIN_22]; assert cmp2 == [False, True, True, True, True]
    assert C.crossover([C.window(k3, lo, hi) for lo, hi in C.WIN_23], [C.window(a3, lo, hi) for lo, hi in C.WIN_23]) == 4
    assert C.crossover([C.window(k2, lo, hi) for lo, hi in C.WIN_22], [C.window(a2, lo, hi) for lo, hi in C.WIN_22]) == 1
    # the start: both arms exact in the first fifty turns of test 23, the chat the faster to the first token
    assert all(r["exact"] for r in k3[:50]) and all(r["exact"] for r in a3) and med(k3, 1, 50) < med(a3, 1, 50)


def test_the_small_functions_of_the_module_on_made_up_rows():
    assert C.stretches([]) == [] and C.stretches([3]) == [(3, 1)] and C.stretches([3, 4, 5, 9, 10, 20]) == [(3, 3), (9, 2), (20, 1)]
    row = lambda t, ex, dist=0, p=100, c=0, tok=5, th=2, s=1.0, f=0.5, rx=True, disk=False: dict(t=t, exact=ex, distance=dist, prompt_tokens=p, cache_tokens=c, tokens=tok, think=th, seconds=s, first_token=f, reply_exact=rx, from_disk=disk)
    rows = [row(1, True), row(2, False, 1, rx=False), row(3, False, 2), row(4, True), row(5, False, 1, p=200, c=150, s=3.0)]
    a = C.account(rows); assert (a["n"], a["wrong"], a["first"], a["longest"], a["stretches"], a["cell_turns"], a["max_dist"], a["exact_replies"], a["exact_replies_wrong"]) == (5, 3, 2, 2, 2, 4, 2, 4, 2)
    assert a["comp"] == 107 * 4 + (200 - 150 + 7) and a["comp_wrong"] == 107 * 2 + 57 and abs(a["sec"] - 7.0) < 1e-9 and abs(a["sec_wrong"] - 5.0) < 1e-9
    assert C.computed(row(1, True)) == 107 and C.computed(row(1, True, c=60)) == 47
    d = [row(1, True, disk=True, s=9.0), row(2, True)]; assert C.account(d)["comp"] == 107 and abs(C.account(d)["sec"] - 1.0) < 1e-9 and C.account(d)["n"] == 2
    w = C.window([row(t, True, f=float(t)) for t in range(1, 11)], 3, 7); assert w["n"] == 5 and w["first"] == 5.0 and w["held"] == 5 and abs(w["prompt"] - 100) < 1e-9 and w["cached"] == 0
    wk = [dict(first=2.0), dict(first=3.0), dict(first=4.0)]; wa = [dict(first=2.5), dict(first=2.5), dict(first=2.5)]
    assert C.crossover(wk, wa) == 1 and C.crossover(wa, wk) is None and C.crossover([dict(first=1.0)] * 3, wa) is None and C.crossover([dict(first=9.0)] * 3, wa) == 0
    kr = [row(t, True, p=1000, c=0, tok=0, th=0) for t in range(1, 5)]; ar = [row(t, True, p=100, c=0, tok=0, th=0) for t in range(1, 5)]
    s0 = C.sweep(kr, ar); assert s0[0]["rate"] == 0.0 and s0[0]["k"] == 1000 and s0[0]["a"] == 100 and s0[-1]["kind"] == "observed" and s0[-1]["rate"] == 0.0
    assert C.weighted(kr, 0.25, 4) == 1000 and C.weighted([row(1, True, p=1000, c=800, tok=10, th=10)], 0.25, 4) == 200 + 800 * 0.25 + 20 * 4


def test_the_page_is_the_render_of_the_two_saved_files_and_nothing_is_typed_by_hand():
    d23, d22 = both(); page = C.render(d23, d22)
    assert open(PAGE).read() == page, "docs/COST_OF_BEING_WRONG.md is not the render of the saved files; regenerate it"
    assert page.startswith("# The cost of being wrong\n") and page.endswith("\n") and "\n\n\n" not in page
    for must in ("test 146", C.FILE_22, C.FILE_23, "Rajnish Choubey", "not joules", "stand-in", "assumed", "not measured", "one run of each arm"):
        assert must.lower() in page.lower(), must
    for sec_ in ("## What is counted", "## 1. How often the state was wrong", "## 2. What the turns that ended wrong used", "## 3. What a step costs through the job", "## 4. Sensitivity to the cache (assumed, not measured)", "## 5. In company terms", "## 6. What this page does not show"):
        assert sec_ in page, sec_
    # the sentences that matter, with the figures the hand count gave
    for s in ("ended 140 of 400 turns", "turn 191 stayed for 140 turns", "29.5 % of the chat's computed tokens", "36.6 % of its seconds", "ended 171 of 457 turns", "from turn 287 to the end of its run", "grew to 17 cells", "no turn of the 1,900 ended wrong",
              "75.9", "38.0", "19.0", "7.6", "3.8", "2.30 times", "higher from turns 201 to 250 on", "against 311 of 857"):
        assert s in page, s


def test_the_wording_rules():
    page = C.render(*both()); low = page.lower()
    for bad in ("$", "proven", "100 percent", "gpus saved", "dollar", "google", "openai", "anthropic", "ultimate", "industry", "market", "profit", "carbon", "kwh", "watt ", "watts", "guarantee", "solves", "breakthrough", "saves money", "millions"):
        assert bad not in low, bad
    assert "100 %" not in page and "100.0 %" not in page                       # a share of 100 is written as 'n of n'
    for must in ("not the training bill", "likely larger than the tokens, and it is not measured here", "a sensitivity", "say nothing about any service", "the file does not say why", "the case for axi is the long job, not the first fifty turns",
                 "not a claim that every one of those tokens was wasted"):
        assert must in low, must
    # no multiple of capacity, no count of machines
    for bad in ("more users", "more sessions", "more agents", "fewer gpus", "fewer machines", "concurrent"): assert bad not in low, bad


def test_the_front_pages_say_what_the_page_is_and_the_counter_moved():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read(); rd = open(README).read(); ref = open(os.path.join(DOCS, "AXI_TECHNICAL_REFERENCE.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1
    j = cm.index("\n120. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "COST_OF_BEING_WRONG.md" in item and "cost_of_wrong.py" in item and "test 146" in item and "Rajnish Choubey" in item and "not measured" in item.lower()
    assert "test 146 checks the cost-of-being-wrong page, item 120" in cm
    row = next(l for l in vf.split("\n") if "COST_OF_BEING_WRONG.md" in l); assert "test_cost_of_wrong.py" in row and "recomputed from the saved files" in row and "not measured" in row.lower()
    assert "docs/COST_OF_BEING_WRONG.md" in rd.split("## Start reading here")[1].split("## Status")[0]
    assert "COST_OF_BEING_WRONG.md" in ref
    for f in (C.FILE_22, C.FILE_23): assert os.path.exists(os.path.join(RES, f))


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

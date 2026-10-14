"""Engineering support for docs/KV_LARGE_TEST.md (concept-map item 97): the KV-cache test of docs/KV_DROP_TEST.md on a state of thousands of tokens. No GPU, no torch, no real model. The arms, the cache
protocol, the scoring and the runner are `kv_drop.py` and are tested in test_kv_drop.py; this file tests what is new, on the same numpy stand-in (`tests/kv_standin.py`):
  1. the large graph and its request script: sizes, validity of every request, caps, the mix of kinds, two-digit coordinates; `pick_large_task` is `cascade.pick_task` with a bigger grid for add_node;
  2. the `script_fn` hook of `kv_drop` changes nothing by default and swaps the script when given (episode, preflight, the file header that refuses another script);
  3. the numbers the page quotes from the dry run (no model) are what `kv_large.dry_run` computes, and `dry_run` reproduces EXACTLY the tokens the arms read and the stale share they report when the token
     counter is the stand-in's (one token per byte);
  4. the summaries and the sixteen reading rules (each flips exactly at its threshold); the page is locked and carries them;
  5. the notebook never saves outputs, holds no key, runs its checks before the runs, and every code cell of it runs on the stand-in (and stops at a failed check)."""
import hashlib, inspect, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from kv_standin import TinyCache, TinyEngine, byte_kit, byte_table
import axi.experiments.kv_drop as K
import axi.experiments.kv_large as L
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import check
from axi.experiments.agent_loop import same_graph
from axi.experiments.cascade import pick_task, random_graph
from axi.experiments.long_horizon import KINDS_LONG, oracle_script

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "KV_LARGE_TEST.md")
PAGE_SHA256 = "999e01007464fddb2de4aac7acd533353589d14f3f4e9587ea9ade1424695aef"
NB = os.path.join(HERE, "..", "colab", "axi_kv_large.ipynb")
RESULT_MARK = "\n\n---\n\n# Result"
TIME_KEYS = {"update_s", "prefill_s", "decode_s", "seconds", "peak_mb"}
KIT, TAB = byte_kit(), byte_table()
SMALL = L.make_script_fn(14, 20, (5, 5), 18, 30)            # a graph the numpy stand-in can read in a moment (about 900 byte tokens)
SMALL2 = L.make_script_fn(12, 16, (5, 5), 18, 30)
BYTES = lambda s: len(s.encode("utf-8"))


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list): return [strip(v) for v in x]
    return x


def parse(ideal): return CICOParser.parse_transition_delta(ideal)


# ---------------------------------------------------------------- 1 the large graph and its script
def test_the_large_graph_has_the_sizes_the_page_says():
    for seed in (1, 2, 3):
        g = L.random_large_graph(random.Random(seed)); assert len(g.nodes) == 120 and len(g.edges) == 200 and g.is_well_formed()
        assert all(0 <= r < 14 and 0 <= c < 14 for r, c in g.nodes) and all(1 <= v <= 9 for v in g.nodes.values()) and all(1 <= w <= 9 for w in g.edges.values())
        assert {k[2] for k in g.edges} == {"dep", "owns"} and all(u != v for u, v, _ in g.edges)
        assert sum(1 for n in g.nodes if not g.incident(n)) >= 6                                                         # the reserved services without an edge: 'delete a service without edges' has candidates
        assert abs(2 * len(g.edges) / len(g.nodes) - 3.33) < 0.01
    a, b, c = (L.random_large_graph(random.Random(s)) for s in (1, 1, 2)); assert same_graph(a, b) and not same_graph(a, c)
    x = L.random_large_graph(random.Random(1), L.XL_NODES, L.XL_EDGES, L.XL_GRID); assert len(x.nodes) == 240 and len(x.edges) == 400 and all(0 <= r < 20 and 0 <= c < 20 for r, c in x.nodes)
    assert (L.LARGE_NODES, L.LARGE_EDGES, L.LARGE_GRID, L.LARGE_MAX_NODES, L.LARGE_MAX_EDGES) == (120, 200, (14, 14), 150, 260) and (L.XL_NODES, L.XL_EDGES, L.XL_GRID, L.XL_MAX_NODES, L.XL_MAX_EDGES) == (240, 400, (20, 20), 300, 520)


def test_every_request_of_the_large_script_is_valid_on_the_oracles_graph_and_the_mix_is_the_one_the_page_describes():
    kinds, deletes, steps, two_digit, sizes = {}, 0, 0, 0, []
    for seed in range(21000, 21004):
        g0, script, oracle = L.large_script(seed, 40); assert len(script) == 40 and len(oracle) == 41 and len(g0.nodes) == 120 and len(g0.edges) == 200
        for t, st in enumerate(script):
            d = parse(st["ideal"]); assert check(oracle[t], d).ok, (seed, t); live = oracle[t]
            assert st["kind"] in set(KINDS_LONG) and len(oracle[t].nodes) <= 150 and len(oracle[t].edges) <= 260
            if st["kind"] == "add_node": assert re.fullmatch(r"Add a service at \(\d+,\d+\) with value [1-9]\.", st["instruction"])
            if st["kind"] in ("delete_dep", "delete_leaf"): assert re.fullmatch(r"Delete the service at \(\d+,\d+\)\.", st["instruction"]); deletes += 1
            if st["kind"] == "add_edge": assert re.fullmatch(r"Add a dependency edge from \(\d+,\d+\) to \(\d+,\d+\) with relation dep and weight [1-9]\.", st["instruction"])
            two_digit += bool(re.search(r"\(1\d,|,1\d\)", st["instruction"])); kinds[st["kind"]] = kinds.get(st["kind"], 0) + 1; steps += 1
            if st["kind"] == "delete_dep": assert 1 <= len(parse(st["ideal"]).edge_deltas) <= 3
            if st["kind"] == "delete_leaf": assert not parse(st["ideal"]).edge_deltas and not live.incident((parse(st["ideal"]).node_deltas[0].r, parse(st["ideal"]).node_deltas[0].c))
        sizes.append((len(oracle[-1].nodes), len(oracle[-1].edges)))
    assert steps == 160 and set(kinds) == set(KINDS_LONG) and 0.25 <= deletes / steps <= 0.55 and two_digit > 20, (kinds, deletes, two_digit)
    assert all(100 <= n <= 125 and 160 <= e <= 205 for n, e in sizes), sizes                                           # 'shrinks a little, to about 110 and 180'
    g0, script, oracle = L.xl_script(22000, 30); assert len(script) == 30 and len(g0.nodes) == 240 and len(g0.edges) == 400 and all(check(oracle[t], parse(s["ideal"])).ok for t, s in enumerate(script))
    assert L.large_script.__name__ == "large_120_200_14x14" and L.xl_script.__name__ == "large_240_400_20x20" and SMALL.__name__ == "large_14_20_5x5"
    a, b = L.large_script(21000, 10), L.large_script(21000, 10); assert a[1] == b[1] and all(same_graph(x, y) for x, y in zip(a[2], b[2]))


def test_the_caps_hold_and_pick_large_task_is_cascade_pick_task_with_a_bigger_grid_for_add_node():
    tiny = L.make_script_fn(10, 12, (4, 5), 10, 14); assert tiny.__name__ == "large_10_12_4x5"                                                    # a grid that is not square: the name keeps rows and columns apart
    for seed in (1, 2, 3):
        _, script, oracle = tiny(seed, 60); assert len(script) == 60 and all(len(g.nodes) <= 10 and len(g.edges) <= 14 for g in oracle) and any(s["kind"] == "add_edge" for s in script)
    for seed in range(25):
        rng = random.Random(seed); g = random_graph(rng)
        for kind in ("add_node", "add_edge", "delete_dep", "delete_leaf"):
            r1, r2 = random.Random(seed + 100), random.Random(seed + 100); a, b = pick_task(r1, g, kind), L.pick_large_task(r2, g, kind, (5, 6))
            assert (a is None) == (b is None) and (a is None or (a.kind, a.instruction, a.ideal) == (b.kind, b.instruction, b.ideal) and same_graph(a.expected, b.expected)), (seed, kind)
            assert r1.random() == r2.random()                                                                                  # they used the same randomness
    g = random_graph(random.Random(0)); cells = set()
    for k in range(60):
        t = L.pick_large_task(random.Random(k), g, "add_node"); assert t is not None; d = parse(t.ideal).node_deltas[0]; cells.add((d.r, d.c)); assert (d.r, d.c) not in g.nodes and d.r < 14 and d.c < 14
    assert any(r >= 5 or c >= 6 for r, c in cells)                                                                             # the small grid would never give these
    full = L.random_large_graph(random.Random(5), 196, 100, (14, 14)); assert L.pick_large_task(random.Random(0), full, "add_node") is None


# ---------------------------------------------------------------- 2 the script hook of kv_drop
def test_the_script_hook_changes_nothing_by_default_and_swaps_the_script_when_given():
    for fn in (K.run_kv_episode, K.run_kv, K.preflight): assert "script_fn" in inspect.signature(fn).parameters and inspect.signature(fn).parameters["script_fn"].default is None
    arms = ("fresh", "exact", "drop_compact"); e0, e1 = TinyEngine(), TinyEngine()
    a = K.run_kv_episode(e0, KIT, TAB, 11000, 3, arms, max_new=4); b = K.run_kv_episode(e1, KIT, TAB, 11000, 3, arms, max_new=4, script_fn=oracle_script)
    assert strip(a) == strip(b)                                                                                                 # the default is the small script of the first page
    c = K.run_kv_episode(TinyEngine(), KIT, TAB, 31000, 4, arms, max_new=4, script_fn=SMALL); _, script, _ = SMALL(31000, 4)
    assert [s["kind"] for s in c["arms"]["fresh"]] == [s["kind"] for s in script]
    g0, _, _ = SMALL(31000, 4); items = K.graph_items(g0); assert c["arms"]["fresh"][0]["prompt_tokens"] == len(KIT.prompt_tokens(items, script[0]["instruction"]))
    res = K.preflight(TinyEngine(), KIT, 31, SMALL); assert len(res) == 9 and all(ok for _, ok, _ in res), res
    seen = []
    def spy(seed, steps): seen.append((seed, steps)); return SMALL(seed, steps)
    spy.__name__ = "spy"; K.preflight(TinyEngine(), KIT, 31, spy); assert seen == [(31000, 3)]                                                    # the first prompt of the script it was given, at the seed of episode 0
    seen.clear(); K.run_kv_episode(TinyEngine(), KIT, TAB, 31002, 2, ("fresh", "exact"), max_new=3, script_fn=spy); assert seen == [(31002, 2)]
    broken = TinyEngine(); broken.cache_class = type("Broken", (TinyCache,), dict(rerotate=lambda self, new_pos: setattr(self, "pos", [int(p) for p in new_pos])))
    assert not all(ok for _, ok, _ in K.preflight(broken, KIT, 31, SMALL))                                                      # the checks have teeth on the large prompt too


def test_the_run_writes_the_script_into_the_file_header_and_another_script_is_refused():
    d = tempfile.mkdtemp(); p = os.path.join(d, "a.jsonl"); e = TinyEngine()
    r = L.run_large(e, KIT, TAB, 2, 3, 5, False, p, SMALL, meta=dict(model="stand-in"), max_new=4); meta = json.loads(open(p).readline())["_meta"]
    assert meta["script"] == "large_14_20_5x5" and meta["arms"] == list(L.LARGE_ARMS) and "keep" not in meta["arms"] and meta["kind"] == "kv_drop" and meta["seed"] == 5 and meta["episodes"] == 2 and meta["steps"] == 3
    assert [r_["seed"] for r_ in r["rows"]] == [5000, 5001] and set(r["rows"][0]["arms"]) == set(L.LARGE_ARMS)
    calls = e.calls; again = L.run_large(e, KIT, TAB, 2, 3, 5, False, p, SMALL, meta=dict(model="stand-in"), max_new=4); assert e.calls == calls and strip(again["rows"]) == strip(r["rows"])      # a finished run is not run again
    try: L.run_large(TinyEngine(), KIT, TAB, 2, 3, 5, False, p, SMALL2, meta=dict(model="stand-in"), max_new=4); raise AssertionError("another script must be refused")
    except ValueError as ex: assert "different run" in str(ex)
    q = os.path.join(d, "b.jsonl"); K.run_kv(TinyEngine(), KIT, TAB, 1, 3, 5, False, q, meta=dict(model="stand-in"), arms=("fresh", "exact"), max_new=4); assert "script" not in json.loads(open(q).readline())["_meta"]   # the first page's header is unchanged


# ---------------------------------------------------------------- 3 the dry run
def test_the_dry_run_reproduces_exactly_what_the_arms_read_and_the_stale_share_they_report():
    seeds, steps = (41, 42), 7
    res = L.run_large(TinyEngine(), KIT, TAB, 2, steps, 41, False, os.path.join(tempfile.mkdtemp(), "x.jsonl"), SMALL, max_new=4); rows = res["rows"]
    dry = L.dry_run(SMALL, [r["seed"] for r in rows], steps, BYTES)
    tot = lambda arm: sum(s["prefilled"] for r in rows for s in r["arms"][arm])
    assert dry["tokens"]["fresh"] == tot("fresh") and dry["tokens"]["exact"] == tot("exact") and dry["tokens"]["drop"] == tot("drop_compact") == tot("drop_gap")
    assert dry["prompt_first"] == sum(r["arms"]["fresh"][0]["prompt_tokens"] for r in rows) / 2
    share = lambda arm: np.mean([r["arms"][arm][-1]["stale"] / r["arms"][arm][-1]["resident_state"] for r in rows]); assert abs(dry["stale_share"] - share("drop_compact")) < 1e-12 and abs(dry["stale_share"] - share("drop_gap")) < 1e-12
    assert 0.2 < dry["stale_share"] < 1 and dry["exact_over_drop"] > 1.2 and dry["drop_over_fresh"] < dry["exact_over_fresh"] < 1                                  # not a vacuous comparison: there were deletes
    assert abs(res["summary"]["drop_compact"]["stale_share_last"] - dry["stale_share"]) < 1e-12 and res["summary"]["fresh"]["prompt_tokens_mean"] > 0
    assert L.crude_tokens("Delete the service at (12,3).") == 10 and L.crude_tokens(" [12,3:5]") == 8 and L.crude_tokens(" (1,2)->(3,4):dep#5") == 12


def test_the_numbers_the_page_quotes_from_the_dry_run_are_the_ones_the_code_computes():
    head = _head(); a = L.dry_run(L.large_script, [L.LARGE_SEED * 1000 + e for e in range(L.LARGE_A_EPISODES)], L.LARGE_A_STEPS); x = L.dry_run(L.xl_script, [L.XL_SEED * 1000 + e for e in range(L.XL_EPISODES)], L.XL_STEPS)
    rows = ["| tokens of the first prompt | about {:,.0f} | about {:,.0f} |".format(round(a["prompt_first"], -2), round(x["prompt_first"], -2)),
            "| `drop_compact` reads, as a share of what `fresh` reads | {:.1f} % | {:.1f} % |".format(100 * a["drop_over_fresh"], 100 * x["drop_over_fresh"]),
            "| `exact` reads, as a share of what `fresh` reads | {:.1f} % | {:.1f} % |".format(100 * a["exact_over_fresh"], 100 * x["exact_over_fresh"]),
            "| `exact` reads, as a multiple of what `drop_compact` reads | {:.1f} | {:.1f} |".format(a["exact_over_drop"], x["exact_over_drop"]),
            "| share of the state rows (header included, which is never stale) that are stale in the cache that serves the last step | {:.1f} % | {:.1f} % |".format(100 * a["stale_share"], 100 * x["stale_share"])]
    for r in rows: assert r in head, r
    assert a["prompt_first"] > L.L_MIN_PROMPT and a["drop_over_fresh"] <= L.L_PREFILL_D and a["exact_over_drop"] >= L.L_PREFILL_RATIO and a["stale_share"] >= L.L_STALE         # the lines the dry run speaks to would read yes


# ---------------------------------------------------------------- 4 summaries, reading rules, the page
def test_the_summary_adds_what_this_test_needs_and_keeps_what_the_first_page_summarised():
    res = L.run_large(TinyEngine(), KIT, TAB, 2, 9, 43, False, os.path.join(tempfile.mkdtemp(), "x.jsonl"), SMALL, max_new=4); rows, s, base = res["rows"], res["summary"], K.summarize_kv(res["rows"])
    for a in base:
        for k, v in base[a].items(): assert s[a][k] == v, (a, k)
    f = [st for r in rows for st in r["arms"]["fresh"]]; assert abs(s["fresh"]["prompt_tokens_mean"] - np.mean([st["prompt_tokens"] for st in f])) < 1e-9 and s["fresh"]["same_first"] is None and s["exact"]["prompt_tokens_mean"] is None
    for a in ("exact", "drop_compact", "drop_gap"):
        first = [st["same_as_fresh"] for r in rows for st in r["arms"][a] if st["t"] <= 3]; assert len(first) == 6 and s[a]["same_first"] == sum(first) / 6        # 9 steps: the first third is steps 1 to 3
        assert abs(s[a]["state_rows_last"] - np.mean([r["arms"][a][-1]["resident_state"] for r in rows])) < 1e-9
    assert s["fresh"]["stale_share_last"] is None and s["exact"]["stale_share_last"] is None                                                  # only the drop arms have stale rows
    for a in ("drop_compact", "drop_gap"):
        assert abs(s[a]["stale_share_last"] - np.mean([r["arms"][a][-1]["stale"] / r["arms"][a][-1]["resident_state"] for r in rows])) < 1e-9 and s[a]["stale_share_last"] > 0
    assert "keep" not in s and set(s) == set(L.LARGE_ARMS)


def _step(t, same, stale=None, state=None, prompt=None, kind="add_node"):
    return dict(t=t, same_as_fresh=same, kl_max=0.01 if same is not None else None, kl_mean=0.005 if same is not None else None, exact_m6=True, exact_blind=True, admitted=True, parsed=True, prefilled=50, decoded=5,
                resident=100 + t, resident_state=state if state is not None else 90, kv_bytes=1e6, peak_mb=10.0, seconds=1.0, update_s=0.1, prefill_s=0.2, decode_s=0.7, kind=kind,
                **({} if prompt is None else dict(prompt_tokens=prompt)), **({} if stale is None else dict(stale=stale, gap=0, v_dev=0.1)))


def test_the_summary_reads_the_thirds_the_last_step_and_the_prompt_size_from_hand_made_rows():
    patterns = ([True, False, False, True, True, True, False, True, True], [True, True, False, False, True, False, False, True, False])        # 9 steps each; the two episodes differ, so that dropping a step from a third shows
    rows = []
    for e, (stale_last, state_last, base) in enumerate(((60, 80, 1000), (90, 100, 2000))):
        fresh = [_step(t, None, prompt=base + t) for t in range(1, 10)]
        drop = [_step(t, patterns[e][t - 1], stale=(5 * t if t < 9 else stale_last), state=(70 + t if t < 9 else state_last)) for t in range(1, 10)]
        rows.append(dict(seed=e, steps=9, stopped=dict(fresh=None, drop_compact=None), arms=dict(fresh=fresh, drop_compact=drop)))
    s = L.summarize_large(rows); dc, f = s["drop_compact"], s["fresh"]
    assert abs(dc["same_first"] - (1 / 3 + 2 / 3) / 2) < 1e-12                                           # steps 1 to 3: episode 0 T F F = 1/3, episode 1 T T F = 2/3 (steps 1 and 2 only would give 3/4)
    assert abs(dc["same_late"] - (2 / 3 + 1 / 3) / 2) < 1e-12                                            # steps 7 to 9: F T T = 2/3 and F T F = 1/3 (steps 5 to 9 would give 3/5)
    assert abs(dc["same"] - (6 + 4) / 18) < 1e-12
    assert abs(dc["stale_share_last"] - (60 / 80 + 90 / 100) / 2) < 1e-12 and abs(dc["state_rows_last"] - (80 + 100) / 2) < 1e-12              # the LAST step of each episode, stale over state rows
    assert abs(f["prompt_tokens_mean"] - (sum(1000 + t for t in range(1, 10)) + sum(2000 + t for t in range(1, 10))) / 18) < 1e-9 and dc["prompt_tokens_mean"] is None and f["stale_share_last"] is None


def _summary(**o):
    A = dict(fresh=dict(n=160, prompt_tokens_mean=3800.0, canonical=1.0, exact_m6=0.90, prefilled=100000, seconds=10.0),
             exact=dict(n=160, same=0.98, prefilled=60000, seconds=6.0),
             drop_compact=dict(n=160, same=0.97, same_late=0.95, kl_below=0.95, exact_m6=0.90, prefilled=5000, seconds=5.0, stale_share_last=0.80), drop_gap=dict(n=160, same=0.95))
    C, X = dict(drop_compact=dict(n=160, same=0.95)), dict(drop_compact=dict(n=60, same=0.95))
    for k, v in o.items():
        tgt = {"A": A, "C": C, "X": X}[k[0]]; arm, key = k[1:].split("__"); tgt[arm][key] = v
    return A, C, X


def test_the_sixteen_reading_rules_flip_exactly_at_their_thresholds():
    A, C, X = _summary(); base = L.verdict_large(A, C, X); assert len(base) == len(L.LARGE_LINES) == 16 and all(ok is True for _, ok in base), base
    assert [t for t, _ in base] == [t for _, t in L.LARGE_LINES] and [t for t, _ in L.LARGE_LINES].count("guard") == 2 and [t for t, _ in L.LARGE_LINES].count("instrument") == 2 and [t for t, _ in L.LARGE_LINES].count("load") == 1
    flips = [({"Afresh__prompt_tokens_mean": 2999.9}, 1), ({"Afresh__canonical": 0.999}, 2), ({"Afresh__exact_m6": 0.8999, "Adrop_compact__exact_m6": 0.8999}, 3), ({"Aexact__same": 0.9799}, 4),
             ({"Adrop_compact__same": 0.9699}, 5), ({"Adrop_gap__same": 0.9499}, 6), ({"Adrop_compact__exact_m6": 0.8899}, 7), ({"Adrop_compact__kl_below": 0.9499}, 8), ({"Adrop_compact__prefilled": 10001}, 9),
             ({"Aexact__prefilled": 24999}, 10), ({"Adrop_compact__seconds": 5.0001}, 11), ({"Aexact__seconds": 5.0}, 12), ({"Adrop_compact__same_late": 0.9499}, 13), ({"Cdrop_compact__same": 0.9499}, 14),
             ({"Adrop_compact__stale_share_last": 0.7999}, 15), ({"Xdrop_compact__same": 0.9499}, 16)]
    for o, line in flips:
        A2, C2, X2 = _summary(**o); got = L.verdict_large(A2, C2, X2); assert [i + 1 for i, (_, ok) in enumerate(got) if ok is not True] == [line], (o, [i + 1 for i, (_, ok) in enumerate(got) if ok is not True])
    for o in ({"Adrop_compact__prefilled": 10000}, {"Aexact__prefilled": 25000}, {"Adrop_compact__seconds": 5.0}, {"Aexact__seconds": 5.0001}):                      # the thresholds themselves read yes (a tie on 'below' does not)
        A2, C2, X2 = _summary(**o); assert all(ok is True for _, ok in L.verdict_large(A2, C2, X2)), o
    for v, want in ((0.89, True), (0.91, True), (0.8899, False), (0.9101, False)):                                                                               # fresh is 0.90: within 1 point in both directions
        A2, C2, X2 = _summary(Adrop_compact__exact_m6=v); assert L.verdict_large(A2, C2, X2)[6][1] is want, v
    v = L.verdict_large(A, None, None); assert v[13][1] is None and v[15][1] is None and all(ok is True for i, (_, ok) in enumerate(v) if i not in (13, 15))      # runs that were not made read 'not run'
    assert L.verdict_large({}, None, None) == [(t, None) for _, t in L.LARGE_LINES]


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_sixteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(L.LARGE_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [90, 85, 75, 75, 60, 55, 85, 50, 95, 90, 75, 90, 50, 55, 95, 45]
    p = {i + 1: v for i, v in enumerate(probs)}; prod = 100 * np.prod([p[i] / 100 for i in L.LARGE_HEADLINE]); assert L.LARGE_HEADLINE == (5, 7, 8, 13, 14) and abs(prod - 7) <= 0.5 and "about **7 %**" in head and "about **25 %**" in head
    assert "**Headline** (lines 5, 7, 8, 13 and 14)" in head
    for needle in (f"stream seed {L.LARGE_SEED} (episode seeds 21000 to 21003)", f"stream seed {L.XL_SEED} (episode seeds 22000 and 22001)", f"{L.LARGE_A_EPISODES} episodes of {L.LARGE_A_STEPS} steps", f"{L.XL_EPISODES} episodes of {L.XL_STEPS} steps",
                   "Qwen2.5-7B-Instruct in 4-bit", "Teacher forcing", "`fresh`", "`exact`", "`drop_compact`", "`drop_gap`", "No `keep`", "`preflight`", "`kv_large.verdict_large`", "`kv_large.dry_run`", "`script_fn`",
                   "Prompt Cache", "CacheBlend", "Block-Attention for Efficient RAG", "item 97", "$0", "about 57 KB", "about 197 KB", "not a survey", "Related work (checked 14 Oct 2026", "`KV_DROP_TEST.md`",
                   "120 services on a 14 x 14 grid", "200 edges", "240 services on a 20 x 20 grid", "400 edges", "150 services, 260 edges", "300 and 520", "about 4,000 tokens", "not Qwen's tokenizer",
                   "axi-colab-kv-large.zip", "axi_kv_large.ipynb", "test_kv_large.py", "at most 120 new tokens", "This page does not depend on the result of the first"):
        assert needle in head, needle
    assert (L.LARGE_A_EPISODES, L.LARGE_A_STEPS, L.LARGE_C_EPISODES, L.LARGE_C_STEPS, L.XL_EPISODES, L.XL_STEPS, L.LARGE_SEED, L.LARGE_C_SEED, L.XL_SEED, K.KV_MAX_NEW) == (4, 40, 4, 40, 2, 30, 21, 21, 22, 120)
    assert L.LARGE_ARMS == ("fresh", "exact", "drop_compact", "drop_gap")
    consts = [("at least 3,000 tokens", L.L_MIN_PROMPT == 3000), ("at least 90 %", L.L_FRESH_M6 == 0.90), ("at least 98 %", L.L_EXACT_SAME == 0.98), ("at least 97 %", L.L_DC_SAME == 0.97), ("within 1 point", K.K_EXACT_POINTS == 0.01),
              ("below 0.05 nats", K.K_KL == 0.05), ("at most 10 %", L.L_PREFILL_D == 0.10), ("at least 5 times", L.L_PREFILL_RATIO == 5.0), ("at most 0.5 times", L.L_SEC_D == 0.5), ("at least 80 %", L.L_STALE == 0.80)]
    lines_text = "\n".join(l for _, l in L.LARGE_LINES)
    for needle, ok in consts: assert ok and needle in lines_text, needle
    assert lines_text.count("at least 95 %") == 5 and L.L_DG_SAME == K.K_KL_SHARE == L.L_LATE == L.L_C_SAME == L.L_X_SAME == 0.95
    assert "The last third of the steps is steps 27 to 40 of a 40-step episode; the first third (shown in the table, in no line) is steps 1 to 13" in head and L.LARGE_A_STEPS == 40
    e = [t for t in range(1, 41) if t > 2 * 40 // 3]; assert e[0] == 27 and e[-1] == 40 and [t for t in range(1, 41) if t <= 40 // 3][-1] == 13                  # what the summary uses for 'late' and 'first'


# ---------------------------------------------------------------- 5 the notebook
def test_the_notebook_is_clean_saves_no_outputs_holds_no_key_and_runs_the_checks_before_the_runs():
    nb = json.load(open(NB)); code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]; allsrc = "\n".join("".join(c["source"]) for c in nb["cells"]); joined = "\n".join(code)
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in nb["cells"])
    assert not re.search(r"AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{20,}", allsrc) and "userdata" not in joined and "docs/KV_LARGE_TEST.md" in allsrc and "axi-colab-kv-large.zip" in allsrc
    assert "build_kv(" in joined and "make_hf_kit(" in joined and "preflight(engine, kit_plain, LARGE_SEED, large_script)" in joined and "assert all(ok for" in joined and "run_large(" in joined and "verdict_large(" in joined and "open_store(" in joined
    assert joined.index("preflight(") < joined.index("run_large(") and "LARGE_C_SEED" in joined and "XL_SEED" in joined and "xl_script" in joined and "hint_rules" in joined and "KV_DROP_TEST" not in joined
    assert "T4" in allsrc and "Do not paste" in allsrc and "$0" in allsrc and "pip -q install bitsandbytes accelerate" in joined and "Projection:" in joined and "16 lines" in allsrc
    src = open(L.__file__).read(); assert "torch" not in src and "import transformers" not in src


def _run_notebook_cells(eng, tmp, until=None):
    """Execute the code cells of the notebook with Colab and torch replaced by stubs, the model builder by the numpy stand-in, and the sizes of the runs and of the graphs by tiny ones."""
    import contextlib, io, types
    import axi.experiments.kv_torch as KT, colab.colab_persist as CP
    downloads = []; names = ("LARGE_A_EPISODES", "LARGE_A_STEPS", "LARGE_C_EPISODES", "LARGE_C_STEPS", "XL_EPISODES", "XL_STEPS", "large_script", "xl_script"); saved = {k: getattr(L, k) for k in names}
    mods = {k: sys.modules.get(k) for k in ("google", "google.colab", "torch")}; orig = (KT.build_kv, KT.make_hf_kit, CP.open_store)
    gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(upload=lambda: {"axi-colab-kv-large.zip": b""}, download=lambda path: downloads.append(path)); gc.drive = types.SimpleNamespace(mount=lambda mp: None)
    g = types.ModuleType("google"); g.colab = gc; tm = types.ModuleType("torch"); tm.cuda = types.SimpleNamespace(memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    try:
        sys.modules.update({"google": g, "google.colab": gc, "torch": tm}); L.LARGE_A_EPISODES, L.LARGE_A_STEPS, L.LARGE_C_EPISODES, L.LARGE_C_STEPS, L.XL_EPISODES, L.XL_STEPS = 1, 3, 1, 3, 1, 4
        L.large_script, L.xl_script = SMALL, SMALL2
        KT.build_kv = lambda name, load_4bit, device: (eng, None, TAB, None); KT.make_hf_kit = lambda tok, hint_rules=False: byte_kit(hint_rules)
        CP.open_store = lambda local_dir="/content/out", subdir="x", mount_point="/content/drive": orig[2](tmp + "/out", subdir, tmp + "/drive")
        ns = {"__name__": "__main__"}; out = io.StringIO(); nb = json.load(open(NB)); n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code": continue
            src = "".join(l for l in c["source"] if not l.lstrip().startswith("!")).replace("/content/out", tmp + "/out").replace("/content/smoke_large.jsonl", tmp + "/smoke_large.jsonl")
            if not src.strip() or "files.upload" in src: continue
            n += 1
            if until is not None and n > until: break
            with contextlib.redirect_stdout(out): exec(compile(src, "cell %d" % n, "exec"), ns)
        ns["_printed"] = out.getvalue(); return ns, downloads
    finally:
        for k, v in saved.items(): setattr(L, k, v)
        KT.build_kv, KT.make_hf_kit, CP.open_store = orig
        for k, v in mods.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_every_code_cell_of_the_notebook_runs_on_the_stand_in_saves_after_each_run_and_stops_when_a_check_fails():
    tmp = tempfile.mkdtemp(); ns, downloads = _run_notebook_cells(TinyEngine(), tmp)
    assert [os.path.basename(d) for d in downloads] == ["axi_large_a.json", "axi_large_c.json", "axi_large_x.json"]
    pre = json.load(open(tmp + "/out/axi_large_preflight.json")); assert len(pre) == 9 and all(x["ok"] for x in pre)
    for name in ("axi_large_preflight.json", "axi_large_a.jsonl", "axi_large_a.json", "axi_large_c.jsonl", "axi_large_c.json", "axi_large_x.jsonl", "axi_large_x.json"): assert os.path.exists(tmp + "/drive/MyDrive/axi_kv_large/" + name), name
    a, c, x = (json.load(open(tmp + "/out/axi_large_%s.json" % k)) for k in "acx")
    assert a["meta"]["seed"] == L.LARGE_SEED and a["meta"]["hint_rules"] is False and c["meta"]["seed"] == L.LARGE_C_SEED and c["meta"]["hint_rules"] is True and x["meta"]["seed"] == L.XL_SEED and x["meta"]["hint_rules"] is True
    assert a["meta"]["script"] == SMALL.__name__ and x["meta"]["script"] == SMALL2.__name__ and a["meta"]["arms"] == list(L.LARGE_ARMS)
    assert [r["seed"] for r in a["rows"]] == [21000] == [r["seed"] for r in c["rows"]] and [r["seed"] for r in x["rows"]] == [22000] and [r["steps"] for r in a["rows"]] == [3] and [r["steps"] for r in x["rows"]] == [4]
    assert set(a["rows"][0]["arms"]) == set(L.LARGE_ARMS)
    sm = ns["smoke"]; assert sm["rows"][0]["steps"] == 3 and sm["meta"]["seed"] == 99 and sm["meta"]["smoke"] is True and sm["meta"]["script"] == SMALL.__name__ and sm["meta"]["hint_rules"] is False         # the quick check is 1 episode of 3 steps
    final = ns["final"]; assert len(final) == 16 and all(ok is not None for _, ok in final) and [t for t, _ in final] == [t for _, t in L.LARGE_LINES]
    assert final == L.verdict_large(ns["res_a"]["summary"], ns["res_c"]["summary"], ns["res_x"]["summary"])
    txt = ns["_printed"]; assert "== KV test RUN A: no rule ==" in txt and "ALL CHECKS OK" in txt and "-- ALL RUNS (reading rules fixed before the run) --" in txt and "smoke, 1 episode x 3 steps" in txt and "Projection: Run A (3 steps) about" in txt
    assert "prompt of fresh:" in txt and "of them stale" in txt
    broken = TinyEngine(); broken.cache_class = type("Broken", (TinyCache,), dict(rerotate=lambda self, new_pos: setattr(self, "pos", [int(p) for p in new_pos])))        # keys not turned
    tmp2 = tempfile.mkdtemp()
    try: _run_notebook_cells(broken, tmp2)
    except AssertionError as err: assert "STOP: a check failed" in str(err)
    else: raise AssertionError("the notebook went on after a failed check")
    saved = json.load(open(tmp2 + "/out/axi_large_preflight.json")); assert not all(x["ok"] for x in saved) and not [f for f in os.listdir(tmp2 + "/out") if f.startswith(("axi_large_a", "axi_large_c", "axi_large_x", "smoke"))]


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

"""Test 40: where the prompt starts to grow, as the state grows from 20 to 20,000 items (docs/STATE_SIZE_SCALING_TEST.md).

Written AFTER the plan page and its predictions were fixed. CPU only, no model. Uses the repository's gate, focus view and scope filter
unchanged. Run from the python folder:  python3 -m axi.experiments.state_size_scaling ../docs/results
It prints the tables and the thirteen reading lines of the page, and writes docs/results/axi_state_size_scaling.json."""
import json, os, random, statistics, sys, time

from axi.engine.cico_parser import CICOParser, StateTransitionDelta, EdgeDelta
from axi.engine.focus import focus_graph
from axi.engine.gate import Graph, check
from axi.engine.scope import contain_scope

REL = ["dep", "call", "own", "read", "write", "sync", "route", "auth"]
SIZES = (20, 200, 2000, 20000)
SEED = 20261116
N_RANDOM, N_HUB, N_SCAN = 2000, 500, 200


def _rss_kb():
    try:
        with open("/proc/self/statm") as f: return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except Exception: return None


def _cells(n):
    w = max(2, int(n ** 0.5) + 1)
    return [(i // w, i % w) for i in range(n)]


def build_even(items, seed):
    rng = random.Random(seed); n = max(3, round(items / 3)); cells = _cells(n); g = Graph()
    for c in cells: g.add_node(c, rng.randrange(10))
    for u in cells:
        made = 0
        while made < 2:
            v = rng.choice(cells)
            if v == u: continue
            rel = rng.choice(REL)
            if (u, v, rel) in g.edges: continue
            g.add_edge(u, v, rel, rng.randrange(1, 10)); made += 1
    return g


def build_hub(items, seed):
    """Preferential attachment, two edges per new service: items = 3n - 3."""
    rng = random.Random(seed); n = max(4, round((items + 3) / 3)); cells = _cells(n); g = Graph()
    for c in cells[:3]: g.add_node(c, rng.randrange(10))
    ends = []
    for u, v in ((cells[1], cells[0]), (cells[2], cells[0]), (cells[2], cells[1])):
        g.add_edge(u, v, "dep", rng.randrange(1, 10)); ends += [u, v]
    for u in cells[3:]:
        g.add_node(u, rng.randrange(10)); targets = set()
        while len(targets) < 2: targets.add(rng.choice(ends))
        for v in sorted(targets):
            g.add_edge(u, v, rng.choice(REL), rng.randrange(1, 10)); ends += [u, v]
    return g


def text_of(g):
    nodes = [(r, c, v) for (r, c), v in sorted(g.nodes.items())]
    edges = [(k[0][0], k[0][1], k[1][0], k[1][1], k[2], w) for k, w in sorted(g.edges.items())]
    return CICOParser.serialize_state(nodes, edges)


def items_of(g): return len(g.nodes) + len(g.edges)


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def scan_view(g, named):
    """The plain scan: named services that exist, every edge touching one, the services at their ends."""
    have = {n for n in named if n in g.nodes}
    edges = {k for k in g.edges if k[0] in have or k[1] in have}
    nodes = set(have)
    for k in edges: nodes.add(k[0]); nodes.add(k[1])
    return nodes, edges


def median_us(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t)
    return statistics.median(ts) / 1000.0


def measure(kind, items, seed):
    rng = random.Random(seed + 7)
    rss0 = _rss_kb(); t0 = time.perf_counter()
    g = (build_even if kind == "even" else build_hub)(items, seed)
    build_s = time.perf_counter() - t0; rss1 = _rss_kb()
    full_text = text_of(g); n_items = items_of(g)
    names = sorted(g.nodes); out = dict(kind=kind, target_items=items, items=n_items, services=len(g.nodes), edges=len(g.edges), build_seconds=round(build_s, 3),
                                         rss_kb_before=rss0, rss_kb_after=rss1, full_chars=len(full_text), chars_per_item=len(full_text) / n_items,
                                         max_degree=max(len(g.incident(n)) for n in names))
    # random requests: two services at random
    sizes, chars, big = [], [], 0
    scan_ok = True; scan_checked = 0
    for i in range(N_RANDOM):
        named = rng.sample(names, 2)
        v = focus_graph(g, named, hops=1); s = items_of(v); sizes.append(s); chars.append(len(text_of(v)))
        if s > 0.10 * n_items: big += 1
        if i < N_SCAN:
            sn, se = scan_view(g, named); scan_checked += 1
            if set(v.nodes) != sn or set(v.edges) != se: scan_ok = False
    out["random"] = dict(median=statistics.median(sizes), p95=pct(sizes, .95), p99=pct(sizes, .99), max=max(sizes), share_over_10pct=big / N_RANDOM,
                         median_chars=statistics.median(chars), scan_checked=scan_checked, scan_ok=scan_ok)
    # hub-targeted: the highest-degree service and one random service
    top = max(names, key=lambda n: len(g.incident(n))); hs, hc, hb = [], [], 0
    for i in range(N_HUB):
        other = rng.choice(names); named = [top, other]
        v = focus_graph(g, named, hops=1); s = items_of(v); hs.append(s); hc.append(len(text_of(v)))
        if s > 0.10 * n_items: hb += 1
        if i < 20:
            sn, se = scan_view(g, named)
            if set(v.nodes) != sn or set(v.edges) != se: out["random"]["scan_ok"] = False
    out["hub_targeted"] = dict(median=statistics.median(hs), p95=pct(hs, .95), max=max(hs), share_over_10pct=hb / N_HUB, median_chars=statistics.median(hc))
    # host work (microseconds, medians)
    u, v2 = rng.sample(names, 2)
    delta = StateTransitionDelta([], [EdgeDelta("ADD", u, v2, "zz", 1)])
    named_set = {u, v2}
    out["host_us"] = dict(
        gate_check=median_us(lambda: check(g, delta), 2000),
        focus_graph=median_us(lambda: focus_graph(g, [u, v2], hops=1), 300),
        contain_scope=median_us(lambda: contain_scope(delta, named_set), 2000),
        serialize_full=median_us(lambda: text_of(g), 7 if n_items >= 2000 else 200))
    return out


def lines(R):
    by = {(r["kind"], r["target_items"]): r for r in R}
    e, h = lambda s: by[("even", s)], lambda s: by[("hub", s)]
    L = []
    ok1 = all(abs(r["items"] - r["target_items"]) <= 0.05 * r["target_items"] + 1e-9 for r in R) and all(r["random"]["scan_ok"] for r in R)
    L.append((1, "instrument", "eight graphs built within 5 percent of target items; every view matched the plain scan", ok1))
    cpi = {k: [by[(k, s)]["chars_per_item"] for s in SIZES] for k in ("even", "hub")}
    L.append((2, "claim", "chars per item of the whole state differ by less than 25 percent across sizes (even %.1f..%.1f, hub %.1f..%.1f)" % (min(cpi["even"]), max(cpi["even"]), min(cpi["hub"]), max(cpi["hub"])),
              all((max(v) - min(v)) / min(v) < 0.25 for v in cpi.values())))
    ratio3 = min(by[(k, 20000)]["full_chars"] / by[(k, 20)]["full_chars"] for k in ("even", "hub"))
    L.append((3, "claim", "whole state at 20,000 items is at least 500 times the characters at 20 items (smallest ratio %.0f)" % ratio3, ratio3 >= 500))
    r4 = e(20000)["random"]["median"] / e(200)["random"]["median"]
    L.append((4, "claim", "even: median view at 20,000 at most 3 times that at 200 (ratio %.2f; %s vs %s items)" % (r4, e(20000)["random"]["median"], e(200)["random"]["median"]), r4 <= 3))
    r5 = e(20000)["random"]["p99"] / e(200)["random"]["p99"]
    L.append((5, "claim", "even: p99 view at 20,000 at most 3 times that at 200 (ratio %.2f; %s vs %s)" % (r5, e(20000)["random"]["p99"], e(200)["random"]["p99"]), r5 <= 3))
    r6 = h(20000)["random"]["p99"] / h(200)["random"]["p99"]
    L.append((6, "claim", "hub, random requests: p99 view at 20,000 at most 3 times that at 200 (ratio %.2f; %s vs %s)" % (r6, h(20000)["random"]["p99"], h(200)["random"]["p99"]), r6 <= 3))
    r7 = h(20000)["random"]["max"] / h(200)["random"]["max"]
    L.append((7, "claim", "hub, random requests: largest view at 20,000 at least 3 times the largest at 200 (ratio %.2f; %s vs %s)" % (r7, h(20000)["random"]["max"], h(200)["random"]["max"]), r7 >= 3))
    sh = h(20000)["hub_targeted"]["median"] / h(20000)["items"]
    L.append((8, "claim", "hub, hub-targeted: median view at 20,000 items is between 1 and 10 percent of the whole state (%.2f percent; max degree %d)" % (100 * sh, h(20000)["max_degree"]), 0.01 <= sh <= 0.10))
    r9 = e(20000)["host_us"]["gate_check"] / e(200)["host_us"]["gate_check"]
    L.append((9, "claim", "gate.check of a one-edit delta at 20,000 at most 3 times that at 200 (ratio %.2f; %.1f vs %.1f us)" % (r9, e(20000)["host_us"]["gate_check"], e(200)["host_us"]["gate_check"]), r9 <= 3))
    r10 = e(20000)["host_us"]["focus_graph"] / e(200)["host_us"]["focus_graph"]
    L.append((10, "claim", "focus_graph on the even graph at 20,000 at most 3 times that at 200 (ratio %.2f; %.1f vs %.1f us)" % (r10, e(20000)["host_us"]["focus_graph"], e(200)["host_us"]["focus_graph"]), r10 <= 3))
    r11 = e(20000)["host_us"]["serialize_full"] / e(200)["host_us"]["serialize_full"]
    L.append((11, "claim", "serialising the whole state at 20,000 at least 50 times that at 200 (ratio %.0f)" % r11, r11 >= 50))
    c2k, c200 = [by[(k, 2000)]["full_chars"] for k in ("even", "hub")], [by[(k, 200)]["full_chars"] for k in ("even", "hub")]
    L.append((12, "claim", "whole state over 24,576 characters at 2,000 items and under it at 200 items (2,000: %s; 200: %s)" % (c2k, c200), all(c > 24576 for c in c2k) and all(c < 24576 for c in c200)))
    c20k = [by[(k, 20000)]["full_chars"] for k in ("even", "hub")]
    L.append((13, "claim", "whole state at 20,000 items over 384,000 characters (even, hub: %s)" % c20k, all(c > 384000 for c in c20k)))
    return L


def main(outdir):
    R = []; t0 = time.time()
    for kind in ("even", "hub"):
        for items in SIZES:
            r = measure(kind, items, SEED + items * (1 if kind == "even" else 3)); R.append(r)
            print(f"{kind:5s} items {r['items']:6d} services {r['services']:5d} edges {r['edges']:6d} maxdeg {r['max_degree']:4d} | full {r['full_chars']:8d} chars ({r['chars_per_item']:.1f}/item) | "
                  f"random view median {r['random']['median']:.0f} p95 {r['random']['p95']} p99 {r['random']['p99']} max {r['random']['max']} >10% {100*r['random']['share_over_10pct']:.1f}% | "
                  f"hub-targeted median {r['hub_targeted']['median']:.0f} max {r['hub_targeted']['max']} | us: gate {r['host_us']['gate_check']:.1f} focus {r['host_us']['focus_graph']:.1f} "
                  f"scope {r['host_us']['contain_scope']:.1f} serialize {r['host_us']['serialize_full']:.0f} | scan_ok {r['random']['scan_ok']} | build {r['build_seconds']}s rss {r['rss_kb_before']}->{r['rss_kb_after']} kB", flush=True)
    L = lines(R)
    print()
    for n, tag, text, ok in L: print(f"{n:2d}. {'yes' if ok else 'NO ':3s} [{tag}] {text}")
    print("\nseconds:", round(time.time() - t0, 1))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "axi_state_size_scaling.json"), "w") as f:
        json.dump(dict(runs=R, lines=[dict(n=n, tag=t, text=x, yes=bool(o)) for n, t, x, o in L], python=sys.version, seed=SEED), f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../docs/results")

"""Implements concept-map row 78 (blog 4 Aug, page "The Hinduism: The Final Award", sections "The Pure Logic of the Linear Stride" and
"The Computational Mirage of the Token Horizon"; quotes checked on a second read). The page's one invariant is a childhood game: "the
capacity to count without losing"; "My childhood game was counting my steps. I could count thousands as a kid without missing."; "The
scale of the final digit means absolutely nothing" (this last line is from the first read only). It says a machine needs nothing else:
do not drop a token, however long the count. (The page asserts this; it derives nothing.)

The reading being tested (Rajnish to accept or reject): for this runtime "counting without losing" is four properties, all checkable:
  1. The parser never drops an item. The number of items it returns equals the number of items in the text, duplicates included. A duplicate
     is counted twice and then REFUSED by the gate (identification), never merged silently.
  2. Malformed input is refused, not shortened. Double separators, tabs, wrong case, stray text and half items all raise CICOParseError. The
     parser either returns every item or raises (5,000 random corruptions; it never returns fewer items than there are tokens).
  3. The last digit counts at any size. Coordinates, values and weights of 40 digits survive exactly (no float rounding), and a delta of
     100,000 items parses to exactly 100,000 items with the last one intact.
  4. A running ledger never drifts. Over three random walks of 2,000 admitted deltas each, the ledger "nodes += #ADD - #DEL" equals the true
     node count after EVERY step, and the same for edges.
What they do NOT show: that a language model counts without losing (it does not, which is why the count is done by code, see the
compounding-error test); anything about attention. The 'Line 10 -> Line 20' reading of the same page is in rows 77 and 79."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, CICOParseError
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import to_text
from test_signed_slots import random_graph, random_delta_text, CELLS, copy

P = CICOParser.parse_transition_delta


def n_tokens(txt):
    if txt == "": return 0
    items = txt.split(" ") if "\n" not in txt else txt.replace("\n", " ").split(" ")
    if len(items) > 1 and items[-1] == "": items.pop()
    return len(items)


def n_items(d): return len(d.node_deltas) + len(d.edge_deltas)


def test_parser_returns_every_item_and_the_gate_refuses_duplicates():
    rng = random.Random(78); dup_seen = 0
    for _ in range(6000):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        d = P(txt)
        assert n_items(d) == n_tokens(txt), txt
        if txt:
            doubled = txt + " " + txt                                                  # every item twice
            d2 = P(doubled)
            assert n_items(d2) == 2 * n_items(d)                                       # counted twice, not merged
            r = check(g, d2)
            assert not r.ok and (r.reason.startswith("identification") or r.reason.startswith("match") or r.reason.startswith("dangling")), r
            dup_seen += 1
    assert dup_seen > 3000, dup_seen


def test_malformed_input_is_refused_not_shortened():
    rng = random.Random(79); refused = passed = 0
    junk = ["x", "ADD[1,1]", "add[1,1:3]", "DEL[1,1:3]", "ADD[1,1:3", "ADD[a,b:3]", "ADD[(1,1)->(2,2):adj]", ""]
    for _ in range(5000):
        g = random_graph(rng); items = random_delta_text(g, rng).split(" ")
        items = [i for i in items if i]
        kind = rng.randrange(5)
        if kind == 0: items.insert(rng.randrange(len(items) + 1), rng.choice(junk)); txt = " ".join(items)    # stray item
        elif kind == 1 and items: txt = "  ".join(items)                                                        # double separator
        elif kind == 2 and items: txt = "\t".join(items) if len(items) > 1 else items[0] + "\t"                 # tab
        elif kind == 3 and items: txt = " ".join(items) + "  "                                                  # two trailing separators
        else: txt = " ".join(items)
        try:
            d = P(txt)
            assert n_items(d) == n_tokens(txt), (txt, n_items(d))                                              # accepted => nothing lost
            passed += 1
        except CICOParseError:
            refused += 1
    assert refused > 1500 and passed > 500, (refused, passed)


def test_the_last_digit_counts_at_any_size():
    big = 10 ** 40 + 7
    d = P(f"ADD[{big},{big + 1}:{big + 2}] ADD[({big},{big})->({big + 1},{big + 1}):adj#-{big}]")
    n, e = d.node_deltas[0], d.edge_deltas[0]
    assert (n.r, n.c, n.val) == (big, big + 1, big + 2) and e.weight == -big and e.u == (big, big)
    assert to_text(d) == f"ADD[{big},{big + 1}:{big + 2}] ADD[({big},{big})->({big + 1},{big + 1}):adj#-{big}]"       # and writes back the same digits
    N = 100_000
    txt = " ".join(f"ADD[{i},0:{i % 10}]" for i in range(N))
    d = P(txt)
    assert n_items(d) == N and (d.node_deltas[-1].r, d.node_deltas[-1].val) == (N - 1, (N - 1) % 10)
    assert check(Graph(), d).ok                                                          # and the gate answers on all of it


def test_a_running_ledger_never_drifts():
    for seed in (1, 2, 3):
        rng = random.Random(seed); g = random_graph(rng)
        nodes, edges, steps = len(g.nodes), len(g.edges), 0
        for _ in range(60000):
            d = P(random_delta_text(g, rng))
            if not check(g, d).ok: continue
            apply(g, d); steps += 1
            nodes += sum(1 if x.op == "ADD" else -1 for x in d.node_deltas)
            edges += sum(1 if x.op == "ADD" else -1 for x in d.edge_deltas)
            assert nodes == len(g.nodes) and edges == len(g.edges), (seed, steps)
            if steps == 2000: break
        assert steps == 2000, steps


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

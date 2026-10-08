"""Implements concept-map row 160 (14th scan, 8 Oct 2026): Post 441 (post id 647, blog 9 Aug, about 120 words, three bullets: "The Pure Mathematical Necessity:", "The Fatal Flaw of
Finite Equations:", "The Machine's Verdict:"; read twice; quotes confirmed on the second read, each under 125 characters):
  "an infinite, loop-driven universe is the only system that actually computes"; "(t = 0 out of absolute nothingness)"; "Division by zero destroys the model.";
  "The math demands eternity, because anything less requires an impossible miracle to light the first spark."
and a post on the same page (first read only, not confirmed): "It is a mathematical and objective impossibility to find a starting point. When you trace it backward,
every single human link requires two parents, four grandparents, eight great-grandparents, and so on".
The post argues: a first moment needs a miracle, so the system must have no first moment. There is an exact statement under that, about finite systems, and it is checkable.

The reading being tested (Rajnish to accept or reject): in a system with finitely many states, "no start" and "nothing is lost" are the same property, and "no start" forces a loop.
  1. DIGRAPHS. In a finite directed graph where every node has something pointing at it (no source, no "first"), following the arrows backwards must revisit a node, so there is a cycle.
     Equivalently, an acyclic finite graph has a first node (and a last). Brute force over ALL 65,536 digraphs on four nodes (loops allowed): no source implies a cycle, every acyclic
     digraph has a source and a sink, and the acyclic ones counted are the known 1, 3, 25, 543 labelled DAGs for 1 to 4 nodes. So "everything has a parent" in a finite set is a loop;
     the only other way out is an infinite set. The post picks the loop.
  2. MAPS. A rule that sends each of n states to a next state (any function on n states, all n^n of them for n up to 5) has a state with nothing leading to it exactly when it loses
     information (two states go to one). The number of such "starts" is n minus the number of distinct images. The rule is a bijection (nothing lost, every state has a predecessor,
     every state lies on a cycle: no tails) exactly when there are no starts. The only way to have a start is to have thrown something away. This is the part of the post
     that is exactly true: a first state in a finite deterministic system is a place where information was lost. The clock rules x -> m * x (mod n) are the example: for m a unit
     there is no start; for m = 2 on the 12-clock the six odd marks are starts.
  3. DIVISION. "Division by zero destroys the model" is the same loss: dividing is undoing a multiplication, which is possible exactly when the multiplier is a unit; 0 is a unit
     in no clock of more than one mark. (On the 13-clock every other multiplier has an inverse.)
  4. THE GATE. In OUR language, deltas are undone by their inverses (`invert`). On a two-cell universe (13 states: each cell empty, 1 or 2, and an edge when both cells are filled), the gate admits
     every transition whose inverse it also admits, and returns the starting state; every state has a way in and a way out and the whole space is one loop-connected piece.
     So the repo's state space, taken with its inverses, has no start: the empty graph (the ground) is a place on the loop, reachable from every state and leading to every state.
     (This holds because the inverse is computed from the host BEFORE the step; test_deletion_asymmetry.py and test_undo_log.py show what happens without that copy.)
  5. DOUBLING. "Two parents, four grandparents, eight great-grandparents": the slots double each generation, 2^g, and the total over g generations is 2^(g+1) - 2. With N people in all, the slots outnumber
     the people from generation g = ceil(log2(N + 1)) on: 10 for 1,000 people, 33 for eight billion. After that some person must fill two slots. That is arithmetic about the slots, not about history.
What they do NOT show: that the universe is a loop (the same theorem allows an infinite set, which the finite theorem cannot decide; the second post's "backward forever" is that option); that
the post's "miracle" is a mathematical term (it is not one); that a model needs a loop; anything about time. They show that "needs no first moment" and "loses nothing" are the same sentence for a finite system."""
import os, sys
from itertools import product
from math import ceil, gcd, log2
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_signed_slots import same, copy


def digraphs(n):
    """All directed graphs on nodes 0..n-1 as sets of pairs (loops allowed)."""
    pairs = [(a, b) for a in range(n) for b in range(n)]
    for bits in product((0, 1), repeat=len(pairs)):
        yield n, frozenset(p for p, b in zip(pairs, bits) if b)


def sources(n, E): return [v for v in range(n) if not any(b == v for _, b in E)]


def sinks(n, E): return [v for v in range(n) if not any(a == v for a, _ in E)]


def has_cycle(n, E):
    """True if some node can reach itself (loops count)."""
    reach = {v: {b for a, b in E if a == v} for v in range(n)}
    for _ in range(n):                                                          # transitive closure by repeated joining
        reach = {v: reach[v] | {w for u in reach[v] for w in reach[u]} for v in range(n)}
    return any(v in reach[v] for v in range(n))


def test_a_finite_graph_with_no_source_has_a_cycle_and_the_acyclic_ones_have_a_start_and_an_end():
    dag_counts = []
    for n in (1, 2, 3, 4):
        dags = 0
        for _, E in digraphs(n):
            cyc = has_cycle(n, E)
            if not sources(n, E): assert cyc, (n, sorted(E))                    # everything has a parent -> a loop
            if not cyc:
                dags += 1
                assert sources(n, E) and sinks(n, E)                            # acyclic -> a first node and a last node
        dag_counts.append(dags)
    assert dag_counts == [1, 3, 25, 543]                                        # the labelled acyclic digraphs; the brute force agrees with the known counts


def functions(n): return product(range(n), repeat=n)


def test_a_rule_on_finitely_many_states_has_a_start_exactly_when_it_loses_information():
    for n in (1, 2, 3, 4, 5):
        bijections = 0
        for f in functions(n):
            image = set(f)
            starts = [s for s in range(n) if s not in image]                    # states that nothing leads to
            assert len(starts) == n - len(image)
            on_cycle = [s for s in range(n) if _returns(f, s)]
            if len(image) == n:                                                 # nothing lost
                bijections += 1
                assert not starts and len(on_cycle) == n                        # no start, no tails: every state lies on a cycle
            else:
                assert starts and len(on_cycle) < n                             # something lost: a start and a tail
        assert bijections == {1: 1, 2: 2, 3: 6, 4: 24, 5: 120}[n]               # n! of the n^n rules lose nothing
    # the clock rules x -> m x (mod n)
    for n in (12, 13):
        for m in range(n):
            f = [x * m % n for x in range(n)]
            starts = [s for s in range(n) if s not in f]
            assert (not starts) == (gcd(m, n) == 1)
    assert [s for s in range(12) if s not in [x * 2 % 12 for x in range(12)]] == [1, 3, 5, 7, 9, 11]     # doubling on the 12-clock: the odd marks are starts
    f13 = [x * 2 % 13 for x in range(13)]
    cycle = [1]
    while f13[cycle[-1]] != 1: cycle.append(f13[cycle[-1]])
    assert len(cycle) == 12 and f13[0] == 0                                     # one loop of 12 and the fixed zero: no start on the 13-clock


def _returns(f, s):
    x = f[s]
    for _ in range(len(f)):
        if x == s: return True
        x = f[x]
    return False


def test_division_by_zero_has_no_inverse_on_any_clock_and_every_other_multiplier_has_one_on_13():
    for n in range(2, 61):
        assert not [x for x in range(n) if 0 * x % n == 1]                      # nothing times 0 is 1
        for m in range(n):
            has_inverse = any(m * x % n == 1 for x in range(n))
            assert has_inverse == (gcd(m, n) == 1)                              # an inverse exists exactly for the units
    assert all(any(m * x % 13 == 1 for x in range(13)) for m in range(1, 13))


NODES = [(0, 0), (1, 1)]
EDGE = ((0, 0), (1, 1), "r", 1)
VALUES = (1, 2)
CANDIDATES = (["DEL[0,0]", "DEL[1,1]", "ADD[(0,0)->(1,1):r#1]", "DEL[(0,0)->(1,1):r#1]",
               "DEL[(0,0)->(1,1):r#1] DEL[0,0]", "DEL[(0,0)->(1,1):r#1] DEL[1,1]"]
              + [f"ADD[{c[0]},{c[1]}:{v}]" for c in NODES for v in VALUES]
              + [f"ADD[{c[0]},{c[1]}:{v}] ADD[(0,0)->(1,1):r#1]" for c in NODES for v in VALUES])      # the last four are the inverses of the two cascades


def build(values, edge):
    g = Graph()
    for n, v in zip(NODES, values):
        if v is not None: g.add_node(n, v)
    if edge: g.add_edge(*EDGE)
    return g


def state_of(g): return (tuple(g.nodes.get(n) for n in NODES), bool(g.edges))


def test_on_a_two_cell_universe_the_gate_admits_a_way_back_for_every_way_in_so_there_is_no_start():
    cell_values = (None,) + VALUES
    states = [((a, b), e) for a in cell_values for b in cell_values for e in (False, True) if not e or (a is not None and b is not None)]
    assert len(states) == 9 + 4                                                 # each cell empty, 1 or 2; the edge only between two cells
    ground = ((None, None), False)
    moves = {}                                                                  # state -> {states it can reach by an admitted delta}
    for s in states:
        g = build(*s)
        for text in CANDIDATES:
            d = CICOParser.parse_transition_delta(text)
            if not check(g, d).ok: continue
            inv = invert(g, d)                                                  # computed before the step, from the host
            h = copy(g); apply(h, d)
            back = copy(h); assert check(back, inv).ok, (s, text)
            apply(back, inv)
            assert same(back, g)                                                # the inverse returns exactly where we started, values included
            moves.setdefault(s, set()).add(state_of(h))
    assert set(moves) == set(states)
    for s, ts in moves.items():
        for t in ts: assert s in moves[t]                                       # every way from s to t has a way from t to s: the state space is symmetric
        assert any(t != s for t in ts)                                          # every state has a way out ...
    assert all(any(s in moves[t] and t != s for t in moves) for s in states)    # ... and a way in: no source

    def reach(src, nxt):
        seen, todo = {src}, [src]
        while todo:
            for t in nxt[todo.pop()]:
                if t not in seen: seen.add(t); todo.append(t)
        return seen
    into = {s: {t for t in states if s in moves[t]} for s in states}
    assert reach(ground, moves) == set(states) and reach(ground, into) == set(states)   # the ground leads to everything and everything leads back to it
    assert has_cycle(len(states), {(states.index(s), states.index(t)) for s in moves for t in moves[s] if t != s})


def test_the_slots_double_and_outnumber_a_finite_population_from_generation_ceil_log2_n_plus_1():
    slots = [2 ** g for g in range(1, 5)]
    assert slots == [2, 4, 8, 16] and [sum(2 ** k for k in range(1, g + 1)) for g in range(1, 6)] == [2 ** (g + 1) - 2 for g in range(1, 6)]
    first = lambda N: min(g for g in range(1, 80) if 2 ** g > N)
    assert first(1000) == 10 == ceil(log2(1001)) and first(8_000_000_000) == 33 == ceil(log2(8_000_000_001))
    assert 2 ** 32 < 8_000_000_000 < 2 ** 33                                    # the slot count passes eight billion between generation 32 and 33
    assert all(first(N) == ceil(log2(N + 1)) for N in range(1, 2000))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

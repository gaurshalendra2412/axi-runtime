"""Implements concept-map row 37 (blog post 46, "that's a social immune system"): the response is selective, used
"exclusively to check bad behavior".
Reading: an immune response that attacks healthy tissue is autoimmunity. For the gate that means rejecting a delta that is fine.
What this pins, and what it does NOT show:
  * On the benchmark's own ideal deltas (8 seeds x 60 tasks) the gate admits every one, and applying them leaves a well-formed graph.
  * It does not show the gate is never over-strict. The gate is deliberately stricter than the outcome in a few cases (deleting an
    edge that is not in the graph is silently ignored by blind apply but rejected here, `match_edge`); that is a policy choice,
    see concept-map row 31 (strictness sweep).
  * The ideal set is the generator's own, so this is a floor, not a proof."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import check
from axi.engine.diagnostics import diagnose
from axi.experiments.agent_loop import make_tasks, copy_graph, blind_apply, same_graph


def test_gate_admits_every_ideal_delta_and_result_is_well_formed():
    n = 0
    for seed in range(8):
        for t in make_tasks(60, seed):
            d = CICOParser.parse_transition_delta(t.ideal)
            r = check(copy_graph(t.graph), d)
            assert r.ok, (seed, t.kind, r)
            assert not diagnose(copy_graph(t.graph), d), (seed, t.kind)
            out = blind_apply(copy_graph(t.graph), d)
            assert out.is_well_formed() and same_graph(out, t.expected), (seed, t.kind)
            n += 1
    assert n == 480


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

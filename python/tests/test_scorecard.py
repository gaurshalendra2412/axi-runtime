import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from test_agent_loop import ScriptedLLM
from axi.experiments.agent_loop import make_tasks, run_experiment
from axi.experiments import scorecard as sc


def _rows(seed=0, n=40):
    return run_experiment(ScriptedLLM(seed=seed), make_tasks(n, seed), hint_rules=False)["rows"]


def test_scorecard_matches_a_hand_count_on_scripted_rows():
    rows = _rows(); s = sc.scorecard(rows, "delete_dep"); dd = [r for r in rows if r["kind"] == "delete_dep"]
    assert s["n"] == len(dd) > 0
    assert s["rule_compliance"]["blind_apply_corrupted"] == sum(r["M1_constrained"]["corrupted"] for r in dd) / len(dd) == 1.0
    assert s["rule_compliance"]["gate_corrupted"] == 0.0                       # the gate never lets corruption through
    assert s["reality_perception"]["proposals_with_dangling"] == 1.0           # the naive stand-in always forgets the edges
    assert s["reality_perception"]["proposals_with_phantom_refs"] == 0.0       # ... and never invents any


def test_ego_defense_separates_feedback_variants():
    s = sc.scorecard(_rows(n=60), "delete_dep")["ego_defense"]
    assert s["structured_imperative"]["repeat_same_proposal"] == 0.0           # exact commands are always followed (scripted)
    assert s["structured_imperative"]["still_rejected"] == 0.0
    assert s["prose_plain"]["repeat_same_proposal"] > s["prose_detailed"]["repeat_same_proposal"] > 0.0
    assert s["prose_plain"]["still_rejected"] > s["structured_imperative"]["still_rejected"]


def test_phantom_edges_are_counted():
    rows = _rows(n=10)
    for r in rows:                                                               # inject a phantom-edge obstruction on delete_dep rows
        if r["kind"] == "delete_dep": r["M2_gate"]["obstructions"] = ["match_edge", "dangling"]
    s = sc.scorecard(rows, "delete_dep")
    assert s["reality_perception"]["proposals_with_phantom_refs"] == 1.0


def test_cli_roundtrip_and_all_kinds():
    rows = _rows(n=20)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump({"rows": rows}, f); p = f.name
    s_all = sc.scorecard(json.load(open(p))["rows"], None)
    assert s_all["n"] == 20
    sc.print_scorecard("t", s_all)                                              # prints without error
    os.unlink(p)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

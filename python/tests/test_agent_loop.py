import random, re
from axi.experiments.agent_loop import (make_tasks, run_experiment, run_task, summarize, blind_apply, lenient_parse, strict_parse,
                                        same_graph, copy_graph, feedback_text, MODES)
from axi.engine.cico_parser import CICOParser
from axi.engine.diagnostics import diagnose

EDGE_RE = re.compile(r"\((\d+),(\d+)\)->\((\d+),(\d+)\):(\w+)#(-?\d+)")


class ScriptedLLM:
    """Stand-in for a real model. Naive about relational rules (like a typical small model)."""
    def __init__(self, seed=0, chatty=0.4, p_plain=0.3, p_detailed=0.85):
        self.rng = random.Random(seed); self.chatty, self.p_plain, self.p_detailed = chatty, p_plain, p_detailed

    def _naive(self, instr):
        m = re.search(r"Delete the service at \((\d+),(\d+)\)", instr)
        if m: return f"DEL[{m[1]},{m[2]}]"
        m = re.search(r"Add a service at \((\d+),(\d+)\) with value (\d+)", instr)
        if m: return f"ADD[{m[1]},{m[2]}:{m[3]}]"
        m = re.search(r"edge from \((\d+),(\d+)\) to \((\d+),(\d+)\) with relation dep and weight (\d+)", instr)
        return f"ADD[({m[1]},{m[2]})->({m[3]},{m[4]}):dep#{m[5]}]"

    def generate(self, messages, constrained):
        first_user = messages[1]["content"]; last = messages[-1]["content"]
        if len(messages) > 3:                                     # retry turn
            node = re.search(r"Delete the service at \((\d+),(\d+)\)", first_user)
            n = (int(node[1]), int(node[2]))
            edges = [e for e in EDGE_RE.findall(first_user.split("EDGES:")[1]) if (int(e[0]), int(e[1])) == n or (int(e[2]), int(e[3])) == n]
            full = f"DEL[{n[0]},{n[1]}] " + " ".join(f"DEL[({a},{b})->({c},{d}):{r}#{w}]" for a, b, c, d, r, w in edges)
            if "OBSTRUCTION" in last: return full, 20, 0.0
            if "weight" in last and self.rng.random() < self.p_detailed: return full, 20, 0.0
            if "weight" not in last and self.rng.random() < self.p_plain: return full, 20, 0.0
            return self._naive(first_user.split("Request:")[1]), 8, 0.0
        txt = self._naive(first_user.split("Request:")[1])
        if not constrained and self.rng.random() < self.chatty: return f"Sure! Here is the delta: {txt} Let me know if you need more.", 30, 0.0
        return txt, 8, 0.0


def test_ideal_deltas_score_as_success_and_blind_naive_corrupts():
    for t in make_tasks(40, 1):
        d = CICOParser.parse_transition_delta(t.ideal); assert same_graph(blind_apply(t.graph, d), t.expected)
        naive = blind_apply(t.graph, CICOParser.parse_transition_delta(ScriptedLLM()._naive(t.instruction)))
        assert (not naive.is_well_formed()) == (t.kind == "delete_dep")


def test_lenient_vs_strict_parse():
    s = "Sure! DEL[1,2] done"
    assert strict_parse(s) is None and lenient_parse(s) is not None and strict_parse("DEL[1,2]") is not None


def test_pipeline_metrics_on_scripted_llm():
    tasks = make_tasks(100, 3); res = run_experiment(ScriptedLLM(0), tasks); S = res["summary"]; a, dd = S["all"], S["delete_dep"]
    # constrained decoding alone still corrupts the graph on every delete-with-dependents task
    assert dd["M1_constrained"]["corrupted"] == 1.0 and dd["M1_constrained"]["strict_parse"] == 1.0
    assert a["M0_unconstrained"]["strict_parse"] < 0.9                      # chatty output fails strict parsing
    # the gate never lets a corrupted graph through, in any gated mode
    for m in MODES[2:]: assert a[m]["corrupted"] == 0.0, m
    # structured retry and auto-repair fully recover; plain prose recovers less
    assert dd["M3_retry_structured"]["success"] == 1.0 and dd["M5_auto_repair"]["success"] == 1.0
    assert dd["M3_retry_prose_plain"]["success"] < dd["M3_retry_prose_detailed"]["success"] < 1.0 or dd["M3_retry_prose_plain"]["success"] < 1.0
    assert dd["M2_gate"]["success"] == 0.0                                  # rejection alone completes nothing
    assert a["M5_auto_repair"]["success"] == 1.0
    # non-trap tasks are unaffected by the machinery
    for r in res["rows"]:
        if r["kind"] != "delete_dep": assert r["M2_gate"]["success"] and not r["M2_gate"]["rejected"]


def test_feedback_variants_carry_the_same_edges():
    t = next(x for x in make_tasks(10, 2) if x.kind == "delete_dep"); d = CICOParser.parse_transition_delta("DEL[" + t.ideal.split("DEL[")[1].split("]")[0] + "]")
    obs = diagnose(t.graph, d); assert obs and obs[0].kind == "dangling"
    st, pd = feedback_text("structured", obs), feedback_text("prose_detailed", obs)
    for (u, v, r, w) in obs[0].edges:
        assert f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{w}]" in st and f"({u[0]},{u[1]})->({v[0]},{v[1]}) {r} weight {w}" in pd
    assert "OBSTRUCTION" not in feedback_text("prose_plain", obs)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

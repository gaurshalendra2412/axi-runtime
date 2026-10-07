import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_agent_loop import ScriptedLLM
from axi.experiments import cascade as C


def test_blind_corruption_accumulates_and_every_gated_policy_stays_clean():
    r = C.run_cascade(ScriptedLLM(0, chatty=0.0), episodes=12, steps=8, seed=1); S = r["summary"]
    blind = [x["corrupted"] for x in S["blind"]["per_step"]]
    assert blind[-1] > blind[0] and blind == sorted(blind)                      # scripted: never repairs, so corruption only grows
    assert S["blind"]["episodes_ever_corrupted"] > 0.8 and S["blind"]["per_step"][-1]["dangling"] > S["blind"]["per_step"][0]["dangling"]
    for p in ("gate", "repair_m5", "repair_m6"):
        assert S[p]["episodes_ever_corrupted"] == 0.0 and all(x["corrupted"] == 0 for x in S[p]["per_step"]), p
    assert S["gate"]["success_overall"] < S["repair_m5"]["success_overall"] <= 1.0   # rejection alone leaves requests undone; repair completes them


def test_tasks_are_generated_from_the_live_graph():
    import random
    g = C.random_graph(random.Random(2)); rng = random.Random(0)
    for kind in C.KINDS:
        t = C.pick_task(rng, g, kind)
        if t is not None: assert C.check(g, C.CICOParser.parse_transition_delta(t.ideal)).ok


def test_unparseable_output_changes_nothing():
    class Bad:
        def generate(self, m, constrained): return "hello", 1, 0.0
    log = C.run_episode(Bad(), "blind", 0, 4); assert log and all(not r["parsed"] and not r["corrupted"] for r in log)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

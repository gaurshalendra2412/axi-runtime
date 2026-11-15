"""AXI quickstart: check the published numbers of the hard test in about a minute. No API key, no model call, no GPU.

Run it from the python folder of a copy of the repo:   python quickstart_replay.py

What it does. The hard test (docs/MILLION_TOKEN_TEST_HARD.md) asked Gemini 3.8 Flash to change a graph of services 4,000 times
(add or delete a service, a dependency, a connection). The model's real replies were saved in docs/results/. This script reads
those saved replies and recomputes, with the repo's own code (the request stream, the gate, the repair, the scope filter),
the held state step by step. Nothing is read from the saved flags: every count below is recomputed here, and the replay
stops with an error if one step differs from what was saved.

Three parts:
  1. the AXI run, 4,000 requests: on how many steps was the held state exactly the right one, and was it ever broken
  2. the model's own replies: how many were exactly the ideal reply, and what the 46 others had in common
  3. the first 400 requests, four ways of using the replies (applied as written, gate only, gate + repair, the AXI run),
     and what a step with the right state cost in each

AXI is a host-side layer: the host holds the exact state and checks every proposal before it touches the state. The model is unchanged.
Architecture by Rajnish Choubey; tests built and run by Shalendra Gaur.
"""
import sys
import time

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, HERE)
from axi.experiments import million as M                      # the repo's own replay machinery
from axi.experiments import million_hard_result as R          # the repo's own recomputation of the page's Result
from axi.experiments.agent_loop import copy_graph, strict_parse

STEPS = 400                                                   # the contrast covers the first 400 requests


def recompute(rows, meta):
    """Replay the saved replies of one arm: same request stream, same policy, held state rebuilt from scratch."""
    stream = M.make_stream(meta["stream"], meta["seed"])
    live = copy_graph(stream.start)
    exact = corrupted = 0
    for rec in rows:
        st, ob = stream.next()
        live, f = M.step_fields(meta["policy"], live, st, ob, stream.live, strict_parse(rec["text"]), meta.get("degree_threshold", 4))
        exact += bool(f["exact"])
        corrupted += bool(f["corrupted"])
    return exact, corrupted


def main():
    t0 = time.time()
    run1, con = R.load(), R.load_contrast()
    rows, meta = run1["rows"], run1["meta"]
    n = len(rows)
    ok = []

    # 1. the AXI run, every step recomputed (strict: one differing flag or re-ask raises an error)
    print(f"1. The AXI run: Gemini 3.8 Flash, {n:,} requests, one job (no model is called, the saved replies are replayed)")
    _, _, bad = M.replay(rows, meta["seed"], meta["policy"], strict=True, stream=meta["stream"], reasks=meta["reasks"], degree_threshold=meta["degree_threshold"])
    exact, corrupted = recompute(rows, meta)
    a = R.analyse(run1)
    print(f"   held state exactly the right one:  {exact:,} of {n:,} steps")
    print(f"   held state broken at any step:     {corrupted} of {n:,} steps")
    print(f"   every saved flag of every step reproduced: {'yes' if not bad else 'NO'}")
    print(f"   tokens sent (prompt + reply + thinking): {a['tokens']['total']:,}")
    ok += [exact == n == 4000, corrupted == 0, not bad, a["tokens"]["total"] == 3661209]

    # 2. the model's replies against the ideal reply of each request
    same = sum(v["same_text"] for v in a["kinds"].values())
    order = sum(v["other_order"] for v in a["kinds"].values())
    other = sum(v["other_items"] for v in a["kinds"].values())
    right = same + order
    print(f"\n2. The model's own replies, each compared with the ideal reply of its request")
    print(f"   exactly the ideal reply:           {right:,} of {n:,} ({100 * right / n:.2f}%)  ({same:,} the same text, {order} the same items in another order)")
    print(f"   something else:                    {other} ({100 * other / n:.2f}%)  in all {a['only_left_out']} the reply is the ideal one with items left out (none added, none changed)")
    print(f"   what a bare 'apply as written' would have broken, one step from the right state: {a['counterfactual']['blind'][0]}; with the repair: {a['counterfactual']['scope_m7'][0]}")
    ok += [right == 3954, other == 46, a["only_left_out"] == 46, a["counterfactual"]["blind"][0] == 46, a["counterfactual"]["scope_m7"][0] == 0]

    # 3. the first 400 requests, four ways
    print(f"\n3. The first {STEPS} requests of the same job, four ways of using the model's replies")
    arms = [("applied as written", con["blind"]["rows"], con["blind"]["meta"], 85, 223),
            ("gate only", con["gate"]["rows"], con["gate"]["meta"], 84, 0),
            ("gate + repair", con["repair_m6"]["rows"], con["repair_m6"]["meta"], 400, 0),
            ("AXI run", rows[:STEPS], meta, 400, 0)]
    print(f"   {'way of using the replies':<22}{'steps with the right state':>28}{'steps with a broken state':>28}")
    for name, arm_rows, arm_meta, want_exact, want_corrupted in arms:
        assert len(arm_rows) == STEPS, name
        e, c = recompute(arm_rows, arm_meta)
        print(f"   {name:<22}{f'{e} of {STEPS} ({100 * e / STEPS:.1f}%)':>28}{f'{c} of {STEPS}':>28}")
        ok.append((e, c) == (want_exact, want_corrupted))
    cost = R.analyse_cost(run1, con)
    print(f"\n   what a step with the right state cost (every call counted, at ${R.PRICE_IN} in / ${R.PRICE_OUT} out per million tokens, thinking counted as output):")
    for key, name in (("scope_m7", "AXI run"), ("blind", "applied as written"), ("gate", "gate only")):
        c = cost[key]
        print(f"   {name:<22}{c['exact']:>4} steps right   ${c['dollars']:.2f} spent   ${c['dollars_per_exact']:.5f} per step with the right state   {c['times_framework']:.1f}x")
    ok += [round(cost["blind"]["times_framework"], 1) == 4.5, round(cost["gate"]["times_framework"], 1) == 4.9]

    same_as_published = all(ok)
    print(f"\nrecomputed in {time.time() - t0:.1f} s from the saved replies.")
    print("matches the published figures (4,000 of 4,000; 3,954 of 4,000; 85 / 84 / 400 / 400 of 400; 4.5x):", "yes" if same_as_published else "NO, the files or the code differ from the published ones")
    print("\nHow to read it: the first two rows of part 3 are separate live runs of the same 400 requests, where the model saw its own held state,"
          "\nso their replies differ from the AXI run's after the first slips. The last two rows use the AXI run's saved replies."
          "\nOne model, one job, one sampled run. Host-side layer, model unchanged. The cost rows are arithmetic added after the results, not a pre-registered line."
          "\nPre-registered page: docs/MILLION_TOKEN_TEST_HARD.md")
    return 0 if same_as_published else 1


if __name__ == "__main__":
    sys.exit(main())

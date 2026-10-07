"""Scorecard: reduce the raw transcripts of an agent-loop run (the `rows` in the results JSON) to a handful of numbers.

Three dimensions, named after the three "boundary challenges" of the Delete Test that the Cinderella Project text lists
(rule compliance, reality perception, ego-defense; the other two are not named in the text we have). The definitions are
mine and operational - they are NOT Rajnish's definitions:

  rule_compliance     blind apply of the (constrained) proposal leaves a corrupted graph          (M1_constrained.corrupted)
  reality_perception  the proposal refers to things that are not in the graph: any match_node / match_edge / match_weight
                      obstruction (phantom nodes, edges, weights)                                  (M2_gate.obstructions)
  ego_defense         after feedback, the model repeats its rejected proposal verbatim, or is still rejected
                      (per retry variant)                                                          (M3_retry_*.text / .rejected)
plus silent-vs-loud failure for M0 / M1: unparseable output (loud, graph untouched) vs a parseable delta that corrupts (silent).

Usage:  python -m axi.experiments.scorecard /content/seed1_nohint.json /content/seed1_hint.json
"""
import json, sys

PHANTOM = ("match_node", "match_edge", "match_weight")


def _rate(n, d):
    return None if d == 0 else n / d


def scorecard(rows, kind="delete_dep"):
    rs = [r for r in rows if kind is None or r["kind"] == kind]
    out = {"n": len(rs)}
    m1 = [r["M1_constrained"] for r in rs]
    out["rule_compliance"] = {"blind_apply_corrupted": _rate(sum(x["corrupted"] for x in m1), len(m1)),
                              "gate_corrupted": _rate(sum(r["M2_gate"]["corrupted"] for r in rs), len(rs))}
    parsed = [r for r in rs if r["M1_constrained"]["strict_parse"]]
    out["reality_perception"] = {"proposals_with_phantom_refs": _rate(sum(any(o in PHANTOM for o in r["M2_gate"]["obstructions"]) for r in parsed), len(parsed)),
                                 "proposals_with_dangling": _rate(sum("dangling" in r["M2_gate"]["obstructions"] for r in parsed), len(parsed))}
    ego = {}
    for key in sorted({k for r in rs for k in r if k.startswith("M3_retry_")}):
        retried = [r for r in rs if r[key].get("retried")]
        rep = sum(r[key]["text"].strip() == r["M1_constrained"]["text"].strip() for r in retried)
        still = sum(r[key]["rejected"] for r in retried)
        ego[key[len("M3_retry_"):]] = {"retried": len(retried), "repeat_same_proposal": _rate(rep, len(retried)), "still_rejected": _rate(still, len(retried)),
                                       "success_overall": _rate(sum(r[key]["success"] for r in rs), len(rs))}
    out["ego_defense"] = ego
    m0 = [r["M0_unconstrained"] for r in rs]
    out["loud_vs_silent"] = {
        "M0_loud(no delta extractable)": _rate(sum(not x["lenient_parse"] for x in m0), len(m0)),
        "M0_silent(corrupted)": _rate(sum(x["corrupted"] for x in m0), len(m0)),
        "M1_loud(unparseable)": _rate(sum(not x["strict_parse"] for x in m1), len(m1)),
        "M1_silent(corrupted)": _rate(sum(x["corrupted"] for x in m1), len(m1))}
    return out


def _fmt(v):
    return "  -  " if v is None else f"{v * 100:5.1f}%"


def print_scorecard(name, sc):
    print(f"\n===== {name}  (kind filter n={sc['n']})")
    rc, rp = sc["rule_compliance"], sc["reality_perception"]
    print(f"rule compliance   : blind apply corrupts {_fmt(rc['blind_apply_corrupted'])}   after the gate {_fmt(rc['gate_corrupted'])}")
    print(f"reality perception: proposals that name things not in the graph {_fmt(rp['proposals_with_phantom_refs'])}   leave dangling edges {_fmt(rp['proposals_with_dangling'])}")
    print("ego-defense (after feedback)      retried   repeats same   still rejected   task success")
    for v, e in sc["ego_defense"].items():
        print(f"  {v:28s} {e['retried']:5d}     {_fmt(e['repeat_same_proposal'])}        {_fmt(e['still_rejected'])}          {_fmt(e['success_overall'])}")
    print("loud vs silent failure: " + "   ".join(f"{k} {_fmt(v)}" for k, v in sc["loud_vs_silent"].items()))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = json.load(open(path))["rows"]
        for kind in ("delete_dep", None):
            print_scorecard(f"{path} [{kind or 'all kinds'}]", scorecard(rows, kind))

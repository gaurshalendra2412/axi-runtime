"""Reading rules of the API test that are new (docs/API_TEST_GEMINI.md; written BEFORE any API number exists).

Part A (the full test on one model reached through an API) reads with the unchanged lines of `bigger_model.py` (`verdict_big_single`, `verdict_big_long`) PLUS one new [format] line below: no grammar mask
exists on an API, so how often the reply does not parse is itself a result. Part B (the 20 requests that decide the question, on a bigger model whose free quota is 20 calls a day) reads with
`verdict_pivotal`. All thresholds are integers, so no floating-point slip can move a verdict."""
from typing import List

from axi.experiments.agent_loop import strict_parse


def unparsed_single(rows):
    """(replies that do not parse, replies) of a one-request-at-a-time file."""
    return sum(1 for r in rows if strict_parse(r["text"]) is None), len(rows)


def unparsed_long(rows):
    """(replies that do not parse, replies) of a long-run file, counting each model reply once: a step whose reply was shared with an earlier policy (`cached`) is not counted again."""
    bad = tot = 0
    for ep in rows:
        for steps in ep["policies"].values():
            for s in steps:
                if s.get("cached"): continue
                tot += 1; bad += 0 if s["parsed"] else 1
    return bad, tot


def verdict_api_format(single_rows=None, long_rows=None) -> List[tuple]:
    """single_rows, long_rows: {hint_rules(bool): rows}. One line: at most 5 percent of all the model's replies fail to parse."""
    bad = tot = 0
    for rs in (single_rows or {}).values(): b, t = unparsed_single(rs); bad += b; tot += t
    for rs in (long_rows or {}).values(): b, t = unparsed_long(rs); bad += b; tot += t
    return [(f"[format] at most 5 percent of the model's replies do not parse ({bad} of {tot})", tot > 0 and 20 * bad <= tot)]


def verdict_pivotal(summaries) -> List[tuple]:
    """summaries: {hint_rules(bool): summarize_scope(rows, tasks)} for the delete_dep requests only (20 of them)."""
    out = []; regs = sorted(summaries)
    out.append(("[guard] gate, repair_m6 and scope_m7 corrupt no graph in any regime", all(summaries[h]["all"][p]["corrupted"] == 0 for h in regs for p in ("gate", "repair_m6", "scope_m7"))))
    if False in summaries:
        s = summaries[False]["all"]; n = s["n"]
        out.append((f"[need] no rule: blind apply is right on at most half of the {n} delete_dep requests", 2 * s["blind"]["success"] <= n))
        out.append((f"[guard] no rule: repair_m6 is right on at least 90 percent of the {n}", 10 * s["repair_m6"]["success"] >= 9 * n))
    if True in summaries:
        s = summaries[True]["all"]; n = s["n"]
        out.append((f"[guard] rule stated: scope_m7 is right on at least 90 percent of the {n}", 10 * s["scope_m7"]["success"] >= 9 * n))
    return out


def smoke_report(rows):
    """Stop-early test for the first requests of an API run: (ok, message). At least two thirds of the replies must parse; a model whose replies are mostly not a delta (wrong id, blocked, empty) is stopped
    before it uses up the day's quota."""
    bad, n = unparsed_single(rows); ok = n > 0 and 3 * (n - bad) >= 2 * n
    msg = (f"SMOKE OK: {n - bad} of {n} replies parse. Go on to the next cell." if ok else
           f"STOP: only {n - bad} of {n} replies parse. Do not run the next cells: send me this printout and the lines above it.")
    return ok, msg

"""Q = O / M from saved result files, no GPU. The idea is from the pasted summaries of Rajnish Choubey's sessions (concept-map item 110): O counts only the steps after which the held state is exact,
M is what the work cost, and the summaries say that M can come down towards a floor and not to zero. This module reads the cells already saved in `docs/results/` and computes, per cell:

  O              the steps after which the state the job held equals the oracle's (the `exact` flag of the log);
  M (time)       the wall seconds of the cell, every round of the engine included (a retry round too);
  M (tokens)     the prompt tokens the engine had to compute (prompt tokens minus cached tokens, every request) plus the tokens it wrote.

and shows the two Q's SEPARATELY: exact steps per GPU hour (3,600 x O / wall seconds) and exact steps per 1,000 tokens (1,000 x O / M tokens). There is no third number that mixes seconds and tokens: that
would need a weight (the summaries write M = time + lambda x tokens) and no weight has been measured, so none is invented here. A Q is only ever stated with the exactness beside it and with the setup
it was measured on (the allowed sentence: "N exact steps per GPU hour against M, at these exactness levels, on this setup"). Never a multiple of capacity, a count of GPUs, dollars.

What this does not show: a floor of M (nothing here says how low M can go), energy (never measured), anything about a cell that was not run, anything about another model, GPU or engine."""
import json, os
from typing import Dict, List, Optional, Tuple

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "results")
FILES: List[Tuple[str, str]] = [("first run, no grammar", "axi_provider_side.json"), ("second run, uncapped grammar", "axi_provider_constrained.json"),
                                ("third run, capped grammar", "axi_provider_bounded.json"), ("fourth run, one retry", "axi_provider_retry.json")]


def cell_id(c: dict) -> str:
    return c["label"] if c.get("label") else f"{c['arm']}{c['K']}"


def q_row(c: dict) -> dict:
    steps = [x for s in c["sessions"] for x in s["log"]]; n = len(steps); O = sum(1 for x in steps if x["exact"]); wall = c["wall_seconds"]; rounds = c["rounds"]
    have = bool(rounds) and all(r.get("cached_tokens") is not None for r in rounds)
    prompt = sum(r["prompt_tokens"] for r in rounds); new = sum(r["new_tokens"] for r in rounds); computed = (prompt - sum(r["cached_tokens"] for r in rounds)) if have else None
    m_tok = (computed + new) if have else None
    return dict(cell=cell_id(c), steps=n, exact_steps=O, exact_share=(O / n if n else None), wall_seconds=wall, computed_prompt_tokens=computed, new_tokens=new, m_tokens=m_tok,
                exact_per_hour=(3600.0 * O / wall if wall else None), exact_per_1000_tokens=(1000.0 * O / m_tok if m_tok else None), seconds_per_exact_step=(wall / O if O else None),
                tokens_per_exact_step=(m_tok / O if m_tok and O else None))


def q_rows(path: str) -> List[dict]:
    return [q_row(c) for c in json.load(open(path))["cells"]]


def all_rows(results_dir: Optional[str] = None) -> Dict[str, List[dict]]:
    d = results_dir or RESULTS
    return {title: q_rows(os.path.join(d, name)) for title, name in FILES if os.path.exists(os.path.join(d, name))}


def render_table(rows: Dict[str, List[dict]]) -> str:
    """The table as it stands on docs/Q_FROM_SAVED_FILES.md (markdown), one block per run."""
    out = []
    for title, rs in rows.items():
        out.append(f"**{title}**\n\n| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |\n|---|---|---|---|---|---|---|---|")
        for r in rs:
            out.append(f"| {r['cell']} | {r['steps']:,} | {r['exact_steps']:,} | {100 * r['exact_share']:.1f} % | {r['wall_seconds']:.1f} | {r['exact_per_hour']:,.0f} | "
                       + (f"{r['m_tokens']:,}" if r["m_tokens"] is not None else "-") + " | " + (f"{r['exact_per_1000_tokens']:.1f}" if r["exact_per_1000_tokens"] is not None else "-") + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render_table(all_rows()))

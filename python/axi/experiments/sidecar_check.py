"""Drives the HTTP sidecar (axi/sidecar.py) with the repository's own request stream and counts, step by step, whether the state the host holds equals the oracle's state.

AXI is a layer on the host side; the model and the engine behind the sidecar are unchanged. The requests come from `million.make_stream` (made on the oracle's graph, so they are the
same whatever the model answers); after every step the state the sidecar returns is compared with the oracle's. This is the same measure as the held-state columns of the repository's
tests, taken through the sidecar and an OpenAI-compatible server instead of the engine's own Python API.

`run(sidecar_url, steps, seed)` returns a dict: steps, steps with the exact state, steps whose reply the strict parser read, corrupted steps, mean step seconds and mean prompt tokens over the
first and the last tenth of the steps (prompt tokens only when the upstream reports `usage`). It is a check of the plumbing and a measurement of whatever model sits behind it;
it proves nothing about other engines, models or kinds of state."""
import json, time, urllib.request
from typing import Optional

from axi.experiments.agent_loop import same_graph
from axi.experiments.million import make_stream
from axi.host import graph_from_json, graph_to_json


def _post(url: str, obj: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)


def _mean(xs):
    xs = [x for x in xs if x is not None]; return round(sum(xs) / len(xs), 4) if xs else None


def run(sidecar_url: str, steps: int, seed: int = 20261021, stream: str = "LiveStream", progress=None) -> dict:
    base = sidecar_url.rstrip("/"); st = make_stream(stream, seed)
    sid = _post(base + "/v1/axi/sessions", {"state": graph_to_json(st.start)})["session"]; rows = []
    for t in range(steps):
        out = st.next()
        if out is None: break
        req, _ = out; oracle = st.live; t0 = time.time()
        r = _post(f"{base}/v1/axi/sessions/{sid}/step", {"instruction": req["instruction"]}); dt = time.time() - t0
        held = graph_from_json(r["state"])
        rows.append(dict(t=t + 1, kind=req["kind"], parsed=bool(r["parsed"]), exact=same_graph(held, oracle), corrupted=not held.is_well_formed(), seconds=dt, prompt_tokens=(r.get("usage") or {}).get("prompt_tokens")))
        if progress: progress(rows[-1])
    n = len(rows); k = max(1, n // 10); first, last = rows[:k], rows[-k:]
    return dict(steps=n, exact=sum(x["exact"] for x in rows), parsed=sum(x["parsed"] for x in rows), corrupted=sum(x["corrupted"] for x in rows),
                seconds_first_tenth=_mean([x["seconds"] for x in first]), seconds_last_tenth=_mean([x["seconds"] for x in last]),
                prompt_tokens_first_tenth=_mean([x["prompt_tokens"] for x in first]), prompt_tokens_last_tenth=_mean([x["prompt_tokens"] for x in last]),
                by_kind={k_: [sum(1 for x in rows if x["kind"] == k_ and x["exact"]), sum(1 for x in rows if x["kind"] == k_)] for k_ in sorted({x["kind"] for x in rows})})

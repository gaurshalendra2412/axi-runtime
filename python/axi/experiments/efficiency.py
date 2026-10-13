"""Two cheap measurements of efficiency claims from a pasted AI audit (docs/EFFICIENCY_MEASURES.md; concept-map item 89). Prediction and reading rules written BEFORE any number existed.

  transport  What does it cost, per step, to turn the state into prompt text, to push it through JSON, and to push it through our Python reference of the binary frame (`axi.wire`, CRC-32C
             included), against one model call? The paste says JSON and text work costs "2 to 8 ms per step" and a shared-memory binary frame "under 350 ns".
  arc        On the 400 public ARC training tasks: how much shorter is a prompt when empty (colour 0) cells are left out (`d4.compile_stream(keep_background=False)`) than when the grid is
             written out? The paste says "often 80 to 90 percent". Also: does the canonical task prompt (`d4.canonicalize_task`) come out identical for all 8 rotated/mirrored images of a real task?

Length is counted in CHARACTERS of the text (a proxy: for the digit-by-digit tokenisers this repo targets, a digit or a punctuation mark is about one token; a real tokenizer may merge some
punctuation, which would shorten both texts; nothing here is a token count of a particular model). All thresholds are integers; timings are in whole nanoseconds."""
import json, os, statistics, sys, time
from typing import List

import numpy as np

from axi import wire
from axi.engine import d4
from axi.engine.cico_parser import CICOParser
from axi.experiments.agent_loop import _serialize
from axi.experiments.cascade import next_state
from axi.experiments.scope_tasks import make_scope_tasks

SEED, N_GRAPHS, INNER, REPEATS = 7, 140, 200, 5
ARC_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"       # fchollet/arc-agi, the checkout the page names
N_ARC_TASKS = 400
REL_IDS = {"dep": 1, "owns": 2}                                 # the relations the tasks use (u16 in the frame)


# ---------------------------------------------------------------- transport
def graph_dict(g) -> dict:
    return {"nodes": [[r, c, v] for (r, c), v in sorted(g.nodes.items())],
            "edges": [[u[0], u[1], w[0], w[1], rel, wt] for (u, w, rel), wt in sorted(g.edges.items())]}


def json_roundtrip(g) -> dict: return json.loads(json.dumps(graph_dict(g)))


def wire_frame(g) -> bytes:
    """The state as one binary frame: COO tuples (src id, dst id, relation id, weight) with the node ids = positions in sorted order."""
    ids = {n: i for i, n in enumerate(sorted(g.nodes))}; keys = sorted(g.edges)
    payload = wire.pack_coo([ids[k[0]] for k in keys], [ids[k[1]] for k in keys], [REL_IDS[k[2]] for k in keys], [g.edges[k] for k in keys])
    return wire.encode(wire.MODE_COO, 1, len(ids), len(keys), payload)


def wire_roundtrip(g):
    h, p = wire.decode(wire_frame(g)); return h, wire.unpack_coo(p)


def per_call_ns(fn, inner=INNER, repeats=REPEATS, timer=time.perf_counter_ns) -> float:
    """Smallest mean time of `fn` over `repeats` blocks of `inner` calls (the minimum is the least disturbed block)."""
    best = None
    for _ in range(repeats):
        t0 = timer()
        for _ in range(inner): fn()
        dt = (timer() - t0) / inner
        best = dt if best is None or dt < best else best
    return best


def time_transport(n=N_GRAPHS, seed=SEED, inner=INNER, repeats=REPEATS, timer=time.perf_counter_ns) -> List[dict]:
    rows = []
    for i, t in enumerate(make_scope_tasks(n, seed)):
        g = t.graph; d = CICOParser.parse_transition_delta(t.ideal); frame = wire_frame(g)
        rows.append(dict(i=i, kind=t.kind, nodes=len(g.nodes), edges=len(g.edges), text_chars=len(_serialize(g)), json_bytes=len(json.dumps(graph_dict(g))), frame_bytes=len(frame),
                         text_ns=per_call_ns(lambda: _serialize(g), inner, repeats, timer), json_ns=per_call_ns(lambda: json_roundtrip(g), inner, repeats, timer),
                         wire_total_ns=per_call_ns(lambda: wire_roundtrip(g), inner, repeats, timer), wire_decode_ns=per_call_ns(lambda: wire.decode(frame), inner, repeats, timer),
                         filter_ns=per_call_ns(lambda: next_state("repair_m6", g, d), inner, repeats, timer)))
    return rows


def model_call_ns(path) -> int:
    """Median seconds per call of a saved run, in whole nanoseconds."""
    rows = json.load(open(path))["rows"]; return int(round(statistics.median(r["seconds"] for r in rows) * 1e9))


def summarize_transport(rows, model_ns: int) -> dict:
    med = lambda k: int(round(statistics.median(r[k] for r in rows)))
    out = {k: med(k) for k in ("text_ns", "json_ns", "wire_total_ns", "wire_decode_ns", "filter_ns", "text_chars", "json_bytes", "frame_bytes")}
    out.update(n=len(rows), model_call_ns=int(model_ns)); return out


def verdict_transport(s) -> List[tuple]:
    """Lines tagged [claim] (a number the paste states) or [scale] (how big the cost is against a model call). All comparisons are on integers."""
    return [("[claim] prompt text and the JSON round trip each take less than 2 ms per step (the paste: 2 to 8 ms)", max(s["text_ns"], s["json_ns"]) < 2_000_000),
            ("[claim] the Python binary-frame path (convert, encode with CRC-32C, decode) is faster than the JSON round trip", s["wire_total_ns"] < s["json_ns"]),
            ("[claim] decoding a frame takes under 350 ns in this Python reference (the paste: under 350 ns for the whole transfer)", s["wire_decode_ns"] < 350),
            ("[scale] prompt text plus the JSON round trip together are under 1 percent of one model call (the saved hosted-model run)", 100 * (s["text_ns"] + s["json_ns"]) < s["model_call_ns"])]


# ---------------------------------------------------------------- arc
def dense_text(g) -> str: return "\n".join("".join(str(int(v)) for v in r) for r in np.asarray(g))


def modal_background(g) -> int:
    """The most frequent colour of the grid (ties: the smallest colour)."""
    vals, counts = np.unique(np.asarray(g), return_counts=True); return int(vals[int(np.argmax(counts))])


def sparse_text(g, background=None) -> str:
    """Header (h, w, orientation id) and one `i,j=colour` per non-empty cell, in the canonical orientation (the repo's `compile_stream`).
    background=None: colour 0 is empty (the pre-registered run; no colour in the header). background=int: that colour is empty and the header names it (`b<colour>`; the exploratory re-read)."""
    s = d4.compile_stream(g, keep_background=False, background=0 if background is None else background); h, w, t = s["header"]
    head = f"{h}x{w}t{t}:" if background is None else f"{h}x{w}b{background}t{t}:"
    return head + ";".join(f"{i},{j}={c}" for i, j, c in s["cells"])


def load_arc_dir(folder) -> dict:
    """{name: task} in the d4 task format, from ARC json files: train pairs and the FIRST test input (the grids a prompt shows)."""
    out = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".json"): continue
        raw = json.load(open(os.path.join(folder, fn)))
        out[fn[:-5]] = {"train": [(np.asarray(p["input"]), np.asarray(p["output"])) for p in raw["train"]], "test_input": np.asarray(raw["test"][0]["input"])}
    return out


def arc_grid_rows(tasks: dict, background="zero"):
    """One row per prompt grid. background="zero": colour 0 is empty (the pre-registered run). background="modal": each grid's most frequent colour is empty (exploratory)."""
    assert background in ("zero", "modal"); grids = []
    for name, task in tasks.items():
        roles = [(f"train{i}_in", x) for i, (x, y) in enumerate(task["train"])] + [(f"train{i}_out", y) for i, (x, y) in enumerate(task["train"])] + [("test_in", task["test_input"])]
        for role, g in roles:
            g = np.asarray(g); bg = 0 if background == "zero" else modal_background(g)
            st = d4.compile_stream(g, keep_background=False, background=bg); back = d4.decompile_stream(st)
            dense, sparse = dense_text(g), sparse_text(g, None if background == "zero" else bg)
            grids.append(dict(task=name, role=role, h=int(g.shape[0]), w=int(g.shape[1]), nonzero=int((g != bg).sum()), dense=len(dense), sparse=len(sparse),
                              lossless=bool(back.shape == g.shape and (back == g).all()), **({} if background == "zero" else {"background": bg})))
    return grids


def arc_rows(tasks: dict):
    grids = arc_grid_rows(tasks, "zero"); per_task = []
    for name, task in tasks.items():
        raw = {d4.task_prompt(d4._transform_task(task, t)) for t in d4.IDS}
        canon = {d4.task_prompt(d4.canonicalize_task(d4._transform_task(task, t))[0]) for t in d4.IDS}
        per_task.append(dict(task=name, raw_distinct=len(raw), canon_distinct=len(canon)))
    return grids, per_task


def summarize_arc(grids, per_task) -> dict:
    n = len(grids); cells = [r["h"] * r["w"] for r in grids]
    share = [1 - r["nonzero"] / c for r, c in zip(grids, cells)]
    bands = {}
    for r, c in zip(grids, cells):
        b = min(9, (10 * r["nonzero"]) // c); k = f"{10 * b}-{10 * b + 10}%"        # share of NON-empty cells, in tenths
        e = bands.setdefault(k, dict(n=0, sparse_shorter=0)); e["n"] += 1; e["sparse_shorter"] += r["sparse"] < r["dense"]
    return dict(grids=n, tasks=len(per_task), lossless=sum(r["lossless"] for r in grids), median_empty_share=statistics.median(share),
                sparse_shorter=sum(r["sparse"] < r["dense"] for r in grids), ge50=sum(2 * r["sparse"] <= r["dense"] for r in grids), ge80=sum(5 * r["sparse"] <= r["dense"] for r in grids),
                total_dense=sum(r["dense"] for r in grids), total_sparse=sum(r["sparse"] for r in grids),
                canon_identical=sum(r["canon_distinct"] == 1 for r in per_task), raw_distinct_sum=sum(r["raw_distinct"] for r in per_task),
                bands={k: bands[k] for k in sorted(bands, key=lambda s: int(s.split("-")[0]))})


def verdict_arc(s) -> List[tuple]:
    return [("[guard] writing a grid as sparse tuples and reading it back restores every grid exactly", s["lossless"] == s["grids"]),
            ("[guard] the canonical task prompt is identical for all 8 rotated/mirrored images of every task", s["canon_identical"] == s["tasks"]),
            ("[claim] at least one grid in four is 80 percent shorter or more when empty cells are left out (the paste: often 80 to 90 percent)", 4 * s["ge80"] >= s["grids"]),
            ("[claim] leaving empty cells out gives a shorter text for at least half of all grids", 2 * s["sparse_shorter"] >= s["grids"]),
            ("[claim] summed over all grids, the sparse text is shorter than the plain grid text", s["total_sparse"] < s["total_dense"])]


# ---------------------------------------------------------------- printing and files
def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(f"  {'yes' if ok else 'NO ':<3}  {text}")


def print_transport(s):
    print(f"== transport, per step, median over {s['n']} graphs (8 nodes, about 8 edges) ==")
    for k, label in (("text_ns", "prompt text (serialize)"), ("json_ns", "JSON dumps + loads"), ("wire_total_ns", "frame: convert + encode + decode"), ("wire_decode_ns", "frame: decode only"), ("filter_ns", "the filter (repair_m6)")):
        print(f"  {label:<36}{s[k]:>12,} ns")
    print(f"  {'one model call (saved run, median)':<36}{s['model_call_ns']:>12,} ns")
    print(f"  sizes: text {s['text_chars']} chars, JSON {s['json_bytes']} bytes, frame {s['frame_bytes']} bytes")


def print_arc(s):
    print(f"== ARC training tasks: {s['tasks']} tasks, {s['grids']} prompt grids (length in characters) ==")
    print(f"  median share of empty cells {s['median_empty_share']:.3f}; sparse shorter on {s['sparse_shorter']} grids; 50% or more shorter on {s['ge50']}; 80% or more shorter on {s['ge80']}")
    print(f"  total plain {s['total_dense']:,} chars, total sparse {s['total_sparse']:,} chars ({s['total_sparse'] / s['total_dense']:.3f} of plain)")
    print(f"  lossless on {s['lossless']} of {s['grids']}; canonical prompt identical across the 8 images on {s['canon_identical']} of {s['tasks']} tasks; mean distinct raw prompts {s['raw_distinct_sum'] / s['tasks']:.2f}")
    for k, e in s["bands"].items(): print(f"  non-empty share {k:>8}: {e['n']:>5} grids, sparse shorter on {e['sparse_shorter']}")


def print_arc_grids(s):
    """Grid-level lines only (the exploratory re-read has no task-level part and no pre-registered reading rules)."""
    print(f"== ARC training grids, each grid's most frequent colour left out (exploratory, decided after the first run): {s['grids']} grids ==")
    print(f"  median share of left-out cells {s['median_empty_share']:.3f}; sparse shorter on {s['sparse_shorter']} grids; 50% or more shorter on {s['ge50']}; 80% or more shorter on {s['ge80']}")
    print(f"  total plain {s['total_dense']:,} chars, total sparse {s['total_sparse']:,} chars ({s['total_sparse'] / s['total_dense']:.3f} of plain); lossless on {s['lossless']} of {s['grids']}")
    for k, e in s["bands"].items(): print(f"  share of cells that are not the most frequent colour {k:>8}: {e['n']:>5} grids, sparse shorter on {e['sparse_shorter']}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["transport", "arc"]); ap.add_argument("--out", required=True)
    ap.add_argument("--model-run", help="saved result file whose median seconds per call is the model-call reference (transport)")
    ap.add_argument("--arc-dir", help="folder with the ARC training json files (arc)")
    ap.add_argument("--background", choices=["zero", "modal"], default="zero", help="arc: which colour counts as empty (modal = exploratory re-read, grids only)")
    ap.add_argument("--n", type=int, default=N_GRAPHS); ap.add_argument("--inner", type=int, default=INNER); ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    if a.what == "transport":
        rows = time_transport(n=a.n, inner=a.inner, repeats=a.repeats); s = summarize_transport(rows, model_call_ns(a.model_run)); print_transport(s); print_verdict(verdict_transport(s), "transport")
        json.dump(dict(meta=dict(test="efficiency_transport", seed=SEED, n=a.n, inner=a.inner, repeats=a.repeats, model_run=os.path.basename(a.model_run), python=sys.version.split()[0]), rows=rows, summary=s), open(a.out, "w"), indent=1)
    elif a.background == "zero":
        tasks = load_arc_dir(a.arc_dir); grids, per_task = arc_rows(tasks); s = summarize_arc(grids, per_task); print_arc(s); print_verdict(verdict_arc(s), "arc")
        json.dump(dict(meta=dict(test="efficiency_arc", arc_commit=ARC_COMMIT, n_tasks=len(tasks)), grids=grids, tasks=per_task, summary=s), open(a.out, "w"), indent=1)
    else:
        tasks = load_arc_dir(a.arc_dir); grids = arc_grid_rows(tasks, "modal"); s = summarize_arc(grids, []); print_arc_grids(s)
        json.dump(dict(meta=dict(test="efficiency_arc_modal", arc_commit=ARC_COMMIT, n_tasks=len(tasks), exploratory=True), grids=grids, summary=s), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

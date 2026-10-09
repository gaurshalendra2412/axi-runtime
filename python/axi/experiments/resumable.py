"""Resumable runner for long Colab experiments (concept-map row 9, structure against scale).

A Colab session can disconnect in the middle of a 40-minute run. `run_resumable` writes one JSON line per finished task to `path`; running it again with
the same arguments skips the tasks already in the file and carries on. The first line of the file records which run it belongs to (model, seed, number of
tasks, rule hint); a file from a different run is refused instead of being mixed in. A half-written last line (the session died while writing) is dropped
and that task is run again. The rows are passed through JSON before they are kept, so a run that was interrupted and resumed returns exactly what an
uninterrupted run returns. Nothing here touches the model or the gate."""
import json, os
from axi.experiments.agent_loop import run_task, summarize


def _roundtrip(row): return json.loads(json.dumps(row, default=str))


def load_rows(path, meta):
    """Rows already saved in `path` (a list, possibly empty). Rewrites the file without a damaged last line. Refuses a file whose header differs from `meta`."""
    if not os.path.exists(path): return []
    good, rows = [], []
    with open(path) as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except json.JSONDecodeError: break                                  # half-written last line: stop here, redo that task
            good.append(line)
            if k == 0:
                if obj.get("_meta") != _roundtrip(meta): raise ValueError(f"{path} belongs to a different run: {obj.get('_meta')} vs {_roundtrip(meta)}. Use another file name or delete it.")
            else: rows.append(obj)
    with open(path, "w") as f:
        for line in good: f.write(line + "\n")
    return rows


def run_resumable(llm, tasks, hint_rules, path, meta=None, progress=None):
    meta = dict(meta or {}, n=len(tasks), hint_rules=bool(hint_rules))
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for i in range(len(rows), len(tasks)):
            row = _roundtrip(run_task(llm, tasks[i], hint_rules))
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
            rows.append(row)
            if progress: progress(i + 1, len(tasks))
    return {"rows": rows, "summary": summarize(rows), "meta": meta}

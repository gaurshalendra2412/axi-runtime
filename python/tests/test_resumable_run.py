"""Engineering support (no concept-map row): axi/experiments/resumable.py, the runner used for the 7B run on Colab (concept-map row 9, structure against scale).
A Colab session can drop in the middle of a long run; the runner must (1) give exactly what an uninterrupted run gives, (2) pick up after the last finished task,
(3) redo a task whose line was half written, (4) refuse a file that belongs to another run, and (5) never run a finished task twice. The model here is a scripted
stand-in whose answers depend only on the messages it is given (so a resumed run and an uninterrupted one can be compared exactly)."""
import json, os, sys, tempfile, zlib
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.agent_loop import make_tasks, run_experiment, summarize
from axi.experiments.resumable import run_resumable, load_rows
from test_agent_loop import ScriptedLLM


class StatelessLLM(ScriptedLLM):
    """Same answers as ScriptedLLM, but the randomness is a function of the messages only."""
    def __init__(self, stop_after=None):
        super().__init__(0); self.calls_by_task = 0; self.stop_after = stop_after; self.total = 0

    def generate(self, messages, constrained, max_new_tokens=400):
        self.total += 1
        if self.stop_after is not None and self.total > self.stop_after: raise RuntimeError("session dropped")
        self.rng.seed(zlib.crc32(json.dumps([messages, constrained], sort_keys=True).encode()))
        return super().generate(messages, constrained)


def tmp(): return os.path.join(tempfile.mkdtemp(), "run.jsonl")


def test_uninterrupted_run_matches_run_experiment():
    tasks = make_tasks(20, 3); path = tmp()
    a = run_resumable(StatelessLLM(), tasks, False, path, meta={"model": "fake", "seed": 3})
    b = run_experiment(StatelessLLM(), tasks)
    assert a["summary"] == json.loads(json.dumps(b["summary"], default=str)) and len(a["rows"]) == 20
    assert len(open(path).read().splitlines()) == 21                            # one header line and one line per task
    assert load_rows(path, dict({"model": "fake", "seed": 3}, n=20, hint_rules=False)) == a["rows"]       # what is kept in memory is what the file holds


def test_interrupted_run_resumes_and_equals_the_uninterrupted_one():
    tasks = make_tasks(20, 3); path = tmp(); meta = {"model": "fake", "seed": 3}
    full = run_resumable(StatelessLLM(), tasks, False, tmp(), meta=meta)
    try: run_resumable(StatelessLLM(stop_after=40), tasks, False, path, meta=meta); raise AssertionError("should have dropped")
    except RuntimeError: pass
    done = len(load_rows(path, dict(meta, n=20, hint_rules=False)))
    assert 0 < done < 20
    llm = StatelessLLM(); resumed = run_resumable(llm, tasks, False, path, meta=meta)
    assert resumed["rows"] == full["rows"] and resumed["summary"] == full["summary"]
    fresh = StatelessLLM(); run_resumable(fresh, tasks, False, tmp(), meta=meta)
    assert llm.total < fresh.total                                              # the resumed run did NOT repeat the finished tasks
    again = StatelessLLM(); run_resumable(again, tasks, False, path, meta=meta)
    assert again.total == 0                                                     # a finished file asks the model for nothing


def test_half_written_last_line_is_redone_and_a_foreign_file_is_refused():
    tasks = make_tasks(6, 1); path = tmp(); meta = {"model": "fake", "seed": 1}
    full = run_resumable(StatelessLLM(), tasks, True, tmp(), meta=meta)
    run_resumable(StatelessLLM(), tasks, True, path, meta=meta)
    lines = open(path).read().splitlines()
    with open(path, "w") as f: f.write("\n".join(lines[:4]) + "\n" + lines[4][: len(lines[4]) // 2])    # header + 3 rows + half of the 4th
    resumed = run_resumable(StatelessLLM(), tasks, True, path, meta=meta)
    assert resumed["rows"] == full["rows"] and len(open(path).read().splitlines()) == 7
    lines = open(path).read().splitlines()                                      # a damaged line in the MIDDLE: everything after it is run again, so the order stays right
    with open(path, "w") as f: f.write("\n".join(lines[:3] + ["{\"kind\": \"half"] + lines[4:]) + "\n")
    resumed = run_resumable(StatelessLLM(), tasks, True, path, meta=meta)
    assert resumed["rows"] == full["rows"] and len(open(path).read().splitlines()) == 7
    for bad in ({"model": "other", "seed": 1}, {"model": "fake", "seed": 2}):
        try: run_resumable(StatelessLLM(), tasks, True, path, meta=bad); raise AssertionError("accepted a foreign file")
        except ValueError: pass
    try: run_resumable(StatelessLLM(), tasks, False, path, meta=meta); raise AssertionError("accepted the other regime")
    except ValueError: pass
    try: run_resumable(StatelessLLM(), tasks[:3], True, path, meta=meta); raise AssertionError("accepted a different n")
    except ValueError: pass


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

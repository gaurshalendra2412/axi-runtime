"""Engineering support (no concept-map row): python/colab/colab_scale_run.py, the helpers behind the 7B notebook (concept-map row 9, structure against scale).
Checks, with a scripted stand-in for the model (no GPU): a regime run writes a resumable file and a result file; the printed comparison names every mode and shows the
3B reference; the reference numbers equal the 'Seed 3' table in docs/EXPERIMENT_RESULTS.md (a typo there would make the comparison misleading); and the plain-words verdict
follows the thresholds written down before the run (50 and 20 percent of delete_dep tasks corrupted by blind apply)."""
import json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from colab.colab_scale_run import run_regime, comparison_table, verdict, REFERENCE_3B, REFERENCE_3B_M1_CORRUPTED_DELETE_DEP, MODE_LABEL
from axi.experiments.agent_loop import MODES
from test_resumable_run import StatelessLLM


def test_regime_run_writes_files_and_the_table_names_every_mode():
    out = tempfile.mkdtemp(); res = {}
    for hint in (False, True):
        r, path = run_regime(StatelessLLM(), "Qwen/Fake-7B", out, hint, seed=3, n=10, progress=False)
        assert os.path.exists(path) and os.path.exists(path.replace(".json", ".jsonl")) and len(r["rows"]) == 10
        assert json.load(open(path))["summary"] == r["summary"] and r["meta"]["hint_rules"] == hint and r["meta"]["model"] == "Qwen/Fake-7B" and r["meta"]["seed"] == 3 and r["meta"]["load_4bit"] is True and r["meta"]["max_new_tokens"] == 400
        res[hint] = r
    table = comparison_table(res)
    for m in MODES: assert MODE_LABEL[m] in table
    assert "no rule in the prompt" in table and "rule stated in the prompt" in table and "3B 100.0%" in table and "3B 76.7%" in table
    again, _ = run_regime(StatelessLLM(stop_after=0), "Qwen/Fake-7B", out, False, seed=3, n=10, progress=False)   # a finished file asks the model for nothing
    assert again["rows"] == res[False]["rows"]


def parse_seed3_table():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "EXPERIMENT_RESULTS.md")
    if not os.path.exists(path): return None
    text = open(path).read(); text = text[text.index("## Seed 3: the first fresh seed"):]; rows = {}
    for line in text.splitlines():
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[1].endswith("%"):
            rows[cells[0]] = [float(re.sub(r"[^0-9.]", "", c)) for c in cells[1:]]
    return rows


def test_reference_numbers_equal_the_seed_3_table_in_the_results_doc():
    rows = parse_seed3_table()
    if rows is None: return                                                    # docs not shipped with this copy of the repo
    assert len(rows) == 9
    for m in MODES:
        nums = rows[MODE_LABEL[m]]                                            # no-rule all, no-rule delete_dep, rule all, rule delete_dep
        assert (nums[0], nums[1]) == REFERENCE_3B[False][m] and (nums[2], nums[3]) == REFERENCE_3B[True][m], m
    assert REFERENCE_3B_M1_CORRUPTED_DELETE_DEP == {False: 100.0, True: 76.7}


def fake_result(m1_corrupted):
    base = {"success": 0.5, "corrupted": 0.0, "strict_parse": 1.0, "avg_tokens": 10.0}
    dd = {"n": 30}; al = {"n": 60}
    for m in MODES: dd[m] = dict(base); al[m] = dict(base)
    dd["M1_constrained"]["corrupted"] = m1_corrupted
    return {"summary": {"all": al, "delete_dep": dd}, "rows": [], "meta": {}}


def test_verdict_follows_the_thresholds_written_before_the_run():
    assert "still corrupts 100%" in verdict({False: fake_result(1.0)}) and "did not fix" in verdict({False: fake_result(0.5)})
    assert "scale helped a lot" in verdict({False: fake_result(0.499)}) and "scale helped a lot" in verdict({False: fake_result(0.2)})
    assert "nearly fixes it" in verdict({False: fake_result(0.199)}) and "nearly fixes it" in verdict({False: fake_result(0.0)})
    assert "not run" in verdict({True: fake_result(0.0)})
    gated = fake_result(1.0); gated["summary"]["delete_dep"]["M5_auto_repair"]["corrupted"] = 0.1
    assert "Worst corruption in any gated mode: 10%" in verdict({False: gated})


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

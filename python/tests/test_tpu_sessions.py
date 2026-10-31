import hashlib
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def test_tpu_page_is_locked_and_module_imports():
    from axi.experiments import tpu_sessions as T
    assert callable(T.measure)
    t = open(os.path.join(DOCS, "TPU_SESSIONS_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "tpu_sessions_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want


def test_tpu_result_figures():
    import json
    d = json.load(open(os.path.join(DOCS, "results", "tpu_sessions.json")))
    c = d["counts"]
    assert d["bytes_limit"] == 33546042880
    assert [c[k]["sessions_held"] for k in c] == [927, 54, 27, 10, 5, 2]
    assert c["AXI step"]["real_over_nominal"] == 1.0454
    t = open(os.path.join(DOCS, "TPU_SESSIONS_TEST.md")).read()
    for needle in ("**927**", "5 of 6 lines read YES", "185 times", "336"):
        assert needle in t

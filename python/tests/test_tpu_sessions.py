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

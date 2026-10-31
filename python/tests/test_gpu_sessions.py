import hashlib
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def test_gpu_sessions_formula_page_and_counter():
    from axi.experiments import gpu_sessions as G
    assert G.bytes_per_token(24, 2, 64) == 12288
    a = G.arithmetic(80, 15)["Qwen2.5-7B"]
    assert a["AXI step"] == 401 and a["kept chat, largest of the long run"] == 1
    left = [1000]
    def alloc(t):
        if left[0] < t:
            raise MemoryError
        left[0] -= t
        return t
    assert G.count_until_full(alloc, lambda h: h.clear(), 100) == 10
    t = open(os.path.join(DOCS, "GPU_SESSIONS_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "gpu_sessions_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want


def test_gpu_sessions_result_figures():
    import json
    d = json.load(open(os.path.join(DOCS, "results", "gpu_sessions.json")))
    assert [m["per_token"] for m in d["measured"]] == [12288.0, 12288.0]
    c = d["counts"]
    assert c["AXI step"]["sessions_held"] == 430 and c["kept chat, mean of the long run"]["sessions_held"] == 2
    assert c["kept chat, largest of the long run"]["sessions_held"] == 1
    t = open(os.path.join(DOCS, "GPU_SESSIONS_TEST.md")).read()
    for needle in ("**430**", "4 of 5 lines read YES", "401 AXI-sized sessions", "540 against 3"):
        assert needle in t

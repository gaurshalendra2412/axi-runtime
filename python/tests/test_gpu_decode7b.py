import hashlib
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def test_l4_plan_arithmetic_and_page_lock():
    from axi.experiments import gpu_decode7b as D
    from axi.experiments import tpu_decode as T
    l4_total = 24e9
    free = l4_total - T.WEIGHT_PARAMS * 2                      # about 8.8 GB before the driver's own share; the run reads the real figure
    assert D.cache_bytes(250000, 1) > free                     # one kept chat at 250k does not fit next to the weights
    assert D.cache_bytes(T.KEPT_MEAN, 1) > free
    assert D.cache_bytes(100000, 1) < free - D.MARGIN
    assert D.cache_bytes(T.AXI_TOKENS, 32) < free - D.MARGIN
    t = open(os.path.join(DOCS, "GPU_DECODE_7B_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "gpu_decode7b_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want


def test_l4_result_figures():
    import json
    d = json.load(open(os.path.join(DOCS, "results", "gpu_decode7b.json")))
    r = {(x["name"], x["sessions"]): x for x in d["rows"]}
    a32, k100 = r[("AXI step", 32)], r[("kept chat at 100k", 1)]
    assert round(a32["tokens_per_second"], 1) == 366.2 and round(k100["tokens_per_second"], 1) == 8.6
    assert a32["tokens_per_second"] / k100["tokens_per_second"] > 15
    assert r[("kept chat at 250k", 1)]["fits"] is False and r[("kept chat, mean of the long run", 1)]["fits"] is False
    assert a32["tokens_per_second"] / r[("AXI step", 1)]["tokens_per_second"] > 10
    t = open(os.path.join(DOCS, "GPU_DECODE_7B_TEST.md")).read()
    for needle in ("7 of 7 lines read YES", "42.5 times", "does not fit (27.3 GB of cache)", "decode only"):
        assert needle in t

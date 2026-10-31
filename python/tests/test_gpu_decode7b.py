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

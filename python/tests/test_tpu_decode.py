import hashlib
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def test_decode_shape_arithmetic_and_page_lock():
    from axi.experiments import tpu_decode as D
    assert D.KV_BYTES_PER_TOKEN == 57344
    assert 7.5e9 < D.WEIGHT_PARAMS < 7.8e9                      # Qwen2.5-7B has about 7.6 billion parameters
    limit = 33546042880
    assert D.need_bytes(D.KEPT_MEAN, 1) > limit                  # one kept chat at the long run's mean does not fit next to the weights
    assert D.need_bytes(D.AXI_TOKENS, 64) < limit - (1 << 30)
    for name, L, bs in D.PLAN:
        if name != "kept chat, mean of the long run":
            assert D.need_bytes(L, max(bs)) < limit - (1 << 30), name
    t = open(os.path.join(DOCS, "TPU_DECODE_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "tpu_decode_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want

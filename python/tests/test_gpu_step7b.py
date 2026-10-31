import hashlib
import json
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def test_combine_with_made_up_numbers_and_page_lock():
    from axi.experiments import gpu_step7b as S
    dec = [dict(name="AXI step", sessions=1, step_ms=60.0), dict(name="AXI step", sessions=32, step_ms=90.0),
           dict(name="kept chat at 100k", sessions=1, step_ms=120.0)]
    pre = [dict(tokens=100, prefill_ms=100.0), dict(tokens=2824, prefill_ms=700.0), dict(tokens=32768, prefill_ms=10000.0)]
    assert S.interp([(100, 100.0), (2824, 700.0)], 2824) == 700.0 and S.interp([(100, 100.0)], 50) is None
    c = S.combine(dec, pre, 40)
    # AXI at 32 sessions: 700 ms of prefill + 40 reply tokens at 90 ms shared by 32 = 700 + 112.5 ms
    assert abs(c[("AXI step", 32)]["chip_seconds_per_step"] - 0.8125) < 1e-9
    # kept chat at 100k: 100 ms prefill + 60 ms cache read + 40 x 120 ms
    assert abs(c[("kept chat at 100k", 1)]["chip_seconds_per_step"] - 4.96) < 1e-9
    f = S.first_token(dec, pre)
    assert f["AXI step"] == 760.0
    t = open(os.path.join(DOCS, "GPU_STEP_7B_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "gpu_step7b_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want

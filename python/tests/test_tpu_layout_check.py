import json
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def _ceil128(t):
    return -(-t // 128) * 128


def test_layout_check_matches_the_tile_rule():
    d = json.load(open(os.path.join(DOCS, "results", "tpu_layout_check.json")))
    assert len(d["rows"]) == 12
    for r in d["rows"]:
        t = r["tokens"]
        padded = 2 * 1 * 2 * _ceil128(t) * 64 * 2
        assert r["nominal_bytes"] == 2 * 1 * 2 * t * 64 * 2
        assert r["on_device_size_in_bytes"] == padded
        assert r["argument_size_in_bytes"] == padded
        assert r["output_size_in_bytes"] == padded
        assert r["temp_size_in_bytes"] == 0 and r["alias_size_in_bytes"] == 0
        assert all("{3,4,2,1,0:T(8,128)(2,1)}" in s for s in r["layout_in_compiled_hlo"])
    first = d["rows"][0]
    assert first["bytes_in_use_delta"] - first["on_device_size_in_bytes"] == 95232
    assert [p["extra_per_array"] for p in d["per_array"]] == [61440.0, 0.0]


def test_test30_v6e_bytes_are_padded_size_plus_a_constant():
    d = json.load(open(os.path.join(DOCS, "results", "tpu_sessions.json")))
    extra = {}
    for k, c in d["counts"].items():
        padded = d["shape"][0] * 2 * 1 * d["shape"][1] * _ceil128(c["tokens"]) * d["shape"][2] * 2
        extra[k] = c["real_session_bytes"] - padded
    assert extra["AXI step"] == 101888
    assert sorted(set(v for k, v in extra.items() if k != "AXI step")) == [94208, 94720]
    t = open(os.path.join(DOCS, "TPU_SESSIONS_TEST.md")).read()
    for needle in ("TPU v5 lite", "{3,4,2,1,0:T(8,128)(2,1)}", "95,232", "101,888", "4.25 points"):
        assert needle in t

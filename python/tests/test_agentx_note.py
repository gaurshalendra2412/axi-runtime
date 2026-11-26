"""Checks the AXI figures of docs/AGENTX_NOTE.md against the saved result files. No model call, no network."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi import price_check as P  # noqa: E402

PAGE = os.path.join(HERE, "..", "..", "docs", "AGENTX_NOTE.md")


def page():
    return open(PAGE, encoding="utf-8").read()


def means():
    a = P.arms()
    return {k: a[k]["prompt"] / a[k]["steps"] for k in ("A", "KC", "KW")}, a


def test_mean_prompt_tokens_and_ratios():
    m, a = means()
    assert [round(m[k]) for k in ("A", "KC", "KW")] == [1177, 97353, 99809]
    assert [round(m["KC"] / m["A"], 1), round(m["KW"] / m["A"], 1)] == [82.7, 84.8]
    assert (a["A"]["prompt_max"], a["KC"]["prompt_max"], a["KW"]["prompt_max"]) == (1390, 193411, 198405)


def test_warm_cache_share():
    a = P.arms()
    assert round(100 * a["KW"]["cached"] / a["KW"]["prompt"], 1) == 97.0
    assert a["A"]["cached"] == 0 and a["KC"]["cached"] == 0


def test_verified_steps_on_the_page():
    a = P.arms()
    assert [(a[k]["verified"], a[k]["steps"]) for k in ("A", "KC", "KW")] == [(400, 400), (328, 390), (260, 400)]


def test_page_states_every_figure_it_tests():
    t = page()
    for s in ("99,809", "97,353", "1,177", "84.8", "82.7", "97.0 percent", "1,390", "193,411", "198,405", "260 of 400", "328 of 390"):
        assert s in t, s

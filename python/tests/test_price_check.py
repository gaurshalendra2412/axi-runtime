"""Checks docs/PRICE_CHECK.md against the saved result files and docs/results/price_check.json. No model call, no network."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi import price_check as P  # noqa: E402

PAGE = os.path.join(HERE, "..", "..", "docs", "PRICE_CHECK.md")


def page():
    return open(PAGE, encoding="utf-8").read()


def test_token_counts_are_the_billing_pages():
    a = P.arms()
    assert (a["A"]["prompt"], a["KC"]["prompt"], a["KW"]["prompt"], a["KW"]["cached"]) == (470669, 37967655, 39923458, 38728988)
    assert (a["A"]["out"], a["KC"]["out"], a["KW"]["out"]) == (57527, 12351, 13456)
    assert [(a[k]["steps"], a[k]["verified"]) for k in ("A", "KC", "KW")] == [(400, 400), (390, 328), (400, 260)]
    assert (a["KC"]["prompt_max"], a["KW"]["prompt_max"]) == (193411, 198405) and max(a["KC"]["prompt_max"], a["KW"]["prompt_max"]) < 200000


def test_google_rates_give_back_the_billing_page():
    r = P.table()["Gemini 3.8 Flash"]
    assert [round(r["per_verified"][k], 4) for k in ("A", "KC", "KW")] == [0.0014, 0.0870, 0.0148]
    assert [round(r["times_verified"][k], 1) for k in ("KC", "KW")] == [61.2, 10.4]
    assert [round(r["per_step"][k], 4) for k in ("A", "KC", "KW")] == [0.0014, 0.0731, 0.0096]
    assert [round(r["times_step"][k], 1) for k in ("KC", "KW")] == [51.4, 6.8]
    assert [round(r["cell"][k], 2) for k in ("A", "KC", "KW")] == [0.57, 28.52, 3.85]


def test_other_providers_figures():
    t = P.table()
    want = {
        "Claude Sonnet 5.5": ([0.0038, 0.2319, 0.0246], [61.2, 6.5]),
        "Claude Opus 5.5": ([0.0076, 0.4638, 0.0492], [61.2, 6.5]),
        "Grok 4.3": ([0.0018, 0.1448, 0.0357], [79.1, 19.5]),
        "Grok 4.7": ([0.0032, 0.2317, 0.0840], [72.1, 26.1]),
    }
    for n, (per, times) in want.items():
        r = t[n]
        assert [round(r["per_verified"][k], 4) for k in ("A", "KC", "KW")] == per, n
        assert [round(r["times_verified"][k], 1) for k in ("KC", "KW")] == times, n
    # the title's ranges are the ends of the rows
    cold = [r["times_verified"]["KC"] for r in t.values()]
    warm = [r["times_verified"]["KW"] for r in t.values()]
    assert (round(min(cold)), round(max(cold)), round(min(warm), 1), round(max(warm))) == (61, 79, 6.5, 26)
    assert round(t["Gemini 3.8 Flash"]["per_verified"]["A"] * 1e6) == 1422 and round(t["Gemini 3.8 Flash"]["per_verified"]["KC"] * 1e6) == 86957


def test_llama_kv_arithmetic():
    c = P.load("price_check.json")["llama_public_configs"]
    b8 = P.kv_bytes_per_token(**{"layers": c["Llama 3.1 8B"]["layers"], "kv_heads": c["Llama 3.1 8B"]["kv_heads"], "head_dim": c["Llama 3.1 8B"]["head_dim"]})
    b70 = P.kv_bytes_per_token(c["Llama 3.1 70B"]["layers"], c["Llama 3.1 70B"]["kv_heads"], c["Llama 3.1 70B"]["head_dim"])
    assert (b8, b70) == (131072, 327680)
    a = P.arms()
    mean_a, mean_k = round(a["A"]["prompt"] / a["A"]["steps"]), round(a["KC"]["prompt"] / a["KC"]["steps"])
    assert (mean_a, mean_k) == (1177, 97353)
    assert round(mean_a * b8 / 1e6, 1) == 154.3 and round(mean_k * b8 / 1e9, 2) == 12.76
    assert round(mean_a * b70 / 1e6, 1) == 385.7 and round(mean_k * b70 / 1e9, 2) == 31.90
    assert round(mean_k / mean_a, 1) == 82.7
    d = P.load("axi_cache_pair_seed20261122_default.json")
    p = d["KC"]["prompt_tokens_per_step"]
    assert next(i for i, x in enumerate(p) if x > 131072) + 1 == 264 and sum(1 for x in p if x > 131072) == 127 and max(d["A"]["prompt_tokens_per_step"]) == 1390


def test_page_carries_the_figures():
    t = page()
    for needle in ("$0.0038", "$0.2319", "$0.0246", "61.2", "6.5", "$0.0018", "$0.1448", "$0.0357", "79.1", "19.5", "$0.0032", "$0.2317", "$0.0840", "72.1", "26.1",
                   "$0.0014", "$0.0870", "$0.0148", "$1,422", "$86,957", "$14,811", "$3,792", "$231,887", "$24,602",
                   "470,669", "37,967,655", "39,923,458", "38,728,988", "198,405", "193,411", "131,072", "327,680", "154.3 MB", "12.76 GB", "385.7 MB", "31.90 GB", "82.7", "step 264",
                   "Last updated: October 10, 2026", "61x to 79x", "6.5x to 26x"):
        assert needle in t, needle
    for banned in ("prototype", "blueprint", "open source", "open-source"):
        assert banned not in t.lower(), banned

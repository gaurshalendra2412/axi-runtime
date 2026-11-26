"""Prices the measured token counts of the cache-pair job (Gemini 3.8 Flash, seed 20261122) at published list prices.

No call to any model and no network. The token counts come from the saved result files; the prices are typed in docs/results/price_check.json
with the page each was read on and the date. Nothing here is a bill: the one bill we have read is Google's (docs/BILLING_CHECK.md).
"""
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "results")


def load(name):
    return json.load(open(os.path.join(RES, name)))


def arms():
    d = load("axi_cache_pair_seed20261122_default.json")
    f = load("axi_gemini_fuzzy_seed20261122_default.json")
    A, K = d["A"], d["KC"]
    kw = f["K"]["summary"]["all"]
    kw_out = sum((r.get("tokens") or 0) + (r.get("think") or 0) for r in f["K"]["rows"])
    kw_prompts = [r["prompt_tokens"] for r in f["K"]["rows"]]
    return {
        "A": dict(prompt=A["prompt_tokens_total"], cached=0, out=A["visible_tokens"] + A["thinking_tokens"], steps=A["steps"], verified=A["verified_steps"], prompt_max=A["prompt_max"]),
        "KC": dict(prompt=K["prompt_tokens_total"], cached=0, out=K["visible_tokens"] + K["thinking_tokens"], steps=K["steps"], verified=K["verified_steps"], prompt_max=K["prompt_max"]),
        "KW": dict(prompt=kw["prompt"], cached=kw["cache"], out=kw_out, steps=len(kw_prompts), verified=d["reference_test_23_saved_file"]["K_verified_steps"], prompt_max=max(kw_prompts)),
    }


def cost(a, p):
    """Dollars for one cell: fresh prompt tokens at the input price, cached ones at the cached price, output at the output price (per million)."""
    return ((a["prompt"] - a["cached"]) * p["input"] + a["cached"] * p["cached"] + a["out"] * p["output"]) / 1e6


def table(prices=None):
    prices = prices or load("price_check.json")["prices"]
    a = arms()
    out = {}
    for name, p in prices.items():
        c = {k: cost(v, p) for k, v in a.items()}
        per_v = {k: c[k] / a[k]["verified"] for k in a}
        per_s = {k: c[k] / a[k]["steps"] for k in a}
        out[name] = dict(cell=c, per_verified=per_v, per_step=per_s,
                         times_verified={k: per_v[k] / per_v["A"] for k in ("KC", "KW")},
                         times_step={k: per_s[k] / per_s["A"] for k in ("KC", "KW")})
    return out


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_value

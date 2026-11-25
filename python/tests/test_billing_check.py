"""Checks docs/BILLING_CHECK.md against the lines copied from Google's billing report (docs/results/google_bill_check_2026-11-25.json) and the saved result files of our own runs, with no call to any model and no network:
  1. the four saved files (hash);
  2. Google's rates: rupees a million tokens on three date ranges, the ratios output/input and cached/input (no exchange rate needed), the rate in rupees to the dollar that the typed prices imply, the lines adding up to the totals;
  3. our logs against Google on 22 Nov (cache pair test) and on 1 to 3 Nov (Test 34): tokens, dollars, differences;
  4. the cost of a verified step of the three cells at Google's rates (A, KC, KW);
  5. that the page carries those figures, and that the wording rules hold (nothing called proven, open source, a prototype or a blueprint)."""
import hashlib, json, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
RES = os.path.join(DOCS, "results")
PAGE = os.path.join(DOCS, "BILLING_CHECK.md")
FILES = {
    "google_bill_check_2026-11-25.json": "fc3d99ddaa9dc1c84464e4a7cb3c1259ab9f553f42aeffdfc588b32ea72785c0",
    "axi_cache_pair_seed20261122_default.json": "c8a1a28ce5777d7f010bbd0a5e06410ad796b488f8afa19383d936db180a468f",
    "axi_gemini_fuzzy_seed20261122_default.json": "fd104d4be5103d4dbb49a225ce5975194b4551c48f6be83c285b35ce83ec00e8",
    "gemini_rulehint.json": "7c5b75b9a48cc20e83d7cce079a865a7eef68e32eee5377afc924ec45cde7ccd",
}
IN, OUT, CACHED = 0.75, 3.75, 0.075            # dollars a million tokens: the typed prices, and a tenth of the input price for a cached token
SKU_IN, SKU_CACHED, SKU_OUT = "4F40-96C8-648E", "A1C3-9773-AA71", "C5EE-B267-EE7E"


def load(name): return json.load(open(os.path.join(RES, name)))


def page(): return open(PAGE, encoding="utf-8").read()


def lines(rng):
    g = load("google_bill_check_2026-11-25.json")["ranges"][rng]
    return g, {l["sku_id"]: l for l in g["lines"]}


def per_million(l): return l["cost_inr"] / (l["tokens"] / 1e6)


def test_the_saved_files_are_the_ones_the_page_was_written_from():
    for name, sha in FILES.items():
        assert hashlib.sha256(open(os.path.join(RES, name), "rb").read()).hexdigest() == sha, name


def test_the_lines_add_up_to_the_totals_of_the_report():
    for rng in ("1_to_25_nov", "22_nov", "1_to_3_nov"):
        g, _ = lines(rng); s = 0.0
        for l in g["lines"]: s += l["cost_inr"]
        assert abs(s - g["total_inr"]) <= 0.02, (rng, s, g["total_inr"])         # the report rounds each line to the paisa
        assert g["tax"] is None
    assert load("google_bill_check_2026-11-25.json")["ai_studio_spend_page_1_to_25_nov_inr"] == 4249.84


def test_google_rates_are_the_typed_ones_and_do_not_move_with_the_day():
    _, m = lines("1_to_25_nov"); _, d = lines("22_nov"); _, e = lines("1_to_3_nov")
    ri, rc, ro = per_million(m[SKU_IN]), per_million(m[SKU_CACHED]), per_million(m[SKU_OUT])
    assert (round(ri, 2), round(rc, 2), round(ro, 2)) == (71.42, 7.14, 357.11)
    assert abs(ro / ri - OUT / IN) < 1e-3 and abs(rc / ri - 0.1) < 1e-5            # 5.000 and 0.100, no exchange rate involved
    fx = ri / IN
    assert round(fx, 2) == 95.23 and abs(ro / OUT - fx) < 0.01 and abs(rc / CACHED - fx) < 0.01
    for r, ref in ((d[SKU_IN], ri), (e[SKU_IN], ri), (d[SKU_OUT], ro), (e[SKU_OUT], ro)):
        assert abs(per_million(r) / ref - 1) < 0.001, r
    assert (round(per_million(d[SKU_IN]), 2), round(per_million(d[SKU_OUT]), 2), round(per_million(e[SKU_IN]), 2), round(per_million(e[SKU_OUT]), 2)) == (71.42, 357.07, 71.42, 357.11)
    assert round(per_million(d[SKU_CACHED]), 1) == 7.1
    # Gemini 3 Flash: cached a tenth of input, output six times input
    m3 = {l["sku_id"]: l for l in lines("1_to_25_nov")[0]["lines"]}
    i3, c3, o3 = per_million(m3["F2C1-F842-5D84"]), per_million(m3["07D6-73CA-C859"]), per_million(m3["ACA6-865B-F153"])
    assert abs(c3 / i3 - 0.1) < 1e-3 and abs(o3 / i3 - 6.0) < 0.01
    assert [round(x / fx, 2) for x in (i3, c3, o3)] == [0.5, 0.05, 3.0]


def our_cache_pair():
    d = load("axi_cache_pair_seed20261122_default.json"); A, K = d["A"], d["KC"]
    return A, K


def test_cache_pair_day_against_the_logs():
    A, K = our_cache_pair(); _, day = lines("22_nov"); fx = per_million(lines("1_to_25_nov")[1][SKU_IN]) / IN
    log_in = A["prompt_tokens_total"] + K["prompt_tokens_total"]
    log_out = A["visible_tokens"] + A["thinking_tokens"] + K["visible_tokens"] + K["thinking_tokens"]
    log_usd = A["dollars_at_typed_prices"] + K["dollars_at_typed_prices"]
    g_in = day[SKU_IN]["tokens"] + day[SKU_CACHED]["tokens"]; g_out = day[SKU_OUT]["tokens"]; g_usd = lines("22_nov")[0]["total_inr"] / fx
    assert (log_in, log_out) == (38438324, 69878) and (g_in, g_out) == (38928349, 75084) and day[SKU_CACHED]["tokens"] == 33833
    assert (g_in - log_in, g_out - log_out) == (490025, 5206)
    assert round((g_in / log_in - 1) * 100, 2) == 1.27 and round((g_out / log_out - 1) * 100, 2) == 7.45
    assert round(log_usd, 2) == 29.09 and round(g_usd, 2) == 29.45 and round(g_usd - log_usd, 2) == 0.36 and round((g_usd / log_usd - 1) * 100, 2) == 1.25
    assert round(A["dollars_at_typed_prices"], 2) == 0.57 and round(A["dollars_at_typed_prices"] / log_usd * 100, 1) == 2.0
    assert abs(log_usd - (log_in * IN + log_out * OUT) / 1e6) < 1e-6                 # the logged dollars are the logged tokens at the typed prices


def test_test_34_against_the_logs():
    t = load("gemini_rulehint.json"); fx = per_million(lines("1_to_25_nov")[1][SKU_IN]) / IN
    l_in = l_out = 0
    for arm in t["arms"].values():
        s = arm["summary"]; l_in += s["prompt_tokens"]; l_out += s["output_tokens"] + s["think_tokens"]
    _, e = lines("1_to_3_nov"); g_in, g_out = e[SKU_IN]["tokens"], e[SKU_OUT]["tokens"]
    l_usd = (l_in * IN + l_out * OUT) / 1e6; g_usd = lines("1_to_3_nov")[0]["total_inr"] / fx
    assert (l_in, l_out, g_in, g_out) == (1031724, 261467, 1043909, 264232) and (g_in - l_in, g_out - l_out) == (12185, 2765)
    assert round((g_in / l_in - 1) * 100, 2) == 1.18 and round((g_out / l_out - 1) * 100, 2) == 1.06
    assert round(l_usd, 2) == 1.75 and round(g_usd, 2) == 1.77 and round(g_usd - l_usd, 2) == 0.02 and round((g_usd / l_usd - 1) * 100, 1) == 1.1


def test_the_cost_of_a_verified_step_at_googles_rates():
    A, K = our_cache_pair(); f = load("axi_gemini_fuzzy_seed20261122_default.json")
    kw = f["K"]["summary"]["all"]; kp, kc = kw["prompt"], kw["cache"]
    ko = 0
    for r in f["K"]["rows"]: ko += (r.get("tokens") or 0) + (r.get("think") or 0)
    assert (kp, kc, ko) == (39923458, 38728988, 13456) and round(kc / kp * 100, 1) == 97.0
    a_out = A["visible_tokens"] + A["thinking_tokens"]; k_out = K["visible_tokens"] + K["thinking_tokens"]
    assert (a_out, k_out) == (57527, 12351)
    a_usd = (A["prompt_tokens_total"] * IN + a_out * OUT) / 1e6
    kc_usd = (K["prompt_tokens_total"] * IN + k_out * OUT) / 1e6
    kw_usd = ((kp - kc) * IN + kc * CACHED + ko * OUT) / 1e6
    assert (round(a_usd, 2), round(kc_usd, 2), round(kw_usd, 2)) == (0.57, 28.52, 3.85)
    assert abs(a_usd - A["dollars_at_typed_prices"]) < 1e-6 and abs(kc_usd - K["dollars_at_typed_prices"]) < 1e-6
    va, vk, vw = A["verified_steps"], K["verified_steps"], 260
    assert (va, vk) == (400, 328)
    ref = load("axi_cache_pair_seed20261122_default.json")["reference_test_23_saved_file"]
    assert ref["K_verified_steps"] == vw and ref["K_prompt_tokens"] == kp and ref["K_cache_tokens"] == kc
    pa, pk, pw = a_usd / va, kc_usd / vk, kw_usd / vw
    assert (round(pa, 4), round(pk, 4), round(pw, 4)) == (0.0014, 0.0870, 0.0148)
    assert (round(pk / pa, 1), round(pw / pa, 1)) == (61.2, 10.4)
    sa, sk, sw = a_usd / A["steps"], kc_usd / K["steps"], kw_usd / 400                # every step counted, right or wrong
    assert (A["steps"], K["steps"]) == (400, 390) and (round(sa, 4), round(sk, 4), round(sw, 4)) == (0.0014, 0.0731, 0.0096)
    assert (round(sk / sa, 1), round(sw / sa, 1)) == (51.4, 6.8)


def test_the_month_and_the_usage_page():
    g = load("google_bill_check_2026-11-25.json"); m, _ = lines("1_to_25_nov"); _, mm = lines("1_to_25_nov"); fx = per_million(mm[SKU_IN]) / IN
    assert round(m["total_inr"] / fx, 2) == 44.63
    assert mm[SKU_IN]["tokens"] + mm[SKU_CACHED]["tokens"] == 104889843 and round(mm[SKU_CACHED]["tokens"] / 104889843 * 100, 1) == 54.5
    flash3 = 0.0
    for k in ("07D6-73CA-C859", "F2C1-F842-5D84", "A78C-B91C-FD61", "ACA6-865B-F153", "1918-CC56-A445"): flash3 += mm[k]["cost_inr"]
    assert round(flash3, 2) == 52.98
    assert round(m["total_inr"] - lines("22_nov")[0]["total_inr"] - lines("1_to_3_nov")[0]["total_inr"] - flash3, 2) == 1222.95
    rep_in = 0
    for k in (SKU_IN, SKU_CACHED, "07D6-73CA-C859", "F2C1-F842-5D84", "A78C-B91C-FD61", "1918-CC56-A445"): rep_in += mm[k]["tokens"]
    rep_out = mm[SKU_OUT]["tokens"] + mm["ACA6-865B-F153"]["tokens"]
    u = g["ai_studio_usage_page_last_28_days"]; u_in = u["input_tokens"] - u["day_28_oct"]["input_tokens"]; u_out = u["output_tokens"] - u["day_28_oct"]["output_tokens"]
    assert (rep_in, rep_out, u_in, u_out) == (112254916, 1074754, 117818503, 1074911)
    assert round((u_in / rep_in - 1) * 100, 2) == 4.96 and round(abs(u_out / rep_out - 1) * 100, 3) == 0.015


def test_the_page_carries_the_figures_and_keeps_the_wording_rules():
    p = page()
    for s in ("₹2,805.00", "$29.09", "$29.45", "1.25 percent", "₹168.92", "$1.75", "$1.77", "1.1 percent", "5.000", "0.100", "₹95.23",
              "47,731,389", "57,158,454", "1,062,778", "38,894,516", "75,084", "1,043,909", "264,232",
              "38,438,324", "38,928,349", "+490,025 (+1.27 %)", "+5,206 (+7.45 %)", "+$0.36 (+1.25 %)",
              "1,031,724", "261,467", "+12,185 (+1.18 %)", "+2,765 (+1.06 %)",
              "470,669", "37,967,655", "39,923,458", "38,728,988 (97.0 %)", "57,527", "12,351", "13,456",
              "0.0014", "0.0870", "0.0148", "61.2", "10.4", "0.0731", "0.0096", "51.4", "6.8", "₹4,249.85", "$44.63", "54.5 percent", "₹52.98", "₹1,222.95",
              "+4.96 percent", "0.015 percent", "117.8M", "112.3M"):
        assert s in p, s
    low = p.lower()
    for bad in ("proven", "open source", "open-source", "prototype", "blueprint", "we cracked", "revolution"): assert bad not in low, bad
    assert "company that runs the model" in p and "host side" in p and "Rajnish Choubey" in p and "not for end users" in p
    for ident in ("@", "gen-lang-client", "015765"): assert ident not in p and ident not in open(os.path.join(RES, "google_bill_check_2026-11-25.json")).read()

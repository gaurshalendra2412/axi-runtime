"""Checks the saved result of the cache pair test (docs/CACHE_PAIR_TEST.md, docs/results/axi_cache_pair_seed20261122_default.json), with no call to any model:
  1. the saved file (hash);
  2. the totals and counts the page carries, recounted below with plain loops from the prompt size of every request that the file holds (sums, requests at least 1,024 / 2,048 / 4,096 tokens, the windows of 50 turns);
  3. the arithmetic of the page (computed tokens, computed tokens per verified step, the ratios, the dollars at the typed prices, the share of the warm kept chat);
  4. that the page carries those figures and the reading of the twelve predictions, and that the wording rules hold (no 'proven', no claim about a provider, nothing called open source or a prototype)."""
import hashlib, json, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
FILE = os.path.join(DOCS, "results", "axi_cache_pair_seed20261122_default.json")
PAGE = os.path.join(DOCS, "CACHE_PAIR_TEST.md")
SHA = "c8a1a28ce5777d7f010bbd0a5e06410ad796b488f8afa19383d936db180a468f"


def load(): return json.load(open(FILE))


def test_the_saved_file_is_the_one_the_page_was_written_from():
    assert hashlib.sha256(open(FILE, "rb").read()).hexdigest() == SHA


def test_the_totals_and_counts_are_recounted_from_the_prompt_size_of_every_request():
    d = load(); A = d["A"]; K = d["KC"]; a = A["prompt_tokens_per_step"]; k = K["prompt_tokens_per_step"]
    assert len(a) == A["steps"] == A["calls"] == 400 and len(k) == K["steps"] == K["calls"] == 390
    s = 0
    for x in a: s += x
    assert s == A["prompt_tokens_total"] == 470669
    s = 0
    for x in k: s += x
    assert s == K["prompt_tokens_total"] == 37967655
    for arm, v in ((A, a), (K, k)):
        assert min(v) == arm["prompt_min"] and max(v) == arm["prompt_max"]
        for lim in (1024, 2048, 4096):
            n = 0
            for x in v:
                if x >= lim: n += 1
            assert n == arm["requests_ge_%d" % lim], (lim, n)
    assert (A["prompt_min"], A["prompt_max"], K["prompt_min"], K["prompt_max"]) == (833, 1390, 864, 193411)
    below = [x for x in a + k if x < 4096]
    assert len(below) == 407                                                     # A 400, KC 7 (the page's reading of prediction 5)
    for arm in (A, K):
        assert arm["cache_tokens_total"] == 0 and arm["raw_cache_field"]["positive"] == 0 and arm["raw_cache_field"]["explicit_zero"] == 0
        assert arm["raw_cache_field"]["none"] == arm["calls"] and arm["unparsable_replies"] == 0
    assert A["requests_ge_4096"] == 0 and K["requests_ge_4096"] == 383
    # the windows of 50 turns
    w = d["windows_of_50_turns"]
    for v, key in ((k, "KC_prompt_per_step"), (a, "A_prompt_per_step")):
        got = []
        for i in range(0, len(v), 50):
            chunk = v[i:i + 50]; t = 0
            for x in chunk: t += x
            got.append(round(t / len(chunk)))
        assert got == w[key], (key, got)
    # the one stretch of wrong turns
    assert K["turns_ending_wrong"] == list(range(329, 391)) and K["verified_steps"] == 390 - len(K["turns_ending_wrong"]) == 328
    assert K["first_turn_ending_wrong"] == 329 and A["verified_steps"] == A["replies_exact"] == 400
    assert w["KC_exact_held_state"][6] == "28/50" and 28 == sum(1 for t in range(301, 351) if t not in K["turns_ending_wrong"])
    assert w["KC_exact_held_state"][7] == "0/40"


def test_the_arithmetic_of_the_page():
    d = load(); A = d["A"]; K = d["KC"]; ref = d["reference_test_23_saved_file"]
    ca = A["prompt_tokens_total"] - A["cache_tokens_total"] + A["visible_tokens"] + A["thinking_tokens"]
    ck = K["prompt_tokens_total"] - K["cache_tokens_total"] + K["visible_tokens"] + K["thinking_tokens"]
    assert (ca, ck) == (528196, 37980006)
    pa, pk, pw = ca / A["verified_steps"], ck / K["verified_steps"], ref["K_computed_tokens"] / ref["K_verified_steps"]
    assert round(pa) == 1320 and round(pk) == 115793 and round(pw) == 4646
    assert round(pk / pa, 1) == 87.7 and round(pk / pw, 1) == 24.9 and round(pw / pa, 1) == 3.5
    ua = (A["prompt_tokens_total"] + 4 * (A["visible_tokens"] + A["thinking_tokens"])) / A["verified_steps"]
    uk = (K["prompt_tokens_total"] + 4 * (K["visible_tokens"] + K["thinking_tokens"])) / K["verified_steps"]
    assert round(ua) == 1752 and round(uk) == 115906 and round(uk / ua, 1) == 66.2
    pi, po = d["prices_typed_usd_per_million_tokens"]["input"], d["prices_typed_usd_per_million_tokens"]["output"]
    da = A["prompt_tokens_total"] * pi / 1e6 + (A["visible_tokens"] + A["thinking_tokens"]) * po / 1e6
    dk = K["prompt_tokens_total"] * pi / 1e6 + (K["visible_tokens"] + K["thinking_tokens"]) * po / 1e6
    assert abs(da - A["dollars_at_typed_prices"]) < 1e-6 and abs(dk - K["dollars_at_typed_prices"]) < 1e-6
    assert round(da, 2) == 0.57 and round(dk, 2) == 28.52 and round(da + dk, 2) == 29.09
    assert round(da / A["verified_steps"], 4) == 0.0014 and round(dk / K["verified_steps"], 4) == 0.0870
    assert round((dk / K["verified_steps"]) / (da / A["verified_steps"])) == 61
    assert ref["A_prompt_tokens"] == A["prompt_tokens_total"]                      # the same input as Test 23's arm A
    assert round(100 * ref["K_cache_tokens"] / ref["K_prompt_tokens"], 1) == 97.0


def test_the_page_carries_the_figures_and_the_wording_rules_hold():
    page = open(PAGE, encoding="utf-8").read()
    for s in ("470,669", "37,967,655", "87.7 times", "24.9 times", "3.5 times", "61 times", "328 of 390", "400 of 400", "383 of 390", "1,390", "193,411",
              "115,793", "1,320", "4,646", "Eleven YES and one NO", "Rajnish Choubey", "47271c207ff7edbd8d5abc07f34fb23c21833fd9462657b5730243b978340374"):
        assert s in page, s
    low = page.lower()
    for bad in ("proven", "open source", "open-source", "prototype", "joule"):
        assert bad not in low, bad
    d = load()
    assert d["driver_sha256"] in page

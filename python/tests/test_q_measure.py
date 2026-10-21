"""Engineering support for docs/Q_FROM_SAVED_FILES.md (concept-map item 111): Q = O / M computed from the result files already saved, no GPU and no model.
  1. the arithmetic of one cell, on a made-up cell whose answer is known by hand (O, seconds, tokens, the two Q's, what happens without cached counts, without exact steps, without seconds);
  2. the five saved runs: the same set-up in all five, O equal to what the earlier pages count, the table character by character as it stands on the page;
  3. what the page says the table shows (kept chat against exact state), recomputed;
  4. the page's rules: two Q's and never one mixed number, no floor of M, no capacity multiple, dollars or energy.
What it does NOT show: a floor of M, a price, anything about a cell that was not run."""
import contextlib, copy, io, json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.q_measure as Q
import axi.experiments.provider_side_constrained as C

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results")
PAGE = os.path.join(ROOT, "docs", "Q_FROM_SAVED_FILES.md")
PAGE_SHA256 = "24567000d5c193c4d2284b896b8b31d307e65bd55077611e6b4bfa02ff121e9b"


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


def fake_cell(exact=(True, True, False, True), wall=10.0, rounds=None, **kw):
    log = [dict(t=i + 1, exact=e) for i, e in enumerate(exact)]
    c = dict(arm="S", K=2, wall_seconds=wall, sessions=[dict(log=log[:2]), dict(log=log[2:])], rounds=rounds if rounds is not None else
             [dict(prompt_tokens=300, cached_tokens=100, new_tokens=20), dict(prompt_tokens=200, cached_tokens=150, new_tokens=10)])
    c.update(kw); return c


def test_one_cell_by_hand_the_exact_steps_the_seconds_the_tokens_and_the_two_ratios():
    r = Q.q_row(fake_cell())
    # O = 3 of 4 steps; M in tokens = (300 - 100) + (200 - 150) computed, plus 20 + 10 written = 280; time = 10 s
    assert r == dict(cell="S2", steps=4, exact_steps=3, exact_share=0.75, wall_seconds=10.0, computed_prompt_tokens=250, new_tokens=30, m_tokens=280,
                     exact_per_hour=3600.0 * 3 / 10.0, exact_per_1000_tokens=1000.0 * 3 / 280, seconds_per_exact_step=10.0 / 3, tokens_per_exact_step=280 / 3)
    assert r["exact_per_hour"] == 1080.0 and abs(r["exact_per_1000_tokens"] - 10.714285714285714) < 1e-12
    assert Q.cell_id(fake_cell(label="FK")) == "FK" and Q.cell_id(fake_cell()) == "S2" and Q.cell_id(dict(arm="KC", K=64)) == "KC64"
    # no cached counts in one round: the token figures are not computed (never a guess), the seconds still are
    rr = fake_cell(rounds=[dict(prompt_tokens=300, cached_tokens=None, new_tokens=20), dict(prompt_tokens=200, cached_tokens=150, new_tokens=10)])
    r = Q.q_row(rr); assert r["computed_prompt_tokens"] is None and r["m_tokens"] is None and r["exact_per_1000_tokens"] is None and r["tokens_per_exact_step"] is None and r["exact_per_hour"] == 1080.0 and r["new_tokens"] == 30
    r = Q.q_row(fake_cell(rounds=[])); assert r["m_tokens"] is None and r["exact_per_hour"] == 1080.0 and r["new_tokens"] == 0
    # no exact step: the ratios are 0 and the cost per exact step does not exist
    r = Q.q_row(fake_cell(exact=(False,) * 4)); assert (r["exact_steps"], r["exact_per_hour"], r["exact_per_1000_tokens"], r["seconds_per_exact_step"], r["tokens_per_exact_step"]) == (0, 0.0, 0.0, None, None)
    # no seconds: no rate in time; no steps: no share
    r = Q.q_row(fake_cell(wall=0)); assert r["exact_per_hour"] is None and r["exact_per_1000_tokens"] is not None
    r = Q.q_row(dict(arm="S", K=0, wall_seconds=1.0, sessions=[], rounds=[])); assert r["steps"] == 0 and r["exact_share"] is None and r["exact_steps"] == 0
    # doubling the exact steps at the same cost doubles both ratios; doubling the cost at the same exact steps halves both
    a = Q.q_row(fake_cell()); b = Q.q_row(fake_cell(exact=(True,) * 4)); assert abs(b["exact_per_hour"] / a["exact_per_hour"] - 4 / 3) < 1e-12 and abs(b["exact_per_1000_tokens"] / a["exact_per_1000_tokens"] - 4 / 3) < 1e-12
    c = Q.q_row(fake_cell(wall=20.0)); assert abs(c["exact_per_hour"] / a["exact_per_hour"] - 0.5) < 1e-12 and c["exact_per_1000_tokens"] == a["exact_per_1000_tokens"]
    big = fake_cell(rounds=[dict(prompt_tokens=300, cached_tokens=100, new_tokens=20)] * 2); d = Q.q_row(big); assert d["m_tokens"] == 2 * (200 + 20) and d["exact_per_hour"] == a["exact_per_hour"]
    # a cached count above the prompt is not repaired here: it shows in the arithmetic
    odd = Q.q_row(fake_cell(rounds=[dict(prompt_tokens=100, cached_tokens=100, new_tokens=0)])); assert odd["m_tokens"] == 0 and odd["exact_per_1000_tokens"] is None


FILES = {"first run, no grammar": "axi_provider_side.json", "second run, uncapped grammar": "axi_provider_constrained.json", "third run, capped grammar": "axi_provider_bounded.json", "fourth run, one retry": "axi_provider_retry.json",
         "fifth run, typed requests": "axi_provider_typed.json"}


def test_the_five_saved_runs_have_the_same_setup_o_is_what_the_earlier_pages_count_and_the_files_are_those_the_page_names():
    assert Q.FILES == list(FILES.items()) and [os.path.basename(p) for p in sorted(os.listdir(RES)) if p.startswith("axi_provider_") and p.endswith(".json")] == sorted(list(FILES.values()) + ["axi_provider_long.json"])   # the sixth run's file (jobs of 100 steps) is in the folder and not in the table of the five runs
    rows = Q.all_rows(); assert list(rows) == list(FILES)
    seen = set()
    for title, name in FILES.items():
        d = json.load(open(os.path.join(RES, name)))
        for c in d["cells"]: seen.add((c["engine"]["model"], c["engine"]["version"], c["engine"]["gpu"], c["engine"]["capacity_tokens"], c["engine"]["capacity_source"], c["engine"]["kind"], c["policy"], c["steps"], c["seed"], c["hint_rules"]))
    assert seen == {("Qwen/Qwen2.5-7B-Instruct-AWQ", "0.31.0", "Tesla T4", 80000, "override", "vllm", "repair_m6", 30, 20261017, False)}, seen
    for title, name in FILES.items():
        d = json.load(open(os.path.join(RES, name))); assert [r["cell"] for r in rows[title]] == [Q.cell_id(c) for c in d["cells"]]
        for r, c in zip(rows[title], d["cells"]):
            S = C.summarize_cell2(c)                                                                                    # the earlier pages' own count
            assert r["exact_steps"] == S["exact_steps"] and r["steps"] == S["steps_done"] and abs(r["exact_share"] - S["exact_share"]) < 1e-12 and r["wall_seconds"] == c["wall_seconds"], (title, r["cell"])
            assert r["m_tokens"] is not None and r["m_tokens"] > 0 and r["computed_prompt_tokens"] + r["new_tokens"] == r["m_tokens"]
            assert abs(r["exact_per_hour"] - 3600 * r["exact_steps"] / r["wall_seconds"]) < 1e-9 and abs(r["exact_per_1000_tokens"] - 1000 * r["exact_steps"] / r["m_tokens"]) < 1e-9
            assert abs(r["seconds_per_exact_step"] * r["exact_steps"] - r["wall_seconds"]) < 1e-6 and abs(r["tokens_per_exact_step"] * r["exact_steps"] - r["m_tokens"]) < 1e-6
    # figures quoted by the earlier pages and by the third, fourth and fifth runs' results
    by = {t: {r["cell"]: r for r in rs} for t, rs in rows.items()}; third = by["third run, capped grammar"]; fourth = by["fourth run, one retry"]
    assert (third["S64"]["exact_steps"], round(third["S64"]["wall_seconds"], 1), round(third["S64"]["exact_per_hour"]), third["S64"]["m_tokens"]) == (1867, 217.4, 30914, 228935) and round(third["KC8"]["exact_per_hour"]) == 5694
    assert (fourth["H0"]["exact_steps"], round(fourth["H0"]["exact_per_hour"]), fourth["H1"]["exact_steps"], round(fourth["H1"]["exact_per_hour"])) == (1884, 31359, 1884, 30951) and round(fourth["H0"]["exact_per_1000_tokens"], 1) == 8.3
    assert [round(100 * third[k]["exact_share"], 1) for k in ("S8", "S32", "S64", "KC8")] == [88.3, 94.6, 97.2, 100.0]
    fifth = by["fifth run, typed requests"]; assert list(fifth) == ["F0", "FK", "FKA", "FKT", "FN", "SK", "SN"]
    assert [(fifth[k]["exact_steps"], round(fifth[k]["wall_seconds"], 1), round(fifth[k]["exact_per_hour"]), fifth[k]["m_tokens"], round(fifth[k]["exact_per_1000_tokens"], 1)) for k in fifth] == [
        (1833, 224.2, 29437, 228379, 8.0), (1835, 223.5, 29560, 228374, 8.0), (1920, 342.3, 20191, 356091, 5.4), (1839, 244.8, 27048, 249814, 7.4), (1920, 223.8, 30882, 228793, 8.4), (1906, 224.8, 30529, 230036, 8.3), (1920, 224.8, 30747, 230050, 8.3)]
    same = [fifth[k] for k in ("F0", "FK", "FN", "SK", "SN")]                                                       # the five cells that did the same work: the page's sentence about them
    assert (round(min(r["wall_seconds"] for r in same), 1), round(max(r["wall_seconds"] for r in same), 1), min(r["m_tokens"] for r in same), max(r["m_tokens"] for r in same)) == (223.5, 224.8, 228374, 230050)
    assert (round(min(r["exact_per_hour"] for r in same)), round(max(r["exact_per_hour"] for r in same)), min(r["exact_steps"] for r in same), max(r["exact_steps"] for r in same)) == (29437, 30882, 1833, 1920)
    assert fifth["FKA"]["exact_per_hour"] < fifth["FKT"]["exact_per_hour"] < min(r["exact_per_hour"] for r in same) and fifth["FKA"]["m_tokens"] > fifth["FKT"]["m_tokens"] > max(r["m_tokens"] for r in same) and fifth["FKA"]["wall_seconds"] > fifth["FKT"]["wall_seconds"] > max(r["wall_seconds"] for r in same)
    # an unreadable or missing file is skipped, not guessed
    with tempfile.TemporaryDirectory() as d:
        json.dump(json.load(open(os.path.join(RES, "axi_provider_retry.json"))), open(os.path.join(d, "axi_provider_retry.json"), "w")); got = Q.all_rows(d)
        assert list(got) == ["fourth run, one retry"] and got["fourth run, one retry"] == rows["fourth run, one retry"] and Q.all_rows(os.path.join(d, "none")) == {}


def test_the_table_on_the_page_is_the_tables_output_character_by_character_and_has_exactly_two_ratios():
    page = open(PAGE).read(); table = Q.render_table(Q.all_rows()).rstrip("\n")
    assert table in page and page.count("| cell | steps | exact steps O | exact share | wall seconds | exact steps per GPU hour | tokens computed + written | exact steps per 1,000 tokens |") == 5
    for line in table.splitlines():
        if line.startswith("| ") and not line.startswith("| cell"): assert line in page
    lines = [l for l in table.splitlines() if l.startswith("| ") and not l.startswith("| cell")]
    assert len(lines) == 6 + 6 + 4 + 5 + 7 and "| S64 | 1,920 | 1,867 | 97.2 % | 217.4 | 30,914 | 228,935 | 8.2 |" in lines and "| KC8 | 240 | 240 | 100.0 % | 151.7 | 5,694 | 49,655 | 4.8 |" in lines
    assert "| H0 | 1,920 | 1,884 | 98.1 % | 216.3 | 31,359 | 225,703 | 8.3 |" in lines and "| KN8 | 240 | 240 | 100.0 % | 1646.3 | 525 | 793,719 | 0.3 |" in lines and "| KC64 | 1,915 | 1,915 | 100.0 % | 10330.5 | 667 | 5,174,334 | 0.4 |" in lines
    assert "| FKA | 1,920 | 1,920 | 100.0 % | 342.3 | 20,191 | 356,091 | 5.4 |" in lines and "| FN | 1,920 | 1,920 | 100.0 % | 223.8 | 30,882 | 228,793 | 8.4 |" in lines and "| F0 | 1,920 | 1,833 | 95.5 % | 224.2 | 29,437 | 228,379 | 8.0 |" in lines
    for l in lines: assert len(l.split("|")) == 10                                                                    # eight columns: no ninth, mixed number
    # a cell without cached counts shows "-", never a guess
    rows = {"x": [Q.q_row(fake_cell(rounds=[dict(prompt_tokens=300, cached_tokens=None, new_tokens=20)]))]}
    assert Q.render_table(rows).splitlines()[-1] == "| S2 | 4 | 3 | 75.0 % | 10.0 | 1,080 | - | - |"
    # the command line prints the same table
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
    out = subprocess.run([sys.executable, "-m", "axi.experiments.q_measure"], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, ".."))
    assert out.returncode == 0 and out.stdout == Q.render_table(Q.all_rows()) + "\n"


def test_what_the_page_says_the_table_shows_about_the_kept_chat_is_what_the_files_give():
    rows = Q.all_rows(); page = open(PAGE).read(); pairs = []
    for title, rs in rows.items():
        by = {r["cell"]: r for r in rs}
        for k in (8, 32, 64):
            if f"S{k}" in by and f"KC{k}" in by: pairs.append((title, k, by[f"S{k}"], by[f"KC{k}"]))
    assert [(t, k) for t, k, _, _ in pairs] == [("first run, no grammar", 8), ("first run, no grammar", 32), ("second run, uncapped grammar", 8), ("second run, uncapped grammar", 32), ("second run, uncapped grammar", 64), ("third run, capped grammar", 8)]
    for t, k, s, kc in pairs:
        assert kc["exact_per_hour"] < s["exact_per_hour"] and kc["exact_per_1000_tokens"] < s["exact_per_1000_tokens"], (t, k)
    kc = lambda k: sorted({round(c["exact_per_1000_tokens"], 1) for t, kk, s, c in pairs if kk == k}); st = lambda k: sorted({round(s["exact_per_1000_tokens"], 1) for t, kk, s, c in pairs if kk == k})
    assert (kc(8), st(8), kc(32), st(32), kc(64), st(64)) == ([4.8], [5.7, 6.6], [0.4], [6.3, 7.6], [0.4], [8.2])
    assert "at 8 jobs 4.8 per 1,000 tokens against 5.7 to 6.6, at 32 jobs 0.4 against 6.3 and 7.6, at 64 jobs 0.4 against 8.2 (the second run)" in page
    ex = {(t, k): (round(100 * s["exact_share"], 1), round(100 * c["exact_share"], 1)) for t, k, s, c in pairs}
    assert ex[("first run, no grammar", 32)] == (78.2, 97.3) and ex[("second run, uncapped grammar", 32)] == (94.6, 100.0) and ex[("second run, uncapped grammar", 64)] == (97.2, 100.0)
    assert "(97.3 %, 100.0 % and 100.0 % against 78.2 %, 94.6 % and 97.2 %)" in page
    # the 64-job exact-state cell of the third run has no kept-chat cell at 64 jobs in that run
    third = {r["cell"] for r in rows["third run, capped grammar"]}; assert third == {"S8", "KC8", "S32", "S64"} and "KC64" in {r["cell"] for r in rows["second run, uncapped grammar"]}
    assert "the 64-job `S` cell therefore has no kept-chat cell at 64 jobs to stand next to in that run (the second run does, `KC64`, and its row is above)" in page


def test_the_page_is_pinned_by_its_sha256_so_that_a_changed_word_is_seen_and_a_changed_table_needs_a_new_run_of_the_checks():
    import hashlib
    page = open(PAGE).read(); assert hashlib.sha256((page.rstrip("\n") + "\n").encode()).hexdigest() == PAGE_SHA256, "the page changed: read it, then re-pin it here"


def test_the_page_states_its_definitions_and_its_limits_and_none_of_the_words_it_refuses():
    page = open(PAGE).read()
    assert page.startswith("# Q = O / M from the saved files (computed 20 Oct 2026, no GPU)\n\nConcept-map item 111.") and "python/axi/experiments/q_measure.py" in page and "python/tests/test_q_measure.py" in page
    for needle in ("the number of steps after which the state the job held equals the oracle's", "**every round of the engine in it** (a retry round, if the cell has one, is inside it)", "the prompt tokens the engine had to compute (prompt tokens minus the tokens served from its cache, over every request) plus the tokens it wrote",
                   "*exact steps per GPU hour* = 3,600 x O / wall seconds, and *exact steps per 1,000 tokens* = 1,000 x O / M in tokens", "They are never added or averaged into one number", "no weight has been measured, so none is invented",
                   "N exact steps per GPU hour against M, at these exactness levels, on this setup", "The exactness is always stated with the Q", "Compare like with like.", "A figure that divides a 64-job `S` cell by an 8-job `KC` cell mixes two things",
                   "**No floor of M is shown.**", "\"Q grows without limit\" is not a statement this page supports.", "**No capacity multiple, no count of GPUs, no dollars, no energy.**", "Energy was never measured", "**Not a claim about any provider**",
                   "Those summaries are second-hand (they are not Sir's own articles)", "python -m axi.experiments.q_measure", "`KN` keeps it with the cache emptied before every round (what a plain engine pays)", "Qwen/Qwen2.5-7B-Instruct-AWQ", "vLLM 0.31.0", "a Tesla T4, a cache of 80,000 tokens (a block override), 30-step jobs from seed 20261017, the policy `repair_m6`",
                   "**What the fifth-run block shows.**", "so what separates their exact steps per GPU hour (29,437 to 30,882) is the number of exact steps (1,833 to 1,920), not the time", "(`F0`, `FK`, `FN`, `SK`, `SN`: 223.5 to 224.8 seconds, 228,374 to 230,050 tokens computed and written)",
                   "`FKA` (the line at the top and the end) has all 1,920 steps exact, 20,191 exact steps per GPU hour and 5.4 per 1,000 tokens, against 1,835 exact, 29,560 and 8.0 for `FK`", "`FKT` (the line at the end only) has 1,839 exact, 27,048 and 7.4",
                   "A cell with fewer exact steps per hour can be the more exact one, and the exactness is read beside the Q.", "test 132 checks that the Result's block and the one above are the same text", "All five runs record the same one:", "the fourth run and the fifth run were separate Colab sessions"):
        assert needle in page, needle
    low = page.lower()
    for bad in ("100 percent", "proven", "saves", "saved gpus", "x faster", "times faster", "multi-million", "per dollar", "$"): assert bad not in low, bad
    assert "5.4 times" not in low
    assert page.count("×") == 0 and "capacity" in low and low.count("dollars") == 1 and low.count("energy") == 2
    # a last check that the page does not contain a single mixed Q: no column or sentence defines one number from seconds and tokens together
    assert "M = time + lambda x tokens" in page and low.count("lambda") == 1


def test_the_module_is_what_it_says_it_is_and_reads_nothing_it_was_not_given():
    src = open(Q.__file__).read(); assert "import json, os" in src and "import requests" not in src and "import torch" not in src and "vllm" not in src.replace("vLLM", "").replace("(the `exact` flag of the log)", "")
    assert Q.RESULTS and os.path.samefile(Q.RESULTS, RES)
    # the arithmetic does not touch the cell it is given
    c = fake_cell(); before = copy.deepcopy(c); Q.q_row(c); assert c == before
    # the table of a cell list is a function of the cells only
    a = Q.render_table({"t": [Q.q_row(fake_cell())]}); b = Q.render_table({"t": [Q.q_row(fake_cell())]}); assert a == b and a.startswith("**t**\n\n| cell |") and a.endswith("|\n")


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

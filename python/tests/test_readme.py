"""Implements the front page (README.md, 14 Oct 2026): every figure the front page quotes is recomputed here from the saved run files, so the page cannot drift away from them. It checks that
  1. every link in README.md and every folder or file it names in backticks exists in the repository, the two links in docs/RFC-001.md (the old front page, kept under a note) resolve, and the old text is still in that file;
  2. every figure on the front page follows from the saved files of the 4,000-step run and the 400-step contrast (steps, tokens, replies right from a correct state, the 46 slips, prompt size and speed from the first tenth to the last,
     steps exactly right and corrupted per policy, how long a wrong state lasted, what each arm spent and what one step with the right state cost, and how many of the fourteen reading rules said yes, NO and not run), the table is
     built from the data and compared with the page character for character, and the figures taken from earlier pages (the 3B and 7B lines) occur in docs/VERIFIED.md;
  3. the front page says nothing the pages do not say: no market, company or energy figure (the word energy occurs once, in the list of what was not measured), no phone-like number, no e-mail address, the credit to Rajnish Choubey is there
     and says the assistant is a tool and not an author or owner, and the limits are stated;
  4. the check notices a doctored front page (a changed figure, a missing link, a removed limit).
What it does NOT show: that a reader will find the page clear, or anything about the results beyond what docs/MILLION_TOKEN_TEST_HARD.md, docs/VERIFIED.md and docs/EXPERIMENT_RESULTS.md measure."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import million_hard_result as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def links_and_paths(text):
    links = [l.split("#")[0] for l in re.findall(r"\]\(([^)]+)\)", text)]
    paths = re.findall(r"`((?:docs|python|crates)/[A-Za-z0-9_./\-]+/?)`", text)
    return [l for l in links if l and not l.startswith("http")], paths


def check_links(text, rel_dir=""):
    for u in re.findall(r"https?://[^\s)\"`]+", text):
        assert u.startswith("https://github.com/gaurshalendra2412/axi-runtime"), ("address", u)
    links, paths = links_and_paths(text)
    for l in links:
        assert os.path.exists(os.path.join(ROOT, os.path.normpath(os.path.join(rel_dir, l)))), ("missing link", l)
    for p in paths:
        assert os.path.exists(os.path.join(ROOT, p)), ("missing path", p)
    return len(links), len(paths)


def facts():
    run1, con = R.load(), R.load_contrast()
    a, ac, cost = R.analyse(run1), R.analyse_contrast(run1, con), R.analyse_cost(run1, con)
    kd = a["kinds"]; tot = lambda k: sum(v[k] for v in kd.values())
    right = tot("same_text") + tot("other_order")
    ten = a["tenths"]
    page = read("docs/MILLION_TOKEN_TEST_HARD.md"); result = page[page.index("\n\n---\n\n# Result"):]
    rules = re.findall(r"^\d+\. (yes|NO|not run) ", result, re.M)
    bl, g = ac["blind"], ac["gate"]
    return dict(meta=run1["meta"], page_result=result, a=a, ac=ac, cost=cost, right=right, steps=tot("requests"), slips=tot("other_items"),
                prompt_ratio=ten[-1]["prompt"] / ten[0]["prompt"], speed_ratio=ten[-1]["seconds"] / ten[0]["seconds"],
                yes=rules.count("yes"), no=rules.count("NO"), notrun=rules.count("not run"),
                lasts=[y - x + 1 for x, y in bl["off"]], gate_open=g["off"][-1])


def check_numbers(text):
    f = facts(); a, ac, cost = f["a"], f["ac"], f["cost"]
    assert f["steps"] == 4000 and a["tokens"]["calls"] == 4000 and round(a["tokens"]["total"] / 1e6, 1) == 3.7
    assert f["right"] == 3954 and round(100 * f["right"] / f["steps"], 2) == 98.85 and f["slips"] == 46 == a["only_left_out"] == 4000 - f["right"]
    assert sum(v["off"] for v in a["kinds"].values()) == 0 and ac["scope_m7"]["exact"] == 400 and ac["scope_m7"]["corrupted"] == 0
    assert round(f["prompt_ratio"], 2) == 1.16 and round(f["speed_ratio"], 2) == 0.94
    assert (f["yes"], f["no"], f["notrun"]) == (12, 1, 1) and round(a["dollars"]["total"], 2) == 4.95
    b, g = ac["blind"], ac["gate"]
    assert (b["exact"], b["corrupted"], g["exact"], g["corrupted"]) == (85, 223, 84, 0) and (min(f["lasts"]), max(f["lasts"])) == (36, 150)
    assert f["gate_open"] == (210, 400) and b["off"][-1][1] == 400
    assert f["meta"]["model"] == "gemini-3.8-flash" and f["meta"]["hint_rules"] is False and f["meta"]["steps"] == 4000
    assert "completed by a deterministic repair 46 times out of 46" in f["page_result"]
    must = [
        "One model (Gemini 3.8 Flash)",
        f"A fixed, rule-based repair completed all {f['slips']}.",
        "one run of **4,000 requests in a row**, 3.7 million tokens sent in all",
        "it was exactly right at 4,000 of 4,000 steps** and never corrupted",
        f"(1.16 times from the first tenth of the run to the last) and the speed did not fall (0.94 times)",
        "its reply was right 3,954 times out of 4,000 (98.85 percent). All 46 slips were of one kind",
        f"left the state exactly right on {b['exact']} steps of {b['steps']} ({round(100 * b['exact'] / b['steps'])} percent) and corrupted it on {b['corrupted']} ({round(100 * b['corrupted'] / b['steps'])} percent)",
        f"the state stayed wrong for {min(f['lasts'])} to {max(f['lasts'])} steps at a time (the last stretch was still going when the 400 steps ended)",
        f"exactly right on {g['exact']} of {g['steps']}, and off the right state on every step from {f['gate_open'][0]} to {f['gate_open'][1]}",
        "twelve passed, one read NO and says so on the page, and one was not run",
        f"(the 4,000-step run cost ${a['dollars']['total']:.2f} at the prices above)",
        "On the first 400 requests of the same stream",
        "With the gate, corruption was 0.",
    ]
    nodes = [round(x["nodes"]) for x in a["tenths"]]
    assert (min(nodes), max(nodes)) == (17, 22) and f"a service graph of {min(nodes)} to {max(nodes)} services on average" in text
    assert all(x == "4,000-step" for x in re.findall(r"\d,\d{3}-step", text)) and "4,000-step" in text
    for m in must:
        assert m in text, ("figure not on the page", m)
    rows = [f"| First 400 requests | Steps with the right state | Spent | Cost of one step with the right state |",
            f"| AXI (repair + scope filter) | {cost['scope_m7']['exact']} | ${cost['scope_m7']['dollars']:.2f} | ${cost['scope_m7']['dollars_per_exact']:.4f} |",
            f"| Replies applied as written | {cost['blind']['exact']} | ${cost['blind']['dollars']:.2f} | ${cost['blind']['dollars_per_exact']:.4f} ({cost['blind']['times_framework']:.1f} times as much) |",
            f"| Gate only (refuses bad replies) | {cost['gate']['exact']} | ${cost['gate']['dollars']:.2f} | ${cost['gate']['dollars_per_exact']:.4f} ({cost['gate']['times_framework']:.1f} times as much) |"]
    for r in rows:
        assert r in text, ("table row differs from the data", r)
    assert "Priced at $%.2f per million tokens in and $%.2f out" % (R.PRICE_IN, R.PRICE_OUT) in text
    cr = read("docs/CREDITS.md")
    for who, year in [("Ehrig, Pfender and Schneider", "1973"), ("Lack and Sobocinski", "2004"), ("Willard and Louf", "2023")]:
        assert f"{who}, {year}" in text and f"{who.replace(' and Schneider', ', Schneider')} {year}" in cr, (who, year)
    assert "gate -> 0% corruption" in read("docs/VERIFIED.md")
    v = read("docs/VERIFIED.md")
    for frag in ["corrupted the graph on 120/120 delete-with-dependents tasks", "(120/120 delete_dep, 240/240 overall)", "corrupted in 12 of 12 episodes, the gate in 0 of 360 steps"]:
        assert frag in v, ("not in VERIFIED.md", frag)
    for frag in ["120 of 120", "(240 of 240)", "12 of 12 episodes", "0 of 360 steps", "Qwen2.5-3B, four seeds, 60 tasks each", "Qwen2.5-7B, 30 requests in a row"]:
        assert frag in text, ("earlier-result figure not on the page", frag)


def check_words(text):
    low = text.lower()
    for w in ["billion", "trillion", "hyperscaler", "industry", "market", "profit", "openai", "anthropic", "google", "kwh", "watt", "carbon", "valuation", "frontier"]:
        assert w not in low, ("word the pages do not support", w)
    assert low.count("energy") == 1 and "or energy" in low
    assert not re.findall(r"\d{8,}", text) and not re.findall(r"\+?\d[\d \-()]{8,}\d", text.replace(",", "")), "phone-like digits"
    assert not re.findall(r"\S+@\S+\.\S+", text), "e-mail address"
    for must in ["The idea, the framework, the architecture and the structural maps of AXI are by **Rajnish Choubey**",
                 "The assistant is a tool here: it is not an author and it owns nothing in this work",
                 "They are his to accept or reject",
                 "**What this does not show.**", "one sampled run on one model and one job", "the model was never told it",
                 "We did not measure a chat that keeps its history, an agent that reads its state back and corrects itself, or energy",
                 "not a product", "the Rust crates are a partly wired sketch"]:
        assert must in text, ("missing", must)
    for h in ["## The result that matters most", "## How the tests are done", "## Run it yourself", "## What is where", "## Start reading here", "## Status", "## Credits"]:
        assert h in text, ("missing section", h)


def test_every_link_and_path_the_front_page_names_exists_and_the_old_front_page_is_kept():
    n_l, n_p = check_links(read("README.md"))
    assert n_l >= 8 and n_p >= 8
    rfc = read("docs/RFC-001.md")
    assert check_links(rfc, "docs")[0] == 2
    assert rfc.startswith("> **Note (14 Oct 2026).**") and "kept below exactly as it was" in rfc
    for old in ["RFC-001: The AXI Architecture", "Dangling Edge Condition: Rejects node deletions", "Copy-on-Write Shadow Paging", "The theoretical origin of this work is Rajnish Choubey"]:
        assert old in rfc, old
    assert os.path.isdir(os.path.join(ROOT, "docs", "results")) and os.path.isfile(os.path.join(ROOT, "python", "tests", "run_all.py"))


def test_every_figure_on_the_front_page_follows_from_the_saved_run_files():
    check_numbers(read("README.md"))


def test_the_front_page_claims_nothing_the_pages_do_not_and_credits_the_idea_to_its_author():
    check_words(read("README.md"))


def test_the_commands_on_the_front_page_are_the_ones_that_exist():
    t = read("README.md")
    assert "PYTHONPATH=python python3 python/tests/run_all.py" in t and "PYTHONPATH=. python3 tests/test_million_hard_result.py" in t and "cargo test --workspace" in t
    assert "cargo test --workspace" in read(".github/workflows/ci.yml")
    assert os.path.isfile(os.path.join(ROOT, "python", "tests", "test_million_hard_result.py"))
    assert "skip" in read("python/tests/run_all.py").lower() and "torch" in read("python/tests/run_all.py")


def test_a_doctored_front_page_is_noticed():
    t = read("README.md")
    def fails(fn, text):
        try: fn(text)
        except AssertionError: return True
        return False
    assert not fails(check_numbers, t) and not fails(check_words, t)
    for old, new in [("| Replies applied as written | 85 |", "| Replies applied as written | 95 |"), ("$0.0052 (4.5 times as much)", "$0.0052 (4.0 times as much)"), ("98.85 percent", "99.85 percent"),
                     ("3.7 million tokens", "4.7 million tokens"), ("twelve passed, one read NO", "thirteen passed, one read NO"), ("36 to 150 steps", "36 to 100 steps"), ("0 of 360 steps", "0 of 300 steps"),
                     ("$0.75 per million tokens in", "$0.50 per million tokens in")]:
        assert old in t and fails(check_numbers, t.replace(old, new, 1)), old
    for bad in ["a market worth billions", "Energy use fell", "call 98765 43210", "write to someone@example.com"]:
        assert fails(check_words, t + "\n" + bad), bad
    assert fails(check_words, t.replace("**What this does not show.**", "**Note.**")) and fails(check_words, t.replace("is not an author", "is an author"))
    assert fails(lambda x: check_links(x), t.replace("docs/VERIFIED.md", "docs/NOPE.md", 1))


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

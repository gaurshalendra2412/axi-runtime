"""Implements the front page (README.md, 14 Oct 2026) and its pictures (docs/img/, drawn by python/tools/readme_figures.py): every figure the front page quotes or draws is recomputed here from the saved run files, so the page cannot drift away from them. It checks that
  1. every link in README.md and every folder or file it names in backticks exists in the repository, the two links in docs/RFC-001.md (the old front page, kept under a note) resolve, and the old text is still in that file;
  2. every figure on the front page follows from the saved files of the 4,000-step run and the 400-step contrast (steps, tokens, replies right from a correct state, the 46 slips, prompt size and speed from the first tenth to the last,
     steps exactly right and corrupted per policy, how long a wrong state lasted, what each arm spent and what one step with the right state cost, and how many of the fourteen reading rules said yes, NO and not run), the table is
     built from the data and compared with the page character for character, and the figures taken from earlier pages (the 3B and 7B lines) occur in docs/VERIFIED.md;
  3. the front page says nothing the pages do not say: no market, company or energy figure (the word energy occurs once, in the list of what was not measured), no phone-like number, no e-mail address, the credit to Rajnish Choubey is there
     and says the assistant is a tool and not an author or owner, and the limits are stated;
  4. the pictures committed in docs/img are, byte for byte, what the drawing code makes from the saved files (so every number drawn is checked), the colours are the validated palette's first three slots, the share card is 1280 x 640,
     and every picture the front page shows is there;
  5. the check notices a doctored front page (a changed figure, a missing link, a removed limit) and a doctored picture.
What it does NOT show: that a reader will find the page clear, or anything about the results beyond what docs/MILLION_TOKEN_TEST_HARD.md, docs/VERIFIED.md and docs/EXPERIMENT_RESULTS.md measure."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import functools, struct
from axi.experiments import million_hard_result as R
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import readme_figures as RF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def links_and_paths(text):
    links = [l.split("#")[0] for l in re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'(?:src|srcset|href)="([^"]+)"', text)]
    paths = re.findall(r"`((?:docs|python|crates)/[A-Za-z0-9_./\-]+/?)`", text)
    return [l for l in links if l and not l.startswith("http")], paths


def check_links(text, rel_dir=""):
    for u in re.findall(r"https?://[^\s)\"`<>]+", text):
        assert u.startswith("https://github.com/gaurshalendra2412/axi-runtime"), ("address", u)
    links, paths = links_and_paths(text)
    for l in links:
        assert os.path.exists(os.path.join(ROOT, os.path.normpath(os.path.join(rel_dir, l)))), ("missing link", l)
    for p in paths:
        assert os.path.exists(os.path.join(ROOT, p)), ("missing path", p)
    return len(links), len(paths)


@functools.lru_cache(maxsize=None)
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
    g400 = RF.facts()
    assert (g400["replies_right"], g400["slips"], g400["steps"]) == (393, 7, 400) and g400["new_offsets"] == [16, 132, 210, 361] and g400["wrong_steps"] == 315 == sum(f["lasts"])
    assert [x for x in g400["series"]["scope_m7"] if x] == [] and max(g400["series"]["blind"]) == 5 and max(g400["series"]["gate"]) == 8
    pct = lambda a_, b_: round(100 * a_ / b_)
    offs = ", ".join(str(x) for x in g400["new_offsets"][:-1]) + f" and {g400['new_offsets'][-1]}"
    alt1 = (f'alt="Line chart over the first 400 requests. AXI is at zero items off the right state on every step. Replies applied as written go wrong at steps {offs} and are off the right state on '
            f'{g400["wrong_steps"]} of 400 steps. A gate that only refuses bad replies is never corrupted but is off the right state from step {f["gate_open"][0]} to the end."')
    alt2 = (f'alt="Bar chart of what one step with the right state cost over the first 400 requests: AXI {cost["scope_m7"]["dollars_per_exact"]:.4f} dollars, replies applied as written '
            f'{cost["blind"]["dollars_per_exact"]:.4f} dollars ({cost["blind"]["times_framework"]:.1f} times as much), gate only {cost["gate"]["dollars_per_exact"]:.4f} dollars ({cost["gate"]["times_framework"]:.1f} times as much)."')
    alt0 = (f'alt="Three numbers. {f["steps"]:,} of {f["steps"]:,} steps with exactly the right state, in one run on Gemini 3.8 Flash with {a["tokens"]["total"] / 1e6:.1f} million tokens. '
            f'A step with the right state cost {cost["blind"]["times_framework"]:.1f} times as much when the replies were applied as written, on the first 400 requests. '
            f'{f["slips"]} of {f["slips"]} slips, where the model left out a service\'s connections, repaired by a fixed rule."')
    must = [
        '<h1 align="center">AXI Runtime</h1>', alt1, alt2,
        f"**Why a model that is right {pct(g400['replies_right'], 400)}% of the time can leave the state right on only {pct(b['exact'], 400)}% of the steps:**",
        f"The model was right {pct(g400['replies_right'], 400)}% of the time. Applied as written, only {pct(b['exact'], 400)}% of its steps ended with the right state.",
        "With AXI between the model and the state: every one of them.",
        "First 400 requests of the same stream, three ways of keeping the state. The whole run was 4,000 requests.",
        alt0,
        f"In these 400 requests the model slipped {g400['slips']} times ({g400['replies_right']} replies right, {pct(g400['replies_right'], 400)}%). Four of those slips, marked on the chart, each started a stretch in which the state was wrong, and together those stretches covered {g400['wrong_steps']} of the 400 steps.",
        f"but its state was off the right one from step {f['gate_open'][0]} to the end. AXI's line is flat at zero.",
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
    assert not re.findall(r"\d{8,}", text) and all(sum(ch.isdigit() for ch in m) < 10 for m in re.findall(r"\+?\d[\d \-()]{8,}\d", text.replace(",", ""))), "phone-like digits"
    assert not re.findall(r"\S+@\S+\.\S+", text), "e-mail address"
    for must in ["The idea, the framework, the architecture and the structural maps of AXI are by **Rajnish Choubey**",
                 "The assistant is a tool here: it is not an author and it owns nothing in this work",
                 "They are his to accept or reject",
                 "**What this does not show.**", "one sampled run on one model and one job", "the model was never told it",
                 "We did not measure a chat that keeps its history, an agent that reads its state back and corrects itself, or energy",
                 "not a product", "the Rust crates are a partly wired sketch"]:
        assert must in text, ("missing", must)
    for h in ["## What AXI is", "## The run, in plain words", "## What it cost", "## How the tests are done", "## Run it yourself", "## What is where", "## Start reading here", "## Status", "## Credits"]:
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
    assert "PYTHONPATH=python python3 python/tools/readme_figures.py docs/img" in t and "PYTHONPATH=python python3 python/tests/run_all.py" in t and "PYTHONPATH=. python3 tests/test_million_hard_result.py" in t and "cargo test --workspace" in t
    assert "cargo test --workspace" in read(".github/workflows/ci.yml")
    assert os.path.isfile(os.path.join(ROOT, "python", "tests", "test_million_hard_result.py"))
    assert "skip" in read("python/tests/run_all.py").lower() and "torch" in read("python/tests/run_all.py")


VALIDATED = {"light": ("#2a78d6", "#eb6834", "#1baf7a"), "dark": ("#3987e5", "#d95926", "#199e70")}        # slots 1 to 3 of the validated palette; light adjacent and all-pairs, dark adjacent and all-pairs, passed the six checks (aqua on light is the one contrast warning, relieved by the labels and the table)


def png_size(path):
    with open(path, "rb") as fh: b = fh.read(24)
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", b[16:24])


def check_pictures(drawn, read_file):
    for name, text in drawn.items():
        assert read_file(name) == text, ("picture differs from what the code draws from the saved files", name)


def test_the_pictures_are_what_the_code_draws_from_the_saved_files_and_say_what_the_page_says():
    f = RF.facts(); drawn = RF.files(f)
    assert sorted(drawn) == ["cost-per-right-step-dark.svg", "cost-per-right-step-light.svg", "headline-distance-dark.svg", "headline-distance-light.svg", "headline-numbers-dark.svg", "headline-numbers-light.svg", "social-preview.svg"]
    read_img = lambda name: open(os.path.join(ROOT, "docs", "img", name), encoding="utf-8", newline="").read()
    check_pictures(drawn, read_img)
    assert png_size(os.path.join(ROOT, "docs", "img", "social-preview.png")) == (1280, 640)
    for mode in ("light", "dark"):                                                                                  # the palette is the validated one, one slot per way of keeping the state, in both modes
        assert (RF.THEMES[mode]["scope_m7"], RF.THEMES[mode]["blind"], RF.THEMES[mode]["gate"]) == VALIDATED[mode] and RF.ORDER == ("scope_m7", "blind", "gate")
        for name in (f"headline-distance-{mode}.svg", f"cost-per-right-step-{mode}.svg", f"headline-numbers-{mode}.svg"):
            for hexv in (VALIDATED[mode][:2] if name.startswith("headline-numbers") else VALIDATED[mode]): assert hexv in drawn[name], (name, hexv)       # the numbers strip uses the first two ways of keeping the state only
            other = "dark" if mode == "light" else "light"
            for hexv in VALIDATED[other]: assert hexv not in drawn[name], (name, hexv)
    d = drawn["headline-distance-light.svg"]                                                                         # what the headline picture says, from the data
    assert "AXI: 0 items off at every step" in d and "exactly right on 85 of 400 steps" in d and "exactly right on 84 of 400 steps" in d and "at most 5 items off" in d and "at most 8 items off" in d
    assert "from steps 16, 132, 210, 361" in d and "The model slipped on 7 of the 400 requests" in d and "AXI · 0" in d and "As written · 2" in d and "Gate only · 5" in d
    assert [f["series"][k][-1] for k in RF.ORDER] == [0, 2, 5] and f["wrong_steps"] == 315 and f["steps"] == 400
    nm = drawn["headline-numbers-dark.svg"]
    for frag in ("4,000 / 4,000", "4.5×", "46 / 46", "3.7 million tokens", "steps with exactly the right state", "what a right step cost without AXI", "slips repaired by a fixed rule", "as written, first 400 requests"): assert frag in nm, frag
    c = drawn["cost-per-right-step-dark.svg"]
    for frag in ("$0.0012", "$0.0052", "$0.0057", "4.5×", "4.9×", "$0.75 in and $3.75 out"): assert frag in c, frag
    sp = drawn["social-preview.svg"]
    for frag in ("4,000 / 4,000", "4.5×", "3.7 million tokens", "Gemini 3.8 Flash"): assert frag in sp, frag
    assert "billion" not in (c + d + sp).lower() and "energy" not in (c + d + sp).lower()


def test_every_picture_the_front_page_shows_is_in_the_repository_and_each_has_a_text_description():
    t = read("README.md")
    srcs = re.findall(r'(?:src|srcset)="(docs/img/[^"]+)"', t)
    assert sorted(set(srcs)) == ["docs/img/cost-per-right-step-dark.svg", "docs/img/cost-per-right-step-light.svg", "docs/img/headline-distance-dark.svg", "docs/img/headline-distance-light.svg", "docs/img/headline-numbers-dark.svg", "docs/img/headline-numbers-light.svg"]
    for sr in srcs: assert os.path.isfile(os.path.join(ROOT, sr)), sr
    assert t.count("<picture>") == 3 and t.count("prefers-color-scheme: dark") == 3
    alts = re.findall(r'<img alt="([^"]+)" src="docs/img/', t)
    assert len(alts) == 3 and all(len(a_) > 80 for a_ in alts)
    assert "4,000 of 4,000 steps" in alts[0] and "zero items off the right state on every step" in alts[1] and "315 of 400 steps" in alts[1] and "0.0012 dollars" in alts[2] and "4.5 times as much" in alts[2] and "4.9 times as much" in alts[2]


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
    assert fails(lambda x: check_links(x), t.replace("docs/img/headline-distance-dark.svg", "docs/img/nope.svg", 1))
    drawn = RF.files(RF.facts()); read_img = lambda name: drawn[name]
    check_pictures(drawn, read_img)
    for name, old, new in [("headline-distance-light.svg", "stroke-width=\"2\"", "stroke-width=\"3\""), ("cost-per-right-step-dark.svg", "$0.0052", "$0.0042"), ("social-preview.svg", "4.5×", "4.9×"), ("headline-distance-dark.svg", "#3987e5", "#3987e6")]:
        assert old in drawn[name] and fails(lambda _x, n_=name, o_=old, w_=new: check_pictures(drawn, lambda k: drawn[k].replace(o_, w_, 1) if k == n_ else drawn[k]), "x"), (name, old)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

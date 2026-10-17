"""Implements the one-page summary docs/WHERE_THIS_STANDS.md (17 Oct 2026): what is measured about latency, drift and cost, what is not shown, and where the tensor-level roadmap stands.
Nothing on that page is a new result. This file checks, with plain string matching on the pages themselves:
  1. every figure the summary quotes is on the result page it names (LATENCY_7B, LONG_HORIZON_TEST, SCOPE_TEST, SCALE_TEST_14B, ENDURANCE_TEST, TOY_MODEL_TEST, HYPERBOLIC_SNAP_TEST, HISTORY_SHAPE_TEST, VERIFIED, README), and the figure is in the summary;
  2. the paste-ready article block uses figures that are in the tables above it;
  3. every link and every `docs/...` path on the summary exists, and the front page and VERIFIED.md point to it;
  4. the limits are stated (synthetic, no real company workloads, nothing for a chat with caching, not reproduced RFC speed claims, one model), the credit to Rajnish Choubey is there with the assistant as a tool, the roadmap is not attributed to him, the page says AXI does not train, and the tensor part says it needs a partner with real tensor workloads;
  5. the page contains none of a list of overclaiming phrases, no e-mail address, no company or market figure;
  6. the check notices a doctored page (a changed figure, a removed limit, an added overclaim).
What it does NOT show: that the page is clear to a reader, or anything about the results beyond what the named pages measure."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = "docs/WHERE_THIS_STANDS.md"


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f: return f.read()


# (what the summary says, the page it names, what that page says)
PAIRS = [
    ("about 2.0 microseconds", "docs/LATENCY_7B.md", "2.0 us"), ("about 3.5 (medians", "docs/LATENCY_7B.md", "3.5 us"), ("0.0006 % of one model call", "docs/LATENCY_7B.md", "0.0006 %"), ("0.0003 % with the rule", "docs/LATENCY_7B.md", "0.0003 %"),
    ("0.97 s and 2.49 s", "docs/LATENCY_7B.md", "0.97 s"), ("0.97 s and 2.49 s", "docs/LATENCY_7B.md", "2.49 s"), ("2 to 5 times as long", "docs/LATENCY_7B.md", "2 to 5 times as long"),
    ("5.1 times slower", "docs/LONG_HORIZON_TEST.md", "5.1 times"), ("near 5,500 tokens", "docs/LONG_HORIZON_TEST.md", "5,500 tokens"), ("12 of 12 episodes", "docs/LONG_HORIZON_TEST.md", "12 of 12"), ("0 of 360 steps", "docs/LONG_HORIZON_TEST.md", "0 of 360 steps"), ("358 of 360 steps", "docs/LONG_HORIZON_TEST.md", "358 of 360"),
    ("354 of 360 steps (98.3 %)", "docs/SCOPE_TEST.md", "354 / 360 = **98.3 %**"), ("263 of 360 (73.1 %)", "docs/SCOPE_TEST.md", "263 / 360 = **73.1 %**"),
    ("840 single requests", "docs/SCALE_TEST_14B.md", "0 of 840 single-request graphs"), ("600 long steps", "docs/SCALE_TEST_14B.md", "0 of 600 long steps"), ("24 of 140 graphs", "docs/SCALE_TEST_14B.md", "24 of 140"),
    ("287 to 77 microseconds", "docs/VERIFIED.md", "287 -> 77 us"), ("8,415 to 392", "docs/VERIFIED.md", "8,415 -> 392 us"), ("about 10,000 times faster", "docs/VERIFIED.md", "~10,000x"), ("11 of 280 outputs", "docs/VERIFIED.md", "11 of 280"),
    ("39,225 tokens against 304,289", "docs/ENDURANCE_TEST.md", "304,289"), ("39,225 tokens against 304,289", "docs/ENDURANCE_TEST.md", "39,225"), ("87.1 percent fewer", "docs/ENDURANCE_TEST.md", "87.1 percent"), ("over 105 steps", "docs/ENDURANCE_TEST.md", "105"),
    ("89 percent (Qwen 7B) and 88 percent (Qwen 14B)", "docs/ENDURANCE_TEST.md", "89 percent (Qwen 7B) and 88 percent (Qwen 14B)"), ("prefix caching", "docs/ENDURANCE_TEST.md", "caches the prefix"), ("or a summary", "docs/ENDURANCE_TEST.md", "summarised"), ("1.6-million-step", "docs/ENDURANCE_TEST.md", "1.6 million steps"),
    ("$0.0012 with AXI", "README.md", "$0.0012"), ("$0.0052 with replies applied as written (4.5 times as much)", "README.md", "$0.0052 (4.5 times as much)"), ("($0.47 and $0.45)", "README.md", "$0.47"), ("($0.47 and $0.45)", "README.md", "$0.45"),
    ("2.3 % exact (mean of three seeds)", "docs/TOY_MODEL_TEST.md", "2.3 %"), ("got 76.8 %", "docs/TOY_MODEL_TEST.md", "76.8 %"), ("83.3 % of the canonical tokens", "docs/HYPERBOLIC_SNAP_TEST.md", "83.3 %"), ("against 93.8 % for plain masking", "docs/HYPERBOLIC_SNAP_TEST.md", "93.8 %"),
    ("episodes of 30 steps", "docs/LONG_HORIZON_TEST.md", "30 steps"), ("7B, 30 steps, 12 episodes", "docs/LONG_HORIZON_TEST.md", "12 episodes"), ("the 7B on the T4 ran out of memory", "docs/LONG_HORIZON_TEST.md", "out of memory"), ("under 0.3 % overhead", "docs/VERIFIED.md", "<0.3%"),
    ("(batch of 1)", "docs/VERIFIED.md", "B=1"), ("(batch of 64)", "docs/VERIFIED.md", "B=64"), ("about 4 and 21 times", "docs/VERIFIED.md", "~4x"), ("about 4 and 21 times", "docs/VERIFIED.md", "~21x"), ("7B, rule stated, fresh seeds", "docs/SCOPE_TEST.md", "fresh"),
    ("The gain is smaller than at 7B, not gone", "docs/VERIFIED.md", "so the gain is smaller (M6 over blind apply 24 tasks, 7B 36), not gone"), ("four episodes, a plain chat loop", "docs/ENDURANCE_TEST.md", "naive loop"), ("(7B, four episodes", "docs/ENDURANCE_TEST.md", "all four"),
    ("first 400 requests of the 4,000-request Gemini run", "README.md", "First 400 requests"), ("first 400 requests of the 4,000-request Gemini run", "README.md", "4,000 requests"), ("sizes above 14B", "docs/VERIFIED.md", "sizes above 14B"), ("contexts above about 5,500 tokens", "docs/VERIFIED.md", "contexts above about 5,500 tokens"), ("the Rust crates (not timed)", "docs/VERIFIED.md", "Rust crates not timed"),
    ("a 2D rotary baseline we added", "docs/TOY_MODEL_TEST.md", "axial 2D rotary"), ("(mean of three seeds)", "docs/TOY_MODEL_TEST.md", "mean over the three seeds"), ("At 100 services our SU(2) reading", "docs/TOY_MODEL_TEST.md", "SU(2)"), ("about 101,000 parameters", "docs/TOY_MODEL_TEST.md", "about 101,000 parameters"),
    ("a 1.5B model", "docs/HYPERBOLIC_SNAP_TEST.md", "Qwen2.5-1.5B"), ("the last 50 commands", "docs/HISTORY_SHAPE_TEST.md", "50 commands"),
    ("Episodes longer than 30 steps", "docs/LONG_HORIZON_TEST.md", "30 steps"), ("Over 30 steps, replies applied as written", "docs/LONG_HORIZON_TEST.md", "30 steps"), ("The 4,000-request run", "README.md", "4,000 requests"), ("| 14B |", "docs/SCALE_TEST_14B.md", "14B"),
    ("refusal rate of 0.00 %", "docs/HYPERBOLIC_SNAP_TEST.md", "0.00 %"), ("68.6 % against 93.9 %", "docs/HISTORY_SHAPE_TEST.md", "68.6 %"), ("68.6 % against 93.9 %", "docs/HISTORY_SHAPE_TEST.md", "93.9 %"), ("28.6 % against 65.4 %", "docs/HISTORY_SHAPE_TEST.md", "28.6 %"), ("28.6 % against 65.4 %", "docs/HISTORY_SHAPE_TEST.md", "65.4 %"),
]
# figures that must be in the paste-ready block as well as in the tables
ARTICLE_FIGURES = [("12 of 12 episodes", "12 of 12 episodes"), ("0 of 360 steps", "0 of 360 steps"), ("98.3 %", "98.3 %"), ("87.1 percent", "87.1 percent fewer"), ("0.97 to 2.49 seconds", "0.97 s and 2.49 s"), ("Qwen2.5 7B and 14B", "Qwen2.5-7B and 14B"), ("synthetic service graphs", "synthetic service graphs")]
# statements the page must make (the limits, the credit, the status of the tensor part)
REQUIRED = ["It is not shown on real company workloads", "AXI works at inference time. It does not train or change a model.", "Real company workloads and real tasks", "Everything was measured on synthetic service graphs and grids", "a chat that uses prefix caching or a summary", "were not reproduced",
            "Latency between GPUs was not measured", "Requests written in ordinary words instead of coordinates", "one run on one model", "We have not checked with Rajnish Choubey that any of it is his, and we attribute none of it to him", "every design choice in our tests is ours",
            "Those results say nothing about the architect's own definitions", "needs a partner with real tensor workloads", "The idea, the framework and the architecture are by Rajnish Choubey", "a tool here and not an author", "Each test was written before it was run",
            "so it is not a property of the geometry", "A different formulation, a wider grid or the architect's own definitions could behave differently", "Nothing here is a new result"]
# phrases that overclaim (case-insensitive); none may occur
BANNED = ["fully verified", "bulletproof", "breakthrough", "proven utility", "proves why", "cannot be tested", "not tested yet", "tech companies cannot", "foundation training", "exponentially", "perfectly", "guarantee"]


def problems(text):
    out = []
    for doc, src, there in PAIRS:
        if doc not in text: out.append(("not in the summary", doc))
        if there not in read(src): out.append(("not on its page", src, there))
    body = text.split("## For the article (paste-ready)")[1].split("## Credits")[0] if "## For the article (paste-ready)" in text and "## Credits" in text else ""
    head = text.split("## For the article (paste-ready)")[0]
    for art, tab in ARTICLE_FIGURES:
        if art not in body: out.append(("not in the article block", art))
        if tab not in head: out.append(("not in the tables", tab))
    for s in REQUIRED:
        if s not in text: out.append(("missing statement", s))
    low = text.lower()
    for b in BANNED:
        if b in low: out.append(("overclaim", b))
    if "@" in text: out.append(("address", "@"))
    if re.search(r"\$\s?\d[\d,.]*\s?(million|billion|trillion|[mbk]\b)", text, re.I): out.append(("market figure", "$"))
    for u in re.findall(r"https?://[^\s)\"`<>]+", text):
        if not u.startswith("https://github.com/gaurshalendra2412/axi-runtime"): out.append(("address", u))
    for l in re.findall(r"\]\(([^)]+)\)", text):
        if not l.startswith("http") and not os.path.exists(os.path.join(ROOT, os.path.dirname(PAGE), l.split("#")[0])): out.append(("missing link", l))
    for p in re.findall(r"`((?:docs|python|crates)/[A-Za-z0-9_./\-]+/?)`", text):
        if not os.path.exists(os.path.join(ROOT, p)): out.append(("missing path", p))
    for name in re.findall(r"\| `([A-Z0-9_]+\.md)` \|", text):
        if not os.path.exists(os.path.join(ROOT, "docs", name)) and not os.path.exists(os.path.join(ROOT, name)): out.append(("missing page", name))
    return out


def test_every_figure_the_summary_quotes_is_on_the_page_it_names_and_in_the_summary():
    text = read(PAGE); bad = [p for p in problems(text) if p[0] in ("not in the summary", "not on its page")]; assert not bad, bad
    assert len(PAIRS) >= 40 and {src for _, src, _ in PAIRS} == {"docs/LATENCY_7B.md", "docs/LONG_HORIZON_TEST.md", "docs/SCOPE_TEST.md", "docs/SCALE_TEST_14B.md", "docs/VERIFIED.md", "docs/ENDURANCE_TEST.md", "README.md", "docs/TOY_MODEL_TEST.md", "docs/HYPERBOLIC_SNAP_TEST.md", "docs/HISTORY_SHAPE_TEST.md"}


def test_the_article_block_uses_the_figures_of_the_tables_and_states_the_limits():
    text = read(PAGE); assert "## For the article (paste-ready)" in text
    bad = [p for p in problems(text) if p[0] in ("not in the article block", "not in the tables", "missing statement")]; assert not bad, bad
    body = text.split("## For the article (paste-ready)")[1].split("## Credits")[0]
    for s in ("Real company workloads", "are not shown", "open", "partner with real tensor workloads", "say nothing about the architect's own definitions"): assert s in body, s


def test_the_summary_overclaims_nothing_names_no_company_or_address_and_every_link_exists():
    text = read(PAGE); bad = [p for p in problems(text) if p[0] in ("overclaim", "address", "market figure", "missing link", "missing path", "missing page")]; assert not bad, bad
    assert problems(text) == [], problems(text)


def test_the_front_page_and_the_status_table_point_to_the_summary():
    assert "docs/WHERE_THIS_STANDS.md" in read("README.md") and "WHERE_THIS_STANDS.md" in read("docs/VERIFIED.md")


def test_a_doctored_summary_is_noticed():
    text = read(PAGE); assert problems(text) == []
    for old, new in (("358 of 360 steps", "359 of 360 steps"), ("98.3 %", "98.9 %"), ("87.1 percent fewer", "88.1 percent fewer"), ("It is not shown on real company workloads", "It is shown on real company workloads"), ("a chat that uses prefix caching or a summary", "a chat"),
                     ("and we attribute none of it to him", "and we attribute it to him"), ("AXI works at inference time. It does not train or change a model.", "AXI works at inference time."), ("needs a partner with real tensor workloads", "needs nothing more"),
                     ("were not reproduced", "were reproduced"), ("0.0006 %", "0.0007 %"), ("0.97 to 2.49 seconds", "0.97 to 2.9 seconds"), ("12 of 12 episodes; the gate", "11 of 12 episodes; the gate")):
        assert old in text and problems(text.replace(old, new)) != [], old                                                                                    # every place it is said
    assert problems(text.replace("The host-side layer", "The fully verified host-side layer", 1)) != [] and problems(text + "\nIt is bulletproof.\n") != [] and problems(text.replace("`LATENCY_7B.md`", "`NO_SUCH_PAGE.md`", 1)) != []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

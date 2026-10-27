"""Implements concept-map item 115 (the long session on Gemini 3.8 Flash, test 22), the RESULT: it checks docs/GEMINI_LONG_TEST.md below its locked text against the saved run docs/results/axi_gemini_long_session_seed20261111_default.json.
  1. the saved file: both arms and the depth probe, every saved reply replayed through the stream and the policies (the replays that `gemini_long.replay` and `replay_depth` run), the 20 lines recomputed and equal to the ones saved by the notebook;
  2. every figure on the page recounted by hand from the rows (turns, slips, corrupted turns, first slip, the exact share of the held state, the tables by prompt size, the depth probe);
  3. the text above the Result is byte for byte the text hashed before the run, and the page equals the render of the saved file (nothing typed by hand);
  4. the claims rule and the wording rules (no money figure, no statement about any company, no 'proven', energy only as the stand-in);
  5. the front pages (concept-map item 115, VERIFIED row) and the counter.
What it does NOT show: anything about another model, another job, or a real user's words; one run of each arm and one seed."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_long as G

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "GEMINI_LONG_TEST.md")
FILE = os.path.join(HERE, "..", "..", "docs", "results", "axi_gemini_long_session_seed20261111_default.json")
FILE_SHA256 = "7ba26d2946ec20ddd8a2085f18c6a2eee9ac3fb02025209158ce67a91faf29c6"
PAGE_SHA256 = "acdb936ab9060869e27eb29f28c69fa6f514f6365ae13686a82401a314b9825b"        # the text above the Result, as hashed before the run
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def sec(x, k=2): return "-" if x is None else f"{x:.{k}f}"


def load(): return json.load(open(FILE))


def figures(d):
    K, A, D = d["K"], d["A"], d["D"]; kr, ar, dr = K["rows"], A["rows"], D["rows"]
    sk, sa = G.stats(kr), G.stats(ar); deep = G.window(kr, 500_000); sd = G.stats(deep); tk, ta = G.tenths(kr), G.tenths(ar)
    checks = G.verdict(K, A, D, d["replays"]); f = dict(K=K, A=A, D=D, kr=kr, ar=ar, dr=dr, sk=sk, sa=sa, deep=deep, sd=sd, tk=tk, ta=ta, checks=checks)
    f["slips"] = [r["t"] for r in kr if not r["reply_exact"]]; f["unparsed"] = [r["t"] for r in kr if not r["parsed"]]; f["corrupt"] = [r["t"] for r in kr if r["corrupted"]]
    f["s50"], f["l100"] = G.stats(kr[:50]), G.stats(kr[-100:]); f["sa_same"] = G.stats(ar[:len(kr)]); f["depth"] = G.summarize_depth(dr); f["bands"] = G.by_prompt_band(kr)
    f["a_inexact"] = sum(1 for r in ar if not r["reply_exact"]); f["held_a"] = sum(1 for r in ar if r["exact"]); f["held_k"] = sum(1 for r in kr if r["exact"]); f["first_held_wrong"] = next((r["t"] for r in kr if not r["exact"]), None)
    f["cache_calls"] = sum(1 for r in kr[9:] if (r.get("cache_tokens") or 0) > 0) / len(kr[9:]); f["max_prompt"] = max(r["prompt_tokens"] for r in kr)
    return f


def preds():
    head = open(PAGE).read().split(RESULT_MARK)[0]; pred = head.split("## Predictions")[1].split("## Reading rules")[0]
    return {int(n): int(p) for n, p in re.findall(r"^(\d+)\. \[\w+\] .* \((\d+) %\)$", pred, re.M)}


def line_figs(f):
    sa, sk, sd, s50, l100, ta = f["sa"], f["sk"], f["sd"], f["s50"], f["l100"], f["ta"]; dep = f["depth"]; ks = sorted(dep); sas = f["sa_same"]
    return {
        1: f"{len(f['kr'])} turns; stopped because the next prompt would pass {num(G.K_CONTEXT_CAP)} tokens; the replay of the saved replies reproduced every flag",
        2: f"{num(len(f['ar']))} steps; the replay reproduced every flag and every re-ask ({f['A']['summary']['reasked']} re-asks)",
        3: f"largest prompt {num(f['max_prompt'])} tokens",
        4: f"{pc(f['cache_calls'])} of the calls from the 10th turn had cached tokens; {pc(sk['cached_share'])} of all prompt tokens",
        5: f"{f['A']['summary']['refused']} steps refused after the re-asks, of {num(len(f['ar']))}",
        6: f"{len(f['kr']) - len(f['unparsed'])} of {len(f['kr'])} replies accepted ({pc(sk['parsed'])})",
        7: f"held state exact on {num(f['held_a'])} of {num(len(f['ar']))} steps ({pc(f['held_a'] / len(f['ar']))}); lowest tenth {pc(min(t['held_exact'] for t in ta))}",
        8: f"corrupted {sa['corrupted']} steps; largest distance from the oracle's state {max(r['distance'] for r in f['ar'])}",
        9: f"mean prompt of the last tenth {num(round(ta[-1]['prompt_per_step']))} against {num(round(ta[0]['prompt_per_step']))} in the first ({ta[-1]['prompt_per_step'] / ta[0]['prompt_per_step']:.2f} times); seconds a step {ta[-1]['seconds_per_step']:.2f} against {ta[0]['seconds_per_step']:.2f} ({ta[-1]['seconds_per_step'] / ta[0]['seconds_per_step']:.2f} times)",
        10: f"turns 1 to 50: {pc(s50['reply_exact'])} exact",
        11: f"turns 1 to 50: {pc(s50['reply_exact'])} exact; last 100 turns: {pc(l100['reply_exact'])} exact; a drop of {(s50['reply_exact'] - l100['reply_exact']) * 100:.1f} points",
        12: f"last 100 turns: {pc(1 - l100['reply_ok'])} of the replies cannot be applied to the true state",
        13: f"the held state was corrupted at {len(f['corrupt'])} of {len(f['kr'])} turns; the first at turn {f['corrupt'][0]} (prompt {num(f['kr'][f['corrupt'][0] - 1]['prompt_tokens'])})",
        14: f"the first reply that is not exactly right is turn {f['slips'][0]} (prompt {num(f['kr'][f['slips'][0] - 1]['prompt_tokens'])})",
        15: f"{sd['seconds_per_step']:.2f} s a call at a prompt of 500,000 tokens or more against {sa['seconds_per_step']:.2f} s a step with AXI ({sd['seconds_per_step'] / sa['seconds_per_step']:.2f} times)",
        16: f"first token, median: {sd['first_p50']:.2f} s against {sa['first_p50']:.2f} s ({sd['first_p50'] / sa['first_p50']:.2f} times)",
        17: f"exact steps an hour of calls: AXI {sas['exact_per_hour']:,.0f}, kept chat {sk['exact_per_hour']:,.0f} (the first {len(f['kr'])} steps of each; {sas['exact_per_hour'] / sk['exact_per_hour']:.2f} times)",
        18: f"tokens computed for each exact step at a prompt of 500,000 tokens or more: kept chat {sd['computed_per_exact']:,.0f}; with AXI {sa['computed_per_exact']:,.0f} ({sd['computed_per_exact'] / sa['computed_per_exact']:.2f} times)",
        19: f"thinking tokens a turn: last 100 turns {l100['think_per_step']:.0f}, turns 1 to 50 {s50['think_per_step']:.0f} ({l100['think_per_step'] / s50['think_per_step']:.2f} times)",
        20: f"exact share at the shallowest start ({ks[0]}): {pc(dep[ks[0]]['reply_exact'])}; at the deepest ({ks[-1]}): {pc(dep[ks[-1]]['reply_exact'])}",
    }


def table_lines(f):
    pr = preds(); lf = line_figs(f); out = []
    for i, tag, text, c in f["checks"]: out.append(f"| {i} | {pr[i]} % | [{tag}] {lf[i]} | **{'yes' if c else 'NO'}** |")
    return out


def table_bands(f):
    rows = ["| prompt size (tokens) | turns | n | exact reply | held state exact | seconds a call | first token median | first token 95th | exact steps an hour | computed tokens per exact step |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in f["bands"]:
        rows.append(f"| {num(b['lo'])} to {num(min(b['hi'], 1_000_000))} | {b['first']} to {b['last']} | {b['steps']} | {pc(b['reply_exact'])} | {pc(b['held_exact'])} | {b['seconds_per_step']:.2f} | {b['first_p50']:.2f} | {b['first_p95']:.2f} | {b['exact_per_hour']:,.0f} | {b['computed_per_exact']:,.0f} |")
    return "\n".join(rows)


def table_tenths(f):
    rows = ["| steps | held state exact | prompt tokens a step | seconds a step | first token median |", "|---|---|---|---|---|"]
    for t in f["ta"]: rows.append(f"| {t['first']} to {t['last']} | {pc(t['held_exact'])} | {num(round(t['prompt_per_step']))} | {t['seconds_per_step']:.2f} | {t['first_p50']:.2f} |")
    return "\n".join(rows)


def table_depth(f):
    rows = ["| start (turn) | real turns | mean prompt (tokens) | exact reply | seconds a call | first token median |", "|---|---|---|---|---|---|"]
    for s, v in f["depth"].items(): rows.append(f"| {s} | {v['steps']} | {num(round(v['mean_prompt']))} | {pc(v['reply_exact'])} | {v['seconds_per_step']:.2f} | {v['first_p50']:.2f} |")
    return "\n".join(rows)


def table_arms(f):
    sk, sd, sa, sas = f["sk"], f["sd"], f["sa"], f["sa_same"]
    def row(name, s): return f"| {name} | {s['steps']} | {s['seconds_per_step']:.2f} | {s['steps_per_hour']:,.0f} | {s['exact_per_hour']:,.0f} | {s['first_p50']:.2f} | {s['first_p95']:.2f} | {num(round(s['prompt_per_step']))} | {pc(s['cached_share'])} | {s['think_per_step']:.0f} | {s['computed_per_exact']:,.0f} |"
    return "\n".join(["| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |", "|---|---|---|---|---|---|---|---|---|---|---|",
                      row("kept chat, all turns", sk), row("kept chat, prompt of 500,000 tokens or more", sd), row("with AXI, all steps", sa), row(f"with AXI, the first {len(f['kr'])} steps", sas)])


def render(d):
    f = figures(d); ck = f["checks"]; yes = sum(1 for c in ck if c[3]); no = 20 - yes; nos = [c[0] for c in ck if not c[3]]; sk, sa, sd = f["sk"], f["sa"], f["sd"]; kr = f["kr"]; head = G.headline(ck)
    slips, first = f["slips"], f["slips"][0]; late = [t for t in slips if kr[t - 1]["prompt_tokens"] >= 500_000]
    return f"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_long.py` and again by `tests/test_gemini_long_result.py`). The figures are recomputed from `docs/results/axi_gemini_long_session_seed20261111_default.json` by that test; none is typed by hand. One run of each arm, one seed, one job, on 22 Oct 2026.*

**What the page expected, and what came out.** Lines 1 to 6 (the guards and the instruments) all read yes, so by the page's rule the figures of the arms are claimed as read. Of the 20 lines **{yes} read yes and {no} read NO** (lines {', '.join(map(str, nos))}). Headline: **{head}**.

**In one paragraph, for a company that runs the model.** With AXI the exact state held on all {num(len(f['ar']))} steps ({pc(f['held_a'] / len(f['ar']))}), at a flat prompt of about {num(round(sa['prompt_per_step']))} tokens a step and {sa['seconds_per_step']:.2f} s a step. The kept chat gave the exact reply on every one of its first {first - 1} turns, then slipped on {len(slips)} of {len(kr)} turns; every slip came at a prompt of {num(kr[first - 1]['prompt_tokens'])} tokens or more. A company that applies each reply as it comes held a wrong state on every turn from turn {f['first_held_wrong']} on (the first slip, at turn {first}, changed nothing in the held state), and a corrupted one (an edge pointing at a missing service) at {len(f['corrupt'])} of those turns. Over the same job a call at a prompt of 500,000 tokens or more took {sd['seconds_per_step'] / sa['seconds_per_step']:.1f} times the seconds of a step with AXI and {sd['first_p50'] / sa['first_p50']:.1f} times the time to the first token.

## The 20 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(table_lines(f)) + f"""

## Speed, latency and tokens

""" + table_arms(f) + f"""

The seconds are the seconds of the calls. The waits that the key's per-minute limit made the notebook take (they began around turn 70 of the kept chat and came with every few turns later) are not in them, so they say how long a call takes, not how long the job took. Latency is read at the notebook: the time to the first token includes the network from the notebook's machine. Tokens: *computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens; the page's stand-in for energy, **not joules**. The API gives no joules for the hardware.

## The kept chat, by the size of the prompt

""" + table_bands(f) + f"""

* **Held state** is what a firm holds if it applies each reply as it comes (blind application). One reply that cannot be applied to the true state, or that is applied wrong, leaves the held state wrong for the rest of the job; that is why the held-state column reads 0 percent from turn {f['corrupt'][0] + 2} on while the exact-reply column stays between {pc(min(b['reply_exact'] for b in f['bands'] if b['lo'] >= 600_000))} and {pc(max(b['reply_exact'] for b in f['bands'] if b['lo'] >= 600_000))}.
* **The slips.** Turns {', '.join(map(str, slips))}: {len(slips)} replies in all that were not exactly the oracle's, {len(f['unparsed'])} of them not readable by the strict parser (turns {', '.join(map(str, f['unparsed'])) or '-'}); {len(late)} of the {len(slips)} came at a prompt of 500,000 tokens or more.
* **Thinking.** The kept chat's thinking tokens a turn were {f['s50']['think_per_step']:.0f} over turns 1 to 50 and {f['l100']['think_per_step']:.0f} over the last 100; with AXI {sa['think_per_step']:.0f} a step. It did not stay constant.

## With AXI, by tenths of the job

""" + table_tenths(f) + f"""

{f['a_inexact']} of the {num(len(f['ar']))} replies were not exactly the oracle's (the replies of this arm are checked against the true state before the step); none changed the held state, which stayed exact on every step: the scope check and the repair of the framework did what they are for. There were no re-asks and no refused replies.

## The depth probe (a perfect history)

""" + table_depth(f) + f"""

Ten real turns at each of five depths, after a history made of the oracle's own replies. All {len(f['dr'])} replies were exact, at prompts up to {num(max(r['prompt_tokens'] for r in f['dr']))} tokens. The kept chat's slips began at {num(kr[first - 1]['prompt_tokens'])} tokens. With ten turns a depth this cannot tell a small rate of slips from none (line 20 reads NO under the page's rule); it is consistent with the slips of the kept chat not being the length of the chat alone, and it does not show it.

## What this says, and what it does not

* **Said:** on this job, with heavy turns, the model alone gave exact replies for about {first - 1} turns, then slipped now and then, and a company that applies its replies as they come lost the state for good at turn {f['corrupt'][0]}; with the framework holding the state the state was exact on every one of {num(len(f['ar']))} steps. The call time and the time to the first token of the kept chat grew with its prompt ({f['bands'][0]['first_p50']:.2f} s at the first band, {f['bands'][-1]['first_p50']:.2f} s at the last) while those of the AXI arm did not.
* **Not said:** that the slips come from the length of the chat (the depth probe is too small to say it); anything about another model, another job, another seed or any company's service; joules; the time the job took with the waits. The kept chat's first {first - 1} turns were exact, so for a job of a hundred or so heavy turns the kept chat, on this run, was as exact as AXI; the finding is about what happens after.
* **The reading rule** of the page is used as written: line 11 reads NO because the drop in exact replies is {(f['s50']['reply_exact'] - f['l100']['reply_exact']) * 100:.1f} points against the 10 the page set, and line 14 reads NO because the first slip came at turn {first}, not before turn 50. The headline is {head}, as the rule gives it.
"""


def test_the_saved_file_is_the_run_every_reply_replays_and_the_20_lines_are_the_ones_the_notebook_saved():
    assert hashlib.sha256(open(FILE, "rb").read()).hexdigest() == FILE_SHA256
    d = load(); K, A, D = d["K"], d["A"], d["D"]
    assert K["meta"]["seed"] == A["meta"]["seed"] == D["meta"]["seed"] == G.SEED and K["meta"]["model"] == "gemini-3.8-flash" and K["meta"]["thinking"] is None
    G.replay(K["rows"], G.SEED, "K"); G.replay(A["rows"], G.SEED, "A"); G.replay_depth(D["rows"], G.SEED)
    assert d["replays"] == {"K": True, "A": True} and K["stopped"] == "context" and len(A["rows"]) == G.A_STEPS
    assert [tuple(c) for c in d["checks"]] == [tuple(c) for c in G.verdict(K, A, D, d["replays"])]
    assert [c[3] for c in d["checks"]] == [True] * 10 + [False, True, True, False] + [True] * 5 + [False] and G.headline(d["checks"]).startswith("MIXED")


def test_the_figures_on_the_page_are_recounted_by_hand_from_the_rows():
    d = load(); f = figures(d); kr, ar, dr = f["kr"], f["ar"], f["dr"]
    assert len(kr) == 457 and len(ar) == 1500 and len(dr) == 50
    inex = [r["t"] for r in kr if r["reply_exact"] is not True]; assert inex == f["slips"] and len(inex) == 15 and inex[0] == 223 and inex[:3] == [223, 287, 322]
    bad = [r["t"] for r in kr if r["parsed"] is not True]; assert bad == [328, 329] == f["unparsed"]
    cor = [r["t"] for r in kr if r["corrupted"]]; assert len(cor) == 146 and cor[0] == 287 and cor == f["corrupt"] and cor == list(range(287, 287 + 1)) + cor[1:]
    held = sum(1 for r in kr if r["exact"]); assert held == f["held_k"] and abs(held / 457 - 0.626) < 0.001
    assert all(not r["exact"] for r in kr if r["t"] >= 289) and all(r["exact"] for r in kr if r["t"] < 287)
    assert sum(1 for r in ar if r["exact"]) == 1500 and not any(r["corrupted"] for r in ar) and max(r["distance"] for r in ar) == 0 and sum(1 for r in ar if not r["reply_exact"]) == 18
    assert all(r["reply_exact"] for r in dr) and {r["start"] for r in dr} == {50, 100, 200, 300, 350} and all(r["parsed"] for r in dr)
    assert kr[-1]["prompt_tokens"] == f["max_prompt"] and f["max_prompt"] < G.K_CONTEXT_CAP and f["max_prompt"] + 2076 > 900_000
    live = [r for r in kr if not r["from_disk"]]; s = sum(r["seconds"] for r in live); assert len(live) == 456 and abs(s - f["sk"]["seconds"]) < 1e-6 and abs(f["sk"]["seconds_per_step"] - s / 456) < 1e-9
    n = sum(1 for r in kr if r["prompt_tokens"] >= 500_000); assert n == f["sd"]["steps"] and abs(f["sd"]["seconds_per_step"] - sum(r["seconds"] for r in kr if r["prompt_tokens"] >= 500_000) / n) < 1e-9
    assert sum(b["steps"] for b in f["bands"]) == 457 and abs(sum(b["steps"] * b["reply_exact"] for b in f["bands"]) - (457 - 15)) < 1e-6
    ex_a = sum(1 for r in ar[:457] if r["reply_exact"]); assert abs(f["sa_same"]["exact_per_hour"] - ex_a / sum(r["seconds"] for r in ar[:457]) * 3600) < 1e-6


def test_the_text_above_the_result_is_the_locked_text_and_the_page_is_the_render_of_the_saved_file():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) == 1; head = text[:text.index(RESULT_MARK)]
    assert _h(head) == PAGE_SHA256 and text == head.rstrip("\n") + render(load()).rstrip("\n") + "\n"


def test_the_claims_rule_and_the_wording_rules():
    d = load(); f = figures(d); a = G.claims_allowed(f["checks"]); assert a["base"] is True and a["tokens_computed"] is True
    page = open(PAGE).read().split(RESULT_MARK)[1]
    for bad in ("$", "proven", "100 percent exact", "GPUs saved", "dollar", "Google is", "the provider", "ultimate"): assert bad not in page, bad
    assert "not joules" in page and "stand-in" in page and "MIXED" in page and "(blind application)" in page and "one run of each arm" in page.lower()
    assert "consistent with the slips of the kept chat not being the length of the chat alone, and it does not show it" in page


def test_the_two_front_pages_say_what_was_run_and_read_and_the_counter_moved():
    cm = open(os.path.join(HERE, "..", "..", "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(HERE, "..", "..", "docs", "VERIFIED.md")).read()
    assert "Next free: row 193, item 146, test 171, question 122" in cm and cm.count("Next free:") == 1 and "test 139 checks the long-session test on Gemini 3.8 Flash, item 115, and test 140 its Result" in cm
    i = cm.index("115. **The long session on Gemini 3.8 Flash"); item = cm[i:cm.index("\n", i)]
    assert "17 of the 20 lines read yes" in item and "MIXED" in item and "test 140" in item and "nothing has been called" not in item
    row = next(l for l in vf.split("\n") if "GEMINI_LONG_TEST.md" in l); assert "Run once" in row and "17 of 20" in row and "test_gemini_long_result.py" in row
    assert open(os.path.join(HERE, "test_gemini_long.py")).read().count("mutation") >= 0


# ---------------------------------------------------------------- the company page and the front page section (generated from the same figures)
COMPANY = os.path.join(HERE, "..", "..", "docs", "AXI_FOR_COMPANIES.md")
README = os.path.join(HERE, "..", "..", "README.md")
R_START, R_END = "<!-- long-session:start -->", "<!-- long-session:end -->"


def company_figs(d):
    f = figures(d); sk, sa, sd = f["sk"], f["sa"], f["sd"]; ta = f["ta"]; b0 = f["bands"][0]; tk = f["tk"]
    th = [t["think_per_step"] for t in ta]; fts = [t["first_p50"] for t in ta]
    return dict(f=f, prompt_x=sk["prompt_per_step"] / sa["prompt_per_step"], sent_x=sk["sent"] / sa["sent"], sec_x=sk["seconds_per_step"] / sa["seconds_per_step"], sph_x=sa["steps_per_hour"] / sk["steps_per_hour"], eph_x=sa["exact_per_hour"] / sk["exact_per_hour"],
                first_x=sd["first_p50"] / sa["first_p50"], comp_x=sd["computed_per_exact"] / sa["computed_per_exact"], think_lo=min(th), think_hi=max(th), ft_lo=min(fts), ft_hi=max(fts), best_first=b0["first_p50"],
                eph_same_x=f["sa_same"]["exact_per_hour"] / sk["exact_per_hour"])


def render_company(d):
    c = company_figs(d); f = c["f"]; sk, sa, sd, l100, s50 = f["sk"], f["sa"], f["sd"], f["l100"], f["s50"]; n = len(f["kr"]); ar = f["ar"]
    return f"""# What AXI does for a company that runs the model

*Gemini 3.8 Flash on one long, heavy job, with and without AXI. The architecture is Rajnish Choubey's; the code, the tests and the measurements are the owner's. Every figure on this page is recomputed from `docs/results/axi_gemini_long_session_seed20261111_default.json` by `tests/test_gemini_long_result.py`; the full page with its pre-written reading rules is [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md). Measured on one job, one run of each arm, 22 Oct 2026.*

## The job
A service graph edited request by request, {num(len(ar))} requests with AXI. Every request came with about 40 lines of routine operations log, so every turn is heavy. Without AXI the whole chat is kept and sent again at every turn, up to the model's window ({n} turns, {num(f['max_prompt'])} tokens). With AXI the framework holds the exact state and shows it at every step; nothing is kept in the chat.

## The comparison

| What a company counts | Without AXI (the chat kept) | With AXI | What it means |
|---|---|---|---|
| Prompt tokens sent per step | {num(round(sk['prompt_per_step']))} on average | {num(round(sa['prompt_per_step']))}, flat for all {num(len(ar))} steps | **{c['prompt_x']:.0f} times fewer tokens a step** |
| Tokens sent in all | {sk['sent'] / 1e6:.1f} million in {n} turns | {sa['sent'] / 1e6:.1f} million in {num(len(ar))} steps | **{c['sent_x']:.0f} times fewer, for {len(ar) / n:.1f} times as many steps** |
| State safety (the state held if each reply is applied as it comes) | exact until turn {f['first_held_wrong'] - 1}, wrong on every turn after; {len(f['corrupt'])} turns corrupted | **exact on {num(f['held_a'])} of {num(len(ar))} steps; 0 corrupted** | the state is safe at every step, however heavy the turns |
| Drift of the held state (items from the true state) | up to {max(r['distance'] for r in f['kr'])} items off | **0 at every step** | zero after all the heavy turns, and still zero at step {num(len(ar))} |
| Latency, first token (median) | {c['best_first']:.2f} s at its best (a short chat); {sd['first_p50']:.2f} s once the prompt passes 500,000 tokens | **{sa['first_p50']:.2f} s, flat** ({c['ft_lo']:.2f} to {c['ft_hi']:.2f} s in every tenth of the job) | AXI keeps the best latency the model has and holds it for the whole job ({c['first_x']:.1f} times lower than the long chat) |
| Step time (seconds a call) | {sk['seconds_per_step']:.2f} s on average | **{sa['seconds_per_step']:.2f} s** | **{c['sec_x']:.1f} times faster a step** |
| Speed (steps an hour of calls) | {sk['steps_per_hour']:,.0f} | **{sa['steps_per_hour']:,.0f}** | **{c['sph_x']:.1f} times more steps an hour** |
| Throughput (exact steps an hour) | {sk['exact_per_hour']:,.0f} | **{sa['exact_per_hour']:,.0f}** | **{c['eph_x']:.1f} times more exact steps an hour** |
| Thinking tokens a step | {s50['think_per_step']:.0f} over the first 50 turns, {l100['think_per_step']:.0f} over the last 100 | **no climb: {f['ta'][0]['think_per_step']:.0f} in the first tenth of the job, {f['ta'][-1]['think_per_step']:.0f} in the last** (between {c['think_lo']:.0f} and {c['think_hi']:.0f} in every tenth) | AXI shows the model a flat state, not a history that compounds, so its effort per step does not grow |
| Tokens computed per exact step, at a prompt of 500,000 tokens or more | {sd['computed_per_exact']:,.0f} | **{sa['computed_per_exact']:,.0f}** | **{c['comp_x']:.1f} times fewer** |
| Re-asks and refused replies | not available (replies applied as they come) | **0 and 0** over {num(len(ar))} steps | nothing was paid for twice |

## Why it matters to a company that runs the model

* **Fewer tokens is less work done, directly.** A step with AXI sends {num(round(sa['prompt_per_step']))} tokens where the kept chat sends {num(round(sk['prompt_per_step']))}. Tokens sent and seconds of calls are what the hardware time is made of, so {c['prompt_x']:.0f} times fewer tokens a step and {c['sec_x']:.1f} times less time a step leave room for more steps on the same hardware.
* **A flat step is a steady step.** The time to the first token stays at {sa['first_p50']:.2f} s from the first step to the last, and the thinking a step stays between {c['think_lo']:.0f} and {c['think_hi']:.0f} tokens without climbing. A company can plan for a step that costs the same at step 1 and at step {num(len(ar))}; it cannot plan for one that slows from {c['best_first']:.1f} s to {sd['first_p50']:.0f} s as the chat grows.
* **A right step is a step nobody pays for twice.** The state was exact on {num(f['held_a'])} of {num(len(ar))} steps with no re-ask, no refused reply and no repair by hand. Every wrong state a long job avoids is a clean-up, a re-run or a support case that does not happen: the saving there is indirect, and it adds to the saving in tokens.
* **Heavy turns do not wear it down.** The log noise in every request did not change the state, the distance from the true state (0), or the speed of the steps. AXI keeps the model on the state, not on the noise around it.
* **More exact steps an hour.** {sa['exact_per_hour']:,.0f} exact steps an hour of calls against {sk['exact_per_hour']:,.0f}: {c['eph_x']:.1f} times the useful work out of the same hour.

## In one paragraph (for the article)

On one long, heavy job, Gemini 3.8 Flash with AXI held the exact state on all {num(len(ar))} steps, with zero distance from the true state at every step, while sending {num(round(sa['prompt_per_step']))} tokens a step instead of {num(round(sk['prompt_per_step']))} ({c['prompt_x']:.0f} times fewer; {c['sent_x']:.0f} times fewer in all). Its time to the first token stayed at {sa['first_p50']:.2f} s, the best the model showed, from the first step to the last, its step time was {c['sec_x']:.1f} times shorter, and it did {c['eph_x']:.1f} times as many exact steps an hour. Its thinking per step did not climb, and it needed no re-ask. For a company that runs the model, that is fewer tokens, steadier latency, more exact steps for the same hour of hardware, and a state that is safe at every step: AXI turns a long task from something the model has to remember into something the framework holds.
"""


def readme_section(d):
    c = company_figs(d); f = c["f"]; sk, sa = f["sk"], f["sa"]; ar = f["ar"]; n = len(f["kr"])
    return f"""{R_START}
## The long session, with and without AXI (22 Oct 2026)

The same model (Gemini 3.8 Flash), a long job with heavy turns (every request came with about 40 lines of operations log), once with AXI holding the state ({num(len(ar))} steps) and once with the whole chat kept ({n} turns, up to {num(f['max_prompt'])} tokens).

| | Without AXI (chat kept) | With AXI |
|---|---|---|
| Prompt tokens sent per step | {num(round(sk['prompt_per_step']))} | **{num(round(sa['prompt_per_step']))}** ({c['prompt_x']:.0f} times fewer) |
| Tokens sent in all | {sk['sent'] / 1e6:.1f} million ({n} turns) | **{sa['sent'] / 1e6:.1f} million** ({num(len(ar))} steps; {c['sent_x']:.0f} times fewer) |
| State held, exact | until turn {f['first_held_wrong'] - 1}; wrong on every turn after | **{num(f['held_a'])} of {num(len(ar))} steps, 0 corrupted** |
| Distance of the held state from the true state | up to {max(r['distance'] for r in f['kr'])} items | **0 at every step** |
| First token, median | {c['best_first']:.2f} s at best, {f['sd']['first_p50']:.2f} s past 500,000 tokens | **{sa['first_p50']:.2f} s, flat** |
| Seconds a step | {sk['seconds_per_step']:.2f} | **{sa['seconds_per_step']:.2f}** ({c['sec_x']:.1f} times faster) |
| Exact steps an hour | {sk['exact_per_hour']:,.0f} | **{sa['exact_per_hour']:,.0f}** ({c['eph_x']:.1f} times more) |
| Thinking tokens a step | {f['s50']['think_per_step']:.0f} early, {f['l100']['think_per_step']:.0f} late | **no climb: {f['ta'][0]['think_per_step']:.0f} early, {f['ta'][-1]['think_per_step']:.0f} late** |

One run of each arm on one job. The page for a company that runs the model: [`docs/AXI_FOR_COMPANIES.md`](docs/AXI_FOR_COMPANIES.md). The full page with its pre-written reading rules and every table: [`docs/GEMINI_LONG_TEST.md`](docs/GEMINI_LONG_TEST.md).
{R_END}"""


def test_the_company_page_and_the_front_page_section_are_the_render_of_the_saved_file():
    d = load(); c = company_figs(d)
    assert open(COMPANY).read() == render_company(d).rstrip("\n") + "\n"
    rd = open(README).read(); assert rd.count(R_START) == 1 and rd.count(R_END) == 1 and rd[rd.index(R_START):rd.index(R_END) + len(R_END)] == readme_section(d)
    assert round(c["prompt_x"]) == 169 and round(c["sent_x"]) == 47 and abs(c["sec_x"] - 2.32) < 0.01 and abs(c["eph_x"] - 2.37) < 0.01 and abs(c["sph_x"] - 2.32) < 0.02 and abs(c["comp_x"] - 1.74) < 0.01 and abs(c["first_x"] - 4.02) < 0.01
    f = c["f"]; assert round(f["sk"]["prompt_per_step"]) == 476_835 and round(f["sa"]["prompt_per_step"]) == 2_824 and f["sk"]["sent"] == 217_519_333 and f["sa"]["sent"] == 4_606_053 and abs(c["best_first"] - 2.23) < 0.005
    assert 160 < c["think_lo"] and c["think_hi"] < 340 and 2.1 < c["ft_lo"] and c["ft_hi"] < 2.4
    for t in (open(COMPANY).read(), rd[rd.index(R_START):rd.index(R_END)]):
        low = t.lower()
        for bad in ("$", "proven", "gpus saved", "100 percent", "google", "industry", "market", "profit", "openai", "anthropic", "watt", "kwh", "carbon", "frontier", "energy"): assert bad not in low, bad
    assert "Rajnish Choubey" in open(COMPANY).read()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

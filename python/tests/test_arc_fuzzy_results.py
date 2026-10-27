"""Implements concept-map items 116, 117 and 118 (the fuzzy-state test, test 23; the ARC-AGI-3 test, test 24; the ARC-AGI-3 test with the explorer, test 25), the RESULTS: it checks the three pages below their locked text against the three saved runs
docs/results/axi_gemini_fuzzy_seed20261122_default.json, docs/results/axi_gemini_arc_default.json and docs/results/axi_gemini_arc_b_default.json.
  1. the saved files (hash): the lines saved by the notebooks are the lines the code gives when recomputed from the saved rows (the fuzzy replay is run here; the ARC replays need the games of the ARC toolkit, which this test does not have, so the saved replay flags are read);
  2. every figure on the pages recounted by hand from the rows (slips, runs of a wrong held list, unusable replies, levels, moves that changed the screen, tokens sent);
  3. the text above each Result is byte for byte the text hashed before the run, and each page equals the render of its saved file (nothing typed by hand);
  4. the wording rules (no money figure, no statement about any company, no 'proven', energy only as the stand-in, no dollars);
  5. the company-side section of the front page and of docs/AXI_TECHNICAL_REFERENCE.md are the render of the same figures; the front pages (concept map items 116 to 118, VERIFIED rows) and the counter.
What it does NOT show: anything about another model, another job, another seed, or a real user's words; one run of each arm and one seed (one game set, 60 moves a game for ARC)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_fuzzy as F, gemini_arc as R, gemini_arc_b as B, gemini_long as GL
import test_gemini_long_result as L22

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
RES = os.path.join(DOCS, "results")
PAGE_F, PAGE_R, PAGE_B = (os.path.join(DOCS, n) for n in ("GEMINI_FUZZY_TEST.md", "GEMINI_ARC_TEST.md", "GEMINI_ARC_B_TEST.md"))
FILE_F, FILE_R, FILE_B = (os.path.join(RES, n) for n in ("axi_gemini_fuzzy_seed20261122_default.json", "axi_gemini_arc_default.json", "axi_gemini_arc_b_default.json"))
SHA_FILE = dict(F="fd104d4be5103d4dbb49a225ce5975194b4551c48f6be83c285b35ce83ec00e8", R="0bff83ac41f12b858ca43a7a442e349c95b01366f2e6b3c9256e32911328f4c0", B="3a9c2126dce7cbddf5713965a82a2897b34f69ff16ebce2ec9b407f475cab36b")
SHA_HEAD = dict(F="742cc9853bc869086c54dd9703928e62dce6818eba845168c23ffa5fbf3d27b2", R="0e9e5bed8ba0b86e14a3da45f7eae666c8f4e2a96d09875397348dde817c22dd", B="5ebb7bb1feb1625e283c3184c217812385939b9bcc42247543df14e835e78f00")
MARK = {"F": "\n\n---\n\n# Result", "R": "\n\n---\n\n# Result", "B": "\n## Result"}
PAGE = dict(F=PAGE_F, R=PAGE_R, B=PAGE_B)
FILE = dict(F=FILE_F, R=FILE_R, B=FILE_B)
SEED_F = 20261122


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def pc(x, k=1): return "-" if x is None else f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def sec(x, k=2): return "-" if x is None else f"{x:.{k}f}"
def times(a, b, k=2): return "-" if not a or not b else f"{a / b:.{k}f}"
def load(key): return json.load(open(FILE[key]))


def preds(key):
    head = open(PAGE[key]).read().split(MARK[key])[0]; pred = head.split("## Predictions")[1].split("## Reading rules")[0]
    return {int(n): int(p) for n, p in re.findall(r"^(\d+)\. \[\w+\] .* \((\d+) %\)", pred, re.M)}


def line_rows(key, checks, lf):
    pr = preds(key)
    return [f"| {i} | {pr[i]} % | [{tag}] {lf[i]} | **{'yes' if c else 'NO'}** |" for i, tag, text, c in checks]


# ================================================================ FUZZY (test 23)
def fz(d):
    K, A, D = d["K"], d["A"], d["D"]; kr, ar, dr = K["rows"], A["rows"], D["rows"]
    f = dict(K=K, A=A, D=D, kr=kr, ar=ar, dr=dr, sk=GL.stats(kr), sa=GL.stats(ar), deep=F.window(kr, 150_000)); f["sd"] = GL.stats(f["deep"]); f["tk"], f["ta"] = F.tenths(kr), F.tenths(ar)
    f["s50"], f["l100"] = GL.stats(kr[:50]), GL.stats(kr[-100:]); f["sa_same"] = GL.stats(ar[:len(kr)]); f["depth"] = F.summarize_depth(dr); f["bands"] = F.by_prompt_band(kr)
    f["checks"] = F.verdict(K, A, D, d["replays"]); f["slips"] = [r["t"] for r in kr if not r["reply_exact"]]; f["held_wrong"] = [r["t"] for r in kr if not r["exact"]]
    f["cache_calls"] = sum(1 for r in kr[9:] if (r.get("cache_tokens") or 0) > 0) / len(kr[9:]); f["max_prompt"] = max(r["prompt_tokens"] for r in kr)
    f["held_k"] = sum(1 for r in kr if r["exact"]); f["held_a"] = sum(1 for r in ar if r["exact"])
    fw = f["held_wrong"]; f["run_first"], f["run_last"] = fw[0], fw[-1]
    f["kinds_k"], f["kinds_a"] = F.by_kind(kr), F.by_kind(ar); f["ages_k"], f["ages_a"] = F.by_age(kr), F.by_age(ar)
    return f


def fz_lines(f):
    sa, sk, sd, s50, l100, ta, dep = f["sa"], f["sk"], f["sd"], f["s50"], f["l100"], f["ta"], f["depth"]; ks = sorted(dep); kr, ar = f["kr"], f["ar"]; sas = f["sa_same"]
    qa = [r for r in ar if r["kind"] == "query"]; yk, ny = F._recall(kr, 0, 10); ok_, no = F._recall(kr, 50, 10 ** 9); ya, nya = F._recall(ar, 0, 10); oa, noa = F._recall(ar, 50, 10 ** 9)
    fk, nk = F._flag_probe(kr); fa, na = F._flag_probe(ar); first = f["slips"][0]
    return {
        1: f"{len(kr)} turns, every one on file; the replay of the saved replies reproduced every flag",
        2: f"{len(ar)} steps; the replay reproduced every flag and every re-ask ({f['A']['summary']['reasked']} re-asks)",
        3: f"largest prompt {num(f['max_prompt'])} tokens",
        4: f"{pc(f['cache_calls'])} of the calls from the 10th turn had cached tokens; {pc(sk['cached_share'])} of all prompt tokens",
        5: f"{f['A']['summary']['refused']} steps refused after the re-asks, of {len(ar)}",
        6: f"{sum(1 for r in kr if r['parsed'])} of {len(kr)} replies had the right shape ({pc(sk['parsed'])})",
        7: f"held list exact on {f['held_a']} of {len(ar)} steps ({pc(f['held_a'] / len(ar))}); lowest tenth {pc(min(t['held_exact'] for t in ta))}",
        8: f"corrupted {sa['corrupted']} steps; largest distance from the true list {max(r['distance'] for r in ar)} cells",
        9: f"mean prompt of the last tenth {num(round(ta[-1]['prompt_per_step']))} against {num(round(ta[1]['prompt_per_step']))} in the second ({times(ta[-1]['prompt_per_step'], ta[1]['prompt_per_step'])} times); seconds a step {sec(ta[-1]['seconds_per_step'])} against {sec(ta[1]['seconds_per_step'])} ({times(ta[-1]['seconds_per_step'], ta[1]['seconds_per_step'])} times)",
        10: f"{sum(1 for r in qa if r['reply_exact'])} of {len(qa)} answers exact ({pc(sum(1 for r in qa if r['reply_exact']) / len(qa))})",
        11: f"turns 1 to 50: {pc(s50['reply_exact'])} exact",
        12: f"turns 1 to 50: {pc(s50['reply_exact'])} exact; last 100 turns: {pc(l100['reply_exact'])}; a drop of {(s50['reply_exact'] - l100['reply_exact']) * 100:.1f} points",
        13: f"last 100 turns: {pc(1 - l100['reply_ok'])} of the replies cannot be applied to the true list or have the wrong shape",
        14: f"the held list was corrupted (an item nobody mentioned, an unknown field or a value outside the allowed set) at {sum(1 for r in kr if r['corrupted'])} of {len(kr)} turns",
        15: f"the first reply that is not exactly right is turn {first} (prompt {num(kr[first - 1]['prompt_tokens'])})",
        16: f"exact answers about facts changed less than 10 turns earlier: {pc(yk)} (n {ny}); 50 or more turns earlier: {pc(ok_)} (n {no})",
        17: f"arm A: {pc(ya)} (n {nya}) against {pc(oa)} (n {noa})",
        18: f"answers exact where the right answer is unknown or ambiguous: arm A {pc(fa)} (n {na}); arm K {pc(fk)} (n {nk})",
        19: f"{sec(sd['seconds_per_step'])} s a call at a prompt of 150,000 tokens or more against {sec(sa['seconds_per_step'])} s a step with AXI ({times(sd['seconds_per_step'], sa['seconds_per_step'])} times)",
        20: f"first token, median: {sec(sd['first_p50'])} s against {sec(sa['first_p50'])} s ({times(sd['first_p50'], sa['first_p50'])} times)",
        21: f"exact steps an hour of calls: AXI {sas['exact_per_hour']:,.0f}, kept chat {sk['exact_per_hour']:,.0f} (the same {len(kr)} steps; {times(sas['exact_per_hour'], sk['exact_per_hour'])} times)",
        22: f"tokens computed for each exact step at a prompt of 150,000 tokens or more: kept chat {sd['computed_per_exact']:,.0f}; with AXI {sa['computed_per_exact']:,.0f} ({times(sd['computed_per_exact'], sa['computed_per_exact'])} times)",
        23: f"thinking tokens a turn: last 100 turns {l100['think_per_step']:.1f}, turns 1 to 50 {s50['think_per_step']:.1f}",
        24: f"exact share at the shallowest start ({ks[0]}): {pc(dep[ks[0]]['reply_exact'])}; at the deepest ({ks[-1]}): {pc(dep[ks[-1]]['reply_exact'])}",
    }


def fz_table_arms(f):
    sk, sd, sa, sas = f["sk"], f["sd"], f["sa"], f["sa_same"]; s50 = f["s50"]
    def row(name, s): return f"| {name} | {s['steps']} | {sec(s['seconds_per_step'])} | {s['steps_per_hour']:,.0f} | {s['exact_per_hour']:,.0f} | {sec(s['first_p50'])} | {sec(s['first_p95'])} | {num(round(s['prompt_per_step']))} | {pc(s['cached_share'])} | {s['think_per_step']:.1f} | {s['computed_per_exact']:,.0f} |"
    return "\n".join(["| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |", "|---|---|---|---|---|---|---|---|---|---|---|",
                      row("kept chat, all turns", sk), row("kept chat, turns 1 to 50", s50), row("kept chat, prompt of 150,000 tokens or more", sd), row("with AXI, all steps", sa), row("with AXI, steps 1 to 50", GL.stats(f["ar"][:50]))])


def fz_table_bands(f):
    rows = ["| prompt size (tokens) | turns | n | exact reply | held list exact | seconds a call | first token median | first token 95th | exact steps an hour | computed tokens per exact step |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in f["bands"]: rows.append(f"| {num(b['lo'])} to {num(min(b['hi'], 1_000_000))} | {b['first']} to {b['last']} | {b['steps']} | {pc(b['reply_exact'])} | {pc(b['held_exact'])} | {sec(b['seconds_per_step'])} | {sec(b['first_p50'])} | {sec(b['first_p95'])} | {b['exact_per_hour']:,.0f} | {b['computed_per_exact']:,.0f} |")
    return "\n".join(rows)


def fz_table_tenths(f):
    rows = ["| turns | kept chat: exact reply | kept chat: held list exact | kept chat: prompt a turn | kept chat: seconds a turn | AXI: held list exact | AXI: prompt a step | AXI: seconds a step | AXI: first token median |", "|---|---|---|---|---|---|---|---|---|"]
    for tk, ta in zip(f["tk"], f["ta"]):
        rows.append(f"| {tk['first']} to {tk['last']} | {pc(tk['reply_exact'])} | {pc(tk['held_exact'])} | {num(round(tk['prompt_per_step']))} | {sec(tk['seconds_per_step'])} | {pc(ta['held_exact'])} | {num(round(ta['prompt_per_step']))} | {sec(ta['seconds_per_step'])} | {sec(ta['first_p50'])} |")
    return "\n".join(rows)


def fz_table_kinds(f):
    ka = {b["kind"]: b for b in f["kinds_a"]}; rows = ["| kind of turn | n | kept chat: exact reply | kept chat: held list exact | AXI: exact reply |", "|---|---|---|---|---|"]
    for b in f["kinds_k"]: rows.append(f"| {b['kind']} | {b['n']} | {pc(b['exact'])} | {pc(b['held_exact'])} | {pc(ka[b['kind']]['exact'])} |")
    return "\n".join(rows)


def fz_table_depth(f):
    rows = ["| start (turn) | real turns | mean prompt (tokens) | exact reply | seconds a call | first token median |", "|---|---|---|---|---|---|"]
    for s, v in f["depth"].items(): rows.append(f"| {s} | {v['live_steps']} | {num(round(v['mean_prompt']))} | {pc(v['reply_exact'])} | {sec(v['seconds_per_step'])} | {sec(v['first_p50'])} |")
    return "\n".join(rows)


def fz_cross(f):
    """The first tenth of the kept chat from which every tenth has more seconds a turn than the mean seconds a step of AXI (all steps)."""
    a = f["sa"]["seconds_per_step"]; tk = f["tk"]
    for i in range(len(tk)):
        if all(t["seconds_per_step"] > a for t in tk[i:]): return tk[i]
    return None


def render_f(d):
    f = fz(d); ck = f["checks"]; yes = sum(1 for c in ck if c[3]); no = len(ck) - yes; nos = [c[0] for c in ck if not c[3]]; sk, sa, sd, kr, ar = f["sk"], f["sa"], f["sd"], f["kr"], f["ar"]; head = F.headline(ck)
    fw = f["held_wrong"]; xt = fz_cross(f); bad_turns = f["slips"]; kinds = {r["t"]: r for r in kr}
    return f"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_fuzzy.py` and again by `tests/test_arc_fuzzy_results.py`). The figures are recomputed from `docs/results/axi_gemini_fuzzy_seed20261122_default.json` by that test; none is typed by hand. One run of each arm, one seed, one job, on 22 Oct 2026.*

**What the page expected, and what came out.** Lines 1 to 6 (the guards and the instruments) all read yes, so by the page's rule the figures of the arms are claimed as read. Of the 24 lines **{yes} read yes and {no} read NO** (lines {', '.join(map(str, nos))}). Headline: **{head}**. The NO lines are findings, and most of them are the kept chat *not* doing what the page guessed it might: at {num(f['max_prompt'])} tokens and {len(kr)} turns of ambiguous text the kept chat did not drift and did not hallucinate, and its answers about old facts were within the page's 10-point threshold of those about new ones.

**In one paragraph, for a company that runs the model.** With AXI the held list was exact on all {len(ar)} steps ({pc(f['held_a'] / len(ar))}), at {num(round(sa['prompt_per_step']))} prompt tokens a step on average ({num(round(f['ta'][0]['prompt_per_step']))} in the first tenth, {num(round(f['ta'][-1]['prompt_per_step']))} in the last, as the list grew) and {sec(sa['seconds_per_step'])} s a step. The kept chat sent {num(round(sk['prompt_per_step']))} prompt tokens a turn on average ({times(sk['prompt_per_step'], sa['prompt_per_step'], 0)} times as many; {sk['prompt'] / 1e6:.1f} million tokens in all against {sa['prompt'] / 1e6:.2f} million) and its replies were exact on {len(kr) - len(bad_turns)} of {len(kr)} turns ({pc(sk['reply_exact'])}). The two slips mattered differently: the slip at turn {bad_turns[0]} (`{kr[bad_turns[0] - 1]['kind']}`: the reply changed the wrong item) was a change to the list, and a company that applies each reply as it comes **held a wrong list for {len(fw)} turns in a row, turns {f['run_first']} to {f['run_last']}**, until a later message happened to overwrite the cell; the list was exact on {pc(f['held_k'] / len(kr))} of the turns although {pc(sk['reply_exact'])} of the replies were exact. The slip at turn {bad_turns[1]} was a wrong answer to a question and changed nothing. On time, the kept chat went from {sec(f['s50']['first_p50'])} s to the first token (turns 1 to 50) to {sec(sd['first_p50'])} s once the prompt passed 150,000 tokens, against {sec(sa['first_p50'])} s with AXI ({times(sd['first_p50'], sa['first_p50'])} times), and from {sec(f['s50']['seconds_per_step'])} s to {sec(sd['seconds_per_step'])} s a call; at a short prompt the kept chat was the faster of the two.

## The 24 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(line_rows("F", ck, fz_lines(f))) + f"""

## Speed, latency and tokens

""" + fz_table_arms(f) + f"""

The seconds are the seconds of the calls, read at the notebook (the time to the first token includes the network from the notebook's machine). Tokens: *computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens; the page's stand-in for energy, **not joules**. The API gives no joules for the hardware. The kept chat was served {pc(sk['cached_share'])} of its prompt tokens from the service's cache, so its *computed* tokens are far fewer than its *sent* tokens ({num(sk['sent'])} sent, {num(sk['computed'])} computed); AXI, which sends a short fresh prompt every step, got no cache ({num(sa['sent'])} sent and {num(sa['computed'])} computed). In computed tokens the kept chat is therefore **{times(sk['computed_per_exact'], sa['computed_per_exact'])} times** AXI per exact step over the whole job, not the {times(sk['sent'], sa['sent'], 0)} times of the tokens sent.

## The kept chat, by the size of the prompt

""" + fz_table_bands(f) + f"""

* **Held list** is what a firm holds if it applies each reply as it comes (blind application). A reply that changes the wrong cell leaves it wrong until a later message overwrites that cell: here from turn {f['run_first']} to turn {f['run_last']} ({len(fw)} turns), at most {max(r['distance'] for r in kr)} cells from the true list. It was not *corrupted* in the page's sense (no item nobody mentioned, no unknown field, no value outside the allowed set): it was a valid list that was not the true one.
* **Crossing.** From the tenth of turns {xt['first']} to {xt['last']} on (prompts around {num(round(xt['prompt_per_step']))} tokens) every tenth of the kept chat took more seconds a turn than the mean seconds a step of AXI ({sec(sa['seconds_per_step'])}); before that the kept chat was the faster arm or about level, in the first 50 turns by {times(f['s50']['exact_per_hour'], GL.stats(ar[:50])['exact_per_hour'])} times in exact steps an hour.
* **Thinking.** With AXI the model thought {sa['think_per_step']:.0f} tokens a step on average (the model's own default thinking); the kept chat {sk['think_per_step']:.1f} a turn on average, {f['s50']['think_per_step']:.1f} over turns 1 to 50 and {f['l100']['think_per_step']:.1f} over the last 100. Line 23 reads yes only because the early value is tiny; it is not a climb that a company would plan around.

## By tenths of the job

""" + fz_table_tenths(f) + f"""

## By kind of turn

""" + fz_table_kinds(f) + f"""

{sum(1 for r in ar if not r['reply_exact'])} of the {len(ar)} replies of AXI were not exactly the oracle's; none changed the held list, which stayed exact on every step (the checks and the repair of the framework did what they are for); there were no re-asks and no refused replies.

## The depth probe (a perfect history)

""" + fz_table_depth(f) + f"""

Ten turns at each of four depths, after a history made of the oracle's own replies ({len(f['dr'])} replies, all exact; the first two depths were answered from the notebook's saved reply file and have no call time). With ten turns a depth this cannot tell a small rate of slips from none (line 24 reads NO under the page's rule).

## What this says, and what it does not

* **Said:** on this job, ambiguous text for {len(kr)} turns, the model alone gave exact replies on {len(kr) - len(bad_turns)} of {len(kr)} turns, so on this job the kept chat did not fail at this length; what the run shows is that the **one** wrong change that did come stayed in the state for as long as no later message overwrote it ({len(fw)} turns), because a state kept in a chat has no check against the truth. With the framework holding the state the list was exact on every one of {len(ar)} steps, at a prompt that grew only with the list, and a step that took about the same time all through (between {sec(min(t['seconds_per_step'] for t in f['ta']))} s and {sec(max(t['seconds_per_step'] for t in f['ta']))} s in every tenth). The time to the first token of the kept chat grew with its prompt ({sec(f['bands'][0]['first_p50'])} s at the first band, {sec(f['bands'][-1]['first_p50'])} s at the last) while that of the AXI arm did not.
* **Not said:** that the kept chat drifts, hallucinates or forgets with age (it did none of these beyond the page's thresholds here: lines 12, 13, 14, 15 and 16 read NO); that AXI is faster at every length (it was not at a short prompt, and line 21 reads NO: {times(f['sa_same']['exact_per_hour'], sk['exact_per_hour'])} times the exact steps an hour, under the 1.5 the page asked); that the two slips would repeat on another seed; anything about another model, another job or any company's service; joules.
* **The reading rule** of the page is used as written. The headline is {head}, as the rule gives it (lines 7 and 8 read yes). Line 9 reads NO because the prompt of AXI grew {times(f['ta'][-1]['prompt_per_step'], f['ta'][1]['prompt_per_step'])} times from the second tenth to the last (the list it shows grows with the job) against the 1.3 the page allowed, while the kept chat's grew {times(f['tk'][-1]['prompt_per_step'], f['tk'][1]['prompt_per_step'], 1)} times.
"""


# ================================================================ ARC-AGI-3 (test 24)
def _mean(xs):
    xs = list(xs); return sum(xs) / len(xs)


def ar_fig(d):
    K, A = d["K"], d["A"]; kr, ar = K["rows"], A["rows"]; f = dict(K=K, A=A, kr=kr, ar=ar, sk=R.summarize(kr), sa=R.summarize(ar), deep=GL.window(kr, 100_000)); f["sd"] = GL.stats(f["deep"])
    f["checks"] = R.verdict(K, A, d["replays"]); f["max_prompt"] = max(r["prompt_tokens"] or 0 for r in kr); f["sent_k"], f["sent_a"] = GL.tokens_sent(kr), GL.tokens_sent(ar)
    f["bad"] = [r for r in kr if not r["parsed"]]; f["bad_empty"] = [r for r in f["bad"] if not r["text"].strip()]; f["bad_cut"] = [r for r in f["bad"] if "MAX_TOKENS" in str(r["finish"])]; f["bad_other"] = [r for r in f["bad"] if r["text"].strip() and "MAX_TOKENS" not in str(r["finish"])]
    f["early_a"], f["late_a"] = R._m_between(ar, 6, 25), R._m_between(ar, 41, 60); f["early_k"], f["late_k"] = R._m_between(kr, 1, 20), R._m_between(kr, 41, 60)
    big = [r for r in kr if (r.get("prompt_tokens") or 0) >= 10_000]; f["cache_big"] = sum(1 for r in big if (r.get("cache_tokens") or 0) > 0) / len(big)
    return f


def ar_lines(f):
    sk, sa, sd, kr, ar = f["sk"], f["sa"], f["sd"], f["kr"], f["ar"]; ea, la, ek, lk = f["early_a"], f["late_a"], f["early_k"], f["late_k"]; shr = R._share
    return {
        1: f"{len(kr)} moves in {len(sk['games'])} games, every one on file; the replay of the saved replies into fresh copies of the games reproduced every flag (read from the saved file: the games come from the ARC toolkit, which this test does not have)",
        2: f"{len(ar)} moves; the replay reproduced every flag and every re-ask ({sa['reasked']} moves asked again)",
        3: f"largest prompt {num(f['max_prompt'])} tokens",
        4: f"{pc(f['cache_big'])} of the calls with a prompt of 10,000 tokens or more had cached tokens",
        5: f"{sa['unusable']} moves of {len(ar)} ended with a reply that could not be used",
        6: f"{len(kr) - len(f['bad'])} of {len(kr)} replies were one JSON object with an action ({pc(sk['all']['parsed'])})",
        7: f"mean prompt of moves 41 to 60 {num(round(_mean(r['prompt_tokens'] for r in la)))} against {num(round(_mean(r['prompt_tokens'] for r in ea)))} for moves 6 to 25 ({times(_mean(r['prompt_tokens'] for r in la), _mean(r['prompt_tokens'] for r in ea))} times)",
        8: f"tokens sent: arm A {num(f['sent_a'])}, arm K {num(f['sent_k'])} (arm A sent {pc(f['sent_a'] / f['sent_k'])} of arm K's; {times(f['sent_k'], f['sent_a'], 1)} times fewer)",
        9: f"effective moves 1 to 20: {pc(shr(ek, 'reply_exact'))}; moves 41 to 60: {pc(shr(lk, 'reply_exact'))}",
        10: f"moves that repeated one that had done nothing from the same screen: arm K {pc(sk['repeat_noop'], 2)}, arm A {pc(sa['repeat_noop'], 2)}",
        11: f"{pc(1 - sk['legal'])} of arm K's replies were not an action the game offered ({len(f['bad'])} of {len(kr)})",
        12: f"effective moves: arm A {pc(sa['effective'])}, arm K {pc(sk['effective'])} ({(sa['effective'] - sk['effective']) * 100:+.1f} points)",
        13: f"levels completed over the games: arm A {sa['levels']}, arm K {sk['levels']}",
        14: f"arm A completed {sa['levels']} level in {sum(1 for g in sa['games'] if g['levels'] >= 1)} game of {len(sa['games'])}",
        15: f"arm K completed {sk['levels']} levels in {sum(1 for g in sk['games'] if g['levels'] >= 1)} games of {len(sk['games'])}",
        16: f"{sec(sd['seconds_per_step'])} s a call at a prompt of 100,000 tokens or more against {sec(sa['all']['seconds_per_step'])} s a move with AXI ({times(sd['seconds_per_step'], sa['all']['seconds_per_step'])} times)",
        17: f"first token, median: {sec(sd['first_p50'])} s against {sec(sa['all']['first_p50'])} s ({times(sd['first_p50'], sa['all']['first_p50'])} times)",
        18: f"effective moves an hour of calls: AXI {sa['all']['exact_per_hour']:,.0f}, kept chat {sk['all']['exact_per_hour']:,.0f} ({times(sa['all']['exact_per_hour'], sk['all']['exact_per_hour'])} times)",
        19: f"tokens computed for each effective move at a prompt of 100,000 tokens or more: kept chat {sd['computed_per_exact']:,.0f}; with AXI {sa['all']['computed_per_exact']:,.0f} ({times(sd['computed_per_exact'], sa['all']['computed_per_exact'])} times)",
    }


def ar_table_arms(f):
    sk, sa, sd = f["sk"]["all"], f["sa"]["all"], f["sd"]
    def row(name, s): return f"| {name} | {s['steps']} | {sec(s['seconds_per_step'])} | {s['steps_per_hour']:,.0f} | {s['exact_per_hour']:,.0f} | {sec(s['first_p50'])} | {sec(s['first_p95'])} | {num(round(s['prompt_per_step']))} | {pc(s['cached_share'])} | {s['think_per_step']:,.0f} | {s['computed_per_exact']:,.0f} |"
    return "\n".join(["| arm | moves | seconds a move | moves an hour | effective moves an hour | first token median | first token 95th | prompt tokens a move | served from the cache | thinking tokens a move | computed tokens per effective move |", "|---|---|---|---|---|---|---|---|---|---|---|",
                      row("kept chat, all moves", sk), row("kept chat, prompt of 100,000 tokens or more", sd), row("with AXI, all moves", sa)])


def ar_table_games(f):
    rows = ["| game | levels to win | arm K: levels | arm K: effective | arm K: largest prompt | arm A: levels | arm A: effective | arm A: largest prompt |", "|---|---|---|---|---|---|---|---|"]
    for gk, ga in zip(f["sk"]["games"], f["sa"]["games"]): rows.append(f"| {gk['game']} | {gk['win_levels']} | {gk['levels']} | {pc(gk['effective'])} | {num(gk['prompt_max'])} | {ga['levels']} | {pc(ga['effective'])} | {num(ga['prompt_max'])} |")
    return "\n".join(rows)


def render_r(d):
    f = ar_fig(d); ck = f["checks"]; yes = sum(1 for c in ck if c[3]); no = len(ck) - yes; nos = [c[0] for c in ck if not c[3]]; sk, sa, sd = f["sk"], f["sa"], f["sd"]; head = R.headline(ck); kr, ar = f["kr"], f["ar"]
    bad = f["bad"]; thinks = sorted(r["think"] for r in f["bad_empty"])
    return f"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_arc.py` and again by `tests/test_arc_fuzzy_results.py`). The figures are recomputed from `docs/results/axi_gemini_arc_default.json` by that test; none is typed by hand. One run of each arm, four of the 25 public games, 60 moves each, on 23 Oct 2026.*

**What the page expected, and what came out.** Of the 19 lines **{yes} read yes and {no} read NO** (lines {', '.join(map(str, nos))}). Lines 1 to 5 (the guards and the first instruments) read yes. **Line 6 reads NO**: {len(bad)} of the {len(kr)} replies of the kept chat ({pc(len(bad) / len(kr))}) were not one JSON object with an action, so by the page's rule *no claim is made until the cause is found*. The cause is found in the saved rows and belongs to the kept chat itself, not to the instrument: {len(f['bad_empty'])} of the {len(bad)} unusable replies are **empty** (the call ended after the model's thinking, {thinks[0]:,} to {thinks[-1]:,} tokens of it, with no reply written), {len(f['bad_cut'])} were **cut off at the token limit** in the middle of a long analysis, and {len(f['bad_other'])} were a reply with the wrong wording. All {len(bad)} came at a prompt of {min(r['prompt_tokens'] for r in bad):,} tokens or more; none came from AXI, whose replies were all usable. So the figures are given as read, with this note. Headline: **{head}**.

**In one paragraph, for a company that runs the model.** On these four games AXI sent **{num(f['sent_a'])} tokens against {num(f['sent_k'])}** for the kept chat ({times(f['sent_k'], f['sent_a'], 1)} times fewer), at a prompt that stayed near {num(round(sa['all']['prompt_per_step']))} tokens a move where the kept chat's reached {num(f['max_prompt'])}; every one of its replies was usable ({pc(sa['legal'])} legal) and {pc(sa['effective'])} of its moves changed the screen, against {pc(sk['legal'])} and {pc(sk['effective'])} for the kept chat. **It was not faster and it did not win more levels**: it completed {sa['levels']} level against {sk['levels']}, took {sec(sa['all']['seconds_per_step'])} s a move against {sec(sk['all']['seconds_per_step'])}, and made {sa['all']['exact_per_hour']:,.0f} effective moves an hour against {sk['all']['exact_per_hour']:,.0f}; the model thought {sa['all']['think_per_step']:,.0f} tokens a move with AXI and {sk['all']['think_per_step']:,.0f} with the kept chat. Computed tokens for each effective move were about level ({sd['computed_per_exact']:,.0f} for the kept chat at a prompt of 100,000 tokens or more, {sa['all']['computed_per_exact']:,.0f} for AXI) because the service served {pc(sk['all']['cached_share'])} of the kept chat's prompt tokens from its cache.

## The 19 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(line_rows("R", ck, ar_lines(f))) + f"""

## Speed, latency and tokens

""" + ar_table_arms(f) + f"""

The seconds are the seconds of the calls, read at the notebook (the time to the first token includes the network from the notebook's machine). Tokens: *computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens; the page's stand-in for energy, **not joules**. The API gives no joules for the hardware.

## By game

""" + ar_table_games(f) + f"""

*Levels* are the levels the game itself counted as completed within 60 moves; *effective* is the share of moves that were an action the game offered and changed the screen. 60 moves is a short play of a game that asks for up to {max(g['win_levels'] for g in sk['games'])} levels; the page makes no claim about how either arm would do on a longer play, on the other public games or on the private ones.

## The unusable replies of the kept chat

{len(bad)} replies, all of arm K: {len(f['bad_empty'])} empty, {len(f['bad_cut'])} cut at the token limit, {len(f['bad_other'])} with the wrong wording. In the games {', '.join(sorted({r['game'] for r in bad}))}; the smallest prompt among them was {min(r['prompt_tokens'] for r in bad):,} tokens. Each one still counts as a move for the kept chat (a wasted one), because the chat is not asked again; AXI asks again up to {R.A_REASKS} times and none of its moves was lost.

## What this says, and what it does not

* **Said:** on this set of games the state-holding arm is **far cheaper to run** ({times(f['sent_k'], f['sent_a'], 1)} times fewer tokens sent, a prompt that does not grow) and **ended no move with an unusable reply** ({sa['legal'] * 100:.0f} % legal, against {pc(sk['legal'])}), and it wastes fewer moves ({pc(sa['effective'])} effective against {pc(sk['effective'])}).
* **Not said, and the run says the opposite:** that AXI makes the model better at the games. It completed {sa['levels']} level where the kept chat completed {sk['levels']}; it was {times(sa['all']['seconds_per_step'], sk['all']['seconds_per_step'])} times slower a move, and the model thought {times(sa['all']['think_per_step'], sk['all']['think_per_step'], 1)} times as many tokens a move with AXI ({sa['all']['think_per_step']:,.0f} against {sk['all']['think_per_step']:,.0f}); the page did not guess this and the run does not say why. Lines 9, 10, 13, 16, 17, 18 and 19 read NO.
* **Why it matters:** the games are short and the kept chat's drift is small at 60 moves (line 9 reads NO). What the run shows is the cost side of the comparison and a better floor on unusable moves, not an advantage in play.
"""


# ================================================================ ARC-AGI-3 with the explorer (test 25)
def ab_fig(d):
    f = dict(B=d["B"], A=d["A"], K=d["K"], sb=B.summarize(d["B"]["rows"]), sa=B.summarize(d["A"]["rows"]), sk=B.summarize(d["K"]["rows"])); f["checks"] = B.verdict(d["B"], d["A"], d["K"], d["replays"])
    f["br"], f["ar"], f["kr"] = d["B"]["rows"], d["A"]["rows"], d["K"]["rows"]; return f


def ab_lines(f):
    sb, sa, sk = f["sb"], f["sa"], f["sk"]; eff = lambda s: s["computed"] / max(1, round(s["effective"] * s["moves"]))
    return {
        1: f"{sb['moves']} moves, every one on file; {sb['forced']} played by the framework; the replay reproduced every saved flag (read from the saved file: the games come from the ARC toolkit)",
        2: f"{sa['moves']} moves of arm A (the rows of Test 24); the replay reproduced every saved flag",
        3: f"{sb['forced']} of {sb['moves']} moves played by the framework ({pc(sb['forced'] / sb['moves'])})",
        4: f"{sb['unusable']} of the model's {sb['moves'] - sb['forced']} moves ended with a reply that could not be used ({sb['reasked']} moves asked again)",
        5: f"effective moves: arm B {pc(sb['effective'])}, arm A {pc(sa['effective'])} ({(sb['effective'] - sa['effective']) * 100:+.1f} points)",
        6: f"moves that repeated one that had done nothing from the same screen: arm B {sb['repeat_noop']}, arm A {sa['repeat_noop']}",
        7: f"screens never seen before per 1,000 computed tokens: arm B {sb['score']:.3f}, arm A {sa['score']:.3f} ({times(sb['score'], sa['score'])} times)",
        8: f"screens never seen before reached: arm B {sb['discovered']}, arm A {sa['discovered']}",
        9: f"levels completed over the games: arm B {sb['levels']}, arm A {sa['levels']}",
        10: f"arm B completed {sb['levels']} level in {sum(1 for g in sb['games'] if g['levels'] >= 1)} game of {len(sb['games'])}",
        11: f"levels: arm B {sb['levels']}, arm A {sa['levels']}",
        12: f"effective moves an hour of calls: arm B {sb['per_hour']:,.0f}, arm A {sa['per_hour']:,.0f} ({times(sb['per_hour'], sa['per_hour'])} times)",
        13: f"thinking tokens a call: arm B {sb['think_per_call']:,.0f}, arm A {sa['think_per_call']:,.0f} ({times(sb['think_per_call'], sa['think_per_call'])} times)",
        14: f"tokens computed for each effective move: arm B {eff(sb):,.0f}, arm A {eff(sa):,.0f} ({times(eff(sb), eff(sa))} times)",
        15: "no move was played by the framework, so there is nothing to read (the conductor never fired)" if not sb["forced"] else f"{sb['forced_changed']} of {sb['forced']} framework moves changed the screen",
        16: f"levels: arm B {sb['levels']}, arm K {sk['levels']}",
        17: f"tokens sent: arm B {num(sb['sent'])}, arm K {num(sk['sent'])} (arm B sent {pc(sb['sent'] / sk['sent'])} of arm K's; {times(sk['sent'], sb['sent'], 1)} times fewer)",
    }


def ab_line_rows(checks, lf):
    pr = preds("B")
    return [f"| {i} | {pr[i]} % | [{tag}] {lf[i]} | **{'not read' if c is None else ('yes' if c else 'NO')}** |" for i, tag, text, c in checks]


def ab_table_arms(f):
    rows = ["| arm | moves | played by the framework | effective moves | repeat no-ops | screens never seen before | per 1,000 computed tokens | levels | thinking tokens a call | seconds a move | effective moves an hour | computed tokens | tokens sent |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("B (explorer)", f["sb"]), ("A (Test 24)", f["sa"]), ("K (Test 24, kept chat)", f["sk"])):
        rows.append(f"| {name} | {s['moves']} | {s['forced']} | {pc(s['effective'])} | {s['repeat_noop']} | {s['discovered']} | {s['score']:.3f} | {s['levels']} | {s['think_per_call']:,.0f} | {sec(s['all']['seconds_per_step'])} | {s['per_hour']:,.0f} | {num(s['computed'])} | {num(s['sent'])} |")
    return "\n".join(rows)


def ab_table_games(f):
    rows = ["| game | levels to win | arm B: levels | arm B: effective | arm A: levels | arm A: effective | arm K: levels | arm K: effective |", "|---|---|---|---|---|---|---|---|"]
    for gb, ga, gk in zip(f["sb"]["games"], f["sa"]["games"], f["sk"]["games"]): rows.append(f"| {gb['game']} | {gb['win_levels']} | {gb['levels']} | {pc(gb['effective'])} | {ga['levels']} | {pc(ga['effective'])} | {gk['levels']} | {pc(gk['effective'])} |")
    return "\n".join(rows)


def render_b(d):
    f = ab_fig(d); ck = f["checks"]; yes = sum(1 for c in ck if c[3] is True); no = sum(1 for c in ck if c[3] is False); nr = sum(1 for c in ck if c[3] is None); nos = [c[0] for c in ck if c[3] is False]; sb, sa, sk = f["sb"], f["sa"], f["sk"]; head = B.headline(ck)
    nrs = [c[0] for c in ck if c[3] is None]
    return f"""
## Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_gemini_arc_b.py` and again by `tests/test_arc_fuzzy_results.py`). The figures are recomputed from `docs/results/axi_gemini_arc_b_default.json` by that test; none is typed by hand. Arm B was run once on 24 Oct 2026; arms A and K are the rows of Test 24 (the same rows, checked equal by the test).*

**What the page expected, and what came out.** Lines 1 to 4 (the guards and the instruments) all read yes, so by the page's rule the figures of the arms are claimed as read. Of the 17 lines **{yes} read yes, {no} read NO** (lines {', '.join(map(str, nos))}) **and {nr} was not read** (line {', '.join(map(str, nrs))}: the framework never played a move itself). Headline: **{head}**. The explorer did not meet the two lines that decide the headline (5 and 7), though on the cost figures it moved in the same direction (lines 13 and 14 read yes).

**In one paragraph, for a company that runs the model.** Adding the object view and the list of untried moves to arm A changed no rule of the framework and no line of its core. On the same four games, the same 240 moves, it gave {pc(sb['effective'])} effective moves against {pc(sa['effective'])}, {sb['discovered']} screens never seen before against {sa['discovered']}, **the same level count** ({sb['levels']} against {sa['levels']}, won in a different game each: {', '.join(g['game'] for g in sb['games'] if g['levels'])} for arm B, {', '.join(g['game'] for g in sa['games'] if g['levels'])} for arm A; arm K, the kept chat, had {sk['levels']}), and **less work for the same result**: {sb['think_per_call']:,.0f} thinking tokens a call against {sa['think_per_call']:,.0f} ({pc(1 - sb['think_per_call'] / sa['think_per_call'], 0)} fewer), {num(sb['computed'])} computed tokens against {num(sa['computed'])}, {sec(sb['all']['seconds_per_step'])} s a move against {sec(sa['all']['seconds_per_step'])}, {sb['per_hour']:,.0f} effective moves an hour against {sa['per_hour']:,.0f}. Against the kept chat it sent {times(sk['sent'], sb['sent'], 1)} times fewer tokens. The differences are small, the games few and the run single, so they are read as a direction, not as a size. The conductor, the part that plays a move itself when the model is stuck, **never fired**: the model was never stuck by the page's rule (4 moves without a new screen, or one screen seen 3 times), so the page says nothing about it.

## The 17 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(ab_line_rows(ck, ab_lines(f))) + f"""

## The three arms

""" + ab_table_arms(f) + f"""

*Computed* is the prompt tokens not served from the service's cache plus the reply and the thinking tokens: the page's stand-in for energy, **not joules**. The kept chat's computed tokens are low next to its sent tokens because the service served most of its prompt from the cache. The seconds are the seconds of the calls, read at the notebook.

## By game

""" + ab_table_games(f) + f"""

## What this says, and what it does not

* **Said:** the explorer's additions are cheap and changed the outcome by little: the same number of levels, slightly more effective moves ({(sb['effective'] - sa['effective']) * 100:+.1f} points), fewer thinking tokens a call, fewer computed tokens and a faster move, with no change to the core of the framework, and {sa['discovered'] - sb['discovered']} screens fewer never seen before. None of the differences reaches the size the page set as its threshold: it asked for 3 points on wasted moves and 25 percent on progress per token, and got {(sb['effective'] - sa['effective']) * 100:+.1f} points and {(sb['score'] / sa['score'] - 1) * 100:.0f} percent.
* **Not said:** that the explorer makes the model solve more ARC levels (it did not: {sb['levels']} level, as arm A; the kept chat had {sk['levels']}); that the conductor works (it never ran); that the object view is the invariant address frame of Sir's notes (whether the notes match his writing is for Sir to confirm); anything about the private games, other models, a longer play or any company's service.
* **What is open:** a stall rule that also fires when the same action is sent again and again (Test 24 saw one game where one action was sent 34 times of 60) would give the conductor a case to act on. That is a new test with its own page, not a change to this one.
"""


# ================================================================ the company side of the front page (generated from the same figures)
README = os.path.join(HERE, "..", "..", "README.md")
R_START, R_END = "<!-- fuzzy-and-arc:start -->", "<!-- fuzzy-and-arc:end -->"
REF = os.path.join(DOCS, "AXI_TECHNICAL_REFERENCE.md")
M_START, M_END = "<!-- measured:start -->", "<!-- measured:end -->"


def company(df, dr, db):
    f, a, b = fz(df), ar_fig(dr), ab_fig(db); sk, sa, sd = f["sk"], f["sa"], f["sd"]; rk, ra, rd = a["sk"], a["sa"], a["sd"]; xt = fz_cross(f)
    return dict(f=f, a=a, b=b, sk=sk, sa=sa, sd=sd, rk=rk, ra=ra, rd=rd, xt=xt)


def readme_section(df, dr, db):
    c = company(df, dr, db); f, a, b = c["f"], c["a"], c["b"]; sk, sa, sd, rk, ra, rd = c["sk"], c["sa"], c["sd"], c["rk"], c["ra"], c["rd"]; kr, ar = f["kr"], f["ar"]; bs = b["sb"]; fw = f["held_wrong"]
    return f"""{R_START}
## For a company that runs the model: three more runs, and the whole picture (22 to 24 Oct 2026)

Gemini 3.8 Flash again, on three jobs that are not the dependency graph of the long session: a **400-turn conversation in plain, ambiguous sentences** (Test 23), **four ARC-AGI-3 games** (Test 24) and the same games **with a host-owned explorer** (Test 25). Each job was run once with the chat kept and once with AXI holding the state. The pages with their pre-written reading lines are in [`docs/`](docs/); every figure below is recomputed from the saved files by `python/tests/test_arc_fuzzy_results.py`. The full technical account is [`docs/AXI_TECHNICAL_REFERENCE.md`](docs/AXI_TECHNICAL_REFERENCE.md).

### Where AXI changes the numbers

| What a company counts | Without AXI (chat kept) | With AXI | Run |
|---|---|---|---|
| Prompt tokens sent a step, conversation job | {num(round(sk['prompt_per_step']))} on average, {num(round(f['tk'][-1]['prompt_per_step']))} by the end | **{num(round(sa['prompt_per_step']))}** ({times(sk['prompt_per_step'], sa['prompt_per_step'], 0)} times fewer) | Test 23 |
| Tokens sent in all, conversation job | {sk['sent'] / 1e6:.1f} million | **{sa['sent'] / 1e6:.2f} million** ({times(sk['sent'], sa['sent'], 0)} times fewer) | Test 23 |
| Tokens sent in all, four ARC games | {a['sent_k'] / 1e6:.1f} million | **{a['sent_a'] / 1e6:.2f} million** ({times(a['sent_k'], a['sent_a'], 1)} times fewer) | Test 24 |
| The list a firm holds if it applies each reply as it comes | exact on {pc(f['held_k'] / len(kr))} of the turns: one wrong change at turn {f['slips'][0]} stayed for **{len(fw)} turns** | **exact on {f['held_a']} of {len(ar)} steps**, distance 0 | Test 23 |
| First token, median, once the prompt passes 150,000 tokens | {sec(sd['first_p50'])} s ({sec(f['s50']['first_p50'])} s in the first 50 turns) | **{sec(sa['first_p50'])} s, flat** ({times(sd['first_p50'], sa['first_p50'])} times lower) | Test 23 |
| Seconds a step at that prompt size | {sec(sd['seconds_per_step'])} | **{sec(sa['seconds_per_step'])}** ({times(sd['seconds_per_step'], sa['seconds_per_step'])} times faster) | Test 23 |
| Tokens computed per exact step at that prompt size | {sd['computed_per_exact']:,.0f} | **{sa['computed_per_exact']:,.0f}** ({times(sd['computed_per_exact'], sa['computed_per_exact'])} times fewer) | Test 23 |
| Replies that could not be used, four ARC games | {len(a['bad'])} of {len(a['kr'])} ({pc(len(a['bad']) / len(a['kr']))}) | **{ra['unusable']} of {len(a['ar'])}** (asked again up to {R.A_REASKS} times) | Test 24 |
| Moves that changed the screen, four ARC games | {pc(rk['effective'])} | **{pc(ra['effective'])}** | Test 24 |
| Tokens sent, ARC games with the explorer | {num(b['sk']['sent'])} (kept chat) | **{num(bs['sent'])}** ({times(b['sk']['sent'], bs['sent'], 1)} times fewer) | Test 25 |

### Where it does not, and what the runs say instead

| What a company counts | What was measured | Run |
|---|---|---|
| Does the kept chat drift or hallucinate on ambiguous text at this length? | **No.** {len(kr) - len(f['slips'])} of {len(kr)} replies were exact ({pc(sk['reply_exact'])}), none was corrupted, and a fact changed 50 or more turns earlier was answered right {round(F._recall(kr, 50, 10 ** 9)[0] * F._recall(kr, 50, 10 ** 9)[1])} times of {F._recall(kr, 50, 10 ** 9)[1]}. The risk is not the rate of slips but that **one** slip stays in the state until something overwrites it. | Test 23 |
| Speed at a short prompt | The kept chat was the faster arm in the first 50 turns ({f['s50']['exact_per_hour']:,.0f} exact steps an hour against {GL.stats(ar[:50])['exact_per_hour']:,.0f}). From turns {c['xt']['first']} to {c['xt']['last']} on it was slower; over all 400 steps AXI made {times(f['sa_same']['exact_per_hour'], sk['exact_per_hour'])} times the exact steps an hour (the page asked for 1.5). | Test 23 |
| Speed on ARC-AGI-3 | **AXI was slower**: {sec(ra['all']['seconds_per_step'])} s a move against {sec(rk['all']['seconds_per_step'])}, {ra['all']['exact_per_hour']:,.0f} effective moves an hour against {rk['all']['exact_per_hour']:,.0f}. The model thought {ra['all']['think_per_step']:,.0f} tokens a move with AXI and {rk['all']['think_per_step']:,.0f} without. | Test 24 |
| Levels completed on ARC-AGI-3 | **AXI did not complete more**: {ra['levels']} level against {rk['levels']} (60 moves a game, four games). The explorer did not add one ({bs['levels']}). | Tests 24, 25 |
| Computed tokens on ARC-AGI-3 | About level ({rd['computed_per_exact']:,.0f} for the kept chat at a prompt of 100,000 tokens or more, {ra['all']['computed_per_exact']:,.0f} for AXI): the service served {pc(rk['all']['cached_share'])} of the kept chat's prompt from its cache. | Test 24 |
| The explorer's conductor | **Never fired** ({bs['forced']} moves played by the framework): the model was never stuck by the rule. The object view and the untried list gave {pc(bs['effective'])} effective moves against {pc(ra['effective'])} and {pc(1 - bs['think_per_call'] / b['sa']['think_per_call'], 0)} fewer thinking tokens a call, below the page's thresholds. | Test 25 |

### How to read the two tables together

AXI is a layer for the company that runs the model, and the runs show what it is for: **the state is kept by the host, so it is exact at every step, the prompt does not grow with the job, and the first token and the step time do not grow with it.** What it costs the model on a hard open-ended task is thinking time, and the ARC-AGI-3 runs show that cost; they do not show a gain in play. Both facts are measured and both are on the pages. A long job where a wrong state is expensive is the case the first table speaks to; a short, open-ended job like these ARC games is the case the second table speaks to.

*Scope: Gemini 3.8 Flash, three jobs, one run of each arm, one seed (four of the 25 public ARC-AGI-3 games, 60 moves each). Joules are not measured; tokens computed and seconds of calls are the stand-in. The ARC pages are not a leaderboard entry. Nothing here is a statement about any company's service.*
{R_END}"""


def growth(f):
    """The kept chat's prompt grows by about the same number of tokens each turn: first prompt, tokens added a turn, and the sum a straight line through them gives against the sum counted."""
    kr = f["kr"]; n = len(kr); p0 = kr[0]["prompt_tokens"]; g = (kr[-1]["prompt_tokens"] - p0) / (n - 1)
    return dict(n=n, p0=p0, g=g, fitted=n * p0 + g * n * (n - 1) / 2, counted=sum(r["prompt_tokens"] for r in kr), at_2n=2 * n * p0 + g * (2 * n) * (2 * n - 1) / 2, ax_per_step=f["sa"]["prompt_per_step"])


def measured_block(df, dr, db):
    c = company(df, dr, db); f, a, b = c["f"], c["a"], c["b"]; d22 = L22.load(); f22 = L22.figures(d22); g = growth(f)
    return f"""{M_START}
### Test 22, the long session on a dependency graph (1,500 steps with AXI, 457 turns of the kept chat)

{L22.table_arms(f22)}

### Test 23, the conversation in ambiguous text (400 turns)

{fz_table_arms(f)}

{fz_table_tenths(f)}

### Test 24, four ARC-AGI-3 games, 60 moves each

{ar_table_arms(a)}

{ar_table_games(a)}

### Test 25, the same games with the host-owned explorer

{ab_table_arms(b)}

### The arithmetic used in section 5 (from the rows of Test 23)

The prompt of the kept chat grew by about {g['g']:,.0f} tokens a turn from {num(g['p0'])} tokens at turn 1. A straight line with that growth gives {num(round(g['fitted']))} prompt tokens over {g['n']} turns; the rows add up to {num(g['counted'])} ({abs(g['counted'] / g['fitted'] - 1) * 100:.1f} % apart). With AXI the prompt a step was {num(round(g['ax_per_step']))} tokens on average. The sum of a growing prompt grows with the square of the number of turns: the same line run on to {2 * g['n']} turns gives {num(round(g['at_2n']))} tokens, {g['at_2n'] / g['fitted']:.1f} times as many for twice the turns, while a step that does not grow gives twice as many. This second figure is arithmetic on a straight line, not a run.
{M_END}"""


# ================================================================ the tests
def test_the_saved_files_are_the_runs_and_the_lines_are_the_ones_the_notebooks_saved():
    for k in "FRB": assert hashlib.sha256(open(FILE[k], "rb").read()).hexdigest() == SHA_FILE[k], k
    d = load("F"); K, A, D = d["K"], d["A"], d["D"]
    assert K["meta"]["seed"] == A["meta"]["seed"] == D["meta"]["seed"] == SEED_F and K["meta"]["model"] == "gemini-3.8-flash" and K["meta"]["thinking"] is None
    F.replay(K["rows"], SEED_F, "K"); F.replay(A["rows"], SEED_F, "A"); F.replay_depth(D["rows"], SEED_F)
    assert d["replays"] == {"K": True, "A": True} and len(K["rows"]) == 400 and len(A["rows"]) == 400 and K["stopped"] is None
    assert [tuple(c) for c in d["checks"]] == [tuple(c) for c in F.verdict(K, A, D, d["replays"])]
    assert [c[3] for c in d["checks"]] == [True] * 8 + [False, True, True, False, False, False, False, False, True, False, True, True, False, True, True, False] and F.headline(d["checks"]) == "AXI HOLDS ON FUZZY TEXT"
    r = load("R"); assert r["K"]["meta"]["model"] == r["A"]["meta"]["model"] == "gemini-3.8-flash" and r["replays"] == {"K": True, "A": True} and r["K"]["meta"]["games"] == r["A"]["meta"]["games"] == list(R.GAMES)
    assert [tuple(c) for c in r["checks"]] == [tuple(c) for c in R.verdict(r["K"], r["A"], r["replays"])]
    assert [c[3] for c in r["checks"]] == [True] * 5 + [False, True, True, False, False, True, True, False, True, True, False, False, False, False] and R.headline(r["checks"]).startswith("MIXED") and R.claims_allowed(r["checks"])["base"] is False
    b = load("B"); assert b["replays"] == {"B": True, "A": True, "K": True} and b["B"]["meta"]["model"] == "gemini-3.8-flash" and b["A"]["rows"] == r["A"]["rows"] and b["K"]["rows"] == r["K"]["rows"]
    assert [tuple(c) for c in b["checks"]] == [tuple(c) for c in B.verdict(b["B"], b["A"], b["K"], b["replays"])]
    assert [c[3] for c in b["checks"]] == [True] * 4 + [False, True, False, False, True, True, False, False, True, True, None, False, True] and B.headline(b["checks"]).startswith("MIXED") and B.claims_allowed(b["checks"])["base"] is True


def test_the_figures_on_the_pages_are_recounted_by_hand_from_the_rows():
    nr = lambda xs, q: sorted(xs)[int(round(q * (len(xs) - 1)))]        # nearest rank, the way the page reads a median
    d = load("F"); f = fz(d); kr, ar, dr = f["kr"], f["ar"], f["dr"]
    assert len(kr) == 400 and len(ar) == 400 and len(dr) == 40 and all(r["reply_exact"] for r in dr)
    assert [r["t"] for r in kr if r["reply_exact"] is not True] == [191, 370] == f["slips"]
    held_wrong = [r["t"] for r in kr if r["exact"] is not True]; assert held_wrong == list(range(191, 331)) and len(held_wrong) == 140 and max(r["distance"] for r in kr) == 2 and not any(r["corrupted"] for r in kr)
    assert sum(1 for r in kr if r["exact"]) == 260 and f["held_k"] == 260 and abs(f["held_k"] / 400 - 0.65) < 1e-12
    assert all(r["exact"] for r in ar) and not any(r["corrupted"] for r in ar) and max(r["distance"] for r in ar) == 0 and all(r["reply_exact"] for r in ar) and all(r["parsed"] for r in ar)
    assert sum(r["prompt_tokens"] for r in kr) == 39_923_458 == f["sk"]["prompt"] and sum(r["prompt_tokens"] for r in ar) == 470_669 == f["sa"]["prompt"] and kr[-1]["prompt_tokens"] == f["max_prompt"] == 198_405
    live = [r for r in kr if not r["from_disk"]]; assert abs(f["sk"]["seconds_per_step"] - sum(r["seconds"] for r in live) / len(live)) < 1e-9
    deep = [r for r in kr if r["prompt_tokens"] >= 150_000]; assert len(deep) == f["sd"]["steps"] == 99 and abs(f["sd"]["seconds_per_step"] - sum(r["seconds"] for r in deep) / len(deep)) < 1e-9
    assert abs(f["sd"]["first_p50"] - nr([r["first_token"] for r in deep], 0.5)) < 1e-9
    assert abs(f["sa"]["first_p50"] - nr([r["first_token"] for r in ar], 0.5)) < 1e-9
    assert abs(f["sa"]["think_per_step"] - sum(r["think"] for r in ar) / 400) < 1e-9 and sum(r["think"] for r in kr) == 1330
    qa = [r for r in ar if r["kind"] == "query"]; qk = [r for r in kr if r["kind"] == "query"]; assert len(qa) == len(qk) == 39 and all(r["reply_exact"] for r in qa) and sum(1 for r in qk if r["reply_exact"]) == 38
    assert sum(b["steps"] for b in f["bands"]) == 400 and abs(sum(b["steps"] * b["reply_exact"] for b in f["bands"]) - 398) < 1e-6
    assert fz_cross(f)["first"] == 121 and abs(f["sa_same"]["exact_per_hour"] / f["sk"]["exact_per_hour"] - 1.3274) < 0.001
    # ARC (test 24)
    r = load("R"); a = ar_fig(r); kr, ar = a["kr"], a["ar"]
    assert len(kr) == len(ar) == 240 and {x["game"] for x in kr} == set(R.GAMES) and all(sum(1 for x in kr if x["g"] == g) == 60 for g in range(4)) and all(sum(1 for x in ar if x["g"] == g) == 60 for g in range(4))
    bad = [x for x in kr if x["parsed"] is not True]; assert len(bad) == 13 and len(a["bad_empty"]) == 9 and len(a["bad_cut"]) == 2 and len(a["bad_other"]) == 2 and all(x["parsed"] for x in ar) and all(x["reply_ok"] for x in ar)
    assert min(x["prompt_tokens"] for x in bad) == 21_041 and sum(1 for x in kr if x["reply_ok"]) == 227 and sum(1 for x in kr if x["reply_exact"]) == 210 and sum(1 for x in ar if x["reply_exact"]) == 228
    lv = lambda rows: sum(max(x["levels"] for x in rows if x["g"] == g) for g in range(4)); assert lv(kr) == 2 and lv(ar) == 1 and a["sk"]["levels"] == 2 and a["sa"]["levels"] == 1
    assert sum(1 for x in kr if x["repeat_noop"]) == 1 and sum(1 for x in ar if x["repeat_noop"]) == 0 and max(x["prompt_tokens"] for x in kr) == 253_440 and max(x["prompt_tokens"] for x in ar) == 4_665
    assert a["sent_k"] == sum(x["prompt_tokens"] + x["tokens"] + x["think"] for x in kr if not x["from_disk"]) and a["sent_k"] == 25_689_461 and a["sent_a"] == 1_951_121
    # explorer (test 25)
    b = load("B"); e = ab_fig(b); br = e["br"]
    assert len(br) == 240 and not any(x.get("forced") for x in br) and sum(1 for x in br if x["reply_exact"]) == 229 and e["sb"]["levels"] == 1 and e["sb"]["unusable"] == 0 and e["sb"]["repeat_noop"] == 0
    assert [g["levels"] for g in e["sb"]["games"]] == [0, 1, 0, 0] and [g["levels"] for g in e["sa"]["games"]] == [0, 0, 0, 1] and [g["levels"] for g in e["sk"]["games"]] == [0, 0, 1, 1]
    assert e["sb"]["sent"] == 1_790_904 and e["sa"]["sent"] == 1_951_121 and e["sk"]["sent"] == 25_689_461 and e["sb"]["computed"] == 1_784_208 and e["sb"]["discovered"] == 221 and e["sa"]["discovered"] == 227


def test_the_text_above_each_result_is_the_locked_text_and_each_page_is_the_render_of_its_saved_file():
    for k, render in (("F", render_f), ("R", render_r), ("B", render_b)):
        text = open(PAGE[k]).read(); assert text.count(MARK[k]) == 1, k; head = text[:text.index(MARK[k])]
        assert _h(head) == SHA_HEAD[k], k
        sep = "\n" if k == "B" else ""                      # the explorer page's Result mark is "\n## Result", the other two "\n\n---\n\n# Result"
        assert text == head.rstrip("\n") + sep + render(load(k)).rstrip("\n") + "\n", k


def test_the_claims_rules_and_the_wording_rules():
    for k, render, must in (("F", render_f, ("AXI HOLDS ON FUZZY TEXT", "not joules", "stand-in", "blind application", "One run of each arm")),
                            ("R", render_r, ("no claim is made until the cause is found", "not joules", "stand-in", "It was not faster and it did not win more levels", "One run of each arm")),
                            ("B", render_b, ("never fired", "not joules", "stand-in", "MIXED", "run once"))):
        page = render(load(k)); low = page.lower()
        for bad in ("$", "proven", "100 percent", "gpus saved", "dollar", "google", "openai", "ultimate", "industry", "market", "profit", "carbon", "kwh", "watt "): assert bad not in low, (k, bad)
        for m in must: assert m.lower() in low, (k, m)


def test_the_company_side_section_of_the_front_page_is_the_render_of_the_saved_files():
    df, dr, db = load("F"), load("R"), load("B"); rd = open(README).read()
    assert rd.count(R_START) == 1 and rd.count(R_END) == 1 and rd[rd.index(R_START):rd.index(R_END) + len(R_END)] == readme_section(df, dr, db)
    c = company(df, dr, db)
    assert round(c["sk"]["prompt_per_step"] / c["sa"]["prompt_per_step"]) == 85 and round(c["sk"]["sent"] / c["sa"]["sent"]) == 76 and abs(c["a"]["sent_k"] / c["a"]["sent_a"] - 13.17) < 0.01 and round(c["sd"]["seconds_per_step"] / c["sa"]["seconds_per_step"], 2) == 2.07
    sec_ = rd[rd.index(R_START):rd.index(R_END)]; low = sec_.lower()
    for bad in ("$", "proven", "gpus saved", "100 percent", "google", "openai", "anthropic", "kwh", "carbon", "ultimate", "profit", "market"): assert bad not in low, bad
    assert "Nothing here is a statement about any company's service" in sec_ and "AXI was slower" in sec_ and "did not complete more" in sec_ and "Never fired" in sec_
    assert "docs/AXI_TECHNICAL_REFERENCE.md" in sec_


def test_the_technical_reference_carries_the_measured_tables_of_the_saved_files_and_follows_the_wording_rules():
    df, dr, db = load("F"), load("R"), load("B"); t = open(REF).read()
    assert t.count(M_START) == 1 and t.count(M_END) == 1 and t[t.index(M_START):t.index(M_END) + len(M_END)] == measured_block(df, dr, db)
    low = t.lower()
    for bad in ("$", "proven", "gpus saved", "100 percent", "ultimate", "openai", "anthropic", "kwh", "carbon"): assert bad not in low, bad
    for need in ("Rajnish Choubey", "not joules", "What is not shown", "tests/test_arc_fuzzy_results.py"): assert need in t, need


def test_the_numbers_typed_into_the_reference_follow_from_the_saved_files():
    df, dr, db = load("F"), load("R"), load("B"); t = open(REF).read(); c = company(df, dr, db); f, a, b = c["f"], c["a"], c["b"]; sk, sa, sd = f["sk"], f["sa"], f["sd"]; rk, ra = c["rk"], c["ra"]; kr, ar = f["kr"], f["ar"]
    assert round(0.9885 ** 100 * 100) == 31 and 0.9885 ** 400 < 0.01 and round(3954 / 4000, 4) == 0.9885
    c22 = L22.company_figs(L22.load()); f22 = c22["f"]
    must = [
        "about **31 percent**", "3,954 of 4,000", "below **1 percent**", f"{len(kr) - len(f['slips'])} of {len(kr)} replies were exact",
        f"**held a wrong list for {len(f['held_wrong'])} turns in a row** (turns {f['run_first']} to {f['run_last']})", f"The list was exact on {round(f['held_k'] / len(kr) * 100)} percent of the turns although {sk['reply_exact'] * 100:.1f} percent of the replies were exact",
        f"({f['s50']['exact_per_hour']:,.0f} exact steps an hour in the first 50 turns against {GL.stats(ar[:50])['exact_per_hour']:,.0f})", f"its first token took {sd['first_p50']:.1f} seconds against {sa['first_p50']:.1f} with AXI",
        f"AXI made {f['sa_same']['exact_per_hour'] / sk['exact_per_hour']:.2f} times the exact steps an hour", f"{times(a['sent_k'], a['sent_a'], 1)} times fewer tokens", f"the kept chat: {len(a['bad'])} of {len(a['kr'])}, nine of them empty replies",
        f"({ra['all']['seconds_per_step']:.1f} seconds a move against {rk['all']['seconds_per_step']:.1f})", f"completed **fewer levels** ({ra['levels']} against {rk['levels']})", f"({ra['all']['think_per_step']:,.0f} tokens a move against {rk['all']['think_per_step']:,.0f})",
        f"({pc(ra['effective'])[:-2]} against {pc(rk['effective'])[:-2]} percent)", f"{pc(b['sb']['effective'])[:-2]} percent effective moves against {pc(b['sa']['effective'])[:-2]}", f"{round((1 - b['sb']['think_per_call'] / b['sa']['think_per_call']) * 100)} percent fewer thinking tokens a call",
        f"{b['sb']['per_hour'] / b['sa']['per_hour']:.1f} times the effective moves an hour", f"{b['sa']['discovered'] - b['sb']['discovered']} fewer screens never seen before",
        f"{round(c22['prompt_x'])} times fewer a step and {round(c22['sent_x'])} times fewer in all", f"{round(sk['prompt_per_step'] / sa['prompt_per_step'])} times fewer prompt tokens a step and {round(sk['sent'] / sa['sent'])} times fewer tokens sent in all",
        f"the ratios are {sk['computed_per_exact'] / sa['computed_per_exact']:.1f} (Test 23, whole job) and {c22['comp_x']:.1f} (Test 22", f"({sd['first_p95']:.1f} seconds against {sa['first_p95']:.1f} in Test 23)",
        f"it used {sa['think_per_step']:.0f} tokens a step where the kept chat used {sk['think_per_step']:.1f}", f"{pc(sk['cached_share'], 0)[:-2]} percent of the kept chat's prompt tokens from its cache",
        f"slipped on {len(f22['slips'])} turns", f"corrupted from turn {f22['corrupt'][0]} on", f"flat prompt of about {round(round(f22['sa']['prompt_per_step'], -2)):,} tokens", "(20 and 7 moves asked again)",
    ]
    for m in must: assert m in t, m
    assert ra["reasked"] == 20 and b["sb"]["reasked"] == 7
    assert [len(x) for x in (a["bad_empty"], a["bad_cut"], a["bad_other"])] == [9, 2, 2]
    assert round(f22["sd"]["first_p50"], 1) == 9.1 and round(f22["bands"][0]["first_p50"], 1) == 2.2 and round(f22["sa"]["first_p50"], 1) == 2.3


def test_the_front_pages_say_what_was_run_and_read_and_the_counter_moved():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert "Next free: row 193, item 146, test 171, question 122" in cm and cm.count("Next free:") == 1
    for n, key, extra in ((116, "116. **", "15 of the 24 lines read yes"), (117, "117. **", "11 of the 19 lines read yes"), (118, "\n118. **", "10 of the 17 lines read yes")):
        i = cm.index(key) + (1 if key.startswith("\n") else 0); item = cm[i:cm.index("\n", i)]
        assert "Status: run once" in item and extra in item and "test 145" in item and "Status: built and checked here; not run" not in item, n
    for name, tag in (("GEMINI_FUZZY_TEST.md", "15 of 24"), ("GEMINI_ARC_TEST.md", "11 of 19"), ("GEMINI_ARC_B_TEST.md", "10 of 17")):
        row = next(l for l in vf.split("\n") if name in l); assert "Run once" in row and tag in row and "test_arc_fuzzy_results.py" in row, name
    j = cm.index("\n119. **") + 1; item = cm[j:cm.index("\n", j)]; assert "AXI_TECHNICAL_REFERENCE.md" in item and "test 145" in item


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

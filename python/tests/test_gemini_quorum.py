"""Implements concept-map item 123 (test 26: Rajnish Choubey's witness quorum on Gemini 3.8 Flash): the INSTRUMENT of docs/GEMINI_QUORUM_TEST.md,
written before any call, test 149. It checks the instrument, not a result:
  1. the witness prompt: the same as arm S's prompt plus one line with the witness's number, so each witness is its own call (its own key in the
     reply file); the canonical change is the list a reply would leave, or the answer;
  2. the rule on a stand-in model that answers with the script's ideal reply, or a legal-but-wrong one, or a garbled one, for chosen witnesses:
     two in agreement commit at once; a lone wrong witness is outvoted; two wrong in the same way commit the wrong change (the stated limit);
     a step where no change gets two votes in five witnesses changes nothing; on the same plan arm S commits the wrong change and arm Q does not;
  3. the saved rows: resume from a file to the same rows, and the replay refuses a doctored witness text, a dropped witness and a doctored vote;
  4. the 16 reading lines on full 400-step stand-in runs, and their flips at the boundaries; the headline;
  5. the page above its Result is what was written before the run (a hash), the page and the module agree, and the constants that quote
     Test 23 are the saved file's; the notebook holds no key, never turns billing on and sets no budget;
  6. the front pages (concept-map item 123, VERIFIED row) and the counter.
What it does NOT show: anything about a real model's replies, or the result of the run."""
import copy, hashlib, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.witnesses import tally_keys
from axi.experiments import gemini_fuzzy as F
from axi.experiments import gemini_long as GL
from axi.experiments import gemini_quorum as Q
from axi.experiments.api_llm import DiskCachedLLM

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "GEMINI_QUORUM_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_quorum.ipynb")
PAGE_SHA256 = "11c77397a82c8fe0c33777ad27052dfbe83e0f648edc84c19dc00c68863e9182"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"
NEXT = "Next free: row 184, item 138, test 163, question 122"
SEED = Q.SEED


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


def alt(field, v, k):
    pool = {"owner": F.OWNERS, "day": F.DAYS, "status": F.STATUSES, "amount": F.AMOUNTS}[field]; others = [x for x in pool if x != v]; return others[k % len(others)]


class FakeW:
    """A stand-in for ChatLLM. It finds the step from the chatter in the prompt and the witness from its line (arm S: witness 1), and answers
    with the script's ideal reply, or by `plan[(t, w)]`: 'wrong' / 'wrong2' (two different legal-but-wrong replies), 'garble' (not JSON)."""
    def __init__(self, seed, plan=None, think=7):
        self.seed, self.plan, self.calls = seed, dict(plan or {}), []
        self.last_prompt_tokens = 0; self.last_think_tokens = think; self.from_disk = False; self.last_cache_tokens = 0; self.last_first_token = 0.2; self.last_finish = "STOP"
        self.st, self.ob, s = {}, {}, F.make_stream(seed)
        for t in range(1, F.STEPS + 1):
            o = s.next()
            if o is None: break
            self.st[t], self.ob[t] = o
        self.chat = {F.chatter(seed, t): t for t in self.st}

    def wrong(self, t, k):
        st, ob = self.st[t], self.ob[t]
        if st["kind"] == "query":
            ans = st["answer"]; cands = [x for x in ("Monday", "Tuesday", "Wednesday") if F.norm_answer(x) != F.norm_answer(ans)]; return json.dumps({"answer": cands[k % len(cands)]})
        ops = copy.deepcopy(json.loads(st["ideal"])["ops"]); o = ops[0]
        if o["op"] == "set": o["value"] = alt(o["field"], o["value"], k)
        elif o["op"] == "add": f = sorted(o["fields"])[0]; o["fields"][f] = alt(f, o["fields"][f], k)
        else:
            others = sorted(n for n in ob if n != o["name"]); ops[0] = {"op": "delete", "name": others[k % len(others)]}
        return json.dumps({"ops": ops})

    def generate(self, messages, constrained=True, max_new_tokens=400):
        content = messages[1]["content"]; part = re.split(r"\n\n(?:Message|Question): ", content)[0]; t = self.chat[part.split("\n\n")[-1]]
        m = re.search(r"\(You are witness (\d+)\.", content); w = int(m.group(1)) if m else 1
        reask = messages[-1]["content"].startswith("That reply could not")
        kind = "ideal" if reask else self.plan.get((t, w), "ideal"); self.calls.append((t, w, kind, reask))
        text = {"ideal": lambda: self.st[t]["ideal"], "wrong": lambda: self.wrong(t, 0), "wrong2": lambda: self.wrong(t, 1), "garble": lambda: "Sure! " + self.st[t]["ideal"]}[kind]()
        self.last_prompt_tokens = sum(len(x["content"]) for x in messages) // 4; self.last_first_token = 0.2
        return text, 12, 0.6


def first_steps(kind, k, lo=1, hi=100):
    f = FakeW(SEED); return [t for t in range(lo, hi + 1) if (f.st[t]["kind"] == "query") == (kind == "query") and (kind == "query" or f.st[t]["kind"] == kind)][:k]


# ---------------------------------------------------------------- 1 the witness prompt and the canonical change
def test_the_witness_prompt_is_arm_s_prompt_plus_its_number_and_each_witness_is_its_own_call():
    s = F.make_stream(SEED); st, ob = s.next(); live = {}
    base = [F._system("A"), {"role": "user", "content": F.user_text(SEED, 1, st, live)}]
    for w in (1, 2, 5):
        m = Q.witness_messages(SEED, 1, st, live, w)
        assert m[0] == base[0] and m[1]["content"] == base[1]["content"] + f"\n\n(You are witness {w}. Answer on your own.)"
    keys = {DiskCachedLLM.key(Q.witness_messages(SEED, 1, st, live, w)) for w in range(1, 6)} | {DiskCachedLLM.key(base)}
    assert len(keys) == 6 and "shown before every message" in base[0]["content"]


def test_the_canonical_change_is_the_list_a_reply_leaves_or_the_answer():
    live = {"Amber Gallery": {"owner": "Priya"}}
    st = dict(kind="set"); a = '{"ops": [{"op": "set", "name": "Amber Gallery", "field": "day", "value": "monday"}]}'
    b = '{"ops":[{"name":"Amber Gallery","op":"set","value":"Monday","field":"day"}]}'
    assert Q.key_of(live, st, a) == Q.key_of(live, st, b) is not None and Q.key_of(live, st, '{"ops": [{"op": "set", "name": "Nowhere", "field": "day", "value": "Monday"}]}') is None
    assert Q.key_of(live, dict(kind="query"), '{"answer": "Monday."}') == Q.key_of(live, dict(kind="query"), '{"answer": "monday"}') == "answer:monday"
    assert Q.key_of(live, st, "Sure!") is None
    assert tally_keys(["x", "y", "x"], need=2)[:3] == ("x", "agreed", 2) and tally_keys(["x", "y"], need=2)[1] == "no quorum" and tally_keys([None, None])[1] == "none admissible"
    assert tally_keys(["x", "y", "x", "y"], need=2)[1] == "tie"


# ---------------------------------------------------------------- 2 the rule on a stand-in
def test_the_rule_on_a_stand_in_two_agree_a_lone_wrong_witness_is_outvoted_two_wrong_commit_and_no_lead_changes_nothing():
    ups = first_steps("set", 4); q = first_steps("query", 1)
    a, b, c, d = ups; plan = {(a, 1): "wrong", (b, 1): "wrong", (b, 2): "wrong", (c, 1): "wrong", (c, 2): "wrong2", (c, 3): "ideal", (c, 4): "ideal"}
    plan.update({(d, w): "garble" for w in range(1, 6)}); plan.update({(q[0], 1): "wrong"})
    fake = FakeW(SEED, plan); res = Q.run_quorum(fake, SEED, 100, tmp()); rows = {r["t"]: r for r in res["rows"]}
    assert rows[a]["heard"] == 3 and rows[a]["committed"] and rows[a]["reply_exact"] and rows[a]["first_two_agree"] is False and rows[a]["votes"] == 2
    assert rows[b]["heard"] == 2 and rows[b]["committed"] and not rows[b]["reply_exact"] and rows[b]["first_two_agree"] is True     # the stated limit
    assert rows[c]["heard"] == 4 and rows[c]["committed"] and rows[c]["reply_exact"]
    assert rows[d]["heard"] == 5 and not rows[d]["committed"] and rows[d]["decision"] == "none admissible" and rows[d]["admissible"] == [False] * 5
    assert rows[q[0]]["heard"] == 3 and rows[q[0]]["reply_exact"]
    plain = [r for t, r in rows.items() if t not in (a, b, c, d, q[0])]
    assert all(r["heard"] == 2 and r["committed"] and r["decision"] == "agreed" for r in plain)
    assert Q.wrong_commits(res["rows"]) == 1 and Q.disagreement(res["rows"]) == (4, 3)          # a, c, d (both refused) and the question; d committed nothing
    assert len(fake.calls) == sum(r["heard"] for r in res["rows"])


def test_on_the_same_plan_arm_s_commits_the_wrong_change_and_arm_q_does_not():
    ups = first_steps("set", 3); plan = {(t, 1): "wrong" for t in ups}
    s = F.run_axi(FakeW(SEED, plan), SEED, 100, tmp()); q = Q.run_quorum(FakeW(SEED, plan), SEED, 100, tmp())
    assert Q.wrong_commits(s["rows"]) == 3 and Q.wrong_commits(q["rows"]) == 0
    assert sum(r["exact"] for r in q["rows"]) == 100 and sum(r["exact"] for r in s["rows"]) < 100


# ---------------------------------------------------------------- 3 the saved rows
def test_a_stopped_run_resumes_to_the_same_rows_and_the_replay_refuses_a_doctored_file():
    ups = first_steps("set", 2); plan = {(ups[0], 1): "wrong", (ups[1], 1): "garble"}; p = tmp()
    full = Q.run_quorum(FakeW(SEED, plan), SEED, 60, tmp())["rows"]
    cut = Q.run_quorum(FakeW(SEED, plan), SEED, 60, p, max_tokens=20_000); assert cut["stopped"] == "token_cap" and 5 < len(cut["rows"]) < 60
    again = Q.run_quorum(FakeW(SEED, plan), SEED, 60, p)["rows"]
    strip = lambda rs: [{k: v for k, v in r.items()} for r in rs]
    assert strip(again) == strip(full) and len(again) == 60
    Q.replay_q(full, SEED)
    i = ups[0] - 1
    bad = copy.deepcopy(full); bad[i]["witness_texts"][2] = bad[i]["witness_texts"][0]; bad[i]["reasks"][1]["text"] = bad[i]["witness_texts"][0]
    for doctored in (bad,):
        try: Q.replay_q(doctored, SEED); assert False, "replay must refuse"
        except F.ReplayMismatch: pass
    drop = copy.deepcopy(full); drop[i]["witness_texts"] = drop[i]["witness_texts"][:2]; drop[i]["reasks"] = drop[i]["reasks"][:1]; drop[i]["heard"] = 2
    try: Q.replay_q(drop, SEED); assert False
    except F.ReplayMismatch as e: assert "rule stops at" in str(e)
    vote = copy.deepcopy(full); vote[5]["votes"] = 3
    try: Q.replay_q(vote, SEED); assert False
    except F.ReplayMismatch: pass
    sc = Q.run_quorum(FakeW(SEED), SEED, 30, tmp(), scripted=20)["rows"]
    assert all(r["decision"] == "scripted" and r["from_disk"] for r in sc[:20]) and all(r["heard"] == 2 for r in sc[20:]); Q.replay_q(sc, SEED)


# ---------------------------------------------------------------- 4 the reading lines
def L(checks): return {n: c for n, _, _, c in checks}


def runs(plan, think=7):
    s = F.run_axi(FakeW(SEED, plan, think), SEED, Q.STEPS, tmp()); q = Q.run_quorum(FakeW(SEED, plan, think), SEED, Q.STEPS, tmp()); return s, q


def test_the_reading_lines_on_full_stand_in_runs_and_their_flips():
    perfect_s, perfect_q = runs({})
    v = L(Q.verdict(perfect_s, perfect_q, {"S": True, "Q": True}))
    assert v[1] and v[2] and v[3] and v[4] and v[5] is False and v[6] and v[7] is False and v[8] and v[9] and v[10] and v[11] is None and v[15] and v[16]
    assert Q.headline(Q.verdict(perfect_s, perfect_q, {"S": True, "Q": True})) == "NOTHING TO FIX"
    assert L(Q.verdict(perfect_s, perfect_q, {"S": True, "Q": False}))[2] is False and Q.headline(Q.verdict(perfect_s, perfect_q, {"S": True, "Q": False})) == "NOT READ"
    ups = [t for t in range(1, 401) if FakeW(SEED).st[t]["kind"] == "set"][:12]
    s, q = runs({(t, 1): "wrong" for t in ups})
    ch = Q.verdict(s, q, {"S": True, "Q": True}); v = L(ch)
    assert v[5] and v[6] and v[7] and v[8] and v[9] and v[10] and v[11] and v[12] and v[15] and v[16] and Q.headline(ch) == "THE QUORUM PAYS"
    assert Q.wrong_commits(s["rows"]) == 12 and Q.wrong_commits(q["rows"]) == 0 and Q.disagreement(q["rows"]) == (12, 12)
    # line 15 flips between 95 % and below: 12 of 400 disagree -> 97 %; 25 -> 93.75 %
    ups25 = [t for t in range(1, 401) if FakeW(SEED).st[t]["kind"] == "set"][:25]
    q25 = Q.run_quorum(FakeW(SEED, {(t, 1): "wrong" for t in ups25}), SEED, Q.STEPS, tmp())
    assert L(Q.verdict(s, q25, {"S": True, "Q": True}))[15] is False and Q.agree_share(q25["rows"]) < 0.95 <= Q.agree_share(q["rows"])
    # line 3 flips at a quarter of 111.5 thinking tokens a call
    thick_s = F.run_axi(FakeW(SEED, {}, think=28), SEED, 50, tmp()); assert L(Q.verdict(thick_s, None))[3] is False
    thin_s = F.run_axi(FakeW(SEED, {}, think=27), SEED, 50, tmp()); assert L(Q.verdict(thin_s, None))[3] is True
    # lines 13 and 14 flip at the reference figures of Test 23's arm A
    sq = GL.stats(q["rows"]); old = (Q.REF_COMPUTED_PER_EXACT, Q.REF_SECONDS_PER_STEP)
    try:
        Q.REF_COMPUTED_PER_EXACT, Q.REF_SECONDS_PER_STEP = sq["computed_per_exact"] + 0.01, sq["seconds_per_step"]
        v = L(Q.verdict(s, q, {"S": True, "Q": True})); assert v[13] and v[14]
        Q.REF_COMPUTED_PER_EXACT, Q.REF_SECONDS_PER_STEP = sq["computed_per_exact"], sq["seconds_per_step"] - 0.01
        v = L(Q.verdict(s, q, {"S": True, "Q": True})); assert v[13] is False and v[14] is False
    finally: Q.REF_COMPUTED_PER_EXACT, Q.REF_SECONDS_PER_STEP = old
    # line 4 reads NO when more than 1 percent of the replies do not parse
    g = [t for t in range(1, 401)][:12]; gs, gq = runs({(t, w): "garble" for t in g for w in (1, 2)})
    assert L(Q.verdict(gs, gq, {"S": True, "Q": True}))[4] is False and Q.headline(Q.verdict(gs, gq, {"S": True, "Q": True})) == "NOT READ"
    # line 7 reads NO and the headline NO GAIN when the witnesses are wrong together as often as one witness is wrong alone
    both = {(t, w): "wrong" for t in ups for w in (1, 2)}; bs, bq = runs(both)
    vb = L(Q.verdict(bs, bq, {"S": True, "Q": True})); assert vb[5] and vb[7] is False and Q.headline(Q.verdict(bs, bq, {"S": True, "Q": True})) == "NO GAIN"


# ---------------------------------------------------------------- 5 the page, the module, the notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert _h(_head()) == PAGE_SHA256


def test_the_page_and_the_module_agree_and_the_reference_figures_are_the_saved_ones():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(Q.LINES) == 16
    for (n, tag, text, pr), (m, rt), (gn, gt, gtext) in zip(ps, rs, Q.LINES): assert int(n) == int(m) == gn and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (Q.MODEL, str(Q.SEED), "400 steps", "minimal", "low", f"{Q.S_TOKEN_CAP:,}", f"{Q.Q_TOKEN_CAP:,}", "111.5", "1,322", "2.76", "THE QUORUM PAYS", "NOTHING TO FIX", "NO GAIN",
                   "MIXED", "NOT READ", "Rajnish Choubey", "The 99 Eyewitnesses of the Ramayana", "legal but wrong", "no joules", "not independent", "seed 98"):
        assert needle in head, needle
    assert "ultimate" not in head.lower() and "$" not in head and "proven" not in head.lower() and "100 percent" not in head
    assert Q.MODEL == "gemini-3.8-flash" and Q.STEPS == 400 and Q.NEED == 2 and Q.MAX_WITNESSES == 5 and Q.THINKING == "minimal" and Q.THINKING_FALLBACK == "low" and Q.SEED == 20261122
    d = json.load(open(os.path.join(DOCS, "results", Q.REF_FILE))); a = d["A"]["summary"]["all"]
    assert abs(Q.REF_THINK_PER_CALL - a["think"] / 400) < 1e-9 and d["A"]["summary"]["reasked"] == 0 and abs(Q.REF_COMPUTED_PER_EXACT - a["computed_per_exact"]) < 1e-9
    assert abs(Q.REF_SECONDS_PER_STEP - a["seconds_per_step"]) < 1e-9 and sum(r["exact"] for r in d["A"]["rows"]) == 400


def test_the_notebook_holds_no_key_never_turns_billing_on_and_sets_no_budget():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 19 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    assert "BUDGET_USD = None" in code and "PAID = None" in code and "PRICE_IN = None" in code and "PRICE_OUT = None" in code and "THINKING = Q.THINKING" in code
    assert "enable billing" not in code.lower()
    for name in ("Q.run_quorum", "F.run_axi", "Q.replay_q", "F.replay", "Q.verdict", "Q.pilot_report", "Q.print_verdict", "F.GL.build_chat_llm", "Q.S_TOKEN_CAP", "Q.Q_TOKEN_CAP", "Q.headline"): assert name in code, name
    assert code.count("need_budget()") >= 3 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code") and "gemini_quorum.py" in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 6 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 149 checks the witness quorum test on Gemini 3.8 Flash, item 123" in cm
    j = cm.index("\n123. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "GEMINI_QUORUM_TEST.md" in item and "gemini_quorum.py" in item and "test 149" in item and "Status: built and checked here; not run" in item and PAGE_SHA256 in item
    row = next(l for l in vf.split("\n") if "GEMINI_QUORUM_TEST.md" in l); assert "Built and checked here; NOT RUN" in row and "test_gemini_quorum.py" in row and "11c77397" in row


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

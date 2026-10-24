"""Implements concept-map item 116 (a long conversation in plain sentences on Gemini 3.8 Flash, test 23): the INSTRUMENT of docs/GEMINI_FUZZY_TEST.md, written before any call. It checks the instrument, not a result:
  1. the stream: deterministic, the true list moves exactly as the script says, every kind of message and question appears, the chatter never holds an item name, a question never holds its answer;
  2. the check before the list is touched (what it accepts and refuses), the blind apply of the kept chat (ghost items), the distance, the corruption test, the reply parser;
  3. the arms on a stand-in model that answers with the script's ideal reply (or a slip / a garbled first reply on chosen turns): the saved flags, the re-asks, the resume from a file, the scripted steps, the replay that refuses a doctored file;
  4. the prompts: the kept chat grows, the AXI prompt is flat, the two system texts differ by one sentence, the list is shown in the order the items were added, the depth history is the ideal replies;
  5. the reading lines flip at their boundaries; the headline needs lines 7 and 8;
  6. the page and the module agree and the page above its Result is byte for byte what was written before the run (a hash); the notebook holds no key, never turns billing on, sets no budget by itself;
  7. the front pages (concept-map item 116, VERIFIED row) and the counter.
What it does NOT show: anything about a real model's replies, or the result of the run."""
import hashlib, io, json, contextlib, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_fuzzy as F
from axi.experiments import api_llm as A

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "GEMINI_FUZZY_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_fuzzy.ipynb")
PAGE_SHA256 = "742cc9853bc869086c54dd9703928e62dce6818eba845168c23ffa5fbf3d27b2"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


class Fake:
    """A stand-in for ChatLLM: answers with the script's ideal reply for the turn it is asked about (a slip on the turns in `slip`, a garbled FIRST reply on `garble`); records the messages it was sent."""
    def __init__(self, seed, slip=(), garble=(), cache_share=0.5, offset=0):
        self.seed, self.slip, self.garble, self.share, self.seen, self.n = seed, set(slip), set(garble), cache_share, [], offset
        self.last_prompt_tokens = 0; self.last_think_tokens = 7; self.from_disk = False; self.last_cache_tokens = 0; self.last_first_token = 0.2; self.last_finish = "STOP"
        self.ideal = {}; s = F.make_stream(seed)
        for i in range(1, F.STEPS + 1):
            o = s.next()
            if o is None: break
            self.ideal[i] = o[0]["ideal"]

    def generate(self, messages, constrained=True, max_new_tokens=400):
        self.seen.append(messages); reask = messages[-1]["content"].startswith("That reply could not")
        if len(messages) == 2 or (reask and "shown before every message" in messages[0]["content"]):
            if not reask: self.n += 1
            t = self.n
        else: t = sum(1 for m in messages if m["role"] == "user" and ("\n\nMessage: " in m["content"] or "\n\nQuestion: " in m["content"]))
        text = self.ideal[t]
        if t in self.slip: text = '{"ops": [{"op": "set", "name": "Zzz Nowhere", "field": "day", "value": "Friday"}]}'
        if t in self.garble and not reask: text = "Sure! Here you go: " + text
        self.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4; self.last_cache_tokens = int(self.last_prompt_tokens * self.share); self.last_first_token = 0.2 + 1e-6 * self.last_prompt_tokens
        return text, 10, 0.5 + 2e-6 * self.last_prompt_tokens


# ---------------------------------------------------------------- 1 the stream
def test_the_stream_is_deterministic_and_has_every_kind_of_message_and_question():
    a = [F.make_stream(5).next() for _ in range(1)]; s1, s2 = F.make_stream(5), F.make_stream(5); kinds = set(); qk = set(); ages = set()
    for _ in range(F.STEPS):
        x, y = s1.next(), s2.next(); assert x[0] == y[0] and x[1] == y[1]; kinds.add(x[0]["kind"])
        if x[0]["kind"] == "query":
            qk.add(x[0]["qkind"])
            if x[0]["qkind"] == "recall": ages.add(next(i for i in range(4) if F.AGE_EDGES[i] <= x[0]["age"] < F.AGE_EDGES[i + 1]))
    assert s1.next() is None and kinds >= {"add", "set", "set2", "correct", "ref_last", "ref_owner", "ref_same", "delete", "query"} and qk == {"recall", "unknown", "ambiguous"} and ages == {0, 1, 2, 3}
    t = F.make_stream(6); assert [t.next()[0]["message"] for _ in range(5)] != [F.make_stream(5).next()[0]["message"]] * 5
    qs = F.make_stream(7); n = 0
    for i in range(F.STEPS):
        st, _ = qs.next()
        assert (st["kind"] == "query") == (i + 1 >= F.QUERY_FROM and (i + 1) % F.QUERY_EVERY == 0) or st["kind"] != "query"                    # a question only on the tenth turns
        if st["kind"] == "query": n += 1; assert st["answer"] not in st["message"] or st["answer"] == "unknown" and False
    assert 30 <= n <= 39


def test_the_true_list_moves_as_the_script_says_and_the_ideal_reply_gives_it_exactly():
    for seed in (1, 2, 3):
        s = F.make_stream(seed); held = {}
        for i in range(F.STEPS):
            st, ob = s.next(); oa = s.live; assert ob == held
            p = F.parse_reply(st["ideal"]); fl = F.reply_flags(ob, oa, st, p); assert fl == dict(reply_ok=True, reply_exact=True), (seed, i, st)
            if st["kind"] == "query": assert oa == ob
            else:
                ok, why, norm = F.check_ops(held, p["ops"]); assert ok, why; held = F.apply_ops(held, norm)
            assert held == oa and F.well_formed(oa, s.introduced) and 0 <= len(oa) <= 28
            assert list(oa) == [n for n in s.order]                                                                                         # the order of the list is the order added


def test_the_chatter_is_the_same_in_every_arm_and_holds_no_item_name_and_no_request():
    a, b = F.chatter(5, 3), F.chatter(5, 3); assert a == b and a != F.chatter(5, 4) and a != F.chatter(6, 3) and len(F.chatter(5, 3, 10).split(". ")) < len(a.split(". "))
    for t in range(1, 60):
        c = F.chatter(9, t); assert not any(n in c for n in F.NAMES) and "Message:" not in c and "Question:" not in c
    assert len(F.NAMES) == 144 and len(set(F.NAMES)) == 144
    st = dict(kind="set", message="X moves to Friday."); u = F.user_text(5, 3, st, {}); assert u.startswith("Current list (exact, kept by the framework):\n(the list is empty)") and u.endswith("Message: X moves to Friday.")
    assert F.user_text(5, 3, st, None).startswith(F.chatter(5, 3)) and F.user_text(5, 3, dict(kind="query", message="q?"), None).endswith("Question: q?")


# ---------------------------------------------------------------- 2 the check, the blind apply, the distance
def test_the_check_before_the_list_is_touched_accepts_the_allowed_and_refuses_the_rest():
    base = {"Amber Gallery": {"owner": "Priya", "day": "Friday"}}
    ok, why, n = F.check_ops(base, [{"op": "set", "name": "Amber Gallery", "field": "day", "value": "monday"}, {"op": "set", "name": "Amber Gallery", "field": "amount", "value": "450"}])
    assert ok and n[0]["value"] == "Monday" and n[1]["value"] == 450 and F.apply_ops(base, n)["Amber Gallery"] == {"owner": "Priya", "day": "Monday", "amount": 450} and base["Amber Gallery"]["day"] == "Friday"
    bad = [[{"op": "set", "name": "Birch Studio", "field": "day", "value": "Monday"}], [{"op": "delete", "name": "Birch Studio"}], [{"op": "add", "name": "Amber Gallery", "fields": {}}], [{"op": "add", "name": "Nowhere Place", "fields": {}}],
           [{"op": "set", "name": "Amber Gallery", "field": "colour", "value": "red"}], [{"op": "set", "name": "Amber Gallery", "field": "day", "value": "Funday"}], [{"op": "set", "name": "Amber Gallery", "field": "amount", "value": 95}],
           [{"op": "set", "name": "Amber Gallery", "field": "amount", "value": 1000}], [{"op": "set", "name": "Amber Gallery", "field": "amount", "value": True}], [{"op": "rename", "name": "Amber Gallery"}],
           [{"op": "add", "name": "Birch Studio", "fields": {"day": "Someday"}}], [{"op": "add", "name": "Birch Studio", "fields": "day=Monday"}], [{"op": "delete", "name": "Amber Gallery"}, {"op": "delete", "name": "Amber Gallery"}]]
    for ops in bad: assert F.check_ops(base, ops)[0] is False, ops
    assert F.check_ops(base, [{"op": "add", "name": "Birch Studio", "fields": {"owner": "aiko", "amount": 450.0}}])[2][0]["fields"] == {"owner": "Aiko", "amount": 450} and F.check_ops(base, [])[0] is True
    assert F.check_ops(base, [{"op": "delete", "name": "Amber Gallery"}, {"op": "add", "name": "Amber Gallery", "fields": {}}])[0] is True                       # a delete then an add of the same item is allowed


def test_the_kept_chat_applies_blind_so_a_wrong_reply_leaves_a_ghost_and_the_list_is_corrupted():
    base = {"Amber Gallery": {"owner": "Priya"}}; intro = {"Amber Gallery"}
    g = F.blind_apply(base, [{"op": "set", "name": "Zzz Nowhere", "field": "day", "value": "Friday"}]); assert "Zzz Nowhere" in g and F.well_formed(g, intro) is False and base == {"Amber Gallery": {"owner": "Priya"}}
    g = F.blind_apply(base, [{"op": "set", "name": "Amber Gallery", "field": "day", "value": "Funday"}]); assert g["Amber Gallery"]["day"] == "Funday" and F.well_formed(g, intro) is False
    g = F.blind_apply(base, [{"op": "delete", "name": "Birch Studio"}, "junk", {"op": "bogus", "name": "Amber Gallery"}, {"name": 5}]); assert g == base
    g = F.blind_apply(base, [{"op": "set", "name": "Birch Studio", "field": "day", "value": "Monday"}]); assert F.well_formed(g, intro | {"Birch Studio"}) is True and F.well_formed(g, intro) is False
    assert F.well_formed({"Amber Gallery": {"owner": "Priya", "day": "Friday"}}, intro) and F.well_formed({"Amber Gallery": {"colour": "red"}}, intro) is False


def test_the_distance_counts_cells_and_the_parser_reads_one_json_object_only():
    a = {"X": {"day": "Monday", "owner": "Priya"}, "Y": {"day": "Friday"}}
    assert F.distance(a, a) == 0 and F.distance(a, {"X": {"day": "Monday", "owner": "Priya"}}) == 2 and F.distance(a, {"X": {"day": "Tuesday", "owner": "Priya"}, "Y": {"day": "Friday"}}) == 1
    assert F.distance(a, {"X": {"day": "Monday"}, "Y": {"day": "Friday"}}) == 1 and F.distance({}, a) == 5 and F.distance(a, {"X": {"day": "Monday", "owner": "Priya"}, "Y": {}}) == 1
    P = F.parse_reply
    assert P('{"answer": "Friday"}') == dict(answer="Friday") and P('```json\n{"answer": 450.0}\n```') == dict(answer="450") and P('{"ops": []}') == dict(ops=[]) and P(' {"ops": [{"op": "delete", "name": "X"}]} ') == dict(ops=[{"op": "delete", "name": "X"}])
    for bad in ("", None, "Friday", "Sure! {\"answer\": \"x\"}", '{"answer": true}', '{"ops": "x"}', '{"ops": [1]}', '[{"op": "delete"}]', '{"foo": 1}', '{"answer": {"a": 1}}', '{"answer": '): assert P(bad) is None, bad
    assert F.norm_answer(" Friday. ") == "friday" and F.norm_answer(450) == "450"


# ---------------------------------------------------------------- 3 the arms
def test_the_two_arms_on_a_perfect_stand_in_are_exact_and_a_slip_corrupts_the_kept_chat_but_not_the_axi_list():
    seed = 4; k = F.run_kept(Fake(seed), seed, 60, tmp()); a = F.run_axi(Fake(seed), seed, 60, tmp())
    for res in (k, a): assert all(r["reply_exact"] and r["exact"] and r["parsed"] and not r["corrupted"] and r["distance"] == 0 for r in res["rows"]) and len(res["rows"]) == 60 and res["stopped"] is None
    k = F.run_kept(Fake(seed, slip=(33,)), seed, 60, tmp()); a = F.run_axi(Fake(seed, slip=(33,)), seed, 60, tmp())
    kr = {r["t"]: r for r in k["rows"]}; ar = {r["t"]: r for r in a["rows"]}
    assert kr[33]["reply_ok"] is False and kr[33]["reply_exact"] is False and kr[33]["corrupted"] is True and all(kr[t]["corrupted"] for t in range(33, 61)) and all(not kr[t]["corrupted"] for t in range(1, 33))
    assert kr[34]["reply_exact"] is True and kr[34]["exact"] is False and kr[34]["distance"] >= 1                                                     # a later reply is right but the held list stays wrong
    assert ar[33]["blocked"] is True and ar[33]["asks"] == 1 + F.A_REASKS and ar[33]["parsed"] is True and ar[33]["exact"] is False and ar[33]["corrupted"] is False and ar[33]["reply_exact"] is False
    assert max(r["distance"] for r in a["rows"]) <= 3 and all(not r["corrupted"] for r in a["rows"])
    F.replay(k["rows"], seed, "K"); F.replay(a["rows"], seed, "A")


def test_a_garbled_first_reply_is_asked_again_once_and_a_reply_that_keeps_failing_is_not_asked_a_fourth_time():
    seed = 4; fk = Fake(seed, garble=(12,)); a = F.run_axi(fk, seed, 15, tmp()); r = {x["t"]: x for x in a["rows"]}
    assert r[12]["asks"] == 2 and r[12]["first_text"].startswith("Sure!") and r[12]["exact"] and r[12]["reply_exact"] and r[12]["parsed"] and "blocked" in r[12] and r[12]["blocked"] is False
    last = fk.seen[[i for i, m in enumerate(fk.seen) if m[-1]["content"].startswith("That reply could not")][0]]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user"] and last[2]["content"].startswith("Sure!") and "one JSON object only" in last[3]["content"]
    k = F.run_kept(Fake(seed, garble=(12,)), seed, 15, tmp()); kr = {x["t"]: x for x in k["rows"]}; assert kr[12]["parsed"] is False and kr[12]["reply_exact"] is False and kr[12]["exact"] is False and kr[12]["distance"] >= 1       # no re-ask, nothing applied
    a3 = F.run_axi(Fake(seed, slip=(9,)), seed, 12, tmp()); r9 = {x["t"]: x for x in a3["rows"]}[9]; assert r9["asks"] == 3 and len(r9["reasks"]) == 2 and r9["blocked"] is True
    F.replay(a["rows"], seed, "A"); F.replay(k["rows"], seed, "K"); F.replay(a3["rows"], seed, "A")


def test_the_scripted_steps_make_no_call_and_a_stopped_run_resumes_from_its_file_to_the_same_rows():
    seed = 4; f = Fake(seed, offset=5); a = F.run_axi(f, seed, 8, tmp(), scripted=5); assert len(f.seen) == 3 and all(r["from_disk"] and r["prompt_tokens"] == 0 and r["exact"] for r in a["rows"][:5])
    assert all(not r["from_disk"] for r in a["rows"][5:]) and F.stats(a["rows"])["live_steps"] == 3 and F.stats(a["rows"])["prompt"] > 0
    p = tmp(); full = F.run_kept(Fake(seed), seed, 12, tmp())
    part = F.run_kept(Fake(seed), seed, 12, p, max_tokens=1500); assert part["stopped"] == "token_cap" and 0 < len(part["rows"]) < 12
    rest = F.run_kept(Fake(seed), seed, 12, p); assert rest["rows"] == full["rows"] and rest["stopped"] is None
    p2 = tmp(); part = F.run_axi(Fake(seed), seed, 12, p2, max_tokens=3000); n = len(part["rows"]); assert part["stopped"] == "token_cap" and n < 12
    rest = F.run_axi(Fake(seed, offset=n), seed, 12, p2); assert rest["rows"] == F.run_axi(Fake(seed), seed, 12, tmp())["rows"]
    try: F.run_kept(Fake(seed), seed + 1, 12, p); raise AssertionError("another run's file accepted")
    except ValueError: pass
    c = F.run_kept(Fake(seed), seed, 40, tmp(), context_cap=10_000); assert c["stopped"] == "context" and len(c["rows"]) < 40


def test_the_replay_refuses_a_doctored_file_and_a_doctored_re_ask():
    seed = 4; k = F.run_kept(Fake(seed), seed, 12, tmp()); a = F.run_axi(Fake(seed, garble=(5,)), seed, 12, tmp())
    for arm, res, key in (("K", k, "reply_exact"), ("K", k, "corrupted"), ("A", a, "exact"), ("A", a, "reply_exact"), ("A", a, "blocked")):
        rows = [dict(r) for r in res["rows"]]; rows[6][key] = not rows[6][key]
        try: F.replay(rows, seed, arm); raise AssertionError(f"accepted a doctored {key}")
        except F.ReplayMismatch: pass
    rows = [dict(r) for r in a["rows"]]; rows[4]["asks"] = 3
    try: F.replay(rows, seed, "A"); raise AssertionError("accepted a doctored ask count")
    except F.ReplayMismatch: pass
    rows = [dict(r) for r in a["rows"]]; rows[2]["asks"] = 2; rows[2]["first_text"] = rows[2]["text"]; rows[2]["reasks"] = [dict(rows[2])]
    try: F.replay(rows, seed, "A"); raise AssertionError("accepted a re-ask after a usable reply")
    except F.ReplayMismatch: pass
    rows = [dict(r) for r in k["rows"]]; rows[3]["kind"] = "delete" if rows[3]["kind"] != "delete" else "add"
    try: F.replay(rows, seed, "K"); raise AssertionError("accepted a doctored kind")
    except F.ReplayMismatch: pass


# ---------------------------------------------------------------- 4 the prompts
def test_the_kept_chat_grows_the_axi_prompt_is_flat_the_system_texts_differ_by_one_sentence_and_the_depth_history_is_ideal():
    seed = 5; k = F.run_kept(Fake(seed), seed, 40, tmp()); a = F.run_axi(Fake(seed), seed, 40, tmp()); kp = [r["prompt_tokens"] for r in k["rows"]]; ap = [r["prompt_tokens"] for r in a["rows"]]
    assert all(y > x for x, y in zip(kp, kp[1:])) and kp[-1] > 15 * kp[0] and max(ap[10:]) < 1.6 * min(ap[10:]) and kp[-1] > 8 * ap[-1]
    fk = Fake(seed); F.run_kept(fk, seed, 4, tmp()); m = fk.seen[3]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"] and m[1]["content"].count("Current list") == 0 and "Message: " in m[7]["content"]
    fa = Fake(seed); F.run_axi(fa, seed, 6, tmp()); assert all(len(x) == 2 for x in fa.seen) and all(x[1]["content"].startswith("Current list (exact, kept by the framework):\n") for x in fa.seen)
    assert F.SYSTEM_K.startswith(F.SYSTEM) and F.SYSTEM_A.startswith(F.SYSTEM) and F.SYSTEM_K[len(F.SYSTEM):].count("\n") == 1 and F.SYSTEM_A[len(F.SYSTEM):].count("\n") == 1 and "starts empty" in F.SYSTEM_K and "shown before every message" in F.SYSTEM_A
    fa2 = Fake(seed); a2 = F.run_axi(fa2, seed, 30, tmp()); shown = fa2.seen[-1][1]["content"].split("\n\n")[0].split("\n")[1:]; s = F.make_stream(seed); [s.next() for _ in range(29)]
    assert [l.split(" | ")[0] for l in shown] == list(s.live)                                                                                         # the list is shown in the order the items were added
    fd = Fake(seed); d = F.run_depth(fd, seed, (5,), 2, tmp()); assert len(d["rows"]) == 2 and len(fd.seen[0]) == 1 + 2 * 4 + 1 and fd.seen[0][-1]["role"] == "user"
    ideal = [x["content"] for x in fd.seen[1] if x["role"] == "assistant"]; stm = F.make_stream(seed); want = [stm.next()[0]["ideal"] for _ in range(5)]; assert ideal == want
    F.replay_depth(d["rows"], seed); rows = [dict(r) for r in d["rows"]]; rows[0]["reply_exact"] = not rows[0]["reply_exact"]
    try: F.replay_depth(rows, seed); raise AssertionError("accepted a doctored depth row")
    except F.ReplayMismatch: pass


def test_the_streamed_chat_backend_and_the_dollar_guard_are_the_ones_of_test_22_and_the_cached_price_is_used():
    assert F.GL.build_chat_llm is not None and F.MODEL == "gemini-3.8-flash" and F.GL.GeminiChatBackend.__name__ == "GeminiChatBackend"
    import types as T
    class Ch:
        def __init__(s, text, um=None, fin=None): s.text, s.usage_metadata, s.candidates = text, um, [T.SimpleNamespace(finish_reason=fin)] if fin else []
    class Cl:
        def __init__(s): s.models = T.SimpleNamespace(generate_content_stream=lambda model, contents, config: iter([Ch("{"), Ch('"answer": "x"}', T.SimpleNamespace(prompt_token_count=1_000_000, candidates_token_count=1000, thoughts_token_count=1000, cached_content_token_count=900_000), "STOP")]), list=lambda: [T.SimpleNamespace(name="models/gemini-3.8-flash")])
    class Ty:
        class Content:
            def __init__(s, role, parts): s.role, s.parts = role, parts
        class Part:
            def __init__(s, text): s.text = text
        class GenerateContentConfig:
            def __init__(s, **kw): s.kw = kw
        class ThinkingConfig:
            def __init__(s, **kw): s.kw = kw
    llm, model = F.GL.build_chat_llm("KEY", tmp(), client=Cl(), types=Ty, price_in=1.0, price_out=10.0, price_cached=0.1, budget_usd=5.0, min_interval=0, say=lambda *_: None)
    t, n, s = llm.generate([{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]); assert t == '{"answer": "x"}' and abs(llm.spent_usd() - (100_000 * 1.0 + 900_000 * 0.1 + 2000 * 10.0) / 1e6) < 1e-9


# ---------------------------------------------------------------- 5 the reading lines
def _row(t, kind="set", prompt=1000, cache=0, vis=10, think=10, sec=1.0, first=0.3, exact=True, held=True, ok=True, parsed=True, corrupted=False, distance=0, qkind=None, age=None, blocked=False):
    return dict(t=t, kind=kind, text="", tokens=vis, think=think, prompt_tokens=prompt, cache_tokens=cache, first_token=first, seconds=sec, from_disk=False, finish="STOP", parsed=parsed, reply_ok=ok, reply_exact=exact, corrupted=corrupted,
                distance=distance, exact=held, nodes=5, oracle_nodes=5, qkind=qkind, age=age, blocked=blocked)


def _k(n, **kw):
    ex = kw.pop("exact_until", 10 ** 9); prompt = kw.pop("prompt", lambda t: 1000 + 700 * t)
    return [_row(t, prompt=prompt(t), cache=int(prompt(t) * 0.5), exact=t <= ex, **kw) for t in range(1, n + 1)]


def _a(n, **kw): return [_row(t, prompt=2000, sec=1.0, first=0.3, **kw) for t in range(1, n + 1)]


def L(checks): return {i: v for i, _, _, v in checks}


def test_the_reading_lines_flip_at_their_boundaries_and_the_headline_needs_lines_7_and_8():
    assert len(F.G_LINES) == 24 and [t for t, _ in F.G_LINES[:6]] == ["guard"] * 3 + ["instrument"] * 3
    kr, ar = _k(400), _a(400); kk, aa = dict(rows=kr, stopped=None), dict(rows=ar, stopped=None); dd = dict(rows=[dict(_row(50 + j, exact=True), start=50) for j in range(5)] + [dict(_row(300 + j, exact=True), start=300) for j in range(5)])
    base = L(F.verdict(kk, aa, dd, {"K": True, "A": True})); assert base[1] and base[2] and base[3] and base[4] and base[5] and base[6] and base[7] and base[8] and base[9] and base[11] and base[17] is None
    assert L(F.verdict(None, None, None)) == {i: None for i in range(1, 25)}
    assert L(F.verdict(kk, dict(rows=_a(399), stopped=None), dd, {}))[2] is False and L(F.verdict(kk, aa, dd, {"K": False}))[1] is False and L(F.verdict(kk, aa, dd, {"A": False}))[2] is False
    assert L(F.verdict(dict(rows=_k(399), stopped=None), aa, dd, {}))[1] is False and L(F.verdict(dict(rows=_k(399), stopped="context"), aa, dd, {}))[1] is True
    assert L(F.verdict(dict(rows=_k(400, prompt=lambda t: 1000 + 300 * t), stopped=None), aa, dd, {}))[3] is False                                  # 121,000 at the end: below 150,000
    a7 = _a(400); [r.update(exact=False) for r in a7[::20]]; assert L(F.verdict(kk, dict(rows=a7, stopped=None), dd, {}))[7] is True                      # 20 of 400 = 95 percent exactly, 2 of 40 in every tenth
    a7[1].update(exact=False); assert L(F.verdict(kk, dict(rows=a7, stopped=None), dd, {}))[7] is False                                                 # 21 of 400 = 94.75 percent
    a7 = _a(400); [r.update(exact=False) for r in a7[:5]]; assert L(F.verdict(kk, dict(rows=a7, stopped=None), dd, {}))[7] is False                     # 5 of the first tenth = 87.5 percent held
    a8 = _a(400); a8[3].update(corrupted=True); assert L(F.verdict(kk, dict(rows=a8, stopped=None), dd, {}))[8] is False
    a8 = _a(400); a8[3].update(distance=3); assert L(F.verdict(kk, dict(rows=a8, stopped=None), dd, {}))[8] is True; a8[3].update(distance=4); assert L(F.verdict(kk, dict(rows=a8, stopped=None), dd, {}))[8] is False
    a9 = [_row(t, prompt=1000 if t <= 40 else 2000) for t in range(1, 401)]; assert L(F.verdict(kk, dict(rows=a9, stopped=None), dd, {}))[9] is True                      # the first tenth is the warm-up (the list is filling): the second is the reference
    a9 = [_row(t, prompt=1000 if t <= 40 else (2000 if t <= 360 else 2700)) for t in range(1, 401)]; assert L(F.verdict(kk, dict(rows=a9, stopped=None), dd, {}))[9] is False        # 2,700 against 2,000 is 1.35 times
    a9 = [_row(t, prompt=1000 if t <= 40 else (2000 if t <= 360 else 2600), sec=1.0 if t <= 360 else 1.6) for t in range(1, 401)]; assert L(F.verdict(kk, dict(rows=a9, stopped=None), dd, {}))[9] is False     # 1.6 times the seconds
    a5 = _a(400); [r.update(blocked=True) for r in a5[:2]]; assert L(F.verdict(kk, dict(rows=a5, stopped=None), dd, {}))[5] is True; a5[2].update(parsed=False); assert L(F.verdict(kk, dict(rows=a5, stopped=None), dd, {}))[5] is False
    k6 = _k(400); [r.update(parsed=False) for r in k6[:8]]; assert L(F.verdict(dict(rows=k6, stopped=None), aa, dd, {}))[6] is True; k6[8].update(parsed=False); assert L(F.verdict(dict(rows=k6, stopped=None), aa, dd, {}))[6] is False
    k4 = _k(400); [r.update(cache_tokens=0) for r in k4[10:10 + 78]]; assert L(F.verdict(dict(rows=k4, stopped=None), aa, dd, {}))[4] is True; k4[200].update(cache_tokens=0); assert L(F.verdict(dict(rows=k4, stopped=None), aa, dd, {}))[4] is False
    k11 = _k(400, exact_until=200); c = L(F.verdict(dict(rows=k11, stopped=None), aa, dd, {})); assert c[11] is True and c[12] is True and c[15] is False
    k13 = _k(400, exact_until=200); [r.update(reply_ok=False) for r in k13[-5:]]; assert L(F.verdict(dict(rows=k13, stopped=None), aa, dd, {}))[13] is True; k13[-5].update(reply_ok=True); assert L(F.verdict(dict(rows=k13, stopped=None), aa, dd, {}))[13] is False
    k14 = _k(400); k14[7].update(corrupted=True); c = L(F.verdict(dict(rows=k14, stopped=None), aa, dd, {})); assert c[14] is True and L(F.verdict(kk, aa, dd, {}))[14] is False
    k15 = _k(400); k15[48].update(reply_exact=False); assert L(F.verdict(dict(rows=k15, stopped=None), aa, dd, {}))[15] is True; k15[48].update(reply_exact=True); k15[49].update(reply_exact=False); assert L(F.verdict(dict(rows=k15, stopped=None), aa, dd, {}))[15] is False
    mixed = [(i, t, x, (False if i == 8 else v)) for i, t, x, v in F.verdict(kk, aa, dd, {})]; assert F.headline(F.verdict(kk, aa, dd, {})) == "AXI HOLDS ON FUZZY TEXT" and F.headline(mixed) == "MIXED: line 7 yes, line 8 NO"
    assert F.headline(F.verdict(kk, None, dd, {})).startswith("NOT READ") and F.claims_allowed(F.verdict(kk, aa, dd, {}))["base"] is True
    assert F.claims_allowed([(i, t, x, (False if i == 4 else v)) for i, t, x, v in F.verdict(kk, aa, dd, {})])["tokens_computed"] is False


def _qrows(arm_rows_n, young_ok, old_ok, young_n=10, old_n=10, flag_ok=None, flag_n=0, a_exact=True):
    rows = [_row(t) for t in range(1, arm_rows_n + 1)]; t = arm_rows_n
    for i in range(young_n): rows.append(_row(t + 1 + i, kind="query", qkind="recall", age=3, exact=i < young_ok))
    t = len(rows)
    for i in range(old_n): rows.append(_row(t + 1 + i, kind="query", qkind="recall", age=80, exact=i < old_ok))
    t = len(rows)
    for i in range(flag_n): rows.append(_row(t + 1 + i, kind="query", qkind="unknown", exact=i < (flag_ok or 0)))
    return rows


def test_the_age_the_questions_and_the_ask_not_guess_lines_flip_at_their_boundaries():
    aa = dict(rows=_a(400), stopped=None); dd = None
    kr = _qrows(380, 10, 7); assert L(F.verdict(dict(rows=kr, stopped=None), aa, dd, {}))[16] is True                                                  # 100 - 70 = 30 points
    kr = _qrows(380, 10, 9); assert L(F.verdict(dict(rows=kr, stopped=None), aa, dd, {}))[16] is True                                                  # exactly 10 points
    kr = _qrows(380, 9, 9, 10, 10); assert L(F.verdict(dict(rows=kr, stopped=None), aa, dd, {}))[16] is False
    kr = _qrows(380, 10, 7, young_n=2); assert L(F.verdict(dict(rows=kr, stopped=None), aa, dd, {}))[16] is None                                      # too few questions to read
    ar = _qrows(380, 10, 10); assert L(F.verdict(None, dict(rows=ar, stopped=None), dd, {}))[17] is True and L(F.verdict(None, dict(rows=ar, stopped=None), dd, {}))[10] is True
    ar = _qrows(380, 10, 9, 20, 20); [r.update() for r in ar]; assert L(F.verdict(None, dict(rows=_qrows(380, 20, 19, 20, 20), stopped=None), dd, {}))[17] is True                                              # 5 points apart
    assert L(F.verdict(None, dict(rows=_qrows(380, 20, 18, 20, 20), stopped=None), dd, {}))[17] is False                                              # 10 points apart
    assert L(F.verdict(None, dict(rows=_qrows(380, 18, 20, 20, 20), stopped=None), dd, {}))[10] is True                                               # 38 of 40 = 95 percent
    assert L(F.verdict(None, dict(rows=_qrows(380, 17, 20, 20, 20), stopped=None), dd, {}))[10] is False                                              # 37 of 40
    def fl(a_ok, k_ok, n=10): return L(F.verdict(dict(rows=_qrows(380, 10, 10, flag_ok=k_ok, flag_n=n), stopped=None), dict(rows=_qrows(380, 10, 10, flag_ok=a_ok, flag_n=n), stopped=None), dd, {}))[18]
    assert fl(8, 7) is True and fl(8, 8) is False and fl(7, 0) is False and fl(10, 9) is True and fl(8, 10) is False                                   # A at 80 percent and 10 points above K; A at 70 is below 80; A at 100 and K at 90 is 10 points
    assert L(F.verdict(dict(rows=_qrows(380, 10, 10), stopped=None), dict(rows=_qrows(380, 10, 10), stopped=None), dd, {}))[18] is None               # no such questions: not read


def test_the_latency_speed_token_and_thinking_lines_flip_at_their_boundaries():
    aa = dict(rows=_a(400), stopped=None); dd = None
    def kk(sec, first, comp_think=10, think_late=10, n=400, exact=True):
        rs = [_row(t, prompt=1000 + 700 * t, cache=0, sec=sec(t), first=first(t), think=think_late if t > n - 100 else comp_think, exact=exact) for t in range(1, n + 1)]; return dict(rows=rs, stopped=None)
    c = L(F.verdict(kk(lambda t: 1.5 if 1000 + 700 * t >= 150_000 else 1.0, lambda t: 0.45 if 1000 + 700 * t >= 150_000 else 0.3), aa, dd, {})); assert c[19] is True and c[20] is True                  # exactly 1.5 times
    c = L(F.verdict(kk(lambda t: 1.49, lambda t: 0.449), aa, dd, {})); assert c[19] is False and c[20] is False
    c = L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3), aa, dd, {})); assert c[21] is False                                                           # same speed
    c = L(F.verdict(kk(lambda t: 1.5, lambda t: 0.3), aa, dd, {})); assert c[21] is True
    c = L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3, think_late=20), aa, dd, {})); assert c[23] is True and L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3, think_late=19), aa, dd, {}))[23] is False
    c = L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3), aa, dd, {})); assert c[22] is True                                                            # 100,000 prompt tokens uncached against 2,000
    d = []
    for s, ex in ((50, 10), (300, 8)):
        for j in range(10): d.append(dict(_row(s + j, exact=j < ex), start=s))
    assert L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3), aa, dict(rows=d), {}))[24] is True
    d[-1].update(reply_exact=True); d[-2].update(reply_exact=True); assert L(F.verdict(kk(lambda t: 1.0, lambda t: 0.3), aa, dict(rows=d), {}))[24] is False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        k = kk(lambda t: 1.0, lambda t: 0.3); F.print_stats("K", k["rows"]); F.print_bands(k["rows"]); F.print_kinds(k["rows"]); F.print_ages(_qrows(10, 5, 5)); F.print_verdict(F.verdict(k, aa, None))
    out = buf.getvalue(); assert "steps/h" in out and "headline:" in out and "age of the fact" in out
    p = F.pilot_report(_k(20, prompt=lambda t: 3000 + 2000 * t), _a(20), 1.0, 10.0, 0.1); assert abs(p["k_growth_per_turn"] - 2000) < 1e-9 and p["cost_k"] < p["cost_k_nocache"] and p["cost_a"] > 0 and F.pilot_report(_k(20), _a(20))["cost_a"] is None
    assert abs(p["k_last_prompt_projected"] - (5000 + 2000 * (F.STEPS - 1))) < 1e-6


def test_the_cuts_by_kind_and_by_age_are_recounted_by_hand():
    rows = [_row(1, kind="add"), _row(2, kind="add", exact=False), _row(3, kind="query", qkind="recall", age=5), _row(4, kind="query", qkind="recall", age=60, exact=False), _row(5, kind="query", qkind="unknown")]
    bk = {b["kind"]: b for b in F.by_kind(rows)}; assert bk["add"]["n"] == 2 and bk["add"]["exact"] == 0.5 and bk["query:recall"]["n"] == 2 and bk["query:unknown"]["exact"] == 1.0
    ba = F.by_age(rows); assert [(b["lo"], b["n"], b["exact"]) for b in ba] == [(0, 1, 1.0), (50, 1, 0.0)]


# ---------------------------------------------------------------- 6 page and notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert _h(_head()) == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F.G_LINES) == 24
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F.MODEL, str(F.SEED), f"{F.K_CONTEXT_CAP:,}", f"{F.STEPS}", f"{F.PAD_SENTENCES} sentences", f"{F.K_TOKEN_CAP:,}", f"{F.A_TOKEN_CAP:,}", f"{F.D_TOKEN_CAP:,}", "Energy in joules", "seconds to the first token", "AXI HOLDS ON FUZZY TEXT", "scratch what I said", "unknown", "ambiguous"): assert needle in head, needle
    assert "ultimate" not in head.lower() and "$" not in head and "proven" not in head.lower() and "100 percent" not in head
    assert F.D_STARTS == (50, 100, 200, 300) and F.MODEL == "gemini-3.8-flash" and F.STEPS == 400


def test_the_notebook_holds_no_key_never_turns_billing_on_and_sets_no_budget():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 21 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    assert "BUDGET_USD = None" in code and "PAID = None" in code and "PRICE_IN = None" in code and "PRICE_CACHED = None" in code and "PRICE_OUT = None" in code
    assert "enable billing" not in code.lower() and "billing" not in code.split("PAID")[0]
    for name in ("run_kept", "run_axi", "run_depth", "pilot_report", "replay_depth", "verdict", "GL.build_chat_llm", "print_kinds", "print_ages"): assert "G." + name in code, name
    assert code.count("need_budget()") >= 3 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code") and "gemini_fuzzy.py" in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 7 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(HERE, "..", "..", "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(HERE, "..", "..", "docs", "VERIFIED.md")).read()
    assert "Next free: row 172, item 125, test 151, question 121" in cm and cm.count("Next free:") == 1 and "test 142 checks the fuzzy-state test on Gemini 3.8 Flash, item 116" in cm and "116. **A long conversation in plain sentences" in cm
    assert "GEMINI_FUZZY_TEST.md" in vf and "gemini_fuzzy.py" in vf and "Run once" in vf[vf.index("GEMINI_FUZZY_TEST.md") - 300: vf.index("GEMINI_FUZZY_TEST.md") + 900]
    i = cm.index("116. **"); item = cm[i:cm.index("\n", i)]; assert "Status: run once on Colab" in item and "docs/GEMINI_FUZZY_TEST.md" in item and "test 142" in item


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

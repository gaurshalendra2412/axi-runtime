"""Checks the INSTRUMENT of docs/RULED_OUT_TEST.md (the ruled-out-causes test), written BEFORE any model run. It checks the instrument, not a result:
  1. the world and the two ways of reading a reply: strict `CHECK[part,fault]` (whole reply, numbers in range) and words (exactly one part and one failure way, any listed alias; two of either or none: not read);
  2. what each arm shows the model: kept and compacted send the chat (the compacted one is emptied into the model's summary every 20 checks); held_prompt and held_gate send a prompt built from the ruled-out
     list alone, and only held_gate refuses a repeat (nothing is checked; one line in the next prompt says so);
  3. what each kind of stand-in model does in each arm: one that never repeats (nobody repeats, the cause is found at its place in the scan, a found episode ends); one that ignores the prompt and picks at
     random (the repeats counted are exactly the repeats that happened, held_gate answers none of them); one that forgets everything when its chat is summarised (only the compacted arm repeats);
     one that writes the same cause in other words (reading the words finds repeats a literal text match misses);
  4. a run that is cut off (also in the middle of a compaction) and started again equals the run that was not; a half-written last line is dropped; a file of another run is refused; a saved flag that the
     replay does not reproduce is refused;
  5. the reading rules read as the page says, at their boundaries, and the page and the notebook agree (the page's text above its Result is byte for byte what was written before any run).
What it does NOT show: anything about a real model's replies (the pre-registered run)."""
import hashlib, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import ruled_out as R
from axi.experiments.resumable import load_rows

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "RULED_OUT_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_ruled_out_test.ipynb")
PAGE_SHA256 = "e98882a1a0c47329870700e03462d9af90d18044d31e4521802b40fac9f97512"          # sha256 of docs/RULED_OUT_TEST.md as written before any model run (text + the final newline)
RESULT_MARK = "\n\n---\n\n# Result"
KEYS = dict(test="ruled_out", mode="keys", steps=R.STEPS_KEYS, compact_every=R.COMPACT_EVERY, seeds=list(R.SEEDS))
WORDS = dict(test="ruled_out", mode="words", steps=R.STEPS_WORDS, compact_every=R.COMPACT_EVERY, seeds=list(R.SEEDS))


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _sentence(key, style=0):
    p, f = key; part = R.PART_NAMES[p].replace("-", " "); way = R.FAULT_ALIASES[f][style % len(R.FAULT_ALIASES[f])]
    return [f"The {part} service {way}.", f"I think {part} is the problem: {way}.", f"{part.capitalize()} - {way}"][style % 3]


class Stand:
    """A stand-in for the model. careful and forgetful are pure functions of the prompt, like a greedy model: careful scans the causes in order and reads how far it got from the prompt (the ruled-out
    list, or the chat, or the position written in its own summary); forgetful is the same, but its summaries do not say how far it got, so after a summary it starts again from the first cause.
    random ignores the prompt; sloppy writes something that is not a check every 5th reply; reworder (words mode) scans 24 causes and then goes over the same 24 in other words (these three keep a counter
    that starts again with each new episode)."""
    def __init__(self, style="careful", mode="keys", seed=0, stop_after=None):
        self.style, self.mode, self.rng, self.i, self.stop_after, self.total, self.last_prompt_tokens, self.summaries, self.log = style, mode, random.Random(seed), 0, stop_after, 0, 0, 0, []

    @staticmethod
    def scan_position(messages):
        last = messages[-1]["content"]; m = re.match(r"Ruled out so far \((\d+) causes", last)
        if m: return int(m.group(1))                                                                                  # a held arm: the list says how many causes are ruled out
        if last.startswith("Nothing is ruled out yet."): return 0
        base = 0
        if messages[1]["content"].startswith(R.SUMMARY_HEAD):
            mm = re.search(r"through cause (\d+)", messages[1]["content"]); base = int(mm.group(1)) if mm else 0       # a chat that starts from a summary
        return base + sum(1 for x in messages[2:] if x["role"] == "assistant" and (x["content"].startswith("CHECK[") or R.ground(x["content"])))

    def generate(self, messages, constrained=False, max_new_tokens=400):
        self.total += 1
        if self.stop_after is not None and self.total > self.stop_after: raise RuntimeError("session dropped")
        self.last_prompt_tokens = len(json.dumps(messages)) // 4; self.log.append(messages)
        last = messages[-1]["content"]
        if messages[0]["content"] == R.COMPACT_SYSTEM:
            self.summaries += 1
            if self.style == "forgetful": return "Summary: I have been checking causes one after another; none of them was the real one.", 12, 0.01
            return f"Summary: I have checked causes through cause {self.scan_position(messages[:-1])}; none of them was the real one.", 12, 0.01
        if last == R.OPEN or last.startswith("Nothing is ruled out yet."): self.i = 0                                  # a new episode
        if self.style in ("careful", "forgetful"):
            key = divmod(self.scan_position(messages), R.N_FAULTS); return (f"CHECK[{key[0]},{key[1]}]" if self.mode == "keys" else _sentence(key)), 6, 0.01
        if self.style == "sloppy" and (self.i + 1) % 5 == 0:
            self.i += 1; return ("I will check the next cause now." if self.mode == "keys" else "Something is wrong somewhere."), 5, 0.01
        if self.style == "random": key = divmod(self.rng.randrange(R.N_PARTS * R.N_FAULTS), R.N_FAULTS)
        elif self.style == "reworder":
            n = self.i; key = divmod(n % 24, R.N_FAULTS); text = _sentence(key, 0 if n < 24 else 1 + (n % 2)); self.i += 1; return text, 9, 0.01   # the first 24 causes, then the same 24 again, in other words
        else: key = divmod(self.i, R.N_FAULTS)
        self.i += 1
        return (f"CHECK[{key[0]},{key[1]}]" if self.mode == "keys" else _sentence(key)), 6, 0.01


def _run(model, arm, seed_i, path, mode="keys", meta=None, steps=None):
    return R.run_arm(model, arm, mode, R.SEEDS[seed_i], path, seed_i, meta=meta if meta is not None else {}, steps=steps)


def _seed_with_cause_at(position):
    for s in range(1, 5000):
        i, j = R.hidden_numbers(s)
        if i * R.N_FAULTS + j == position: return s                                       # the position of the real cause in an in-order scan of the numbers
    raise AssertionError


# ---------------------------------------------------------------- 1. the world and the two readers
def test_the_world_and_the_two_ways_of_reading_a_reply():
    assert (R.N_PARTS, R.N_FAULTS, len(R.SEEDS), R.STEPS_KEYS, R.STEPS_WORDS, R.COMPACT_EVERY) == (30, 8, 3, 60, 30, 20) and len(set(R.PART_NAMES)) == 30 and len(set(R.FAULT_NAMES)) == 8
    hs = [R.hidden(s) for s in R.SEEDS]; assert len(set(hs)) == 3 and all(0 <= p < 30 and 0 <= f < 8 for p, f in hs) and R.hidden(101) == R.hidden(101)
    assert R.parse_key("CHECK[3,5]") == (3, 5) and R.parse_key("  CHECK[29,7]\n") == (29, 7) and R.parse_key("CHECK[30,0]") is None and R.parse_key("CHECK[0,8]") is None
    for bad in ("CHECK[3,5] ok", "check[3,5]", "CHECK(3,5)", "CHECK[3, 5]", "CHECK[3]", "", "I check part 3, fault 5", "CHECK[-1,2]"): assert R.parse_key(bad) is None, bad
    good = {"The billing service times out.": (1, 0), "Webhooks are dropping messages": (29, 6), "the feature flags service sends duplicates": (10, 5), "The sessions are slow": (23, 7),
            "Database: out of memory": (6, 3), "The tax part returns an error": (26, 2), "dns has stale data": (7, 1), "Login - bad credentials": (15, 4), "Check the cache layer timing out!": (2, 0)}
    for t, k in good.items(): assert R.ground(t) == k, t
    for t in ("billing and cache time out", "The billing service is fine", "Something is wrong somewhere.", "the cache is slow and times out", ""): assert R.ground(t) is None, t
    ek, ew = R.Episode("kept", "keys", 101), R.Episode("kept", "words", 101); P, F = R.perms(101)
    assert sorted(P) == list(range(30)) and sorted(F) == list(range(8)) and P != list(range(30)) and R.perms(101) == R.perms(101) and len({tuple(R.perms(s)[0]) for s in R.SEEDS}) == 3 and len({tuple(R.perms(s)[1]) for s in R.SEEDS}) == 3
    assert ek.read("CHECK[1,1]") == (P[1], F[1]) and ek.read("The cache times out") is None and ew.read("CHECK[1,1]") is None and ew.read("The cache times out") == (2, 0)      # keys are read in the episode's numbering, words are not
    assert ek.cause_text((P[3], F[4])) == "[3,4]" and ew.cause_text((P[3], F[4])) == f"{R.PART_NAMES[P[3]]} / {R.FAULT_NAMES[F[4]]}"
    hp, hf = R.hidden(101); assert R.hidden_numbers(101) == (P.index(hp), F.index(hf)) and ek.hidden == (hp, hf)
    sk, sw = R.system_text("keys", 101), R.system_text("words", 101)
    assert "CHECK[part,fault]" in sk and all(f"{i} {R.PART_NAMES[g]}" in sk for i, g in enumerate(P)) and all(f"{j} {R.FAULT_NAMES[g]}" in sk for j, g in enumerate(F))
    assert "CHECK[" not in sw and "plain sentence" in sw and "webhooks" in sw and sk != R.system_text("keys", 102) and sw != R.system_text("words", 102) and sk == ek.system and sw == ew.system


def test_every_cause_can_be_written_in_words_and_read_back_in_two_other_ways():
    n = 0
    for p in range(R.N_PARTS):
        for f in range(R.N_FAULTS):
            for style in range(3): assert R.ground(_sentence((p, f), style)) == (p, f), (p, f, style, _sentence((p, f), style)); n += 1
    assert n == 720


# ---------------------------------------------------------------- 2. what each arm shows
def test_each_arm_shows_the_model_what_the_page_says():
    tmp = tempfile.mkdtemp(); m = {a: Stand("careful") for a in R.ARMS_KEYS}
    ep = {a: _run(m[a], a, 0, os.path.join(tmp, a + ".jsonl"), steps=45)["episode"] for a in R.ARMS_KEYS}
    k, c, hp, hg = (m[a].log for a in R.ARMS_KEYS)
    assert [len(x) for x in (k[0], k[1], k[44])] == [2, 4, 90] and k[0][1]["content"] == R.OPEN and k[1][2] == dict(role="assistant", content="CHECK[0,0]") and k[1][3]["content"] == "NOT [0,0]. Next check?"       # kept: the whole chat
    assert all(len(x) == 2 and x[0]["role"] == "system" and x[1]["role"] == "user" for x in hp + hg) and hp[0] == hg[0] and hp[0][1]["content"] == "Nothing is ruled out yet.\nName ONE cause that is not in this list."
    P = R.perms(R.SEEDS[0])[0]; nm = lambda i: R.PART_NAMES[P[i]]
    assert hp[3][1]["content"] == f"Ruled out so far (3 causes, none of them is the real one):\n  part 0 ({nm(0)}): faults 0, 1, 2\nName ONE cause that is not in this list."
    assert hp[17][1]["content"] == (f"Ruled out so far (17 causes, none of them is the real one):\n  part 0 ({nm(0)}): faults 0, 1, 2, 3, 4, 5, 6, 7\n  part 1 ({nm(1)}): faults 0, 1, 2, 3, 4, 5, 6, 7\n"
                                    f"  part 2 ({nm(2)}): faults 0\nName ONE cause that is not in this list.")                   # the list is grouped by part, so it grows slower than the chat
    assert all("refused" not in x[1]["content"] for x in hp + hg)                                                     # a careful model is never refused
    assert m["compacted"].summaries == 2 and ep["compacted"].compactions == 2 and ep["compacted"].checks == 45                                      # summaries before reply 21 and reply 41
    assert [len(x) for x in c[:3]] == [2, 4, 6] and c[20][0]["content"] == R.COMPACT_SYSTEM and c[20][-1]["content"] == R.COMPACT_GENERIC and len(c[20]) == 1 + 1 + 2 * 20 + 1        # system, opening, 20 pairs, request
    after = [x for x in c if len(x) >= 2 and x[1]["content"].startswith(R.SUMMARY_HEAD)]; assert len(after) >= 3 and after[0][1]["content"].endswith("\n\nNext check?")
    assert len(c[21]) == 2 and c[21][1]["content"].startswith(R.SUMMARY_HEAD) and len(c[22]) == 4                    # after a summary the chat starts again from the summary
    assert max(len(x) for x in c) < len(k[44]) and ep["kept"].checks == 45 and ep["kept"].compactions == 0
    for a in ("kept", "held_prompt", "held_gate"): assert m[a].summaries == 0


def test_a_found_episode_ends_and_the_real_cause_is_never_on_the_ruled_out_list_before_it_is_found():
    seed = _seed_with_cause_at(37); tmp = tempfile.mkdtemp()
    for arm in R.ARMS_KEYS:
        m = Stand("careful"); path = os.path.join(tmp, arm + ".jsonl")
        r = R.run_arm(m, arm, "keys", seed, path, 0, meta={}); ep = r["episode"]
        assert ep.found == 38 and ep.done() and R.hidden_numbers(seed) == (4, 5) and ep.hidden not in ep.ruled and len(ep.ruled) == 37 and len(r["rows"]) - ep.compactions == 38 and m.total - ep.compactions == 38, arm
        assert [x["found"] for x in r["rows"] if x["kind"] == "check"].count(True) == 1 and r["rows"][-1]["found"] and tuple(r["rows"][-1]["key"]) == ep.hidden
        again = R.run_arm(Stand("careful"), arm, "keys", seed, path, 0, meta={}); assert again["rows"] == r["rows"]            # a finished episode asks for nothing more


# ---------------------------------------------------------------- 3. what each stand-in does in each arm
def test_a_model_that_ignores_the_prompt_repeats_and_only_held_gate_refuses_the_repeats():
    tmp = tempfile.mkdtemp(); S = {}
    for arm in ("kept", "held_prompt", "held_gate"):
        m = Stand("random", seed=5); r = _run(m, arm, 0, os.path.join(tmp, arm + ".jsonl"), steps=60); rows = r["rows"]
        seen, rep = set(), 0                                                                                                # recount the repeats from the texts alone
        for x in rows:
            kk = R.parse_key(x["text"]); rep += kk in seen; seen.add(kk)
        hid = R.hidden_numbers(R.SEEDS[0]); distinct = {R.parse_key(x["text"]) for x in rows} - {hid}
        S[arm] = dict(rows=rows, rep=rep, ep=r["episode"], distinct=distinct)
        assert sum(x["repeat"] for x in rows) == rep and len(r["episode"].ruled) == len(distinct), arm
    assert S["kept"]["rep"] == S["held_prompt"]["rep"] > 0 and all(x["refused"] is False for x in S["kept"]["rows"] + S["held_prompt"]["rows"])        # the same replies (the model ignores the prompt)
    g = S["held_gate"]; assert g["rep"] == S["kept"]["rep"] and sum(x["refused"] for x in g["rows"]) == g["rep"] and all(x["refused"] == x["repeat"] for x in g["rows"])
    sm = R.summarize(load_rows(os.path.join(tmp, "held_gate.jsonl"), KEYS), "keys")["held_gate"]
    assert sm["accepted_repeats"] == 0 and sm["refused"] == g["rep"] and sm["refusals_followed"] > 0 and sm["new_ruled"] == len(g["distinct"])
    k = R.summarize(load_rows(os.path.join(tmp, "kept.jsonl"), KEYS), "keys")["kept"]; assert k["accepted_repeats"] == k["repeats"] > 0 and k["new_ruled"] == len(S["kept"]["distinct"])
    # the refusal line: the prompt after a refused reply says so once, and the prompt after the next reply does not
    m = Stand("random", seed=5); _run(m, "held_gate", 0, os.path.join(tmp, "h2.jsonl"), steps=60)
    notes = [i for i, x in enumerate(S["held_gate"]["rows"]) if x["refused"]]; i = notes[0]
    assert m.log[i + 1][1]["content"].endswith(f"Your last reply ({R.Episode('held_gate', 'keys', R.SEEDS[0]).cause_text(tuple(S['held_gate']['rows'][i]['key']))}) was refused: it is already ruled out.") and "refused" not in m.log[i][1]["content"]
    nxt = [j for j in range(i + 1, 59) if not S["held_gate"]["rows"][j]["refused"]][0]; assert "refused" not in m.log[nxt + 1][1]["content"]


def test_a_model_that_forgets_when_its_chat_is_summarised_repeats_only_in_the_compacted_arm():
    tmp = tempfile.mkdtemp(); out = {}
    for arm in R.ARMS_KEYS:
        r = _run(Stand("forgetful"), arm, 0, os.path.join(tmp, arm + ".jsonl"), steps=60); out[arm] = R.summarize(load_rows(os.path.join(tmp, arm + ".jsonl"), KEYS), "keys")[arm]
    assert [out[a]["repeats"] for a in R.ARMS_KEYS] == [0, 40, 0, 0] and out["compacted"]["new_ruled"] == 20 and out["kept"]["new_ruled"] == out["held_gate"]["new_ruled"] == 60 and out["compacted"]["summaries"] == 2
    w = out["compacted"]["windows"]; assert [x["repeats"] for x in w] == [0, 0, 10, 10, 10, 10]
    kw = out["kept"]["windows"]; assert [x["ptok_sum"] // x["ptok_n"] for x in kw] == sorted(x["ptok_sum"] // x["ptok_n"] for x in kw) and kw[-1]["ptok_sum"] // kw[-1]["ptok_n"] > 5 * (kw[0]["ptok_sum"] // kw[0]["ptok_n"]) // 3
    hw = out["held_gate"]["windows"]; assert hw[-1]["ptok_sum"] // hw[-1]["ptok_n"] < kw[-1]["ptok_sum"] // kw[-1]["ptok_n"]


def test_replies_that_are_not_checks_are_counted_and_do_not_change_the_state():
    tmp = tempfile.mkdtemp()
    for arm in R.ARMS_KEYS:
        m = Stand("sloppy"); r = _run(m, arm, 0, os.path.join(tmp, arm + ".jsonl"), steps=30); ep = r["episode"]; rows = [x for x in r["rows"] if x["kind"] == "check"]
        bad = [x for x in rows if x["status"] != "ok"]; assert len(bad) == 6 and all(x["key"] is None and not x["repeat"] and not x["refused"] for x in bad) and ep.checks == 30 and len(ep.ruled) == 24, arm
    k = [x for x in m.log if x[0]["role"] == "system" and x[0]["content"] == R.system_text("keys", R.SEEDS[0])]
    assert any("Your last reply was not a valid check." in x[1]["content"] for x in k)                                  # held arm: the next prompt says so
    m = Stand("sloppy"); _run(m, "kept", 0, os.path.join(tmp, "k2.jsonl"), steps=10)
    assert any(x[-1]["content"] == "Invalid reply. Write exactly CHECK[part,fault]. Next check?" for x in m.log)        # kept arm: the answer in the chat says so
    sm = R.summarize(load_rows(os.path.join(tmp, "kept.jsonl"), KEYS), "keys")["kept"]; assert sm["unread"] == 6 and sm["new_ruled"] == 24 and sm["checks"] == 30


def test_in_words_mode_reading_the_words_finds_repeats_a_literal_match_misses():
    tmp = tempfile.mkdtemp()
    for arm in R.ARMS_WORDS:
        r = _run(Stand("reworder", mode="words"), arm, 0, os.path.join(tmp, arm + ".jsonl"), mode="words", steps=30)
    S = R.summarize(load_rows(os.path.join(tmp, "kept.jsonl"), WORDS), "words"), R.summarize(load_rows(os.path.join(tmp, "held_gate.jsonl"), WORDS), "words")
    for s, arm in zip(S, R.ARMS_WORDS): assert s[arm]["repeats"] == 6 and s[arm]["literal_repeats"] == 0 and s[arm]["unread"] == 0 and s[arm]["new_ruled"] == 24, arm      # replies 25 to 30 go over causes 1 to 6 again, in other words
    assert S[1]["held_gate"]["refused"] == 6 and S[1]["held_gate"]["accepted_repeats"] == 0 and S[0]["kept"]["accepted_repeats"] == 6
    assert all(ok for t, ok in R.verdict_words({"kept": S[0]["kept"], "held_gate": S[1]["held_gate"]}))
    m = Stand("careful", mode="words"); _run(m, "held_gate", 1, os.path.join(tmp, "w2.jsonl"), mode="words", steps=3)
    assert m.log[1][1]["content"] == "Ruled out so far (1 causes, none of them is the real one):\n  auth: timeout\nName ONE cause that is not in this list."
    same = Stand("careful", mode="words"); _run(same, "held_gate", 2, os.path.join(tmp, "w3.jsonl"), mode="words", steps=30)                       # the same sentence twice is a repeat that a literal match finds too
    rows = load_rows(os.path.join(tmp, "w3.jsonl"), WORDS); assert R.summarize(rows, "words")["held_gate"]["literal_repeats"] == 0


# ---------------------------------------------------------------- 4. cut off, started again
def test_a_run_that_is_cut_off_in_the_middle_of_a_compaction_equals_the_run_that_was_not():
    tmp = tempfile.mkdtemp()
    for style in ("careful", "forgetful"):
        ref = _run(Stand(style), "compacted", 0, os.path.join(tmp, style + "_ref.jsonl"), steps=60)
        assert len(ref["rows"]) == 62 and [x["kind"] for x in ref["rows"]].count("summary") == 2 and ref["rows"][20]["kind"] == "summary" and ref["rows"][20]["step"] == 20
        for cut in (7, 20, 21, 22, 45):                                                       # just before, at and just after the first summary, and near the second
            p = os.path.join(tmp, f"{style}_cut{cut}.jsonl")
            try: _run(Stand(style, stop_after=cut), "compacted", 0, p, steps=60); raise AssertionError("did not stop")
            except RuntimeError: pass
            assert len(open(p).read().splitlines()) == 1 + cut                               # header + one line per finished call
            with open(p, "a") as f: f.write('{"arm": "compacted", "ep": 0, "ki')               # a half-written last line
            m2 = Stand(style); again = _run(m2, "compacted", 0, p, steps=60)
            assert again["rows"] == ref["rows"] and m2.total == 62 - cut, (style, cut)       # the same rows, and only the missing calls were made
    for arm in R.ARMS_KEYS:
        base = _run(Stand("careful"), arm, 1, os.path.join(tmp, arm + "_b.jsonl"), steps=40)["rows"]; p = os.path.join(tmp, arm + "_c.jsonl")
        try: _run(Stand("careful", stop_after=13), arm, 1, p, steps=40)
        except RuntimeError: pass
        assert _run(Stand("careful"), arm, 1, p, steps=40)["rows"] == base, arm


def test_a_file_of_another_run_is_refused_and_a_saved_flag_that_the_replay_does_not_give_is_refused():
    tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "a.jsonl"); _run(Stand("careful"), "kept", 0, p, steps=5)
    try: _run(Stand("careful"), "kept", 0, p, mode="words", steps=5); raise AssertionError("a file of the other mode was accepted")
    except (ValueError, AssertionError) as e: assert "different run" in str(e) or "not part of mode" in str(e) or "different run" in repr(e)
    q = os.path.join(tmp, "b.jsonl"); _run(Stand("careful"), "kept", 0, q, steps=5)
    lines = open(q).read().splitlines(); row = json.loads(lines[3]); row["repeat"] = True; lines[3] = json.dumps(row); open(q, "w").write("\n".join(lines) + "\n")
    try: _run(Stand("careful"), "kept", 0, q, steps=8); raise AssertionError("a doctored flag was accepted")
    except ValueError as e: assert "repeat" in str(e)
    lines = open(p).read().splitlines(); row = json.loads(lines[2]); row["kind"] = "summary"; lines[2] = json.dumps(row); open(p, "w").write("\n".join(lines) + "\n")
    try: _run(Stand("careful"), "kept", 0, p, steps=8); raise AssertionError("a wrong kind of row was accepted")
    except ValueError as e: assert "state needs" in str(e)
    try: R.Episode("compacted", "words", 1); raise AssertionError("an arm of another mode was accepted")
    except ValueError: pass


def test_run_mode_runs_every_arm_over_every_seed_in_the_order_of_the_page_and_a_second_call_asks_nothing():
    tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "k.jsonl"); m = Stand("careful"); res = R.run_mode(m, "keys", p, meta={})
    S = res["summary"]; assert [S[a]["episodes"] for a in R.ARMS_KEYS] == [3, 3, 3, 3]
    assert all(S[a]["checks"] == 180 and S[a]["found"] == 0 for a in R.ARMS_KEYS) and S["compacted"]["summaries"] == 6                  # an in-order scan reaches the three causes at replies 215, 155 and 76
    order = [(r["arm"], r["ep"]) for r in res["rows"] if r["step"] == 1 and r["kind"] == "check"]; assert order == [(a, e) for e in range(3) for a in R.ARMS_KEYS]
    n = m.total; again = R.run_mode(m, "keys", p, meta={}); assert m.total == n and again["rows"] == res["rows"]
    assert len(res["rows"]) == 4 * 180 + 6 == 726                                              # the most calls keys mode can need


# ---------------------------------------------------------------- 5. the reading rules
def _cell(**kw):
    w = [dict(first=51, last=60, n=30, repeats=0, unread=0, ptok_sum=3000, ptok_n=3)]
    base = dict(episodes=3, checks=180, repeats=0, refused=0, accepted_repeats=0, unread=0, new_ruled=100, found=0, found_steps=0, summaries=0, tokens=0, seconds=0.0, windows=w,
                refusals_followed=10, refusals_then_repeat=0, literal_repeats=0); base.update(kw); return base


def _K(**over):
    S = dict(kept=_cell(repeats=4, windows=[dict(first=51, last=60, n=30, repeats=0, unread=0, ptok_sum=9000, ptok_n=3)]), compacted=_cell(repeats=9, new_ruled=60), held_prompt=_cell(), held_gate=_cell(repeats=3, refused=3, new_ruled=80))
    for k, v in over.items(): S[k] = {**S[k], **v}
    return S


def _ok(S): return [ok for _, ok in R.verdict_keys(S)]


def test_the_reading_rules_for_keys_mode_read_as_the_page_says_at_their_boundaries():
    assert _ok(_K()) == [True] * 8 and [t.split("]")[0] for t, _ in R.verdict_keys(_K())] == ["[guard", "[guard", "[format", "[need", "[need", "[need", "[need", "[need"]
    assert _ok(_K(held_gate=dict(accepted_repeats=1)))[0] is False and _ok(_K(held_gate=dict(checks=0)))[0] is False and _ok(_K(held_prompt=dict(episodes=2)))[1] is False
    assert _ok(_K(kept=dict(unread=9)))[2] is True and _ok(_K(kept=dict(unread=10)))[2] is False and _ok(_K(held_gate=dict(unread=10)))[2] is False                      # 20 x unread at most 180
    assert _ok(_K(compacted=dict(repeats=5)))[3] is True and _ok(_K(compacted=dict(repeats=4)))[3] is False
    assert _ok(_K(held_gate=dict(new_ruled=75)))[4] is True and _ok(_K(held_gate=dict(new_ruled=74)))[4] is False                                                         # 60 + 15
    assert _ok(_K(held_gate=dict(repeats=4)))[5] is True and _ok(_K(held_gate=dict(repeats=5)))[5] is False
    assert _ok(_K(held_gate=dict(refusals_then_repeat=2)))[6] is True and _ok(_K(held_gate=dict(refusals_then_repeat=3)))[6] is False and _ok(_K(held_gate=dict(refusals_followed=0, refusals_then_repeat=0)))[6] is False
    w = lambda ps, pn: [dict(first=51, last=60, n=30, repeats=0, unread=0, ptok_sum=ps, ptok_n=pn)]
    assert _ok(_K(held_gate=dict(windows=w(8999, 3))))[7] is True and _ok(_K(held_gate=dict(windows=w(9000, 3))))[7] is False and _ok(_K(held_gate=dict(windows=w(0, 0))))[7] is False


def test_the_reading_rules_for_words_mode_read_as_the_page_says_at_their_boundaries():
    S = lambda **o: {"kept": {**_cell(repeats=3, literal_repeats=2), **o.get("kept", {})}, "held_gate": {**_cell(repeats=2, literal_repeats=2), **o.get("held_gate", {})}}
    ok = lambda s: [x for _, x in R.verdict_words(s)]
    assert ok(S()) == [True, True, True] and [t.split("]")[0] for t, _ in R.verdict_words(S())] == ["[format", "[need", "[need"]
    assert ok(S(kept=dict(unread=18)))[0] is True and ok(S(kept=dict(unread=19)))[0] is False and ok(S(held_gate=dict(unread=19)))[0] is False                            # 10 x unread at most 180
    assert ok(S(kept=dict(repeats=2)))[1] is False and ok(S(kept=dict(repeats=3)))[1] is True                                                                         # (3 + 2) - (2 + 2) >= 1
    assert ok(S(held_gate=dict(repeats=3)))[2] is True and ok(S(held_gate=dict(repeats=4)))[2] is False


# ---------------------------------------------------------------- the page and the notebook
def _cells(): nb = json.load(open(NB)); return nb, nb["cells"]


def _src(c): return "".join(c["source"])


def _no_bang(s): return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith(("!", "%")))


def test_the_page_and_the_notebook_agree_and_the_page_is_locked():
    import ast
    nb, cells = _cells(); text = open(PAGE).read(); code = "\n".join(_no_bang(_src(c)) for c in cells if c["cell_type"] == "code"); ast.parse(code)
    assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    assert PAGE_SHA256 is not None and _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    assert "R.run_mode(llm, \"keys\"" in code and "R.run_mode(llm, \"words\"" in code and "R.verdict_keys(" in code and "R.verdict_words(" in code and 'WANTED = "gemini-3.5-flash-lite"' in code and "daily_limit=480" in code
    for w in ("[format]", "[guard]", "[need]"): assert w in text
    assert "## Predictions" in text and "## Reading rules" in text and "## What this cannot show" in text and "726 calls" in text and "180 calls" in text and "906" in text
    nlines = len(R.verdict_keys(_K())) + len(R.verdict_words({"kept": _cell(repeats=3), "held_gate": _cell(repeats=2)})); assert nlines == 11 and "Eleven lines" in text
    assert "`python/colab/axi_ruled_out_test.ipynb`" in text and "`python/axi/experiments/ruled_out.py`" in text and "`python/tests/test_ruled_out.py`" in text
    assert KEY_FREE(nb)


def KEY_FREE(nb): s = json.dumps(nb); return "AIza" not in s and "GEMINI_API_KEY" in s



# ---------------------------------------------------------------- the notebook, run cell by cell against a fake Colab and a fake Gemini
class ModeStand:
    """One stand-in for both modes: it reads the mode from the system text of the prompt. keys: forgetful (so the compacted arm repeats); words: careful."""
    def __init__(self): self.s = {"keys": Stand("forgetful", "keys"), "words": Stand("careful", "words")}; self.last_prompt_tokens = 0
    def generate(self, messages):
        mode = "keys" if "CHECK[part,fault]" in messages[0]["content"] else "words"; out = self.s[mode].generate(messages); self.last_prompt_tokens = self.s[mode].last_prompt_tokens; return out


def _fake_world_chat(clock, key=None):
    import types
    from test_api_gemini import KEY
    from test_colab_persist import _fake_drive
    key = key or KEY; mods, _ = _fake_drive(True); st = types.SimpleNamespace(calls=0, models=[], keys=[], downloaded=[], prompts=[], inner=ModeStand())
    class Files:
        def upload(self): return {"axi-colab-ruled-out.zip": b""}
        def download(self, p): st.downloaded.append(os.path.basename(p))
    class Secrets:
        @staticmethod
        def get(name):
            if name != "GEMINI_API_KEY": raise KeyError("Secret does not exist")
            return key
    class Content:
        def __init__(s, role, parts): s.role, s.parts = role, parts
    class Part:
        def __init__(s, text): s.text = text
    class Cfg:
        def __init__(s, **kw): s.kw = kw
    class Chunk:
        def __init__(s, text, um, fin): s.text, s.usage_metadata, s.candidates = text, um, [types.SimpleNamespace(finish_reason=fin)]
    class Client:
        def __init__(self, api_key=None):
            st.keys.append(api_key); self.models = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(name=n) for n in ("models/gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite-tts", "models/gemini-3.8-flash")],
                                                                          generate_content_stream=self._gen)
        def _gen(self, model, contents, config):
            st.calls += 1; st.models.append(model); clock.advance(1.0)
            msgs = [dict(role="system", content=config.kw["system_instruction"])] + [dict(role="assistant" if c.role == "model" else "user", content=c.parts[0].text) for c in contents]
            st.prompts.append(json.dumps(msgs, sort_keys=True)); text, ntok, sec = st.inner.generate(msgs)
            um = types.SimpleNamespace(prompt_token_count=st.inner.last_prompt_tokens, candidates_token_count=ntok, thoughts_token_count=0, cached_content_token_count=0)
            return iter([Chunk(text, um, "STOP")])
    mods["google.colab"].files = Files(); mods["google.colab"].userdata = Secrets; mods["google.colab.userdata"] = Secrets
    google = mods["google"]; genai = types.ModuleType("google.genai"); gt = types.ModuleType("google.genai.types")
    gt.Content, gt.Part, gt.GenerateContentConfig = Content, Part, Cfg; gt.ThinkingConfig = Cfg; genai.Client, genai.types, google.genai = Client, gt, genai
    mods["google.genai"], mods["google.genai.types"] = genai, gt
    return mods, st


def _code_index(contains): return [i for i, c in enumerate(_cells()[1]) if c["cell_type"] == "code" and contains in _src(c)][0]


def _run_notebook(cells_to_run, work, drive_root, clock, daily=None, ns=None, world=None):
    import contextlib, io, time
    from unittest import mock
    from test_colab_persist import patched_modules
    nb = json.load(open(NB)); out = io.StringIO(); mods, st = world or _fake_world_chat(clock); ns = {} if ns is None else ns; early = False
    with patched_modules(mods), contextlib.redirect_stdout(out), mock.patch.object(time, "time", clock.now), mock.patch.object(time, "sleep", clock.advance):
        for i in cells_to_run:
            c = nb["cells"][i]; assert c["cell_type"] == "code", i
            src = _no_bang(_src(c)).replace('open_store("/content/out", "axi_ruled_out")', f'open_store("{work}", "axi_ruled_out", "{drive_root}")').replace('"/content/smoke_ruled.jsonl"', f'"{work}/smoke_ruled.jsonl"')
            src = src.replace('sys.path.insert(0, "/content/axi/python")', "")
            if daily is not None: src = src.replace("daily_limit=480", f"daily_limit={daily}")
            assert "/content" not in src, src
            try: exec(compile(src, f"cell{i}", "exec"), ns)
            except AssertionError as e:
                if "stopped for today" in str(e): early = True; break
                raise
    return ns, out.getvalue(), st, early


def test_the_notebook_runs_end_to_end_stops_at_the_limit_and_finishes_the_next_day_like_an_unstopped_run():
    from test_api_gemini import Clock, KEY, ts
    tmp = tempfile.mkdtemp(); A = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'name = "axi_ruled_out_keys"', 'name = "axi_ruled_out_words"')]
    c = Clock(); work = os.path.join(tmp, "ref"); os.makedirs(work)
    ref, out_ref, st_ref, early = _run_notebook(A, work, os.path.join(tmp, "drive_ref"), c, daily=10 ** 6); assert not early
    total = st_ref.calls; assert "SMOKE OK" in out_ref and len(ref["smoke"]["rows"]) == 6 and ref["res_keys"] is not None and ref["res_words"] is not None
    kr, wr = ref["res_keys"]["rows"], ref["res_words"]["rows"]; assert len(kr) == 726 and len(wr) == 180 and total == len(set(st_ref.prompts))      # nothing asked twice (the smoke calls are the first 6 of the run)
    assert 660 <= total <= 726 + 180 - 3 * 20 - 180, total   # at most 666: compacted shares its first 20 replies with kept in each episode and held_gate is held_prompt when nobody repeats (the same prompts, read from the file); this stand-in's constant summary makes 4 more
    for need in ("[guard]", "[format]", "[need]", "keys mode (reading rule fixed before the run)", "words mode (reading rule fixed before the run)", "the model's summary after reply 20 of episode 0"): assert need in out_ref, need
    assert out_ref.count("  yes  ") + out_ref.count("  NO   ") == 11 or out_ref.count("yes") + out_ref.count("NO ") >= 11
    for limit in (200, 500):
        drive = os.path.join(tmp, f"drive_{limit}"); saved = os.path.join(drive, "MyDrive", "axi_ruled_out")
        c = Clock(); w1 = os.path.join(tmp, f"d1_{limit}"); os.makedirs(w1)
        ns1, out1, st1, early1 = _run_notebook(A, w1, drive, c, daily=limit)
        assert "STOPPED FOR NOW" in out1 and "Daily budget reached" in out1 and "12:30 pm in India until 1 November, 1:30 pm after" in out1 and st1.calls == limit
        assert KEY not in out1 and not any(KEY in open(os.path.join(d, f)).read() for d, _, fs in os.walk(tmp) for f in fs if f.endswith((".json", ".jsonl")))
        days, day, ns2 = [st1.calls], 0, ns1
        while ns2.get("res_words") is None:                                                   # each day the notebook is opened again: Cells 2 to 5, then the cell that stopped
            day += 1; assert day <= 6, "never finished"; c.t = ts(2026, 10, 9 + day, 20); w2 = os.path.join(tmp, f"d{day + 1}_{limit}"); os.makedirs(w2)
            ns2, out2, st2, early2 = _run_notebook(A, w2, drive, c, daily=limit); days.append(st2.calls)
        assert "reply file: restored" in out2 and "Drive connected" in out2 and sum(days) == total and all(d <= limit for d in days), (limit, days, total)      # no prompt asked twice across the days
        assert ns2["res_keys"]["rows"] == ref["res_keys"]["rows"] and ns2["res_words"]["rows"] == ref["res_words"]["rows"] and ns2["res_keys"]["summary"] == ref["res_keys"]["summary"]
        for f in ("axi_ruled_out_cache.jsonl", "axi_ruled_out_keys.jsonl", "axi_ruled_out_words.jsonl", "axi_ruled_out_keys.json"): assert os.path.exists(os.path.join(saved, f)), f
    assert set(st_ref.models) == {"gemini-3.5-flash-lite"} and st_ref.keys == [KEY]
    nb = json.load(open(NB)); assert KEY not in json.dumps(nb) and "AIza" not in json.dumps(nb)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(f"{n} tests passed")

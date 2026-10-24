"""Test 26 on a hosted model (docs/GEMINI_QUORUM_TEST.md): Rajnish Choubey's witness quorum (axi/engine/witnesses.py, concept-map item 122) on
Gemini 3.8 Flash, on the fuzzy-state job of Test 23.

Why this job and this setting. With its default thinking, Gemini 3.8 Flash held Test 23's list exactly on 400 of 400 steps with AXI, so a
quorum would have nothing to fix there. With thinking set to 'minimal' one reply should slip more often, and the company question becomes
real: can a few cheap witnesses that do not think replace one call that does?

Arms (the SAME stream and chatter as Test 23: seed 20261122, 400 steps; thinking 'minimal' in both, or 'low' if the service refuses
'minimal', decided in the pilot and recorded in the run's meta; temperature the model's own default):
  S  one witness: exactly arm A of Test 23 (`gemini_fuzzy.run_axi`): the exact list shown at every step, the check before the list is
     touched, up to two re-asks with the reason; a reply that still fails changes nothing.
  Q  witnesses: the same prompt, each witness told its number (so that each is its own call); witnesses are heard one after another; after
     each from the second on, a change (or an answer) that exactly NEED = 2 witnesses proposed, with more votes than any other, is
     committed and no further witness is heard; at most MAX_WITNESSES = 5; if no change has that lead after five, nothing changes. A reply
     the check refuses is set aside (no re-ask with the reason). Two proposals are the same change when they leave the same list.
Reference, no call: Test 23's saved arm A (default thinking) and arm K, `docs/results/axi_gemini_fuzzy_seed20261122_default.json`.

Measured per call as in Test 23. The reading lines are fixed in the page before any call. It imports gemini_fuzzy, gemini_long, api_llm and
the witnesses module and edits none of them. Nothing of the framework's core is used or changed."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional, Tuple

from axi.engine.witnesses import tally_keys
from axi.experiments import gemini_fuzzy as F
from axi.experiments import gemini_long as GL
from axi.experiments.resumable import load_rows, _roundtrip

# ---------------------------------------------------------------- the settings written on the page
MODEL, SEED, SEED_PILOT, STEPS = F.MODEL, F.SEED, F.SEED_PILOT, F.STEPS
THINKING, THINKING_FALLBACK = "minimal", "low"
NEED, MAX_WITNESSES = 2, 5
S_TOKEN_CAP, Q_TOKEN_CAP = 1_500_000, 4_000_000              # tokens sent: the guard behind the dollar budget
REF_FILE = "axi_gemini_fuzzy_seed20261122_default.json"
REF_THINK_PER_CALL = 44607 / 400                             # Test 23 arm A, default thinking (no re-asks): 111.5 thinking tokens a call
REF_COMPUTED_PER_EXACT = 1322.09                             # Test 23 arm A: computed tokens per exact step
REF_SECONDS_PER_STEP = 2.75701025                            # Test 23 arm A: seconds a step


# ---------------------------------------------------------------- one witness
def witness_tag(w: int) -> str: return f"\n\n(You are witness {w}. Answer on your own.)"


def witness_messages(seed: int, t: int, st: dict, live: dict, w: int, sentences: int = F.PAD_SENTENCES) -> list:
    return [F._system("A"), {"role": "user", "content": F.user_text(seed, t, st, live, sentences) + witness_tag(w)}]


def key_of(live: dict, st: dict, text: str) -> Optional[str]:
    """The canonical change a witness proposes: for a message, the list it would leave; for a question, the answer. None if the check refuses it."""
    ok, _ = F.usable(live, st, text)
    if not ok: return None
    p = F.parse_reply(text)
    if st["kind"] == "query": return "answer:" + F.norm_answer(p["answer"])
    return "list:" + json.dumps(F.apply_ops(live, F.check_ops(live, p["ops"])[2]), sort_keys=True)


def decide(keys: List[Optional[str]]) -> Tuple[Optional[str], str, int]:
    """The key proposed by at least NEED witnesses with more votes than any other, else None (and why)."""
    k, why, votes, _ = tally_keys(keys, need=NEED); return k, why, votes


def stop_point(keys: List[Optional[str]]) -> Tuple[Optional[str], str, int, int]:
    """The rule as a function of the keys of all witnesses in the order heard: (winner, reason, votes, witnesses heard)."""
    win, why, votes = None, "none admissible", 0
    for w in range(1, min(len(keys), MAX_WITNESSES) + 1):
        if w >= NEED:
            win, why, votes = decide(keys[:w])
            if win is not None: return win, why, votes, w
    return None, why, votes, min(len(keys), MAX_WITNESSES)


# ---------------------------------------------------------------- arm Q
def _row(t: int, st: dict, live: dict, ob: dict, oa: dict, introduced, calls: list, keys: list, win, why, votes) -> Tuple[dict, dict]:
    text = next((c["text"] for c, k in zip(calls, keys) if win is not None and k == win), calls[0]["text"])
    p = F.parse_reply(text) if win is not None else None; after = F.copy_state(live)
    if win is not None and st["kind"] != "query": after = F.apply_ops(live, F.check_ops(live, p["ops"])[2])
    fl = F.row_fields(after, oa, ob, st, p, introduced)
    first = calls[0]; top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
    rec = dict(t=t, kind=st["kind"], text=text, seconds=round(sum(c["seconds"] for c in calls), 4), **top, **fl, committed=win is not None, decision=why, votes=votes,
               heard=len(calls), witness_texts=[c["text"] for c in calls], admissible=[k is not None for k in keys],
               first_two_agree=(keys[0] is not None and keys[0] == keys[1]) if len(keys) > 1 else None, reasks=calls[1:])
    return _roundtrip(rec), after


def run_quorum(llm, seed: int, steps: int, path: str, meta=None, sentences: int = F.PAD_SENTENCES, max_tokens: Optional[int] = None, progress=None, scripted: int = 0) -> dict:
    """Arm Q. `scripted`: the first steps are answered by the script (no call), so that the pilot sees the list at its full size."""
    meta = dict(meta or {}, arm="Q", seed=seed, steps=steps, sentences=sentences, stream=F.STREAM, policy="witnesses", need=NEED, max_witnesses=MAX_WITNESSES)
    if scripted: meta["scripted"] = scripted
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} steps, more than the {steps} asked for")
    stream, live = replay_q(rows, seed, sentences); stopped = None; spent = F.tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live
            if t < scripted:
                calls = [dict(text=st["ideal"], tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, seconds=0.0, from_disk=True, finish="scripted")]
                keys = [key_of(live, st, st["ideal"])]; win, why, votes = keys[0], "scripted", 1
            else:
                calls, keys = [], []; win, why, votes = None, "none admissible", 0
                for w in range(1, MAX_WITNESSES + 1):
                    text, ntok, sec = llm.generate(witness_messages(seed, t + 1, st, live, w, sentences), constrained=True)
                    calls.append(F._att(llm, text, ntok, sec)); keys.append(key_of(live, st, text))
                    if w >= NEED:
                        win, why, votes = decide(keys)
                        if win is not None: break
            rec, after = _row(t + 1, st, live, ob, oa, stream.introduced, calls, keys, win, why, votes)
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); live = after; spent += F.billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=F.summarize(rows), meta=_roundtrip(meta), stopped=stopped)


def replay_q(rows: list, seed: int, sentences: int = F.PAD_SENTENCES):
    """Run the stream and the witness rule over the saved witness replies (no call). Returns (stream, held list). Raises at the first vote, stop or flag that differs."""
    stream = F.make_stream(seed); live: dict = {}
    for rec in rows:
        out = stream.next()
        if out is None: raise F.ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = stream.live
        if st["kind"] != rec["kind"]: raise F.ReplayMismatch(f"step {rec['t']}: kind {st['kind']} against saved {rec['kind']}")
        texts = rec["witness_texts"]; keys = [key_of(live, st, x) for x in texts]
        if rec["decision"] == "scripted":
            if len(texts) != 1 or keys[0] is None: raise F.ReplayMismatch(f"step {rec['t']}: a scripted step must hold one usable reply")
            win, why, votes, heard = keys[0], "scripted", 1, 1
        else:
            win, why, votes, heard = stop_point(keys)
            expected = heard if win is not None else MAX_WITNESSES
            if expected != len(texts): raise F.ReplayMismatch(f"step {rec['t']}: {len(texts)} witnesses on file, the rule stops at {expected}")
        calls = [dict(text=x["text"], seconds=x["seconds"], tokens=x.get("tokens"), think=x.get("think"), prompt_tokens=x.get("prompt_tokens"), cache_tokens=x.get("cache_tokens"),
                      first_token=x.get("first_token"), from_disk=x.get("from_disk"), finish=x.get("finish")) for x in [rec] + rec.get("reasks", [])]
        calls[0]["text"] = texts[0]
        new, after = _row(rec["t"], st, live, ob, oa, stream.introduced, calls, keys, win, why, votes)
        diff = [k for k in ("text", "committed", "decision", "votes", "heard", "admissible", "first_two_agree", "parsed", "reply_ok", "reply_exact", "corrupted", "distance", "exact",
                            "nodes", "oracle_nodes", "qkind", "age") if new.get(k) != rec.get(k)]
        if [x["text"] for x in rec.get("reasks", [])] != texts[1:]: diff.append("witness texts")
        if diff: raise F.ReplayMismatch(f"step {rec['t']}: {diff} differ from the saved ones")
        live = after
    return stream, live


# ---------------------------------------------------------------- what the reading lines read
def live_rows(rows: list) -> list: return [r for r in rows if r.get("finish") != "scripted"]


def call_texts(r: dict) -> List[str]:
    """Every reply text of a step, in the order received (arm S: first ask and re-asks; arm Q: the witnesses)."""
    if "witness_texts" in r: return list(r["witness_texts"])
    return ([r["first_text"]] + [x["text"] for x in r["reasks"]]) if r.get("reasks") else [r["text"]]


def committed(r: dict) -> bool:
    return bool(r["committed"]) if "committed" in r else bool(r["parsed"] and not r.get("blocked"))


def wrong_commits(rows: list) -> int:
    """Steps where a change or an answer was accepted (it passed the check) and was not the true one: the legal-but-wrong case."""
    return sum(1 for r in rows if committed(r) and not r["reply_exact"])


def shape_ok(kind: str, text: str) -> bool:
    p = F.parse_reply(text); return p is not None and (("answer" in p) if kind == "query" else ("ops" in p))


def parse_share(rows: list) -> float:
    xs = [shape_ok(r["kind"], x) for r in live_rows(rows) for x in call_texts(r)]; return sum(xs) / len(xs) if xs else None


def think_per_call(rows: list) -> Optional[float]:
    calls = [p for r in live_rows(rows) for p in GL.parts_of(r) if not p["disk"]]; return sum(p["think"] for p in calls) / len(calls) if calls else None


def witnesses_per_step(rows: list) -> Optional[float]:
    lr = live_rows(rows); return sum(r["heard"] for r in lr) / len(lr) if lr else None


def disagreement(rows: list) -> Tuple[int, int]:
    """(steps where the first two witnesses did not propose the same change, of those the steps whose committed change was exact)."""
    d = [r for r in live_rows(rows) if r.get("first_two_agree") is False]; return len(d), sum(1 for r in d if r["reply_exact"])


def agree_share(rows: list) -> Optional[float]:
    lr = [r for r in live_rows(rows) if r.get("first_two_agree") is not None]; return sum(1 for r in lr if r["first_two_agree"]) / len(lr) if lr else None


def answered_exact(rows: list) -> int: return sum(1 for r in rows if r["kind"] == "query" and r["reply_exact"])


def load_reference(results_dir: str) -> dict:
    d = json.load(open(os.path.join(results_dir, REF_FILE))); return dict(A=d["A"], K=d["K"])


# ---------------------------------------------------------------- the reading lines (fixed in the page before any call)
G_S_DONE = "arm S reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask"
G_Q_DONE = "arm Q reached step 400 with every witness reply on file, and the replay reproduces every vote, every stop and every flag"
G_THINK = "the minimal thinking took effect: the thinking tokens a call of arm S are at most a quarter of those of Test 23's arm A (111.5)"
G_PARSE = "at least 99 percent of all replies of arms S and Q (every re-ask and every witness) are one JSON object of the right shape"
G_COST_EXACT = "minimal thinking costs exactness: arm S held the list exactly on fewer than 400 of the 400 steps"
G_Q_NOT_WORSE = "arm Q held the list exactly on at least as many steps as arm S"
G_Q_BETTER = "arm Q held the list exactly on more steps than arm S"
G_Q_ALL = "arm Q held the list exactly on 400 of the 400 steps"
G_WRONG = "arm Q committed at most half as many legal-but-wrong changes and answers as arm S (when arm S committed none, arm Q must commit none)"
G_WITNESSES = "arm Q heard at most 2.3 witnesses a step on average"
G_SPLIT = "on the steps where the first two witnesses did not agree, the committed change was exact on at least 80 percent (not read with fewer than 5 such steps)"
G_COST_Q = "arm Q's computed tokens per exact step are at most 2.5 times arm S's"
G_CHEAPER = "arm Q's computed tokens per exact step are below those of Test 23's arm A with default thinking (1,322)"
G_FASTER = "arm Q's seconds a step are at most those of Test 23's arm A with default thinking (2.76 s)"
G_AGREE = "the first two witnesses of arm Q proposed the same change on at least 95 percent of the steps"
G_QUESTIONS = "arm Q answered at least as many of the questions exactly as arm S"
LINES = [(1, "guard", G_S_DONE), (2, "guard", G_Q_DONE), (3, "guard", G_THINK), (4, "instrument", G_PARSE), (5, "claim", G_COST_EXACT), (6, "claim", G_Q_NOT_WORSE), (7, "claim", G_Q_BETTER),
         (8, "claim", G_Q_ALL), (9, "claim", G_WRONG), (10, "claim", G_WITNESSES), (11, "claim", G_SPLIT), (12, "claim", G_COST_Q), (13, "claim", G_CHEAPER), (14, "claim", G_FASTER),
         (15, "claim", G_AGREE), (16, "claim", G_QUESTIONS)]


def verdict(s: Optional[dict], q: Optional[dict], replays: Optional[dict] = None) -> List[Tuple[int, str, str, Optional[bool]]]:
    """The 16 lines, read from the two runs: True (yes), False (NO), None (not read: the run or the steps it needs are missing)."""
    replays = replays or {}; sr = s["rows"] if s else None; qr = q["rows"] if q else None
    ss = GL.stats(sr) if sr else None; sq = GL.stats(qr) if qr else None
    held = lambda rows: sum(1 for r in rows if r["exact"])
    c = {}
    c[1] = None if sr is None else (len(sr) == STEPS and bool(replays.get("S")))
    c[2] = None if qr is None else (len(qr) == STEPS and bool(replays.get("Q")))
    c[3] = None if sr is None else (think_per_call(sr) is not None and think_per_call(sr) <= REF_THINK_PER_CALL / 4)
    c[4] = None if (sr is None or qr is None) else (min(parse_share(sr), parse_share(qr)) >= 0.99)
    c[5] = None if sr is None or len(sr) < STEPS else held(sr) < STEPS
    both = sr is not None and qr is not None and len(sr) == STEPS and len(qr) == STEPS
    c[6] = held(qr) >= held(sr) if both else None
    c[7] = held(qr) > held(sr) if both else None
    c[8] = (held(qr) == STEPS) if qr is not None and len(qr) == STEPS else None
    if both:
        ws, wq = wrong_commits(sr), wrong_commits(qr); c[9] = (wq == 0) if ws == 0 else (wq * 2 <= ws)
    else: c[9] = None
    c[10] = None if qr is None else (witnesses_per_step(qr) is not None and witnesses_per_step(qr) <= 2.3)
    n_split, n_ok = disagreement(qr) if qr is not None else (0, 0); c[11] = None if (qr is None or n_split < 5) else (n_ok / n_split >= 0.8)
    c[12] = None if not both or not ss["computed_per_exact"] or not sq["computed_per_exact"] else sq["computed_per_exact"] <= 2.5 * ss["computed_per_exact"]
    c[13] = None if qr is None or not sq["computed_per_exact"] else sq["computed_per_exact"] < REF_COMPUTED_PER_EXACT
    c[14] = None if qr is None or sq["seconds_per_step"] is None else sq["seconds_per_step"] <= REF_SECONDS_PER_STEP
    c[15] = None if qr is None or agree_share(qr) is None else agree_share(qr) >= 0.95
    c[16] = answered_exact(qr) >= answered_exact(sr) if both else None
    return [(n, tag, text, c[n]) for n, tag, text in LINES]


def headline(checks) -> str:
    v = {n: c for n, _, _, c in checks}
    if not (v[1] and v[2] and v[4]): return "NOT READ"
    if v[5] is False: return "NOTHING TO FIX"
    if v[7] and v[12]: return "THE QUORUM PAYS"
    if v[7] is False: return "NO GAIN"
    return "MIXED"


def pilot_report(s_rows: list, q_rows: list, price_in=None, price_out=None) -> dict:
    """What the live pilot steps say about 400 steps of each arm: tokens a step, thinking a call, witnesses a step, seconds, and the projected cost at the prices typed in."""
    def per(rows):
        calls = [p for r in live_rows(rows) for p in GL.parts_of(r) if not p["disk"]]; n = max(1, len(live_rows(rows)))
        return dict(prompt=sum(p["prompt"] for p in calls) / n * STEPS, out=sum(p["vis"] + p["think"] for p in calls) / n * STEPS, seconds=sum(r["seconds"] for r in live_rows(rows)) / n)
    a, b = per(s_rows), per(q_rows)
    cost = lambda x: None if price_in is None or price_out is None else (x["prompt"] * price_in + x["out"] * price_out) / 1e6
    return dict(s_think_per_call=think_per_call(s_rows), q_think_per_call=think_per_call(q_rows), q_witnesses_per_step=witnesses_per_step(q_rows), s_seconds_per_step=a["seconds"],
                q_seconds_per_step=b["seconds"], s_tokens_400=a["prompt"] + a["out"], q_tokens_400=b["prompt"] + b["out"], cost_s=cost(a), cost_q=cost(b),
                cost_both=None if cost(a) is None else cost(a) + cost(b))


# ---------------------------------------------------------------- printing (the notebook)
def print_q(rows: list):
    lr = live_rows(rows); from collections import Counter
    heard = Counter(r["heard"] for r in lr); dec = Counter(r["decision"] for r in lr); n_split, n_ok = disagreement(rows)
    print(f"witnesses a step: {_f(witnesses_per_step(rows), 2)} (distribution {dict(sorted(heard.items()))}) | decisions {dict(dec)} | first two agree {_f(None if agree_share(rows) is None else 100 * agree_share(rows))}% "
          f"| disagreements {n_split}, committed exact on {n_ok} | legal-but-wrong commits {wrong_commits(rows)}")


def print_verdict(checks, title="WITNESS QUORUM TEST (reading lines fixed before the run)"):
    print(title)
    for n, tag, text, c in checks: print(f"  {n:>2}. [{tag}] {'yes' if c else ('NO' if c is False else 'not read')}  {text}")
    print("  HEADLINE:", headline(checks))


def _f(x, k=1): return "n/a" if x is None else f"{x:,.{k}f}"

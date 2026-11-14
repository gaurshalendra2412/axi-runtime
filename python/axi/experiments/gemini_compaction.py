"""The head-to-head test: the kept chat WITH COMPACTION against AXI (page: docs/COMPACTION_TEST.md, written and hash-locked BEFORE any call).

The long-session test (gemini_long.py, docs/GEMINI_LONG_TEST.md) put AXI against a kept chat that is never shortened. Nobody runs a long agent session that way: agent tools COMPACT the chat when it gets
long (the model is asked to summarise the conversation, and the summary replaces it). This test runs that rival, on the SAME job (`million_hard.HardStream`, seed 20261111), the SAME heavy turns (`gemini_long.ticket`)
and the SAME model (gemini-3.8-flash) as the long-session test, and reads it against the SAVED arm A of that test (AXI, 1,500 steps; its first 600 steps are used). Arm A is not run again.

Three compaction arms, 600 steps each, each reply applied as it comes (blind apply, no re-ask, no repair: what a firm gets from the chat):
  S  generic summary, when the last prompt reached 32,000 tokens   (what an agent tool does, early)
  L  generic summary, when the last prompt reached 128,000 tokens  (the same, later: fewer summaries, longer chats)
  V  a summary that is asked to write out the complete current service graph (the rival built by someone who knows the job), when the last prompt reached 32,000 tokens
When the last prompt reached the trigger, ONE extra call is made before the next turn: the same chat, a different system text (`COMPACT_SYSTEM`) and the request for the summary (`COMPACT_*`). The chat is then
emptied, and the summary is put at the top of the next turn's message. Every call, the summary calls included, is counted (tokens, cached tokens, seconds, first token), and every summary is kept in the file.

It imports `gemini_long.py`, `million.py`, `million_hard.py`, `api_llm.py`, `agent_loop.py`, `cascade.py` and the framework and edits none of them.
What this file does not do: it calls no model by itself (the notebook does), and it says nothing about summaries written by another prompt, about memory kept in tools or in a search index, or about another model."""
from __future__ import annotations
import json, os
from typing import List, Optional, Tuple

from axi.experiments import gemini_long as G
from axi.experiments import million as M
from axi.experiments.agent_loop import SYSTEM, copy_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.resumable import load_rows, _roundtrip

# ---------------------------------------------------------------- the settings written on the page
SEED, SEED_PILOT = G.SEED, G.SEED_PILOT                  # 20261111 (the long-session test's seed: the same requests and heavy turns), and 99 for the pilot
STEPS = 600
ARMS = {"S": dict(trigger=32_000, style="generic"), "L": dict(trigger=128_000, style="generic"), "V": dict(trigger=32_000, style="verbatim")}
PILOT_STEPS = 45
TOKEN_CAP = 60_000_000                                   # tokens sent per arm (prompt + visible + thinking, summary calls included): the guard behind the dollar budget
A_FILE_SHA256 = "7ba26d2946ec20ddd8a2085f18c6a2eee9ac3fb02025209158ce67a91faf29c6"      # docs/results/axi_gemini_long_session_seed20261111_default.json (the saved arm A)
SUMMARY_HEAD = "Summary of our conversation so far, written by you:\n"
COMPACT_SYSTEM = "You write summaries of conversations. The summary replaces the conversation: whoever reads it will have nothing else to go on."
COMPACT_GENERIC = ("Please write a summary of our conversation so far. It will replace the conversation: from now on you will see only your summary and then the next request. "
                   "Include everything you will need to carry on the work correctly.")
COMPACT_VERBATIM = ("Please write a summary of our conversation so far. It will replace the conversation: from now on you will see only your summary and then the next request. "
                    "Write out the complete current service graph exactly as it stands now, every service as [row,col:value] and every edge as (r,c)->(r,c):relation#weight, "
                    "and add any rule or convention you have noticed that you will need to carry on the work correctly.")
COMPACT_REQUEST = {"generic": COMPACT_GENERIC, "verbatim": COMPACT_VERBATIM}


# ---------------------------------------------------------------- the chat
def compaction_due(rows: list, trigger: int) -> bool:
    """A summary is asked for before the next turn when the prompt of the LAST turn on file reached `trigger` tokens. A function of the saved rows alone, so a replay can check every compaction."""
    return bool(rows) and (rows[-1].get("prompt_tokens") or 0) >= trigger


def compaction_messages(history: List[dict], style: str) -> List[dict]:
    """The call that makes the summary: the compaction system text, the chat as it stands, the request. The chat ends with an assistant reply, so the request is the last (user) message."""
    return [{"role": "system", "content": COMPACT_SYSTEM}] + history + [{"role": "user", "content": COMPACT_REQUEST[style]}]


def turn_text(seed: int, t: int, instruction: str, start_graph, pad_lines: int, carry: Optional[str]) -> str:
    """The user message of turn t: the heavy turn of the long-session test (the first turn carries the first state), and after a compaction the summary on top of it."""
    base = G.user_text(seed, t, instruction, start_graph, pad_lines)
    return base if not carry else SUMMARY_HEAD + carry + "\n\n" + base


def replay(rows: list, seed: int, trigger: int, style: str, pad_lines: int = G.PAD_LINES):
    """Run the stream and the blind policy over the saved replies and rebuild the chat, with the compactions found on file. Returns (stream, held state, chat history, summary waiting to be sent or None).
    Raises at the first flag that differs, at a compaction that was made when none was due (or not made when due), or at a compaction with no summary."""
    stream = M.make_stream(G.STREAM, seed); live = copy_graph(stream.start); history: List[dict] = []; carry = None
    for i, rec in enumerate(rows):
        due = compaction_due(rows[:i], trigger)
        if due != ("compaction" in rec): raise G.ReplayMismatch(f"step {rec['t']}: a compaction was {'made' if 'compaction' in rec else 'not made'} where one was {'not due' if 'compaction' in rec else 'due'}")
        if due:
            if not (rec["compaction"].get("text") or "").strip(): raise G.ReplayMismatch(f"step {rec['t']}: the compaction has no summary")
            carry = rec["compaction"]["text"]; history = []
        out = stream.next()
        if out is None: raise G.ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = stream.live
        if st["kind"] != rec["kind"]: raise G.ReplayMismatch(f"step {rec['t']}: request kind {st['kind']} against saved {rec['kind']}")
        d = strict_parse(rec["text"]); after = next_state("blind", live, d, st["instruction"]); f = G.kept_fields(after, oa, ob, d)
        diff = [k for k, v in f.items() if rec.get(k) != v]
        if diff: raise G.ReplayMismatch(f"step {rec['t']}: flags {diff} differ from the saved ones")
        history += [{"role": "user", "content": turn_text(seed, rec["t"], st["instruction"], stream.start if rec["t"] == 1 else None, pad_lines, carry)}, {"role": "assistant", "content": rec["text"]}]
        carry = None; live = after
    return stream, live, history, carry


def run_compact(llm, seed: int, steps: int, arm: str, path: str, meta=None, pad_lines: int = G.PAD_LINES, max_tokens: Optional[int] = None, progress=None, trigger: Optional[int] = None) -> dict:
    """One compaction arm (`ARMS`; `trigger` overrides the arm's trigger, for the pilot and the tests). One JSON line per finished turn after a header line; the same call again carries on from the lines it finds
    (the chat is rebuilt from the saved replies and summaries). A turn that came after a compaction carries it as `compaction` (text, tokens, thinking, prompt, cached tokens, seconds, first token). `stopped`: 'token_cap', 'stream' or None."""
    cfg = dict(ARMS[arm]); cfg["trigger"] = trigger if trigger is not None else cfg["trigger"]; style = cfg["style"]; trg = cfg["trigger"]
    meta = dict(meta or {}, arm=arm, seed=seed, steps=steps, pad_lines=pad_lines, stream=G.STREAM, policy="blind", trigger=trg, style=style)
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} turns, more than the {steps} asked for")
    stream, live, history, carry = replay(rows, seed, trg, style, pad_lines); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live; comp = None
            if compaction_due(rows, trg):
                text, ntok, sec = llm.generate(compaction_messages(history, style), constrained=False); comp = G._att(llm, text, ntok, sec)
                if not text.strip(): raise ValueError("the model returned an empty summary; nothing was saved for this turn, run the same cell again")
                carry = text; history = []
            user = {"role": "user", "content": turn_text(seed, t + 1, st["instruction"], stream.start if t == 0 else None, pad_lines, carry)}
            text, ntok, sec = llm.generate([{"role": "system", "content": SYSTEM}] + history + [user], constrained=True); a = G._att(llm, text, ntok, sec)
            d = strict_parse(text); after = next_state("blind", live, d, st["instruction"])
            rec = dict(t=t + 1, kind=st["kind"], **a, **G.kept_fields(after, oa, ob, d))
            if comp: rec["compaction"] = comp
            rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); history += [user, {"role": "assistant", "content": text}]; carry = None; live = after; spent += billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


# ---------------------------------------------------------------- counting: the summary call is part of the step that follows it
def as_long(r: dict) -> dict:
    """The row as `gemini_long` counts it: the summary call is one more call of the step (its tokens, cache and first token), and the step's seconds include it."""
    c = r.get("compaction")
    if not c: return r
    x = dict(r); x["reasks"] = list(r.get("reasks", [])) + [c]; x["seconds"] = round(r["seconds"] + c["seconds"], 4); return x


def billed(r: dict) -> int: return G.billed(as_long(r))
def tokens_sent(rows: list) -> int: return sum(billed(r) for r in rows)
def compactions(rows: list) -> List[dict]: return [r["compaction"] for r in rows if "compaction" in r]
def stats(rows: list) -> dict: return G.stats([as_long(r) for r in rows])
def summarize(rows: list) -> dict: return dict(all=stats(rows), compactions=len(compactions(rows)), summary_tokens=sum(c["tokens"] for c in compactions(rows)), first_inexact=next((r["t"] for r in rows if not r.get("reply_exact")), None),
                                                max_prompt=max((r.get("prompt_tokens") or 0 for r in rows), default=0), max_distance=max((r.get("distance", 0) for r in rows), default=0), refused=sum(1 for r in rows if not r["parsed"]))


def cost_usd(rows: list, price_in: Optional[float], price_out: Optional[float], price_cached: Optional[float] = None) -> Optional[float]:
    """Dollars for the calls that were really made (summary calls included) at the prices typed in (dollars per million tokens; thinking tokens are billed as output). Prompt tokens served from the cache are priced at
    `price_cached` when one is typed, else at the full input price. None if a price is missing."""
    if price_in is None or price_out is None: return None
    pc = price_cached if price_cached is not None else price_in; tot = 0.0
    for r in rows:
        for p in G.parts_of(as_long(r)):
            if not p["disk"]: tot += (p["prompt"] - p["cache"]) * price_in + p["cache"] * pc + (p["vis"] + p["think"]) * price_out
    return tot / 1e6


def arm_figures(rows: list, price_in=None, price_out=None, price_cached=None) -> dict:
    """The figures the page reads, for a set of steps of one arm (a compaction arm, or the first steps of the saved arm A, whose rows have no summary)."""
    n = len(rows); s = stats(rows); held = [r["exact"] for r in rows]; wrong = [r["t"] for r in rows if not r["exact"]]
    last_ok = max((i for i, x in enumerate(held) if x), default=-1)
    cost, cost0 = cost_usd(rows, price_in, price_out, price_cached), cost_usd(rows, price_in, price_out)
    ex = sum(1 for x in held if x)
    return dict(steps=n, held_exact=ex / n if n else None, held_exact_steps=ex, reply_exact=s["reply_exact"], final_exact=bool(held[-1]) if n else None, first_slip=next((r["t"] for r in rows if not r["reply_exact"]), None),
                first_held_wrong=wrong[0] if wrong else None, held_wrong_turns=len(wrong), lost_for_good=(rows[last_ok + 1]["t"] if last_ok + 1 < n else None), corrupted=s["corrupted"],
                max_distance=max((r["distance"] for r in rows), default=0), compactions=len(compactions(rows)), prompt_per_step=s["prompt_per_step"], seconds_per_step=s["seconds_per_step"], exact_per_hour=s["exact_per_hour"],
                cached_share=s["cached_share"], parsed=s["parsed"], sent=s["sent"], computed=s["computed"], cost=cost, cost_nodiscount=cost0,
                cost_per_step=None if cost is None or not n else cost / n, cost_per_exact=None if cost is None else (cost / ex if ex else float("inf")),
                cost_per_step_nodiscount=None if cost0 is None or not n else cost0 / n, cost_per_exact_nodiscount=None if cost0 is None else (cost0 / ex if ex else float("inf")))


# ---------------------------------------------------------------- the reading rules (fixed in the page before any call)
H_S = "arm S finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction"
H_L = "arm L finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction"
H_V = "arm V finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction"
H_A = "the saved arm A of the long-session test is the file whose SHA-256 is written in the test, and its first 600 steps replay without a mismatch"
H_SUM = "every compaction has a non-empty summary, and there were at least 10 compactions in arm S, at least 3 in arm L and at least 10 in arm V"
H_PARSED = "at least 98 percent of the replies of each of the three compaction arms were accepted by the strict parser"
H_EX_S = "arm S: the held state equals the oracle's on at least 98 percent of its 600 steps"
H_EX_L = "arm L: the held state equals the oracle's on at least 98 percent of its 600 steps"
H_EX_V = "arm V: the held state equals the oracle's on at least 98 percent of its 600 steps"
H_AXI_S = "AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm S's"
H_AXI_L = "AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm L's"
H_AXI_V = "AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm V's"
H_END_S = "arm S: the held state is not exact at step 600"
H_END_L = "arm L: the held state is not exact at step 600"
H_END_V = "arm V: the held state is not exact at step 600"
H_EARLY = "arm S: the first reply that is not exactly right comes before turn 100"
H_TIME = "the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A over its first 600 steps"
H_COST = "(read only if the three prices were typed) the cost of a step of arm A is lower than that of each of the arms S, L and V"
H_COST_EX = "(read only if the three prices were typed) the cost of an exact step of arm A is lower than that of each of the arms S, L and V"
LINES = [("guard", H_S), ("guard", H_L), ("guard", H_V), ("guard", H_A), ("instrument", H_SUM), ("instrument", H_PARSED), ("claim", H_EX_S), ("claim", H_EX_L), ("claim", H_EX_V), ("claim", H_AXI_S), ("claim", H_AXI_L),
         ("claim", H_AXI_V), ("claim", H_END_S), ("claim", H_END_L), ("claim", H_END_V), ("claim", H_EARLY), ("claim", H_TIME), ("claim", H_COST), ("claim", H_COST_EX)]
EXACT_MIN, AXI_MARGIN, PARSED_MIN, MIN_COMPACTIONS = 0.98, 0.05, 0.98, {"S": 10, "L": 3, "V": 10}


def verdict(arms: dict, a_rows: Optional[list], replays: Optional[dict] = None, a_file_ok: Optional[bool] = None, prices: Tuple = (None, None, None)) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `arms`: {'S': run dict, 'L': ..., 'V': ...} as `run_compact` returns (missing arms are not run); `a_rows`: the first 600 rows of the saved arm A; `replays`: {'S': bool, ...,
    'A': bool}, whether the saved replies replayed without a mismatch; `a_file_ok`: the saved arm A file has the hash written here; `prices`: (price_in, price_out, price_cached)."""
    replays = replays or {}; out: List[Optional[bool]] = []; pin, pout, pc = prices
    rows = {k: (v["rows"] if v else None) for k, v in arms.items()}; fig = {k: arm_figures(r, pin, pout, pc) for k, r in rows.items() if r}
    figs = arm_figures(a_rows, pin, pout, pc) if a_rows else None
    for k in "SLV": out.append(None if not rows.get(k) else bool(replays.get(k, True)) and len(rows[k]) >= STEPS)                                           # 1-3
    out.append(None if a_rows is None or a_file_ok is None else bool(a_file_ok) and bool(replays.get("A", True)) and len(a_rows) >= STEPS)                   # 4
    ok5 = [rows.get(k) and all((c.get("text") or "").strip() for c in compactions(rows[k])) and len(compactions(rows[k])) >= MIN_COMPACTIONS[k] for k in "SLV"]
    out.append(None if not all(rows.get(k) for k in "SLV") else all(ok5))                                                                                    # 5
    out.append(None if not all(rows.get(k) for k in "SLV") else all(fig[k]["parsed"] >= PARSED_MIN for k in "SLV"))                                           # 6
    for k in "SLV": out.append(None if k not in fig else fig[k]["held_exact"] >= EXACT_MIN)                                                                    # 7-9
    for k in "SLV": out.append(None if k not in fig or not figs else figs["held_exact"] - fig[k]["held_exact"] >= AXI_MARGIN - 1e-12)                          # 10-12
    for k in "SLV": out.append(None if k not in fig else not fig[k]["final_exact"])                                                                            # 13-15
    out.append(None if "S" not in fig else fig["S"]["first_slip"] is not None and fig["S"]["first_slip"] < 100)                                              # 16
    out.append(None if "S" not in fig or not figs or fig["S"]["seconds_per_step"] is None or figs["seconds_per_step"] is None else fig["S"]["seconds_per_step"] <= figs["seconds_per_step"] + 1e-12)   # 17
    have = pin is not None and pout is not None and pc is not None and figs and all(k in fig for k in "SLV")
    out.append(None if not have else all(figs["cost_per_step"] < fig[k]["cost_per_step"] for k in "SLV"))                                                      # 18
    out.append(None if not have else all(figs["cost_per_exact"] < fig[k]["cost_per_exact"] for k in "SLV"))                                                    # 19
    assert len(out) == len(LINES)
    return [(i + 1, LINES[i][0], LINES[i][1], out[i]) for i in range(len(LINES))]


def claims_allowed(checks) -> dict:
    ok = {i: c for i, _, _, c in checks}; return dict(base=all(ok[i] is True for i in (1, 2, 3, 4, 5, 6)), cost=bool(ok.get(18) is not None))


def headline(checks) -> str:
    ok = {i: c for i, _, _, c in checks}; need = (10, 11, 12)
    if any(ok[i] is None for i in need): return "NOT READ (a headline line was not run)"
    held = [k for k, i in zip("SLV", (7, 8, 9)) if ok[i]]
    if all(ok[i] for i in need): return "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM" + (f" (but arm {', '.join(held)} also reached 98 percent)" if held else "")
    return "MIXED: " + ", ".join(f"line {i} {'yes' if ok[i] else 'NO'}" for i in need)


# ---------------------------------------------------------------- the check before the run
def project(rows: list, steps: int, trigger: int, price_in: Optional[float], price_out: Optional[float], price_cached: Optional[float] = None) -> dict:
    """What a pilot says about a whole arm of `steps` turns at `trigger`: the prompt grows by the pilot's median growth a turn, restarts at the size seen after its compactions when the last prompt reaches the trigger, the summary call
    sends the chat as it stands. Cached share and output tokens a turn are the pilot's own; the first turn after a compaction and the summary call are priced with no cache. Cost with the cached price typed (if any) and with none."""
    ps = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]; grow = sorted(b - a for a, b in zip(ps, ps[1:]) if b > a)
    after = [r["prompt_tokens"] for r in rows if "compaction" in r and r.get("prompt_tokens")]
    if not grow or not rows: return dict(note="the pilot has too few turns to project")
    g = grow[len(grow) // 2]; restart = (sum(after) / len(after)) if after else ps[0]
    live = [r for r in rows if "compaction" not in r and (r.get("prompt_tokens") or 0) > 0]; share = (sum(r.get("cache_tokens") or 0 for r in live) / sum(r["prompt_tokens"] for r in live)) if live else 0.0
    out_per = sum((r.get("tokens") or 0) + (r.get("think") or 0) for r in rows) / len(rows); sumout = (sum(c["tokens"] + c.get("think", 0) for c in compactions(rows)) / len(compactions(rows))) if compactions(rows) else 0.0
    p = ps[0]; prompt = cached = 0.0; comps = 0; comp_prompt = 0.0; fresh = True                 # the first turn of the arm and the first turn after a summary start a chat the service has not seen: no cache
    for t in range(steps):
        if t > 0 and p >= trigger:
            comps += 1; comp_prompt += p + g; p = restart; fresh = True
        prompt += p; cached += 0.0 if fresh else p * share; fresh = False; p += g
    out_tok = steps * out_per + comps * sumout
    cost = lambda pc: None if price_in is None or price_out is None else ((prompt + comp_prompt - cached) * price_in + cached * (pc if pc is not None else price_in) + out_tok * price_out) / 1e6
    return dict(growth_per_turn=g, restart_prompt=restart, cached_share=share, compactions=comps, prompt_tokens=prompt + comp_prompt, output_tokens=out_tok, cost=cost(price_cached), cost_nodiscount=cost(None))


# ---------------------------------------------------------------- printing
def _f(x, k=1): return "-" if x is None else f"{x:.{k}f}"
def _pct(x): return "-" if x is None else f"{x * 100:.1f}%"


def print_arm(title: str, rows: list, prices=(None, None, None)):
    f = arm_figures(rows, *prices)
    print(f"{title}: {f['steps']} steps | held exact {_pct(f['held_exact'])} ({f['held_exact_steps']}) | replies exact {_pct(f['reply_exact'])} | first slip {f['first_slip']} | held wrong from {f['first_held_wrong']} on {f['held_wrong_turns']} turns | "
          f"never exact again from {f['lost_for_good']} | corrupted {f['corrupted']} | summaries {f['compactions']} | prompt/step {_f(f['prompt_per_step'], 0)} (cached {_pct(f['cached_share'])}) | {_f(f['seconds_per_step'], 2)} s/step | "
          f"cost/step {'-' if f['cost_per_step'] is None else '$%.4f' % f['cost_per_step']} (no discount {'-' if f['cost_per_step_nodiscount'] is None else '$%.4f' % f['cost_per_step_nodiscount']})")


def print_verdict(checks, title="COMPACTION HEAD-TO-HEAD (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks))
    print("  rule on the page:", "lines 1 to 6 read yes: the figures of the arms are claimed as read." if claims_allowed(checks)["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")

"""The long-session test on a hosted model (concept-map item 115, docs/GEMINI_LONG_TEST.md, test 22): Gemini 3.8 Flash, heavy turns, a chat that grows toward the model's 1,000,000-token window, alone and with AXI.

Two arms on the SAME request stream (the hard test's dense service graph, `million_hard.HardStream`) and the SAME heavy turns (every request comes with a block of operations-log noise, `ticket`):
  K  the kept chat, "alone": one chat, system text + the first state + every turn and every reply kept and re-sent; each reply is taken as it comes (blind apply, no re-ask, no repair). This is what a firm
     gets when it just keeps talking to the model. It stops when the next prompt would pass `K_CONTEXT_CAP` tokens (950,000) or at `K_MAX_STEPS` turns.
  A  "with AXI": the exact state held by the framework (`scope_m7`, up to two re-asks, the repair deleting up to 8 dependants) is shown at every step with the same heavy turn; nothing is kept in the chat. `A_STEPS` steps.
  D  the depth probe (a control): a chat whose history is the IDEAL replies for turns 1..s-1 (no errors in the history, no call for them) and then ten real turns, at several depths. If the model slips there, depth
     alone is the cause, not its own earlier mistakes.
Measured per call: prompt tokens, tokens served from the service's cache (`cached_content_token_count`), visible and thinking tokens (as the service counts them), seconds in all, and **seconds to the first token**
(the calls are streamed), so speed (steps and exact steps per hour of calls), latency (first token and total, by context size) and tokens (computed = not served from the cache + written) are measured, not estimated.
Energy: the API gives no joules for Google's hardware. The computed tokens and the seconds are the stand-in, and the page says so.
It imports `million.py`, `million_hard.py`, `api_llm.py` and the framework and edits none of them. The money is guarded by `ChatLLM` (a dollar budget at prices typed in by the owner, a cached-token price if typed) and a token cap per run."""
from __future__ import annotations
import hashlib, json, os, random, time
from typing import Dict, List, Optional, Tuple

from axi.engine.gate import apply, check
from axi.experiments import million as M
from axi.experiments import million_hard as H
from axi.experiments import api_llm as API
from axi.experiments.agent_loop import SYSTEM, copy_graph, same_graph, strict_parse, _serialize
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import distance
from axi.experiments.resumable import load_rows, _roundtrip

# ---------------------------------------------------------------- the settings written on the page
MODEL = H.HARD_MODEL                                    # "gemini-3.8-flash"
STREAM = "HardStream"
SEED, SEED_PILOT = 20261111, 99
PAD_LINES = 40                                          # lines of operations log in every turn (about 1,900 tokens; the pilot prints the real figure)
K_MAX_STEPS, K_CONTEXT_CAP = 700, 950_000
A_STEPS, A_REASKS, A_DEGREE = 1500, H.HARD_REASKS, H.HARD_DEGREE
D_STARTS, D_ASKS = (50, 150, 250, 350, 450), 10          # depth probe: first real turn of each cell, real turns asked in each
K_TOKEN_CAP, A_TOKEN_CAP, D_TOKEN_CAP = 400_000_000, 8_000_000, 60_000_000     # tokens sent (prompt + visible + thinking): the guard behind the dollar budget
MAX_OUT = 8192
BANDS = (0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_001)


# ---------------------------------------------------------------- the heavy turn
_SVC = ["gateway", "billing", "auth", "search", "cart", "mailer", "indexer", "scheduler", "uploader", "exporter", "ledger", "notifier"]
_MSG = ["handled request", "cache refresh done", "retry scheduled", "connection reused", "batch flushed", "token validated", "queue depth sampled", "heartbeat ok", "config reloaded", "job finished"]
_LVL = ["INFO", "INFO", "INFO", "WARN", "DEBUG"]


def ticket(seed: int, t: int, lines: int = PAD_LINES) -> str:
    """The operations-log block that comes with request number t: deterministic in (seed, t), the same text in every arm. It contains no coordinates, no service graph and no rule."""
    rng = random.Random(f"{seed}:{t}:pad"); base = 36000.0 + 61.0 * t
    out = [f"Shift log {rng.randrange(10 ** 6):06d} (routine output, nothing in it needs an action):"]
    for i in range(lines):
        out.append(f"{base + i * 1.37:.2f}s {rng.choice(_LVL)} {rng.choice(_SVC)}-{rng.randrange(1, 40)} pod={rng.getrandbits(32):08x} id={rng.getrandbits(48):012x} {rng.choice(_MSG)} in {rng.randrange(3, 900)}ms code={rng.choice([200, 200, 200, 204, 301, 404, 500])}")
    return "\n".join(out)


def user_text(seed: int, t: int, instruction: str, state_graph=None, lines: int = PAD_LINES) -> str:
    head = (_serialize(state_graph) + "\n\n") if state_graph is not None else ""
    return f"{head}{ticket(seed, t, lines)}\n\nRequest: {instruction}"


# ---------------------------------------------------------------- the API, with a chat, streaming and the cache count
class GeminiChatBackend(API.GeminiBackend):
    """`GeminiBackend` that also takes a chat (user / assistant messages), streams the reply to record the seconds to the first token, and records the tokens the service served from its cache.
    `client` and `types` can be passed in (tests do). The key is only handed to the SDK."""
    last_cache_tokens = 0; last_first_token = None

    def call(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system"); contents = []
        for m in messages:
            if m["role"] == "system": continue
            if m["role"] not in ("user", "assistant"): raise ValueError(f"unknown role {m['role']!r}")
            contents.append(self.types.Content(role="user" if m["role"] == "user" else "model", parts=[self.types.Part(text=m["content"])]))
        if not contents or contents[-1].role != "user": raise ValueError("the chat must end with a user message")
        kw = dict(system_instruction=system, max_output_tokens=self.max_output_tokens)
        if self.temperature is not None: kw["temperature"] = self.temperature
        if self.thinking is not None:
            kw["thinking_config"] = self.types.ThinkingConfig(thinking_level=self.thinking) if isinstance(self.thinking, str) else self.types.ThinkingConfig(thinking_budget=int(self.thinking))
        t0 = time.perf_counter(); first = None; parts = []; um = None; finish = None
        for ch in self.client.models.generate_content_stream(model=self.model, contents=contents, config=self.types.GenerateContentConfig(**kw)):
            try: tx = ch.text or ""
            except Exception: tx = ""
            if tx and first is None: first = time.perf_counter() - t0
            if tx: parts.append(tx)
            if getattr(ch, "usage_metadata", None) is not None: um = ch.usage_metadata
            cands = getattr(ch, "candidates", None) or []
            if cands and getattr(cands[0], "finish_reason", None) is not None: finish = cands[0].finish_reason
        g = lambda name: (getattr(um, name, None) or 0) if um is not None else 0
        self.last_first_token = first; self.last_cache_tokens = g("cached_content_token_count")
        return dict(text="".join(parts), tokens=g("candidates_token_count"), think_tokens=g("thoughts_token_count"), prompt_tokens=g("prompt_token_count") or None, finish=finish)


class ChatLLM(API.DiskCachedLLM):
    """`DiskCachedLLM` that also saves, for every answer, the cache tokens and the seconds to the first token, and prices the saved answers with a cached-token price when one was typed in."""
    def __init__(self, *a, price_cached=None, **kw):
        self.price_cached = price_cached; self.last_cache_tokens = 0; self.last_first_token = None
        super().__init__(*a, **kw)

    def _append(self, rec):
        if "text" in rec and not rec.get("fail"): rec["cache_tokens"] = getattr(self.backend, "last_cache_tokens", 0) or 0; rec["first_token"] = getattr(self.backend, "last_first_token", None)
        super()._append(rec)

    def spent_usd(self):
        if self.price_in is None or self.price_out is None: return None
        rs = list(self.cache.values()); pc = self.price_cached if self.price_cached is not None else self.price_in
        cached = sum(r.get("cache_tokens") or 0 for r in rs); prompt = sum(r.get("prompt_tokens") or 0 for r in rs)
        return ((prompt - cached) * self.price_in + cached * pc + sum((r.get("tokens") or 0) + (r.get("think") or 0) for r in rs) * self.price_out) / 1e6

    def generate(self, messages, constrained=True, max_new_tokens=400):
        out = super().generate(messages, constrained, max_new_tokens); rec = self.cache.get(self.key(messages)) or {}
        self.last_cache_tokens = rec.get("cache_tokens") or 0; self.last_first_token = rec.get("first_token"); self.last_finish = rec.get("finish"); return out


def build_chat_llm(api_key, cache_path, daily_limit=None, min_interval=0.3, on_write=None, client=None, types=None, thinking=None, say=print, **kw):
    """List the models the key can see, pick `gemini-3.8-flash`, and return (llm, model id). `kw`: budget_usd, price_in, price_out, price_cached."""
    if client is None:
        from google import genai
        client = genai.Client(api_key=api_key)
    model = API.resolve_model(API.list_model_names(client), MODEL); say(f"model id: {model}")
    backend = GeminiChatBackend(model=model, thinking=thinking, temperature=None, max_output_tokens=MAX_OUT, client=client, types=types)
    return ChatLLM(backend, cache_path, backend.config_dict(), daily_limit=daily_limit, min_interval=min_interval, on_write=on_write, say=say, **kw), model


# ---------------------------------------------------------------- what a step did
def reply_flags(ob, oa, d) -> dict:
    """What the REPLY did to the oracle's state before the step (`ob`): was it readable, can it be applied to the true state (False = it names something that is not there or leaves an edge dangling: a
    hallucinated reply), and does it give exactly the oracle's next state. Independent of what the arm does with the reply."""
    if d is None: return dict(reply_ok=False, reply_exact=False)
    ok = bool(check(ob, d).ok); ex = False
    if ok:
        g = copy_graph(ob); apply(g, d); ex = same_graph(g, oa)
    return dict(reply_ok=ok, reply_exact=ex)


def kept_fields(after, oa, ob, d) -> dict:
    return dict(parsed=d is not None, **reply_flags(ob, oa, d), corrupted=not after.is_well_formed(), distance=distance(after, oa), exact=same_graph(after, oa), nodes=len(after.nodes), edges=len(after.edges),
                oracle_nodes=len(oa.nodes), oracle_edges=len(oa.edges))


def _att(llm, text, ntok, sec) -> dict:
    return dict(text=text, tokens=int(ntok), think=int(getattr(llm, "last_think_tokens", 0) or 0), prompt_tokens=getattr(llm, "last_prompt_tokens", None), cache_tokens=int(getattr(llm, "last_cache_tokens", 0) or 0),
                first_token=getattr(llm, "last_first_token", None), seconds=round(float(sec), 4), from_disk=bool(getattr(llm, "from_disk", False)), finish=str(getattr(llm, "last_finish", None)))


def parts_of(r: dict) -> List[dict]:
    """The calls of one step, first ask first (a step of the arm A may have re-asks)."""
    one = lambda x: dict(prompt=x.get("prompt_tokens") or 0, vis=x.get("tokens") or 0, think=x.get("think") or 0, cache=x.get("cache_tokens") or 0, seconds=x.get("seconds") or 0.0, first=x.get("first_token"), disk=bool(x.get("from_disk")))
    return [one(r)] + [one(x) for x in r.get("reasks", [])]


def billed(r: dict) -> int: return sum(0 if p["disk"] else p["prompt"] + p["vis"] + p["think"] for p in parts_of(r))
def computed(r: dict) -> int: return sum(0 if p["disk"] else p["prompt"] - p["cache"] + p["vis"] + p["think"] for p in parts_of(r))
def tokens_sent(rows: list) -> int: return sum(billed(r) for r in rows)


# ---------------------------------------------------------------- replay (no call): every saved flag of every step must be reproduced
class ReplayMismatch(M.ReplayMismatch): pass


def replay(rows: list, seed: int, arm: str, pad_lines: int = PAD_LINES):
    """Run the stream and the arm's policy over the saved replies. Returns (stream, held state after the last saved step, history of the chat (arm K), prompt of each saved step). Raises at the first flag that differs."""
    stream = M.make_stream(STREAM, seed); live = copy_graph(stream.start); history: List[dict] = []
    for rec in rows:
        out = stream.next()
        if out is None: raise ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = stream.live
        if st["kind"] != rec["kind"]: raise ReplayMismatch(f"step {rec['t']}: request kind {st['kind']} against saved {rec['kind']}")
        d = strict_parse(rec["text"])
        if arm == "K":
            after = next_state("blind", live, d, st["instruction"]); f = kept_fields(after, oa, ob, d)
            history += [{"role": "user", "content": user_text(seed, rec["t"], st["instruction"], stream.start if rec["t"] == 1 else None, pad_lines)}, {"role": "assistant", "content": rec["text"]}]
        else:
            after, f = M.step_fields("scope_m7", live, st, ob, oa, d, A_DEGREE); f.update(reply_flags(ob, oa, d))
        diff = [k for k, v in f.items() if rec.get(k) != v] + (M._ask_problems(rec, A_REASKS) if arm == "A" else [])
        if diff: raise ReplayMismatch(f"step {rec['t']}: flags {diff} differ from the saved ones")
        live = after
    return stream, live, history


# ---------------------------------------------------------------- the runs
def _system() -> dict: return {"role": "system", "content": SYSTEM}


def projected_prompt(rows: list) -> Optional[int]:
    """The prompt of the next turn of the kept chat, projected from the last two saved prompts (None before the second turn)."""
    p = [r["prompt_tokens"] for r in rows[-2:] if r.get("prompt_tokens")]
    return None if len(p) < 2 else p[-1] + (p[-1] - p[-2])


def run_kept(llm, seed: int, steps: int, path: str, meta=None, pad_lines: int = PAD_LINES, context_cap: int = K_CONTEXT_CAP, max_tokens: Optional[int] = None, progress=None) -> dict:
    """Arm K. One JSON line per finished turn after a header line; the same call again carries on from the lines it finds (the chat is rebuilt from the saved replies). `stopped`: 'context' (the next prompt would
    pass `context_cap`), 'token_cap', 'stream' or None."""
    meta = dict(meta or {}, arm="K", seed=seed, steps=steps, pad_lines=pad_lines, context_cap=context_cap, stream=STREAM, policy="blind")
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} turns, more than the {steps} asked for")
    stream, live, history = replay(rows, seed, "K", pad_lines); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            nxt = projected_prompt(rows)
            if nxt is not None and nxt > context_cap: stopped = "context"; break
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live; user = {"role": "user", "content": user_text(seed, t + 1, st["instruction"], stream.start if t == 0 else None, pad_lines)}
            text, ntok, sec = llm.generate([_system()] + history + [user], constrained=True); a = _att(llm, text, ntok, sec)
            d = strict_parse(text); after = next_state("blind", live, d, st["instruction"])
            rec = _roundtrip(dict(t=t + 1, kind=st["kind"], **a, **kept_fields(after, oa, ob, d)))
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); history += [user, {"role": "assistant", "content": text}]; live = after; spent += billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


def run_axi(llm, seed: int, steps: int, path: str, meta=None, pad_lines: int = PAD_LINES, max_tokens: Optional[int] = None, progress=None, scripted: int = 0) -> dict:
    """Arm A: `scope_m7`, the exact state shown at every step with the same heavy turn, up to two re-asks, the repair deleting up to 8 dependants. `scripted`: the first steps are answered by the oracle (no call)."""
    meta = dict(meta or {}, arm="A", seed=seed, steps=steps, pad_lines=pad_lines, stream=STREAM, policy="scope_m7", reasks=A_REASKS, degree_threshold=A_DEGREE)
    if scripted: meta["scripted"] = scripted
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} steps, more than the {steps} asked for")
    stream, live, _ = replay(rows, seed, "A", pad_lines); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live
            if t < scripted: text, secs = st["ideal"], 0.0; first = dict(text=text, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, seconds=0.0, from_disk=True, finish="scripted"); extra = []
            else:
                msgs = [_system(), {"role": "user", "content": user_text(seed, t + 1, st["instruction"], live, pad_lines)}]
                text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = _att(llm, text, ntok, sec); extra = []
                while strict_parse(text) is None and len(extra) < A_REASKS:
                    text, ntok, sec = llm.generate(M.reask_messages(msgs, len(extra) + 1), constrained=True); secs += float(sec); extra.append(_att(llm, text, ntok, sec))
            d = strict_parse(text); after, fl = M.step_fields("scope_m7", live, st, ob, oa, d, A_DEGREE); fl.update(reply_flags(ob, oa, d))
            top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
            rec = dict(t=t + 1, kind=st["kind"], text=text, seconds=round(secs, 4), **top, **fl)
            if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
            rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); live = after; spent += billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


def run_depth(llm, seed: int, starts, asks: int, path: str, meta=None, pad_lines: int = PAD_LINES, max_tokens: Optional[int] = None, progress=None) -> dict:
    """The depth probe. For each start s: a chat whose history is turns 1..s-1 with the IDEAL replies (no call), then `asks` real turns s..s+asks-1, each followed by the ideal reply (so the history stays perfect).
    One JSON line per real turn, with `start`; the same call again carries on."""
    meta = dict(meta or {}, arm="D", seed=seed, starts=list(starts), asks=asks, pad_lines=pad_lines, stream=STREAM)
    rows = load_rows(path, meta); done = {(r["start"], r["t"]) for r in rows}; stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for s in starts:
            stream = M.make_stream(STREAM, seed); history: List[dict] = []
            for t in range(1, s + asks):
                out = stream.next()
                if out is None: stopped = "stream"; break
                st, ob = out; oa = stream.live; user = {"role": "user", "content": user_text(seed, t, st["instruction"], stream.start if t == 1 else None, pad_lines)}
                if t >= s and (s, t) not in done:
                    if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                    text, ntok, sec = llm.generate([_system()] + history + [user], constrained=True); a = _att(llm, text, ntok, sec); d = strict_parse(text)
                    rec = _roundtrip(dict(start=s, t=t, kind=st["kind"], ideal=st["ideal"], **a, parsed=d is not None, **reply_flags(ob, oa, d)))
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); done.add((s, t)); spent += billed(rec)
                    if progress: progress(s, t, rec)
                history += [user, {"role": "assistant", "content": st["ideal"]}]
            if stopped: break
    return dict(rows=rows, summary=summarize_depth(rows), meta=_roundtrip(meta), stopped=stopped)


def replay_depth(rows: list, seed: int) -> None:
    """The flags of a depth-probe row are a function of the saved reply and the stream."""
    by = {}
    for r in rows: by.setdefault(r["start"], []).append(r)
    for s, rs in by.items():
        stream = M.make_stream(STREAM, seed); need = {r["t"]: r for r in rs}
        for t in range(1, max(need) + 1):
            out = stream.next(); st, ob = out; oa = stream.live
            if t in need:
                r = need[t]; d = strict_parse(r["text"]); f = dict(parsed=d is not None, **reply_flags(ob, oa, d))
                if st["kind"] != r["kind"] or st["ideal"] != r["ideal"] or any(r.get(k) != v for k, v in f.items()): raise ReplayMismatch(f"depth probe start {s} turn {t}: differs from the saved flags")


# ---------------------------------------------------------------- summaries
def _mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else None


def _pct(xs, p):
    xs = sorted(xs)
    return None if not xs else xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def stats(rows: list) -> dict:
    """Speed, latency, tokens and accuracy of a set of steps. Only calls that were really made count for tokens and seconds."""
    n = len(rows); live = [p for r in rows for p in parts_of(r) if not p["disk"]]; secs = sum(r["seconds"] for r in rows if not all(p["disk"] for p in parts_of(r)))
    nl = sum(1 for r in rows if not all(p["disk"] for p in parts_of(r))); ex = sum(1 for r in rows if r.get("reply_exact")); ft = [p["first"] for p in live if p["first"] is not None]; ts = [r["seconds"] for r in rows if not all(p["disk"] for p in parts_of(r))]
    prompt = sum(p["prompt"] for p in live); cache = sum(p["cache"] for p in live); vis = sum(p["vis"] for p in live); th = sum(p["think"] for p in live)
    return dict(steps=n, live_steps=nl, reply_exact=ex / n if n else None, reply_ok=_mean(r["reply_ok"] for r in rows), parsed=_mean(r["parsed"] for r in rows), held_exact=_mean(r["exact"] for r in rows) if rows and "exact" in rows[0] else None,
                corrupted=sum(1 for r in rows if r.get("corrupted")), mean_distance=_mean(r["distance"] for r in rows) if rows and "distance" in rows[0] else None,
                prompt=prompt, cache=cache, vis=vis, think=th, sent=prompt + vis + th, computed=prompt - cache + vis + th, cached_share=(cache / prompt if prompt else None),
                prompt_per_step=(prompt / nl if nl else None), think_per_step=(th / nl if nl else None), seconds=secs, seconds_per_step=(secs / nl if nl else None),
                steps_per_hour=(nl / secs * 3600 if secs else None), exact_per_hour=(sum(1 for r in rows if r.get("reply_exact") and not all(p["disk"] for p in parts_of(r))) / secs * 3600 if secs else None),
                first_p50=_pct(ft, 0.5), first_p95=_pct(ft, 0.95), first_mean=_mean(ft), total_p50=_pct(ts, 0.5), total_p95=_pct(ts, 0.95), total_max=max(ts) if ts else None,
                computed_per_exact=_per_exact(prompt - cache + vis + th, sum(1 for r in rows if r.get("reply_exact") and not all(p["disk"] for p in parts_of(r))), nl))


def _per_exact(computed, exact, live_steps):
    """Computed tokens for each exact step; no live step: None; live steps but no exact one: infinity (the tokens bought nothing exact)."""
    if not live_steps: return None
    return computed / exact if exact else float("inf")


def by_prompt_band(rows: list, edges=BANDS) -> List[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in rows if lo <= (r.get("prompt_tokens") or 0) < hi]
        if b: out.append(dict(lo=lo, hi=hi, first=b[0]["t"], last=b[-1]["t"], **stats(b)))
    return out


def tenths(rows: list, k: int = 10) -> List[dict]:
    n = len(rows); return [dict(first=b[0]["t"], last=b[-1]["t"], **stats(b)) for i in range(k) for b in [rows[i * n // k:(i + 1) * n // k]] if b]


def summarize(rows: list) -> dict:
    return dict(all=stats(rows), first50=stats(rows[:50]), last100=stats(rows[-100:]), first_inexact=next((r["t"] for r in rows if not r.get("reply_exact")), None), max_prompt=max((r.get("prompt_tokens") or 0 for r in rows), default=0),
                max_distance=max((r.get("distance", 0) for r in rows), default=0), reasked=sum(1 for r in rows if r.get("asks", 1) > 1), refused=sum(1 for r in rows if not r["parsed"]))


def summarize_depth(rows: list) -> dict:
    by = {}
    for r in rows: by.setdefault(r["start"], []).append(r)
    return {s: dict(mean_prompt=_mean(r["prompt_tokens"] or 0 for r in rs), **stats(rs)) for s, rs in sorted(by.items())}


def window(rows: list, lo_prompt: int) -> list: return [r for r in rows if (r.get("prompt_tokens") or 0) >= lo_prompt]


# ---------------------------------------------------------------- the reading rules (fixed in the page before any call)
G_FINISH_K = "the kept chat (arm K) ended either at 700 turns or because its next prompt would pass 950,000 tokens, with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn"
G_FINISH_A = "the arm with AXI (arm A) reached step 1,500 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask"
G_DEEP = "the kept chat grew to a prompt of at least 500,000 tokens"
G_CACHE = "the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn"
G_REFUSED = "at most 0.5 percent of the steps of arm A ended with a reply the strict parser refused (after the re-asks)"
G_PARSED = "at least 98 percent of the replies of arm K were accepted by the strict parser"
G_A_EXACT = "arm A: the held state equals the oracle's on at least 98 percent of all steps and on at least 95 percent in every tenth"
G_A_CORRUPT = "arm A: the held state was never corrupted and never more than 4 items from the oracle's"
G_A_FLAT = "arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the first tenth and the mean seconds a step at most 1.5 times"
G_K_EARLY = "arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the oracle's next state"
G_K_DRIFT = "arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50"
G_K_HALLU = "arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true state (they name something that is not there or leave an edge dangling)"
G_K_CORRUPT = "arm K: the state a firm would hold by applying the replies as they come was corrupted (an edge pointing at a missing service) at some turn"
G_K_FIRST = "arm K: the first reply that is not exactly right comes before turn 50"
G_LATENCY = "latency: the mean seconds of a call of arm K at a prompt of 500,000 tokens or more are at least 3 times the mean seconds of a step of arm A"
G_FIRST_TOKEN = "latency to the first token: the median seconds to the first token of arm K at a prompt of 500,000 tokens or more are at least 3 times those of arm A"
G_SPEED = "speed: the exact steps per hour of calls of arm A are at least 2 times those of arm K, over the same number of steps (the first as many steps of A as K has turns)"
G_TOKENS = "tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 500,000 tokens or more"
G_THINK = "arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50"
G_DEPTH = "depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest"
G_HEAD = "headline: lines 7, 8 and 11 all read yes (with AXI the state holds; the kept chat drifts)"
G_LINES = [("guard", G_FINISH_K), ("guard", G_FINISH_A), ("guard", G_DEEP), ("instrument", G_CACHE), ("instrument", G_REFUSED), ("instrument", G_PARSED), ("claim", G_A_EXACT), ("claim", G_A_CORRUPT), ("claim", G_A_FLAT),
           ("claim", G_K_EARLY), ("claim", G_K_DRIFT), ("claim", G_K_HALLU), ("claim", G_K_CORRUPT), ("claim", G_K_FIRST), ("claim", G_LATENCY), ("claim", G_FIRST_TOKEN), ("claim", G_SPEED), ("claim", G_TOKENS), ("claim", G_THINK), ("claim", G_DEPTH)]


def _gt(x, y): return None if x is None or y is None else x >= y - 1e-12
def _le(x, y): return None if x is None or y is None else x <= y + 1e-12


def verdict(k: Optional[dict], a: Optional[dict], d: Optional[dict] = None, replays: Optional[dict] = None) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `k`, `a`, `d`: the dicts `run_kept`, `run_axi`, `run_depth` return (or None). `replays`: {'K': bool, 'A': bool}, whether the saved replies were replayed without a mismatch."""
    replays = replays or {}; out: List[Optional[bool]] = []; kr = k["rows"] if k else None; ar = a["rows"] if a else None
    out.append(None if not k else bool(replays.get("K", True)) and (len(kr) >= K_MAX_STEPS or k["stopped"] == "context"))                                         # 1
    out.append(None if not a else bool(replays.get("A", True)) and len(ar) >= A_STEPS)                                                                              # 2
    out.append(None if not k else max((r.get("prompt_tokens") or 0 for r in kr), default=0) >= 500_000)                                                             # 3
    out.append(None if not k else (lambda xs: bool(xs) and sum(1 for r in xs if (r.get("cache_tokens") or 0) > 0) / len(xs) >= 0.8)(kr[9:]) if kr and len(kr) > 10 else None)       # 4
    out.append(None if not a else sum(1 for r in ar if not r["parsed"]) <= 0.005 * len(ar))                                                                          # 5
    out.append(None if not k else sum(1 for r in kr if r["parsed"]) >= 0.98 * len(kr))                                                                              # 6
    out.append(None if not a else (lambda ts: sum(1 for r in ar if r["exact"]) >= 0.98 * len(ar) and all(t["held_exact"] >= 0.95 for t in ts))(tenths(ar)))         # 7
    out.append(None if not a else all(not r["corrupted"] for r in ar) and max(r["distance"] for r in ar) <= 4)                                                      # 8
    out.append(None if not a else (lambda ts: ts[-1]["prompt_per_step"] <= 1.3 * ts[0]["prompt_per_step"] and ts[-1]["seconds_per_step"] <= 1.5 * ts[0]["seconds_per_step"])(tenths(ar)) if ar and len(ar) >= 10 else None)   # 9
    s50 = stats(kr[:50]) if kr else None; l100 = stats(kr[-100:]) if kr else None
    out.append(None if not k else _gt(s50["reply_exact"], 0.9))                                                                                                     # 10
    out.append(None if not k or len(kr) < 150 else _gt(s50["reply_exact"] - l100["reply_exact"], 0.10))                                                             # 11
    out.append(None if not k or len(kr) < 150 else _gt(1 - l100["reply_ok"], 0.05))                                                                                 # 12
    out.append(None if not k else any(r["corrupted"] for r in kr))                                                                                                  # 13
    fi = next((r["t"] for r in kr if not r["reply_exact"]), None) if kr else None
    out.append(None if not k else (fi is not None and fi < 50))                                                                                                     # 14
    deep = window(kr, 500_000) if kr else []; sd = stats(deep) if deep else None; sa = stats(ar) if ar else None
    out.append(None if not deep or not a else _gt(sd["seconds_per_step"], 3 * sa["seconds_per_step"]))                                                              # 15
    out.append(None if not deep or not a or sd["first_p50"] is None or sa["first_p50"] is None else _gt(sd["first_p50"], 3 * sa["first_p50"]))                      # 16
    if k and a and len(ar) >= len(kr):
        sk, sa2 = stats(kr), stats(ar[:len(kr)]); out.append(_gt(sa2["exact_per_hour"], 2 * sk["exact_per_hour"]) if sk["exact_per_hour"] and sa2["exact_per_hour"] else None)                  # 17
    else: out.append(None)
    out.append(None if not deep or not a or sd["computed_per_exact"] is None or sa["computed_per_exact"] is None else _gt(sd["computed_per_exact"], 1.5 * sa["computed_per_exact"]))                          # 18
    out.append(None if not k or len(kr) < 150 else _gt(l100["think_per_step"], 2 * s50["think_per_step"]))                                                          # 19
    if d and d["rows"]:
        sm = summarize_depth(d["rows"]); ks = sorted(sm)
        out.append(None if len(ks) < 2 else _gt(sm[ks[0]]["reply_exact"] - sm[ks[-1]]["reply_exact"], 0.10))                                                         # 20
    else: out.append(None)
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    """Lines 1, 2, 3, 4, 5 and 6 read yes: the figures are claimed as read. Line 4 NO: no computed-token (energy stand-in) figure is claimed, only prompt tokens. A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (1, 2, 3, 5, 6))
    return dict(base=base, tokens_computed=bool(base and ok[4] is True))


def headline(checks) -> str:
    ok = {i: c for i, _, _, c in checks}; need = (7, 8, 11)
    if any(ok[i] is None for i in need): return "NOT READ (a headline line was not run)"
    return "AXI HOLDS AND THE KEPT CHAT DRIFTS" if all(ok[i] for i in need) else "MIXED: " + ", ".join(f"line {i} {'yes' if ok[i] else 'NO'}" for i in need)


# ---------------------------------------------------------------- the check before the run, and the printout
def estimate_cost(tokens: dict, price_in: Optional[float], price_out: Optional[float], price_cached: Optional[float] = None) -> Optional[float]:
    if price_in is None or price_out is None: return None
    pc = price_cached if price_cached is not None else price_in
    return ((tokens["prompt"] - tokens["cache"]) * price_in + tokens["cache"] * pc + (tokens["vis"] + tokens["think"]) * price_out) / 1e6


def pilot_report(k_rows: list, a_rows: list, price_in=None, price_out=None, price_cached=None, cached_share_assumed: float = 0.9) -> dict:
    """What 20 real turns of each arm say about the whole run: tokens per turn (padding included), thinking tokens, seconds, how many turns fit in the window, the total prompt tokens of arm K (which grows with
    the square of the turns), and the projected cost of the two arms at the prices typed in (arm K with the cache share measured in the pilot, and with none)."""
    sk, sa = stats(k_rows), stats(a_rows); grow = (k_rows[-1]["prompt_tokens"] - k_rows[0]["prompt_tokens"]) / max(1, len(k_rows) - 1) if len(k_rows) > 1 else None
    base = k_rows[0]["prompt_tokens"] if k_rows else None
    if grow:
        n_k = min(K_MAX_STEPS, int((K_CONTEXT_CAP - (base or 0)) / grow) + 1); k_prompt = sum((base or 0) + grow * i for i in range(n_k)); share = sk["cached_share"] if sk["cached_share"] is not None else cached_share_assumed
        k_tok = dict(prompt=k_prompt, cache=k_prompt * share, vis=n_k * sk["vis"] / max(1, sk["live_steps"]), think=n_k * sk["think"] / max(1, sk["live_steps"]))
        k_tok_nocache = dict(k_tok, cache=0)
    else: n_k = k_tok = k_tok_nocache = None
    a_tok = dict(prompt=sa["prompt"] / max(1, sa["live_steps"]) * A_STEPS, cache=0, vis=sa["vis"] / max(1, sa["live_steps"]) * A_STEPS, think=sa["think"] / max(1, sa["live_steps"]) * A_STEPS)
    return dict(k_growth_per_turn=grow, k_first_prompt=base, k_turns_that_fit=n_k, k_total_prompt_tokens=(k_tok or {}).get("prompt"), k_think_per_turn=sk["think_per_step"], k_seconds_per_turn=sk["seconds_per_step"], k_first_token=sk["first_mean"],
                a_prompt_per_step=sa["prompt_per_step"], a_think_per_step=sa["think_per_step"], a_seconds_per_step=sa["seconds_per_step"], a_total_tokens=a_tok["prompt"] + a_tok["vis"] + a_tok["think"],
                a_hours=(sa["seconds_per_step"] or 0) * A_STEPS / 3600, cost_a=estimate_cost(a_tok, price_in, price_out), cost_k=estimate_cost(k_tok, price_in, price_out, price_cached) if k_tok else None,
                cost_k_nocache=estimate_cost(k_tok_nocache, price_in, price_out) if k_tok_nocache else None)


def _f(x, k=1): return "-" if x is None else f"{x:.{k}f}"


def print_stats(title: str, rows: list, bands=None):
    s = stats(rows)
    print(f"{title}: {s['steps']} steps | reply exact {_f(None if s['reply_exact'] is None else s['reply_exact'] * 100)}% | reply applicable to the true state {_f(None if s['reply_ok'] is None else s['reply_ok'] * 100)}% | held state exact "
          f"{_f(None if s['held_exact'] is None else s['held_exact'] * 100)}% | corrupted {s['corrupted']} | {_f(s['seconds_per_step'], 2)} s a step | first token p50 {_f(s['first_p50'], 2)} s p95 {_f(s['first_p95'], 2)} s | "
          f"{_f(s['steps_per_hour'], 0)} steps/h, {_f(s['exact_per_hour'], 0)} exact steps/h | prompt {s['prompt']:,} (cached {_f(None if s['cached_share'] is None else s['cached_share'] * 100)}%) + reply {s['vis']:,} + thinking {s['think']:,}")


def print_bands(rows: list):
    print(f"{'prompt band':>19} {'turns':>11} {'n':>4} | {'exact':>6} {'ok':>6} {'held':>6} | {'s/call':>6} {'first p50':>9} {'p95':>6} | {'steps/h':>7} {'exact/h':>7} | {'think':>6} {'computed/exact':>14}")
    for b in by_prompt_band(rows):
        print(f"{b['lo']:>9,}-{min(b['hi'], 1_000_000):>9,} {b['first']:>5}-{b['last']:<5} {b['steps']:>4} | {_f(b['reply_exact'] * 100):>5}% {_f(b['reply_ok'] * 100):>5}% {_f(None if b['held_exact'] is None else b['held_exact'] * 100):>5}% | {_f(b['seconds_per_step'], 2):>6} {_f(b['first_p50'], 2):>9} "
              f"{_f(b['first_p95'], 2):>6} | {_f(b['steps_per_hour'], 0):>7} {_f(b['exact_per_hour'], 0):>7} | {_f(b['think_per_step'], 0):>6} {_f(b['computed_per_exact'], 0):>14}")


def print_verdict(checks, title="LONG-SESSION TEST (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks))
    a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2, 3, 5 and 6 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")

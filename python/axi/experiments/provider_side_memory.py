"""Test 44 (docs/MEMORY_ARMS_TEST.md): a chat that has to remember the state (kept, summarised, windowed, retrieved) against the exact-state arm, on a 14B model, jobs of 600 steps.

A NEW file: it imports the harness of Tests 35 to 43 (provider_side.py, provider_side_typed.py, provider_side_long.py, provider_side_big.py, agent_loop.py, cascade.py, long_horizon.py) and edits none of them.

The chat arms are LEAN chats: the first user message shows the whole state and the first request; every later user message is only the request ("Request: ..."). The model's own replies stay in the
chat, so the model has to carry the state in its memory: the state is never shown again, except through what a summary keeps. Replies are asked under the same grammar as the exact-state arm
(`named`: the kind of the request and the coordinates it names; the model still writes the numbers and, for a delete, the edges). A step of a chat arm has these flags:
  step_ok       the reply, applied with no check to the ORACLE's state before the step, gives the oracle's state after it (the step itself was right; earlier mistakes do not count)
  exact_blind   the state the host holds if it applies every reply as it comes, with no check, equals the oracle's after the step (what the chat alone gives)
  exact_policy  the same, with the host's gate and repair (`repair_m6`, as the exact-state arm)
Arms:  A  exact state each step (`provider_side_typed.TypedSession`, unchanged)
       K  the lean chat kept whole until its next prompt would pass LIMIT tokens
       S  K, and when the next prompt would pass BUDGET tokens the model is asked for a generic summary, the chat is emptied and the summary leads the next message
       V  K, and the model is asked to write out the whole current state (and any rule it noticed) instead
       W  sliding window: the first turn (which shows the state) is pinned, then only the last WINDOW_TURNS turns
       R  retrieval: the first turn pinned, the R_TOP earlier turns that share most coordinates with the request, and the last R_LAST turns
The summary call reuses the chat as its prefix (same system text, same history, one more user message), so it gets the prefix cache like any other call.
Test doubles (`TypedStandIn`) test the harness only. Never a result."""
import hashlib, json, re, time
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
from axi.experiments.agent_loop import SYSTEM, copy_graph, same_graph, strict_parse, _serialize
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script

SEED = PS.SEED
MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
JOBS = (192, 194, 196)
STEPS = 600
SHAPE = "named"
BUDGET = 4000            # a summary is asked for when the next prompt would pass this many tokens (arms S and V)
LIMIT = 16000            # the kept chat stops before its next prompt would pass this many tokens (arm K)
WINDOW_TURNS = 12        # arm W keeps the first turn and the last 12 turns
R_LAST, R_TOP = 4, 8     # arm R keeps the first turn, the last 4 turns and the 8 best-matching earlier turns
MAX_SUMMARY = 1000       # the longest summary, in tokens
ARMS = ("A", "K", "S", "V", "W", "R")
ARM_TEXT = {"A": "exact state each step", "K": "lean chat kept whole", "S": "generic summary", "V": "state written out", "W": "sliding window", "R": "retrieval"}

COMPACT_GENERIC = ("Stop the delta format for this one message. Summarise our conversation so far. Your summary will replace the conversation, so include everything you will need to carry on the work correctly.")
COMPACT_VERBATIM = ("Stop the delta format for this one message. Write out the complete current list of services and the complete current list of edges exactly as they stand now, after every change made in "
                    "this conversation, in the same format as the state shown at the start (a line NODES: ... and a line EDGES: ...). Then add any rule you have noticed. Your answer will replace the conversation.")
LEAD = "Summary of our conversation so far:\n"


# ---------------------------------------------------------------- small tools
def count_tokens(engine, messages: List[dict]) -> int:
    """The prompt length the engine will see for these messages (the real tokenizer on a real engine; the test double's own count otherwise)."""
    tok = getattr(engine, "tok", None)
    if tok is not None:
        return len(tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False))
    return len(engine._render(messages))


def coords(text: str) -> set:
    return set(re.findall(r"(\d+),(\d+)", text))


_NODE = re.compile(r"\[(\d+),(\d+):(-?\d+)\]")
_EDGE = re.compile(r"\((\d+),(\d+)\)->\((\d+),(\d+)\):([A-Za-z_]+)#(-?\d+)")


def parse_state_text(text: str) -> Optional[Tuple[dict, dict]]:
    """The NODES and EDGES lines of a state written out (the format of the state shown to the model). None if there is no NODES line or no EDGES line."""
    n = re.search(r"NODES:(.*)", text); e = re.search(r"EDGES:(.*)", text)
    if n is None or e is None: return None
    nodes = {(int(a), int(b)): int(v) for a, b, v in _NODE.findall(n.group(1))}
    edges = {((int(a), int(b)), (int(c), int(d)), r): int(w) for a, b, c, d, r, w in _EDGE.findall(e.group(1))}
    return nodes, edges


def state_fidelity(text: str, graph) -> Optional[bool]:
    """True if the summary contains a state written out that equals the graph exactly; False if it contains one that differs; None if it contains none."""
    p = parse_state_text(text)
    if p is None: return None
    return p[0] == graph.nodes and p[1] == graph.edges


def flat(turns: List[Tuple[str, str]]) -> List[dict]:
    out: List[dict] = []
    for u, a in turns: out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def free_generate(engine, messages_list: List[List[dict]], standins: List[str], max_tokens: int = MAX_SUMMARY):
    """One round of calls with no grammar (the summary calls). A real engine: its own `llm`. A test double: its `generate_round`, with the text it is told to write."""
    if hasattr(engine, "llm"):
        from vllm import SamplingParams
        prompts = [engine.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        t0 = time.perf_counter(); outs = engine.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False); dt = time.perf_counter() - t0
        return [PS.Reply(o.outputs[0].text, len(o.prompt_token_ids), getattr(o, "num_cached_tokens", None), len(o.outputs[0].token_ids)) for o in outs], dt
    return engine.generate_round([PS.Request(m, s) for m, s in zip(messages_list, standins)])


# ---------------------------------------------------------------- the chat arms
class MemSession:
    def __init__(self, arm: str, seed: int, steps: int = STEPS, shape: str = SHAPE, budget: int = BUDGET, limit: int = LIMIT):
        assert arm in ("K", "S", "V", "W", "R"), arm
        self.arm, self.seed, self.shape, self.budget, self.limit = arm, seed, shape, budget, limit
        self.g0, self.script, self.oracle = oracle_script(seed, steps)
        self.blind = copy_graph(self.g0); self.pol = copy_graph(self.g0)
        self.turns: List[Tuple[str, str]] = []; self.lead = ""
        self.log: List[dict] = []; self.summaries: List[dict] = []; self.stopped: Optional[str] = None; self.t = 0; self._pending = None

    @property
    def done(self) -> bool: return self.stopped is not None or self.t >= len(self.script)

    # --- what the model is shown
    def _user(self) -> str:
        ins = self.script[self.t]["instruction"]
        body = f"{_serialize(self.g0)}\n\nRequest: {ins}" if self.t == 0 else f"Request: {ins}"
        return (LEAD + self.lead + "\n\n" + body) if self.lead else body

    def _history(self) -> List[Tuple[str, str]]:
        n = len(self.turns)
        if self.arm in ("K", "S", "V") or n == 0: return self.turns
        if self.arm == "W": return [self.turns[0]] + self.turns[max(1, n - WINDOW_TURNS):]
        now = coords(self.script[self.t]["instruction"]); lo = max(1, n - R_LAST)
        scored = sorted(((len(coords(u + " " + a) & now), i) for i, (u, a) in enumerate(self.turns[1:lo], start=1)), key=lambda x: (-x[0], -x[1]))
        pick = sorted(i for sc, i in scored[:R_TOP] if sc > 0)
        return [self.turns[0]] + [self.turns[i] for i in pick] + self.turns[lo:]

    def _messages(self, user: str) -> List[dict]:
        return [{"role": "system", "content": SYSTEM}] + flat(self._history()) + [{"role": "user", "content": user}]

    # --- the round protocol
    def prepare(self, engine) -> Optional[str]:
        if self.done: return None
        msgs = self._messages(self._user()); n = count_tokens(engine, msgs)
        if self.arm in ("S", "V") and n > self.budget and len(self.turns) >= 2: return "compact"
        if self.arm == "K" and n > self.limit: self.stopped = "context"; return "stop"
        self._pending = (msgs, self._user()); return "ready"

    def compaction_messages(self) -> List[dict]:
        ask = COMPACT_GENERIC if self.arm == "S" else COMPACT_VERBATIM
        return [{"role": "system", "content": SYSTEM}] + flat(self.turns) + [{"role": "user", "content": ask}]

    def canned_summary(self) -> str:                       # read by a test double only
        return _serialize(self.oracle[self.t]) + "\nRule: delete a service together with its edges." if self.arm == "V" else "We keep a service graph and apply the requested changes."

    def accept_summary(self, rep: PS.Reply) -> None:
        self.summaries.append(dict(t=self.t, text=rep.text, n_prompt=rep.n_prompt, n_cached=rep.n_cached, n_new=rep.n_new, truncated=rep.n_new >= MAX_SUMMARY,
                                   state_fidelity=state_fidelity(rep.text, self.oracle[self.t])))
        self.turns = []; self.lead = rep.text.strip()

    def request(self) -> T.TRequest:
        st = self.script[self.t]; msgs, _ = self._pending; rx = T.regex_for(self.shape, st["instruction"])
        assert re.fullmatch(rx, st["ideal"]), f"the expression of this request excludes the right answer: {st['instruction']!r} {st['ideal']!r}"
        return T.TRequest(msgs, st["ideal"], seed=self.seed, step=self.t + 1, prompt="old", shape=self.shape, regex=rx)

    def accept(self, rep: PS.Reply) -> None:
        st = self.script[self.t]; d = strict_parse(rep.text); prev, want = self.oracle[self.t], self.oracle[self.t + 1]
        self.blind = next_state("blind", self.blind, d); self.pol = next_state(PS.POLICY, self.pol, d)
        self.log.append(dict(t=self.t + 1, kind=st["kind"], text=rep.text, n_prompt=rep.n_prompt, n_cached=rep.n_cached, n_new=rep.n_new, parsed=d is not None,
                             step_ok=same_graph(next_state("blind", prev, d), want), exact_blind=same_graph(self.blind, want), exact_policy=same_graph(self.pol, want),
                             corrupted_blind=not self.blind.is_well_formed(), text_equal=rep.text == st["ideal"], n_msgs=len(self._pending[0])))
        self.turns.append((self._pending[1], rep.text)); self.lead = ""; self.t += 1


# ---------------------------------------------------------------- the run
def run_memory(engine, sess: List[MemSession], progress=None) -> dict:
    """Lockstep rounds as in `provider_side_long.run_sessions`; before a round, every chat that is over its budget makes its summary call (one round of its own, no grammar)."""
    engine.reset_cache(); rounds: List[dict] = []; wall = comp_secs = 0.0; n_comp = 0; seen: set = set(); guard = 0
    while not all(s.done for s in sess):
        guard += 1; assert guard < 100000, "the round loop does not advance"
        status = {id(s): s.prepare(engine) for s in sess if not s.done}
        need = [s for s in sess if status.get(id(s)) == "compact"]
        if need:
            reps, secs = free_generate(engine, [s.compaction_messages() for s in need], [s.canned_summary() for s in need])
            for s, r in zip(need, reps): s.accept_summary(r)
            wall += secs; comp_secs += secs; n_comp += len(need)
            for s in need: status[id(s)] = s.prepare(engine)
        active = [s for s in sess if status.get(id(s)) == "ready"]
        if not active: continue
        rqs = [s.request() for s in active]; fresh = {r.regex for r in rqs} - seen; seen |= fresh
        reps, secs = engine.generate_round(rqs, reset_cache=False); assert len(reps) == len(active)
        for s, r in zip(active, reps): s.accept(r)
        rounds.append(dict(T._round(len(rounds) + 1, len(active), secs, reps, len(fresh)), compactions=len(need))); wall += secs
        if progress: progress(len(rounds))
    return dict(wall_seconds=wall, compaction_seconds=comp_secs, compactions=n_comp, rounds=rounds, distinct_regexes=len(seen),
                sessions=[dict(seed=s.seed, arm=s.arm, script_steps=len(s.script), stopped=s.stopped, log=s.log, summaries=s.summaries) for s in sess])


def make_sessions(arm: str, steps: int = STEPS, jobs=JOBS, seed: int = SEED) -> list:
    if arm == "A": return [T.TypedSession(seed * 1000 + e, steps, "old", SHAPE) for e in jobs]
    return [MemSession(arm, seed * 1000 + e, steps) for e in jobs]


def run_arm(engine, arm: str, steps: int = STEPS, jobs=JOBS, seed: int = SEED, progress=None, meter=None) -> dict:
    import axi.experiments.provider_side_big as B
    sess = make_sessions(arm, steps, jobs, seed); meter = meter if meter is not None else B.PowerMeter()
    meter.start()
    r = L.run_sessions(engine, sess, progress) if arm == "A" else run_memory(engine, sess, progress)
    power = meter.stop(r["wall_seconds"])
    return dict(label=arm, arm=arm, jobs=list(jobs), steps=steps, seed=seed, shape=SHAPE, policy=PS.POLICY, budget=BUDGET, limit=LIMIT, window_turns=WINDOW_TURNS, r_last=R_LAST, r_top=R_TOP,
                max_summary=MAX_SUMMARY, engine=engine.info(), power=power, **r)


# ---------------------------------------------------------------- the numbers
def f(x, d=2): return None if x is None else round(x, d)


def share(xs: List[dict], key: str) -> Optional[float]:
    return f(100.0 * sum(1 for x in xs if x[key]) / len(xs)) if xs else None


def add_flags(cell: dict) -> None:
    """Arm A's log has `exact` (held state, policy); `step_ok` and `exact_blind` are worked out from its replies and the oracle, so that the same flags exist for every arm."""
    if cell["arm"] != "A": return
    for s in cell["sessions"]:
        if s["log"] and "step_ok" in s["log"][0]: continue
        _, _, oracle = oracle_script(s["seed"], cell["steps"]); blind = copy_graph(oracle[0])
        for x in s["log"]:
            t = x["t"]; d = strict_parse(x["text"]); blind = next_state("blind", blind, d)
            x["step_ok"] = same_graph(next_state("blind", oracle[t - 1], d), oracle[t]); x["exact_blind"] = same_graph(blind, oracle[t]); x["exact_policy"] = x["exact"]; x["text_equal"] = None


def summarize_arm(cell: dict, win: int = 100) -> dict:
    add_flags(cell)
    logs = [x for s in cell["sessions"] for x in s["log"]]; n = len(logs); out = dict(arm=cell["arm"], jobs=cell["jobs"], steps=n, wall_seconds=f(cell["wall_seconds"], 1))
    out["stops"] = [(s["seed"] % 1000, len(s["log"]), s["stopped"], s["script_steps"]) for s in cell["sessions"]]
    for k in ("step_ok", "exact_blind", "exact_policy", "parsed"): out[k] = share(logs, k)
    out["corrupted_blind"] = sum(1 for x in logs if x.get("corrupted_blind")) if cell["arm"] != "A" else None
    out["first_miss"] = [next((x["t"] for x in s["log"] if not x["step_ok"]), None) for s in cell["sessions"]]
    out["first_blind_off"] = [next((x["t"] for x in s["log"] if not x["exact_blind"]), None) for s in cell["sessions"]]
    p = sum(x["n_prompt"] for x in logs); c = sum((x["n_cached"] or 0) for x in logs)
    sums = [m for s in cell["sessions"] for m in s.get("summaries", [])]
    sp = sum(m["n_prompt"] for m in sums); sc = sum((m["n_cached"] or 0) for m in sums); sn = sum(m["n_new"] for m in sums)
    out["compactions"] = len(sums); out["summary_tokens_mean"] = f(sn / len(sums), 1) if sums else None; out["summaries_truncated"] = sum(1 for m in sums if m["truncated"])
    out["summary_state_fidelity"] = dict(exact=sum(1 for m in sums if m["state_fidelity"] is True), differs=sum(1 for m in sums if m["state_fidelity"] is False), none=sum(1 for m in sums if m["state_fidelity"] is None))
    out["prompt_per_step"] = f(p / n, 1) if n else None
    out["computed_per_step"] = f(((p - c) + (sp - sc)) / n, 1) if n else None                 # summary calls included
    out["computed_per_step_steps_only"] = f((p - c) / n, 1) if n else None
    out["cached_share"] = f(c / p, 4) if p else None
    out["new_tokens_per_step"] = f((sum(x["n_new"] for x in logs) + sn) / n, 1) if n else None
    out["sec_per_step"] = f(cell["wall_seconds"] / n, 4) if n else None; out["compaction_seconds"] = f(cell.get("compaction_seconds"), 1)
    out["max_prompt"] = max((x["n_prompt"] for x in logs), default=None); out["power"] = cell.get("power")
    ws = []
    for lo in range(1, cell["steps"] + 1, win):
        xs = [x for x in logs if lo <= x["t"] < lo + win]
        if xs: ws.append(dict(lo=lo, n=len(xs), step_ok=share(xs, "step_ok"), exact_blind=share(xs, "exact_blind"), exact_policy=share(xs, "exact_policy"), prompt=f(sum(x["n_prompt"] for x in xs) / len(xs), 1)))
    out["windows"] = ws
    dk = [x for x in logs if x["kind"] in ("delete_dep", "delete_leaf")]; out["step_ok_delete"] = share(dk, "step_ok"); out["step_ok_add"] = share([x for x in logs if x["kind"] not in ("delete_dep", "delete_leaf")], "step_ok")
    return out


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save_arm(cell: dict, path: str) -> dict:
    json.dump(cell, open(path, "w"), default=str)
    return dict(file=path.rsplit("/", 1)[-1], bytes=len(open(path, "rb").read()), sha256=sha256_file(path))

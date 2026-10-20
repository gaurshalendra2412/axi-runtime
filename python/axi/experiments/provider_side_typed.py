"""Provider-side serving test, fifth run: the grammar of the reply is chosen by the TYPE of the request. Page with the predictions written BEFORE any run:
docs/PROVIDER_SIDE_TYPED_TEST.md (concept-map item 111). The fourth run (docs/PROVIDER_SIDE_RETRY_TEST.md, its Result) asked a refused reply once more and found that this did not raise the exactness
on jobs nobody had read (98.1 % with the retry, 98.1 % without). The eight first misses seen in jobs 0 to 127 are all replies to an ADD request, and the page of this test shows that five of them are
outside the answer shape that the request itself allows (a node written as an edge, an extra DEL in front of an ADD), and the other three have the right shape and a wrong number (a coordinate copied
from the state list). This test changes what the engine is ALLOWED to write, per request, and asks whether the held state moves; it does not ask the model again.

What is the same (nothing below is changed; the files are imported and not edited, their sha256 are pinned on the page and in the test): the model, the engine class, the 80,000-token cache, the
30-step jobs and their seeds, the prompts of the earlier runs, the lockstep rounds, the policy repair_m6 (`cascade.next_state`), the gate, the repair, the measures, the capped expression of the third run.
What is new, and only this:
  * `parse_request`: the three request sentences the jobs use are read by the host, in Python, with no model call (`Add a service at (x,y) with value v.`, `Add a dependency edge from (a,b) to (c,d) with
    relation r and weight w.`, `Delete the service at (x,y).`). A sentence of another form is an error: there is no fallback;
  * `regex_for(shape, instruction)`: the expression the engine must apply to THIS request. `capped` is the third run's expression for every request; `kind` is the expression of the kind of request
    (an add_node reply is one ADD node item, an add_edge reply is one ADD edge item, a delete reply is only DEL items); `named` is the kind AND the coordinates the request names (the model still writes
    the number, and for a delete the edges). A delete request under `named` may only delete edges that touch the named service;
  * three prompts: `old` (the earlier runs'), `anchor` (concept-map item 110: a line `TARGET ...` at the top of the user message and again as its last line) and `tail` (the same line as the last line
    only, which keeps the engine's prefix cache);
  * `TypedVLLMEngine`: one set of sampling parameters per request, so that each request carries its own expression;
  * seven cells of 64 jobs (below), 17 reading lines, a probe at the eight known first misses, and `q_measure.py` (O / M from saved files, no GPU).

Honest limits: as the third run (one model, one small GPU, one engine and its version, synthetic graphs, greedy decoding, lockstep rounds, GPU seconds and not dollars, no energy), plus: the host reads the
request from the sentence the job was built from; in a real system the kind of a request and the coordinates it names would come from whoever made the request, and a request in free words is a different
test. A grammar makes a reply well formed, not right: under `named` the model is left with the numbers and the edge list, and the test says so. `ScriptedStandIn` (a model of a prefix cache that obeys
the grammar) is used to test the harness only; its numbers are never a result."""
import hashlib, json, os, re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
from axi.experiments.long_horizon import oracle_script
from axi.experiments.provider_side import _ge, _le

# ---------------------------------------------------------------- the settings written on the page
MODEL, STEPS, SEED, POLICY, HINT_RULES, MAX_NEW, CAPACITY_TOKENS = B.MODEL, B.STEPS, B.SEED, B.POLICY, B.HINT_RULES, B.MAX_NEW, B.CAPACITY_TOKENS
NAME_CAP, REGEX, REGEX_SHA256 = B.NAME_CAP, B.REGEX, B.REGEX_SHA256
SHAPES = ("capped", "kind", "named")
PROMPTS = ("old", "anchor", "tail")

# label: (jobs side by side, first job number, prompt, grammar). F = fresh jobs (128 to 191, no earlier run has seen them), S = seen jobs (0 to 63). Job e has seed SEED * 1000 + e.
CELLS: Dict[str, Tuple[int, int, str, str]] = {
    "F0": (64, 128, "old", "capped"),
    "FK": (64, 128, "old", "kind"),
    "FKA": (64, 128, "anchor", "kind"),
    "FKT": (64, 128, "tail", "kind"),
    "FN": (64, 128, "old", "named"),
    "SK": (64, 0, "old", "kind"),
    "SN": (64, 0, "old", "named"),
}
ORDER = ["F0", "FK", "FKA", "FKT", "FN", "SK", "SN"]
THIRD_S64 = 64                                    # the third run's `S` cell on the same jobs as SK and SN
REPLAY_PAIRS = [("FK", "F0", "kind"), ("FN", "F0", "named"), ("SK", None, "kind"), ("SN", None, "named")]      # (cell, base, grammar); base None = the third run's `S` cell at 64 jobs

T_PARSED = 0.995                                  # line 2
T_GAIN = 0.01                                     # line 6: one point
T_HIGH, T_HIGHER, T_FRESH_NAMED = 0.99, 0.995, 0.995     # lines 8, 9, 10
T_TIME_KIND, T_TIME_NAMED, T_TIME_TOP, T_TIME_TAIL = 0.05, 0.15, 0.30, 0.20          # lines 13 to 16

TL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens, and every reply is in the language of its own request"),
    ("guard", "the grammar changed only what it forbids: for each of FK and FN against F0 and SK and SN against the third run's `S` cell at 64 jobs, every job's replies are the same text as the base's up to the job's first step whose base reply is outside the new language, and the reply at that step is inside it"),
    ("guard", "the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs, and the oracle's own reply is inside the expression of every request"),
    ("claim", "on the fresh jobs (128 to 191) the exact share with the kind grammar (FK) is not below the share with the capped grammar (F0)"),
    ("claim", "on those jobs the exact share with the kind grammar (FK) is at least one point above the share with the capped grammar (F0)"),
    ("claim", "on those jobs no job holds the right state on fewer steps with the kind grammar (FK) than with the capped grammar (F0)"),
    ("claim", "on the seen jobs (0 to 63) the exact share with the kind grammar (SK) is at least 99 percent (the third run: 97.2; the most it could be is 99.3)"),
    ("claim", "on the seen jobs (0 to 63) the exact share with the kind and coordinates grammar (SN) is at least 99.5 percent"),
    ("claim", "on the fresh jobs the exact share with the kind and coordinates grammar (FN) is at least 99.5 percent"),
    ("claim", "on the fresh jobs the exact share with the anchor line at the top and at the end (FKA) is not below that of FK"),
    ("claim", "on the fresh jobs the exact share with the anchor line at the end only (FKT) is not below that of FK"),
    ("claim", "the seconds per step of FK are at most 5 percent above those of F0 (both in this session)"),
    ("claim", "the seconds per step of FN are at most 15 percent above those of F0 (both in this session)"),
    ("claim", "the seconds per step of FKA are at most 30 percent above those of FK (both in this session)"),
    ("claim", "the seconds per step of FKT are at most 20 percent above those of FK (both in this session)"),
    ("claim", "the exact steps per GPU hour of FK are higher than those of F0 (both in this session)"),
]
TL_HEADLINE = (6, 17)

# the eight first misses seen in jobs 0 to 127 (job, step, reply of the earlier runs): six in jobs 0 to 63 (third and fourth run), two in jobs 64 to 127 (fourth run's H0)
KNOWN_SLIPS: List[Tuple[int, int, str]] = [
    (3, 16, "ADD[(4,4)->(4,4):value#6]"), (4, 18, "ADD[1,0:6]"), (17, 20, "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]"), (19, 12, "ADD[(3,0)->(3,0):4tworowsdown4col#4]"),
    (20, 30, "ADD[(1,5)->(2,1):dep#3]"), (62, 12, "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]"), (101, 4, "ADD[(3,4)->(3,4):2alue#2]"), (122, 22, "ADD[(0,0)->(2,0):dep#3]")]


# ---------------------------------------------------------------- the host reads the request (no model call)
_P_NODE = re.compile(r"Add a service at \(([0-9]+),([0-9]+)\) with value (-?[0-9]+)\.")
_P_EDGE = re.compile(r"Add a dependency edge from \(([0-9]+),([0-9]+)\) to \(([0-9]+),([0-9]+)\) with relation ([A-Za-z0-9_]+) and weight (-?[0-9]+)\.")
_P_DEL = re.compile(r"Delete the service at \(([0-9]+),([0-9]+)\)\.")


def parse_request(instruction: str) -> dict:
    """The kind and the numbers a request names. A sentence that is not one of the three forms is an error: nothing falls back to a looser reading."""
    m = _P_NODE.fullmatch(instruction)
    if m: return dict(kind="add_node", node=(m[1], m[2]), value=m[3])
    m = _P_EDGE.fullmatch(instruction)
    if m: return dict(kind="add_edge", src=(m[1], m[2]), dst=(m[3], m[4]), rel=m[5], weight=m[6])
    m = _P_DEL.fullmatch(instruction)
    if m: return dict(kind="delete", node=(m[1], m[2]))
    raise ValueError(f"not a request this test can read: {instruction!r}")


def target_body(instruction: str) -> str:
    """The target of the request in the words of the anchor line, without the word TARGET."""
    p = parse_request(instruction)
    if p["kind"] == "add_node": return f"({p['node'][0]},{p['node'][1]}) ADD NODE value {p['value']}"
    if p["kind"] == "add_edge": return f"({p['src'][0]},{p['src'][1]})->({p['dst'][0]},{p['dst'][1]}) ADD EDGE {p['rel']} weight {p['weight']}"
    return f"({p['node'][0]},{p['node'][1]}) DELETE NODE"


def target_line(instruction: str) -> str:
    return "TARGET " + target_body(instruction)


def anchored(content: str, instruction: str, prompt: str) -> str:
    """The user message of a prompt variant. `old`: unchanged. `anchor`: the target line at the top and again as the last line (concept-map item 110). `tail`: the last line only."""
    assert prompt in PROMPTS, prompt
    t = target_line(instruction)
    return content if prompt == "old" else f"{t}\n\n{content}\n{t}" if prompt == "anchor" else f"{content}\n{t}"


# ---------------------------------------------------------------- the grammars
_PT = r"[0-9]+,[0-9]+"
_NUM = r"-?[0-9]+"
ADD_NODE = r"ADD\[" + _PT + r":" + _NUM + r"\]"
DEL_NODE = r"DEL\[" + _PT + r"\]"
SEP, TAIL = r"[ \n]", r"[ \n]?"


def _edge(op: str, a: str = _PT, b: str = _PT, name: str = B._NAME) -> str:
    return op + r"\[\(" + a + r"\)->\(" + b + r"\):" + name + r"#" + _NUM + r"\]"


def regex_for(shape: str, instruction: str) -> str:
    """The expression the engine must apply to this request. `capped` is the third run's expression whatever the request is. `kind`: one ADD node item, one ADD edge item, or only DEL items, by the
    kind of the request. `named`: the kind and the coordinates the request names (the number of a node or the weight of an edge is still the model's; a delete begins with the named service and may
    only delete edges that touch it)."""
    assert shape in SHAPES, shape
    if shape == "capped": return REGEX
    p = parse_request(instruction); k = p["kind"]
    if shape == "kind":
        if k == "add_node": return ADD_NODE + TAIL
        if k == "add_edge": return _edge("ADD") + TAIL
        item = "(?:" + DEL_NODE + "|" + _edge("DEL") + ")"
        return item + "(?:" + SEP + item + ")*" + TAIL
    if k == "add_node":
        x, y = p["node"]; return f"ADD\\[{x},{y}:{_NUM}\\]" + TAIL
    if k == "add_edge":
        (a, b), (c, d) = p["src"], p["dst"]; return f"ADD\\[\\({a},{b}\\)->\\({c},{d}\\):{re.escape(p['rel'])}#{_NUM}\\]" + TAIL
    x, y = p["node"]; own = f"{x},{y}"
    touching = "(?:" + _edge("DEL", a=own) + "|" + _edge("DEL", b=own) + ")"
    return f"DEL\\[{own}\\]" + "(?:" + SEP + touching + ")*" + TAIL


@lru_cache(maxsize=None)
def script_of(seed: int, steps: int = STEPS) -> Tuple[dict, ...]:
    """The requests of a job (kind, sentence, the oracle's reply). Deterministic in the seed."""
    return tuple(oracle_script(seed, steps)[1])


def in_language(shape: str, instruction: str, text: str) -> bool:
    return re.fullmatch(regex_for(shape, instruction), text) is not None


def grammar_sha256(shape: str, first: int, K: int, steps: int = STEPS, seed: int = SEED) -> str:
    """sha256 of the sorted set of expressions a cell's requests carry, recomputed from its jobs."""
    rx = {regex_for(shape, st["instruction"]) for e in range(first, first + K) for st in script_of(seed * 1000 + e, steps)}
    return hashlib.sha256("\n".join(sorted(rx)).encode()).hexdigest()


# ---------------------------------------------------------------- the reply read in the words of the target (used by the probe and by the descriptions on the page)
_FIRST = re.compile(r"(ADD|DEL)\[(?:\(([0-9]+),([0-9]+)\)->\(([0-9]+),([0-9]+)\):([A-Za-z0-9_]+)#(-?[0-9]+)|([0-9]+),([0-9]+)(?::(-?[0-9]+))?)\]")


def describe_reply(text: str) -> Optional[str]:
    """The first item of a reply, in the words of the target line (None if it cannot be read)."""
    m = _FIRST.match(text.strip())
    if not m: return None
    op, a, b, c, d, name, w, x, y, v = m.groups()
    if x is not None: return f"({x},{y}) ADD NODE value {v}" if op == "ADD" else f"({x},{y}) DELETE NODE"
    return f"({a},{b})->({c},{d}) ADD EDGE {name} weight {w}" if op == "ADD" else f"({a},{b})->({c},{d}) DELETE EDGE {name}"


def parity_ok(instruction: str, text: str) -> bool:
    """The reply's first item says what the request says; an add request has exactly one item. Descriptive: no cell of this test uses it to decide a step."""
    items = re.findall(r"(?:ADD|DEL)\[[^\]]*\]", text)
    if parse_request(instruction)["kind"] != "delete" and len(items) != 1: return False
    return describe_reply(text) == target_body(instruction)


def parity_text(instruction: str, text: str) -> str:
    """The user turn of a second request that says, in the same words, what the request names and what the reply named."""
    return f"PARITY request {target_line(instruction)}; reply {describe_reply(text) or 'unreadable'}. Reply with only the corrected delta."


# ---------------------------------------------------------------- what a session asks and what it gets back
@dataclass
class TRequest(PS.Request):
    seed: int = 0                                 # the job's seed (read by the test double only)
    step: int = 0                                 # the step, from 1
    prompt: str = "old"
    shape: str = "capped"
    regex: str = REGEX                            # the expression the engine must apply to this request


class TypedSession(PS.Session):
    """The exact-state job of `provider_side.Session` (unchanged: it adds, it overrides nothing that decides a step), with the prompt variant and the grammar of its cell."""
    def __init__(self, seed: int, steps: int = STEPS, prompt: str = "old", shape: str = "capped"):
        assert prompt in PROMPTS and shape in SHAPES, (prompt, shape)
        super().__init__("S", seed, steps, HINT_RULES); self.prompt, self.shape = prompt, shape; self._rx = REGEX

    def request(self) -> TRequest:
        rq = super().request(); st = self.script[self.t]; msgs = rq.messages
        if self.prompt != "old": msgs = [msgs[0], dict(msgs[1], content=anchored(msgs[1]["content"], st["instruction"], self.prompt))]
        self._rx = regex_for(self.shape, st["instruction"])
        assert re.fullmatch(self._rx, st["ideal"]), f"the expression of this request excludes the right answer: {st['instruction']!r} {st['ideal']!r}"       # a grammar that cannot say the answer is not this test
        return TRequest(msgs, rq.stand_in_reply, seed=self.seed, step=self.t + 1, prompt=self.prompt, shape=self.shape, regex=self._rx)

    def accept(self, rep: PS.Reply) -> None:
        super().accept(rep); self.log[-1]["in_language"] = re.fullmatch(self._rx, rep.text) is not None


# ---------------------------------------------------------------- the engines and the test double
def constraint_info(kind: str) -> dict:
    return dict(kind=kind, api="structured_outputs(regex), one set of sampling parameters per request", regex=REGEX, regex_sha256=REGEX_SHA256, name_cap=NAME_CAP, shapes=list(SHAPES), prompts=list(PROMPTS))


class TypedVLLMEngine(PS.VLLMEngine):
    """The first run's engine; every request carries its own expression (`TRequest.regex`). Not testable without a GPU; tests/test_provider_side_typed.py runs it against a fake `vllm` module, and the
    notebook's safety cell shows within minutes whether the installed version takes one expression per request."""
    def __init__(self, *a, **kw):
        params = C._structured_params(REGEX)                                 # before the model is loaded: a version that cannot constrain fails in a second, not after ten minutes
        super().__init__(*a, **kw)
        from vllm import SamplingParams
        self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=params)           # the warm-up request

    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("regex_per_request"); return d

    def generate_round(self, requests, reset_cache: bool = False):
        import time
        from vllm import SamplingParams
        if reset_cache: self.reset_cache()
        prompts = [self.tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True) for r in requests]
        sps = [SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=C._structured_params(getattr(r, "regex", REGEX))) for r in requests]
        t0 = time.perf_counter(); outs = self.llm.generate(prompts, sps, use_tqdm=False); dt = time.perf_counter() - t0
        return [PS.Reply(o.outputs[0].text, len(o.prompt_token_ids), getattr(o, "num_cached_tokens", None), len(o.outputs[0].token_ids)) for o in outs], dt


class TypedStandIn(B.BoundedStandIn):
    """The stand-in with the per-request constraint recorded. It answers with the oracle's own reply (which is inside every expression). Tests the harness only. Never a result."""
    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("stand_in"); return d


class ScriptedStandIn(TypedStandIn):
    """A stand-in that obeys the grammar. `faults` maps (prompt, seed, step) to the reply the unconstrained model would write there; if that reply is outside the request's expression the engine cannot
    write it and writes `forced[(prompt, seed, step)]` (default: the oracle's reply) instead. Tests the harness only. Never a result."""
    def __init__(self, faults: Optional[dict] = None, forced: Optional[dict] = None, **kw):
        super().__init__(**kw); self.faults = dict(faults or {}); self.forced = dict(forced or {})

    def generate_round(self, requests, reset_cache: bool = False):
        rqs = []
        for r in requests:
            key = (getattr(r, "prompt", "old"), getattr(r, "seed", None), getattr(r, "step", None)); text = self.faults.get(key)
            if text is not None and re.fullmatch(getattr(r, "regex", REGEX), text) is None: text = self.forced.get(key, r.stand_in_reply)
            rqs.append(r if text is None else TRequest(r.messages, text, seed=r.seed, step=r.step, prompt=r.prompt, shape=r.shape, regex=r.regex))
        return super().generate_round(rqs, reset_cache)


# ---------------------------------------------------------------- one cell
def _round(n: int, active: int, secs: float, reps: List[PS.Reply], new_grammars: int) -> dict:
    cached = None if any(r.n_cached is None for r in reps) else sum(r.n_cached for r in reps)
    return dict(round=n, active=active, seconds=secs, prompt_tokens=sum(r.n_prompt for r in reps), cached_tokens=cached, new_tokens=sum(r.n_new for r in reps), new_grammars=new_grammars)


def run_jobs(engine, episodes: List[int], prompt: str, shape: str, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    """K jobs side by side in lockstep, as the third run: a round for every unfinished job's request, then every job applies its reply. Returns the rounds, the wall seconds and the jobs."""
    sess = [TypedSession(seed * 1000 + e, steps, prompt, shape) for e in episodes]
    engine.reset_cache(); rounds: List[dict] = []; wall = 0.0; seen: set = set()                   # every cell starts with a cold cache
    while True:
        active = [s for s in sess if not s.done]
        if not active: break
        rqs = [s.request() for s in active]; fresh = {r.regex for r in rqs} - seen; seen |= fresh
        reps, secs = engine.generate_round(rqs, reset_cache=False); assert len(reps) == len(active)
        for s, r in zip(active, reps): s.accept(r)
        rounds.append(_round(len(rounds) + 1, len(active), secs, reps, len(fresh))); wall += secs
        if progress: progress(len(rounds))
    return dict(wall_seconds=wall, rounds=rounds, distinct_regexes=len(seen), sessions=[dict(seed=s.seed, script_steps=len(s.script), stopped=s.stopped, log=s.log) for s in sess])


def run_cell(engine, label: str, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    K, first, prompt, shape = CELLS[label]
    r = run_jobs(engine, list(range(first, first + K)), prompt, shape, steps, seed, progress)
    return dict(label=label, arm="S", K=K, first_episode=first, prompt=prompt, shape=shape, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(),
                grammar_sha256=grammar_sha256(shape, first, K, steps, seed), **r)


# ---------------------------------------------------------------- the probe: the eight known first misses, rebuilt, under each variant
PROBE_VARIANTS = [("old", "old", "capped"), ("kind", "old", "kind"), ("named", "old", "named"), ("kind+top", "anchor", "kind"), ("kind+tail", "tail", "kind")]


def _prefix_session(job: int, step: int, prompt: str, shape: str, seed: int) -> TypedSession:
    s = TypedSession(seed * 1000 + job, STEPS, prompt, shape)
    while s.t < step - 1:
        rq = s.request(); s.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
    return s


def known_slip_probe(engine, seed: int = SEED, slips: Optional[List[Tuple[int, int, str]]] = None) -> List[dict]:
    """For each known first miss: the exact-state prompt of that step rebuilt with the oracle's own replies for the steps before (the jobs held the right state up to there), asked under each variant
    (capped, kind, named; kind with the anchor line at the top and at the end), and once more under the capped expression with the earlier run's wrong reply as the assistant's turn and a
    `PARITY` user turn that says what the request names and what the reply named. The answers are descriptions: a wrong answer is not a STOP."""
    out = []; slips = KNOWN_SLIPS if slips is None else slips; reqs = []; meta = []
    for job, step, old in slips:
        for name, prompt, shape in PROBE_VARIANTS:
            s = _prefix_session(job, step, prompt, shape, seed); reqs.append(s.request()); meta.append((job, step, name))
        s = _prefix_session(job, step, "old", "capped", seed); rq = s.request(); instr = s.script[s.t]["instruction"]
        reqs.append(TRequest(list(rq.messages) + [{"role": "assistant", "content": old}, {"role": "user", "content": parity_text(instr, old)}], rq.stand_in_reply, seed=rq.seed, step=rq.step, prompt="old", shape="capped", regex=REGEX))
        meta.append((job, step, "parity"))
    reps, secs = engine.generate_round(reqs)
    by: Dict[Tuple[int, int], dict] = {}
    for (job, step, name), rq, rep in zip(meta, reqs, reps): by.setdefault((job, step), {})[name] = (rep.text, rep.n_new, rep.n_cached)
    for job, step, old in slips:
        s = _prefix_session(job, step, "old", "capped", seed); instr = s.script[s.t]["instruction"]; oracle = s.script[s.t]["ideal"]
        out.append(dict(job=job, step=step, kind=s.script[s.t]["kind"], instruction=instr, oracle=oracle, earlier=old, earlier_in_kind=in_language("kind", instr, old), earlier_in_named=in_language("named", instr, old),
                        parity=parity_text(instr, old), replies={k: v[0] for k, v in by[(job, step)].items()}, n_new={k: v[1] for k, v in by[(job, step)].items()}, cached={k: v[2] for k, v in by[(job, step)].items()}, seconds=secs))
    return out


# ---------------------------------------------------------------- summaries
def jobs_exact(c: dict) -> List[int]:
    """For every job of a cell, the number of steps after which the held state was exactly the oracle's."""
    return [sum(1 for x in s["log"] if x["exact"]) for s in c["sessions"]]


def summarize_cell4(c: dict) -> dict:
    s = C.summarize_cell2(c); steps = [x for ss in c["sessions"] for x in ss["log"]]; n = len(steps)
    firsts = []
    for ss in c["sessions"]:
        x = next((y for y in ss["log"] if not y["exact"]), None)
        if x is not None: firsts.append((ss["seed"] - c["seed"] * 1000, x["t"], x["kind"], x["text"]))
    s.update(label=c["label"], prompt=c["prompt"], shape=c["shape"], first_episode=c["first_episode"], out_of_language=sum(1 for x in steps if not x.get("in_language", False)),
             distinct_regexes=c.get("distinct_regexes"), jobs_exact_to_end=sum(1 for s_ in c["sessions"] if all(x["exact"] for x in s_["log"]) and len(s_["log"]) == s_["script_steps"]),
             lost_steps=sum(1 for x in steps if not x["exact"]), first_misses=firsts, new_grammar_rounds=sum(1 for r in c["rounds"] if r.get("new_grammars")))
    return s


def summarize_all(cells: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cells:
        assert c["label"] not in out, f"cell {c['label']} twice"; out[c["label"]] = summarize_cell4(c)
    return out


# ---------------------------------------------------------------- merge
def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pinned_hashes() -> dict:
    """sha256 of the files this test imports and does not edit."""
    import axi.engine.gate as G, axi.engine.diagnostics as D, axi.experiments.agent_loop as A, axi.experiments.cascade as K, axi.experiments.long_horizon as L
    return dict(provider_side=PS.fingerprint(), constrained=C.fingerprint(), bounded=B.fingerprint(), gate=sha256_file(G.__file__), diagnostics=sha256_file(D.__file__), agent_loop=sha256_file(A.__file__),
                cascade=sha256_file(K.__file__), long_horizon=sha256_file(L.__file__))


def cells_planned() -> List[str]:
    """In the order they are run: the control and the kind grammar on the fresh jobs first, then the anchor variants, then the grammar with the coordinates, then the seen jobs. About 25 minutes of GPU."""
    return list(ORDER)


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings, a cell twice, a cell that is not one of this test's, a cell whose settings are not those of its label, a cell
    that carries another base constraint, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        lab = c.get("label")
        if lab not in CELLS: raise ValueError(f"cell {lab!r} is not one of this test's cells")
        if (c["K"], c["first_episode"], c["prompt"], c["shape"]) != CELLS[lab]: raise ValueError(f"cell {lab} was not run with the settings of its label: {(c['K'], c['first_episode'], c['prompt'], c['shape'])} against {CELLS[lab]}")
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {lab} was not run with this test's constraint")
        if key(c) != key(ref): raise ValueError(f"cell {lab} was run under other settings: {key(c)} against {key(ref)}")
        if lab in seen: raise ValueError(f"cell {lab} twice")
        seen.add(lab)
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP,
                          cells={k: list(CELLS[k]) for k in ORDER}), cells=sorted(cells, key=lambda c: ORDER.index(c["label"])))


def default_third_run_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "results", "axi_provider_bounded.json")


def load_third_run(path: Optional[str] = None) -> dict:
    return json.load(open(path or default_third_run_path()))


def merge_files(paths: List[str], out: str, third_run: Optional[str] = None) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells); tr = third_run or default_third_run_path()
    m["third_run_file"] = os.path.basename(tr); m["third_run_sha256"] = sha256_file(tr); m["module_sha256"] = fingerprint(); m["pinned_sha256"] = pinned_hashes()
    with open(out, "w") as f: json.dump(m, f)
    return m


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_TYPED_TEST.md before the run)
def replies_of(c: dict) -> List[Tuple[bool, int, Optional[int], bool]]:
    """(parsed, new tokens, cached tokens, inside the language of its own request) of every request of a cell."""
    return [(x["parsed"], x["n_new"], x["n_cached"], bool(x.get("in_language", False))) for s in c["sessions"] for x in s["log"]]


def same_until_forced(new: dict, base: dict, shape: str) -> bool:
    """Job by job (same seed): while the base's reply is inside the language of the request, the new reply is the same text; at the first step whose base reply is outside, the new reply is inside the language
    (the grammar forced it) and the comparison stops; a job whose base replies are all inside has the same replies throughout and the same number of steps."""
    if len(new["sessions"]) != len(base["sessions"]): return False
    for a, b in zip(new["sessions"], base["sessions"]):
        if a["seed"] != b["seed"]: return False
        script = script_of(a["seed"], max(a["script_steps"], b["script_steps"])); forced = False
        for i, (x, y) in enumerate(zip(a["log"], b["log"])):
            instr = script[i]["instruction"]
            if in_language(shape, instr, y["text"]):
                if x["text"] != y["text"]: return False
            else:
                if not in_language(shape, instr, x["text"]): return False
                forced = True; break
        if not forced and len(a["log"]) != len(b["log"]): return False
    return True


def cell_grammar_ok(c: dict) -> bool:
    """The cell's recorded set of expressions is the one recomputed from its jobs, and the oracle's reply is inside the expression of every request of every job."""
    if c.get("grammar_sha256") != grammar_sha256(c["shape"], c["first_episode"], c["K"], c["steps"], c["seed"]): return False
    return all(in_language(c["shape"], st["instruction"], st["ideal"]) for s in c["sessions"] for st in script_of(s["seed"], c["steps"]))


def verdict_typed(new_cells: List[dict], third_cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `new_cells` are the cells of this run, `third_cells` those of the third run (`S` at 8, 32 and 64 jobs). A line that needs a cell that was not run
    is None, never 'no'."""
    new = {c["label"]: c for c in new_cells if any(s["log"] for s in c["sessions"])}; third = {c["K"]: c for c in third_cells if c["arm"] == "S"}
    Sn = {k: summarize_cell4(c) for k, c in new.items()}
    has = lambda *ks: all(k in Sn for k in ks)
    out: List[Optional[bool]] = []
    allrep = [r for k in new for r in replies_of(new[k])]
    out.append(all(r[2] is not None for r in allrep) if allrep else None)                                                                                                                          # 1
    out.append((sum(1 for r in allrep if not r[0]) <= (1 - T_PARSED) * len(allrep) + 1e-9 and all(r[1] < MAX_NEW for r in allrep) and all(r[3] for r in allrep)) if allrep else None)                # 2
    pairs = []
    for cell, base, shape in REPLAY_PAIRS:
        b = third.get(THIRD_S64) if base is None else new.get(base)
        if cell in new and b is not None: pairs.append(same_until_forced(new[cell], b, shape))
    out.append(all(pairs) if pairs else None)                                                                                                                                                       # 3
    out.append(all(cell_grammar_ok(c) for c in new.values()) if new else None)                                                                                                                      # 4
    out.append(None if not has("F0", "FK") else _ge(Sn["FK"]["exact_share"], Sn["F0"]["exact_share"]))                                                                                              # 5
    out.append(None if not has("F0", "FK") else _ge(Sn["FK"]["exact_share"], Sn["F0"]["exact_share"] + T_GAIN))                                                                                     # 6
    out.append(None if not has("F0", "FK") else all(a >= b for a, b in zip(jobs_exact(new["FK"]), jobs_exact(new["F0"]))))                                                                          # 7
    out.append(None if not has("SK") else _ge(Sn["SK"]["exact_share"], T_HIGH))                                                                                                                      # 8
    out.append(None if not has("SN") else _ge(Sn["SN"]["exact_share"], T_HIGHER))                                                                                                                    # 9
    out.append(None if not has("FN") else _ge(Sn["FN"]["exact_share"], T_FRESH_NAMED))                                                                                                               # 10
    out.append(None if not has("FK", "FKA") else _ge(Sn["FKA"]["exact_share"], Sn["FK"]["exact_share"]))                                                                                            # 11
    out.append(None if not has("FK", "FKT") else _ge(Sn["FKT"]["exact_share"], Sn["FK"]["exact_share"]))                                                                                            # 12
    out.append(None if not has("F0", "FK") else _le(Sn["FK"]["seconds_per_step"], Sn["F0"]["seconds_per_step"] * (1 + T_TIME_KIND)))                                                               # 13
    out.append(None if not has("F0", "FN") else _le(Sn["FN"]["seconds_per_step"], Sn["F0"]["seconds_per_step"] * (1 + T_TIME_NAMED)))                                                              # 14
    out.append(None if not has("FK", "FKA") else _le(Sn["FKA"]["seconds_per_step"], Sn["FK"]["seconds_per_step"] * (1 + T_TIME_TOP)))                                                              # 15
    out.append(None if not has("FK", "FKT") else _le(Sn["FKT"]["seconds_per_step"], Sn["FK"]["seconds_per_step"] * (1 + T_TIME_TAIL)))                                                              # 16
    out.append(None if not has("F0", "FK") else Sn["FK"]["exact_per_hour"] > Sn["F0"]["exact_per_hour"])                                                                                            # 17
    assert len(out) == len(TL_LINES)
    return [(i + 1, TL_LINES[i][0], TL_LINES[i][1], out[i]) for i in range(len(TL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim, from the lines: with lines 2, 3 and 4 yes (the constraint held, the grammar changed only what it forbids, the cells used the expressions this test defines)
    the figures of the cells are claimed as read. That a grammar raises the exactness is claimed only from the fresh jobs (lines 5 to 7, 10 to 12); lines 8 and 9 are on the jobs on which the slips were
    read. A statement about time or cost comes only from cells of one session (lines 13 to 17). A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = ok[2] is True and ok[3] is True and ok[4] is True
    pick = lambda ks: ({i: ok[i] for i in ks} if base else {i: None for i in ks})
    return dict(base=base, constraint_held=ok[2], grammar_changed_only_what_it_forbids=ok[3], expressions_as_defined=ok[4], fresh_jobs_kind=pick((5, 6, 7)), fresh_jobs_named=pick((10,)), anchor_lines=pick((11, 12)),
                seen_jobs=pick((8, 9)), time_and_cost_in_one_session=pick((13, 14, 15, 16, 17)))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[str, dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    print(f"\n{'cell':4s} {'K':>3s} {'jobs':>9s} {'prompt':>6s} {'grammar':>7s} | {'steps':>5s} {'seconds':>8s} {'s/step':>7s} {'comp/step':>9s} | {'exact':>6s} {'jobs ok':>7s} {'lost':>4s} | {'unparsed':>8s} {'outside':>7s} {'regexes':>7s} {'exact/h':>8s}")
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; K, first, prompt, shape = CELLS[lab]
        print(f"{lab:4s} {K:3d} {str(first) + '-' + str(first + K - 1):>9s} {prompt:>6s} {shape:>7s} | {s['steps_done']:5d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 3):>7s} {f(s['computed_per_step'], 0):>9s} | "
              f"{pc(s['exact_share']):>6s} {s['jobs_exact_to_end']:7d} {s['lost_steps']:4d} | {s['unparsed']:8d} {s['out_of_language']:7d} {f(s['distinct_regexes'], 0):>7s} {f(s['exact_per_hour'], 0):>8s}")


def print_misses(S: Dict[str, dict]) -> None:
    """The first miss of every job that missed, per cell (descriptive)."""
    for lab in ORDER:
        if lab in S and S[lab]["first_misses"]:
            print(f"first misses in {lab}: " + "; ".join(f"job {j} step {t} ({k}) {x!r}" for j, t, k, x in S[lab]["first_misses"]))


def print_verdict(checks, title="provider-side test, typed requests") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {TL_HEADLINE[0]} and {TL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in TL_HEADLINE) + ")")
    a = claims_allowed(checks)
    if not a["base"]: print("  rule on the page: line 2, 3 or 4 did not read yes, so NO claim is made from this run until the cause is found.")
    else:
        print("  rule on the page: lines 2, 3 and 4 read yes, so the figures of the cells are claimed as read. That a grammar raises the exactness rests on the fresh jobs only (lines 5 to 7, 10 to 12); lines 8 and 9 are on the jobs the slips were read on.")
        print("  a statement about time or cost comes only from cells of one session (lines 13 to 17), never from a cell against the third run's.")
        print("  never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about requests in free words, anything about a model that was not run.")


def dry_run(steps: int = STEPS, labels=None) -> Dict[str, dict]:
    """Every planned cell on the stand-in, with the earlier runs' slips written by the test double where the unconstrained model would write them and the oracle's own reply where the grammar forces it.
    Prints the table. NOT a result."""
    faults = {("old", SEED * 1000 + j, t): text for j, t, text in KNOWN_SLIPS[:6]}
    eng = ScriptedStandIn(faults=faults, capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for lab in ORDER:
        if labels is None or lab in labels: cells.append(run_cell(eng, lab, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache that obeys the grammar; the earlier runs' slips are written by the test double), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, typed requests (docs/PROVIDER_SIDE_TYPED_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_typed.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_misses(S); print_verdict(verdict_typed(m["cells"], load_third_run()["cells"])); print("\nwritten to", a.out)
    else: ap.print_help()

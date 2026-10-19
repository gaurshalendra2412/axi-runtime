"""Provider-side serving test, fourth run: ONE retry when the repair cannot fix a refused reply. Page with the predictions written BEFORE any run:
docs/PROVIDER_SIDE_RETRY_TEST.md (concept-map item 109). The third run (docs/PROVIDER_SIDE_BOUNDED_TEST.md, its Result) held the exact state on 88.3, 94.6 and 97.2 percent of the steps of the exact-state
arm `S` at 8, 32 and 64 jobs. Of the six jobs that missed at 64 jobs, four missed at a step where the gate REFUSED the reply and the repair of policy repair_m6 could not make it admissible
(jobs 3, 4, 19 and 20); the job then stays one step behind to its last step. In that test a refused reply was never asked again. The framework already has the next rung of its own ladder
(rules, feedback, ONE retry, then accept the refusal: `agent_loop.py` mode M3, concept-map rows 26 and 42); this test puts that rung into the provider-side loop and counts what it costs.

What is the same (nothing below is changed; the files are imported and not edited, their sha256 are pinned on the page and in the test): the model, the engine class, the capped expression, the
80,000-token cache, the 30-step jobs and their seeds, the prompts, the lockstep rounds, the policy repair_m6 (`cascade.next_state`), the gate (`gate.check`), the repair
(`diagnostics.ground_and_complete`), the diagnosis (`diagnostics.diagnose`), the retry feedback (`agent_loop.feedback_text`, variant "structured_imperative"), the measures.
What is new, and only this:
  * `RetrySession`: a job that, when the reply it just got is parsed, refused by the gate and not repairable by `ground_and_complete`, can send ONE more request (its first request, the first
    reply as the assistant's turn, and the framework's feedback as the user's turn). The second reply decides the step, as any reply does under repair_m6; if it is refused again the state stays
    (ask once and walk away);
  * `run_cell`: the lockstep round, plus a second, smaller round for the jobs that asked again. Its seconds are part of the cell's wall seconds: a retry is paid for;
  * five cells (below): the three exact-state cells of the third run with the retry on (the same jobs), and a pair on 64 jobs that no earlier run has seen (jobs 64 to 127), without and with the retry;
  * 12 reading lines.

Honest limits: as the third run (one model, one small GPU, one engine and its version, synthetic graphs, greedy decoding, lockstep rounds, GPU seconds and not dollars, no energy; the grammar makes a
reply well formed, not right), plus: one retry only (a budget sweep is not run); in the lockstep model every round with a retry is charged as a whole extra round, which a server that mixes
retries into other traffic would not pay. `FaultyStandIn` (a model of a prefix cache that writes the slips of the third run, for jobs 3, 4, 19 and 20) is used to test the harness only; its numbers
are never a result."""
import hashlib, json, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
from axi.engine.gate import check
from axi.engine.diagnostics import diagnose, ground_and_complete
from axi.experiments.agent_loop import feedback_text, strict_parse
from axi.experiments.provider_side import _ge, _le

# ---------------------------------------------------------------- the settings written on the page
MODEL, STEPS, SEED, POLICY, HINT_RULES, MAX_NEW, CAPACITY_TOKENS = B.MODEL, B.STEPS, B.SEED, B.POLICY, B.HINT_RULES, B.MAX_NEW, B.CAPACITY_TOKENS
NAME_CAP, REGEX, REGEX_SHA256 = B.NAME_CAP, B.REGEX, B.REGEX_SHA256
FEEDBACK = "structured_imperative"                # the variant of agent_loop.feedback_text that recovered 100 percent of the dangling deletes in the earlier tests (docs/EXPERIMENT_RESULTS.md)
DEGREE_THRESHOLD = 4                              # the default of cascade.next_state and of ground_and_complete
MAX_RETRIES = 1                                   # ask once and walk away

# label: (jobs side by side, first job number, retry on). Jobs are numbered from 0; job e has seed SEED * 1000 + e.
CELLS: Dict[str, Tuple[int, int, bool]] = {"R8": (8, 0, True), "R32": (32, 0, True), "R64": (64, 0, True), "H0": (64, 64, False), "H1": (64, 64, True)}
ORDER = ["R8", "R32", "R64", "H0", "H1"]
THIRD_K = {"R8": 8, "R32": 32, "R64": 64}         # the third run's `S` cell with the same jobs
HELD_OUT = ("H0", "H1")

R_PARSED = 0.995                                  # line 2
R_MAX_SHARE = 0.03                                # line 5: at most 3 percent of the steps of a cell ask again
R_HIGH, R_VERY_HIGH = 0.98, 0.99                  # lines 7 and 8
R_HELD_GAIN = 0.01                                # line 9: one point
R_TIME_TOL = 0.10                                 # line 10

RL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run, the second requests included"),
    ("instrument", "the constraint held: of all the replies of all the cells that were run (first and second) at least 99.5 percent were accepted by the strict parser, and none used all 96 new tokens"),
    ("guard", "the first replies are the earlier run's: in the three cells on jobs 0 to K-1, every job's first replies up to and including its first retried step are the same text as the third run's, and every job that asked again nowhere is the same text throughout; in the pair on jobs 64 to 127, the same between the cell with the retry and the cell without"),
    ("guard", "the retry fired only where the rule says: every second request was made for a reply that the strict parser accepted, the gate refused and the repair could not make admissible; and the cell without the retry made none"),
    ("guard", "the retry is rare: in every cell with the retry at most 3 percent of the steps asked again"),
    ("claim", "on the same jobs as the third run, the exact share is higher with the retry at every K: above 88.3, 94.6 and 97.2 percent for K = 8, 32 and 64"),
    ("claim", "on jobs 0 to 63 (K = 64) the exact share with the retry is at least 98 percent (the third run: 97.2; the most it could be is 99.7)"),
    ("claim", "on jobs 0 to 63 (K = 64) the exact share with the retry is at least 99 percent"),
    ("claim", "on the jobs that no earlier run has seen (64 to 127), the exact share with the retry is at least one point above the share without it"),
    ("claim", "on those jobs, the seconds per step with the retry are at most 10 percent above those without it (both in this session)"),
    ("claim", "on those jobs, the exact steps per GPU hour with the retry are higher than without it (both in this session)"),
    ("claim", "on those jobs, no job holds the right state on fewer steps with the retry than without it"),
]
RL_HEADLINE = (9, 11)


# ---------------------------------------------------------------- what a session asks and what it gets back
@dataclass
class RRequest(PS.Request):
    seed: int = 0                                 # the job's seed (read by the test double only)
    step: int = 0                                 # the step, from 1
    retry: bool = False                           # True for the second request of a step


class RetrySession(PS.Session):
    """The exact-state job of `provider_side.Session` (unchanged: this class adds, it overrides nothing that decides a step), plus the rule for asking once more."""
    def __init__(self, seed: int, steps: int = STEPS):
        super().__init__("S", seed, steps, HINT_RULES)

    def request(self) -> RRequest:
        rq = super().request(); return RRequest(rq.messages, rq.stand_in_reply, seed=self.seed, step=self.t + 1, retry=False)

    def read_reply(self, text: str) -> dict:
        """What the framework's own rules say about a first reply, against the state the job holds now. `fires`: the reply parsed, the gate refuses it, and the repair of repair_m6
        (`ground_and_complete`) cannot make it admissible: exactly the replies after which the policy leaves the state unchanged although the model did answer."""
        d = strict_parse(text)
        if d is None: return dict(parsed=False, admitted=False, repairable=False, fires=False, reason="not parsed")
        ck = check(self.live, d)
        if ck.ok: return dict(parsed=True, admitted=True, repairable=False, fires=False, reason="")
        g = ground_and_complete(self.live, d, DEGREE_THRESHOLD)
        rep = bool(g is not None and g.repairable and check(self.live, g.repaired).ok)
        return dict(parsed=True, admitted=False, repairable=rep, fires=not rep, reason=ck.reason)

    def retry_request(self, rq: PS.Request, first_text: str) -> RRequest:
        """The first request, the first reply as the assistant's turn, the framework's feedback as the user's turn."""
        d = strict_parse(first_text); assert d is not None, "a retry is asked for a reply that parsed"
        fb = feedback_text(FEEDBACK, diagnose(self.live, d))
        return RRequest(list(rq.messages) + [{"role": "assistant", "content": first_text}, {"role": "user", "content": fb}], rq.stand_in_reply, seed=self.seed, step=self.t + 1, retry=True)

    def accept(self, rep: PS.Reply) -> None:
        super().accept(rep); self.log[-1]["retried"] = False

    def accept_retried(self, first: PS.Reply, info: dict, rq2: RRequest, rep: PS.Reply) -> None:
        """The second reply decides the step (policy repair_m6 as for any reply). The record is that of the second reply, plus what the first one was."""
        self.accept(rep)
        self.log[-1].update(retried=True, first_text=first.text, first_n_prompt=first.n_prompt, first_n_cached=first.n_cached, first_n_new=first.n_new,
                            first_parsed=info["parsed"], first_admitted=info["admitted"], first_repairable=info["repairable"], first_reason=info["reason"], feedback=rq2.messages[-1]["content"])


# ---------------------------------------------------------------- the engines (the third run's, unchanged) and the test double
BoundedVLLMEngine = B.BoundedVLLMEngine
BoundedStandIn = B.BoundedStandIn


class FaultyStandIn(B.BoundedStandIn):
    """The stand-in that writes a given wrong reply to a given first request (job seed, step) and a given reply to a given second request. Tests the harness only. Never a result.
    `faults` and `retry_faults` map (seed, step) to the text."""
    def __init__(self, faults: Optional[dict] = None, retry_faults: Optional[dict] = None, **kw):
        super().__init__(**kw); self.faults = dict(faults or {}); self.retry_faults = dict(retry_faults or {})

    def generate_round(self, requests, reset_cache: bool = False):
        rqs = []
        for r in requests:
            table = self.retry_faults if getattr(r, "retry", False) else self.faults; key = (getattr(r, "seed", None), getattr(r, "step", None))
            rqs.append(RRequest(r.messages, table[key], seed=r.seed, step=r.step, retry=r.retry) if key in table else r)
        return super().generate_round(rqs, reset_cache)


def third_run_slips() -> Dict[Tuple[int, int], str]:
    """The four first replies of the third run that the gate refused and the repair could not fix (jobs 3, 4, 19, 20; seed SEED * 1000 + job, step), for the test double."""
    return {(SEED * 1000 + 3, 16): "ADD[(4,4)->(4,4):value#6]", (SEED * 1000 + 4, 18): "ADD[1,0:6]", (SEED * 1000 + 19, 12): "ADD[(3,0)->(3,0):4tworowsdown4col#4]", (SEED * 1000 + 20, 30): "ADD[(1,5)->(2,1):dep#3]"}


# ---------------------------------------------------------------- one cell
def _round(n: int, kind: str, active: int, secs: float, reps: List[PS.Reply]) -> dict:
    cached = None if any(r.n_cached is None for r in reps) else sum(r.n_cached for r in reps)
    return dict(round=n, kind=kind, active=active, seconds=secs, prompt_tokens=sum(r.n_prompt for r in reps), cached_tokens=cached, new_tokens=sum(r.n_new for r in reps))


def run_jobs(engine, episodes: List[int], retry: bool, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    """K jobs side by side in lockstep: a round for every unfinished job's request, then (if `retry`) a second round for the jobs whose reply the rule says to ask again, then every job applies
    the reply that decides its step. Returns the rounds, the wall seconds (both kinds of round) and the jobs."""
    sess = [RetrySession(seed * 1000 + e, steps) for e in episodes]
    engine.reset_cache(); rounds: List[dict] = []; wall = 0.0                      # every cell starts with a cold cache
    while True:
        active = [s for s in sess if not s.done]
        if not active: break
        rqs = [s.request() for s in active]
        reps, secs = engine.generate_round(rqs, reset_cache=False); assert len(reps) == len(active)
        rounds.append(_round(len(rounds) + 1, "first", len(active), secs, reps)); wall += secs
        infos = [s.read_reply(r.text) for s, r in zip(active, reps)]
        pend = [i for i, f in enumerate(infos) if retry and f["fires"]]; second: Dict[int, Tuple[RRequest, PS.Reply]] = {}
        if pend:
            rq2 = [active[i].retry_request(rqs[i], reps[i].text) for i in pend]
            reps2, secs2 = engine.generate_round(rq2, reset_cache=False); assert len(reps2) == len(pend)
            rounds.append(_round(len(rounds) + 1, "retry", len(pend), secs2, reps2)); wall += secs2
            second = {i: (q, r) for i, q, r in zip(pend, rq2, reps2)}
        for i, (s, r) in enumerate(zip(active, reps)):
            if i in second: s.accept_retried(r, infos[i], *second[i])
            else: s.accept(r)
        if progress: progress(len(rounds))
    return dict(wall_seconds=wall, rounds=rounds, sessions=[dict(seed=s.seed, script_steps=len(s.script), stopped=s.stopped, log=s.log) for s in sess])


def run_cell(engine, label: str, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    K, first, retry = CELLS[label]
    r = run_jobs(engine, list(range(first, first + K)), retry, steps, seed, progress)
    return dict(label=label, arm="S", K=K, first_episode=first, retry=retry, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(),
                retry_rule=dict(feedback=FEEDBACK, degree_threshold=DEGREE_THRESHOLD, max_retries=MAX_RETRIES), **r)


def retry_probe(engine, seed: int = SEED) -> dict:
    """One first request and, if the rule fires, one second: the exact-state prompt of step 12 of job 19, rebuilt with the oracle's own replies for steps 1 to 11, and the reply the third run's
    engine gave there (`ADD[(3,0)->(3,0):4tworowsdown4col#4]`, refused by the gate: the edge's end is not in the graph). Returns what the engine answers when told why."""
    (jseed, step), first_text = next(iter(((k, v) for k, v in third_run_slips().items() if k[0] == seed * 1000 + 19)))
    s = RetrySession(jseed, STEPS)
    while s.t < step - 1:
        rq = s.request(); s.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
    rq = s.request(); info = s.read_reply(first_text); assert info["fires"], info
    rq2 = s.retry_request(rq, first_text); reps, secs = engine.generate_round([rq2]); text = reps[0].text
    d = strict_parse(text); ok = d is not None and check(s.live, d).ok
    return dict(job=19, step=step, kind=s.script[s.t]["kind"], oracle=rq.stand_in_reply, first=first_text, why=info["reason"], feedback=rq2.messages[-1]["content"], reply=text, parsed=d is not None,
                admitted=ok, n_new=reps[0].n_new, seconds=secs)


# ---------------------------------------------------------------- summaries
def jobs_exact(c: dict) -> List[int]:
    """For every job of a cell, the number of steps after which the held state was exactly the oracle's."""
    return [sum(1 for x in s["log"] if x["exact"]) for s in c["sessions"]]


def replies_of(c: dict) -> List[Tuple[bool, int, Optional[int]]]:
    """(parsed, new tokens, cached tokens) of every request of a cell: the first request of every step and the second where there was one."""
    out = []
    for s in c["sessions"]:
        for x in s["log"]:
            if x.get("retried"): out.append((x["first_parsed"], x["first_n_new"], x["first_n_cached"]))
            out.append((x["parsed"], x["n_new"], x["n_cached"]))
    return out


def summarize_cell3(c: dict) -> dict:
    s = C.summarize_cell2(c); steps = [x for ss in c["sessions"] for x in ss["log"]]; n = len(steps); asked = [x for x in steps if x.get("retried")]
    rr = [r for r in c["rounds"] if r.get("kind") == "retry"]; allrep = replies_of(c)
    s.update(label=c["label"], retry=c["retry"], first_episode=c["first_episode"], retries=len(asked), retry_share=(len(asked) / n if n else None), retry_rounds=len(rr), retry_seconds=sum(r["seconds"] for r in rr),
             retry_exact=sum(1 for x in asked if x["exact"]), retry_admitted=sum(1 for x in asked if x["admitted"]), requests=len(allrep), unparsed_all=sum(1 for p, _, _ in allrep if not p),
             truncated_all=sum(1 for _, k, _ in allrep if k >= MAX_NEW), jobs_exact_to_end=sum(1 for s_ in c["sessions"] if all(x["exact"] for x in s_["log"]) and len(s_["log"]) == s_["script_steps"]))
    return s


def summarize_all(cells: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cells:
        assert c["label"] not in out, f"cell {c['label']} twice"; out[c["label"]] = summarize_cell3(c)
    return out


# ---------------------------------------------------------------- merge
def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pinned_hashes() -> dict:
    """sha256 of the files this test imports and does not edit."""
    import axi.engine.gate as G, axi.engine.diagnostics as D, axi.experiments.agent_loop as A, axi.experiments.cascade as K
    return dict(provider_side=PS.fingerprint(), constrained=C.fingerprint(), bounded=B.fingerprint(), gate=sha256_file(G.__file__), diagnostics=sha256_file(D.__file__), agent_loop=sha256_file(A.__file__), cascade=sha256_file(K.__file__))


def cells_planned() -> List[str]:
    """In the order they are run: the three small cells, then the held-out pair, without the retry first. About 15 minutes of GPU in all."""
    return list(ORDER)


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings (the constraint and the retry rule included), a cell twice, a cell that is not one of this test's, a cell whose
    settings are not those of its label, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"), json.dumps(c.get("retry_rule"), sort_keys=True))
    seen = set()
    for c in cells:
        lab = c.get("label")
        if lab not in CELLS: raise ValueError(f"cell {lab!r} is not one of this test's cells")
        if (c["K"], c["first_episode"], c["retry"]) != CELLS[lab]: raise ValueError(f"cell {lab} was not run with the settings of its label: {(c['K'], c['first_episode'], c['retry'])} against {CELLS[lab]}")
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {lab} was not run with this test's constraint")
        if c.get("retry_rule") != dict(feedback=FEEDBACK, degree_threshold=DEGREE_THRESHOLD, max_retries=MAX_RETRIES): raise ValueError(f"cell {lab} was not run with this test's retry rule: {c.get('retry_rule')}")
        if key(c) != key(ref): raise ValueError(f"cell {lab} was run under other settings: {key(c)} against {key(ref)}")
        if lab in seen: raise ValueError(f"cell {lab} twice")
        seen.add(lab)
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP,
                          retry_rule=ref["retry_rule"], cells={k: list(CELLS[k]) for k in ORDER}), cells=sorted(cells, key=lambda c: ORDER.index(c["label"])))


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


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_RETRY_TEST.md before the run)
def first_text_of(x: dict) -> str:
    return x["first_text"] if x.get("retried") else x["text"]


def same_first_replies(new: dict, base: dict) -> bool:
    """Job by job: the first reply of every step, up to and including the first step at which the job asked again, is the same text as `base`'s reply of the same job at the same step; a job that
    never asked again is the same throughout (and has the same number of steps)."""
    if len(new["sessions"]) != len(base["sessions"]): return False
    for a, b in zip(new["sessions"], base["sessions"]):
        asked = any(x.get("retried") for x in a["log"])
        if not asked and len(a["log"]) != len(b["log"]): return False
        for x, y in zip(a["log"], b["log"]):
            if first_text_of(x) != y["text"]: return False
            if x.get("retried"): break
    return True


def verdict_retry(new_cells: List[dict], third_cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `new_cells` are the cells of this run, `third_cells` those of the third run (`S` at 8, 32 and 64 jobs). A line that needs a cell that was not
    run is None, never 'no'."""
    new = {c["label"]: c for c in new_cells if any(s["log"] for s in c["sessions"])}; third = {c["K"]: c for c in third_cells if c["arm"] == "S"}
    Sn = {k: summarize_cell3(c) for k, c in new.items()}; St = {k: C.summarize_cell2(c) for k, c in third.items()}
    with_retry = [k for k in new if CELLS[k][2]]; small = [k for k in new if k in THIRD_K and THIRD_K[k] in third]
    out: List[Optional[bool]] = []
    out.append(all(c is not None for k in new for _, _, c in replies_of(new[k])) if new else None)                                                                                          # 1
    allrep = [r for k in new for r in replies_of(new[k])]
    out.append((sum(1 for p, _, _ in allrep if not p) <= (1 - R_PARSED) * len(allrep) + 1e-9 and all(k_ < MAX_NEW for _, k_, _ in allrep)) if allrep else None)                              # 2
    g3 = [same_first_replies(new[k], third[THIRD_K[k]]) for k in small]; g3 += [same_first_replies(new["H1"], new["H0"])] if "H0" in new and "H1" in new else []
    out.append(all(g3) if g3 else None)                                                                                                                                                     # 3
    asked = [x for k in new for s in new[k]["sessions"] for x in s["log"] if x.get("retried")]
    out.append((all(x["first_parsed"] and not x["first_admitted"] and not x["first_repairable"] for x in asked) and all(Sn[k]["retries"] == 0 for k in new if not CELLS[k][2])) if new else None)       # 4
    out.append(all(_le(Sn[k]["retry_share"], R_MAX_SHARE) for k in with_retry) if with_retry else None)                                                                                     # 5
    pairs = [(Sn[k]["exact_share"], St[THIRD_K[k]]["exact_share"]) for k in small]; out.append(all(a > b + 1e-12 for a, b in pairs) if pairs else None)                                       # 6
    out.append(None if "R64" not in Sn else _ge(Sn["R64"]["exact_share"], R_HIGH))                                                                                                          # 7
    out.append(None if "R64" not in Sn else _ge(Sn["R64"]["exact_share"], R_VERY_HIGH))                                                                                                     # 8
    held = "H0" in Sn and "H1" in Sn
    out.append(None if not held else _ge(Sn["H1"]["exact_share"], Sn["H0"]["exact_share"] + R_HELD_GAIN))                                                                                    # 9
    out.append(None if not held else _le(Sn["H1"]["seconds_per_step"], Sn["H0"]["seconds_per_step"] * (1 + R_TIME_TOL)))                                                                     # 10
    out.append(None if not held else Sn["H1"]["exact_per_hour"] > Sn["H0"]["exact_per_hour"])                                                                                               # 11
    out.append(None if not held else all(a >= b for a, b in zip(jobs_exact(new["H1"]), jobs_exact(new["H0"]))))                                                                             # 12
    assert len(out) == len(RL_LINES)
    return [(i + 1, RL_LINES[i][0], RL_LINES[i][1], out[i]) for i in range(len(RL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim, from the lines: with lines 2, 3 and 4 yes (the constraint held, the first replies are the earlier run's, the retry fired only where the rule says) the figures
    of the cells are claimed as read. That the retry raises the exactness is claimed only from the jobs no earlier run has seen (lines 9 and 12); lines 6 to 8 are on the jobs on which the slips were read.
    A statement about time or cost comes only from the pair in one session (lines 10 and 11), never from a cell against the third run's (another Colab session). A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = ok[2] is True and ok[3] is True and ok[4] is True
    return dict(base=base, constraint_held=ok[2], first_replies_as_before=ok[3], fired_by_the_rule=ok[4],
                same_jobs=({i: ok[i] for i in (6, 7, 8)} if base else {i: None for i in (6, 7, 8)}),
                new_jobs_exactness=({i: ok[i] for i in (9, 12)} if base else {9: None, 12: None}),
                time_and_cost_in_one_session=({i: ok[i] for i in (10, 11)} if base else {10: None, 11: None}))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[str, dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    print(f"\n{'cell':4s} {'K':>3s} {'jobs':>9s} {'retry':>5s} | {'steps':>5s} {'seconds':>8s} {'s/step':>7s} | {'asked':>5s} {'rounds':>6s} {'retry s':>7s} {'fixed':>5s} | {'exact':>6s} {'jobs ok':>7s} {'unparsed':>8s} {'exact/h':>8s}")
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; K, first, rt = CELLS[lab]
        print(f"{lab:4s} {K:3d} {str(first) + '-' + str(first + K - 1):>9s} {'on' if rt else 'off':>5s} | {s['steps_done']:5d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 3):>7s} | {s['retries']:5d} {s['retry_rounds']:6d} {f(s['retry_seconds'], 1):>7s} {s['retry_exact']:5d} | "
              f"{pc(s['exact_share']):>6s} {s['jobs_exact_to_end']:7d} {s['unparsed_all']:8d} {f(s['exact_per_hour'], 0):>8s}")


def print_verdict(checks, title="provider-side test, one retry") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {RL_HEADLINE[0]} and {RL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in RL_HEADLINE) + ")")
    a = claims_allowed(checks)
    if not a["base"]: print("  rule on the page: line 2, 3 or 4 did not read yes, so NO claim is made from this run until the cause is found.")
    else:
        print("  rule on the page: lines 2, 3 and 4 read yes, so the figures of the cells are claimed as read. That the retry raises the exactness rests on lines 9 and 12 (new jobs) only; lines 6 to 8 are on the jobs the slips were read on.")
        print("  a statement about time or cost comes only from the pair in one session (lines 10 and 11), never from a cell against the third run's.")
        print("  never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about more than one retry.")


def dry_run(steps: int = STEPS, labels=None) -> Dict[str, dict]:
    """Every planned cell on the stand-in, with the third run's four slips written by the test double and the oracle's own reply on the second request. Prints the table. NOT a result."""
    eng = FaultyStandIn(faults=third_run_slips(), capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for lab in ORDER:
        if labels is None or lab in labels: cells.append(run_cell(eng, lab, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache; the third run's four slips are written by the test double and the second request is answered with the oracle's own reply), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, one retry (docs/PROVIDER_SIDE_RETRY_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_retry.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_verdict(verdict_retry(m["cells"], load_third_run()["cells"])); print("\nwritten to", a.out)
    else: ap.print_help()

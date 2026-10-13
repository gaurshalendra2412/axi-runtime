"""Million-token test on a real model (concept-map item 93; rows 3, 9, 27). Page: docs/MILLION_TOKEN_TEST.md, written and hash-locked BEFORE any run.

The question: does a real model, shown the exact held state at every step and wrapped by scope_m7, stay on the oracle's state over a job of about one million tokens (3,600 steps of
about 330 tokens), or does something the offline stand-in could not show (a slip that stays, a state that grows, a step that gets slower) turn up? `endurance.py` showed that the code
between a reply and the held state survives 1.6 million steps of damaged replies; it called no model. This test calls one.

What is run. ONE episode of 3,600 requests (`LiveStream`, seed 20261021; a second, optional, on seed 20261022). The requests are made on the ORACLE's graph (so they are meaningful there and
the same for every policy), the policy under test (scope_m7) holds its own graph, shows THAT graph to the model at every step (fresh context: system, exact state, request), and updates it
from the reply with its own rule (`cascade.next_state`). No cache is used: every step is a real call, so every token counted is a token sent. After every step the file on disk holds the step
(a dropped Colab session costs the unfinished step only): running the same call again replays the saved replies through the same stream and policy, checks that every saved flag is reproduced
(`ReplayMismatch` otherwise) and carries on. A short contrast (blind apply, the gate and repair_m6 on the first 300 requests of the same stream) shows what the same task does to the policies
that do not have the scope filter, on this model.

Honest limits are in the page. In short: one model (Qwen2.5-7B-Instruct in 4-bit), greedy decoding, grammar-constrained output, synthetic graphs of at most 14 services and 26 edges, one
episode per seed; a graph that never grows does not test a state that must; the tokens are counted by the model's own tokenizer; the history-kept side at this length is NOT run (the GPU
runs out of memory near 5,500 tokens).

The same harness runs over a hosted model: `api_llm.DiskCachedLLM` has the same `generate` and `last_prompt_tokens`, plus `last_think_tokens` and `from_disk`. Tokens "sent" are the tokens billed: prompt, visible reply and
thinking tokens, of the calls that were really made (an answer that came from the saved reply file, because the same prompt had been asked before, is marked `cached` and not counted). A quota or busy-server stop
(`api_llm.QuotaStop`) leaves the file as it was after the last finished step; the same call again, on another day, carries on.

The hard test (`million_hard.py`, docs/MILLION_TOKEN_TEST_HARD.md) uses the same harness with two additions, both off by default: `stream=` names another request stream (`STREAMS`), and `reasks=` re-asks the model
(up to twice, each time with a short note appended to the request) when a reply is refused by the strict parser. A re-asked step stores the final reply in `text`, the first reply in `first_text` and the other
attempts in `reasks`; every attempt is billed, and the replay checks that only unreadable replies were re-asked."""
import json, os
from typing import Dict, List, Optional

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import apply, check
from axi.experiments.agent_loop import Task, base_messages, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.endurance import Stream
from axi.experiments.long_horizon import distance
from axi.experiments.resumable import _roundtrip, load_rows

T = CICOParser.parse_transition_delta
MODEL = "Qwen/Qwen2.5-7B-Instruct"
KINDS_LIVE = ["add_node"] * 3 + ["add_edge"] * 4 + ["delete_dep"] + ["delete_leaf"] + ["delete_edge"] + ["delete_pair"]        # chosen by simulating the ORACLE only (no model): about 10 services, 5 to 10 edges, 17 % deletes with dependants
SEEDS = (20261021, 20261022)
STEPS = 3600
BLOCKS = 10
MAIN_POLICY = "scope_m7"
CONTRAST_STEPS = 300
CONTRAST_POLICIES = ("blind", "gate", "repair_m6")
PROBE_EVERY = 100
GEMINI_MODEL = "gemini-3.5-flash-lite"           # the hosted run (docs/MILLION_TOKEN_TEST_GEMINI.md); the same stream, lines and thresholds as the Qwen run
GEMINI_TOKEN_CAP = 2_000_000                     # a guard on the bill: the run stops once this many tokens were sent (about 1.5 times the expected total)
GEMINI_FREE_DAILY = 480                          # calls a day the notebook allows itself on the free tier
THINKING_LIMIT = 20                              # thinking tokens a step above which the 20-step check advises thinking level 'minimal'


class ReplayMismatch(Exception):
    """The saved replies do not reproduce the saved flags (another stream, another code, or a damaged file)."""


class LiveStream(Stream):
    """`endurance.Stream` (never stored, same picker and caps) with the request kinds of this test."""
    def __init__(self, seed: int):
        super().__init__(seed, "steady"); self.kinds_all = list(KINDS_LIVE)


STREAMS = {"LiveStream": LiveStream}                       # name -> class; `million_hard` adds "HardStream"
REASK_NOTES = ("Your previous reply could not be read by the parser. Reply again with only the change, in the required format, and nothing else.",
               "Your previous two replies could not be read by the parser. Reply with only the change, in exactly the required format (no explanation, no code fences).")


def make_stream(name: str, seed: int):
    if name not in STREAMS and name == "HardStream": import axi.experiments.million_hard          # registers itself
    if name not in STREAMS: raise ValueError(f"unknown request stream {name!r}")
    return STREAMS[name](seed)


def reask_messages(messages: list, k: int) -> list:
    """The prompt of re-ask number k (1 or 2): the same system text and request, with a short note after the request. The note differs from one re-ask to the next, so the saved-reply file never returns the refused reply."""
    out = [dict(m) for m in messages]; out[-1]["content"] = out[-1]["content"] + "\n\n" + REASK_NOTES[k - 1]; return out


# ---------------------------------------------------------------- one step
def step_fields(policy: str, live, st: dict, ob, oa, d, degree_threshold: int = 4):
    """What one step did, from the held state `live`, the request `st`, the oracle's state before (`ob`) and after (`oa`) and the parsed reply `d` (None if the strict parser refused it).
    Returns (the new held state, the flags). `applicable`: the ideal change can be applied as written to the HELD state; `wrong`: it was applicable and the new state is not the held state plus
    that change. The flags are a function of the saved reply alone, so a replay reproduces them."""
    after = next_state(policy, live, d, st["instruction"], degree_threshold=degree_threshold)
    ideal = T(st["ideal"]); applicable = bool(check(live, ideal).ok); wrong = None
    if applicable:
        want = copy_graph(live); apply(want, ideal); wrong = not same_graph(after, want)
    f = dict(parsed=d is not None, corrupted=not after.is_well_formed(), distance=distance(after, oa), exact=same_graph(after, oa), start_same=same_graph(live, ob),
             applicable=applicable, wrong=wrong, nodes=len(after.nodes), edges=len(after.edges), oracle_nodes=len(oa.nodes), oracle_edges=len(oa.edges))
    if policy == "scope_m7" and d is not None:
        from axi.engine.scope import admit_scoped
        f["dropped"] = len(admit_scoped(live, d, st["instruction"], "named", degree_threshold).dropped)          # items of the reply the scope filter removed at this step
    return after, f


def _ask_problems(rec: dict, reasks: int) -> list:
    """What is wrong with the way a step was re-asked, from the saved record alone: a reply that was readable was re-asked, a refused one was not (when re-asking was on), the counts disagree, or too many re-asks."""
    extra = rec.get("reasks") or []; asks = rec.get("asks", 1); out = []
    if asks != 1 + len(extra) or len(extra) > reasks or (extra and "first_text" not in rec): return ["asks"]
    texts = ([rec["first_text"]] + [x["text"] for x in extra[:-1]]) if extra else []
    if any(strict_parse(t) is not None for t in texts): out.append("a readable reply was re-asked")
    if extra and extra[-1]["text"] != rec["text"]: out.append("the last re-ask is not the reply used")
    if not extra and reasks > 0 and strict_parse(rec["text"]) is None: out.append("a refused reply was not re-asked")
    return out


def replay(rows: list, seed: int, policy: str, strict: bool = True, stream: str = "LiveStream", reasks: int = 0, degree_threshold: int = 4):
    """Run the stream and the policy over the saved replies. Returns (stream, held state after the last saved step, list of step numbers whose flags differ from the saved ones).
    With strict=True the first difference raises ReplayMismatch. `reasks`: how many re-asks the run allowed (0 for the runs of the first two pages); the saved re-asks are checked against it."""
    stream = make_stream(stream, seed); live = copy_graph(stream.start); bad = []
    for rec in rows:
        out = stream.next()
        if out is None: raise ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = stream.live
        if st["kind"] != rec["kind"]:
            if strict: raise ReplayMismatch(f"step {rec['t']}: request kind {st['kind']} against saved {rec['kind']}")
            bad.append(rec["t"])
        after, f = step_fields(policy, live, st, ob, oa, strict_parse(rec["text"]), degree_threshold)
        diff = [k for k, v in f.items() if rec.get(k) != v] + _ask_problems(rec, reasks)
        if diff:
            if strict: raise ReplayMismatch(f"step {rec['t']}: flags {diff} differ from the saved ones")
            bad.append(rec["t"])
        live = after
    return stream, live, bad


# ---------------------------------------------------------------- the run
def run_million(llm, seed: int, steps: int, policy: str, path: str, meta=None, hint_rules: bool = False, progress=None, probe=None, probe_every: int = PROBE_EVERY, max_tokens: Optional[int] = None, stream: str = "LiveStream", reasks: int = 0, degree_threshold: int = 4, scripted: int = 0) -> dict:
    """One episode of `steps` requests. `llm.generate(messages, constrained=True)` returns (text, output tokens, seconds); `llm.last_prompt_tokens` is read after each call. One JSON line per finished
    step after a header line; the same call again carries on from the lines it finds. `progress(t, steps, record)` after each step; `probe()` (a dict) is stored in every `probe_every`-th record.
    `max_tokens`: stop (stopped = "token_cap") as soon as the tokens sent reach it, a guard on the bill; the file is complete up to that step and the same call without the cap carries on.
    `stream`: the name of the request stream (`STREAMS`). `reasks` (0, 1 or 2): when the strict parser refuses a reply, ask again up to this many times (`reask_messages`); the header records it only when it is above 0,
    so the files of the first two pages keep the header they had. `degree_threshold`: how many dependants a deleted service may have for the repair to delete them with it (the repair keeps a service with more; 4 is the framework's
    default and is not written to the header; the hard test uses 8).
    `scripted`: the first this many steps are not asked to the model at all: the reply is the ideal one, marked as an answer from the saved file (so nothing is counted as sent), so that a short check can start on a graph
    that has already grown (the header records it only when above 0, and `million_hard.registered` refuses such a file as a registered run)."""
    if reasks not in (0, 1, 2): raise ValueError("reasks must be 0, 1 or 2")
    if not 0 <= scripted <= steps: raise ValueError("scripted must be between 0 and the number of steps")
    meta = dict(meta or {}, seed=seed, steps=steps, policy=policy, hint_rules=bool(hint_rules), kinds=list(make_stream(stream, seed).kinds_all), stream=stream)
    if reasks: meta["reasks"] = reasks
    if scripted: meta["scripted"] = scripted
    if degree_threshold != 4: meta["degree_threshold"] = degree_threshold
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} steps, more than the {steps} asked for")
    stream, live, _ = replay(rows, seed, policy, strict=True, stream=stream, reasks=reasks, degree_threshold=degree_threshold); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live
            task = Task(st["kind"], copy_graph(live), st["instruction"], st["ideal"], oa)
            if t < scripted: text, secs = st["ideal"], 0.0; first = dict(text=text, tokens=0, think=0, cached=True, prompt_tokens=0); extra = []
            else:
                msgs = base_messages(task, hint_rules); text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec)
                part = lambda tx, nt: dict(text=tx, tokens=int(nt), think=int(getattr(llm, "last_think_tokens", 0) or 0), cached=bool(getattr(llm, "from_disk", False)), prompt_tokens=getattr(llm, "last_prompt_tokens", None))
                first = part(text, ntok); extra = []
                while strict_parse(text) is None and len(extra) < reasks:
                    text, ntok, sec = llm.generate(reask_messages(msgs, len(extra) + 1), constrained=True); secs += float(sec); extra.append(part(text, ntok))
            after, fl = step_fields(policy, live, st, ob, oa, strict_parse(text), degree_threshold)
            rec = dict(t=t + 1, kind=st["kind"], text=text, tokens=first["tokens"], think=first["think"], cached=first["cached"], prompt_tokens=first["prompt_tokens"], seconds=round(secs, 4), **fl)
            if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
            if probe is not None and (t + 1) % probe_every == 0: rec["probe"] = probe()
            rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); live = after; spent += _billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize_million(rows), meta=_roundtrip(meta), stopped=stopped)


def gpu_probe() -> dict:
    """Memory held by the model process, after freeing what is not in use. For the notebook; the tests use a stand-in."""
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return dict(allocated_mb=round(torch.cuda.memory_allocated() / 1e6, 1), reserved_mb=round(torch.cuda.memory_reserved() / 1e6, 1))


# ---------------------------------------------------------------- numbers
def _mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else None


def block(rows: list, i: int, blocks: int = BLOCKS) -> list:
    n = len(rows); return rows[i * n // blocks:(i + 1) * n // blocks]


def wrong_rate(rows: list) -> tuple:
    """(steps that did not do what was asked, steps counted): among steps that start from the oracle's state and ask for a change that can be applied to it."""
    sel = [r for r in rows if r["start_same"] and r["applicable"]]
    return sum(1 for r in sel if r["wrong"]), len(sel)


def _parts(r: dict) -> list:
    """The calls of one step, first ask first: prompt tokens, visible reply tokens, thinking tokens and whether the answer came from the saved reply file. A step that was re-asked has more than one."""
    one = lambda x: dict(prompt_tokens=x["prompt_tokens"] or 0, tokens=x["tokens"] or 0, think=x.get("think") or 0, cached=bool(x.get("cached")))
    return [one(r)] + [one(x) for x in r.get("reasks", [])]


def _billed(r: dict) -> int:
    """Tokens of one step that were really sent: prompt, visible reply and thinking tokens of every call of the step; nothing for an answer that came from the saved reply file."""
    return sum(0 if p["cached"] else p["prompt_tokens"] + p["tokens"] + p["think"] for p in _parts(r))


def tokens_sent(rows: list) -> int:
    return sum(_billed(r) for r in rows)


def summarize_million(rows: list, blocks: int = BLOCKS) -> dict:
    n = len(rows); tenth = []
    for i in range(blocks):
        b = block(rows, i, blocks)
        tenth.append(dict(first=b[0]["t"] if b else None, last=b[-1]["t"] if b else None, steps=len(b), distance=_mean(r["distance"] for r in b), exact=_mean(r["exact"] for r in b),
                          prompt_tokens=_mean(r["prompt_tokens"] or 0 for r in b), seconds=_mean(r["seconds"] for r in b), refused=_mean(not r["parsed"] for r in b),
                          wrong=wrong_rate(b), nodes=_mean(r["nodes"] for r in b), edges=_mean(r["edges"] for r in b), dropped=_mean(r.get("dropped", 0) for r in b)))
    live = [p for r in rows for p in _parts(r) if not p["cached"]]
    return dict(steps=n, tokens=tokens_sent(rows), prompt_tokens=sum(p["prompt_tokens"] for p in live), output_tokens=sum(p["tokens"] for p in live), think_tokens=sum(p["think"] for p in live),
                cached_steps=sum(1 for r in rows if all(p["cached"] for p in _parts(r))), reasked_steps=sum(1 for r in rows if r.get("asks", 1) > 1), calls=sum(len(_parts(r)) for r in rows),
                seconds=sum(r["seconds"] for r in rows), live_seconds=sum(r["seconds"] for r in rows if not all(p["cached"] for p in _parts(r))), refused=sum(1 for r in rows if not r["parsed"]), corrupted=sum(1 for r in rows if r["corrupted"]),
                exact=sum(1 for r in rows if r["exact"]), max_distance=max((r["distance"] for r in rows), default=0), max_nodes=max((r["nodes"] for r in rows), default=0),
                max_edges=max((r["edges"] for r in rows), default=0), tenths=tenth, halves=[wrong_rate(rows[:n // 2]), wrong_rate(rows[n // 2:])],
                probes=[(r["t"], r["probe"]) for r in rows if "probe" in r])


# ---------------------------------------------------------------- reading rules (fixed in the page before the run)
L_FINISH = "run 1 reached step 3,600 with every step on file, and the replay of its saved replies reproduces every saved flag of every step"
L_CORRUPT = "scope_m7 was never corrupted (no edge pointing at a missing service) at any step of run 1"
L_REFUSED = "the strict parser refused at most 2 percent of the replies of run 1"
L_BLIND = "blind apply, on the same requests, was corrupted on at least 50 percent of its first 300 steps"
L_TOKENS = "tokens sent in run 1 (prompt plus reply, counted by the model's own tokenizer) are at least 1,000,000"
L_DRIFT = "the mean distance to the oracle in the last tenth of run 1 is at most 1.0 above that of the first tenth"
L_BOUND = "the distance to the oracle never exceeded 10 items at any step of run 1"
L_EXACT = "at least 80 percent of the steps of the last tenth of run 1 hold exactly the oracle's state"
L_WORSE = "among the steps of run 1 that start from the oracle's state and ask for a change that can be applied, the share that did not do what was asked is in the second half at most 2 percentage points above that of the first half"
L_TOKFLAT = "the mean prompt tokens of the last tenth of run 1 are at most 1.5 times those of the first tenth"
L_TIMEFLAT = "the mean seconds a step takes in the last tenth of run 1 are at most 1.5 times those of the first tenth"
L_CONTRAST = "in steps 201 to 300, scope_m7's mean distance to the oracle is smaller than blind apply's and smaller than the gate's, on the same requests"
L_TWO = "run 2 (the second seed) also reached step 3,600, and lines 2, 3 and 6 to 11 also hold for it, so that the two runs together sent at least 2,000,000 tokens (only if run 2 was made)"
LINES_M = [("guard", L_FINISH), ("guard", L_CORRUPT), ("instrument", L_REFUSED), ("instrument", L_BLIND), ("claim", L_TOKENS), ("claim", L_DRIFT), ("claim", L_BOUND), ("claim", L_EXACT),
           ("claim", L_WORSE), ("claim", L_TOKFLAT), ("claim", L_TIMEFLAT), ("claim", L_CONTRAST), ("claim", L_TWO)]
LIMIT_DISTANCE, REFUSED_MAX, DRIFT_ADD, EXACT_MIN, WORSE_ADD, FLAT_TOKENS, FLAT_SECONDS = 10, 0.02, 1.0, 0.8, 0.02, 1.5, 1.5
TOKENS_MIN, TOKENS_MIN_TWO, BLIND_MIN = 1_000_000, 2_000_000, 0.5


def _run_lines(run: dict) -> dict:
    """The lines that are about one run, as booleans keyed by line number (1, 2, 3, 5 to 11)."""
    rows, meta = run["rows"], run["meta"]; n = len(rows); steps = meta["steps"]
    if n == 0: return {k: False for k in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)}
    _, _, bad = replay(rows, meta["seed"], meta["policy"], strict=False, stream=meta.get("stream", "LiveStream"), reasks=meta.get("reasks", 0), degree_threshold=meta.get("degree_threshold", 4))
    first, last = block(rows, 0), block(rows, BLOCKS - 1)
    (w1, d1), (w2, d2) = wrong_rate(rows[:n // 2]), wrong_rate(rows[n // 2:])
    pf, pl = _mean(r["prompt_tokens"] or 0 for r in first), _mean(r["prompt_tokens"] or 0 for r in last)
    sf, sl = _mean(r["seconds"] for r in first), _mean(r["seconds"] for r in last)
    return {1: n == steps and [r["t"] for r in rows] == list(range(1, steps + 1)) and not bad,
            2: all(not r["corrupted"] for r in rows),
            3: sum(1 for r in rows if not r["parsed"]) <= REFUSED_MAX * n,
            5: tokens_sent(rows) >= TOKENS_MIN,
            6: _mean(r["distance"] for r in last) <= _mean(r["distance"] for r in first) + DRIFT_ADD,
            7: max(r["distance"] for r in rows) <= LIMIT_DISTANCE,
            8: _mean(r["exact"] for r in last) >= EXACT_MIN,
            9: d1 > 0 and d2 > 0 and w2 / d2 <= w1 / d1 + WORSE_ADD,
            10: pf > 0 and pl <= FLAT_TOKENS * pf,
            11: sf > 0 and sl <= FLAT_SECONDS * sf}


def verdict_million(run1: dict, contrast: Optional[Dict[str, dict]] = None, run2: Optional[dict] = None) -> List[tuple]:
    """(tag, line, answer) for the 13 lines; answer is 'yes', 'NO' or 'not run' (line 4 and 12 need the contrast, line 13 needs run 2)."""
    a = _run_lines(run1); out = {k: ("yes" if v else "NO") for k, v in a.items()}
    full = contrast is not None and all(p in contrast and len(contrast[p]["rows"]) == CONTRAST_STEPS for p in CONTRAST_POLICIES)
    if full:
        b = contrast["blind"]["rows"]; out[4] = "yes" if sum(1 for r in b if r["corrupted"]) >= BLIND_MIN * len(b) else "NO"
        s = _mean(r["distance"] for r in run1["rows"][200:300]); g = _mean(r["distance"] for r in contrast["gate"]["rows"][200:300]); bl = _mean(r["distance"] for r in b[200:300])
        out[12] = "yes" if len(run1["rows"]) >= 300 and s is not None and s < g and s < bl else "NO"
    else: out[4] = out[12] = "not run"
    if run2 is None: out[13] = "not run"
    else:
        b = _run_lines(run2); ok = b[1] and all(b[k] for k in (2, 3, 6, 7, 8, 9, 10, 11)) and tokens_sent(run1["rows"]) + tokens_sent(run2["rows"]) >= TOKENS_MIN_TWO
        out[13] = "yes" if ok else "NO"
    return [(tag, text, out[i + 1]) for i, (tag, text) in enumerate(LINES_M)]


# ---------------------------------------------------------------- the short check before a hosted run
def estimate_cost(prompt_tokens: float, output_tokens: float, price_in: Optional[float], price_out: Optional[float]) -> Optional[float]:
    """Dollars for that many tokens at `price_in` and `price_out` dollars per million (the price page's numbers, typed in by the person; nothing is assumed here). `output_tokens` includes thinking tokens, which are billed as output.
    None if a price is missing."""
    if price_in is None or price_out is None: return None
    return prompt_tokens * price_in / 1e6 + output_tokens * price_out / 1e6


def smoke_report(summary: dict, steps: int = STEPS, price_in: Optional[float] = None, price_out: Optional[float] = None, think_limit: int = THINKING_LIMIT, lowest: str = "minimal", budget_usd: Optional[float] = None, minutes_limit: Optional[float] = None, token_cap: int = GEMINI_TOKEN_CAP) -> dict:
    """What a short check says about the whole run: tokens, thinking tokens and seconds a step, the projection to `steps` steps, the cost at the typed-in prices, and advice. Only steps that were really sent count.
    `lowest`: the lowest thinking level the model accepts ('minimal' for Flash Lite; 'low' for Gemini 3.8 Flash, which has no 'minimal'). `budget_usd`: a projected cost above it is advised against. `minutes_limit`: a projected time of calls above it is advised against."""
    n = summary["steps"] - summary["cached_steps"]
    if n <= 0: return dict(steps=0, advice=["no step was sent to the model in this check (all came from the saved reply file); run it with another seed"])
    per, think, sec = summary["tokens"] / n, summary["think_tokens"] / n, summary.get("live_seconds", summary["seconds"]) / n
    proj = per * steps; cost = estimate_cost(summary["prompt_tokens"] / n * steps, (summary["output_tokens"] + summary["think_tokens"]) / n * steps, price_in, price_out)
    advice = []
    if think > think_limit: advice.append(f"the model spends {think:.0f} thinking tokens a step (limit {think_limit}): thinking tokens are billed as output; set THINKING = \"{lowest}\" in the previous cell and run it and this cell again")
    if proj > token_cap: advice.append(f"the projection, {proj / 1e6:.2f} million tokens, is above the cap of {token_cap / 1e6:.1f} million: the run would stop before the end")
    if minutes_limit is not None and sec * steps / 60 > minutes_limit: advice.append(f"{steps} steps would take about {sec * steps / 3600:.1f} hours of calls (limit {minutes_limit / 60:.1f} hours): a lower thinking level makes each call faster")
    if cost is not None and budget_usd is not None and cost > budget_usd: advice.append(f"the projected cost of one run, ${cost:.2f}, is above your budget of ${budget_usd:.2f}: the run would stop at the budget before its end")
    if cost is None: advice.append("no prices were typed in, so no cost is shown: put the two prices of the AI Studio price page (dollars per million tokens) in PRICE_IN and PRICE_OUT to see it")
    return dict(steps=n, tokens_per_step=per, think_per_step=think, seconds_per_step=sec, projected_tokens=proj, projected_minutes=sec * steps / 60, projected_cost=cost, advice=advice)


# ---------------------------------------------------------------- printing
def _f(x, k=1): return "-" if x is None else f"{x:.{k}f}"


def print_million(s: dict, title: str = "run"):
    extra = (f" + {s['think_tokens']:,} thinking" if s["think_tokens"] else "") + (f"; {s['cached_steps']} steps answered from the saved reply file, not counted" if s["cached_steps"] else "") + (f"; {s['reasked_steps']} steps needed a re-ask" if s.get("reasked_steps") else "")
    print(f"\n== {title}: {s['steps']:,} steps, {s['tokens']:,} tokens sent ({s['prompt_tokens']:,} prompt + {s['output_tokens']:,} reply{extra}), {s['seconds'] / 60:.0f} model minutes ==")
    print(f"corrupted steps {s['corrupted']}, refused by the strict parser {s['refused']}, steps equal to the oracle {s['exact']:,}, largest distance {s['max_distance']}, most services held {s['max_nodes']}, most edges held {s['max_edges']}")
    print(f"{'steps':>13s} {'distance':>9s} {'exact':>7s} {'prompt tok':>11s} {'sec/step':>9s} {'wrong asks':>11s} {'refused':>8s} {'services':>9s} {'edges':>6s}")
    for t in s["tenths"]:
        w, d = t["wrong"]
        print(f"{str(t['first']) + '-' + str(t['last']):>13s} {_f(t['distance'], 2):>9s} {_f(100 * t['exact'] if t['exact'] is not None else None, 1) + '%':>7s} {_f(t['prompt_tokens'], 0):>11s} {_f(t['seconds'], 2):>9s} {f'{w}/{d}':>11s} "
              f"{_f(100 * t['refused'] if t['refused'] is not None else None, 1) + '%':>8s} {_f(t['nodes']):>9s} {_f(t['edges']):>6s}")
    (w1, d1), (w2, d2) = s["halves"]; print(f"steps that did not do what was asked (start from the oracle's state, applicable): first half {w1}/{d1}, second half {w2}/{d2}")
    if s["probes"]: print("GPU memory (allocated MB) at steps " + ", ".join(f"{t}: {p['allocated_mb']:.0f}" for t, p in s["probes"][::max(1, len(s['probes']) // 8)]))


def print_verdict_million(v: List[tuple], title: str = "MILLION-TOKEN TEST"):
    print(f"\n== reading rules, {title} ==")
    for i, (tag, text, ans) in enumerate(v, 1): print(f"{ans:7s} {i}. [{tag}] {text}")

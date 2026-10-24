"""A conversation with ambiguous changes, for the readings layer (concept-map item 125, docs/SUPERPOSITION.md, test 151). No model is called.

Written by a script from a hidden true list, like Test 23's stream, but with what Test 23 did not have: CHANGES that fit more than one item
("whatever Priya is handling: the day is now Friday" when Priya has two), and, a few steps later in most cases, a message that says which
one was meant. The speaker always meant one item; the script knows which. Every message is assumed to be read correctly (the host is
given the parsed request), so the comparison is ONLY between what the host does with an ambiguous change:

  NOTHING  today's rule: an ambiguous change applies nothing (safe; the change is lost), a clarification cannot bring it back.
  GUESS    what a blind application does: the change goes to the first item that fits.
  HOLD     the readings layer (engine/superpose.py): every reading is kept; a clarification keeps the one meant; a question is answered
           only when every reading gives the same answer, else "not settled yet"; a reading not clarified within 8 steps falls back to
           today's rule (applied nowhere); above 9 readings it applies nothing, as today.

Every step is marked by an exact match with the script's true list. The stream is ours, written for this comparison, not a real workload."""
from __future__ import annotations
import random
from typing import Dict, List, Optional

from axi.engine.superpose import Readings
from axi.experiments import gemini_fuzzy as F

FIELDS3 = ("day", "amount", "status")
STEPS, ITEMS, OWNERS_USED, CAP, WINDOW = 400, 12, 5, 9, 8
WEIGHTS = (65, 5, 5, 25)                     # set, owner, amb, query: one step in twenty is an ambiguous change
SEEDS = (1, 2, 3, 4, 5)
P_CLARIFY = 0.8


def _val(rng, f): return {"day": lambda: rng.choice(F.DAYS), "amount": lambda: rng.choice(F.AMOUNTS), "status": lambda: rng.choice(F.STATUSES)}[f]()


def make_stream(seed: int, steps: int = STEPS) -> dict:
    """{'start': list, 'steps': [...]} where a step is one of: set (named), owner (named), amb (by owner, with the hidden item), clarify, query."""
    rng = random.Random(f"amb:{seed}"); names = rng.sample(F.NAMES, ITEMS); owners = rng.sample(F.OWNERS, OWNERS_USED)
    start = {n: {"owner": owners[i % OWNERS_USED], "day": rng.choice(F.DAYS), "amount": rng.choice(F.AMOUNTS), "status": rng.choice(F.STATUSES)} for i, n in enumerate(names)}
    truth = {n: dict(v) for n, v in start.items()}; out, due = [], {}
    for t in range(1, steps + 1):
        if t in due:
            k, item, o = due.pop(t); out.append(dict(t=t, kind="clarify", of=k, item=item, message=f"By the one {o} is handling (step {k}) I meant {item}.")); continue
        kind = rng.choices(["set", "owner", "amb", "query"], WEIGHTS)[0]
        if kind == "amb":
            groups = {o: sorted(n for n in truth if truth[n]["owner"] == o) for o in owners}; multi = [o for o, ns in groups.items() if len(ns) >= 2]
            if not multi: kind = "set"
            else:
                o = rng.choice(sorted(multi)); item = rng.choice(groups[o]); f = rng.choice(FIELDS3); v = _val(rng, f)
                while v == truth[item][f]: v = _val(rng, f)
                truth[item][f] = v; out.append(dict(t=t, kind="amb", owner=o, field=f, value=v, item=item, message=f"Whatever {o} is handling: the {f} is now {v}."))
                if rng.random() < P_CLARIFY:
                    k = t + rng.randint(1, 6)
                    while k in due: k += 1
                    due[k] = (t, item, o)
                continue
        if kind == "set":
            n = rng.choice(sorted(truth)); f = rng.choice(FIELDS3); v = _val(rng, f)
            while v == truth[n][f]: v = _val(rng, f)
            truth[n][f] = v; out.append(dict(t=t, kind="set", name=n, field=f, value=v, message=f"{n}: the {f} is now {v}."))
        elif kind == "owner":
            n = rng.choice(sorted(truth)); v = rng.choice([o for o in owners if o != truth[n]["owner"]])
            truth[n]["owner"] = v; out.append(dict(t=t, kind="owner", name=n, field="owner", value=v, message=f"Hand {n} to {v}."))
        elif kind == "query":
            n = rng.choice(sorted(truth)); f = rng.choice(FIELDS3); out.append(dict(t=t, kind="query", name=n, field=f, answer=truth[n][f], message=f"What is the {f} of {n}?"))
    return dict(start=start, steps=out, seed=seed)


def truth_after(stream: dict) -> List[dict]:
    """The true list after every step."""
    cur = {n: dict(v) for n, v in stream["start"].items()}; out = []
    for s in stream["steps"]:
        if s["kind"] in ("set", "owner"): cur[s["name"]][s["field"]] = s["value"]
        elif s["kind"] == "amb": cur[s["item"]][s["field"]] = s["value"]
        out.append({n: dict(v) for n, v in cur.items()})
    return out


def _set(state, name, f, v):
    if name not in state: return None
    new = {n: dict(x) for n, x in state.items()}; new[name][f] = v; return new


def run(stream: dict, policy: str) -> List[dict]:
    """The host's list (or readings) after every step, with what the step did."""
    truths = truth_after(stream); rows = []
    if policy == "HOLD": R = Readings({n: dict(v) for n, v in stream["start"].items()}, cap=CAP)
    else: cur = {n: dict(v) for n, v in stream["start"].items()}
    for s, tr in zip(stream["steps"], truths):
        rec = dict(t=s["t"], kind=s["kind"])
        if policy == "HOLD":
            R.expire(s["t"], WINDOW)
            if s["kind"] in ("set", "owner"): rec["did"] = R.apply(lambda st, s=s: _set(st, s["name"], s["field"], s["value"]))
            elif s["kind"] == "amb":
                rec["did"] = R.branch(s["t"], lambda st, s=s: [(n, _set(st, n, s["field"], s["value"])) for n in sorted(st) if st[n]["owner"] == s["owner"]])
            elif s["kind"] == "clarify": rec["did"] = R.clarify(s["of"], s["item"])
            else:
                a, ok = R.ask(lambda st, s=s: st.get(s["name"], {}).get(s["field"])); rec["answer"] = a if ok else None; rec["settled_answer"] = ok
            rec["worlds"] = len(R.worlds); rec["exact"] = R.settled and R.state() == tr; rec["truth_held"] = any(x == tr for x in R.states())
        else:
            if s["kind"] in ("set", "owner"): cur = _set(cur, s["name"], s["field"], s["value"]) or cur
            elif s["kind"] == "amb":
                fits = [n for n in sorted(cur) if cur[n]["owner"] == s["owner"]]
                if policy == "GUESS" and fits: cur = _set(cur, fits[0], s["field"], s["value"]); rec["guess"] = fits[0]
            elif s["kind"] == "query": rec["answer"] = cur.get(s["name"], {}).get(s["field"]); rec["settled_answer"] = True
            rec["worlds"] = 1; rec["exact"] = cur == tr; rec["truth_held"] = rec["exact"]
        if s["kind"] == "query": rec["answer_exact"] = rec["settled_answer"] and rec["answer"] == s["answer"]
        rows.append(rec)
    if policy == "HOLD": rows[-1]["max_worlds"], rows[-1]["over_cap"], rows[-1]["contradictions"], rows[-1]["expired"] = R.max_seen, R.over_cap, R.contradictions, R.expired
    return rows


def summary(stream: dict, rows: List[dict]) -> dict:
    amb = [s for s in stream["steps"] if s["kind"] == "amb"]; clar = {s["of"] for s in stream["steps"] if s["kind"] == "clarify"}
    q = [r for r in rows if r["kind"] == "query"]
    return dict(steps=len(rows), exact=sum(r["exact"] for r in rows), truth_held=sum(r["truth_held"] for r in rows), end_exact=rows[-1]["exact"], ambiguous=len(amb),
                clarified=len(clar), questions=len(q), answers_exact=sum(r["answer_exact"] for r in q), answers_wrong=sum(1 for r in q if r["settled_answer"] and not r["answer_exact"]),
                not_settled=sum(1 for r in q if not r["settled_answer"]), max_worlds=max(r["worlds"] for r in rows), over_cap=rows[-1].get("over_cap", 0))


def compare(seeds=SEEDS) -> Dict[str, dict]:
    """The three policies summed over the seeds."""
    tot: Dict[str, dict] = {}
    for seed in seeds:
        st = make_stream(seed)
        for p in ("NOTHING", "GUESS", "HOLD"):
            s = summary(st, run(st, p)); acc = tot.setdefault(p, {k: 0 for k in s})
            for k, v in s.items(): acc[k] = max(acc[k], v) if k == "max_worlds" else acc[k] + v
    return tot

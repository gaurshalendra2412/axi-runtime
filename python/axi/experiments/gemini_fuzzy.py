"""The fuzzy-state test on a hosted model (docs/GEMINI_FUZZY_TEST.md, test 23): Gemini 3.8 Flash, a long conversation in plain sentences, alone and with AXI.

The state here is NOT a graph. It is a short list of items (an events desk) whose facts (owner, day, amount, status) arrive as ordinary sentences, in many wordings, with corrections ("scratch what I said"),
loose references ("the last one we added", "whatever Priya is handling", "the same day as ..."), items dropped, and a paragraph of office chatter around every message that mentions other days, other amounts and
the same first names without changing anything. Every tenth turn from the 20th is a QUESTION about the list, asked about facts of every age, about a fact never given (the right answer is `unknown`) and about a
loose reference that fits two items (the right answer is `ambiguous`).
The text is written by a script from a hidden true list, so every reply is marked by an exact match with what the script knows, never by another model's opinion.

Two arms on the SAME stream and the SAME chatter:
  K  the kept chat: system text + every turn and every reply kept and re-sent; each reply is applied as it comes (blind apply). What a firm gets when it just keeps talking to the model.
  A  with AXI: the exact list held by the framework is shown at every turn and nothing is kept in the chat; the reply goes through a check (the item must exist, the field and the value must be in the allowed
     set) before it touches the list, with up to two re-asks; a reply that still fails changes nothing.
  D  the depth probe (a control): a chat whose history is the IDEAL replies, then ten real turns, at several depths.
Measured per call as in Test 22 (prompt tokens, tokens served from the cache, visible and thinking tokens, seconds, seconds to the first token): speed, latency, tokens (the stand-in for energy), drift,
distance from the true list. It imports `gemini_long.py` (the chat backend, the stats) and `api_llm.py` and edits none of them. Nothing of the framework's core is used or changed."""
from __future__ import annotations
import json, os, random
from typing import Dict, List, Optional, Tuple

from axi.experiments import api_llm as API
from axi.experiments import gemini_long as GL
from axi.experiments.resumable import load_rows, _roundtrip

# ---------------------------------------------------------------- the settings written on the page
MODEL = GL.MODEL                                            # "gemini-3.8-flash"
STREAM = "FuzzyDesk"
SEED, SEED_PILOT = 20261122, 98
STEPS = 400                                                  # turns of arm K and steps of arm A
PAD_SENTENCES = 30                                           # sentences of office chatter around every message (the pilot prints the real token count)
QUERY_EVERY, QUERY_FROM = 10, 20                             # a question every 10th turn from the 20th
A_REASKS = 2
D_STARTS, D_ASKS = (50, 100, 200, 300), 10
K_CONTEXT_CAP = 950_000
K_TOKEN_CAP, A_TOKEN_CAP, D_TOKEN_CAP = 150_000_000, 3_000_000, 20_000_000
MAX_OUT = GL.MAX_OUT
BANDS = (0, 25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 1_000_001)
AGE_EDGES = (0, 10, 50, 150, 10 ** 9)                        # question age bands: facts last changed < 10, 10-49, 50-149, 150 or more turns earlier

OWNERS = ["Priya", "Marcus", "Elena", "Tomas", "Aiko", "Dmitri", "Fatima", "Lucas", "Nadia", "Hiro"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STATUSES = ["pending", "confirmed", "on hold", "waiting on payment"]
AMOUNTS = list(range(100, 1000, 10))
FIELDS = ("owner", "day", "amount", "status")
_ADJ = ["Amber", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Granite", "Harbor", "Indigo", "Juniper", "Kestrel", "Lantern"]
_NOUN = ["Gallery", "Pavilion", "Studio", "Terrace", "Workshop", "Atelier", "Lounge", "Market", "Garden", "Annex", "Depot", "Foyer"]
NAMES = [f"{a} {n}" for a in _ADJ for n in _NOUN]

SYSTEM = (
    "You keep a short list of items for a small events desk. Each item has a name and up to four facts: owner (a first name), day (a weekday), amount (a number of 100 to 990, digits only) and status "
    "(pending, confirmed, on hold or waiting on payment). Facts arrive one message at a time, in ordinary sentences, with a paragraph of unrelated office chatter around them. Only the line that starts "
    "with 'Message:' or 'Question:' matters; the chatter never changes the list, even when it mentions days, amounts or first names.\n"
    "Reply with ONE JSON object and nothing else.\n"
    "For a Message reply {\"ops\": [...]} where each op is one of: {\"op\": \"add\", \"name\": \"<item>\", \"fields\": {\"owner\": ..., \"day\": ..., \"amount\": ..., \"status\": ...}} (only the facts given), "
    "{\"op\": \"set\", \"name\": \"<item>\", \"field\": \"<owner|day|amount|status>\", \"value\": ...}, {\"op\": \"delete\", \"name\": \"<item>\"}. Use the item names exactly as written. A correction replaces the "
    "earlier value. A loose reference ('the last one we added', 'the one Priya is handling', 'the same day as <item>') must be turned into the item's name.\n"
    "For a Question reply {\"answer\": \"<value>\"} with the value only (a day, a first name, a status or digits). The question never changes the list. If the fact was never given, answer unknown. "
    "If more than one item fits the description, answer ambiguous.")
SYSTEM_K = SYSTEM + "\nThe list starts empty. You are not shown it: keep it from the conversation."
SYSTEM_A = SYSTEM + "\nThe current list is shown before every message and is exact."


# ---------------------------------------------------------------- values, the list, the check
_CANON = {"owner": {x.lower(): x for x in OWNERS}, "day": {x.lower(): x for x in DAYS}, "status": {x.lower(): x for x in STATUSES}}


def norm_value(field: str, v):
    """The canonical value for a field, or None if it is outside the allowed set (case and digits-as-text are cleaned, nothing else)."""
    if field not in FIELDS or isinstance(v, bool): return None
    if field == "amount":
        if isinstance(v, float) and v.is_integer(): v = int(v)
        if isinstance(v, str) and v.strip().isdigit(): v = int(v.strip())
        return v if isinstance(v, int) and v in AMOUNTS else None
    if not isinstance(v, str): return None
    return _CANON[field].get(v.strip().lower())


def show_state(state: dict) -> str:
    if not state: return "(the list is empty)"
    return "\n".join((f"{n} | " + " | ".join(f"{f}={state[n][f]}" for f in FIELDS if f in state[n])) if state[n] else f"{n} |" for n in state)      # in the order the items were added


def copy_state(s: dict) -> dict: return {n: dict(v) for n, v in s.items()}


def distance(a: dict, b: dict) -> int:
    """Cells that differ: for an item in one list only, 1 plus its facts; for an item in both, each fact that differs or is missing on one side."""
    d = 0
    for n in set(a) | set(b):
        if n not in a or n not in b: d += 1 + len(a.get(n) or b.get(n))
        else: d += sum(1 for f in set(a[n]) | set(b[n]) if a[n].get(f) != b[n].get(f))
    return d


def parse_reply(text: str) -> Optional[dict]:
    """One JSON object (a code fence around it is removed). {'ops': [...]} or {'answer': '...'}; anything else is None."""
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.split("\n"); lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        s = "\n".join(lines).strip()
    try: obj = json.loads(s)
    except (ValueError, TypeError): return None
    if not isinstance(obj, dict): return None
    if isinstance(obj.get("ops"), list) and all(isinstance(o, dict) for o in obj["ops"]): return dict(ops=obj["ops"])
    a = obj.get("answer")
    if isinstance(a, (str, int, float)) and not isinstance(a, bool):
        if isinstance(a, float) and a.is_integer(): a = int(a)
        return dict(answer=str(a).strip())
    return None


def norm_answer(x) -> str: return str(x).strip().strip(".").lower()


def check_ops(state: dict, ops: list) -> Tuple[bool, str, list]:
    """The framework's check before a reply touches the list. Returns (ok, why not, ops with canonical values). The item must exist for set and delete, must not exist for add, field and value must be allowed."""
    cur = copy_state(state); out = []
    for i, o in enumerate(ops):
        kind = o.get("op"); name = o.get("name")
        if kind == "add":
            if not isinstance(name, str) or name not in NAMES: return False, f"op {i + 1}: '{name}' is not an item name", []
            if name in cur: return False, f"op {i + 1}: '{name}' is already on the list", []
            fl = o.get("fields", {})
            if not isinstance(fl, dict): return False, f"op {i + 1}: fields must be an object", []
            clean = {}
            for f, v in fl.items():
                nv = norm_value(f, v)
                if nv is None: return False, f"op {i + 1}: {f}={v!r} is not allowed", []
                clean[f] = nv
            cur[name] = clean; out.append(dict(op="add", name=name, fields=clean))
        elif kind == "set":
            if name not in cur: return False, f"op {i + 1}: '{name}' is not on the list", []
            f = o.get("field"); nv = norm_value(f, o.get("value"))
            if nv is None: return False, f"op {i + 1}: {f}={o.get('value')!r} is not allowed", []
            cur[name][f] = nv; out.append(dict(op="set", name=name, field=f, value=nv))
        elif kind == "delete":
            if name not in cur: return False, f"op {i + 1}: '{name}' is not on the list", []
            del cur[name]; out.append(dict(op="delete", name=name))
        else: return False, f"op {i + 1}: unknown op {kind!r}", []
    return True, "", out


def apply_ops(state: dict, norm_ops: list) -> dict:
    g = copy_state(state)
    for o in norm_ops:
        if o["op"] == "add": g[o["name"]] = dict(o["fields"])
        elif o["op"] == "set": g[o["name"]][o["field"]] = o["value"]
        else: del g[o["name"]]
    return g


def blind_apply(state: dict, ops: list) -> dict:
    """What a firm does that takes each reply as it comes: no check. A set on a missing item creates it, a delete of a missing item does nothing, an odd value is stored as it is."""
    g = copy_state(state)
    for o in ops:
        if not isinstance(o, dict) or not isinstance(o.get("name"), str): continue
        n, kind = o["name"], o.get("op")
        if kind == "add": g[n] = {f: (norm_value(f, v) if norm_value(f, v) is not None else v) for f, v in (o.get("fields") or {}).items()} if isinstance(o.get("fields"), dict) else {}
        elif kind == "set" and isinstance(o.get("field"), str):
            nv = norm_value(o["field"], o.get("value")); g.setdefault(n, {})[o["field"]] = nv if nv is not None else o.get("value")
        elif kind == "delete": g.pop(n, None)
    return g


def well_formed(state: dict, introduced) -> bool:
    """No item the conversation never mentioned, no unknown field, no value outside the allowed set (the counterpart of a dangling edge)."""
    for n, fs in state.items():
        if n not in introduced or not isinstance(fs, dict): return False
        for f, v in fs.items():
            if norm_value(f, v) is None: return False
    return True


# ---------------------------------------------------------------- the chatter around every message
_SUBJ = ["the café on the corner", "Dana from accounts", "the night shuttle", "our landlord", "the printer on floor two", "Sam in facilities", "the courier", "the new intern", "the parking office",
         "the bakery downstairs", "the window cleaners", "the lift engineer", "the IT helpdesk", "Mr Okafor from legal", "the postroom"]
_TEMPL = ["{s} said it would be {day} before anything happens.", "{s} quoted {amt} for the lunch order, which nobody asked for.", "{o} mentioned that the weather was odd this morning.",
          "{s} is still {st}, as far as anyone can tell.", "{o} asked to leave early on {day}, which is none of this list's business.", "Somebody left {amt} in an envelope at the front desk for {s}.",
          "{s} will be out until {day}.", "{o} thinks the coffee machine has gone off again.", "The meeting about {s} was moved to {day}, but that is a different matter.",
          "{o} says {s} charged {amt} last time, give or take.", "Nothing is {st} here that has anything to do with {s}.", "{o} will not be in on {day} because of a dentist visit.",
          "{s} rang about a parcel and said {day} at the latest.", "The budget for the staff party is {amt}, and {o} is organising it.", "Please ignore the noise from {s}; it is {st} again."]


def chatter(seed: int, t: int, sentences: int = PAD_SENTENCES) -> str:
    """The paragraph of office chatter that comes with turn t: deterministic in (seed, t), the same text in every arm. It never contains an item name and never asks for anything."""
    rng = random.Random(f"{seed}:{t}:chat"); out = []
    for _ in range(sentences):
        out.append(rng.choice(_TEMPL).format(s=rng.choice(_SUBJ), day=rng.choice(DAYS), amt=rng.choice(AMOUNTS), o=rng.choice(OWNERS), st=rng.choice(STATUSES)))
    return " ".join(out)


def user_text(seed: int, t: int, st: dict, state: Optional[dict] = None, sentences: int = PAD_SENTENCES) -> str:
    head = ("Current list (exact, kept by the framework):\n" + show_state(state) + "\n\n") if state is not None else ""
    return f"{head}{chatter(seed, t, sentences)}\n\n{'Question' if st['kind'] == 'query' else 'Message'}: {st['message']}"


# ---------------------------------------------------------------- the stream: the hidden true list and the messages written from it
_VAL = {"owner": lambda r: r.choice(OWNERS), "day": lambda r: r.choice(DAYS), "amount": lambda r: r.choice(AMOUNTS), "status": lambda r: r.choice(STATUSES)}


def _other(rng, field, cur):
    for _ in range(40):
        v = _VAL[field](rng)
        if v != cur: return v
    return v


_FRAG = {"owner": ["{v} is handling it", "it is {v}'s job", "assign it to {v}"], "day": ["it is set for {v}", "scheduled for {v}", "the date is {v}"],
         "amount": ["the budget is {v}", "it costs {v}", "we agreed on {v}"], "status": ["status: {v}", "it is currently {v}", "right now it is {v}"]}
_INTRO = ["New item: {n}.", "Adding {n} to the list.", "We just took on {n}.", "Please track a new one called {n}."]
_SET = {"owner": ["{v} takes over {n}.", "Hand {n} to {v}.", "{n}: the owner is now {v}."], "day": ["{n} moves to {v}.", "Push {n} to {v}.", "New date for {n}: {v}."],
        "amount": ["{n} now costs {v}.", "Revised budget for {n}: {v}.", "Update the amount on {n} to {v}."], "status": ["{n} is now {v}.", "Mark {n} as {v}.", "Status of {n} changed to {v}."]}
_REFLAST = {"owner": ["The last one we added goes to {v}."], "day": ["The last item we added moves to {v}."], "amount": ["The last one we added now costs {v}."], "status": ["The last one we added is now {v}."]}
_DROP = ["Drop {n}, it is cancelled.", "Remove {n} from the list.", "{n} is off, forget it."]


class FuzzyStream:
    """Deterministic in (seed, n). `next()` returns (step, list before) and moves `live` to the list after (a question leaves it as it is). step: t, kind, message, ideal (the right reply as text), answer / qkind / age for a question."""
    def __init__(self, seed: int, n: int = STEPS):
        self.seed, self.n, self.t, self.live = seed, n, 0, {}
        self.introduced: set = set(); self.order: List[str] = []; self.last: Dict[Tuple[str, str], int] = {}; self.qcount = 0

    def next(self):
        if self.t >= self.n: return None
        self.t += 1; t = self.t; ob = copy_state(self.live); rng = random.Random(f"{self.seed}:{t}:step")
        st = self._query(t, rng) if (t >= QUERY_FROM and t % QUERY_EVERY == 0 and self.live) else None
        if st is None: st = self._update(t, rng)
        return st, ob

    # -- questions
    def _query(self, t, rng):
        q = self.qcount; self.qcount += 1; cands = [(n, f, t - self.last.get((n, f), t)) for n, fs in self.live.items() for f in fs]
        if q % 7 == 6:                                                                                    # a fact never given
            miss = [(n, f) for n in sorted(self.live) for f in FIELDS if f not in self.live[n]]
            if miss:
                n, f = rng.choice(miss); return dict(t=t, kind="query", qkind="unknown", age=None, answer="unknown", message=f"what is the {f} of {n}? It has not been given.", ideal=json.dumps(dict(answer="unknown")))
        if q % 11 == 10:                                                                                  # a reference that fits two items
            for o in rng.sample(OWNERS, len(OWNERS)):
                hit = [n for n in self.live if self.live[n].get("owner") == o]
                if len(hit) >= 2:
                    f = rng.choice(["day", "amount", "status"]); return dict(t=t, kind="query", qkind="ambiguous", age=None, answer="ambiguous", message=f"what is the {f} of the item {o} is handling?", ideal=json.dumps(dict(answer="ambiguous")))
        if not cands: return None
        band = q % 4; inb = [c for c in cands if AGE_EDGES[band] <= c[2] < AGE_EDGES[band + 1]] or sorted(cands, key=lambda c: abs(c[2] - AGE_EDGES[band]))[:3]
        n, f, age = rng.choice(inb); v = self.live[n][f]
        return dict(t=t, kind="query", qkind="recall", age=age, answer=str(v), message=f"what is the {f} of {n}?", ideal=json.dumps(dict(answer=str(v))))

    # -- updates
    def _update(self, t, rng):
        n = len(self.live)
        if n < 5: kind = "add"
        elif n > 26: kind = "delete"
        else: kind = rng.choices(["add", "set", "set2", "correct", "ref_last", "ref_owner", "ref_same", "delete"], [20, 28, 8, 8, 8, 6, 6, 10])[0]
        for _ in range(3):
            st = getattr(self, "_k_" + kind)(t, rng)
            if st is not None: return self._finish(st, t)
            kind = "set"
        return self._finish(self._k_add(t, rng), t)

    def _finish(self, st, t):
        ops = st["ops"]
        for o in ops:
            if o["op"] == "add":
                self.live[o["name"]] = dict(o["fields"]); self.introduced.add(o["name"]); self.order.append(o["name"])
                for f in o["fields"]: self.last[(o["name"], f)] = t
            elif o["op"] == "set": self.live[o["name"]][o["field"]] = o["value"]; self.last[(o["name"], o["field"])] = t
            else:
                del self.live[o["name"]]; self.order.remove(o["name"])
                for f in FIELDS: self.last.pop((o["name"], f), None)
        return dict(t=t, kind=st["kind"], message=st["message"], ideal=json.dumps(dict(ops=ops)), ops=ops)

    def _k_add(self, t, rng):
        pool = [x for x in NAMES if x not in self.live and x not in self.introduced] or [x for x in NAMES if x not in self.live]
        name = rng.choice(pool); fs = rng.sample(FIELDS, rng.randint(1, 4)); fields = {f: _VAL[f](rng) for f in fs}
        parts = [rng.choice(_FRAG[f]).format(v=fields[f]) for f in FIELDS if f in fields]
        return dict(kind="add", ops=[dict(op="add", name=name, fields={f: fields[f] for f in FIELDS if f in fields})], message=rng.choice(_INTRO).format(n=name) + " " + ", ".join(parts)[:1].upper() + ", ".join(parts)[1:] + ".")

    def _k_set(self, t, rng):
        name = rng.choice(sorted(self.live)); f = rng.choice(FIELDS); v = _other(rng, f, self.live[name].get(f))
        return dict(kind="set", ops=[dict(op="set", name=name, field=f, value=v)], message=rng.choice(_SET[f]).format(n=name, v=v))

    def _k_set2(self, t, rng):
        name = rng.choice(sorted(self.live)); f1, f2 = rng.sample(FIELDS, 2); v1 = _other(rng, f1, self.live[name].get(f1)); v2 = _other(rng, f2, self.live[name].get(f2))
        return dict(kind="set2", ops=[dict(op="set", name=name, field=f1, value=v1), dict(op="set", name=name, field=f2, value=v2)],
                    message=f"For {name}: {rng.choice(_FRAG[f1]).format(v=v1)}, and {rng.choice(_FRAG[f2]).format(v=v2)}.")

    def _k_correct(self, t, rng):
        rec = sorted((n, f) for (n, f), at in self.last.items() if at >= t - 6 and n in self.live and f in self.live[n])
        if not rec: return None
        name, f = rng.choice(rec); old = self.live[name][f]; v = _other(rng, f, old)
        return dict(kind="correct", ops=[dict(op="set", name=name, field=f, value=v)], message=f"Scratch what I said about {name}: the {f} is {v}, not {old}.")

    def _k_ref_last(self, t, rng):
        if not self.order: return None
        name = self.order[-1]; f = rng.choice(FIELDS); v = _other(rng, f, self.live[name].get(f))
        return dict(kind="ref_last", ops=[dict(op="set", name=name, field=f, value=v)], message=rng.choice(_REFLAST[f]).format(v=v))

    def _k_ref_owner(self, t, rng):
        once = [o for o in OWNERS if sum(1 for n in self.live if self.live[n].get("owner") == o) == 1]
        if not once: return None
        o = rng.choice(once); name = next(n for n in sorted(self.live) if self.live[n].get("owner") == o); f = rng.choice(["day", "amount", "status"]); v = _other(rng, f, self.live[name].get(f))
        return dict(kind="ref_owner", ops=[dict(op="set", name=name, field=f, value=v)], message=f"Whatever {o} is handling: the {f} is now {v}.")

    def _k_ref_same(self, t, rng):
        f = rng.choice(["day", "amount"]); have = sorted(n for n in self.live if f in self.live[n])
        if len(have) < 2 or len(self.live) < 2: return None
        b = rng.choice(have); a = rng.choice([x for x in sorted(self.live) if x != b]); v = self.live[b][f]
        return dict(kind="ref_same", ops=[dict(op="set", name=a, field=f, value=v)], message=f"Put {a} on the same day as {b}." if f == "day" else f"{a} costs the same as {b}.")

    def _k_delete(self, t, rng):
        name = rng.choice(sorted(self.live)); return dict(kind="delete", ops=[dict(op="delete", name=name)], message=rng.choice(_DROP).format(n=name))


def make_stream(seed: int, n: int = STEPS) -> FuzzyStream: return FuzzyStream(seed, n)


# ---------------------------------------------------------------- what a reply did
def usable(state: dict, st: dict, text: str) -> Tuple[bool, str]:
    """Is the reply one the framework would accept for this step (right shape, and for an update it passes the check on the list it holds)? Returns (ok, why not)."""
    p = parse_reply(text)
    if p is None: return False, "it is not one JSON object with ops or answer"
    if st["kind"] == "query": return ("answer" in p, "" if "answer" in p else "a question needs {\"answer\": ...}")
    if "ops" not in p: return False, "a message needs {\"ops\": [...]}"
    ok, why, _ = check_ops(state, p["ops"]); return ok, why


def reply_flags(ob: dict, oa: dict, st: dict, p: Optional[dict]) -> dict:
    """What the REPLY did against the oracle's list before the step (`ob`): was it of the right shape and (for an update) can it be applied to the true list, and does it give exactly the oracle's next list (or the right answer)."""
    if p is None: return dict(reply_ok=False, reply_exact=False)
    if st["kind"] == "query":
        if "answer" not in p: return dict(reply_ok=False, reply_exact=False)
        return dict(reply_ok=True, reply_exact=norm_answer(p["answer"]) == norm_answer(st["answer"]))
    if "ops" not in p: return dict(reply_ok=False, reply_exact=False)
    ok, _, norm = check_ops(ob, p["ops"])
    return dict(reply_ok=ok, reply_exact=bool(ok and apply_ops(ob, norm) == oa))


def row_fields(after: dict, oa: dict, ob: dict, st: dict, p: Optional[dict], introduced) -> dict:
    shape = p is not None and (("answer" in p) if st["kind"] == "query" else ("ops" in p))
    return dict(parsed=bool(shape), **reply_flags(ob, oa, st, p), corrupted=not well_formed(after, introduced), distance=distance(after, oa), exact=after == oa, nodes=len(after), oracle_nodes=len(oa),
                qkind=st.get("qkind"), age=st.get("age"))


_att, parts_of, billed, computed, tokens_sent = GL._att, GL.parts_of, GL.billed, GL.computed, GL.tokens_sent


class ReplayMismatch(Exception): pass


def replay(rows: list, seed: int, arm: str, sentences: int = PAD_SENTENCES):
    """Run the stream and the arm's policy over the saved replies (no call). Returns (stream, held list after the last saved step, chat history of arm K). Raises at the first flag that differs."""
    stream = make_stream(seed); live: dict = {}; history: List[dict] = []
    for rec in rows:
        out = stream.next()
        if out is None: raise ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = stream.live
        if st["kind"] != rec["kind"]: raise ReplayMismatch(f"step {rec['t']}: kind {st['kind']} against saved {rec['kind']}")
        p = parse_reply(rec["text"])
        if arm == "K":
            after = copy_state(live) if st["kind"] == "query" or p is None or "ops" not in p else blind_apply(live, p["ops"])
            f = row_fields(after, oa, ob, st, p, stream.introduced)
            history += [{"role": "user", "content": user_text(seed, rec["t"], st, None, sentences)}, {"role": "assistant", "content": rec["text"]}]
        else:
            ok, why = usable(live, st, rec["text"]); after = copy_state(live)
            if ok and st["kind"] != "query": after = apply_ops(live, check_ops(live, p["ops"])[2])
            f = row_fields(after, oa, ob, st, p, stream.introduced); f["blocked"] = bool(p is not None and not ok)
        diff = [k for k, v in f.items() if rec.get(k) != v] + (_ask_problems(rec, live, st) if arm == "A" else [])
        if diff: raise ReplayMismatch(f"step {rec['t']}: flags {diff} differ from the saved ones")
        live = after
    return stream, live, history


def _ask_problems(rec: dict, live: dict, st: dict) -> List[str]:
    """The re-asks of a saved step of arm A must follow the rule: ask again only while the reply is not usable, at most A_REASKS times."""
    re = rec.get("reasks", []); asks = rec.get("asks", 1); bad = []
    if asks != 1 + len(re) or len(re) > A_REASKS: bad.append("asks")
    texts = ([rec["first_text"]] if re else []) + [x["text"] for x in re[:-1]]
    if any(usable(live, st, x)[0] for x in texts): bad.append("asked again after a usable reply")
    if re and len(re) < A_REASKS and not usable(live, st, rec["text"])[0]: bad.append("stopped asking before a usable reply")
    if re and re[-1]["text"] != rec["text"]: bad.append("final text")
    return bad


# ---------------------------------------------------------------- the runs
def _system(arm: str) -> dict: return {"role": "system", "content": SYSTEM_K if arm == "K" else SYSTEM_A}


def projected_prompt(rows: list) -> Optional[int]: return GL.projected_prompt(rows)


def run_kept(llm, seed: int, steps: int, path: str, meta=None, sentences: int = PAD_SENTENCES, context_cap: int = K_CONTEXT_CAP, max_tokens: Optional[int] = None, progress=None) -> dict:
    """Arm K. One JSON line per finished turn after a header; the same call again carries on (the chat is rebuilt from the saved replies). `stopped`: 'context', 'token_cap', 'stream' or None."""
    meta = dict(meta or {}, arm="K", seed=seed, steps=steps, sentences=sentences, context_cap=context_cap, stream=STREAM, policy="blind")
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} turns, more than the {steps} asked for")
    stream, live, history = replay(rows, seed, "K", sentences); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            nxt = projected_prompt(rows)
            if nxt is not None and nxt > context_cap: stopped = "context"; break
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live; user = {"role": "user", "content": user_text(seed, t + 1, st, None, sentences)}
            text, ntok, sec = llm.generate([_system("K")] + history + [user], constrained=True); a = _att(llm, text, ntok, sec); p = parse_reply(text)
            after = copy_state(live) if st["kind"] == "query" or p is None or "ops" not in p else blind_apply(live, p["ops"])
            rec = _roundtrip(dict(t=t + 1, kind=st["kind"], **a, **row_fields(after, oa, ob, st, p, stream.introduced)))
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); history += [user, {"role": "assistant", "content": text}]; live = after; spent += billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


def reask_messages(msgs: list, text: str, why: str) -> list:
    return msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": f"That reply could not be used: {why}. Reply again with the one JSON object only."}]


def run_axi(llm, seed: int, steps: int, path: str, meta=None, sentences: int = PAD_SENTENCES, max_tokens: Optional[int] = None, progress=None, scripted: int = 0) -> dict:
    """Arm A: the exact list shown at every step with the same chatter, the check before the list is touched, up to two re-asks; a reply that still fails changes nothing.
    `scripted`: the first steps are answered by the script (no call), so that the pilot sees the list at its full size."""
    meta = dict(meta or {}, arm="A", seed=seed, steps=steps, sentences=sentences, stream=STREAM, policy="gate", reasks=A_REASKS)
    if scripted: meta["scripted"] = scripted
    rows = load_rows(path, meta)
    if len(rows) > steps: raise ValueError(f"{path} holds {len(rows)} steps, more than the {steps} asked for")
    stream, live, _ = replay(rows, seed, "A", sentences); stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for t in range(len(rows), steps):
            if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
            out = stream.next()
            if out is None: stopped = "stream"; break
            st, ob = out; oa = stream.live
            if t < scripted:
                text, secs = st["ideal"], 0.0; first = dict(text=text, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, seconds=0.0, from_disk=True, finish="scripted"); extra = []; ok, why = usable(live, st, text)
            else:
                msgs = [_system("A"), {"role": "user", "content": user_text(seed, t + 1, st, live, sentences)}]
                text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = _att(llm, text, ntok, sec); extra = []
                ok, why = usable(live, st, text)
                while not ok and len(extra) < A_REASKS:
                    text, ntok, sec = llm.generate(reask_messages(msgs, text, why), constrained=True); secs += float(sec); extra.append(_att(llm, text, ntok, sec)); ok, why = usable(live, st, text)
            p = parse_reply(text); after = copy_state(live)
            if ok and st["kind"] != "query": after = apply_ops(live, check_ops(live, p["ops"])[2])
            fl = row_fields(after, oa, ob, st, p, stream.introduced); fl["blocked"] = bool(p is not None and not ok)
            top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
            rec = dict(t=t + 1, kind=st["kind"], text=text, seconds=round(secs, 4), **top, **fl)
            if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
            rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); live = after; spent += billed(rec)
            if progress: progress(t + 1, steps, rec)
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


def run_depth(llm, seed: int, starts, asks: int, path: str, meta=None, sentences: int = PAD_SENTENCES, max_tokens: Optional[int] = None, progress=None) -> dict:
    """The depth probe: a chat whose history is turns 1..s-1 with the IDEAL replies (no call), then `asks` real turns, each followed by the ideal reply. One line per real turn, with `start`."""
    meta = dict(meta or {}, arm="D", seed=seed, starts=list(starts), asks=asks, sentences=sentences, stream=STREAM)
    rows = load_rows(path, meta); done = {(r["start"], r["t"]) for r in rows}; stopped = None; spent = tokens_sent(rows)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for s in starts:
            stream = make_stream(seed, s + asks); history: List[dict] = []
            for t in range(1, s + asks):
                out = stream.next()
                if out is None: stopped = "stream"; break
                st, ob = out; oa = stream.live; user = {"role": "user", "content": user_text(seed, t, st, None, sentences)}
                if t >= s and (s, t) not in done:
                    if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                    text, ntok, sec = llm.generate([_system("K")] + history + [user], constrained=True); a = _att(llm, text, ntok, sec); p = parse_reply(text)
                    shape = p is not None and (("answer" in p) if st["kind"] == "query" else ("ops" in p))
                    rec = _roundtrip(dict(start=s, t=t, kind=st["kind"], ideal=st["ideal"], **a, parsed=bool(shape), **reply_flags(ob, oa, st, p)))
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); done.add((s, t)); spent += billed(rec)
                    if progress: progress(s, t, rec)
                history += [user, {"role": "assistant", "content": st["ideal"]}]
            if stopped: break
    return dict(rows=rows, summary=summarize_depth(rows), meta=_roundtrip(meta), stopped=stopped)


def replay_depth(rows: list, seed: int) -> None:
    by = {}
    for r in rows: by.setdefault(r["start"], []).append(r)
    for s, rs in by.items():
        stream = make_stream(seed, max(r["t"] for r in rs)); need = {r["t"]: r for r in rs}
        for t in range(1, max(need) + 1):
            st, ob = stream.next(); oa = stream.live
            if t in need:
                r = need[t]; p = parse_reply(r["text"]); shape = p is not None and (("answer" in p) if st["kind"] == "query" else ("ops" in p))
                f = dict(parsed=bool(shape), **reply_flags(ob, oa, st, p))
                if st["kind"] != r["kind"] or st["ideal"] != r["ideal"] or any(r.get(k) != v for k, v in f.items()): raise ReplayMismatch(f"depth probe start {s} turn {t}: differs from the saved flags")


# ---------------------------------------------------------------- summaries
stats, tenths, window, _gt, _le, _mean, _pct = GL.stats, GL.tenths, GL.window, GL._gt, GL._le, GL._mean, GL._pct
summarize, summarize_depth, estimate_cost = GL.summarize, GL.summarize_depth, GL.estimate_cost


def by_prompt_band(rows: list, edges=BANDS) -> List[dict]: return GL.by_prompt_band(rows, edges)


def by_kind(rows: list) -> List[dict]:
    """Exact share by kind of message / question, so that a wording or a reference type that fails shows up."""
    out = []
    for k in sorted({r["kind"] for r in rows}):
        for qk in ([None] if k != "query" else sorted({r.get("qkind") for r in rows if r["kind"] == "query"})):
            b = [r for r in rows if r["kind"] == k and (qk is None or r.get("qkind") == qk)]
            out.append(dict(kind=k if qk is None else f"query:{qk}", n=len(b), exact=sum(1 for r in b if r["reply_exact"]) / len(b), held_exact=_mean(r["exact"] for r in b)))
    return out


def by_age(rows: list) -> List[dict]:
    """Exact share of the recall questions by the age of the fact asked about (turns since it was last changed)."""
    out = []
    for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:]):
        b = [r for r in rows if r["kind"] == "query" and r.get("qkind") == "recall" and r.get("age") is not None and lo <= r["age"] < hi]
        if b: out.append(dict(lo=lo, hi=hi, n=len(b), exact=sum(1 for r in b if r["reply_exact"]) / len(b)))
    return out


def _recall(rows, lo, hi):
    b = [r for r in rows if r["kind"] == "query" and r.get("qkind") == "recall" and r.get("age") is not None and lo <= r["age"] < hi]
    return (sum(1 for r in b if r["reply_exact"]) / len(b), len(b)) if b else (None, 0)


def _flag_probe(rows):
    b = [r for r in rows if r["kind"] == "query" and r.get("qkind") in ("unknown", "ambiguous")]
    return (sum(1 for r in b if r["reply_exact"]) / len(b), len(b)) if b else (None, 0)


# ---------------------------------------------------------------- the reading rules (fixed in the page before any call)
G_FINISH_K = "the kept chat (arm K) reached turn 400 (or stopped because its next prompt would pass 950,000 tokens) with every turn on file, and the replay of its saved replies reproduces every saved flag of every turn"
G_FINISH_A = "the arm with AXI (arm A) reached step 400 with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask"
G_DEEP = "the kept chat grew to a prompt of at least 150,000 tokens"
G_CACHE = "the service reported cached tokens for at least 80 percent of the calls of arm K from its 10th turn"
G_REFUSED = "at most 0.5 percent of the steps of arm A ended with a reply that could not be used (after the re-asks)"
G_PARSED = "at least 98 percent of the replies of arm K had the right shape (one JSON object with ops for a message, answer for a question)"
G_A_EXACT = "arm A: the held list equals the true list on at least 95 percent of all steps and on at least 90 percent in every tenth"
G_A_CORRUPT = "arm A: the held list was never corrupted (an item nobody mentioned, an unknown field or a value outside the allowed set) and never more than 3 cells from the true list"
G_A_FLAT = "arm A: the mean prompt tokens of the last tenth are at most 1.3 times those of the second tenth and the mean seconds a step at most 1.5 times"
G_A_Q = "arm A: at least 95 percent of the answers to the questions are exactly right"
G_K_EARLY = "arm K: at least 90 percent of the replies of turns 1 to 50 give exactly the true next list (or the right answer)"
G_K_DRIFT = "arm K drifts: the share of exact replies over its last 100 turns is at least 10 points below that of turns 1 to 50"
G_K_HALLU = "arm K hallucinates: at least 5 percent of the replies of its last 100 turns cannot be applied to the true list (they name an item that is not there, or a value that is not allowed) or have the wrong shape"
G_K_CORRUPT = "arm K: the list a firm would hold by applying the replies as they come was corrupted at some turn"
G_K_FIRST = "arm K: the first reply that is not exactly right comes before turn 50"
G_K_AGE = "arm K forgets with age: its share of exact answers to questions about facts last changed 50 or more turns earlier is at least 10 points below that for facts changed less than 10 turns earlier"
G_A_AGE = "arm A does not forget with age: its share of exact answers about facts last changed 50 or more turns earlier is within 5 points of that for facts changed less than 10 turns earlier"
G_FLAG = "ask, not guess: on the questions whose right answer is unknown or ambiguous, arm A answers exactly at least 80 percent of the time and at least 10 points more often than arm K"
G_LATENCY = "latency: the mean seconds of a call of arm K at a prompt of 150,000 tokens or more are at least 1.5 times the mean seconds of a step of arm A"
G_FIRST_TOKEN = "latency to the first token: the median seconds to the first token of arm K at a prompt of 150,000 tokens or more are at least 1.5 times those of arm A"
G_SPEED = "speed: the exact steps per hour of calls of arm A are at least 1.5 times those of arm K, over the same number of steps"
G_TOKENS = "tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per exact step as arm A, over the turns at a prompt of 150,000 tokens or more"
G_THINK = "arm K: the mean thinking tokens per turn over its last 100 turns are at least twice those of turns 1 to 50"
G_DEPTH = "depth alone (the probe, with a perfect history): at the deepest start the share of exact replies is at least 10 points below that of the shallowest"
G_LINES = [("guard", G_FINISH_K), ("guard", G_FINISH_A), ("guard", G_DEEP), ("instrument", G_CACHE), ("instrument", G_REFUSED), ("instrument", G_PARSED), ("claim", G_A_EXACT), ("claim", G_A_CORRUPT), ("claim", G_A_FLAT),
           ("claim", G_A_Q), ("claim", G_K_EARLY), ("claim", G_K_DRIFT), ("claim", G_K_HALLU), ("claim", G_K_CORRUPT), ("claim", G_K_FIRST), ("claim", G_K_AGE), ("claim", G_A_AGE), ("claim", G_FLAG),
           ("claim", G_LATENCY), ("claim", G_FIRST_TOKEN), ("claim", G_SPEED), ("claim", G_TOKENS), ("claim", G_THINK), ("claim", G_DEPTH)]
G_HEAD = "headline: lines 7 and 8 both read yes (with AXI the list holds on fuzzy text); the comparison with the kept chat is read from lines 12 to 18"


def verdict(k: Optional[dict], a: Optional[dict], d: Optional[dict] = None, replays: Optional[dict] = None) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `k`, `a`, `d`: the dicts the runs return (or None). `replays`: {'K': bool, 'A': bool}."""
    replays = replays or {}; out: List[Optional[bool]] = []; kr = k["rows"] if k else None; ar = a["rows"] if a else None
    out.append(None if not k else bool(replays.get("K", True)) and (len(kr) >= STEPS or k["stopped"] == "context"))                                                    # 1
    out.append(None if not a else bool(replays.get("A", True)) and len(ar) >= STEPS)                                                                                 # 2
    out.append(None if not k else max((r.get("prompt_tokens") or 0 for r in kr), default=0) >= 150_000)                                                              # 3
    out.append(None if not k or len(kr) <= 10 else (lambda xs: sum(1 for r in xs if (r.get("cache_tokens") or 0) > 0) / len(xs) >= 0.8)(kr[9:]))                    # 4
    out.append(None if not a else sum(1 for r in ar if not r["parsed"] or r.get("blocked")) <= 0.005 * len(ar))                                                       # 5
    out.append(None if not k else sum(1 for r in kr if r["parsed"]) >= 0.98 * len(kr))                                                                               # 6
    out.append(None if not a else (lambda ts: sum(1 for r in ar if r["exact"]) >= 0.95 * len(ar) and all(t["held_exact"] >= 0.90 for t in ts))(tenths(ar)))          # 7
    out.append(None if not a else all(not r["corrupted"] for r in ar) and max(r["distance"] for r in ar) <= 3)                                                       # 8
    out.append(None if not a else (lambda ts: ts[-1]["prompt_per_step"] <= 1.3 * ts[1]["prompt_per_step"] and ts[-1]["seconds_per_step"] <= 1.5 * ts[1]["seconds_per_step"])(tenths(ar)) if len(ar) >= 20 else None)   # 9
    qa = [r for r in ar if r["kind"] == "query"] if a else []
    out.append(None if not a or not qa else sum(1 for r in qa if r["reply_exact"]) / len(qa) >= 0.95)                                                               # 10
    s50 = stats(kr[:50]) if kr else None; l100 = stats(kr[-100:]) if kr else None
    out.append(None if not k else _gt(s50["reply_exact"], 0.9))                                                                                                      # 11
    out.append(None if not k or len(kr) < 150 else _gt(s50["reply_exact"] - l100["reply_exact"], 0.10))                                                              # 12
    out.append(None if not k or len(kr) < 150 else _gt(1 - l100["reply_ok"], 0.05))                      # 13
    out.append(None if not k else any(r["corrupted"] for r in kr))                                                                                                   # 14
    fi = next((r["t"] for r in kr if not r["reply_exact"]), None) if kr else None
    out.append(None if not k else (fi is not None and fi < 50))                                                                                                      # 15
    if k:
        yk, ny = _recall(kr, 0, 10); ok_, no = _recall(kr, 50, 10 ** 9); out.append(None if yk is None or ok_ is None or ny < 3 or no < 3 else (yk - ok_) >= 0.10 - 1e-12)   # 16
    else: out.append(None)
    if a:
        ya, ny = _recall(ar, 0, 10); oa_, no = _recall(ar, 50, 10 ** 9); out.append(None if ya is None or oa_ is None or ny < 3 or no < 3 else abs(ya - oa_) <= 0.05 + 1e-12)   # 17
    else: out.append(None)
    if k and a:
        fk, nk = _flag_probe(kr); fa, na = _flag_probe(ar); out.append(None if fk is None or fa is None or na < 3 else fa >= 0.8 and (fa - fk) >= 0.10 - 1e-12)             # 18
    else: out.append(None)
    deep = window(kr, 150_000) if kr else []; sd = stats(deep) if deep else None; sa = stats(ar) if ar else None
    out.append(None if not deep or not a else _gt(sd["seconds_per_step"], 1.5 * sa["seconds_per_step"]))                                                             # 19
    out.append(None if not deep or not a or sd["first_p50"] is None or sa["first_p50"] is None else _gt(sd["first_p50"], 1.5 * sa["first_p50"]))                      # 20
    if k and a and len(ar) >= len(kr):
        sk, sa2 = stats(kr), stats(ar[:len(kr)]); out.append(_gt(sa2["exact_per_hour"], 1.5 * sk["exact_per_hour"]) if sk["exact_per_hour"] and sa2["exact_per_hour"] else None)   # 21
    else: out.append(None)
    out.append(None if not deep or not a or sd["computed_per_exact"] is None or sa["computed_per_exact"] is None else _gt(sd["computed_per_exact"], 1.5 * sa["computed_per_exact"]))   # 22
    out.append(None if not k or len(kr) < 150 else _gt(l100["think_per_step"], 2 * s50["think_per_step"]))                                                           # 23
    if d and d["rows"]:
        sm = summarize_depth(d["rows"]); ks = sorted(sm)
        out.append(None if len(ks) < 2 else _gt(sm[ks[0]]["reply_exact"] - sm[ks[-1]]["reply_exact"], 0.10))                                                          # 24
    else: out.append(None)
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    """Lines 1, 2, 3, 5 and 6 read yes: the figures are claimed as read. Line 4 NO: no computed-token (energy stand-in) figure is claimed, only prompt tokens."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (1, 2, 3, 5, 6))
    return dict(base=base, tokens_computed=bool(base and ok[4] is True))


def headline(checks) -> str:
    ok = {i: c for i, _, _, c in checks}; need = (7, 8)
    if any(ok[i] is None for i in need): return "NOT READ (a headline line was not run)"
    return "AXI HOLDS ON FUZZY TEXT" if all(ok[i] for i in need) else "MIXED: " + ", ".join(f"line {i} {'yes' if ok[i] else 'NO'}" for i in need)


# ---------------------------------------------------------------- the check before the run, and the printout
def pilot_report(k_rows: list, a_rows: list, price_in=None, price_out=None, price_cached=None, cached_share_assumed: float = 0.9) -> dict:
    """What 20 real turns of each arm say about the whole run: tokens per turn, thinking tokens, seconds, the total prompt tokens of arm K over STEPS turns (they grow with the square of the turns), and the projected cost
    of the two arms at the prices typed in."""
    sk, sa = stats(k_rows), stats(a_rows); grow = (k_rows[-1]["prompt_tokens"] - k_rows[0]["prompt_tokens"]) / max(1, len(k_rows) - 1) if len(k_rows) > 1 else None; base = k_rows[0]["prompt_tokens"] if k_rows else None
    if grow:
        k_prompt = sum((base or 0) + grow * i for i in range(STEPS)); share = sk["cached_share"] if sk["cached_share"] is not None else cached_share_assumed
        k_tok = dict(prompt=k_prompt, cache=k_prompt * share, vis=STEPS * sk["vis"] / max(1, sk["live_steps"]), think=STEPS * sk["think"] / max(1, sk["live_steps"])); k_nc = dict(k_tok, cache=0)
    else: k_tok = k_nc = None
    a_tok = dict(prompt=sa["prompt"] / max(1, sa["live_steps"]) * STEPS, cache=0, vis=sa["vis"] / max(1, sa["live_steps"]) * STEPS, think=sa["think"] / max(1, sa["live_steps"]) * STEPS)
    return dict(k_growth_per_turn=grow, k_first_prompt=base, k_last_prompt_projected=(base + grow * (STEPS - 1)) if grow else None, k_total_prompt_tokens=(k_tok or {}).get("prompt"), k_think_per_turn=sk["think_per_step"],
                k_seconds_per_turn=sk["seconds_per_step"], a_prompt_per_step=sa["prompt_per_step"], a_think_per_step=sa["think_per_step"], a_seconds_per_step=sa["seconds_per_step"], a_total_tokens=a_tok["prompt"] + a_tok["vis"] + a_tok["think"],
                cost_a=estimate_cost(a_tok, price_in, price_out), cost_k=estimate_cost(k_tok, price_in, price_out, price_cached) if k_tok else None, cost_k_nocache=estimate_cost(k_nc, price_in, price_out) if k_nc else None)


print_stats, print_bands, _f = GL.print_stats, GL.print_bands, GL._f


def print_kinds(rows: list):
    print(f"{'kind':>18} {'n':>4} | {'exact':>6} {'held list exact':>15}")
    for b in by_kind(rows): print(f"{b['kind']:>18} {b['n']:>4} | {_f(b['exact'] * 100):>5}% {_f(None if b['held_exact'] is None else b['held_exact'] * 100):>14}%")


def print_ages(rows: list):
    print(f"{'age of the fact (turns)':>24} {'n':>4} | {'exact answers':>13}")
    for b in by_age(rows): print(f"{b['lo']:>10}-{min(b['hi'], 9999):<12} {b['n']:>4} | {_f(b['exact'] * 100):>12}%")


def print_verdict(checks, title="FUZZY-STATE TEST (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks))
    a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2, 3, 5 and 6 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")

"""Receive without adding: the host applies a plain change itself, checked by the gate, with no model call (concept-map item 129,
docs/RECEIVE_WITHOUT_ADDING.md, test 155).

Prompted by Rajnish Choubey's article "a durga diary: Why do we park in driveways and drive on parkways?" (Medium): the user's input is
complete, "the addition is the failure"; a tool that "cannot receive without immediately responding"; an architecture that "doesn't know how
to stop". The same line runs through "What Exactly Will AI do?" (a good loudspeaker passes the sound without adding anything of its own) and
"SAARE AI ke bacho YAHAN AAO" ("the language can be cleaned, but the pulse should stay raw"). Item 126 (`answer.py`) took the model off the
questions the host can answer; this takes it off the changes the host can read.

`receive` reads a message with one word, one reading (item 127, `spans`). It acts only when the message is plain:
  * exactly one sentence names a held item, and it names exactly one (a sentence that names none, such as office chatter, is not read);
  * that sentence asks nothing ("?"), gives no reason ("so", "since", "because", "given", "therefore": a reason is a premise, item 128;
    "as" is not read as a reason, it also marks a role: "Mark Amber Gallery as confirmed"), adds no aside ("which", "who", brackets: an
    aside is a premise too), hedges nothing ("if", "maybe", "might", "could", "should", "perhaps", "probably", "unless", "when", "think")
    and negates nothing ("not" only as the mark of an old value: "Marcus, not Elena");
  * it carries one new value for each field it changes (an old value marked "not / from / instead of / rather than" is left out), and every
    such value differs from the list; or, with no value at all, a word of removal (drop, remove, delete, cancel, scrap, forget, off).
Then it builds the change and hands it to the gate (`check`), the same check every model reply passes. Anything else, and anything the gate
refuses, goes to the model as before. It never adds an item (a new name is not on the list it can read), never reads a loose reference
("the last one we added", "the one Priya is handling", "the same day as ...": those name no item or two), never writes a reply text.
It edits no framework file.
What it does NOT show: how real messages are worded (the measured messages follow Test 23's templates and ours); a bare statement of a new
value ("Amber Gallery is on Friday.") is taken as a change, as the model and Test 23's ideal take it, and so is a statement joined to the
change by a plain "and" ("Dune Gallery is on Wednesday and hand Dune Gallery to Marcus.": both are written): if it was meant as a premise,
the premise check (item 128) cannot see it either."""
from __future__ import annotations
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from axi.engine.answer import spans
from axi.engine.premise import sentences

_REASON = re.compile(r"\b(?:so|since|because|given|therefore|thus|hence)\b", re.I)
_ASIDE = re.compile(r"\b(?:which|who|whose)\b|[()\[\]]", re.I)
_HEDGE = re.compile(r"\b(?:if|maybe|might|could|should|would|perhaps|probably|possibly|unless|when|whether|think|guess|suppose)\b", re.I)
_NEG = re.compile(r"\b(?:not|never|no\s+longer|nothing|nobody|cannot)\b|n't\b", re.I)
_OLD = re.compile(r"\b(?:not|from|instead\s+of|rather\s+than)\s+(?:on\s+|at\s+|for\s+|to\s+)?$", re.I)
_DROP = re.compile(r"\b(?:drop|dropped|remove|removed|delete|deleted|cancel|cancelled|canceled|scrap|scrapped|forget|off)\b", re.I)


def receive(state: Mapping[str, Mapping[str, Any]], message: str, values: Mapping[str, Iterable[Any]],
            check: Optional[Callable[[Mapping, list], Tuple[bool, str, list]]] = None) -> Tuple[Optional[List[dict]], str]:
    """(ops, how). ops None: the host hands the message to the model. `values`: field -> the values it allows. `check(state, ops)`: the
    gate, returning (ok, why not, ops as committed); without it the ops are returned unchecked (for reading tests only)."""
    kinds: Dict[Any, List[str]] = {("item",): sorted(state)}
    for f, vs in values.items(): kinds[("value", f)] = [str(v) for v in vs]
    read = spans(message, kinds); held = {n.lower() for n in state}
    hits = []
    for a, b in sentences(message):
        inside = [s for s in read if a <= s[0] < b]
        named = {s[3] for s in inside if s[2] == ("item",)}
        double = [s for s in inside if s[2] is None]
        if double and (named or any(s[3] in held for s in double)): return None, "one word, two readings"
        if named: hits.append((a, b, named, inside))
    if not hits: return None, "no held item named"
    if len(hits) > 1: return None, "two sentences name held items"
    a, b, named, inside = hits[0]
    if len(named) != 1: return None, "the sentence names two items"
    n = next(iter(named)); text = message[a:b]
    if "?" in text: return None, "a question"
    if _REASON.search(text): return None, "a reason given: a premise, not a change"
    if _ASIDE.search(text): return None, "an aside (which, who, brackets): a premise, not a change"
    if _HEDGE.search(text): return None, "hedged"
    new: Dict[str, List[str]] = {}
    old_marks = []
    for start, end, kind, said in inside:
        if kind is None or kind[0] != "value": continue
        if _OLD.search(message[a:start]): old_marks.append(start - a); continue
        new.setdefault(kind[1], [])
        if said not in new[kind[1]]: new[kind[1]].append(said)
    for m in _NEG.finditer(text):
        if m.group(0).lower() != "not" or not any(0 <= p - m.end() <= 8 for p in old_marks): return None, "negated"
    if any(len(v) > 1 for v in new.values()): return None, "two values for one field"
    if not new:
        if _DROP.search(text): ops = [dict(op="delete", name=n)]
        else: return None, "no value and no removal"
    else:
        if _DROP.search(text): return None, "a value and a removal"
        ops = []
        for f in values:
            if f not in new: continue
            v = new[f][0]; held = state[n].get(f)
            if held is not None and str(held).lower() == v.lower(): return None, "no change: the list already has it"
            ops.append(dict(op="set", name=n, field=f, value=int(v) if v.isdigit() else v))
    if check is None: return ops, "read"
    ok, why, done = check(state, ops)
    return (done, "read and checked") if ok else (None, "the gate refused: " + why)

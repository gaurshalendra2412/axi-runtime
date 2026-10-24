"""The premise check: what a message takes for granted is checked against the exact state (concept-map item 128, docs/PREMISE_CHECK.md, test 154).

Prompted by Rajnish Choubey's article "The Cinderella Project: Testing AI Understanding of Modern Visual Communication" (Medium, The
Cinderella Project). In the AI chats it shows, the AI took each new premise it was given, then reversed it, and kept agreeing with whatever
framing came last: a chat with no held facts lets the latest message win.

AXI already keeps the facts with the host, and its check before a change asks whether the change is VALID (the item is on the list, the
value is allowed). Nothing asked whether what a message TAKES FOR GRANTED is TRUE. `premises` reads a message with one word, one reading
(item 127) and, in every sentence that names exactly one item the host holds, ties each value it carries (a day, an amount, a status, an
owner) to that item; then it compares:
  * a value marked as the old one ("not Monday", "from Monday", "instead of Monday", "rather than Monday") with the list BEFORE the step;
  * every other value with the list the host would commit AFTER the step (for a question, the list does not change).
A value that the list does not hold is reported: "you said Friday for Amber Gallery; the list has Monday". The report changes nothing:
the check, the repair and the commit run as before. A sentence that names no item or two items ties nothing, so chatter around a message
cannot raise an alarm. It edits no framework file.
What it does NOT show: a premise the model turns into a change ("Amber Gallery, which is on Friday, now costs 500" answered by setting
the day to Friday as well) passes, because the list after the step then holds it; how real messages are worded."""
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from axi.engine.answer import spans

_END = re.compile(r"[.?!;]+(?:\s|$)")
_OLD = re.compile(r"\b(?:not|from|instead\s+of|rather\s+than)\s+(?:on\s+|at\s+|for\s+)?$", re.I)


def sentences(text: str) -> List[tuple]:
    """(start, end) of every sentence of `text`."""
    out, a = [], 0
    for m in _END.finditer(text): out.append((a, m.end())); a = m.end()
    if a < len(text): out.append((a, len(text)))
    return out


def premises(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]], message: str,
             values: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """The values `message` ties to an item that the list does not hold. `values`: field -> the values it allows. Each report: item,
    field, said, held (None: the fact is not on the list), against ("before" or "after"), sentence."""
    items = sorted(set(before) | set(after))
    kinds: Dict[Any, List[str]] = {("item",): items}
    for f, vs in values.items(): kinds[("value", f)] = [str(v) for v in vs]
    read = spans(message, kinds); out = []
    for a, b in sentences(message):
        inside = [s for s in read if a <= s[0] < b]
        named = {s[3] for s in inside if s[2] == ("item",)}
        if len(named) != 1 or any(s[2] is None for s in inside): continue
        n = next(iter(named))
        for start, end, kind, said in inside:
            if kind is None or kind[0] != "value": continue
            f = kind[1]; old = bool(_OLD.search(message[a:start]))
            ref = before if old else after
            if n not in ref: continue
            held = ref[n].get(f)
            if held is None or str(held).lower() != said.lower():
                out.append(dict(item=n, field=f, said=said, held=None if held is None else str(held), against="before" if old else "after",
                                sentence=message[a:b].strip()))
    return out


def note(reports: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """What the host tells the user, or None when every premise holds."""
    parts = []
    for r in reports:
        when = "had" if r["against"] == "before" else "has"
        parts.append(f"you said {r['said']} for the {r['field']} of {r['item']}; the list {when} " + (r["held"] if r["held"] is not None else "no " + r["field"]))
    return ("Note: " + "; ".join(parts) + ".") if parts else None

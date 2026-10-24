"""R = 0 for questions: the host answers a question about the state it holds (concept-map item 126, docs/HOST_ANSWERS.md, test 152),
read with one word, one reading (concept-map item 127, docs/ONE_READING.md, test 153).

Prompted by Rajnish Choubey's open letter "The Failure of Thought Imposition" and its Spherical Mind: the radius R is everything the machine
is made to engage; at R = 0 "the question is the answer" and "how you process: silence"; at R = Max the output is exhaustive and made for
show. The method is a plain lookup in the exact state.

A question about the state ("what is the day of Amber Gallery?") does not need the model: the host holds the exact answer. `answer`
reads the question for the one field it asks about and for the item: by name, or by a description (a value of a describing field, such
as an owner) resolved against the held facts (witnesses.resolve). Then:
  * one item, the fact held  -> the value;
  * one item, the fact never given -> "unknown";
  * a description that fits two or more items -> "ambiguous";
  * anything the host cannot place (no field, two fields, no item, a description that fits nothing) -> None: the model is asked, as today.

One word, one reading (`read`, item 127). Prompted by the text on Devanagari and the Latin alphabet from Rajnish Choubey's conversation
(pasted 24 Oct): Devanagari has one sound for one symbol, and every consonant has one place, set by where in the mouth it is made; English
has "their / there / they're" (one sound, several spellings) and "read / read" (one spelling, several sounds). Read separately, the field
words, the item names and the describing values let one word be read twice: in "how much is Day Spa?" the word "Day" is part of the name
AND the field "day", and the host answered with the day. `read` looks for every phrase of every kind at once, longest first; a span taken
by one phrase is never read again inside another; a phrase that belongs to two kinds (an item named "Priya" when Priya is also an owner)
is reported, and the question goes to the model. The host never invents a value, and with one reading per word it never reads a word
two ways. It edits no framework file.
What it does NOT show: how real questions are worded; the questions it was measured on follow the templates of the Test 23 stream."""
from __future__ import annotations
import re
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

from axi.engine.witnesses import resolve


def spans(text: str, kinds: Mapping[Hashable, Iterable[str]]) -> List[Tuple[int, int, Optional[Hashable], str]]:
    """Every span read, in text order: (start, end, kind, phrase). Every phrase of every kind is looked for at once, as whole words, case
    ignored, longest first; every occurrence a phrase takes is closed to the shorter phrases inside it. kind None: the phrase belongs to
    two kinds, or to one kind in two spellings (no reading is chosen), and the phrase is given in lower case."""
    owners: Dict[str, set] = {}
    for k, phrases in kinds.items():
        for p in phrases: owners.setdefault(str(p).lower(), set()).add((k, str(p)))
    taken: List[Tuple[int, int]] = []; out = []; tl = text.lower()
    for low in sorted(owners, key=lambda s: (-len(s), s)):
        if low not in tl: continue                                         # a phrase that is not in the text at all needs no search
        for m in re.finditer(r"(?<!\w)" + re.escape(low) + r"(?!\w)", text, re.I):
            if any(m.start() < b and a < m.end() for a, b in taken): continue
            taken.append((m.start(), m.end()))
            if len(owners[low]) > 1: out.append((m.start(), m.end(), None, low))
            else: (k, p), = owners[low]; out.append((m.start(), m.end(), k, p))
    return sorted(out, key=lambda s: s[0])


def read(text: str, kinds: Mapping[Hashable, Iterable[str]]) -> Tuple[Dict[Hashable, List[str]], List[str]]:
    """(found, double), from `spans`. `found[kind]`: the phrases of that kind that were read, each once, longest first. `double`: phrases
    read that belong to two kinds, or to one kind in two spellings (no reading is chosen)."""
    found: Dict[Hashable, List[str]] = {k: [] for k in kinds}; double: List[str] = []
    for _, _, k, p in sorted(spans(text, kinds), key=lambda s: (-len(s[3]), s[3].lower(), s[0])):
        if k is None:
            if p not in double: double.append(p)
        elif p not in found[k]: found[k].append(p)
    return found, double


def answer(state: Mapping[str, Mapping[str, Any]], question: str, fields: Iterable[str], describers: Optional[Mapping[str, Iterable[str]]] = None) -> Tuple[Optional[str], str]:
    """(answer, how): answer None means the host hands the question to the model."""
    kinds: Dict[Hashable, List[str]] = {("field",): list(fields), ("item",): list(state)}
    for df, values in (describers or {}).items(): kinds[("desc", df)] = [str(v) for v in values]
    found, double = read(question, kinds)
    if double: return None, "one word, two readings"
    fs = found[("field",)]
    if len(fs) != 1: return None, "field not placed"
    f = fs[0]; names = found[("item",)]; desc, clash = {}, False
    for df in (describers or {}):
        hits = found[("desc", df)]
        if len(hits) > 1: clash = True
        elif hits: desc[df] = hits[0]
    if clash: return None, "two values for one description"
    if len(names) == 1 and not desc:
        facts = state[names[0]]; return (str(facts[f]) if f in facts else "unknown"), "named"
    if not names and desc:
        status, items = resolve({n: {k: str(v) for k, v in x.items()} for n, x in state.items()}, desc)
        if status == "bound": facts = state[items[0]]; return (str(facts[f]) if f in facts else "unknown"), "described"
        if status == "ambiguous": return "ambiguous", "described, fits several"
        return None, "description fits nothing"
    return None, "item not placed"

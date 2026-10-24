"""R = 0 for questions: the host answers a question about the state it holds (concept-map item 126, docs/HOST_ANSWERS.md, test 152).

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
The host never invents a value, so an answer it gives cannot be a hallucination. It edits no framework file.
What it does NOT show: how real questions are worded; the questions it was measured on follow the templates of the Test 23 stream."""
from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional, Tuple

from axi.engine.focus import mentioned
from axi.engine.witnesses import resolve


def answer(state: Mapping[str, Mapping[str, Any]], question: str, fields: Iterable[str], describers: Optional[Mapping[str, Iterable[str]]] = None) -> Tuple[Optional[str], str]:
    """(answer, how): answer None means the host hands the question to the model."""
    fs = mentioned(question, list(fields))
    if len(fs) != 1: return None, "field not placed"
    f = fs[0]; names = mentioned(question, list(state)); desc, clash = {}, False
    for df, values in (describers or {}).items():
        hits = mentioned(question, [str(v) for v in values])
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

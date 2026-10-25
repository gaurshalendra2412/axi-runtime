"""Escalate when nothing is committed: ask the next model only on a step the host itself sees fail. A MEASUREMENT, NOT KEPT in the engine
(concept-map item 131, docs/ESCALATE_ON_REFUSAL.md, test 157).

Prompted by Rajnish Choubey's line quoted, seven times, in the AI replies of "How Perplexity Made Google a backend team of brick layers"
(Medium, 4 May): "The only thing that scales is AI-to-AI communication + human-centric routing". Item 109 re-asked the SAME model once
where the repair could not fix a refused reply; this asks the NEXT model instead, at the same point.

`escalate(askers, commit, before, same)`: ask the models in their order; run the host's own commit (the gate and the repair, unchanged) on
each reply; stop at the first reply that commits a change. The trigger is what the host sees for itself: the state after the commit is
the state before it (not a raw refusal of the gate, and not a wrong reply the gate admits, which the host cannot see).

Why it is not kept: replayed from the saved seed-7 replies, its whole gain (flash-lite 108 -> 138 of 140 with the rule stated) came from
one slip of flash-lite's (a node delete written with a value, which the framework cannot read); a reader that drops that value gives
flash-lite alone the same 138 with no second call, and with Qwen first (grammar on) nothing is ever passed on. It edits no framework file.
What it does NOT show: a live run of the router; that needs a run."""
from __future__ import annotations
from typing import Any, Callable, Sequence, Tuple, TypeVar

S = TypeVar("S")


def escalate(askers: Sequence[Callable[[], Any]], commit: Callable[[Any], S], before: S, same: Callable[[S, S], bool]) -> Tuple[S, dict]:
    """(state after, why). `askers`: one call per model, in the order to ask them (each returns that model's reply). `commit(reply)`: the
    host's own step on a reply (parse, gate, repair) returning the state it would commit. why: asked (how many models were asked), by (the
    index of the model whose reply committed a change; None when none did and the state is left as it was), replies (what each wrote)."""
    replies = []
    for i, ask in enumerate(askers):
        reply = ask(); replies.append(reply)
        after = commit(reply)
        if not same(after, before): return after, dict(asked=i + 1, by=i, replies=replies)
    return before, dict(asked=len(replies), by=None, replies=replies)

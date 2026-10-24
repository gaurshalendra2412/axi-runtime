# The 1 % view: show only what the request needs

*Concept-map item 124. Code: `python/axi/engine/focus.py`. Checked by test 150 (`python/tests/test_focus.py`). No model is called; the figures below are counted offline from the Test 23 stream and its saved calls.*

## Where it comes from

Rajnish Choubey's text "The Trilogy of the Canvas: The Art of the Void" (the owner pasted it on 24 Oct). Art has had two acts, creation and editing; the text adds a third, **deliberate deletion**. Of everything made, 99 % is deleted and 1 % is shown, and the value of an idea is the noise cleared around it.

Read for structure, most of the text is already AXI: deleting the history and keeping only the exact state is what AXI does at every step, and independent voices whose agreement is picked up is the witness quorum (item 122). One part is new.

## What is new

AXI deletes the history, but it still shows the model the **whole** state at every step. Its prompt therefore still grows with the size of the state: in Test 23 the prompt went from 959 tokens a step in the first 50 steps to 1,337 in the last 100, as the list grew. A state of thousands of items would not fit.

The 1 % view applies deliberate deletion to the view. For each request the host shows only:

* the items the request **names**;
* the items a **description** in it points to (for example "whatever Priya is handling": every item Priya has, so an ambiguous description stays visible as ambiguous);
* the most recently added item, when the request says "**last**";
* nothing else for a request that adds a **new** item;
* the **whole** state when the host cannot place the request (the fallback).

For a graph, `focus_graph` shows the named nodes, every edge that touches them and the nodes at the other ends: everything a delete and its repair need.

The view changes what the model reads, never what the host checks: the check, the repair and the commit still run on the **whole exact state**.

## What it gives on Test 23 (offline, no call)

Over the 400 steps of the Test 23 stream (seed 20261122), the view held every item the right reply depends on at every step: **400 of 400**, with **no fallback**. The list held 17.5 items on average; the view showed 1.2.

With the view in place of the whole list, the prompt of each step would be, estimated from the tokens of the prompts Test 23 really sent:

| Steps | Whole list (sent) | 1 % view (estimated) |
|---|---|---|
| All 400 | 1,177 | 920 |
| First 50 | 959 | 912 |
| Last 100 | 1,337 | 921 |

The prompt no longer grows with the list. What is left (about 900 tokens) is the instructions and the 30 sentences of office chatter the test puts around every message.

## Why it matters for a large state

For a request that names one item, the list part of the prompt, at the same tokens per character:

| Items held | Whole list | 1 % view |
|---|---|---|
| 20 | about 354 tokens | about 23 tokens |
| 1,000 | about 17,686 tokens | about 24 tokens |
| 10,000 | about 176,856 tokens | about 24 tokens |

With the whole list AXI's prompt is the size of the state; with the view it is the size of the request. That is the step a company with a large state (an inventory, a ticket queue, a codebase's dependency graph) needs.

## What this does not show

That the model is as exact with the view as with the whole list: that needs a run, and its cost would be stated before any run. The view is chosen from the request's names, descriptions and the word "last"; a request that points at items another way gets the whole state. The large-state figures are arithmetic on made-up states, not measurements.

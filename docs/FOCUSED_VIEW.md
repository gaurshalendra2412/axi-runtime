# The 1 % view: show only what the request needs

*Concept-map item 124, and item 130 for the meeting rings. Code: `python/axi/engine/focus.py`. Checked by test 150 (`python/tests/test_focus.py`) and test 156 (`python/tests/test_focus_meet.py`). No model is called; the figures below are counted offline from the Test 23 stream and its saved calls, and from graphs made by the repository's own generators.*

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

## Widen on refusal (a rule for use, not measured)

Rajnish Choubey's open letter "The Failure of Thought Imposition" describes a Spherical Mind whose radius R is everything the machine is made to engage, and places intelligence at R = 0. Read for the view: start every step at the smallest view, and widen to the whole state only when the host's check refuses the reply, so that a reply refused for want of context costs one re-ask with the whole state. This rule is written here and not measured: how often it would widen needs a run.

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

## Meeting rings: the view for a request about two things (item 130)

**Where it comes from.** Rajnish Choubey's lines in "The Gen 2 Reset" (Medium, 6 June; a chat with Gemini, so the wording may be the AI's). An old model files a word in a pyramid of folders (animal, reptile, snake). In the new one, every word is a centre with rings expanding around it, and meaning sparks where two words' rings intersect: the snake's ring meets the ring of time.

Read for structure, the rings around one word are `focus_graph` (item 124): the named node, then one ring of edges, then the next. What is new is the second centre. A request can be about two things: how a service is linked to another, or whether one depends on the other through others. For such a request the view should hold the link between them, and a fixed number of rings around each does not know how far apart they are.

**What is new.** `focus_meet(g, a, b)` grows rings from both nodes, one whole ring at a time, the smaller ring first, until the rings intersect. It shows only the nodes and edges of the shortest paths between the two, nothing else. It can follow edges one way (does a reach b along its edges) and only some relations. When the rings run out without meeting, the host knows exactly that the two are **not connected**. The search from both ends is a standard method; what is new is using it to choose the view. As before, the view changes what the model reads, never what the host checks.

**What it gives (no model; graphs made by the repository's own generators).**

The view was checked against a plain search on every pair of nodes: 200 small service graphs (the scope tests' generator) and a 120-node graph (the large KV test's generator), both ways and one way. On every pair it held exactly the nodes and edges of the shortest paths.

On the four 120-node, 200-edge graphs of the large KV test (seeds 21000 to 21003), over every connected pair of nodes:

| View | Holds a path between the two | Nodes | Edges |
|---|---|---|---|
| Meeting rings | **24,321 of 24,321** | 6.7 | 6.7 |
| 1 ring around each | 13.5 % | 9.0 | 7.2 |
| 2 rings around each | 70.1 % | 29.1 | 30.6 |
| 3 rings around each | 98.0 % | 66.5 | 88.2 |
| Whole graph | all | 120 | 200 |

A fixed view holds a path exactly when the two are at most twice its rings apart. In the large test's crude count, the whole graph is 3,554 tokens, the 3-ring view 1,652, and the meeting view 151.

The rings from both ends reached 27.9 nodes on average before meeting. Rings from one end, out to the same distance, reach 70.0.

On a made-up graph of 10,000 nodes with the same density, 252 of 300 random pairs were connected (the other 48 were reported as not connected):

* the meeting view held a path for all 252, with 14.5 nodes on average;
* the 3-ring view held one for 62 of 252, with 119 nodes;
* the rings from both ends reached 300 nodes before meeting, where rings from one end reach 5,728;
* a call takes well under a millisecond on average.

On the 8-node graphs of the scope tests the gain is small: the whole graph is already 8 nodes, and 2 rings around each node hold a path for 99.2 % of pairs.

**What it does not show.** That a model reads a connection better from this view; that needs a run. **No test of AXI asks about a connection yet**: its requests name one node, or two that an edge will join. So this view serves a kind of request the tests do not have. Until a test asks one, it is a tool, not a measured gain in exactness.

## What this does not show

That the model is as exact with the view as with the whole list: that needs a run, and its cost would be stated before any run. The view is chosen from the request's names, descriptions and the word "last"; a request that points at items another way gets the whole state. The large-state figures are arithmetic on made-up states, not measurements.

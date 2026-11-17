# Test 40: where the prompt starts to grow, as the state grows from 20 to 20,000 items, with and without the focused view (written BEFORE the run, 16 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written before the harness exists and before any of its numbers; the Result goes below it and nothing above the Result line is edited afterwards. No model is called, no GPU and no key are used: the test runs on a CPU.*

## Why this test
Every serving test so far held a state of a few dozen to a few hundred items, and said so. AXI's claim is that the prompt does not grow with the history. It does grow with the state: the exact-state arm shows the model the whole state at every step (`FOCUSED_VIEW.md`, item 124, measured on one 400-step stream: 959 to 1,337 tokens as the list grew). The focused view (`python/axi/engine/focus.py`, `focus_graph`) is the repository's answer: show the named services, every edge that touches them and the services at the other ends. **The question a company with a large state asks: from 20 to 20,000 items, how big is what the model is shown with the whole state and with the view, how does that depend on the shape of the graph (even degrees against hubs), and does the host's own work (the gate, the view, the serialisation) grow with the state?** This is the plain place where AXI's prompt starts to grow; it is measured, not argued.

## What is run
The repository's code, unchanged (`python/axi/engine`: `gate.py`, `focus.py`, `scope.py`, `cico_parser.py`, repo commit `55631ec`), CPython 3.13.18, two cores; no framework file is edited. The harness is new, written after this page: `python/axi/experiments/state_size_scaling.py`. Items are services plus edges.

**Graphs.** Four sizes, **20, 200, 2,000 and 20,000 items** (one service to two edges, so about 7, 67, 667 and 6,667 services), in two kinds, so eight graphs, built from fixed seeds:
* **even:** every service gets two outgoing edges to services chosen uniformly at random, with a relation name from a set of eight and a weight from 1 to 9 (degrees close to equal);
* **hub:** preferential attachment with two edges per new service (a few services collect a large part of the edges; the maximum degree grows with the size). Real service graphs have hubs; this is one model of them, not the real thing.
Services sit on a grid just wide enough for their number.

**What is measured, on each of the eight graphs.**
* **Whole state.** The characters of the state as the repository writes it (`CICOParser.serialize_state`), and characters per item. Tokens are not counted (no tokenizer is reachable from this computer); where a token count is needed it is the characters divided by 3, an assumption, marked as one.
* **The view.** 2,000 requests, each naming two services chosen uniformly at random, and 500 requests each naming the service of highest degree and one random service ("hub-targeted"). For each, `focus_graph(g, named, hops=1)`: the number of items shown and the characters of the view written the same way; median, 95th percentile, 99th percentile and largest; and the share of requests whose view is more than 10 percent of the whole state.
* **Checked against a plain scan.** For 200 requests per graph, the view must contain every named service that exists and every edge that touches one (a scan over all edges, written separately).
* **The host's own work, in microseconds, median of repeated calls:** `gate.check` of a one-edit delta (an edge addition) on the whole state; `focus_graph`; `scope.contain_scope` on the same delta; and the serialisation of the whole state.
* **Memory:** the process's resident memory before and after the 20,000-item graphs are built, and the time to build them.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". A line that comes out "NO" is a finding, not an error.

1. [instrument] all eight graphs were built with the number of items within 5 percent of the target, and every view held every named service that exists and every edge that touches one, in all 200 plain-scan checks of each graph (95 %)
2. [claim] the characters per item of the whole state differ by less than 25 percent across the four sizes (80 %)
3. [claim] the whole state at 20,000 items is at least 500 times as many characters as at 20 items (85 %)
4. [claim] even graph: the median number of items in the view at 20,000 items is at most 3 times that at 200 items (90 %)
5. [claim] even graph: the 99th percentile of the view at 20,000 items is at most 3 times that at 200 items (80 %)
6. [claim] hub graph, random requests: the 99th percentile of the view at 20,000 items is at most 3 times that at 200 items (70 %): in preferential attachment the tail of the degree distribution has a fixed shape, so a percentile should stay put while the largest hub grows
7. [claim] hub graph, random requests: the largest view among the 2,000 requests at 20,000 items is at least 3 times the largest at 200 items (70 %)
8. [claim] hub graph, hub-targeted requests: the median view at 20,000 items is between 1 and 10 percent of the items of the whole state (65 %): I expect the largest hub to have about 160 edges, so a view of about 330 items out of 20,000
9. [claim] the time of `gate.check` on a one-edit delta at 20,000 items is at most 3 times that at 200 items (85 %)
10. [claim] the time of `focus_graph` on the even graph at 20,000 items is at most 3 times that at 200 items (75 %)
11. [claim] the time to serialise the whole state at 20,000 items is at least 50 times that at 200 items (90 %)
12. [claim] the whole state is over 24,576 characters (8,192 tokens at the assumed 3 characters per token) at the 2,000-item size and under it at the 200-item size (85 %); from a quick count of about 19 characters per item I expect the crossing near 1,300 items
13. [claim] the whole state at 20,000 items is over 384,000 characters (128,000 tokens at the assumed 3 characters per token), so it would not fit a 128,000-token window (35 %): by the same count of about 19 characters per item it is near 375,000, which is close to the line

What would count against AXI here: a view that grows with the state on the even graph, or a gate that slows with the state. A hub graph whose view grows is not against AXI's gate; it is the place where the prompt starts to grow again, and it is reported as such.

## Honest limits
Synthetic random graphs, two kinds; no model is called, so nothing here says a model is as exact with the view as with the whole state (that needs a run); requests name coordinates, and a request in ordinary words needs the host's reader first (Tests 36 and 37 show how far that goes); one hop only; characters, not tokens; a two-core sandbox, one interpreter, times in microseconds that depend on the machine; the Python reference model, not the Rust crate. The even and hub graphs are two points, not the space of real graphs.

## Result
*Written after the run, 16 Nov 2026. Nothing above this line was changed after the run.*

Run: `python3 -m axi.experiments.state_size_scaling` (new file `python/axi/experiments/state_size_scaling.py`, written after the plan; sha256 `a563ecd28ba44b84b67b124ad0013c4ed3ea20aab48d8e6d00b5b5227b049377`), repo code at commit `55631ec` unchanged, CPython 3.13.18, two cores, fixed seeds, one run of 3.6 seconds. Raw file: `docs/results/axi_state_size_scaling.json` (sha256 `2820a5656f5b32664d715a0cf369303ed4cfa57535ca603a9a488a32edc38de2`). The plan page was revised once, before the harness existed, after the first hash was written down: lines 6 to 13 were re-written because I worked out that the tail of a preferential-attachment degree distribution has a fixed shape (a percentile stays put while the largest hub grows) and that the whole state is about 19 characters per item. No number from the harness existed then.

| graph | items | services | edges | largest degree | whole state, characters | view, random requests: median / 99th pct / largest (items) | view, hub-targeted: median (items) | `gate.check` µs | `focus_graph` µs | serialise whole state µs |
|---|---|---|---|---|---|---|---|---|---|---|
| even | 21 | 7 | 14 | 5 | 384 | 14 / 16 / 16 | 13 | 1.3 | 6.9 | 10 |
| even | 201 | 67 | 134 | 8 | 3,250 | 16 / 29 / 33 | 24 | 3.9 | 17.1 | 176 |
| even | 2,001 | 667 | 1,334 | 10 | 36,146 | 18 / 28 / 34 | 30 | 15.1 | 8.3 | 1,226 |
| even | 20,001 | 6,667 | 13,334 | 11 | 378,399 | 18 / 28 / 34 | 32 | 257.0 | 22.8 | 21,618 |
| hub | 21 | 8 | 13 | 6 | 360 | 12 / 18 / 18 | 15 | 2.6 | 13.5 | 18 |
| hub | 201 | 68 | 133 | 19 | 3,241 | 14 / 46 / 68 | 44 | 5.0 | 9.1 | 151 |
| hub | 2,001 | 668 | 1,333 | 74 | 35,300 | 14 / 80 / 161 | 154 | 23.8 | 9.2 | 1,517 |
| hub | 20,001 | 6,668 | 13,333 | 174 | 371,793 | 14 / 62 / 356 | 355 | 297.1 | 18.1 | 15,694 |

Characters of the view (median, random requests): 242, 253, 307, 323 on the even graphs and 207, 227, 239, 254 on the hub graphs, from 20 to 20,000 items; for hub-targeted requests on the hub graph at 20,000 items the median view is 5,553 characters (355 items, 1.8 percent of the items) against 371,793 for the whole state. The share of requests whose view is over 10 percent of the whole state: 100 percent at 20 items, about 20 percent at 200, and 0 percent at 2,000 and 20,000, on both kinds. Memory: the first 20,001-item graph (even) took the process from 17,556 to 22,524 kB resident (about 5 MB) and 0.03 s to build; the hub graph was built later in the same process and the resident figure did not move, so only the first figure means anything. Every view matched the plain scan.

Readings against the predictions:
1. Eight graphs built within 5 percent, every view equal to the plain scan: **YES.**
2. Characters per item differ by less than 25 percent across sizes: **YES** (16.2 to 18.9 even, 16.1 to 18.6 hub).
3. Whole state at 20,000 items at least 500 times the characters at 20: **YES** (ratio 985 at the smaller of the two kinds).
4. Even graph, median view at 20,000 at most 3 times that at 200: **YES** (1.12; 18 against 16 items).
5. Even graph, 99th percentile at 20,000 at most 3 times that at 200: **YES** (0.97; 28 against 29).
6. Hub graph, random requests, 99th percentile at most 3 times: **YES** (1.35; 62 against 46).
7. Hub graph, random requests, largest view at least 3 times: **YES** (5.2; 356 against 68).
8. Hub graph, hub-targeted, median view between 1 and 10 percent of the whole state: **YES** (1.8 percent; the largest hub has 174 edges).
9. `gate.check` of a one-edit delta at 20,000 items at most 3 times that at 200: **NO.** It is 66 times (257 µs against 3.9 µs on the even graph; 297 µs on the hub graph).
10. `focus_graph` on the even graph at 20,000 at most 3 times that at 200: **YES** (1.33; 22.8 against 17.1 µs).
11. Serialising the whole state at 20,000 at least 50 times that at 200: **YES** (123).
12. Whole state over 24,576 characters at 2,000 items and under it at 200: **YES** (36,146 and 35,300 against 3,250 and 3,241).
13. Whole state at 20,000 items over 384,000 characters: **NO**, by a little: 378,399 (even) and 371,793 (hub), which at the assumed 3 characters per token is about 124,000 to 126,000 tokens, just under a 128,000-token window and with no room left for anything else.

What this shows and what it does not. With the whole state shown, the prompt grows in a straight line with the state from the first item (about 19 characters per item) and passes 8,192 tokens at about 1,300 items; at 20,000 items it takes essentially an entire 128,000-token window. With the focused view the part of the prompt that carries the state stays near 250 to 320 characters on a graph of even degrees, and that does not move from 200 to 20,000 items; on a graph with hubs it stays near 250 for most requests but a request that names the biggest hub shows 355 items (5,553 characters at 20,000 items, and growing with the size of the hub): the prompt does start to grow again there, and it grows with the degree of the named service, not with the size of the state. One of my predictions failed and it is a finding about the code: **`gate.check` is not flat in the size of the state.** It took 3.9 µs at 201 items and 257 µs at 20,001 items. From reading `gate.py` the cause is the line that builds the set of live services (`alive = (set(g.nodes) - del_nodes) | add_nodes`) on every call, so each check copies every service; I did not test this explanation and nothing was changed. In absolute terms 0.3 ms at 6,700 services is small beside the time a model takes to write a reply (tens to hundreds of milliseconds, Tests 21 and 35), but it is the one host cost here that grows with the state, and a straight-line extension (not measured) would put it near 10 ms at a million services. Not shown: that a model is as exact when it reads the view as when it reads the whole state (that needs a run); anything about graphs other than these two random kinds; the Rust crate; tokens (characters only, with an assumed 3 characters per token).

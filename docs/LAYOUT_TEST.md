# Does the order a graph is written in change what a stock model finds in it? A layout test on Qwen2.5-7B (written BEFORE the run, 15 Oct 2026)

This page fixes the question, the conditions, the measures, the predictions and the reading rules before any number from this test exists. Code: `python/axi/experiments/kv_layout.py`; notebook `python/colab/axi_tensor_tests.ipynb` (Cell 8); instrument checks `python/tests/test_kv_layout.py`. Concept-map item 98. It is the first of two pages written on 15 Oct 2026; the second is `KV_BLOCK_TEST.md` (item 99).

## Where the idea comes from, and what is and is not tested
On 15 Oct 2026 the owner shared a text (author and source not stated to me) that describes a "structural geometry language" at the level of a transformer's tensors. Its sections 1 and 3 (my reading of them) say that a model reads where things are from their position in the text, so that the same graph written in another order, or turned, is a different problem for it; the remedies named (orthogonal subspaces, Lie-group position codes, gauge connections) need a model trained for them. **None of those remedies is tested here, and none can be on a stock model.** The text claims things about Rajnish Choubey's formulations; I have not checked any of them with him and do not attribute them to him.

What a stock model does allow is the premise, and one cheap remedy on the host side:
* **The premise:** write the SAME graph and the SAME request in different orders and see whether the model finds the same things.
* **A remedy that needs no training:** write the graph so that what the request needs sits next to the thing it names ("structure-local" order), and see whether the model finds more.

## What I read from Run C before writing this page
Run C of the large test (`axi_large_c.json`, 160 steps, the rule stated, prompts of about 3,900 tokens) was read on 15 Oct 2026. Its `fresh` arm (the model reading the whole prompt afresh) answered 52 requests of the kind "delete the service at (r,c)", where the reply must list every edge that hangs on the service. I scored those replies with the scoring this page uses (`kv_layout.reply_metrics`, which reads a reply item by item so that a reply cut off at the token limit still counts for what it wrote):

| | Run C, `fresh`, 52 delete replies with dependents |
|---|---|
| replies exactly right when applied as written | 4 (8 %) |
| share of the service's own edges that the reply lists (**recall**) | **0.23** |
| ... of the edges that START at the service | **0.42** |
| ... of the edges that END at the service | **0.04** |
| edges written that are not in the graph, per reply | 2.85 |
| replies that ran into the 120-token limit | 14 of 52 (27 %) |

In the layout of every earlier test (services sorted, then edges sorted by where they start) the edges that start at a service sit next to each other; the edges that end at it are scattered over the list. The model found the first kind four times in ten and the second kind almost never. That is the observation this test is built on; the predictions below use it, so they are not blind to it. The test asks whether it is a property of where the edges sit.

## The question
For one graph and one request, does the order the items are written in change (1) how many of a service's own edges the reply lists, (2) whether the reply is exactly right, and (3) how many edges it invents?

## The five conditions
Every condition shows the model the same graph and the same request; only the order of the items (and, for `rot90`, the coordinates; for `adjacency`, one sentence) differs.

| condition | the state is written as |
|---|---|
| `sorted` | services sorted by coordinates, then edges sorted by (from, to, relation): the layout of every earlier KV test. The edges that START at a service are next to each other. |
| `shuffled` | the same services and the same edges in a random order (services still before edges). |
| `grouped` | each service followed by the edges that start at it. |
| `adjacency` | each service followed by ALL its edges, those that start at it and those that end at it, so every edge appears twice. The system prompt gets one more sentence: "In the state below every edge is written twice, once after each of its two services; in a reply write it once." About 1.65 times the tokens of `sorted`. |
| `rot90` | the whole problem turned a quarter turn (every coordinate in the state and in the request, (r,c) becomes (c,9-r)), then written sorted. One of the eight symmetries of the grid, so it is a probe and cannot show "rotation blindness" in general. |

## The problems
* **Graph:** 70 services on random cells of a 10 x 10 grid (values 1 to 9), 120 distinct edges (relation `dep` or `owns`, weight 1 to 9), one service in twenty with no edge; the generator of the large test (`kv_large.random_large_graph`). By the crude counter of the dry run (not Qwen's tokenizer) the prompt holds 2,230 tokens in `sorted` and 3,695 in `adjacency`; on the large test's prompts Qwen's own count was about 2 to 3 % higher than the crude one (3,785 crude, 3,877 real), so I expect about 2,300 and 3,800. That is the size at which Run C's observation was made, a little smaller, and `adjacency` stays under the 3,900 tokens that Run C's prompts reached on the same T4.
* **42 delete problems:** 14 graphs, 3 services each, chosen among the services with 1 to 3 edges (5 with one edge, 15 with two, 22 with three; 2.4 on average; of their 101 edges, 53 start at the service and 48 end at it). The request is "Delete the service at (r,c)." and the exactly right reply lists the service and every edge that hangs on it.
* **8 control problems:** on the first 8 graphs, "add an edge" (4) and "add a service" (4). The reply is a copy of the request and needs no search; they show whether a layout breaks the model's basic job.
* **The rule is stated** in the system prompt (a service can only be removed in the same delta that removes every edge attached to it), as in Run C.
* Model, grammar and decoding as in every earlier 7B test: Qwen2.5-7B-Instruct in 4-bit (NF4) on the free Colab T4, greedy decoding under the delta grammar, at most 120 new tokens. Seed 31.
* **Each problem is read afresh under each condition** (no cache is kept), 50 problems x 5 conditions = 250 reads. **Cost: $0.** Time on a free T4, my estimate and nothing more: 35 to 45 minutes; the smoke test in the notebook prints the real projection.

## What is measured
Per problem and condition: the reply; whether it is exactly right when applied as written and with `repair_m6` (`kv_drop.score_reply`); **recall** (the share of the service's own edges that the reply lists, exactly as written), split into edges that start at the service and edges that end at it; the edges the reply lists that are not in the graph (**invented**) or are in the graph but are not the service's (**other**); whether the reply ran into the token limit; the prompt size; seconds. Problem by problem against `sorted`: how many are better, how many worse, and an exact sign test.

## Checks before the run (the notebook stops if one fails)
`kv_drop.preflight` and `kv_block.preflight_blocks` on the real model (Cell 6, shared with the block test); the chat template of the tokenizer must equal the ChatML text the test builds; one more line checks that the `adjacency` prompt (the system prompt with its extra sentence) is made of the tokenizer's own tokens; the tokens built piece by piece must equal the tokenizer's own tokens for the whole prompt text (recorded for every probe, line 2).

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error, except line 4: if `sorted` leaves no room, the lines about change say little.

1. [instrument] The graph is large enough to matter: the prompt of `sorted` holds on average at least 2,000 tokens. (85 %)
2. [instrument] The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every probe of every condition. (85 %)
3. [instrument] Every condition answered the same 42 delete problems and the same 8 control problems. (99 %)
4. [guard] sorted: the recall (the share of a service's own edges that the reply lists) is between 0.15 and 0.90, on the delete problems (there is room to see a change in either direction). (85 %)
5. [claim] shuffled: the recall is at least 0.10 below sorted's, on the delete problems. (65 %)
6. [claim] sorted: the recall of the edges that START at the service is at least 0.10 above the recall of the edges that END at it. (92 %)
7. [claim] adjacency: the recall is at least 0.15 above sorted's. (75 %)
8. [claim] adjacency: the share of delete replies that are exactly right when applied as written is at least 15 points above sorted's. (55 %)
9. [claim] grouped: the recall is at least 0.05 above sorted's. (35 %)
10. [claim] rot90: the recall is within 0.10 of sorted's. (70 %)
11. [claim] adjacency: the edges a delete reply invents (edges that are not in the graph), per reply, are at least 30 % fewer than sorted's. (60 %)
12. [claim] control: in every condition at least 85 % of the add replies are exactly right when applied as written. (80 %)
13. [claim] shuffled: the reply is not the same text as sorted's on at least 20 % of the delete problems (the order changes what the model writes). (95 %)

**Headline** (lines 5, 7 and 8): if all three read "yes", then on this model and this task the order the same facts are written in changes how many of a service's edges the model finds, and writing each service's edges next to it, in both directions, finds clearly more of them and gets more replies exactly right. The product of my three probabilities is about 27 %; they are positively correlated (if the model can read a local list, 7 and 8 move together), so my single number is about **35 %**.

Why I think so, line by line. *Line 6 (92 %)* is already seen in Run C (0.42 against 0.04) at a larger size; the doubt is only that the smaller graph might lift the second number. *Line 5 (65 %):* in `sorted` the edges that start at a service are contiguous and the model finds some of them; scattering them should cost some, but the model may also find them by matching "(r,c)->" anywhere, and at 2,300 tokens the loss may be under 0.10. *Line 7 (75 %):* in `adjacency` the answer sits in the lines right after the service's own line, both directions; the experience of Run C says the model does not find what is far away, so I expect a gain, not a certain one: the doubled list might confuse it, and one added sentence about the format is a thin guide. *Line 8 (55 %):* "exactly right" needs every edge, no invented ones and a stop; Run C shows 27 % of replies running on to the limit, and a better recall does not by itself stop that. *Line 9 (35 %):* `grouped` changes little: the contiguous run is already there; what it adds is that the service's own line is next to it. *Line 10 (70 %):* digits are separate tokens and the model reads coordinates as strings, so a turned grid is a different string problem of the same difficulty; a single frame is noisy with 42 problems. *Line 11 (60 %):* if the right edges are near, fewer invented ones are needed to fill the reply. *Line 12 (80 %):* Run A and Run C each got every add reply right (89 of 89 in each); the doubled list is the only layout that could disturb that. *Line 13 (95 %):* greedy decoding on a shuffled list will not write the same edges.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *Lines 7 and 8 yes:* the model's trouble in Run C is where the edges sit, not what it can do; a structure-local layout is a fix available today on the host side with no training, at the price of a longer prompt; it does not show that Lie-group or gauge position codes work (untested) and the next experiment is a bigger graph and a second model. *7 yes, 8 NO:* the model finds more of the edges and still does not stop or still invents; the next fix is on the output side (a limit, a grammar that knows the graph). *7 NO:* the layout is not the lever for this model at this size, and the premise of the text (that order is what hurts) is not supported here. *5 yes and 7 NO:* order hurts when it is scrambled but a better order than the usual one does not help: the usual layout is already as good as this model gets. *5 NO and 6 yes:* the model is order-insensitive in the large and direction-sensitive in the small. *Line 4 NO:* no room; read lines 5 to 11 as direction only. *Line 10 NO:* a quarter turn changes what the model finds, so a host that turns the problem into a fixed frame (the D4 canonical form of the repository) would be worth testing as a model-side fix, and one frame is not eight.

## Reading rules (fixed now; the notebook prints exactly these)
The thirteen lines above are the reading rules; `kv_layout.verdict_layout` computes them from the saved rows. "Recall" is the share of the service's own edges (those that start at it and those that end at it) that the reply lists exactly as written (relation and weight right); a reply is read item by item, so a reply cut off at the limit counts for what it wrote before the cut. "Exactly right when applied as written" is `exact_blind` of `kv_drop.score_reply` on the reply as parsed (a reply cut off at the limit does not parse and is not exactly right). "Invented" counts the edge deletions of a reply that name an edge that is not in the graph exactly as written. The control lines use the 8 add problems only. With 42 problems a difference of ten points is about four problems: the lines are thresholds fixed in advance, not tests of significance, and the printout adds the exact sign test for the exactness counts.

## What this cannot show
One model (4-bit weights), one task (find a service's edges and list them), one graph size, 42 problems, greedy decoding, single reads (no episode, no cache, no teacher forcing). One list format and one added sentence for `adjacency`. The `adjacency` condition changes the length of the prompt and the system prompt along with the order, so a gain could in part come from the sentence; I did not add a "sorted with the sentence" condition. `rot90` is one of eight frames. Nothing about training, so nothing about the position codes the text proposes. A bigger graph (the 4,000-token prompts of Run C), a bigger model and a second model family are not covered; `adjacency` at 4,000 tokens of services and edges would need about 7,000 tokens and does not fit comfortably on the free T4.

## Related work (the three papers exist as named; titles, authors, years and venues were confirmed by a search on 15 Oct 2026, what they report is from my memory of their abstracts and was not re-read; check before citing)
That the position of information in a long prompt changes whether a model uses it is reported in *Lost in the Middle* (Liu et al., 2023; TACL 2024); that the order of premises changes reasoning accuracy in *Premise Order Matters in Reasoning with Large Language Models* (Chen et al., ICML 2024); that the way a graph is written as text changes what a model can do with it in *Talk like a Graph: Encoding Graphs for Large Language Models* (Fatemi et al., 2023; ICLR 2024). If these hold as I remember them, the finding that layout matters would not be new; what would be new here is the measured size on this task and this model, and the host-side fix.

## How it is run
`axi_tensor_tests.ipynb` on Colab (T4): upload `axi-colab-tensor.zip`, connect Google Drive, run the cells top to bottom: GPU check, upload, install, Drive, load the model, the checks (stops if one fails), a smoke test with a projection, then Cell 8 (this test). Each finished problem is saved to Google Drive, so a dropped session continues where it stopped. The result file is JSON; send it back and this page gets a "Result" section without changing anything above this line.

---

# Result (run 15 Oct 2026; everything above this line is unchanged from before the run)

*The run was made on a free Colab T4 on 15 Oct 2026 (Qwen2.5-7B-Instruct, 4-bit, seed 31, 50 problems read under 5 conditions = 250 reads, 37.6 minutes, `axi_tensor_tests.ipynb` Cell 8). The saved file is `docs/results/axi_layout_Qwen2.5-7B_seed31.json`. Every number below is recomputed from the saved rows, and every reply is scored again from its text, by `python/tests/test_kv_layout_result.py` (test 111). The block test (Cells 9 and 10) was not run.*

**The headline failed, on all three of its lines.** Lines 5, 7 and 8 all read NO. Of the thirteen lines, seven read yes (1, 2, 3, 4, 6, 12, 13) and six read NO (5, 7, 8, 9, 10, 11). I had put the three headline lines together at about 27 % and my single number was 35 %.

## The thirteen lines, against what I wrote before the run
| # | my probability | read | what the saved rows say |
|---|---|---|---|
| 1 | 85 % | yes | the prompt of `sorted` held 2,336 tokens on average (2,335.6) |
| 2 | 85 % | yes | the pieces equal the tokenizer's own tokens on all 250 probes (`canonical` = 1.0 in every condition) |
| 3 | 99 % | yes | 42 delete problems and 8 control problems in every condition |
| 4 | 85 % | yes | `sorted` recall 0.333 (between 0.15 and 0.90) |
| 5 | 65 % | **NO** | `shuffled` recall 0.329 against 0.333: lower by 0.004 (the line needed 0.10) |
| 6 | 92 % | yes | `sorted`: edges that START at the service 0.534, edges that END at it 0.103: a gap of 0.431 (Run C of the large test: 0.42 and 0.04) |
| 7 | 75 % | **NO** | `adjacency` recall 0.373 against 0.333: higher by 0.040 (the line needed 0.15) |
| 8 | 55 % | **NO** | `adjacency` exactly right as written on 4 of 42 replies (9.5 %), `sorted` on 9 of 42 (21.4 %): **11.9 points worse**, not 15 better |
| 9 | 35 % | **NO** | `grouped` recall 0.302 against 0.333: lower by 0.032 (the line needed +0.05) |
| 10 | 70 % | **NO** | `rot90` recall 0.440 against 0.333: higher by 0.107, outside the 0.10 band by 0.007 |
| 11 | 60 % | **NO** | `adjacency` invents 2.71 edges per reply, `sorted` 1.64: **65 % more**, not 30 % fewer |
| 12 | 80 % | yes | control: 8 of 8 add replies exactly right in every condition |
| 13 | 95 % | yes | `shuffled` wrote a different text from `sorted` on 37 of 42 delete problems (88 %) |

## The table the notebook printed (42 delete problems per condition; the repair-step and cut-off columns are measures the page named and no rule uses)
| condition | exact as written | exact with the repair step | cut off at 120 tokens | recall | recall, edges that start at the service | ... that end at it | invented per reply | prompt tokens | seconds per read |
|---|---|---|---|---|---|---|---|---|---|
| `sorted` | 9 (21.4 %) | 33 (78.6 %) | 5 (12 %) | 0.333 | 0.534 | 0.103 | 1.64 | 2,336 | 7.5 |
| `shuffled` | 2 (4.8 %) | 19 (45.2 %) | 21 (50 %) | 0.329 | 0.529 | 0.098 | 2.81 | 2,336 | 8.8 |
| `grouped` | 8 (19.0 %) | 41 (97.6 %) | 0 (0 %) | 0.302 | 0.552 | 0.034 | 1.45 | 2,336 | 6.6 |
| `adjacency` | 4 (9.5 %) | 35 (83.3 %) | 7 (17 %) | 0.373 | 0.379 | 0.391 | 2.71 | 3,921 | 13.7 |
| `rot90` | 8 (19.0 %) | 26 (61.9 %) | 11 (26 %) | 0.440 | 0.695 | 0.103 | 2.07 | 2,336 | 8.5 |

Problem by problem against `sorted` (better / worse, exact two-sided sign test, as printed): exact as written, `shuffled` 1 / 8 (p = 0.04), `grouped` 2 / 3 (p = 1.0), `adjacency` 3 / 8 (p = 0.23), `rot90` 3 / 4 (p = 1.0); with the repair step, `shuffled` 3 / 17 (p = 0.003), `grouped` 8 / 0 (p = 0.008), `adjacency` 6 / 4 (p = 0.75), `rot90` 5 / 12 (p = 0.14). "Exact with the repair step" is `repair_m6` (`diagnostics.ground_and_complete`, the repository's own repair: it keeps the edge deletions that are true, cuts those that are not in the graph, and adds the deletions the reply left out when it deletes a service). The recall start and end columns are taken over the 29 problems of the 42 that have edges of that kind.

## Where my numbers were off
* **Line 5 (65 %).** I expected the loss from scrambling the list to show in the recall. It does not: 0.329 against 0.333, and the edges that start at the service are found as often in a random order (0.529) as in the sorted one (0.534). The model does not use the fact that they sit next to each other. This was the alternative I named on the page ("the model may also find them by matching `(r,c)->` anywhere").
* **Lines 7, 8 and 11.** I gave 75 % to the doubled list finding clearly more edges, and it does not (+0.04); the exact replies fell instead of rising, and it invents more edges, not fewer. I had said the doubled list "might confuse it". It did, in a way I did not foresee (the arrow, below).
* **Line 6, second number.** I wrote that the smaller graph "might lift the second number": 0.10 here against 0.04 at Run C's size. The gap is still 0.43.
* **Line 9 (35 %), line 10 (70 %).** Both NO, both by small margins (0.032 and 0.007).
* **What I did not expect at all:** how much of what looks like a miss is the right edge written with its arrow turned around, and how much the cut-off at 120 tokens depends on the order (50 % of the `shuffled` replies, 0 % of the `grouped` ones).

## What the pre-registered rules say, in the words I wrote before the run
Lines 5 NO and 6 yes: *"the model is order-insensitive in the large and direction-sensitive in the small."* Line 7 NO: *"the layout is not the lever for this model at this size, and the premise of the text (that order is what hurts) is not supported here."* Line 10 NO: *"a quarter turn changes what the model finds, so a host that turns the problem into a fixed frame ... would be worth testing as a model-side fix, and one frame is not eight."* I keep the first two as written. For the third, see below: the data cannot tell the shift from chance (p = 0.42).

## What I read from the rows afterwards (NOT pre-registered; each of these is a look at a table I had already seen, so read them as reasons to test again, not as findings)
1. **Scrambling the order does not lose edges; it makes the reply not end.** `shuffled`: 21 of 42 replies run into the limit (`sorted` 5, `grouped` 0), 2.81 invented edges per reply against 1.64, 2 replies exactly right as written against 9, 19 with the repair step against 33 (3 better, 17 worse). Of these, the repair figure survives a correction for the eight sign tests printed above (p = 0.003 x 8 = 0.02).
2. **`grouped` (each service followed by the edges that start at it) is the layout whose replies end and stay inside the service.** None of its 42 replies ran on, it names 1 real edge that is not the service's against 24 for `sorted` (11 for `shuffled`, 36 for `rot90`, 0 for `adjacency`), and with the repair step 41 of 42 are exactly right, against 33 (8 better, 0 worse, p = 0.008, which does NOT survive a correction for eight tests: 0.06). Its recall is no higher (0.302). I take it as a candidate to test with a new seed and more problems, not as a result. Four of its eight gains are replies of `sorted` that ran into the limit; the other four are replies that ended but named edges that are not the service's.
3. **With the repair step the outcome follows the reply's ending and what it deletes by mistake, not the length of its edge list.** None of the 44 replies that failed to parse was repaired to the right state, and almost every reply that did parse was: 33 of 37 for `sorted`, 41 of 42 for `grouped`, 35 of 35 for `adjacency`, 19 of 21 for `shuffled`, 26 of 31 for `rot90`. **Each of the 12 replies that parsed and were still wrong after the repair had deleted at least one real edge that belongs to another service** (the repair keeps what is true, so it cannot tell that the edge is not the service's): 4, 2, 1, 0 and 5 of them in the five conditions. The model finds about a third of the service's edges (0.30 to 0.44 in every condition) and the host's repair turns that into 62 to 98 % of deletes exactly right in four of the five orders (45 % when the order is scrambled). That is the split of work the repository is built on (the model proposes, the host closes the boundary), measured here on a delete the model gets right only one time in five as written.
4. **A good part of the "misses" is the arrow.** Counting an edge as found when the reply writes it with the same relation and weight but the arrow turned around: 13 of the 101 edges that hang on the 42 services in `sorted` were written that way (12 of the 53 edges that start at the service, 1 of the 48 that end at it), 1 in `shuffled`, 14 in `grouped`, **22 in `adjacency`** (16 and 6), 8 in `rot90`. Found in either direction the shares are 0.475, 0.366, 0.455, **0.604** and 0.505 (pooled over the 101 edges). So in `sorted` the edges that end at the service are not written backwards (1 of 48); they are not written at all (4 of 48 as written), which is the Run C observation, here again. In `adjacency` the model finds more neighbours (paired against `sorted`, direction ignored: +0.151, 95 % interval -0.03 to +0.33, 19 problems better and 8 worse, p = 0.052) and writes the arrow of the edges that start at the service backwards 16 times in 53.
5. **`adjacency` does what I hoped for the edges that end at the service and loses what it gained on the others.** Recall of the edges that end at the service: 0.391 against 0.103 (+0.287, 95 % interval +0.09 to +0.48, 14 problems better and 2 worse, p = 0.004). Recall of those that start at it: 0.379 against 0.534 (-0.155, interval -0.39 to +0.07, 5 better and 10 worse). The sum is line 7's +0.04. It costs 1.68 times the tokens (3,921 against 2,336), 1.8 times the time per read (13.7 s against 7.5 s) and 10.1 GB of GPU memory against 7.3 GB, and the page said that its added sentence ("write it once") could account for part of the change: I did not run `sorted` with the sentence.
6. **`rot90` is higher by 0.107 and the data cannot tell that from chance:** 95 % interval -0.01 to +0.23, 9 problems better and 5 worse, p = 0.42. By the rule it reads NO, by 0.007. It does not say anything about rotation in general (one frame of eight; 42 problems).

## What this shows and does not show
* **Shows (this model, 70 services and 120 edges, 42 delete problems, greedy decoding, the rule stated):** the order the same facts are written in does not change how many of a service's own edges the model finds (0.30 to 0.37 in four of the five conditions, 0.44 in the quarter-turned one, a gap the data cannot tell from chance); the usual layout is not distinguishably worse than the alternatives on that measure; scrambling it makes the replies run on and invent more; and the edges that end at the service stay unfound in the usual layout (4 of 48 written) as in Run C.
* **Does not show:** that `grouped` is better (post hoc, p = 0.06 after correction); anything about rotation; anything about bigger graphs, other models, sampling or other tasks; anything about the position codes of the shared text (they need training and were not tested); whether the added sentence or the order is what changed `adjacency`; whether the arrow errors are the cause of the low recall or a symptom (a prompt that states the direction was not run).
* **What it does to the text's premise:** the premise (the same graph in another order is another problem for the model) does not hold for how many true edges are found, and, as a look afterwards, does hold for how the reply ends and for the number of invented edges.

## What to do next (not run)
* A pre-registered replication of `grouped` against `sorted` on a new seed with about 100 delete problems: the line to write first is "fewer replies cut off and more exactly right with the repair step", with the thresholds set from this table.
* The direction: a request that states which way the arrows run, tested on `sorted` and `adjacency`, and the same with the sentence added to `sorted`.
* The block test (`KV_BLOCK_TEST.md`, Cells 9 and 10) is still waiting; this result does not change what it can show. *(Added 16 Oct 2026: the sentence above is as it stood on 15 Oct. The block test has since been run twice, B1 and B2; the headline of its first page failed. See the Result of `KV_BLOCK_TEST.md`.)*

## Reproduce
`python/tests/test_kv_layout_result.py` (no GPU, a few seconds) re-reads the saved file, scores all 250 replies again from their text with `kv_layout.reply_metrics` and `kv_drop.score_reply`'s own functions, recomputes the thirteen lines and every figure above, and checks that the part of this page above the Result line is byte for byte the text written before the run (sha256 `c9c3cd72018428d55338b2f350b3605a4fc50a7091319badbbbe439ed2f142cc`). The test was itself checked with 89 deliberate changes (to the saved file, to this page and to the scoring code): 88 are caught by it; the one that is not (the order of the DEL items in the ideal reply, which no score reads) is caught by `test_kv_layout.py`.

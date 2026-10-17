# Can a stock model read every item of its state in its own block? Block independence without training, on Qwen2.5-7B (written BEFORE the run, 15 Oct 2026)

This page fixes the question, the arms, the measures, the predictions and the reading rules before any number from this test exists. Code: `python/axi/experiments/kv_block.py` (the reading rules, the checks on the real model) and `BlockArm` in `kv_drop.py`, on top of `kv_large.py` and `kv_torch.py`; notebook `python/colab/axi_tensor_tests.ipynb` (Cells 9 and 10); instrument checks `python/tests/test_kv_block.py`. Concept-map item 99. It is the second of two pages written on 15 Oct 2026; the first is `LAYOUT_TEST.md` (item 98). It builds on `KV_DROP_TEST.md` (item 96) and `KV_LARGE_TEST.md` (item 97) and reuses their machinery unchanged.

## Where the idea comes from, and what is tested
On 15 Oct 2026 the owner shared a text (author and source not stated to me) that describes a "structural geometry language" at the level of a transformer's tensors. Its section 2 proposes "sheaf-factorized block attention": once a block of the state is committed, only a thin interface of it is visible to later blocks, and its table lists this as "no pre-training required: applied as an attention mask at test time". Nothing in the text says it was measured, and I have not checked with Rajnish Choubey that it is his formulation; I do not attribute it to him.

A stock model can be asked exactly one version of this without training: **compute the key/value rows of every item on its own, with the header in view and the other items NOT in view, put the rows side by side, and let the request read all of them.** No mask is needed; each item is read in a forward pass of its own, at the position it would have in a normal read. If a stock model keeps its answers under that, two things follow that matter for a runtime: (a) **deleting an item is exact for that model** (the cut leaves no stale row, because nothing was computed in the item's presence), and (b) **adding an item costs only that item.** If it does not keep them, "no pre-training required" is false for this model and this task, and the literature below says why that would not be a surprise.

## What the earlier pages found, which this page is built on
`KV_LARGE_TEST.md` (Runs A and C on the same T4, read on 15 Oct 2026): in the cache arms that cut a deleted item out (`drop_compact`), at the last step of an episode 93 % (Run A) and 92 % (Run C) of the state rows were stale (computed while a since-deleted item was visible), and the value rows of the cache differed from those of a fresh read by 0.22 and 0.21 (mean relative L2 difference over the layers, at that step).
* **No rule in the prompt (Run A):** the replies are short copies of the request (11 tokens on average); `drop_compact` wrote exactly the same reply as a full rebuild on 160 of 160 steps.
* **Rule stated (Run C):** the replies have to find the service's edges in the list; the rebuild itself is poor at that (4 of 52 delete replies exactly right, recall 0.23, 14 of 52 replies running on to the 120-token limit), and `drop_compact` wrote the same reply as the rebuild on 71.9 % of steps (37 % on the 71 delete steps), with the accuracy after repair 3 points lower (79.4 % against 82.5 %, 11 steps lost, 6 gained: not distinguishable from chance with 160 steps). The differences were different invented edges, not edges brought back from deleted items (none were).

So a stock model already tolerates a moderate amount of "wrong context" in its rows when the reply is a copy, and the reply of a model that is guessing is not a fair thing to demand identity of. Independent blocks are a much bigger change than a stale tail: **every** row loses the context of **every** other item. That is why the test has two regimes, and why the predictions below are lower than Run A's results would suggest.

## The question
Does a stock model, with every item (or every 16 items) of its state read in a block of its own, write the same reply as when everything is read together, and is the saving what the design promises?

## The arms
Each step, every arm answers the same logical prompt (header, the items of the oracle's state, the request); the arms differ in how the key/value rows of the state are obtained.

| arm | how the rows for the step are obtained | status |
|---|---|---|
| `fresh` | the whole prompt is read at once | the reference |
| `drop_compact` | the cache of the large test: a deleted item is cut out, the keys behind it are turned to close the gap, nothing is read again | approximate: stale rows (here for comparison) |
| `block` | **every item is a block of its own**: read with the header and itself in view, no other item. The cache is kept between steps: a deleted item's rows are cut, **nothing is turned** (a row's deeper layers were computed at its position and its distance to the header, so turning the keys alone would not equal a rebuild), an added item is read as a new block (it sees the header only) after the last row, leaving a hole in the position numbers as `drop_gap` does. The request and the reply are cropped after each step | exact for the independent-block model |
| `block16` | blocks of 16 consecutive items, rebuilt from nothing every step with contiguous positions | accuracy only; no speed is claimed |

Every `CHECK_EVERY` = 10 steps (steps 1, 11 and 21 of an episode) the kept `block` cache is compared with one rebuilt from nothing at the same positions: the mean relative difference of the values and the largest relative difference of the keys (first and last layer). The first comparison in each episode is the first build against itself and is zero by construction.

## Design
* **Model, grammar, decoding:** as in every earlier 7B test (Qwen2.5-7B-Instruct, 4-bit NF4, free Colab T4, greedy under the delta grammar, at most 120 new tokens).
* **Prompt and script:** exactly the large test's: a graph of 120 services and 200 edges (about 3,800 tokens), the same four kinds of request, teacher forcing (every arm sees the oracle's state), the **first steps of the same request scripts** (seed 21, episode seeds 21000 to 21002). Nothing about the request or the layout changes, so the numbers can be read against Runs A and C.
* **Run B1 (no rule):** 3 episodes x 30 steps, four arms. **Run B2 (rule stated, optional):** 3 episodes x 20 steps, the same four arms.
* **The checks on the real model before the run** (`kv_block.preflight_blocks`, seven lines): one block holding the whole state gives the same logits as a normal read; the rows of block B do not change when block A has other tokens; and that check can tell (A's own rows do change); in a normal read B's rows DO change when A changes; a cut leaves exactly the cache of a build without A at the same positions; a block added afterwards equals the same block built together with the others; the request reads every block. The tolerances are the ones of the first page (0.5 in logits, 0.05 relative in keys).
* **Cost: $0.** Time on a free T4, my estimate and nothing more: each item is its own forward pass, so the first build of an episode is about 320 passes (a minute or less); B1 25 to 30 minutes, B2 25 to 30 minutes. The smoke test in the notebook prints seconds per step.

## What the script alone fixes (a dry run, no model)
`kv_block.block_dry_run` counts with the crude counter of the large test (a word, a digit or a run of punctuation is one token; **not Qwen's tokenizer**, so the real figures differ by some percent): the first prompt of these scripts holds about 3,800 tokens; `block` (which appends the same items and the same request as `drop_compact` and reads nothing again) reads **4.5 %** of what `fresh` reads over run B1 (3 x 30 steps, first builds included) and 6.1 % over run B2 (3 x 20). Line 10 is this, and my probability for it is high because of it.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error, except lines 3 and 4: if the kept cache does not equal the rebuilt one, or if `fresh` itself cannot do the task, the other lines say little.

1. [instrument] The graph is large: the prompt of `fresh` holds on average at least 3,000 tokens over the steps of run B1. (97 %)
2. [instrument] The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run B1. (95 %)
3. [instrument] The kept block cache equals one rebuilt from nothing at the same positions: at every step where this was checked (run B1) the mean relative difference of the values is at most 0.01 and that of the keys at most 0.05. (93 %)
4. [guard] fresh: the share of steps on which repair_m6 turns the reply into the right next state is at least 90 % (run B1). (95 %)
5. [claim] block: the reply is identical to fresh's on at least 90 % of the steps of run B1. (55 %)
6. [claim] block: the share of steps on which repair_m6 turns the reply into the right next state is at most 3 points below fresh's (run B1). (65 %)
7. [claim] block: on at least 90 % of the steps of run B1 the largest KL along the reply, against fresh's distribution, is below 0.5 nats. (50 %)
8. [claim] block16: the reply is identical to fresh's on at least 90 % of the steps of run B1. (65 %)
9. [claim] block: the mean relative difference of its values from fresh's is larger than drop_compact's (run B1): leaving out all other items is a bigger change than a stale tail. (85 %)
10. [claim] block reads at most 10 % of the tokens that fresh reads (run B1, whole episodes, the first build of each included). (95 %)
11. [claim] block: mean seconds per step at most 0.5 times fresh's (run B1). (85 %)
12. [claim] run B2 (rule stated): the share of steps on which repair_m6 turns the reply into the right next state is, for block, at most 5 points below fresh's. (40 %)
13. [claim] run B2 (rule stated): the share of replies that are exactly right when applied as written is, for block, at most 5 points below fresh's. (55 %)
14. [claim] run B2 (rule stated): block's reply is identical to fresh's on at least 70 % of the steps. (40 %)

**Headline** (lines 5, 6, 10 and 11): if all four read "yes", then on this model, on this task, in the regime where the reply is a copy, reading every item of a 4,000-token state in a block of its own, with no training, kept the replies and made a delete exact and an add cost one item, at a tenth of the reading and half the time. The product of my four probabilities is about 29 %; 5 and 6 are positively correlated (if the replies are the same the repaired state is the same), so my single number is about **40 %**, and I would still bet slightly against it.

Why I think so, line by line. *For:* the items are short (about ten tokens) and self-contained (a coordinate and a value, or two coordinates, a relation and a weight); the first layers of a row depend only on its own tokens; Run A showed identical replies with 93 % of the rows stale and a 22 % value difference; the replies of run B1 are copies of the request. *Against:* every row loses every other item here, so the difference is certainly larger than Run A's (line 9); in Run C a smaller change moved two replies in three on the delete steps; the three papers I know on independent chunks all report a loss without fine-tuning or recomputation (below); the 4-bit weights add noise. *Line 5 (55 %) and line 7 (50 %):* a reply of 11 tokens has few places to differ, but "identical" is strict and KL < 0.5 is looser than it looks on a model that moves. *Line 6 (65 %):* `repair_m6` absorbs differences of the kind Run C showed. *Line 8 (65 %):* bigger blocks are closer to a normal read. *Line 11 (85 %):* the first build of 320 blocks is slow and is counted. *Lines 12 to 14* are about the regime where the reply has to find the service's edges: the rebuild is poor there, so identity is a demanding thing to ask of any variant, and I put them at 40 to 55 %.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *Lines 5, 6, 10, 11 yes:* "no pre-training required" holds for this model in the copy regime; the next questions are whether it holds when the reply has to find things (lines 12 to 14 say so for this task), a second model family and a bigger model, and what the design costs when the items are longer. *5 NO and 6 yes with 8 yes:* the replies move but repair absorbs it, and bigger blocks lose less: a stock model needs some cross-item context, and the next test is the block size (2, 4, 8) and blocks that follow the graph (a service with its edges). *5 and 8 NO:* the claim is false for this model; a model trained with the block mask (Block-Attention does this for retrieval) is what the design would need. *Line 3 NO:* the kept cache is not the rebuilt one, the implementation or the numerics are to be looked at before anything else is read. *Line 9 NO:* independence is a smaller change than I think and the stale-row results are not a good guide to it. *Lines 12 to 14 NO with 5 and 6 yes:* independent blocks work where the reply is a copy and fail where it has to search, which is the shape of the earlier results.

## Reading rules (fixed now; the notebook prints exactly these)
The fourteen lines above are the reading rules; `kv_block.verdict_block` computes them from the saved rows. "Identical" means the same tokens including the end token. "Steps" are all steps of all episodes of the run. "Largest KL along the reply" is the largest, over the positions where the two replies agree so far, of the KL divergence in nats between the two distributions over the tokens the grammar allows. A line whose run was not made reads "not run". Line 3 uses the checks of steps 1, 11 and 21 of every episode of run B1. Line 9 compares the mean over all steps of the relative L2 difference of the value rows with `fresh`'s, averaged over layers (`value_deviation`).

## What this cannot show
One model (4-bit weights), one task family, greedy decoding, 30 and 20 steps, a free T4, teacher forcing. Only the first steps of the large scripts, and a graph written in one layout (services, then edges): blocks that follow the graph are not tested here (see `LAYOUT_TEST.md` for the layout). Independence is tested at the level of the **rows** (a forward pass per block); a trained model, a block mask inside the attention kernel and the interface idea of the text (a block that exposes only a few of its tokens) are not. The positions of the kept cache have holes, as in `drop_gap`; a build with contiguous positions (`block16` does this) is a slightly different model. Speed is measured with one forward pass per item and no batching, which is how the test is built, not how a serving stack would build it.

## Related work (checked 14 Oct 2026 for the first page, not a survey)
* *Prompt Cache* (Gim et al., MLSys 2024): reuses precomputed attention states of prompt modules at other positions, accuracy comparable to the baseline on the modules and tasks they report.
* *CacheBlend* (Yao et al., EuroSys 2025): chunks cached independently lose the attention between chunks; recomputing about 15 % of the tokens restores quality. This is the closest published relative of the question here: it says independent caches degrade and that a small recomputation repairs them.
* *Block-Attention for Efficient RAG*: independent blocks with positions re-encoded by RoPE rotations; the model has to be fine-tuned with the block mask.
If these are right, a loss without training is the expected result for passages of hundreds of tokens; what this test adds is the case of ten-token self-contained lines, with a measured size.

## How it is run
`axi_tensor_tests.ipynb` on Colab (T4): the cells of `LAYOUT_TEST.md` up to the checks, then Cell 9 (run B1) and, optionally, Cell 10 (run B2). Each finished episode is saved to Google Drive, so a dropped session continues where it stopped. The result files are JSON; send them back and this page gets a "Result" section without changing anything above this line.

---

# Result (run 16 Oct 2026; everything above this line is unchanged from before the run)

*The run was made on a free Colab T4 on 16 Oct 2026 (Qwen2.5-7B-Instruct, 4-bit, seed 21, no rule in the prompt, 3 episodes x 30 steps x 4 arms = 360 replies; the progress counter said 2,196 s, the step times in the file add up to 1,914 s; `axi_tensor_tests.ipynb` Cell 9). The saved file is `docs/results/axi_block_b1_Qwen2.5-7B_seed21.json`. Every number below is recomputed from the saved rows, and every reply is scored again from its text, by `python/tests/test_kv_block_result.py` (test 112). Run B2 (Cell 10, the rule stated) was made a few hours later and is read in its own section below, "Run B2: the rule stated"; the part of this Result before that section was written before run B2 and is changed only where it says "updated after run B2".*

**The headline failed on two of its four lines.** Lines 5 and 6 read NO; lines 10 and 11 read yes. Of the eleven lines this run can read, eight read yes (1, 2, 3, 4, 8, 9, 10, 11) and three read NO (5, 6, 7). I had put the four headline lines together at about 29 % and my single number was 40 %.

In plain words: the part of the design that is arithmetic held (the kept cache equals a rebuild exactly, and `block` reads 4.3 % of the tokens in 0.33 times the time). The part that needs the model to cooperate did not: with every item in a block of its own, the stock model wrote a different reply on **every one of the 40 delete steps** and the repair step could not rescue a single one. Blocks of 16 kept the replies (95.6 %) in this run (updated after run B2: with the rule stated in the prompt, run B2 below, they did not).

## The fourteen lines, against what I wrote before the run
| # | my probability | read | what the saved rows say |
|---|---|---|---|
| 1 | 97 % | yes | the prompt of `fresh` held 3,849.8 tokens on average (3,545 to 4,008) |
| 2 | 95 % | yes | the pieces equal the tokenizer's own tokens on all 90 steps (`canonical` = 1.0) |
| 3 | 93 % | yes | kept `block` cache against a rebuild at the same positions: 9 checks, the difference of the values and of the keys is exactly 0.0 in every one (the 3 checks at step 1 are the build against itself and are zero by construction; the 6 at steps 11 and 21 are real comparisons) |
| 4 | 95 % | yes | `fresh`: `repair_m6` turns the reply into the right state on 90 of 90 steps (100 %) |
| 5 | 55 % | **NO** | `block` identical to `fresh` on 43 of 90 steps (**47.8 %**); the line needed 90 % |
| 6 | 65 % | **NO** | `block` right after `repair_m6` on 43 of 90 (**47.8 %**), `fresh` on 90 of 90: **52.2 points below**, the line allowed 3 |
| 7 | 50 % | **NO** | largest KL below 0.5 nats on 36 of 90 steps (**40.0 %**; the line needed 90 %); the largest KL anywhere is 5.325 |
| 8 | 65 % | yes | `block16` identical to `fresh` on 86 of 90 steps (95.6 %) |
| 9 | 85 % | yes | value rows differ from `fresh`'s by 0.936 (mean relative L2, over layers and steps) against 0.134 for `drop_compact` |
| 10 | 95 % | yes | `block` read 14,827 tokens against 346,483 for `fresh`: **4.28 %** (the dry run with the crude counter said 4.5 %) |
| 11 | 85 % | yes | 2.81 s per step against 8.64 s: **0.325 times** |
| 12 | 40 % | **NO** | (updated after run B2) `block` right after `repair_m6` on 45 of 60 steps (75.0 %), `fresh` on 50 of 60 (83.3 %): **8.3 points below**, the line allowed 5; step by step 8 better and 13 worse (exact sign test p = 0.38: the data cannot tell the gap from chance) |
| 13 | 55 % | **NO** | (updated after run B2) exactly right as written: `block` 31 of 60 (51.7 %), `fresh` 37 of 60 (61.7 %): **10.0 points below**, the line allowed 5 (2 better, 8 worse, p = 0.11) |
| 14 | 40 % | **NO** | (updated after run B2) `block` identical to `fresh` on 28 of 60 steps (**46.7 %**); the line needed 70 % |

## The table the notebook printed (90 steps per arm: 28 add-edge, 22 add-node, 31 delete a service with dependants, 9 delete a leaf service)
| arm | identical to `fresh` | right after `repair_m6` | right as written | KL below 0.5 | largest KL | tokens read per step | seconds per step | value rows differ by |
|---|---|---|---|---|---|---|---|---|
| `fresh` | - | 100.0 % | 65.6 % | - | - | 3,849.8 | 8.64 | - |
| `drop_compact` | 100.0 % | 100.0 % | 65.6 % | 100.0 % | 0.005 | 164.7 | 1.55 | 0.134 |
| `block` | 47.8 % | 47.8 % | 47.8 % | 40.0 % | 5.325 | 164.7 | 2.81 | 0.936 |
| `block16` | 95.6 % | 96.7 % | 65.6 % | 95.6 % | 2.210 | 3,849.8 | 8.26 | 0.595 |

Identical to `fresh`, by kind of request:

| kind | `drop_compact` | `block` | `block16` |
|---|---|---|---|
| add edge (28) | 100 % | 78.6 % (22) | 100 % |
| add service (22) | 100 % | 95.5 % (21) | 100 % |
| delete a service with dependants (31) | 100 % | **0 %** | 87.1 % (27) |
| delete a leaf service (9) | 100 % | **0 %** | 100 % |

Paired against `fresh`, after the repair step: `block` 0 steps better and 47 worse (exact two-sided sign test, p < 0.0001), `block16` 0 better and 3 worse (p = 0.25), `drop_compact` 0 and 0. `block16` was rebuilt from nothing at every step (the page said: accuracy only), so it reads the whole prompt and takes `fresh`'s time by construction. Time, by phase: `block` 1.07 s of cache update per step (one forward pass per new item), 0.15 s of prefill, 1.59 s of decoding; `drop_compact` 0.32, 0.15 and 1.07. Peak GPU memory: `fresh` 11.0 GB, `drop_compact` 10.4 GB, `block` and `block16` 6.9 GB and 7.0 GB.

## Where my numbers were off
* **Lines 5 and 7 (55 % and 50 %) were not near misses.** 47.8 % against 90 % and 40 % against 90 %. I wrote that a reply of 11 tokens "has few places to differ"; the replies differ at the same place every time (below).
* **Line 6 (65 %).** I wrote that `repair_m6` "absorbs differences of the kind Run C showed". The differences here are of another kind. The repair can complete a reply that deletes the service and leaves out some of its edges; it cannot turn a reply that opens as an edge deletion into the deletion of a service. None of the 40 delete steps ends in the right state.
* **The anchor I used was Run A, and it was the wrong one.** I leaned on "identical replies with 93 % of the rows stale and a 22 % value difference". Here the value rows differ by 0.94 from a normal read at step 1 already (0.939, 0.940, 0.940 in the three episodes) against 0.13 for the stale tail, seven times more. Line 9 said "larger"; I did not expect it to be nearly the size of the rows themselves.
* **Line 8 (65 %) read yes with room** (95.6 % against 90 %); lines 3, 10 and 11 read as expected. The kept cache is not only close to the rebuild, it is bit for bit the same in the file (0.0).
* **My list of outcomes had no entry for this combination** (5 NO, 6 NO, 8 yes). The closest one I wrote reads: *"5 NO and 6 yes with 8 yes: the replies move but repair absorbs it, and bigger blocks lose less: a stock model needs some cross-item context, and the next test is the block size (2, 4, 8) and blocks that follow the graph (a service with its edges)."* Line 6 came out NO, so the repair does not absorb it; the rest of that reading stands. The entry *"5 and 8 NO: the claim is false for this model"* does not apply as written, because 8 read yes. For the claim as the shared text states it (every item its own block, no training), line 5 NO is the answer for this model and this task.

## What I read from the rows afterwards (NOT pre-registered; each of these is a look at tables I had already seen, so read them as reasons to test again, not as findings)
1. **The failure is one decision, taken on every delete.** In all 40 delete replies of `block` the first token is `DEL`, as in `fresh`, and the second token is `[(` (token 9697, the opening of an edge item) where `fresh` writes `[` (token 58, the opening of a service item). The request says "Delete the service at (6,0)"; `fresh` writes `DEL[6,0]` 40 times out of 40, `block` writes an edge deletion such as `DEL[(6,0)->(6,0):owns#8]`, an edge that is not in the graph. On the 31 deletes with dependants `block` names 31 invented edges in all and finds none of the service's own edges; the 9 leaf replies have the same shape. **All 43 replies of `block` that equal `fresh`'s are adds** (22 of 28 add-edge, 21 of 22 add-service). The 6 add-edge replies that differ differ in the coordinate of the target of the new edge; the one add-service reply that differs is written as an edge item. I do not know why the second token flips. The format is described in the system header, which every block and the request can see, so the description of the forms is not what is missing; which rows the request reads at that token was not looked at.
2. **Independence alone is enough; the holes in the position numbers are not needed.** At step 1 of each episode the positions have no hole yet (`gap` = 0, the first build is contiguous) and the replies already differ in all three episodes (one delete, two add-edge). The holes grow to 519, 361 and 397 positions by step 30. `block` and `block16` differ in two things (the size, and holes against contiguous positions), so the run cannot separate the two; the step-1 rows say the size is enough.
3. **The drop from 56.7 % in the first third to 40.0 % in the last (printed by the notebook) is mostly the mix of requests.** Deletes were 10, 16 and 14 of the 30 steps of the three thirds; on adds `block` was identical on 17 of 20, 14 of 14 and 12 of 16, on deletes on 0 of 10, 0 of 16 and 0 of 14. I do not read it as drift.
4. **`block16`'s 4 differing replies are 4 of the 31 deletes with dependants, and three of them have `block`'s shape** (edge deletions where the request names a service; one of the three names an edge that does exist, which the gate would admit as a different edit). The fourth lists four edges (their targets are invented) and then deletes the service; the repair step turns it into the right state, which is why `block16`'s repaired figure (96.7 %) is above its identical figure (95.6 %). So bigger blocks lose the same thing at a much lower rate (4 of 31 deletes against 31 of 31) (updated after run B2: this holds for the prompt without the rule only; with the rule stated blocks of 16 were the worst arm, see below). Because `block16` is rebuilt at every step here, this says that blocks of 16 are close to a normal read, not that a cache of blocks of 16 would be fast; a version that keeps the blocks and rebuilds only the block that changed was not built.
5. **Against `drop_compact` (the cut-out cache of the large test), `block` is worse on identical replies, on the repaired state, on KL and on time, equal on tokens read, and better on two things: it has no stale row, and its peak memory is lower.** Both read the same 164.7 tokens a step. `drop_compact` kept 90 of 90 replies (largest KL 0.005) at 1.55 s a step with 93.8 % of its state rows stale at the last step; `block` has no stale row and kept 43 of 90 at 2.81 s. What `block` buys is a delete that leaves no trace in the rows; in this regime a trace in the rows cost nothing in replies and independence cost more than half of them. The extra time is the update (1.07 s against 0.32 s: one pass per new item) and longer replies (15.9 tokens a step against 11.0).
6. **The first build is where the time goes.** Step 1 of an episode took 32.3 s for `block` (about 320 forward passes of one item each) against 9.2 s for `fresh`; steps 2 to 30 took 1.79 s against 8.62 s, 0.21 times. The 0.325 of line 11 includes the first build (the page guessed "a minute or less" for it; 32 s is inside that).

## Run B2: the rule stated (run 16 Oct 2026, a few hours after run B1)
*Free Colab T4, the same model and seed, the rule stated in the system prompt (`hint_rules` true), 3 episodes x 20 steps x 4 arms = 240 replies; the progress counter said 2,303 s, the step times add up to 2,109 s; Cell 10. The saved file is `docs/results/axi_block_b2_Qwen2.5-7B_seed21.json`, recomputed by test 112 in the same way as run B1. Above I wrote that I would not spend the time on this run before the block-size run; it was made anyway, and it changed one thing I had written (blocks of 16, below).*

**Lines 12, 13 and 14 all read NO** (the rows are in the table of the fourteen lines above). With run B1 that makes eight lines yes (1, 2, 3, 4, 8, 9, 10, 11) and six NO (5, 6, 7, 12, 13, 14) of the fourteen. Lines 12 and 13 miss by margins the paired data cannot tell from chance (p = 0.38 and 0.11); line 14 does not (46.7 % against 70 %).

### The table the notebook printed (60 steps per arm: 17 add-edge, 17 add-service, 21 delete a service with dependants, 5 delete a leaf service)
| arm | identical to `fresh` | right after `repair_m6` | right as written | KL below 0.5 | largest KL | tokens read per step | seconds per step | value rows differ by |
|---|---|---|---|---|---|---|---|---|
| `fresh` | - | 83.3 % | 61.7 % | - | - | 3,963.6 | 11.89 | - |
| `drop_compact` | 76.7 % | 83.3 % | 58.3 % | 90.0 % | 3.652 | 233.3 | 4.41 | 0.107 |
| `block` | 46.7 % | 75.0 % | 51.7 % | 45.0 % | 9.777 | 233.3 | 4.02 | 0.925 |
| `block16` | 33.3 % | 61.7 % | 48.3 % | 31.7 % | 11.007 | 3,963.6 | 14.83 | 0.589 |

By kind of request, identical to `fresh` / right after the repair step (`fresh` itself, right after the repair step, in the first column):

| kind | `fresh` | `drop_compact` | `block` | `block16` |
|---|---|---|---|---|
| add edge (17) | 100 % | 100 % / 100 % | 94.1 % / 94.1 % | 100 % / 100 % |
| add service (17) | 100 % | 100 % / 100 % | 70.6 % / 70.6 % | 17.6 % / 35.3 % |
| delete a service with dependants (21) | 66.7 % | 38.1 % / 71.4 % | 0 % / 66.7 % | 0 % / 47.6 % |
| delete a leaf service (5) | 40.0 % | 80.0 % / 20.0 % | 0 % / 60.0 % | 0 % / 80.0 % |

Paired against `fresh`, step by step (better / worse, exact two-sided sign test): after the repair step `drop_compact` 3 / 3 (p = 1.0), `block` 8 / 13 (p = 0.38), `block16` 5 / 18 (p = 0.011); as written `drop_compact` 0 / 2 (p = 0.5), `block` 2 / 8 (p = 0.11), `block16` 3 / 11 (p = 0.057). The kept `block` cache equals a rebuild exactly again (6 checks, 0.0 for the values and the keys). Peak GPU memory: `fresh` 11.1 GB, `drop_compact` 10.5 GB, `block` and `block16` 7.0 GB.

### Where my numbers were off
* **Lines 12 to 14 (40 %, 55 %, 40 %):** all NO. For lines 12 and 14 that was the likelier outcome by my numbers (60 %); for line 13 (55 % yes) it was the less likely one, 10.0 points below against an allowance of 5. I had expected nothing about `block16` here beyond line 8's 65 %; see 4 below.
* **My reading of the possible outcomes** ("independent blocks work where the reply is a copy and fail where it has to search") was written for 5 and 6 yes; they read NO, so it does not apply. What the two runs show together is that single-item blocks change the replies in both regimes.

### What I read from the rows afterwards (NOT pre-registered; reasons to test again, not findings)
1. **With the rule stated `block` finds none of the service's own edges, and the repair step hides that on the deletes.** On the 21 deletes with dependants `block` finds 0 of the service's own edges (recall 0.0; `fresh` 0.19, `drop_compact` 0.13, `block16` 0.14) and names 28 invented edges (1.33 a reply; `fresh` 3.62, `block16` 4.57). Its replies are shorter (21.2 tokens on average over all kinds of request, against 37.6 for `fresh`). After the repair step it is right on 14 of 21, the same as `fresh`: the repair completes the deletion of the edges of a service that the reply deletes, and `fresh`'s own lists are mostly wrong as well. So the 8.3 points of line 12 are not lost on the deletes.
2. **They are lost on the adds.** Add-service: `block` is right on 12 of 17, `fresh` on 17 of 17; the five that differ are all written as an edge item with the service's own coordinates at both ends (`ADD[(13,7)->(13,7):owns#6]` for "add a service at (13,7) with value 6"), the same flip of the second token as in run B1. Add-edge: 16 of 17.
3. **The flip is less total here.** 9 of the 26 delete replies of `block` open in the edge form (`DEL[(`) and 17 open in the service form; in run B1 it was 40 of 40. `fresh`, `drop_compact` and `block16` open in the service form on all 26.
4. **Blocks of 16 are not better here; they are the worst arm.** Identical on 33.3 % (20 of 60), below `block`'s 46.7 %, and right after the repair step on 61.7 % (37 of 60). 14 of its 17 add-service replies write the service and then go on to write `DEL[...]` items for edges of the new service (`fresh`, `drop_compact` and `block`: 0 of 17), and 22 of its 60 replies run to the 120-token limit (11 deletes with dependants, 10 add-service, 1 leaf; `fresh` 5, `drop_compact` 3, `block` 0). So what I wrote after run B1, that bigger blocks lose the same thing at a much lower rate, holds for the prompt without the rule and does not carry over to the prompt with it: the loss moves from the first token of a delete to the model applying the delete rule to an add. I do not know why. The paired data say 5 better and 18 worse than `fresh` after the repair step (p = 0.011), which with the six paired tests of this section would be about 0.06 after a correction.
5. **`drop_compact` again keeps most of it.** Identical on 46 of 60 steps (76.7 %): 8 of 21 deletes with dependants, 4 of 5 leaves, all 34 adds; right after the repair step on 50 of 60 (83.3 %), the same as `fresh`. This agrees with Run C of the large test (71.9 % identical). `block` is below it on identical replies (46.7 against 76.7 %), on the repaired state (75.0 against 83.3 %) and on KL (45.0 against 90.0 % of the steps below 0.5 nats); about equal on time (4.02 s against 4.41 s) and tokens read (233.3 each); better on stale rows (none, against 92.2 %) and peak memory (7.0 GB against 10.5 GB).
6. **Cost.** `block` read 233.3 tokens a step against 3,963.6 for `fresh` (5.89 %; the dry run said 6.1 %) and took 4.02 s a step against 11.89 s (0.338 times); after the first build (35.1 s on average, against 12.8 s for `fresh`) it took 2.38 s against 11.84 s (0.20 times).

## What this shows and does not show
* **Shows (this model with 4-bit weights, 120 services and 200 edges, 3 episodes, the first 30 steps (no rule in the prompt, run B1) and the first 20 steps (rule stated, run B2) of the scripts of Runs A and C, greedy decoding, teacher forcing):**
  1. a cache in which every item is read in a block of its own can be kept, cut and extended with an exact result at the level of the rows (the kept cache equals the rebuild, 0.0) and at a cost that is arithmetic (4.3 % of the tokens read);
  2. on this stock model that cache does not keep the replies: without the rule not on any of the 40 deletes, on 43 of 50 adds, and the repair step does not recover them; with the rule stated 28 of 60 replies are identical and 75.0 % are right after the repair step against 83.3 % for a normal read; so "no pre-training required" is false for independent single-item blocks on this model and this task, in both prompts;
  3. blocks of 16, rebuilt every step, keep 95.6 % of the replies without the rule and 33.3 % with it, so no block size tested here is shown to keep the replies in both prompts.
* **Not seen by me:** the output of the seven checks of Cell 6 on the real model (`preflight_blocks`, saved to `axi_tensor_preflight.json` on your Drive) was not sent to me. The notebook is built not to go on after a failed check, and in the run itself the nine comparisons of the kept cache with a rebuild are 0.0, but I have not seen the seven lines.
* **Does not show:** that a model trained or fine-tuned with the block mask would fail (the related work says it is what is needed, and it was not tested); anything about block sizes between 1 and 16, blocks that follow the graph, or a kept cache of blocks of 16; more than 20 steps with the rule stated; other models, bigger models, other precisions, sampling; longer items; the interface idea of the shared text (a block that exposes a few of its tokens), which is not what was built; why the second token flips; whether the same happens with the examples of the system header moved next to the request. It does not show that the shared text's table is wrong about a model that is trained for it.
* **Related work.** The result goes the way the papers named on this page go: independent pieces lose something without recomputation or fine-tuning (CacheBlend: recomputing about 15 % of the tokens repairs it; Block-Attention: the model is fine-tuned with the mask). This run did not reproduce those papers' numbers. What it adds is the case of ten-token self-contained lines, measured: small on adds that copy the request, total on the step whose reply form depends on what the request names.

## What to do next (not run)
* **The block size**, on the same scripts and with the same measures: 1, 2, 4, 8, 16. The line to write first: the smallest size at which the replies are identical on at least 90 % of the steps in both prompts, with the saving that size would give if the blocks were kept (it is not the 4.3 %).
* **Blocks that follow the graph** (a service with the edges that start at it), the version the page named, and a kept `block16` that rebuilds only the touched block.
* **The second token.** One look at which rows the request reads at the `[` or `[(` decision, in `fresh` and in `block`, would say whether this is a fault of the format or of independence; it is a few minutes on the T4.
* **Run B2** was made (above). It means the block-size run should be made in both prompts, with and without the rule, because size 16 is right in one and wrong in the other.

## Reproduce
`python/tests/test_kv_block_result.py` (no GPU, a few seconds) re-reads the two saved files (run B1: 360 replies, run B2: 240), scores every reply again from its text with `kv_drop.score_reply`'s own functions on the graph the code rebuilds from the seed (0 differences from the saved scores), recomputes the fourteen lines and every figure above, and checks that the part of this page above the Result line is byte for byte the text written before the run (sha256 `8f893996d9df330886651ac4cecdd957da76d41a501e5a72e5bd7ecaa38d6b04`). The test was itself checked with 204 deliberate changes (to the two saved files, to this page and to the code the test reads), each made alone in a copy and each required to make the test fail. The first version, written for run B1 alone, let 5 of the first 134 through (two quoted figures that also stand elsewhere on the page, a second Result heading, the scoring function `kv_drop.score_reply` itself, and a change aimed at a step already above its line); after run B2 was added, 3 of the 204 got through (two page quotes that were too loose and the text of one `drop_compact` reply that no figure on the page used). The test was tightened each time, and all 204 are now caught.

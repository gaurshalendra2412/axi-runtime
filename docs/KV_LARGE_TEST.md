# Keep or drop the interior tokens when the state is thousands of tokens long? A KV-cache test on a stock model (written BEFORE the run, 14 Oct 2026)

This page fixes the questions, the arms, the measures, the predictions and the reading rules before any number from this test exists. Code: `python/axi/experiments/kv_large.py` (the large graph, the request script, the reading rules) on top of `kv_drop.py` and `kv_torch.py`; notebook `python/colab/axi_kv_large.ipynb`; instrument checks `python/tests/test_kv_large.py`. Concept-map item 97. It is the second page of the pair that starts with `KV_DROP_TEST.md` (item 96): that page uses graphs of 8 to 14 services (prompts of a few hundred tokens); this one uses about 120 services and 200 edges (prompts of about 4,000 tokens). **This page does not depend on the result of the first**: when it was written the first had not been run. The two are read side by side once both exist.

## The question
`KV_DROP_TEST.md` asks whether cutting the tokens of a deleted item out of a stock model's key/value cache (nothing behind the hole read again) changes what the model writes. On a few hundred tokens the saving is small in absolute terms and the cache is small too. Size changes three things, and each of them can go either way:

1. **What a rebuild costs.** `fresh` reads the whole prompt at every step; at 4,000 tokens that is most of a step. A step that only appends a small change reads about 50 tokens. The saving grows with the state.
2. **What a prefix cache saves.** `exact` (the cache every serving stack keeps) can only keep the part *before* the first change. In this prompt the services come first and the edges after them, so a deleted service sits in the first quarter of the prompt and everything behind it, the whole edge list, has to be read again. At this size the prefix cache saves much less on a delete than the drop arms do.
3. **How much of the cache is stale.** A row behind a hole was computed while the deleted item was visible. After a few deletes more than nine rows in ten are behind some hole. In the small test that is a few hundred rows each carrying a faint trace; here it is thousands. A trace that is faint per row may or may not add up. This is the risk the test is built to look at.

And the model's job is harder: to delete a service it has to find that service's edges among about 200. So the measures are:

1. **Exactness.** On the same prompt, does the model write the same reply when the cache was *patched* as when it was *rebuilt*?
2. **Tokens** read and kept, per step and per way of holding the cache.
3. **Time per step** on a free Colab T4.
4. **Does it get worse as the stale rows pile up?** First third of an episode against the last third.

## What changes against `KV_DROP_TEST.md`
| | `KV_DROP_TEST.md` | this page |
|---|---|---|
| start graph | 8 services on a 3 x 4 grid, about 9 edges | **120 services on a 14 x 14 grid, 200 edges** (run X: 240 services on a 20 x 20 grid, 400 edges) |
| limits during an episode | 14 services, 26 edges | 150 services, 260 edges (run X: 300 and 520) |
| prompt | a few hundred tokens | **about 4,000 tokens** (run X about 8,000) |
| arms | `fresh`, `exact`, `drop_compact`, `drop_gap`, `keep` | `fresh`, `exact`, `drop_compact`, `drop_gap`. **No `keep`**: one state is already thousands of tokens, a chat that keeps every state would pass 20,000 tokens within a few steps |
| runs | A: 8 x 30; C: same scripts with the rule; B: 2 x 100 | **A: 4 x 40; C: same scripts with the rule; X: 2 x 30 on the bigger graph, rule stated** |
| reading rules | 15 | 16 (own thresholds, below) |

Everything else is the same code: the prompt layout, the cache protocol, teacher forcing, the arms, the KL measure, the scoring, the resumable runner and the `preflight` on the real model. `kv_drop.py` gained one optional argument (`script_fn`, which request script to use) and nothing else; with the default it behaves as before.

## Interface and interior (the definition used here)
As in `KV_DROP_TEST.md`. **Interface:** the fixed header and the items of the live state. **Interior:** the request and the reply of a step (cropped by every cache arm: exact), and the items a step deleted (cut out by the two drop arms: the approximation). An added item is appended after the last item and is exact in every arm; only deletes are approximate. This is my operational reading, not Rajnish's definition, and a model trained or masked for block independence is not tested.

## The arms
| arm | how the cache for the step is obtained | exact? |
|---|---|---|
| `fresh` | the whole prompt is read again from nothing | the reference |
| `exact` | the cache is kept; request and reply are cropped; an add is appended; after a delete everything from the first deleted item on is read again | exact (a prefix cache) |
| `drop_compact` | as `exact`, but a deleted item is cut out and nothing after it is read again; the keys behind the hole are turned back to the positions a fresh read would give | approximate: stale rows only |
| `drop_gap` | as `drop_compact` without the turn: a hole in the position numbers | approximate: stale rows and a position gap |

## Design
* **Model:** Qwen2.5-7B-Instruct in 4-bit (NF4), free Colab T4, greedy decoding, the grammar of the earlier 7B tests (the reply must be a delta), at most 120 new tokens. Same system prompt as every earlier 7B test; run C and run X add the rule hint.
* **Prompt:** exactly the layout of `KV_DROP_TEST.md`: header, then one piece per item (a space, then `[r,c:v]` or `(r,c)->(r,c):rel#w`), then the closing tag after a space, the request and the assistant header. The initial items are the services sorted, then the edges sorted. A later add goes to the end. Deleted items leave a hole in the list. **This layout decides the stale share:** the first hole of a delete is almost always in the service list, so the whole edge list, about three quarters of the prompt, is behind it. An edges-first layout, or one that lists each service followed by its edges, would give different shares; those are not tested.
* **Graph:** `kv_large.random_large_graph`: 120 services on random cells of a 14 x 14 grid (values 1 to 9; coordinates up to 13, so some are two digits), 200 distinct edges (relation `dep` or `owns`, weight 1 to 9) between random pairs, one service in twenty with no edge (so that "delete a service without edges" has candidates). A service has on average 3.3 edges.
* **Requests:** the same four kinds as `long_horizon.oracle_script`, in the same proportions (add a service, add an edge, delete a service with 1 to 3 edges, delete a service without edges), chosen on the oracle's graph so that each is meaningful there. `add_node` takes its free cells from the bigger grid; the other three are `cascade.pick_task` unchanged. About 40 % of the steps delete something. The graph shrinks a little over 40 steps (from 120 and 200 to about 110 and 180).
* **Teacher forcing:** the state at every step is the oracle's, as before. The arms never feed their own mistakes back; they differ in nothing but the cache.
* **Run A (no rule):** 4 episodes of 40 steps, stream seed 21 (episode seeds 21000 to 21003), four arms. **Run C (rule stated):** the same 4 request scripts with the rule of the earlier tests in the system prompt; the answer then depends on the state (the model must name the edges of the service), so the cache has something to get wrong. **Run X (bigger graph):** 2 episodes of 30 steps, stream seed 22 (episode seeds 22000 and 22001), 240 services and 400 edges, rule stated.
* **Cost:** $0 (a local model, no key). Time on a free T4, my estimate and nothing more: 45 to 70 minutes for run A, as long for run C, 30 to 45 for run X (the smoke test in the notebook prints seconds per step and a projection for run A from the real machine before the long runs start). The notebook saves after every episode to Google Drive and continues after a drop.

## What is measured, per step and per arm
As in `KV_DROP_TEST.md`: reply identical to `fresh`'s (token for token); KL along the reply while the two agree; task exactness (blind and `repair_m6`) and whether the reply was legal as written; tokens read and kept, bytes, peak memory; seconds split into cache update, prompt and decode, GPU synchronised; stale rows and the position gap of the drop arms; value deviation. Added here: **the size of the prompt** (tokens of `fresh`'s prompt), **agreement in the first third against the last third** of the steps, and **the share of the state rows that are stale at the last step**.

## Checks before the run (the notebook stops if one fails)
* `preflight` on the real model, **on the first prompt of the large script (about 4,000 tokens)**: the nine checks of `KV_DROP_TEST.md` (pieces = one pass; keys turned by 211 positions = keys read at those positions, and the check can tell; next-token logits; cut and re-read; select-all identity; a hole in the positions does not hide earlier rows; size reporting; tokenizer pieces). The tolerances are the ones of the first page (0.5 in logits, 0.05 relative in keys) and are guesses for 16-bit numbers with 4-bit weights; at 4,000 tokens the rounding differences between a prompt read in two pieces and one read at once are larger than at 500, so a line may fail by a small margin. The notebook then stops and prints its numbers. I would look at them before changing a tolerance, and the change would be written below the locked text.
* At every step: the item list describes exactly the oracle's state; the cache of `exact`, `drop_compact` and `drop_gap` holds exactly the tokens of the logical prompt (an assertion).
* The chat template of the tokenizer must equal the ChatML text the test builds.

## What the script alone fixes (a dry run, no model)
Some of the lines below follow from the request script and the design, not from what the model does. I counted them before writing the predictions with `kv_large.dry_run`, which uses a crude token counter (a word, a digit or a run of punctuation is one token; **not Qwen's tokenizer**, so the real figures will differ by some tens of percent), the bookkeeping of the arms, and the seeds of run A and run X:

| | run A (4 x 40 steps) | run X (2 x 30 steps) |
|---|---|---|
| tokens of the first prompt | about 3,800 | about 7,800 |
| `drop_compact` reads, as a share of what `fresh` reads | 3.7 % | 3.9 % |
| `exact` reads, as a share of what `fresh` reads | 36.5 % | 37.6 % |
| `exact` reads, as a multiple of what `drop_compact` reads | 9.9 | 9.7 |
| share of the state rows (header included, which is never stale) that are stale in the cache that serves the last step | 92.8 % | 93.7 % |

Lines 1, 9, 10 and 15 are these. They are in the list so that the table is complete and the instrument is checked on the real tokenizer, not as risky claims, and my probabilities for them are high accordingly. The table also says how hard the test is on the cache: when it ends, more than nine rows in ten of what the drop arms hold were computed with some deleted item visible.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error, except line 4: if `exact` does not match `fresh`, the instrument or the numerics are to be looked at before anything else is read, and line 3: if `fresh` itself cannot do the task at this size, the other lines say little.

1. [instrument] The graph is large: the prompt of `fresh` holds on average at least 3,000 tokens over the steps of run A. (90 %)
2. [instrument] The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run A. (85 %)
3. [guard] fresh: the share of steps on which repair_m6 turns the reply into the right next state is at least 90 % (run A). (75 %)
4. [guard] exact: the reply is identical to fresh's, token for token, on at least 98 % of the steps of run A. (75 %)
5. [claim] drop_compact: the reply is identical to fresh's on at least 97 % of the steps of run A. (60 %)
6. [claim] drop_gap: the reply is identical to fresh's on at least 95 % of the steps of run A. (55 %)
7. [claim] drop_compact: the share of steps on which repair_m6 turns the reply into the right next state is within 1 point of fresh's (run A). (85 %)
8. [claim] drop_compact: on at least 95 % of the steps of run A the largest KL along the reply, against fresh's distribution, is below 0.05 nats. (50 %)
9. [claim] drop_compact reads at most 10 % of the tokens that fresh reads (run A, whole episodes, the first build of each included). (95 %)
10. [claim] exact reads at least 5 times as many tokens as drop_compact (run A). (90 %)
11. [claim] drop_compact: mean seconds per step at most 0.5 times fresh's (run A). (75 %)
12. [claim] drop_compact: mean seconds per step below exact's (run A). (90 %)
13. [claim] drop_compact: the reply is identical to fresh's on at least 95 % of the steps in the last third of the steps of run A. (50 %)
14. [claim] run C (rule stated): drop_compact's reply is identical to fresh's on at least 95 % of the steps. (55 %)
15. [load] drop_compact: at the last step of run A, at least 80 % of the state rows in its cache are stale. (95 %)
16. [claim] run X (about 8,000 tokens): drop_compact's reply is identical to fresh's on at least 95 % of the steps. (45 %)

**Headline** (lines 5, 7, 8, 13 and 14): if all five read "yes", then on this model, this task and these sizes, cutting the deleted items out of the cache did not change what the model wrote even though more than nine rows in ten of what it held were stale, in the regime where it does not need the state much (run A) and in the one where it does (run C), and the effect did not grow over the episode. The product of my five probabilities is about **7 %**. They are not independent: if the cache is faithful they all hold, and if it is not, 5 and 13 fail together. Taking that into account my own single number for all five holding is about **25 %**, and I would still bet against it.

**A direction, not one of the sixteen lines:** I expect (about 65 %) that `drop_compact`'s agreement with `fresh` in run A here is lower than in run A of `KV_DROP_TEST.md`. That is read by hand from the two result files, and only if both runs were made.

Why I think so, line by line. *Exactness (4 to 8, 13, 14, 16):* the argument for "faithful" is dilution: a deleted item is about 40 tokens among 4,000, a stale row attended to it with a small weight among thousands, and the replies are mostly a copy of numbers in the request, so a faint trace should rarely change a token. The argument against is the load: more than nine rows in ten are stale, the model has to find the edges of one service among 200, and at 4,000 tokens its own margins between the best token and the second are narrower than at 500, so a small shift flips a token more easily. I end at a little better than even for 97 % (60 %), and lower for the stricter KL line (50 %), because the largest KL along a reply is a maximum over 10 to 60 positions. The guard `exact` gets 75 % at 98 %, a little less than the first page gave it at 99 %: the same inputs read in one pass or several give slightly different 16-bit numbers, and a long prompt gives them more room to add up. `drop_gap` (55 %): RoPE models are known to tolerate gaps in position numbers (see the related work), but by the end of an episode the gap is some hundreds of positions. *Line 3 (75 %):* `repair_m6` completes the edges of a deleted service and cuts claims that are not in the graph, so a reply that misses an edge is still repaired; what hurts is a wrong coordinate or a copied-wrong number, and the 7B did not do that on small graphs. *Tokens (9, 10, 15):* arithmetic, see the dry run. *Time (11, 12):* a decode of 20 to 30 tokens is about 2 seconds on this setup (the earlier runs took about 1.1 s per call); a prefill of 4,000 tokens on a T4 with 4-bit weights I guess at 4 to 10 seconds (the earlier 0.5-billion-parameter measurement, `VERIFIED.md`, was 683 ms for 4k, and a 7B does about 15 times the work per token with better efficiency). So `fresh` should take 6 to 12 seconds per step and the drop arms about 2.5, and the line at 0.5 times (75 %) has room; `drop_compact` below `exact` (90 %) because `exact` reads about 1,000 tokens more per step on average.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *All five headline lines yes:* at this size the cut-out is not visible in the replies; the next experiments are a second model family, an edit in the middle, and a bigger model. *Line 5 NO but 7 yes:* the replies change but `repair_m6` absorbs it: the approximation is real and the gate hides it; the next experiment is how much re-reading restores it (a few percent of the stale rows, as CacheBlend does). *Line 6 NO with 5 yes:* the positions matter and the rotation is needed. *Line 13 NO with 5 yes:* the effect is there but small, and it grows with the number of deletes. *Line 3 NO:* the task is too hard for this model at this size; the other lines are then about a model that is already guessing, and a smaller graph or another model is the next test. *Line 4 NO:* look at the numerics and the instrument first.

## Reading rules (fixed now; the notebook prints exactly these)
The sixteen lines above are the reading rules; `kv_large.verdict_large` computes them from the saved rows. "Identical" means the same tokens including the end token. "Steps" are all steps of all episodes of the run. The last third of the steps is steps 27 to 40 of a 40-step episode; the first third (shown in the table, in no line) is steps 1 to 13. A line whose run was not made reads "not run". The share of stale rows (line 15) is the number of stale rows divided by all state rows (the header and the items, but not the request) in the cache that serves the last step of each episode, averaged over the episodes. "Prompt of `fresh`" (line 1) is the number of tokens it reads at a step, request included.

## What this cannot show
One model (4-bit weights, which add noise of their own), one task family (a graph, deltas), greedy decoding, 40 or 30 steps, a free T4. Teacher forcing: the arms are never fed their own mistakes. Deletes only: adds are exact by construction, and a *changed* value is not tested. One layout of the prompt (services first, then edges). **One model size:** a 14-billion-parameter model has more layers and more key/value heads (its cache is about 197 KB per token against about 57 KB), so four caches of 4,000 tokens would be about 3 GB next to weights of about 9 GB in 4-bit, with the activations of a 4,000-token prompt on top: borderline on the 15 GB of a free T4, and very likely too tight for the bigger graph of run X. It would want a paid GPU (a few dollars an hour). That is a separate test with its own page. The cache bytes of this model are still small at 4,000 tokens (about 230 MB per cache), so the saving in tokens is real, the saving in bytes modest. `exact` is my own implementation of a prefix cache on Hugging Face, not vLLM or SGLang, whose kernels would shift the time numbers. My cache wrapper copies the whole cache once per forward call (it hands a fresh Hugging Face cache object to the model): at 4,000 rows that is about 230 MB moved per generated token, a few milliseconds on a T4 (my estimate, not measured), against roughly 70 ms per token for reading the weights (a guess); it is paid by every arm that holds a cache and slightly raises their time compared with a serving stack. Nothing here supports a statement about months of continuous operation, about constant compute in the session length, or about models other than this one; and passing every line would not show that dropping is safe in general, only that it was safe in this setting. A model trained or fine-tuned with block masks is a different experiment.

## Related work (checked 14 Oct 2026, not a survey)
The same three papers as `KV_DROP_TEST.md`, and I did not search beyond them.
* *Prompt Cache: Modular Attention Reuse for Low-Latency Inference* (Gim et al., MLSys 2024): reuses precomputed attention states of prompt modules at other positions and reports that the models work with discontinuous position ids, accuracy comparable to the baseline. Bears on `drop_gap`.
* *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion* (Yao et al., EuroSys 2025): chunks cached independently lose the attention between chunks; recomputing about 15 % of the tokens restores quality to within about 0.02 F1 or Rouge-L, with 2.2 to 3.3 times lower time to first token. Bears on the stale rows, and on what a repair could be.
* *Block-Attention for Efficient RAG*: independent blocks with positions re-encoded by RoPE rotations; the model has to be fine-tuned with the block mask. Bears on `drop_compact` and on "a stock model is an approximation".

## How it is run
`axi_kv_large.ipynb` on Colab (T4): upload `axi-colab-kv-large.zip` (not the zip of the first notebook), connect Google Drive, run the cells top to bottom. The cells: GPU check, upload, install, Drive, load the model, `preflight` on the large prompt (stops if a line fails), a smoke test (one episode of 3 steps, all four arms, which prints seconds per step and a projection for run A), run A, run C, run X (optional). Each cell writes one line per finished episode, so a dropped session continues where it stopped. The result files are JSON; send them back and this page gets a "Result" section without changing anything above this line.

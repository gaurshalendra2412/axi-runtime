# Keep or drop the interior tokens? A KV-cache test on a stock model (written BEFORE the run, 14 Oct 2026)

This page fixes the questions, the arms, the measures, the predictions and the reading rules before any number from this test exists. Code: `python/axi/experiments/kv_drop.py` (the logic, no torch) and `kv_torch.py` (the PyTorch side); notebook `python/colab/axi_kv_drop.ipynb`; instrument checks `python/tests/test_kv_drop.py`. Concept-map item 96, with items 89 (block independence) and 92 (the token bill). It follows `LONG_HORIZON_TEST.md`, where keeping the whole chat made a step 5.1 times slower, but only because that test read the whole chat again at every step.

## The question
Block independence (RFC-001, item 89) says: if the next step depends only on an interface, the tokens in between can be dropped. For a model trained with ordinary full attention that is an **approximation**, because a token that stays in memory has already read the tokens that are removed. `tests/test_block_independence.py` shows it on a toy: a mask that only looks like block independence leaks from two layers on, and the strict one (interface queries never see the interior) is bit-exact. Nobody here has measured how large the approximation is on a real model. This test does, for one stock model and one task, and it measures three things next to it: tokens read, tokens kept, and time per step.

1. **Exactness.** On the same prompt, does the model write the same reply when the cache for the prompt was *patched* (the deleted items cut out, nothing after them read again) as when the cache is *rebuilt* from nothing?
2. **Tokens.** How many tokens does each way of holding the cache read per step, and how many does it keep?
3. **Time per step**, on a free Colab T4.
4. **The chat, with its cache reused.** `LONG_HORIZON_TEST.md` measured the chat without reusing the cache. With the cache reused, is a long chat still slow?

## Interface and interior (the definition used here)
* **Interface:** the fixed header and the items of the live state (every service and every edge, each written the way the system prompt describes them).
* **Interior:** the request and the reply of a step, and the items a step deleted. The request and the reply are cropped from the cache after every step by every cache arm; this is exact, because nothing after them is kept. The tokens of a deleted item are cut out of the cache by the two drop arms; the tokens after the hole stay. **That is the approximation.** Their keys and values were computed while the deleted item was visible (they are *stale*).
* This is my operational reading of "interior" and "interface" for a stock model. It is not Rajnish's definition, and a model trained or masked for block independence is not tested.
* An *added* item is appended after the last item. Causal attention then gives it exactly the cache a fresh read would give it, so adds are exact in every arm. Only deletes are approximate.

## The arms (every arm answers the same logical prompt at every step, except `keep`)
| arm | how the cache for the step is obtained | exact? |
|---|---|---|
| `fresh` | the whole prompt (header, items, request) is read again from nothing; one forward pass | the reference: what the stock model says about this prompt |
| `exact` | the cache is kept; request and reply are cropped; an add is appended; after a delete everything from the first deleted item on is read again | exact (a prefix cache, as serving stacks keep) |
| `drop_compact` | as `exact`, but a deleted item is **cut out** and nothing after it is read again; the keys behind the hole are turned back to the positions a fresh read would give them (a rotation; rotations add, so this part is exact) | approximate: stale rows only |
| `drop_gap` | as `drop_compact`, but the positions are left as they were (a hole in the position numbers, no rotation) | approximate: stale rows and a position gap |
| `keep` | a chat: every step's state, request and raw reply stays in the cache; nothing is dropped | exact for its own, longer input; not the same input as `fresh` |

`exact` is the honest baseline for time and tokens. Prefix caching is standard in serving stacks, so a saving against `fresh` alone would mostly be that.

## Design
* **Model:** Qwen2.5-7B-Instruct in 4-bit (NF4), free Colab T4, greedy decoding, the grammar of the earlier 7B tests (the reply must be a delta), at most 120 new tokens. Same system prompt as every earlier 7B test; run C adds the rule hint.
* **Prompt:** `<|im_start|>system ...<|im_end|><|im_start|>user <CICO_STATE>`, then one piece per item (a space, then `[r,c:v]` or `(r,c)->(r,c):rel#w`), then the request and the assistant header. The closing tag `</CICO_STATE>` follows a space and not a line break: by my reading of Qwen's pre-tokenizer rule a bracket followed by a line break is one pre-token that the tokenizer could merge into one token, and then the pieces would not be the tokenizer's own tokens (a toy tokenizer with that rule shows it in `tests/test_kv_drop.py`; the real tokenizer is checked by `preflight` and by line 1). Items stay in the order they were added; deleted items leave a hole in the list, not in the order. The initial items are the starting graph sorted, as in the earlier tests. This is a different prompt layout from the earlier 7B tests (items in one list, no `NODES:` and `EDGES:` labels), so the numbers of `fresh` here are not the numbers of `LONG_HORIZON_TEST.md`.
* **Requests:** the scripts of `long_horizon.oracle_script` (add a service, add an edge, delete a service with dependants, delete a service without edges; at most 14 services and 26 edges).
* **Teacher forcing:** the state at every step is the **oracle's**. The arms never feed their own mistakes back into the state; the model's replies do not enter the cache of any arm except `keep`. So the arms differ in nothing but the cache. Compounding of the model's own errors is what the earlier long-horizon tests measured and is not measured here.
* **Run A (no rule):** 8 episodes of 30 steps, stream seed 11 (episode seeds 11000 to 11007), all five arms. **Run C (rule stated):** the same 8 request scripts with the rule of the earlier tests in the system prompt; the answer then depends on the state (the model must name the edges of the service), so the cache has something to get wrong. **Run B (long):** 2 episodes of 100 steps, stream seed 12 (episode seeds 12000 and 12001), rule stated; `keep` stops before its cache would pass 20,000 tokens.
* **Cost:** $0 (a local model, no key). About 1 hour of a free T4 for all three runs (my estimate); the notebook saves after every episode to Google Drive and continues after a drop.

## What is measured, per step and per arm
* **Reply identical to `fresh`'s:** the same tokens, token for token (greedy decoding, so the same reply means the same behaviour). Every arm writes its own reply; none is told what `fresh` wrote.
* **KL:** while an arm's reply agrees with `fresh`'s, the KL divergence (nats) between `fresh`'s distribution and the arm's over the tokens the grammar allows, at each position. The largest per step is kept. An arm can write the same reply with a shifted distribution; this shows it.
* **Task exactness:** the reply is parsed and applied to the oracle's state by blind apply and by `repair_m6`; the share of steps that give exactly the oracle's next state. Also whether the reply was legal as written.
* **Tokens read:** tokens put through the model for the prompt (header and items at the first step, the request, items to append, the items read again after a delete; for `keep` the whole new turn). Reply tokens are counted apart. **Tokens kept:** the rows in the cache when the reply is finished, and at the end of the step; bytes of keys and values; peak GPU memory.
* **Time:** wall seconds per step, split into cache update (cut, turn, append, re-read), prompt and decode, with the GPU synchronised. First third against last third of the steps.
* **Staleness** (for the drop arms, not predicted): rows whose keys and values were computed while a since-deleted item was visible; the position gap.

## Checks before the run (the notebook stops if one fails)
* `preflight`, on the real model: a prompt read in two pieces gives the same logits as read at once; keys turned by 211 positions equal the keys of a prompt read at those positions, in every layer, and the same keys without the turn are clearly different (so the check can tell); the next token's logits agree; rows cut and read again give the same logits; selecting every row changes nothing; a hole in the position numbers does not hide the earlier rows (the logits change when the earlier rows are different); the cache reports its size; the three pieces of the prompt are the tokenizer's own tokens. Differences of keys are measured relative to the size of the keys, and the tolerances are set for 16-bit numbers with 4-bit weights; a failed line stops the notebook and prints its numbers.
* At every step: the item list describes exactly the oracle's state; the cache of `exact`, `drop_compact` and `drop_gap` holds exactly the tokens of the logical prompt (an assertion; the run stops otherwise).
* The chat template of the tokenizer must equal the ChatML text the test builds.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error, except line 2: if `exact` does not match `fresh`, the instrument or the numerics are to be looked at before anything else is read.

1. [instrument] The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run A. (90 %)
2. [guard] exact: the reply is identical to fresh's, token for token, on at least 99 % of the steps of run A. (85 %)
3. [claim] drop_compact: the reply is identical to fresh's on at least 97 % of the steps of run A. (80 %)
4. [claim] drop_gap: the reply is identical to fresh's on at least 95 % of the steps of run A. (75 %)
5. [claim] drop_compact: the share of steps on which repair_m6 turns the reply into the right next state is within 1 point of fresh's (run A). (90 %)
6. [claim] drop_compact: on at least 95 % of the steps of run A the largest KL along the reply, against fresh's distribution, is below 0.05 nats. (70 %)
7. [claim] drop_compact reads at most 25 % of the tokens that fresh reads (run A, whole episodes, the first build of each included). (80 %)
8. [claim] exact reads at most 40 % of the tokens that fresh reads (run A). (80 %)
9. [claim] keep: at the last step of run A its cache holds at least 10 times as many tokens as drop_compact's. (90 %)
10. [claim] drop_compact: mean seconds per step at most 0.85 times fresh's (run A). (70 %)
11. [claim] drop_compact: mean seconds per step below exact's (run A). (75 %)
12. [claim] keep: mean seconds per step in the last third of the steps at most 1.5 times the first third (run A). (75 %)
13. [claim] keep: the reply is identical to fresh's on at least 90 % of the steps of run A. (65 %)
14. [claim] run C (rule stated): drop_compact's reply is identical to fresh's on at least 95 % of the steps. (60 %)
15. [claim] run B (episodes of 100 steps): drop_compact's reply is identical to fresh's on at least 95 % of steps 67 to 100. (60 %)

**Headline** (lines 2, 3, 5, 6 and 14): if all five read "yes", then on this model, this task and these sizes, cutting the deleted items out of the cache did not change what the model wrote, in the regime where it does not need the state much (run A) and in the one where it does (run C). The product of my five probabilities is about **26 %**. The lines are not independent, so the real chance is somewhat higher; I would still bet against all five.

Why I think so, line by line. *Exactness (2 to 6, 13 to 15):* the task is easy for a 7B. Most replies are a copy of numbers in the request, and with no rule it never lists the edges of a deleted service (`LONG_HORIZON_TEST.md`, 0 of 95), so a perturbation of the cache has little to change; a stale row carries a faint trace of an item that is gone. I expect the replies to stay the same on almost every step and the distribution to move a little. I expect run C to be harder for the cache (60 %): the reply depends on finding the edges in the state. Position gaps are known to be tolerated by RoPE models (see below), so `drop_gap` should be only a little worse than `drop_compact`. *Tokens (7 to 9):* arithmetic of the design more than findings. A prompt is perhaps 350 to 550 tokens (my estimate, not measured); a request with its header about 45; an added item 7 to 14. `drop_compact` reads about the request and the added item at each step plus the first build, which is why the threshold is 25 % and not 15 %. `exact` reads the items after the first deleted one again on delete steps. `keep` holds a whole turn per step. *Time (10 to 12):* a call took about 1.1 s on this setup in the earlier runs (`LATENCY_7B.md`, `LONG_HORIZON_TEST.md`), and the reply is only 10 to 20 tokens, so I expect decoding to be the larger part (a guess: no read/decode split has been measured here yet; this test measures it) and the saving to be real but modest, 15 to 30 %; the difference between `drop_compact` and `exact` is smaller still. With the cache reused, the chat should NOT be 5 times slower: the extra attention over 10,000 rows is small next to reading the weights.

## Reading rules (fixed now; the notebook prints exactly these)
The fifteen lines above are the reading rules; `kv_drop.verdict_kv` computes them from the saved rows. "Identical" means the same tokens including the end token. "Steps" are all steps of all episodes of the run. Third of the steps: the first third is steps 1 to 10 of a 30-step episode, the last third steps 21 to 30. A line whose run was not made reads "not run".

## What this cannot show
One model (4-bit weights, which add noise of their own and could hide or add to the effect), one task family (small graphs, deltas), greedy decoding, 30 or 100 steps, a free T4. Teacher forcing: the arms are never fed their own mistakes. Deletes only: adds are exact by construction, and a *changed* value (an edit in the middle) is not tested. The cache bytes of this model are small (grouped-query attention, about 57 KB per token), so on a T4 the cache is not the constraint at these sizes; the saving in tokens is real, the saving in bytes is modest. `exact` is my own implementation of a prefix cache on Hugging Face, not vLLM or SGLang, whose kernels would shift the time numbers. Time on a T4 with 4-bit weights depends on the kernels. My cache wrapper copies the whole cache once per forward call (it hands a fresh Hugging Face cache object to the model): nothing visible for a few hundred tokens, but at 15,000 tokens a few percent of a step (my estimate, not measured), which slightly raises the time of `keep` late in an episode and so works against line 12. Nothing here supports a statement about months of continuous operation, about constant compute in the session length, or about models other than this one; and passing every line would not show that dropping is safe in general, only that it was safe in this setting. A model trained or fine-tuned with block masks is a different experiment.

## Related work (checked 14 Oct 2026, not a survey)
* *Prompt Cache: Modular Attention Reuse for Low-Latency Inference* (Gim et al., MLSys 2024): reuses precomputed attention states of prompt modules at other positions and reports that the models work with discontinuous position ids, accuracy comparable to the baseline. Bears on `drop_gap`.
* *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion* (Yao et al., EuroSys 2025): chunks cached independently lose the attention between chunks; recomputing about 15 % of the tokens restores quality to within about 0.02 F1 or Rouge-L, with 2.2 to 3.3 times lower time to first token. Bears on the stale rows, and on what a repair could be.
* *Block-Attention for Efficient RAG*: independent blocks with positions re-encoded by RoPE rotations; the model has to be fine-tuned with the block mask. Bears on `drop_compact` and on "a stock model is an approximation".
I am not aware of a measurement of this exact case (items cut out of a structured state held in the cache, a stock 7B, greedy, replies compared with a rebuild), but I did not search for one beyond these three.

## How it is run
`axi_kv_drop.ipynb` on Colab (T4): upload `axi-colab-kv.zip`, connect Google Drive, run the cells top to bottom. The cells: GPU check, upload, install, Drive, load the model, `preflight` (stops if a line fails), a smoke test (one episode of 4 steps, all arms), run A, run C, run B (optional). Each cell writes one line per finished episode, so a dropped session continues where it stopped. The result files are JSON; send them back and this page gets a "Result" section without changing anything above this line.

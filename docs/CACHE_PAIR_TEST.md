# The cache pair test: the same job with the kept chat cold and warm, and the host-held arm counted for cache eligibility

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. The plan and the twelve predictions below were written and hash-locked before the 400-turn run; the Result follows them. Saved data: `docs/results/axi_cache_pair_seed20261122_default.json`. Model: gemini-3.8-flash. AXI is a layer on the host side, between a model and the state it edits; the model is unchanged and every claim is about the state the host holds.*

**In one paragraph.** A reader of the cost article asked for the comparison to be set up as a cold run and a warm run for each way of holding the state, with the cache read of every request logged. This test did that on the Test 23 job (seed 20261122). The host-held arm (A) ran 400 steps: held state exact on all 400, 470,669 prompt tokens (the same number as in Test 23), the largest request 1,390 tokens, and not one request showed a cache hit. The kept chat ran cold for 390 turns (a changing line in the system text of every request, so that no request shared its beginning with an earlier one; the raw cache field was None on all 390 requests, 383 of them at least 4,096 tokens): 37,967,655 prompt tokens, all fresh. Counted per verified step, the cold kept chat used 87.7 times the computed tokens of A, and 61 times the dollars at the typed prices. Test 23's warm kept chat, with 97.0 percent of its prompt read from the cache, used 3.5 times A's computed tokens by the same count, with cached tokens not counted.

## What was asked

The reader (a comment on the Medium copy of the cost article, 21 and 22 November 2026) asked for: a cold and a warm run for each way of holding the state, the same task sequence, model and stopping rule; the token count split into cache reads, fresh input and output; one exact-state check across all runs; spend per verified step including the steps that went wrong; the raw cache field logged on every request with a missing field kept apart from an explicit zero; and, since the service caches only requests of at least 4,096 input tokens (his figure), a count of how many host-held requests were eligible, without padding them. Test 23 ran the kept chat warm and the host-held arm with no cache reads; the cold kept chat existed only as arithmetic in `docs/COST_OF_BEING_WRONG.md`, and the harness of Test 23 stored a missing cache field as 0.

## What was run

One model, **gemini-3.8-flash**, the model's own default thinking, the Test 23 job (seed 20261122, the same stream). Code: `python/axi/experiments/gemini_fuzzy.py`, `gemini_long.py` and `api_llm.py`, unchanged. The driver (sha256 `47271c207ff7edbd8d5abc07f34fb23c21833fd9462657b5730243b978340374`) edits no repository file: a proxy around the SDK client records the raw `cached_content_token_count` of every call (absent, None, 0 or a number), and the changing line is put on a copy of the messages, never into the saved chat.

| cell | what it is |
|---|---|
| A | the host-held arm, 400 steps, input exactly as in Test 23; no line, no padding; the raw field logged on every request |
| KC | the kept chat, cold: one line, `Request id: <10 hex characters of a hash of the request and of a per-run random value>.`, as the last line of the system text of every request, just before the chat, so that no earlier request, in this run or in an earlier one, shares its beginning from that point on. Cold is read request by request from the raw field |
| KW | not re-run: Test 23's saved kept chat (`docs/results/axi_gemini_fuzzy_seed20261122_default.json`, arm K), the same stream, warm (97.0 percent of its prompt tokens from the cache) |

Prices typed by the owner (0.75 and 3.75 dollars per million input and output tokens, as in the earlier Gemini tests); no cached price was typed because neither cell had a cached token. One dollar cap of 29 dollars for the two cells together; the run stops by itself at the cap. A ran its 400 steps first; KC then ran until the cap stopped it at turn 390.

Where the changing line goes was settled on 15- and 25-turn pilots before the run: as the first line of the system text it changed the model's reply format on 6 of 40 turns, as the last line on 0 of 25, so the last line was used.

## What is measured

For every request: prompt, visible and thinking tokens, the raw cache field, seconds. For every cell: the three-way split (cache reads, fresh input, output), the requests of at least 4,096 prompt tokens and how many showed a hit, any hit below 4,096 (and the count of requests of at least 1,024 and 2,048 tokens), the verified steps (held state equal to the oracle's), and the units per verified step with every call counted.

## Predictions (written first)

Each line carries my probability that it reads "yes"; the reading rules are in the same order.

| # | what | probability |
|---|---|---|
| 1 | both cells finished (A 400 steps; KC 400 turns or stopped by the dollar cap or the context, every step on file, the stop said) | 80 % |
| 2 | KC: no request shows a positive cache count | 85 % |
| 3 | A: none of its requests reached 4,096 prompt tokens | 97 % |
| 4 | A: none of its requests shows a positive cache count | 95 % |
| 5 | no request below 4,096 prompt tokens, in either cell, shows a positive cache count | 65 % |
| 6 | A: the held state equals the oracle's on all 400 steps | 80 % |
| 7 | KC: the held state equals the oracle's on at least 95 percent of its turns | 40 % |
| 8 | KC: its prompt tokens add up to at least 35 million (Test 23's warm kept chat sent 39.9 million) | 80 % |
| 9 | KC: computed tokens per verified step are at least 10 times Test 23's kept chat (1,207,926 over 260 verified steps) | 90 % |
| 10 | KC: those tokens per verified step are at least 25 times A's | 90 % |
| 11 | A: its prompt tokens are within 10 percent of Test 23's arm A (470,669) | 85 % |
| 12 | A: no request has an explicit 0 in the raw field (every call None or absent) | 85 % |

How the result could have counted against AXI (fixed before the run): line 6 reading NO while KC is exact; KC verified steps above A's; or A's units per verified step above Test 23's kept chat at the same weights. None of the three happened.

## Result

A ran 400 steps (about 19 minutes). KC ran 390 turns (about 26 minutes) and was stopped at turn 390 by the 29-dollar cap (29.09 dollars for the two cells together by the typed prices; the check comes before the next call). Every row up to turn 390 is on file; no call failed.

| | A (host-held, 400 steps) | KC (kept chat, cold, 390 turns) | Test 23's warm kept chat (saved, 400 turns) |
|---|---|---|---|
| prompt tokens | 470,669 | 37,967,655 | 39,923,458 |
| of them read from the cache | 0 | 0 | 38,728,988 (97.0 %) |
| visible + thinking tokens | 13,666 + 43,861 | 11,770 + 581 | 12,126 + 1,330 |
| raw cache field: None / explicit 0 / a number | 400 / 0 / 0 | 390 / 0 / 0 | not logged |
| requests of at least 4,096 prompt tokens | 0 (largest 1,390) | 383 of 390 (largest 193,411) | 393 of 400 |
| requests of at least 1,024 / 2,048 tokens | 294 / 0 | 389 / 387 | |
| requests with a cache hit | 0 | 0 | |
| held state equal to the oracle's after the turn | 400 of 400 | 328 of 390 | 260 of 400 |
| replies exact | 400 of 400 | 385 of 390 | 398 of 400 |
| first turn that ended wrong | none | 329 | 191 |
| largest distance from the true state (cells) | 0 | 6 | 2 |
| unparsable replies | 0 | 0 | 0 |
| computed tokens (prompt - cache + visible + thinking) | 528,196 | 37,980,006 | 1,207,926 |
| computed tokens per verified step | 1,320 | 115,793 | 4,646 |
| dollars at the typed prices, whole cell | 0.57 | 28.52 | |
| dollars per verified step | 0.0014 | 0.0870 | |
| call seconds, whole cell | 1,058 | 1,467 | 1,457 |

Per verified step, the cold kept chat computed **87.7 times** what A computed (115,793 over 1,320); with output tokens weighted 4, 66.2 times (115,906 over 1,752 units); in dollars at the typed prices, 61 times (0.0870 over 0.0014). Against Test 23's warm kept chat the cold chat computed 24.9 times as much per verified step (115,793 over 4,646); the warm chat, by the same count, computed 3.5 times A's (4,646 over 1,320), with cached tokens counted as free. A's 470,669 prompt tokens are the same number as in Test 23 to the digit, and its raw cache field was None on all 400 requests; 294 of them were of 1,024 to 1,390 tokens and none showed a hit.

By windows of 50 turns (first-token time is the median of the saved first-token seconds):

| turns | KC prompt a step | KC computed a step | KC first token (s) | KC exact held state | A prompt a step | A computed a step | A first token (s) |
|---|---|---|---|---|---|---|---|
| 1 to 50 | 13,109 | 13,140 | 1.62 | 50 of 50 | 959 | 1,104 | 2.38 |
| 51 to 100 | 37,816 | 37,848 | 1.99 | 50 of 50 | 987 | 1,117 | 2.29 |
| 101 to 150 | 62,704 | 62,734 | 2.46 | 50 of 50 | 1,041 | 1,190 | 2.41 |
| 151 to 200 | 87,511 | 87,543 | 2.79 | 50 of 50 | 1,130 | 1,279 | 2.40 |
| 201 to 250 | 112,177 | 112,217 | 3.18 | 50 of 50 | 1,289 | 1,448 | 2.50 |
| 251 to 300 | 136,966 | 136,996 | 3.97 | 50 of 50 | 1,333 | 1,470 | 2.43 |
| 301 to 350 | 161,911 | 161,940 | 4.91 | 28 of 50 | 1,340 | 1,483 | 2.16 |
| 351 to 390 (A: 351 to 400) | 183,950 | 183,978 | 6.19 | 0 of 40 | 1,334 | 1,472 | 2.23 |

Test 23's warm kept chat had a median first token of 1.86, 2.14, 2.29, 2.59, 2.83, 3.85, 5.06 and 6.12 s in the same windows. The cache cut the computed tokens of the kept chat; its time to the first token and its seconds came out close to the cold chat's. A's first token stayed between 2.16 and 2.50 s through the 400 steps.

**The one wrong turn of KC.** Turn 329 (a reference to the last thing mentioned) was answered with a wrong but well-formed change; the blind application kept it, and the held state stayed 2 cells from the truth, growing to at most 6, through turn 390: 62 turns ending wrong, 288 cells off in all. Test 23's kept chat slipped once too (turn 191), at a different place. The host-held arm did not slip in either run.

## The twelve predictions, read

| # | reading |
|---|---|
| 1 | YES (A 400; KC stopped by the cap at 390) |
| 2 | YES (390 of 390 None) |
| 3 | YES (largest request 1,390) |
| 4 | YES (0 hits in 400) |
| 5 | YES (0 hits among the 407 requests below 4,096 tokens: A 400, KC 7) |
| 6 | YES (400 of 400) |
| 7 | NO (328 of 390 turns, 84.1 %; the kept chat's one slip carried to the end) |
| 8 | YES (37,967,655) |
| 9 | YES (24.9 times) |
| 10 | YES (87.7 times) |
| 11 | YES (470,669) |
| 12 | YES (400 of 400 None) |

Eleven YES and one NO; the NO is the cold kept chat's held state.

## What this answers

* **Cold and warm.** For the host-held arm the two are the same here: its requests never reached the size at which, by the reader's figure, the service caches (largest 1,390 tokens against 4,096), and the raw field never carried a number (400 of 400 None). For the kept chat: warm (Test 23, 97.0 percent of the prompt from the cache) and cold (this run, 0 cached tokens on all 390 requests, 383 of them at least 4,096 tokens). Per verified step the cold kept chat computed 87.7 times A's; the warm one 3.5 times A's with cached tokens counted as free. `docs/COST_OF_BEING_WRONG.md` section 4 gives the warm chat at other assumed weights for a cached token.
* **Missing against zero.** In all 790 requests of the two cells the field came back as None; there was no explicit 0 and no positive number.
* **The 4,096 figure** is the reader's; no request below 4,096 tokens showed a hit in the final cells (407 of them).
* **Spend per verified step including the steps that went wrong** is in the first table: verified steps are the turns that ended with the held state equal to the oracle's, and KC's 62 wrong turns count as turns and not as verified steps.

## What the numbers cover

One model (gemini-3.8-flash), one job and seed (Test 23's), 400 steps of the host-held arm and 390 turns of the cold kept chat (the dollar cap stopped it at turn 390); the dollar figures use the typed prices, and the cache-weight columns elsewhere in the repository are assumptions, not a price list. The raw rows of the two cells are in the Colab session of the run; the saved JSON holds the prompt size of every request, the totals, the windows and the counts above, and `python/tests/test_cache_pair_result.py` recounts them.

Back to the [README](../README.md).

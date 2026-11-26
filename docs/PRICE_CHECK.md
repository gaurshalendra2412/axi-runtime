# AXI at Claude, Grok and Gemini list prices: 61x to 79x cheaper per verified step than a kept chat (cold), 6.5x to 26x with a warm cache

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model, not for end users. The model is unchanged, and everything below is about the state the host holds. It fits any LLM application that holds state, short jobs and long, from the first step. Written 26 Nov 2026, after the runs. No model was called for this page and nothing was bought. Prices: `docs/results/price_check.json`. Check: `python/tests/test_price_check.py`.*

**In one paragraph.** Our cost pages count tokens on one job, run on Gemini 3.8 Flash, and [`BILLING_CHECK.md`](BILLING_CHECK.md) shows that Google's own bill agrees with those counts to about 1.3 percent. This page takes the same measured token counts and prices them at the published list prices of Anthropic (Claude Sonnet 5.5 and Opus 5.5) and xAI (Grok 4.3 and 4.7), next to Google's. The job is the one of the cache pair test: 400 steps that edit a service graph, with the host-held state (A), a kept chat cold (KC) and a kept chat with a warm cache (KW). One **verified step**, a step after which the held state equals the oracle's, costs **$0.0038 with AXI on Claude Sonnet 5.5 against $0.2319 for the kept chat cold (61.2 times) and $0.0246 warm (6.5 times); on Grok 4.3 $0.0018 against $0.1448 (79.1 times) and $0.0357 (19.5 times); on Gemini 3.8 Flash $0.0014 against $0.0870 and $0.0148**. The token counts are the same on every row (they are Gemini's): about 1,177 tokens a step with the host-held state, 97,353 on average for the kept chat.

## What was measured and what is arithmetic
The token counts are measured: they are in the saved result files of the cache pair test and of Test 23, and Google's bill confirms them in total. The prices are the providers' own, typed in from their pages on 26 Nov 2026 (sources below). The dollar figures are the first multiplied by the second. No Anthropic or xAI bill was read; the only bill read is Google's, on the page named above. The same arithmetic on Google's rates gives back the figures of that page ($0.0014, $0.0870, $0.0148; 61.2 and 10.4 times), which is the check that the method is the same.

| | A (host-held) | KC (kept chat, cold) | KW (kept chat, warm, Test 23) |
|---|---|---|---|
| steps run | 400 | 390 | 400 |
| verified steps | 400 | 328 | 260 |
| prompt tokens | 470,669 | 37,967,655 | 39,923,458 |
| of them cached | 0 | 0 | 38,728,988 |
| output tokens (visible and thinking) | 57,527 | 12,351 | 13,456 |
| largest prompt of one step | 1,390 | 193,411 | 198,405 |

A and KC are read from the logs of the 22 Nov run; KW is Test 23's saved run, as on the billing page.

## Where each price was read
| model | price page | what it says (dollars a million tokens) |
|---|---|---|
| Gemini 3.8 Flash | Google's billing report, [`BILLING_CHECK.md`](BILLING_CHECK.md) | input 0.75, cached 0.075, output 3.75 |
| Claude Sonnet 5.5 | platform.claude.com/docs/en/about-claude/pricing | base input 2, cache hits 0.10 (0.05 of base), output 10 |
| Claude Opus 5.5 | same page | base input 4, cache hits 0.20 (0.05 of base), output 20 |
| Grok 4.3 | docs.x.ai/docs/pricing, "Last updated: October 10, 2026" | under 200k tokens: input 1.25, cached 0.20, output 2.50 |
| Grok 4.7 | same page | under 200k tokens: input 2, cached 0.50, output 6 |

## The cost of a verified step
Fresh prompt tokens at the input price, cached ones at the cached price, output tokens (visible and thinking, as counted on Gemini) at the output price, divided by the verified steps of the cell (400, 328 and 260).

| model (list price, $ a million tokens: input / cached / output) | A (host-held state) | kept chat, cold | times A | kept chat, warm | times A |
|---|---|---|---|---|---|
| Gemini 3.8 Flash (0.75 / 0.075 / 3.75) | $0.0014 | $0.0870 | 61.2 | $0.0148 | 10.4 |
| Claude Sonnet 5.5 (2 / 0.1 / 10) | $0.0038 | $0.2319 | 61.2 | $0.0246 | 6.5 |
| Claude Opus 5.5 (4 / 0.2 / 20) | $0.0076 | $0.4638 | 61.2 | $0.0492 | 6.5 |
| Grok 4.3 (1.25 / 0.2 / 2.5) | $0.0018 | $0.1448 | 79.1 | $0.0357 | 19.5 |
| Grok 4.7 (2 / 0.5 / 6) | $0.0032 | $0.2317 | 72.1 | $0.0840 | 26.1 |

Counting every step, right or wrong (the same tokens divided by 400, 390 and 400 steps), so the gap is the tokens alone:

| model | A | kept chat, cold | times A | kept chat, warm | times A |
|---|---|---|---|---|---|
| Gemini 3.8 Flash | $0.0014 | $0.0731 | 51.4 | $0.0096 | 6.8 |
| Claude Sonnet 5.5 | $0.0038 | $0.1950 | 51.4 | $0.0160 | 4.2 |
| Claude Opus 5.5 | $0.0076 | $0.3900 | 51.4 | $0.0320 | 4.2 |
| Grok 4.3 | $0.0018 | $0.1218 | 66.5 | $0.0232 | 12.7 |
| Grok 4.7 | $0.0032 | $0.1949 | 60.6 | $0.0546 | 17.0 |

## What a million verified steps cost
The same per-step figures times a million: dollars for a million verified steps.

| model | A (host-held state) | kept chat, cold | kept chat, warm |
|---|---|---|---|
| Gemini 3.8 Flash | $1,422 | $86,957 | $14,811 |
| Claude Sonnet 5.5 | $3,792 | $231,887 | $24,602 |
| Claude Opus 5.5 | $7,583 | $463,773 | $49,203 |
| Grok 4.3 | $1,830 | $144,788 | $35,664 |
| Grok 4.7 | $3,216 | $231,736 | $83,978 |

The whole cells, for reference (400, 390 and 400 steps):

| model | A, whole cell | kept chat, cold, whole cell | kept chat, warm, whole cell |
|---|---|---|---|
| Gemini 3.8 Flash | $0.57 | $28.52 | $3.85 |
| Claude Sonnet 5.5 | $1.52 | $76.06 | $6.40 |
| Claude Opus 5.5 | $3.03 | $152.12 | $12.79 |
| Grok 4.3 | $0.73 | $47.49 | $9.27 |
| Grok 4.7 | $1.29 | $76.01 | $21.83 |

## What the arithmetic assumes
* The token counts are Gemini 3.8 Flash's. A different model counts a different number of tokens for the same text (Anthropic's page says its newer tokenizer produces about 30 percent more tokens than the older one). Both arms of a comparison scale together, so the ratios move little; the dollars would move with the count.
* Output tokens are Gemini's counts, visible and thinking, priced at each provider's output rate.
* On Claude the fresh tokens of the warm kept chat are priced at the base input price. Anthropic charges cache writes at 1.25 times base (5-minute cache) on top; adding that would raise the kept chat's warm figure. It is not added here.
* A pays no cache price: its prompt is rebuilt from the held state each step and is not cached on any provider here.
* No step reaches a long-context tier. The largest prompt of the kept chat cells is 198,405 tokens (warm) and 193,411 (cold); Grok's higher rates start at 200,000. Anthropic's page lists no surcharge for these two Claude models.
* The verified-step counts (400, 328, 260) are from the Gemini run. They are part of the job's result, not of the price.

## Meta's Llama
Llama is open-weight and priced by whoever serves it, so there is no single list price to put in the table. What Meta's models share with every host is the memory of the cache. Per token it is 2 (keys and values) times layers times key-value heads times head size times 2 bytes (16-bit); from the public model configurations that is 131,072 bytes for Llama 3.1 8B (32 layers, 8 key-value heads, head size 128) and 327,680 for 70B (80 layers, 8, 128). At the mean prompt of each arm on this job:

| | KV bytes a token | A: a session at its mean prompt (1,177 tokens) | kept chat: a session at its mean prompt (97,353 tokens) | times |
|---|---|---|---|---|
| Llama 3.1 8B | 131,072 | 154.3 MB | 12.76 GB | 82.7 |
| Llama 3.1 70B | 327,680 | 385.7 MB | 31.90 GB | 82.7 |


This is the arithmetic of Tests 29 and 30, which put real cache-shaped arrays on a T4 GPU and a TPU chip and counted sessions until memory ran out; here it is applied to the Llama shapes. Llama 3.1's window is 131,072 tokens, and the kept chat of this job passes it at step 264 (127 of its 390 steps are over it); the prompt of A never goes above 1,390.

## What this does and does not show
* It shows what the same measured job costs per verified step at published prices, and that the gap is in the tokens a step carries, not in any provider's price: a prompt rebuilt from the held state is about 83 times shorter here than a kept chat's mean.
* It does not show a bill from Anthropic or xAI, a run on their models, or their quality on this job. A run of the same job on those models would replace the Gemini token counts and step counts with their own; the arithmetic above is what it would be multiplied by.
* The ratio between the arms is the host's choice of what to put in the prompt. It is the same on every row because the token counts are the same; what changes from row to row is the price list.

## Check it
1. `python -m pytest python/tests/test_price_check.py` recomputes every figure above from the saved result files and the prices in `docs/results/price_check.json`, and checks that the page carries them.
2. To use other prices, change the numbers in `docs/results/price_check.json` and run `python -c "import sys; sys.path.insert(0,'python'); from axi import price_check as P; print(P.table())"`.

This page is also written up as an [article on dev.to](https://dev.to/shalendra_gaur_ecd0567a49/axi-at-claude-grok-and-gemini-list-prices-61x-to-79x-cheaper-per-verified-step-than-a-kept-chat-i5j) (26 November 2026).

Back to the [README](../README.md).

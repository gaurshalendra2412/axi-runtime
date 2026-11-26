# AXI, checked against Google's own bill: $0.0014 per verified step against $0.0870 for a kept chat (cold) and $0.0148 (warm)

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model, not for end users. The model is unchanged, and everything below is about the state the host holds. It fits any LLM application that holds state, short jobs and long, from the first step. Written 25 Nov 2026, after the runs. No model was called for this page and nothing was bought. Saved data: `docs/results/google_bill_check_2026-11-25.json`. Check: `python/tests/test_billing_check.py`.*

**In one paragraph.** The dollar figures on the cost pages are token counts from our own logs times prices the owner typed in. This page reads Google's own billing report for the owner's paid Gemini account and puts the two side by side. Google's rates are the typed ones: in the report an output token costs 5.000 times an input token (3.75 over 0.75) and a cached input token costs 0.100 of a fresh one. On the day of the cache pair test Google charged ₹2,805.00 (about $29.45) against $29.09 from the logs, 1.25 percent more; for Test 34 (2 to 3 Nov) it charged ₹168.92 (about $1.77) against $1.75, 1.1 percent more. At Google's rates, on the Test 23 job, one verified step costs **$0.0014 with the host-held state, $0.0870 with the kept chat cold (61 times) and $0.0148 with the kept chat warm (10.4 times)**. Counting every step, right or wrong, the same cells cost $0.0014, $0.0731 and $0.0096 a step: 51.4 and 6.8 times.

## Where each number can be read

Google's side is read in the account itself: AI Studio, *Billing* (balance, transactions), *Usage* (tokens a day by model) and *Spend*; and in the Cloud console, *Billing, Reports, Group by SKU*, with a date range, whose table has a *Download CSV* button. The report is in rupees, shows no tax line, and prints no exchange rate. The lines below were copied from that table into `docs/results/google_bill_check_2026-11-25.json`; account, project and key identifiers are left out. Our side is read from the saved files `axi_cache_pair_seed20261122_default.json` (arms A and KC), `axi_gemini_fuzzy_seed20261122_default.json` (Test 23, arm K) and `gemini_rulehint.json` (Test 34).

## Google's rates

Gemini 3.8 Flash, the three kinds of token, on three date ranges of the report (rupees a million tokens = rupees over tokens times a million):

| range of the report | kind | tokens | rupees | rupees a million |
|---|---|---|---|---|
| 1 to 25 Nov | input | 47,731,389 | 3,409.10 | 71.42 |
| 1 to 25 Nov | cached input | 57,158,454 | 408.24 | 7.14 |
| 1 to 25 Nov | output | 1,062,778 | 379.53 | 357.11 |
| 22 Nov | input | 38,894,516 | 2,777.94 | 71.42 |
| 22 Nov | cached input | 33,833 | 0.24 | 7.1 |
| 22 Nov | output | 75,084 | 26.81 | 357.07 |
| 1 to 3 Nov | input | 1,043,909 | 74.56 | 71.42 |
| 1 to 3 Nov | output | 264,232 | 94.36 | 357.11 |

The rate does not move with the day or the volume. Output over input is 5.000 and cached over input is 0.100, which are the typed 3.75 over 0.75 and the 10 percent for a cached token; neither ratio needs an exchange rate. The typed $0.75 a million input tokens is ₹71.42, so the report's rupees are ₹95.23 to the dollar; the output line gives the same 95.23 from $3.75. The dollar figures below use that rate; the percentage differences do not depend on it. The Gemini 3 Flash lines of the month (5 Nov, another model, not used below) show the same pattern: cached 0.100 of input, output 6.000 times input, which is $0.50, $0.05 and $3.00 at that rate.

## The cache pair test, 22 Nov

Both arms of the test ran on 22 Nov, so Google's lines for that day are their sum.

| | our logs, A and KC together | Google, 22 Nov | difference |
|---|---|---|---|
| input tokens (fresh and cached) | 38,438,324 | 38,928,349 | +490,025 (+1.27 %) |
| of them cached | 0 | 33,833 | |
| output tokens (visible and thinking) | 69,878 | 75,084 | +5,206 (+7.45 %) |
| money | $29.09 | ₹2,805.00 = $29.45 | +$0.36 (+1.25 %) |

The logs hold the calls of the two cells. Google's day holds 490,025 more input tokens, 33,833 of them cached, and 5,206 more output tokens; whether those are calls that the cells' logs do not have or a difference in counting, the bill does not say. Per our logs, A's share is $0.57 of the $29.09 (2.0 percent); the bill has one line per kind of token per day, so it does not split the day between the two cells.

## Test 34, 2 to 3 Nov

| | our logs, five arms | Google, 1 to 3 Nov | difference |
|---|---|---|---|
| input tokens | 1,031,724 | 1,043,909 | +12,185 (+1.18 %) |
| output tokens (visible and thinking) | 261,467 | 264,232 | +2,765 (+1.06 %) |
| money | $1.75 | ₹168.92 = $1.77 | +$0.02 (+1.1 %) |

## The cost of a verified step at Google's rates

A verified step is a step after which the held state equals the oracle's, as on [`CACHE_PAIR_TEST.md`](CACHE_PAIR_TEST.md). Cold and warm are as there. Dollars: fresh prompt tokens at $0.75 a million, cached ones at $0.075, output at $3.75.

| | A (host-held) | KC (kept chat, cold) | KW (kept chat, warm, Test 23) |
|---|---|---|---|
| prompt tokens | 470,669 | 37,967,655 | 39,923,458 |
| of them cached | 0 | 0 | 38,728,988 (97.0 %) |
| output tokens | 57,527 | 12,351 | 13,456 |
| dollars, whole cell | 0.57 | 28.52 | 3.85 |
| verified steps | 400 | 328 | 260 |
| dollars per verified step | 0.0014 | 0.0870 | 0.0148 |
| times A | 1 | 61.2 | 10.4 |
| dollars per step, every step counted whether right or wrong | 0.0014 | 0.0731 | 0.0096 |
| times A, wrongness ignored | 1 | 51.4 | 6.8 |

The last two rows leave out whether a step was right: the same tokens, divided by every step the cell ran (400, 390 and 400), so the gap there is the tokens alone. A and KC are read from the logs of the 22 Nov run, which Google's bill confirms in total. KW is Test 23's saved run: its dollars are worked out from its tokens at Google's rates, not read from a bill line of its own, because it ran on other days among the rest of the month. The cache pair page typed no cached price; this is the first place the cached price is Google's.

## The month, as Google has it

1 to 25 Nov: ₹4,249.85 (about $44.63), no tax line; the Spend page shows ₹4,249.84. Of the 104,889,843 Gemini 3.8 Flash input tokens of the month, 54.5 percent were billed as cached. By date range: 22 Nov ₹2,805.00 (the cache pair test), 1 to 3 Nov ₹168.92 (Test 34), the Gemini 3 Flash lines ₹52.98, and ₹1,222.95 on the other days (the usage page puts them on 13 and 14 Nov). That last part is not tied to a test on this page.

## What the bill does not show

* The split of 22 Nov between A and KC. The bill confirms the total and the rates; the split is from the logs.
* KW's own line, for the reason above.
* Other models and other providers. This is Gemini 3.8 Flash, one paid account, one month.
* The usage page and the report differ on input tokens: AI Studio's *Usage* page shows about 117.8M input tokens for 1 to 25 Nov against 112.3M in the report (+4.96 percent); output tokens agree to 0.015 percent (about 1,074,911 against 1,074,754). The report is used here because it is what is charged.

## Check it

1. In AI Studio, open *Billing*, *Usage* and *Spend* for the billing account; in the Cloud console open *Billing, Reports*, group by SKU, set the dates (22 Nov 2026 or 1 to 3 Nov 2026) and read the table or press *Download CSV*. The lines should be those in the JSON.
2. `python -m pytest python/tests/test_billing_check.py` recomputes every figure above from the JSON and from the three saved result files, and checks that the page carries them.

Back to the [README](../README.md).

# Test 28: a model plain against a model given the pointer gate's findings
## Gemini 3.8 Flash on 300 Juliet functions (use after free, double free). Pre-registered 30 Oct 2026

**Idea of the framework: Rajnish Choubey. Code, tests and measurements: Shalendra Gaur with an AI coding assistant (a tool).**

## What this is, in plain words
The pointer gate (`ptrgate.py`) is a program with set rules that reads C/C++ and says whether a function uses memory after it was freed or frees it twice. This test asks whether handing a language model **the gate's exact findings** makes the model more right than the same model alone. This is the "model with and without AXI" shape of a test, for one small job.

## Read this first: what this test can and cannot say
* **The gate was fitted on these cases** (pages `PTRGATE_CHECK` 1 to 4: it was changed after I saw its misses). Arm X will therefore read close to the gate's own score. That is a **demonstration of what exact host facts do for a model**, not a result on new code and not a benchmark score. It is **not** CWE-Bench, not a real-vulnerability result.
* Juliet is synthetic and small. Plain Gemini may already be near the ceiling; then the gain will be small, and that is a fair reading too.
* One model, one run per arm, one sample of 300 functions. Temperature is the API default.

## Setup
* Items: 130 vulnerable and 130 clean functions from the data-flow and multi-file variants (21 to 68), plus 20 vulnerable and 20 clean functions from C++ classes and containers (`rest`), which the gate marks **unsupported**. 300 items, drawn with seed 28, shuffled.
* One function is one question: "can it use memory after it was freed or deleted, or free it twice (counting the functions it calls)?" Reply is one JSON object `{"memory_error": bool, "why": ...}`.
* Arm **P** (plain): the source only. Arm **X**: the same plus the gate's findings on that function (its exact path, or the statement that it cannot read the code).
* Prompts: comments and preprocessor lines removed, strings removed, every name carrying a label word or a case name replaced by a random `f<number>`, the `main` function and the `good` wrapper removed. The gate is run on that same changed source. A test checks that no label word (bad, good, G2B, B2G, CWE, flaw, fix, use_after_free, double_free) is in any prompt.
* Same model in both arms (`gemini-3.8-flash`, thinking default), 600 calls, one reply per call, cached on disk. Cost about one to two dollars at the prices you type in.
* Right = the JSON `memory_error` equals the label. A reply that cannot be read counts as wrong.
* Groups: **all** (300), **hard** (the 260 data-flow and multi-file cases; the gate marks 27 of the clean ones unsupported), **rest** (40 C++ class/container cases, gate unsupported), **cross** (variants 45 and 54 and 61 to 68: the freeing happens in another function or file).

## Predictions (written before any call; probability that the line reads YES)
1. Both arms give a readable JSON on at least 95 % of items: **90 %**.
2. Arm P accuracy on all 300 is at least 0.90: **45 %** (my guess is 0.85 to 0.95; plain models are good at short C but miss data passed through function pointers and other files).
3. Arm X accuracy on all 300 is at least 0.97: **85 %**.
4. Arm X beats arm P by at least 3 points on all 300: **55 %**.
5. Arm P accuracy on the cross-function group is at most 0.90: **50 %**.
6. Arm X false-alarm rate (clean functions called vulnerable) is at most 0.02: **85 %**.
7. On the 40 gate-unsupported `rest` items, arm X is within 3 points of arm P (the model should not be helped or hurt when the gate says it cannot read): **60 %**.
8. Arm P false-alarm rate is higher than arm X's: **70 %**.

## How the result will be read
Every figure is recomputed by a test from the saved replies (`docs/results/ptrgate_model.json`). Each line gets YES or NO; a NO is written as plainly as a YES. If arm P is already at the ceiling, the page will say the demonstration shows no gain on this data. No claim will be made about real-world vulnerabilities.

## Result
*Run on 30 Oct 2026 in Colab, `gemini-3.8-flash`, 600 calls, all replies readable (300 of 300 in each arm). Figures are recomputed by a test from `docs/results/ptrgate_model.json`.*

| group | n | plain: right | plain: false alarms | with gate: right | with gate: false alarms |
|---|---|---|---|---|---|
| all | 300 | 295 (0.983) | 2 of 150 clean (0.013) | 297 (0.990) | 0 of 150 (0.000) |
| hard (data-flow and multi-file) | 260 | 258 (0.992) | 2 of 130 | 260 (1.000) | 0 of 130 |
| cross-function (variants 45, 54, 61 to 68) | 134 | 133 (0.993) | 1 | 134 (1.000) | 0 |
| rest (C++ classes and containers; gate says unsupported) | 40 | 37 (0.925) | 0 | 37 (0.925) | 0 |

Recall on vulnerable functions: 0.980 in both arms (the 3 missed ones are the same C++ class/container cases, where the gate said "cannot read").

### The eight lines
1. Both arms readable on at least 95 %: **YES** (100 % and 100 %).
2. Plain accuracy at least 0.90: **YES** (0.983). Plain was already near the ceiling.
3. With-gate accuracy at least 0.97: **YES** (0.990).
4. With-gate beats plain by at least 3 points: **NO** (0.7 points: 2 items out of 300).
5. Plain accuracy on the cross-function group at most 0.90: **NO** (0.993). I expected the model to miss freeing done in another function or file; it did not.
6. With-gate false-alarm rate at most 0.02: **YES** (0.000).
7. On the 40 unsupported C++ items the arms are within 3 points: **YES** (identical, 0.925).
8. Plain false-alarm rate higher than with-gate: **YES** (0.013 against 0.000, 2 items).

6 of 8 lines read YES, 2 read NO.

### What this says, and what it does not
* **On this data the model plain is already almost as good as the gate.** The gate added two corrected false alarms (clean functions the plain model called vulnerable) and no new misses. It did not lift accuracy by the 3 points I guessed. My guess that plain models would miss cross-function cases was wrong for this model on this synthetic data.
* The two corrected items are tiny numbers (2 of 300). They could be chance; one run, one sample. They are on clean functions where the gate itself said "unsupported", so the gate's facts did not decide them; I do not know why the model did better there.
* **Where the gate cannot read (C++ classes and containers), neither arm helps:** both miss 3 of 20 vulnerable functions there. Nothing in the host facts fixes that.
* **Time and thinking (descriptive, not a pre-registered line):** mean seconds per call 7.42 plain against 6.27 with the gate; mean thinking tokens 498 against 299. With the gate the model thought about 40 % less for the same or slightly better answers. One run; not a tested claim.
* **Not shown:** anything on real vulnerabilities or real repositories; any gain in accuracy on Juliet; any other model. The gate was fitted on these cases, so the with-gate arm is a demonstration and not an independent result. Juliet is small and synthetic, which is the likeliest reason plain Gemini scores so high.
* **What is supported:** giving the model exact host facts did not hurt it on any group, removed the plain model's two false alarms, and was associated with less thinking per call. A fair next test needs code a model plain does not already solve (real repositories), which this page did not attempt.

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

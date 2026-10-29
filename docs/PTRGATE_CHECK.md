# The pointer gate: a free check on Juliet use-after-free and double-free cases

*Written before the held-out run. The text above `## Result` is locked by a SHA-256 hash recorded in `docs/results/ptrgate_check_hash.txt`. No model and no cost: the gate is a host-side rule.*

## What this is

A small host-side piece of AXI, `python/axi/engine/ptrgate.py`. It reads the dangling-edge condition of the double pushout (Rajnish Choubey's gluing conditions, which the framework already uses on service graphs) on C and C++ heap pointers: a heap object is a node, a pointer variable is an edge to it. Freeing the node while an edge still points at it leaves a dangling edge; using that edge is a use after free (CWE-416); freeing a deleted node is a double free (CWE-415). The gate says so by fixed rules, in microseconds, with no model. It gives a model, if one is used, the exact findings.

It is **our** reading of the idea on code: how a pointer becomes an edge, how an alias shares its node, which branches are reachable. Nothing here is a claim that Sir wrote this rule for code.

## The data and the way it is cut

NIST's Juliet C/C++ test cases for CWE-416 and CWE-415 (public domain), control-flow variants 01 to 18 only (one file each). One function is one item: `*_bad` is vulnerable, `goodG2B`, `goodB2G` and the like are clean, the `*_good` wrapper is left out. Comments are removed before the gate sees the code, so the words "FLAW" and "FIX" carry no hint.

* **Development part** (where the gate was written and fixed): the `malloc_free_char` and `malloc_free_int` families, 72 files, 320 functions. The gate reads **320 of 320** correct there. That number is **not evidence**: the code was fitted to it.
* **Held-out part** (this page's test): every other family (`malloc_free_int64_t`, `_long`, `_struct`, `_wchar_t`; all `new_delete` and `new_delete_array` families, with class and struct; `return_freed_ptr`). The gate is **not changed** after this page is locked.

Scope of the gate: one function at a time, if/else, loops taken once or not at all, switch on a constant, goto to a later label, branches with a constant condition, malloc/calloc/new, free/delete, aliasing by plain assignment. Anything else makes the function `unsupported` and the gate claims nothing. A call into another function that frees for us is **not** followed.

## The lines (my probability that the line reads yes)

| # | line | my probability |
|---|---|---|
| 1 | [guard] on the held-out part at least 90 % of functions are analysed (unsupported at most 10 %) | 85 % |
| 2 | recall at least 0.95 (of the vulnerable functions that are analysed) | 55 % |
| 3 | false-alarm rate at most 0.02 (of the clean functions that are analysed) | 75 % |
| 4 | accuracy at least 0.97 | 55 % |
| 5 | every missed vulnerable function (if any) belongs to the `return_freed_ptr` family, where the freeing is done in another function | 55 % |
| 6 | the CWE-415 (double free) recall is 1.0 | 70 % |
| 7 | the C++ `new_delete` families together reach accuracy at least 0.97 | 55 % |

**Headline:** lines 2 and 3 both read yes ("THE POINTER GATE FINDS THE USE AFTER FREE AND THE DOUBLE FREE ON CASES IT WAS NOT WRITTEN ON"); otherwise MIXED.

## What each reading would mean

* If the headline reads yes, the dangling-edge reading of pointers works as a deterministic host check on this synthetic suite, and a model could be given its findings.
* **Juliet is synthetic and the cases are small.** A reading of yes here says nothing about real code and real vulnerabilities, where a model plain may already be strong and where the code needs a real parser. This check can say whether the idea is sound on clean cases, and nothing more.
* The model test (a model plain against a model given the gate's findings) is **not** made here. It is the next step and would have to be on harder data.

## Result

*Written after the run, below the locked text; nothing above this line has been changed. The figures are in `docs/results/ptrgate_check_held_frozen_gate.json` (the gate as locked) and `docs/results/ptrgate_check_held_after_fix.json` (after one bug fix, below). Free, no model. Written for Sir to confirm.*

**What happened, in order, with nothing left out.**

1. **The first look at the held-out part had a mistake in my reader, not in the gate.** In the C++ files the vulnerable function is called `bad()`, not `*_bad`, so my reader left every one of them out and only the clean ones were counted. That run read accuracy 0.9939 on 2,929 functions and was wrong in its own terms (no C++ vulnerable function in it). I fixed the reader and discarded those figures.
2. **The gate as locked, on the held-out part (3,433 functions, 666 files, 0 unsupported):** recall **0.8378** (558 of 666), false-alarm rate **0.0000** (0 of 2,767), accuracy **0.9685**. Of the seven lines: **1 yes, 2 NO, 3 yes, 4 NO, 5 NO, 6 yes, 7 NO**. Headline: **MIXED** (line 2 NO). The CWE-415 double-free recall was 1.0 (324 of 324).
3. **The misses (108) were of two kinds.** 18 are the `return_freed_ptr` family (the freeing is in another function: the gate does not follow calls, as stated). 90 are in the C++ `new_delete` families (`char`, `int`, `int64_t`, part of `long`), and they were a **bug in the gate**: `*data = 5;` was read as the pointer being reassigned, so the gate forgot that `data` had been deleted.
4. **After fixing that one rule** (a store through a pointer is a use, not a reassignment), the held-out part reads: recall **0.9730** (648 of 666), false alarms **0** of 2,767, accuracy **0.9948**; **every one of the remaining 18 misses is a `return_freed_ptr` function**. On the development part the gate still reads 320 of 320. **This second reading is not a pre-registered one:** the fix was made after seeing the held-out failures, so it counts as development for the next test, and a fresh held-out part is needed before it is claimed.

**What this says.** The dangling-edge reading of pointers works as a deterministic host check on these cases: no false alarm in 2,767 clean functions, every double free found, and every use after free found except where the freeing is in another function. It also found a real bug in its own first version, which is what a held-out part is for.

**What it does not say.** Juliet is synthetic and its cases are tiny; a model plain is likely already strong on them, so this check cannot show that the gate helps a model. It says nothing about real vulnerabilities. Cross-function freeing is not covered. The model test (a model plain against a model given these findings) is not made here and would need harder data.

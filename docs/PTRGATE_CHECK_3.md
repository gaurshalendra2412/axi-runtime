# The pointer gate, third check: the multi-file variants nobody fitted it to

*Written before the run. The text above `## Result` is locked by a SHA-256 hash recorded in `docs/results/ptrgate_check_3_hash.txt`. No model and no cost.*

## What changed since the second check, and why this page exists

The second check (`PTRGATE_CHECK_2.md`) read **MIXED**: recall 0.636, 120 false alarms on the data-flow variants 21 to 45. We then rebuilt the gate to read what it had only guessed before, in the way the second page named as the next step:

* **transport against dereference:** passing, copying, returning or comparing a pointer is safe whatever it points at; only `*p`, `p->f`, `p[i]`, a library call that reads it and `free(p)` need a live node;
* **a function of the file that is called as a statement is entered** with the caller's pointers (depth at most 4), so what it does with its pointer parameters is read from its body;
* **references, pointers to pointers, unions, function pointers, global and static variables, and global integer flags** whose value is known.

With these, the gate reads every function of the three parts it had seen (control-flow 01 to 18, development and held-out, and the data-flow 21 to 45) **correctly: 4,605 of 4,605**. That number is **not evidence**; the code was fitted to those parts, three times.

## The data: cases it has never been run on

The Juliet **multi-file** variants (CWE-416 and CWE-415, all families), the functions of one case joined into one text: variants **51 to 54** (the pointer passed down a chain of one to four functions in other files) and **61 to 68** (returned from a function, through a reference, through a pointer to a pointer, as `void *`, through a function pointer, through an array, a struct, and a global pointer). 300 cases, 720 files. Not included, and not attempted: 72 to 74 (standard containers), 81 to 84 (virtual classes). One function is one item: `*_bad` and `bad()` are vulnerable; `goodG2B`, `goodB2G` and the like are clean; the `...Sink` and `...Source` helpers are not counted. Comments are removed first.

**The gate is not changed after this page is locked.**

## The lines (my probability that the line reads yes)

| # | line | my probability |
|---|---|---|
| 1 | [guard] at least 90 % of the functions are analysed (unsupported at most 10 %) | 85 % |
| 2 | recall at least 0.90 | 40 % |
| 3 | false-alarm rate at most 0.02 | 55 % |
| 4 | recall at least 0.95 on variants 51 to 54 (the chain of calls) | 55 % |
| 5 | recall at least 0.90 on variants 61 to 65 (return, reference, pointer to pointer, `void *`, function pointer) | 40 % |
| 6 | recall at least 0.90 on variants 66 to 68 (array, struct, global pointer kept in another file) | 25 % |
| 7 | every false alarm (if any) is in a clean function of the `goodB2G` kind (the freed pointer handed on to a sink that ignores it) | 50 % |

**Headline:** lines 2 and 3 both read yes ("THE POINTER GATE FINDS THE FREED POINTER AND RAISES NO FALSE ALARM ON CASES SPREAD OVER SEVERAL FILES"); otherwise MIXED.

## What each reading would mean

* A yes would say the transport / dereference rule and the entering of called functions carry beyond the cases they were fitted to. A MIXED would say where the rules stop; the per-variant table in the Result is the list of what to build next.
* Juliet is synthetic, and a yes still says nothing about real code. The model test (a model plain against a model given the gate's findings) is **not** made here and waits for a gate that is right on cases like these.

## Result

*Written after the run, below the locked text; nothing above this line has been changed. Figures: `docs/results/ptrgate_check_3.json`, recomputed by `tests/test_ptrgate.py`. Free, no model. For Sir to confirm.*

**The gate was not changed between the lock and the run.** On the 300 multi-file cases (720 files, 1,374 functions counted, 0 unsupported): 240 true positives, 60 misses, 14 false alarms, 1,060 true negatives. Recall **0.800** (240 of 300), false-alarm rate **0.0130** (14 of 1,074), accuracy **0.9461**.

Of the seven lines: **1 yes, 2 NO, 3 yes, 4 yes, 5 NO, 6 NO, 7 NO**. Headline: **MIXED** (line 2 NO).

| line | reading |
|---|---|
| 1 guard | 0 unsupported of 1,374: yes |
| 2 recall ≥ 0.90 | 0.800: **NO** |
| 3 false alarms ≤ 0.02 | 0.0130: yes |
| 4 variants 51 to 54, recall ≥ 0.95 | 80 of 80 = 1.0: yes |
| 5 variants 61 to 65, recall ≥ 0.90 | 120 of 160 = 0.75: **NO** (variant 64 found 0 of 40) |
| 6 variants 66 to 68, recall ≥ 0.90 | 40 of 60 = 0.667: **NO** (variant 66 found 0 of 20) |
| 7 every false alarm is in a `goodB2G`-kind function | all 14 are in the C++ wrapper `good()` of CWE-415 variant 54, not in a `goodB2G` function: read strictly, **NO** (the cause is the one named, see below) |

**By variant** (TP, FN, FP, TN): 51: 20, 0, 0, 54. 52: 20, 0, 0, 110. 53: 20, 0, 0, 138. 54: 20, 0, **14**, 152. 61: 20, 0, 0, 54. 62: 40, 0, 0, 120. 63: 40, 0, 0, 108. **64: 0, 40**, 0, 108. 65: 20, 0, 0, 54. **66: 0, 20**, 0, 54. 67: 20, 0, 0, 54. 68: 20, 0, 0, 54.

**What this says, plainly.** On cases it was never fitted to, the gate finds the freed pointer through a chain of calls (51 to 54), a return (61), a reference (62), a pointer to a pointer (63), a function pointer (65), a struct (67) and a global pointer kept in another file (68), with no miss in any of them, and with a false-alarm rate of 1.3 %. It reads nothing in two kinds of case: a pointer hidden behind `void *` (variant 64, 40 misses) and a pointer kept in an array (variant 66, 20 misses). Its 14 false alarms have one cause, which the locked page stated as a limit: a call chain of four files plus the wrapper that calls it is deeper than the depth limit of 4, and past the limit the gate falls back to counting the argument as used.

**What to build next, in order of size.** The reading of `void *` casts and of arrays of pointers (60 misses); a deeper limit or a summary kept for each called function (14 false alarms). These failures are now known, so a gate changed for them needs another part it has not seen: the variants 72 to 74 and 81 to 84 are still unseen, and so are the other CWEs.

**What it does not say.** Juliet is synthetic; none of this is a statement about real code, and the model test (a model plain against a model given the gate's findings) is not made here.

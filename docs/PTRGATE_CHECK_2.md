# The pointer gate, second check: data-flow variants it was not written on

*Written before the run. The text above `## Result` is locked by a SHA-256 hash recorded in `docs/results/ptrgate_check_2_hash.txt`. No model and no cost.*

## Why a second check

The first check (`PTRGATE_CHECK.md`) ended with a gate that, after two fixes made while looking at the held-out failures, reads every function of the control-flow variants 01 to 18 correctly (3,433 held-out functions and 320 development ones). Those numbers are now development numbers: the gate was fitted to them. This page asks the question again on cases it has never been run on and has not been fitted to.

## The data

The same NIST Juliet C/C++ test cases (CWE-416 use after free, CWE-415 double free), but the **data-flow variants that fit in one file**: 21 (flags kept in static variables), 31 (the pointer copied into another variable), 32 (the pointer reached through a pointer to it), 33 (through a C++ reference), 34 (through a union), 41 (the pointer passed to another function), 42 (the pointer returned by another function), 43 (a source function writes it through a reference), 44 (the sink called through a function pointer), 45 (the pointer kept in a static global). 220 files, all families. One function is one item: `*_bad` and `bad()` are vulnerable; the main `goodG2B`, `goodB2G` and the like are clean; the helper functions called `...Sink` and `...Source` are left out of the count. Comments are removed before the gate sees the code.

**The gate is not changed after this page is locked.** What it does: one function at a time; a call to another function in the same file that returns a pointer it has freed is followed (`return_freed_ptr`); a pointer passed to another function counts as used. What it does not do: it does not follow calls that free for us, does not read global or static variables set in another function, does not follow pointers to pointers, references or unions.

## The lines (my probability that the line reads yes)

| # | line | my probability |
|---|---|---|
| 1 | [guard] at least 90 % of the functions are analysed (unsupported at most 10 %) | 80 % |
| 2 | recall at least 0.90 (of the vulnerable functions analysed) | 45 % |
| 3 | false-alarm rate at most 0.02 (of the clean functions analysed) | 60 % |
| 4 | CWE-415 (double free) recall at least 0.90 | 50 % |
| 5 | variant 31 (pointer copied) recall is 1.0 | 70 % |
| 6 | variants 44 and 45 (function pointer, static global) recall at least 0.90 | 30 % |
| 7 | every missed vulnerable function (if any) is in a variant among 21, 32, 33, 34, 43, 45 (static flags, pointer to pointer, reference, union, source writing through a reference, static global) | 55 % |

**Headline:** lines 2 and 3 both read yes ("THE POINTER GATE STILL FINDS THE FREED POINTER WHEN THE POINTER TRAVELS THROUGH OTHER VARIABLES AND FUNCTIONS"); otherwise MIXED.

## What each reading would mean

* If the headline reads yes, the dangling-edge reading carries beyond the control-flow cases. If it reads MIXED, the lines 4 to 7 say where the host would need more (following frees done in other functions, global state, indirection), which is what we would build next.
* Juliet is synthetic; a yes here still says nothing about real code. The model test is still not made here.

## Result

*Written after the run, below the locked text; nothing above this line has been changed. Figures: `docs/results/ptrgate_check_2.json`, recomputed by `tests/test_ptrgate.py`. Free, no model. For Sir to confirm.*

**The gate was not changed between the lock and the run.** On the 220 single-file data-flow files (852 functions counted, 0 unsupported): 140 true positives, 80 misses, 120 false alarms, 512 true negatives. Recall **0.636** (140 of 220), false-alarm rate **0.190** (120 of 632), accuracy 0.765.

Of the seven lines: **1 yes, 2 NO, 3 NO, 4 NO, 5 yes, 6 NO, 7 yes**. Headline: **MIXED** (lines 2 and 3 both read NO).

| line | reading |
|---|---|
| 1 guard | 0 unsupported of 852: yes |
| 2 recall ≥ 0.90 | 0.636: **NO** |
| 3 false alarms ≤ 0.02 | 0.190: **NO** |
| 4 CWE-415 recall ≥ 0.90 | 140 of 200 = 0.70: **NO** (CWE-416: 0 of 20) |
| 5 variant 31 recall = 1.0 | 20 of 20: yes |
| 6 variants 44 and 45 ≥ 0.90 | 44: 20 of 20; 45: 0 of 20; together 0.50: **NO** |
| 7 every miss is in 21, 32, 33, 34, 43, 45 | misses are in 33, 43 and 45: yes |

**By variant** (TP, FN, FP, TN): 31 copy: 20, 0, 0, 54 (exact). 42 return from function: 20, 0, 0, 54 (exact). 41 pass to function: 20, 0, **20**, 34. 44 function pointer: 20, 0, **20**, 34. 32 pointer to pointer: 20, 0, **20**, 34. 34 union: 20, 0, **20**, 34. 21 static flags: 20, 0, **40**, 34. 33 reference: **0**, 20, 0, 60. 43 source writes through a reference: **0**, 40, 0, 120. 45 static global: **0**, 20, 0, 54.

**What this says, plainly.** The gate carries through copies (31) and returned pointers (42) without an error, and finds the freed pointer in 41, 44, 32, 34 and 21 as well. It does not follow a pointer that is written through a reference (33, 43) or kept in a static global (45): 80 vulnerable functions missed. And its rule "a pointer passed to another function counts as used" gives **120 false alarms**, all in clean functions that hand a freed pointer to a sink that ignores it. Both are limits stated on the locked page, and both are the next things to build: a summary of what a called function does with its pointer parameters (uses it, frees it, ignores it) and the state of static globals. Because these failures are now known, any gate changed for them has to be tested again on cases it has not seen: the multi-file variants 51 to 84 are unseen.

**What it does not say.** Juliet is synthetic; none of this is a statement about real code, and the model test is still not made.

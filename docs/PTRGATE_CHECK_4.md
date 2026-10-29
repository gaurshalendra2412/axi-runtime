# The pointer gate, fourth check: containers and virtual classes

*Written before the run. The text above `## Result` is locked by a SHA-256 hash recorded in `docs/results/ptrgate_check_4_hash.txt`. No model and no cost.*

## What changed since the third check, and what is left unseen

The third check (`PTRGATE_CHECK_3.md`) read **MIXED** on the multi-file variants 51 to 68: recall 0.800, false alarms 1.3 %. Its failures were three: a pointer hidden behind `void *` (variant 64), a pointer kept in an array (66), and a call chain deeper than the depth limit of 4 (variant 54). We corrected all three: casts and outer parentheses are read through; an array of pointers carries the pointer put into it; the depth limit is 8. The gate now reads every function of all four parts it has seen (control-flow 01 to 18, data-flow 21 to 45, multi-file 51 to 68) correctly: **8,166 of 8,166**. That is **not evidence**: it was fitted to them.

## The data: what nobody fitted it to

The Juliet variants left out so far (CWE-416 and CWE-415, all families, the files of one case joined into one text): **72 to 74**, where the pointer travels through a standard container (`std::vector`, `std::deque`, `std::list`), and **81 to 84**, where the sink is a method of a virtual class called through a base reference. One function is one item: `bad()` is vulnerable; `goodG2B`, `goodB2G` are clean. Comments are removed first.

What the gate does not do on these, and what I expect to go wrong: it does not model a container (`vector.push_back(data)`, `vector[2]`), and it does not know which class a virtual call lands in, so a call `baseObject.action(data)` is read as an unknown call whose argument is used. That should keep the freed pointer in the bad cases and raise a false alarm in the clean `goodB2G` ones.

**The gate is not changed after this page is locked.**

## The lines (my probability that the line reads yes)

| # | line | my probability |
|---|---|---|
| 1 | [guard] at least 90 % of the functions are analysed (unsupported at most 10 %) | 80 % |
| 2 | recall at least 0.90 | 35 % |
| 3 | false-alarm rate at most 0.02 | 40 % |
| 4 | recall at least 0.90 on variants 72 to 74 (containers) | 15 % |
| 5 | recall at least 0.90 on variants 81 to 84 (virtual classes) | 70 % |
| 6 | false-alarm rate at most 0.02 on variants 81 to 84 | 25 % |
| 7 | every missed vulnerable function (if any) is in variants 72 to 74 | 60 % |

**Headline:** lines 2 and 3 both read yes ("THE POINTER GATE HOLDS ON CONTAINERS AND VIRTUAL CLASSES"); otherwise MIXED.

## What each reading would mean

A MIXED here would be the expected reading and would tell what to build next: a model of containers and of virtual dispatch. Juliet is synthetic and a yes would still say nothing about real code. The model test is **not** made here.

## Result

*Written after the run, below the locked text; nothing above this line has been changed. Figures: `docs/results/ptrgate_check_4.json`, recomputed by `tests/test_ptrgate.py`. Free, no model. For Sir to confirm.*

**The gate was not changed between the lock and the run.** On the 140 cases of variants 72 to 74 and 81 to 84 (440 files; all of them CWE-415, because CWE-416 has no such variants; 680 functions counted, 0 unsupported): 80 true positives, 100 misses, 160 false alarms, 340 true negatives. Recall **0.444** (80 of 180), false-alarm rate **0.320** (160 of 500), accuracy **0.618**.

Of the seven lines: **1 yes, 2 NO, 3 NO, 4 NO, 5 NO, 6 NO, 7 NO**. Headline: **MIXED**, and a weak one.

| line | reading |
|---|---|
| 1 guard | 0 unsupported of 680: yes |
| 2 recall ≥ 0.90 | 0.444: **NO** |
| 3 false alarms ≤ 0.02 | 0.320: **NO** |
| 4 variants 72 to 74, recall ≥ 0.90 | 40 of 60 = 0.667: **NO** |
| 5 variants 81 to 84, recall ≥ 0.90 | 40 of 140 = 0.286: **NO** (my 70 % was wrong) |
| 6 variants 81 to 84, false alarms ≤ 0.02 | 80 of 320 = 0.25: **NO** |
| 7 every miss is in 72 to 74 | misses are in 74 (20), 83 (40) and 84 (40): **NO** |

**By variant** (TP, FN, FP, TN): 72: 20, 0, 40, 20. 73: 20, 0, 40, 20. 74: **0, 20**, 0, 60. 81: 20, 0, 40, 20. 82: 20, 0, 40, 20. 83: **0, 40**, 0, 100. 84: **0, 40**, 0, 100.

**What this says, plainly.** The gate does not carry to C++ code in which the pointer travels through a standard container (72 to 74) or through the methods, constructors and destructors of a class (81 to 84). In 72, 73, 81 and 82 it finds every double free but also raises a false alarm on every clean `goodB2G` case, as the locked page expected: a method call or a container call is read as an unknown call that uses its argument. In 74, 83 and 84 it finds nothing, because the pointer is kept in an object field or freed by a constructor or destructor the gate does not read.

**What it does not say.** The three corrections of the previous round held on what they were fitted to and nothing here contradicts them, but this check shows how far the gate goes: **C-style functions and pointers, yes; C++ classes and containers, no.** The gate does not claim them; it should return `unsupported` there instead of a verdict. Juliet is synthetic and none of this is a statement about real code. The model test is not made here.

**A correction to the locked text.** The locked page says the gate reads "8,166 of 8,166" functions of the four parts it had seen. The right count is **5,979** (320 development + 3,433 held-out + 852 data-flow + 1,374 multi-file, all read correctly after the three corrections). I mistyped the sum; the locked text is left as it was, because it was locked before the run, and the error changes no prediction and no figure of the result.

**After the page: what the gate does now on these cases.** The gate was changed after this Result was written: it now answers `unsupported` for a source with a C++ method definition (`Class::method`), a virtual call, a template, a destructor, or a standard container (`vector<...>`, `std::list`, and the like), and claims nothing about it. Run again on the same 440 files, the gate reads **680 of 680 functions as unsupported**, with no false alarm and no miss; the three parts that are C-style code and the C++ files that only use `new` and `delete` on plain types are unchanged (320, 3,433, 852 and 1,374 functions, all read correctly; fitted data, not evidence). This changes how the gate reports on code it cannot read, and does not add anything it can read. The figures above stay the figures of the gate as it was locked.

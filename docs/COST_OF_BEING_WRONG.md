# The cost of being wrong

*A company's view of two measured runs. Generated from `docs/results/axi_gemini_long_session_seed20261111_default.json` (test 22, the long session) and `docs/results/axi_gemini_fuzzy_seed20261122_default.json` (test 23, the fuzzy-state text), both on Gemini 3.8 Flash. Nothing below is typed by hand; test 146 checks that this page is the render of those two files. The architecture is Rajnish Choubey's; the code, the tests and the measurements are the owner's, with an AI assistant as a tool.*

**In one paragraph.** A company that runs a model pays for every token and every second whether the state the model is working on is right or wrong. In test 23 the kept chat ended 140 of 400 turns with a wrong held list: one wrong change at turn 191 stayed for 140 turns, and those turns used 29.5 % of the chat's computed tokens and 36.6 % of its seconds. In test 22 the kept chat ended 171 of 457 turns with a wrong held state, from turn 287 to the end of its run, and the distance from the true state grew to 17 cells. With AXI no turn of the 1,900 ended wrong, and a step stayed in a band of cost: 1,102 to 1,489 computed tokens a step through test 23, 2,422 to 2,981 prompt tokens a step through test 22. This is the part of the work that a kept chat makes heavier with the length of the job (the prompt it sends, the time to the first token and, once a wrong change is in, the state it works on) and that AXI keeps in a band; it is not the training bill and not the price of a token.

## What is counted

* A turn *ends wrong* when the held state is not the true state after it (the saved `exact` flag is false, the distance is above 0 cells). The kept chat applies each reply as it comes (blind application); AXI holds the state on the host, checks each change and re-asks up to twice.
* Computed tokens = prompt tokens not served from the cache + visible + thinking tokens. Seconds = the wall time of the call. The API gives no joules for the hardware, so the computed tokens and the seconds are the stand-in for energy, not joules.
* Test 22 asked the kept chat for up to 700 turns; it stopped at turn 457 because its next prompt would have passed 950,000 tokens. The AXI arm is shown for the same 457 steps and for all 1,500. The first turn of the kept chat in test 22 was read from the disk, not called; as in the saved summary it is left out of the tokens and the seconds.

## 1. How often the state was wrong

| Arm | Turns | Replies exact | Turns ending with a wrong held state | Share of the turns | First such turn | Longest stretch (turns) | Cells off, added over the turns | Largest distance (cells) |
|---|---|---|---|---|---|---|---|---|
| Test 23, kept chat | 400 | 398 of 400 | 140 | 35.0 % | 191 | 140 | 146 | 2 |
| Test 23, with AXI | 400 | 400 of 400 | 0 | 0.0 % | - | 0 | 0 | 0 |
| Test 22, kept chat | 457 | 442 of 457 | 171 | 37.4 % | 287 | 171 | 2,024 | 17 |
| Test 22, with AXI, first 457 steps | 457 | 453 of 457 | 0 | 0.0 % | - | 0 | 0 | 0 |
| Test 22, with AXI, all 1,500 steps | 1,500 | 1,482 of 1,500 | 0 | 0.0 % | - | 0 | 0 | 0 |

Both arms were exact at the start: the kept chat's first wrong turn is 191 in test 23 and 287 in test 22. What the table adds is what happens after a wrong change: in the kept chat it stayed (140 turns in test 23; to the end of the run in test 22, where the distance grew), and with AXI there was no such stretch in 1,900 turns.

## 2. What the turns that ended wrong used

| Arm | Computed tokens, whole arm | on turns ending wrong | Share | Seconds, whole arm | on turns ending wrong | Share |
|---|---|---|---|---|---|---|
| Test 23, kept chat | 1,207,926 | 356,199 | 29.5 % | 1,457 | 533 | 36.6 % |
| Test 23, with AXI | 528,836 | 0 | 0.0 % | 1,103 | 0 | 0.0 % |
| Test 22, kept chat | 2,085,068 | 416,029 | 20.0 % | 3,253 | 1,997 | 61.4 % |
| Test 22, with AXI, first 457 steps | 1,369,436 | 0 | 0.0 % | 1,274 | 0 | 0.0 % |
| Test 22, with AXI, all 1,500 steps | 4,606,053 | 0 | 0.0 % | 4,612 | 0 | 0.0 % |

This is an accounting of where the compute and the seconds went, not a claim that every one of those tokens was wasted: on the turns that ended wrong, 139 of 140 replies were still exact in test 23 and 157 of 171 in test 22. What was wrong was the state the next turn started from.

## 3. What a step costs through the job

Test 23 (the text is ambiguous; 400 turns):

| Turns | Chat: prompt tokens a step | Chat: computed a step | Chat: first token (s, median) | Chat: share of prompt from the cache | AXI: prompt tokens a step | AXI: computed a step | AXI: first token (s, median) |
|---|---|---|---|---|---|---|---|
| 1 to 50 | 13,094 | 6,338 | 1.86 | 51.8 % | 959 | 1,102 | 2.37 |
| 51 to 100 | 37,801 | 2,543 | 2.14 | 93.4 % | 987 | 1,125 | 2.29 |
| 101 to 150 | 62,690 | 2,542 | 2.29 | 96.0 % | 1,041 | 1,190 | 2.59 |
| 151 to 200 | 87,497 | 2,546 | 2.59 | 97.1 % | 1,130 | 1,293 | 2.78 |
| 201 to 250 | 112,162 | 2,543 | 2.83 | 97.8 % | 1,289 | 1,435 | 2.47 |
| 251 to 300 | 136,951 | 2,546 | 3.85 | 98.2 % | 1,333 | 1,479 | 2.33 |
| 301 to 350 | 161,896 | 2,564 | 5.06 | 98.4 % | 1,340 | 1,489 | 2.35 |
| 351 to 400 | 186,378 | 2,537 | 6.12 | 98.7 % | 1,334 | 1,462 | 2.37 |

The kept chat's median first token was lower than AXI's in the windows up to turn 200 and higher from turns 201 to 250 on, when its prompt was 112,162 tokens a step. AXI's computed tokens a step moved between 1,102 and 1,489 as its short state grew; its median first token stayed between 2.29 and 2.78 s. At a short prompt the kept chat was the faster one (1.86 s against 2.37 s in the first window); AXI's replies also carried more thinking tokens (112 a step against 3). The gain is not at the start; it is that AXI stays where it was while the chat does not.

Test 22 (a dependency graph with heavy turns; the first 457 steps of each arm):

| Turns | Chat: prompt tokens a step | Chat: computed a step | Chat: first token (s, median) | Chat: held state exact | Chat: mean distance (cells) | AXI: prompt tokens a step | AXI: computed a step | AXI: first token (s, median) | AXI: held state exact |
|---|---|---|---|---|---|---|---|---|---|
| 1 to 50 | 54,308 | 3,453 | 2.41 | 49 of 49 | 0.00 | 2,422 | 2,767 | 2.58 | 50 of 50 |
| 51 to 150 | 208,875 | 2,177 | 2.48 | 100 of 100 | 0.00 | 2,537 | 2,709 | 2.14 | 100 of 100 |
| 151 to 250 | 416,510 | 6,271 | 3.58 | 100 of 100 | 0.00 | 2,845 | 3,052 | 2.15 | 100 of 100 |
| 251 to 350 | 624,292 | 8,203 | 6.89 | 36 of 100 | 3.94 | 2,915 | 3,126 | 2.15 | 100 of 100 |
| 351 to 457 | 839,327 | 2,344 | 11.41 | 0 of 107 | 15.23 | 2,981 | 3,201 | 2.21 | 107 of 107 |

In test 22 the first window was close (2.41 s for the kept chat, 2.58 s for AXI) and the kept chat's median first token was above AXI's from the window 51 to 150 on. AXI's median first token stayed between 2.14 and 2.58 s through the 457 steps; the kept chat's rose with its prompt.

## 4. Sensitivity to the cache (assumed, not measured)

The kept chat's prompt was served from the cache for 97.0 % of its tokens in test 23, and a cache is easiest to use for a chat that only appends. A real service may do worse (a changing head of the prompt, a request routed to a worker without the prefix) or better. The table re-counts the same job at other hit rates; only the last row was observed, the others are assumptions and say nothing about any service. AXI's prompt is not cached in the measured run.

| Cache hit rate of the kept chat | Kept chat: computed tokens per exact reply | AXI: computed tokens per exact reply | Kept chat over AXI (times) | Status |
|---|---|---|---|---|
| 0.0 % | 100,344 | 1,322 | 75.9 | assumed |
| 50.0 % | 50,189 | 1,322 | 38.0 | assumed |
| 75.0 % | 25,111 | 1,322 | 19.0 | assumed |
| 90.0 % | 10,065 | 1,322 | 7.6 | assumed |
| 95.0 % | 5,049 | 1,322 | 3.8 | assumed |
| 97.0 % | 3,035 | 1,322 | 2.3 | observed |

At the observed rate the kept chat computed 2.30 times the tokens per exact reply of AXI. In test 23 the computed tokens of the kept chat stayed near 2,543 a step from turn 51 on because the cache carried the history; what grew for the kept chat was the prompt it sent and the time to the first token. In test 22 the cache did not serve every window alike: the computed tokens a step of the kept chat were between 2,177 and 8,203 by window (table above); the file does not say why.

The same job with weights instead of a price list. The weights are assumptions for a sensitivity: an uncached prompt token weighs 1, a cached prompt token and an output token (visible plus thinking) weigh what the first two columns say.

| Weight of a cached prompt token | Weight of an output token | Kept chat: units per exact reply | AXI: units per exact reply | Kept chat over AXI (times) |
|---|---|---|---|---|
| 0.25 | 1 | 27,362 | 1,322 | 20.7 |
| 0.25 | 4 | 27,464 | 1,758 | 15.6 |
| 0.25 | 8 | 27,599 | 2,340 | 11.8 |
| 0.10 | 1 | 12,766 | 1,322 | 9.7 |
| 0.10 | 4 | 12,867 | 1,758 | 7.3 |
| 0.10 | 8 | 13,003 | 2,340 | 5.6 |

## 5. In company terms

* **A step stays in a band of cost.** With AXI the computed tokens a step stayed between 1,102 and 1,489 (test 23) and the prompt between 2,422 and 2,981 tokens (test 22), over 400 and 1,500 steps. A company can budget a step from the first one.
* **No turn ended on a wrong state with AXI** in 1,900 turns, against 311 of 857 for the kept chat; the check and the undo belong to the host.
* **The time to the first token stayed in a band** (2.14 to 2.58 s in test 22, 2.29 to 2.78 s in test 23), which is what a service level can be written against; the kept chat's rose with its prompt.
* **Where the kept chat is as good or better:** both arms were exact at the start, and at a short prompt the kept chat answered sooner in test 23 (1.86 s against 2.37 s in the first window). The case for AXI is the long job, not the first fifty turns.

## 6. What this page does not show

* The cost of a wrong action in a real business (a wrong record, a failed deployment, a person who has to look). It is likely larger than the tokens, and it is not measured here.
* Joules, or the memory a session takes on a GPU, or how many sessions fit on one: every call went to a hosted model. A shorter prompt means a smaller cache for the same model, but that is arithmetic here, not a measurement.
* The cache hit rates of any real service (section 4 is a sensitivity), or a price list (section 4 uses weights).
* Another model, another seed, another job, or a real user's words: one run of each arm, one seed in each test, synthetic streams.

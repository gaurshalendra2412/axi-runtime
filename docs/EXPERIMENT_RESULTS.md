# Experiment results (real runs, Colab T4, no training)

Setup: synthetic service graphs; 60 tasks (half are "delete a service that still has dependents"); greedy decoding;
success = final graph exactly equals the expected graph. Prompt describes syntax only unless noted.
Qwen/Qwen2.5-Instruct models, fp16. Scripts: `python/colab/colab_llm_experiment.py`.

## Qwen2.5-0.5B-Instruct (too weak to test the idea)
| | all | delete_dep |
|---|---|---|
| unconstrained / constrained success | 26.7% / 26.7% | 0% / 0% |
| corrupted graph (blind apply) | 3.3% | 0% |
| strict parse | 96.7% -> 100% (constrained) | 93.3% -> 100% |
The model never emitted a plain service delete (it wrote self-loop edge deletes), so the gate/repair had nothing to fix.

## Qwen2.5-3B-Instruct — two seeds, both regimes

Seed 0 was the development seed: the imperative retry wording and `normalize()` were written after reading its failures.
**Seed 1 was run afterwards on fresh tasks and is the confirmation.** Because seed-1 failures have now also been read, any
change made from them needs a seed-2 run before it counts. Cells are `seed 0 / seed 1`.

### Regime A: no rule in the prompt (syntax only)
| mode | success (all, n=60) | corrupted (all) | success (delete_dep, n=30) |
|---|---|---|---|
| M0 unconstrained, lenient parse, blind apply | 50.0 / 50.0 % | 50.0 / 41.7 % | 0.0 / 0.0 % |
| M1 constrained, blind apply | 50.0 / 50.0 % | 50.0 / 50.0 % | 0.0 / 0.0 % (corrupted 100 / 100 %) |
| M2 gate only | 50.0 / 50.0 % | 0 / 0 % | 0.0 / 0.0 % |
| M3 retry, vague prose | 58.3 / 60.0 % | 0 / 0 % | 16.7 / 20.0 % |
| M3 retry, detailed prose | 96.7 / 93.3 % | 0 / 0 % | 93.3 / 86.7 % |
| M3 retry, structured (v1) | 68.3 / 73.3 % | 0 / 0 % | 36.7 / 46.7 % |
| **M3 retry, structured imperative** | **100 / 100 %** | 0 / 0 % | **100 / 100 %** |
| M5 deterministic repair (no 2nd LLM call) | 100 / 100 % | 0 / 0 % | 100 / 100 % |

Strict parse of the *unconstrained* output: 100 / 91.7 % overall, and 100 / 83.3 % on delete_dep (seed 1). The grammar mask removes
those parse failures (-> 100 %), and every delete_dep proposal it rescues is a corrupting one: M0 corrupted 83.3 %, M1 100 % (seed 1).
Retry-wording order is the same on both seeds: vague prose < structured v1 < detailed prose < structured imperative.
Combined, imperative retry recovered 60/60 delete_dep tasks over the two seeds.

### Regime B: rule stated in the prompt (`--hint-rules`)
| mode | success (all) | corrupted (all) | success (delete_dep) |
|---|---|---|---|
| M0 unconstrained | 55.0 / 56.7 % | 40.0 / 38.3 % | 13.3 / 16.7 % |
| M1 constrained | 55.0 / 55.0 % | 38.3 / 41.7 % | 13.3 / 16.7 % |
| M2 gate only | 45.0 / 46.7 % | 0 / 0 % | 6.7 / 10.0 % |
| M3 retry, vague prose | 53.3 / 50.0 % | 0 / 0 % | 16.7 / 16.7 % |
| M3 retry, detailed prose | 60.0 / 60.0 % | 0 / 0 % | 36.7 / 36.7 % |
| M3 retry, structured (v1) | 55.0 / 58.3 % | 0 / 0 % | 23.3 / 33.3 % |
| M3 retry, structured imperative | 60.0 / 56.7 % | 0 / 0 % | 33.3 / 30.0 % |
| M5 deterministic repair | 65.0 / 66.7 % | 0 / 0 % | 46.7 / 50.0 % |

Putting the rule in the prompt did not rescue the 3B model on either seed (imperative retry 100 -> 33.3 / 30.0 % on delete_dep).
Strict parse of the constrained output: 98.3 / 100 % (seed 0 / seed 1). The seed-0 shortfall is unexplained (seed-0 transcripts
were not kept); on seed 1 every constrained output parsed in both regimes.

### What the rule-in-prompt transcripts show (seed 1, delete_dep)
With the rule the model usually *does* try to delete the incident edges, but a listed edge often does not exist in the graph
(wrong relation, direction or weight), e.g. `DEL[1,3] DEL[(0,0)->(1,3):dep#1] DEL[(0,0)->(1,3):owns#9] DEL[(1,2)->(2,3):dep#8]
DEL[(1,2)->(2,3):owns#1]` -> obstruction `match_edge`. One phantom edge makes the whole delta inadmissible. Other proposals were
`ADD[0,0:9] DEL[0,0] ...` (`ident_node`). So the failure looks like *recall errors about the graph*, not (shown) attention
entropy. In the no-rule regime the gate hands the model the exact edge list to copy, which is what the imperative retry does.
**Gap:** feedback and M5 repair currently handle only `dangling`; `match_edge` and `ident_node` are reported but not repaired
(`propose_repair(...).repairable == False`, checked), which is why M5 only reaches 50 % on delete_dep in this regime.

### Gate rejected a proposal that blind apply got right (hint regime: 6 on seed 0, 5 on seed 1)
Reproduced on toy graphs (`blind_apply` silently ignores deletes of edges that do not exist, the gate does not):
* 4 of the 5 seed-1 cases contain a **phantom edge delete** (`match_edge`) on a graph where it is a no-op, e.g. `DEL[1,3] DEL[(2,3)->(1,3):dep#8]`;
* 1 is `ADD[0,0:9] DEL[0,0] DEL[(0,0)->(1,1):dep#6]` (`ident_node`: add then delete the same node = net delete).
This is the DPO match/identification conditions being stricter than the task outcome needs. It is a policy choice, not a bug:
the alternatives are "deleting something absent is a no-op" (idempotent delete) or dropping the phantom op in a repair step.
It cost the gate-only mode 8.3 points of success (55.0 -> 46.7 %) against a drop in corruption from 41.7 % to 0 %.

## KV-cache timing (Tesla T4, Qwen2.5-0.5B, no logits computed)
| context | full prefill | append 64 spec tokens | rollback = crop | edit at 25%, re-prefill rest |
|---|---|---|---|---|
| 4,096 | 683 ms | 43 ms | 813 us | 286 ms |
| 16,384 | 3,170 ms | 78 ms | 349 us | 2,507 ms |
| 32,768 | 10,076 ms | 137 ms | 668 us | 8,937 ms |

## Caveats
One model family, greedy decoding, synthetic graphs, 60 tasks; the trap share (50%) is a design choice, so the headline rate is
not a real-world rate. "Success" encodes the policy that deleting a service also deletes its edges, which the auto-repair follows.

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

## Qwen2.5-3B-Instruct — three seeds, both regimes

Seed 0 was the development seed (the imperative retry wording and `normalize()` were written after reading its failures).
**Seeds 1 and 2 were run afterwards on fresh tasks, with unchanged code.** Cells are `seed 0 / seed 1 / seed 2`. 60 tasks per seed,
30 of them delete-with-dependents (`delete_dep`).

### Regime A: no rule in the prompt (syntax only)
| mode | success (all) | success (delete_dep) | corrupted (delete_dep) |
|---|---|---|---|
| M0 unconstrained, lenient parse, blind apply | 50.0 / 50.0 / 50.0 % | 0 / 0 / 0 % | 100 / 83.3 / 96.7 % |
| M1 constrained, blind apply | 50.0 / 50.0 / 50.0 % | 0 / 0 / 0 % | 100 / 100 / 100 % |
| M2 gate only | 50.0 / 50.0 / 50.0 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| M3 retry, vague prose | 58.3 / 60.0 / 53.3 % | 16.7 / 20.0 / 6.7 % | 0 |
| M3 retry, detailed prose | 96.7 / 93.3 / 96.7 % | 93.3 / 86.7 / 93.3 % | 0 |
| M3 retry, structured (v1) | 68.3 / 73.3 / 66.7 % | 36.7 / 46.7 / 33.3 % | 0 |
| **M3 retry, structured imperative** | **100 / 100 / 100 %** | **100 / 100 / 100 %** | 0 |
| M5 deterministic repair (no 2nd LLM call) | 100 / 100 / 100 % | 100 / 100 / 100 % | 0 |

* Imperative retry: **180/180 tasks, 90/90 delete_dep** over three seeds. With 90/90 the true rate is plausibly anywhere above ~96% (Wilson 95%).
* The ordering of retry wordings is the same on every seed: vague prose < structured v1 < detailed prose < structured imperative.
* Strict parse of the *unconstrained* output (overall): 100 / 91.7 / 98.3 %; the mask makes it 100% every time. On delete_dep the
  unconstrained corruption rate is 100 / 83.3 / 96.7 % and the constrained one 100% every time: the mask removes the loud failures
  (unparseable output) and what is left corrupts the graph.
* Gate-rejected-but-blind-right: 0 on every seed in this regime.
* Obstruction mix of the 30 delete_dep proposals (seed 2): **30/30 are pure `dangling`** (the model forgets the incident edges and nothing else).

### Regime B: rule stated in the prompt (`--hint-rules`)
| mode | success (all) | success (delete_dep) | corrupted (delete_dep) |
|---|---|---|---|
| M0 unconstrained | 55.0 / 56.7 / 58.3 % | 13.3 / 16.7 / 20.0 % | 80.0 / 76.7 / 76.7 % |
| M1 constrained | 55.0 / 55.0 / 55.0 % | 13.3 / 16.7 / 13.3 % | 76.7 / 76.7 / 76.7 % |
| M2 gate only | 45.0 / 46.7 / 46.7 % | 6.7 / 10.0 / 6.7 % | 0 |
| M3 retry, vague prose | 53.3 / 50.0 / 53.3 % | 16.7 / 16.7 / 13.3 % | 0 |
| M3 retry, detailed prose | 60.0 / 60.0 / 63.3 % | 36.7 / 36.7 / 40.0 % | 0 |
| M3 retry, structured (v1) | 55.0 / 58.3 / 56.7 % | 23.3 / 33.3 / 26.7 % | 0 |
| M3 retry, structured imperative | 60.0 / 56.7 / 66.7 % | 33.3 / 30.0 / 46.7 % | 0 |
| M5 deterministic repair | 65.0 / 66.7 / 58.3 % | 46.7 / 50.0 / 30.0 % | 0 |

* The collapse is real and repeats: imperative retry on delete_dep goes from 100% (no rule) to 33.3 / 30.0 / 46.7 %. The size of the
  drop varies by seed, so report "roughly 30-47%", not one number. M5 is noisy too (30-50%).
* Strict parse of the *constrained* output: 98.3 / 100 / 96.7 % overall (96.7 / 100 / 93.3 % on delete_dep), so the mask does not
  always guarantee a parsed delta here. Cause not yet established (hypothesis: output cut off at `max_new_tokens=120` when the
  model lists many edges; check `tokens` on the failing rows). Unconstrained strict parse falls to 88.3 / 88.3 / 78.3 % (56.7% on delete_dep, seed 2).
* Gate-rejected-but-blind-right: 6 / 5 / 5 overall, 2 / 2 / 2 on delete_dep.

### Why the rule in the prompt hurts: the failure mode changes (seed 2, delete_dep, n=30)
| gate obstructions on the model's proposal | count |
|---|---|
| none (proposal admissible) | 4 |
| `dangling` only (forgot the incident edges) | 7 |
| `dangling` + other kinds | 15 |
| no `dangling`, other kinds only | 4 |

Of the 30 proposals: 9 contain a phantom reference (`match_edge`, sometimes `match_weight`: deleting an edge that does not exist or
has another weight), 8 contain `ident_node` (e.g. `ADD[r,c:v] DEL[r,c]`), 8 contain `malformed` (an `ADD` edge touching the node being
deleted: the model *adds* the listed edges instead of deleting them), 4 contain `ident_edge`.
So without the rule 100% of failures are the single kind the repair and the imperative wording handle; with the rule only 23% are.
This is why imperative retry and M5 stall at 30-50%: the current feedback text and `propose_repair` only handle `dangling`; the other
kinds are reported but not repaired (`propose_repair(...).repairable == False`, checked).

### Gate stricter than the outcome needs (hint regime: 6 / 5 / 5 per seed)
Reproduced on toy graphs: `blind_apply` silently ignores deletes of edges that do not exist, the gate does not. In the seed-1
examples 4 of 5 contain a phantom edge delete (a no-op), and 1 is `ADD[0,0:9] DEL[0,0] ...` (add-then-delete of the same node = net delete).
This is the DPO match/identification conditions being stricter than the task outcome: a policy choice, not a bug (alternatives:
idempotent delete, or drop the phantom op in a repair step). It cost the gate-only mode about 8 points of success for a drop in
corruption from ~40% to 0%.

### Expressiveness gap found while reading the concept texts
The delta language cannot express an in-place update: `ADD` of an existing node -> `ident_node`; `DEL`+`ADD` of the same key ->
`ident_node` / `ident_edge`. The benchmark has no update tasks.

## KV-cache timing (Tesla T4, Qwen2.5-0.5B, no logits computed)
| context | full prefill | append 64 spec tokens | rollback = crop | edit at 25%, re-prefill rest |
|---|---|---|---|---|
| 4,096 | 683 ms | 43 ms | 813 us | 286 ms |
| 16,384 | 3,170 ms | 78 ms | 349 us | 2,507 ms |
| 32,768 | 10,076 ms | 137 ms | 668 us | 8,937 ms |

## Caveats
One model family, greedy decoding, synthetic graphs, 60 tasks; the trap share (50%) is a design choice, so the headline rate is
not a real-world rate. "Success" encodes the policy that deleting a service also deletes its edges, which the auto-repair follows.

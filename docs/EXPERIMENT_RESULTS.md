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

## Qwen2.5-3B-Instruct, no rule in prompt
| mode | success (all) | corrupted | success (delete_dep) |
|---|---|---|---|
| M0 unconstrained | 50.0% | 50.0% | 0.0% (100% corrupted) |
| M1 constrained | 50.0% | 50.0% | 0.0% (100% corrupted) |
| M2 gate | 50.0% | 0.0% | 0.0% |
| M3 retry, vague prose | 58.3% | 0.0% | 16.7% |
| M3 retry, detailed prose | 96.7% | 0.0% | 93.3% |
| M3 retry, structured (v1) | 68.3% | 0.0% | 36.7% |
| M5 deterministic repair | 100.0% | 0.0% | 100.0% |
Strict parse was already 100% unconstrained. Structured-v1 failures: the model rewrote the listed `DEL[...]` edges as `ADD[...]` with swapped direction.

## Qwen2.5-3B-Instruct, rule stated in the prompt (--hint-rules)
| mode | success (all) | corrupted | success (delete_dep) |
|---|---|---|---|
| M0 unconstrained | 55.0% | 40.0% | 13.3% (80% corrupted), strict parse 76.7% |
| M1 constrained | 55.0% | 38.3% | 13.3% (76.7% corrupted) |
| M2 gate | 45.0% | 0.0% | 6.7% |
| M3 retry, detailed prose | 60.0% | 0.0% | 36.7% |
| M5 deterministic repair (v1) | 56.7% | 0.0% | 30.0% |
Putting the rule in the prompt did NOT fix this 3B model (it often emitted `ADD[0,1:9] DEL[0,1]`). The redundant ADD blocked v1 auto-repair; `normalize()` in `diagnostics.py` was added for exactly this and needs a re-run.
The gate also blocked some proposals blind-apply happened to get right (M2 45% < M1 55%): the identification rule is stricter than necessary for redundant items.

## KV-cache timing (Tesla T4, Qwen2.5-0.5B, no logits computed)
| context | full prefill | append 64 spec tokens | rollback = crop | edit at 25%, re-prefill rest |
|---|---|---|---|---|
| 4,096 | 683 ms | 43 ms | 813 us | 286 ms |
| 16,384 | 3,170 ms | 78 ms | 349 us | 2,507 ms |
| 32,768 | 10,076 ms | 137 ms | 668 us | 8,937 ms |

## Caveats
One model family, greedy decoding, synthetic graphs, 60 tasks; the trap share (50%) is a design choice, so the headline rate is
not a real-world rate. "Success" encodes the policy that deleting a service also deletes its edges, which the auto-repair follows.

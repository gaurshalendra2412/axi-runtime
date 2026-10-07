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
* Strict parse of the *constrained* output: 98.3 / 100 / 96.7 % overall (96.7 / 100 / 93.3 % on delete_dep), so the mask did not
  always give a parsed delta here. **Cause found (seed 2): truncation.** Both failing outputs have `tokens 120` and stop mid-item
  (`... DEL[(0,0)->(2,1):dep#`). The model lists many edges and the 120-token cap cuts the output; the mask only guarantees that every
  *prefix* is valid, not that the output is finished. Seed 0's single failing row was not inspected. The cap is now 400
  (`colab_llm_experiment.py`); seeds 0-2 in this document all ran with 120, so the hint-regime numbers (about 2 of 60 outputs per seed) are
  to be re-run with 400 before the paper quotes them. Unconstrained strict parse falls to 88.3 / 88.3 / 78.3 % (56.7% on delete_dep, seed 2).
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

### Reproducibility (seed 2 run twice)
Seed 2 was run a second time on a different Colab notebook (the first hit its GPU limit; fresh runtime, model re-downloaded). Every
number printed (both regimes, all modes, all kinds, strict-parse counts, the two truncated outputs) is identical to the first run.
This shows greedy decoding on a T4 gives the same result twice; it does **not** replace more seeds or more models.

### Delete Test scorecard on the real logs (seed 2, delete_dep, n=30; `python -m axi.experiments.scorecard`)
The three columns are named after the challenges in the Cinderella Project text; the operational definitions are ours, not Rajnish's
(see `RAJNISH_CONCEPT_MAP.md` row 23).

| | no rule in prompt | rule in prompt |
|---|---|---|
| **rule compliance**: blind apply corrupts the graph (M1) / after the gate | 100% / 0% | 76.7% / 0% |
| **reality perception**: proposals that name things not in the graph | 0% | 32.1% (9 of 28 parsed) |
| proposals that leave dangling edges | 100% | 78.6% (22 of 28 parsed) |
| **silent vs loud**, unconstrained: loud (nothing extractable) / silent (corrupts) | 3.3% / 96.7% | 0% / 76.7% |
| **silent vs loud**, constrained: loud (unparseable) / silent (corrupts) | 0% / 100% | 6.7% / 76.7% |

The 9 and 22 equal the hand count of the obstruction mix above (the denominator is the 28 proposals that parsed).
All kinds together (n=60): no rule, constrained silent 50.0% / loud 0%; rule, constrained silent 38.3% / loud 3.3%, phantom references 22.4%.

After feedback, per retry wording (delete_dep; `retried` = proposals the gate rejected first):

| wording | no rule: retried / repeats same / still rejected / success | rule: retried / repeats same / still rejected / success |
|---|---|---|
| vague prose | 30 / 3.3% / 90.0% / 6.7% | 26 / 0.0% / 92.3% / 13.3% |
| detailed prose | 30 / 0.0% / 6.7% / 93.3% | 26 / 3.8% / 61.5% / 40.0% |
| structured (v1) | 30 / 10.0% / 66.7% / 33.3% | 26 / 7.7% / 76.9% / 26.7% |
| structured imperative | 30 / 0.0% / 0.0% / 100% | 26 / 3.8% / 53.8% / 46.7% |

Reading:
* **Failure is silent.** Without the gate, 76.7-100% of constrained delete_dep proposals corrupt the graph with a perfectly parseable
  delta; the loud share is 0-6.7%. This is the snake-vs-tiger row (16) on real data.
* **A wrong hypothesis, corrected.** I expected vague feedback to make the model *repeat* its rejected proposal. It does not: it
  repeats 0-10% of the time. It changes the answer, and the new answer is still rejected (90.0% / 92.3% for vague prose). The better
  description is *misdirected compliance*: the model reacts to the feedback but has not been told what to change. Whether that is
  what Rajnish means by ego-defense is for him to say; the "repeats the same proposal" reading is not supported.
* **Reality perception is a property of the prompt, not only of the model.** With the rule stated, a third of the proposals refer to
  edges that are not in the graph (0% without it). The gate names these (`match_edge`) but nothing repairs them yet.

### Gate stricter than the outcome needs (hint regime: 6 / 5 / 5 per seed)
Reproduced on toy graphs: `blind_apply` silently ignores deletes of edges that do not exist, the gate does not. In the seed-1
examples 4 of 5 contain a phantom edge delete (a no-op), and 1 is `ADD[0,0:9] DEL[0,0] ...` (add-then-delete of the same node = net delete).
This is the DPO match/identification conditions being stricter than the task outcome: a policy choice, not a bug (alternatives:
idempotent delete, or drop the phantom op in a repair step). It cost the gate-only mode about 8 points of success for a drop in
corruption from ~40% to 0%.

### Expressiveness gap found while reading the concept texts
The delta language cannot express an in-place update: `ADD` of an existing node -> `ident_node`; `DEL`+`ADD` of the same key ->
`ident_node` / `ident_edge`. The benchmark has no update tasks.

## Built, not yet run on a real model
* **M6 ground-and-complete** (mode `M6_ground_complete`): cuts operations that name things not in the graph, corrects a wrong weight, then deletes every
  remaining incident edge. Sandbox only (synthetic fuzz); `python/tests/test_ground_complete.py`. Needs `--seed 3` or later.
* **Containment test** (`--cascade-episodes 10 --cascade-steps 8`): 8 sequential requests on one live graph under four policies (blind / gate /
  gate+M5 / gate+M6). Scripted stand-in only so far.
Seeds 0-2 numbers above are unchanged and do not include either.

## KV-cache timing (Tesla T4, Qwen2.5-0.5B, no logits computed)
| context | full prefill | append 64 spec tokens | rollback = crop | edit at 25%, re-prefill rest |
|---|---|---|---|---|
| 4,096 | 683 ms | 43 ms | 813 us | 286 ms |
| 16,384 | 3,170 ms | 78 ms | 349 us | 2,507 ms |
| 32,768 | 10,076 ms | 137 ms | 668 us | 8,937 ms |

## Caveats
One model family, greedy decoding, synthetic graphs, 60 tasks; the trap share (50%) is a design choice, so the headline rate is
not a real-world rate. "Success" encodes the policy that deleting a service also deletes its edges, which the auto-repair follows.

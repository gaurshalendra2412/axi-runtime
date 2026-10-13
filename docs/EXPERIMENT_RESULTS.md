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

## Qwen2.5-3B-Instruct — seeds 0-2 (tables below) and seed 3 (own section after them), both regimes

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

### Reproducibility (seed 3, rule regime, run twice)
The seed-3 rule-in-prompt run was repeated in a fresh Colab session (7 Oct): every printed number in the summary tables was identical to the first run
(all nine modes, both panels, the "blind right but gate rejected" counts 8 and 3). Same model, greedy decoding, T4. One seed, one model.

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

## Seed 3: the first fresh seed for M6 and the containment test (Colab T4, Qwen2.5-3B, `max_new_tokens=400`)
M6 (ground-and-complete) and the containment test were written after seed 2 was read, so seed 3 is the first run that counts for them.
M0-M5 ran on unchanged code (the 400-token cap is the only change since seeds 0-2). 60 tasks, 30 `delete_dep`.

| mode | no rule: success (all) | no rule: delete_dep | rule in prompt: success (all) | rule in prompt: delete_dep |
|---|---|---|---|---|
| M0 unconstrained | 50.0 % | 0.0 % | 53.3 % | 10.0 % |
| M1 constrained | 50.0 % | 0.0 % | 55.0 % | 13.3 % |
| M2 gate only | 50.0 % | 0.0 % | 41.7 % | 3.3 % |
| M3 retry, vague prose | 51.7 % | 3.3 % | 51.7 % | 16.7 % |
| M3 retry, detailed prose | 98.3 % | 96.7 % | 53.3 % | 26.7 % |
| M3 retry, structured (v1) | 65.0 % | 30.0 % | 46.7 % | 13.3 % |
| M3 retry, structured imperative | 100 % | 100 % | 48.3 % | 16.7 % |
| M5 deterministic repair | 100 % | 100 % | 51.7 % | 23.3 % |
| **M6 ground-and-complete** | **100 %** | **100 %** | **95.0 %** | **90.0 %** |

Corruption: blind apply (M1) corrupts 100 % of delete_dep without the rule and 76.7 % with it; every gated mode, including M6, is 0 %.
* **Four seeds now agree** on the no-rule ordering of the retry wordings (delete_dep, s0/s1/s2/s3): vague 16.7/20.0/6.7/3.3 %, structured v1 36.7/46.7/33.3/30.0 %,
  detailed prose 93.3/86.7/93.3/96.7 %, imperative **100/100/100/100 %** (120 of 120 delete_dep tasks, 240 of 240 overall).
* With the rule in the prompt, imperative retry on delete_dep is 33.3/30.0/46.7/16.7 % and M5 is 46.7/50.0/30.0/23.3 %: report a range, not a number.
* **M6 in the rule regime: 27 of 30 correct (26 repaired, plus one admissible proposal applied untouched; the count of 1 is inferred).** The 3 left are misses; they were left alone on purpose and
  the graph stayed intact (0 % corruption).
  **Correction (7 Oct, after reading one raw output).** I first wrote that all 3 are `ident_node` updates the language cannot express. That was not supported.
  On the re-run, the filter "M6 gave a reason it could not repair" printed **one** output, not three, so at most one of the three is that case and the other
  two are still unread. The one that was read: request "Delete the service at (1,2)." (node value 5, three incident edges); model text
  `ADD[1,2:0] DEL[(0,0)->(1,2):dep#2] DEL[(1,2)->(0,0):dep#7] DEL[(1,2)->(1,0):owns#2]`. The request was a plain delete, not an update. The model wrote
  `ADD[1,2:0]` where `DEL[1,2]` was needed (it looks like a "set it to 0" improvisation) and named the first edge `dep` instead of `owns`. M6 refuses to guess
  around that. It is a model error, not a gap in the language. The benchmark has no update tasks, so no failure in it can be an update the user wanted.
* M6 recovers the gate-rejected-but-blind-right cases (8 overall, 3 on delete_dep): the harmless phantom edge deletes are cut instead of rejected.
* **How to read M6.** The model only has to name the right service. The runtime cuts the false claims and writes the missing edge deletes. So the
  headline is *safety plus completion by the runtime* (0 % corruption, 90 % of the intended edits), not "the model reached 90 %". The M3 retry numbers are the
  ones that involve a second model call, and in the rule regime they stay at 13-27 %.
* **Constrained strict-parse: 100 % in both regimes, 0 of 60 unparsed with the rule in the prompt** (seeds 0-2 at the 120-token cap: 98.3 / 100 / 96.7 %).
  Consistent with the truncation explanation; the clean test (re-run seed 2 with the rule and the 400 cap, the two failing outputs should now parse) has not been done.
  Unconstrained strict-parse with the rule: 76.7 % overall, 53.3 % on delete_dep (seed 2: 78.3 / 56.7 %).

Scorecard, seed 3, delete_dep (same definitions as above):

| | no rule | rule in prompt |
|---|---|---|
| blind apply corrupts / after the gate | 100 % / 0 % | 76.7 % / 0 % |
| proposals naming things not in the graph | 0 % | 36.7 % (seed 2: 32.1 %) |
| proposals leaving dangling edges | 100 % | 76.7 % (seed 2: 78.6 %) |
| constrained: loud / silent | 0 % / 100 % | 0 % / 76.7 % |
| after feedback, vague prose: repeats same / still rejected | 6.7 % / 86.7 % | 3.7 % / 85.2 % |
| after feedback, imperative: repeats same / still rejected | 0 % / 0 % | 0 % / 85.2 % |

The "models change their answer but stay rejected" finding (misdirected compliance) repeats on a second seed: repeats 0-15 %, still rejected 70-89 % in the rule regime.

### Containment test (8 episodes x 8 steps per policy, no rule in the prompt, seed 3; concept-map row 27)
| policy | share of episodes with a corrupted graph after step 1 / 2 / 3 / 4 ... 8 | mean dangling edges at step 8 |
|---|---|---|
| blind apply | 38 % / 75 % / 88 % / 100 % ... 100 % | 7.75 |
| gate only | 0 % at every step | 0 |
| gate + M5 repair | 0 % at every step | 0 |
| gate + M6 repair | 0 % at every step | 0 |

Read with care: (1) with blind apply, corruption persists **by construction** (nothing ever removes a dangling edge), so the informative part is how fast it
appears (38 % of episodes at step 1, all by step 4) and how much of it piles up. (2) 8 episodes = 64 steps per policy; percentages move in steps of 12.5 points.
(3) The step-success column the script prints (blind 45.3 %, gate 35.9 %, repair 100 %) is **not comparable across policies**: each policy leaves a different graph,
so the later requests differ. Quote the corruption rows only. (4) One model, synthetic graphs.

## Qwen2.5-7B (4-bit), seed 3: the same 60 tasks, both regimes (Colab T4, run 8 Oct 2026)
Prediction and reading rule were written before the run (`SCALE_TEST_7B.md`, which also has the full reading and the limits). Raw outputs: `docs/results/axi_Qwen2.5-7B-Instruct_seed3_{norule,rule}.json`.
3B is fp16, 7B is NF4 4-bit (the 7B does not fit the T4 in fp16), so size and quantization are mixed in every comparison below.

**No rule in the prompt** (success; 'corrupted' is the share of delete_dep tasks that ended with a broken graph)

| mode | 3B all | 3B delete_dep | 7B all | 7B delete_dep | 7B corrupted (delete_dep) |
|---|---|---|---|---|---|
| M0 unconstrained | 50.0 % | 0.0 % | 50.0 % | 0.0 % | 100.0 % |
| M1 constrained | 50.0 % | 0.0 % | 50.0 % | 0.0 % | 100.0 % |
| M2 gate only | 50.0 % | 0.0 % | 50.0 % | 0.0 % | 0.0 % |
| M3 retry, vague prose | 51.7 % | 3.3 % | 55.0 % | 10.0 % | 0.0 % |
| M3 retry, detailed prose | 98.3 % | 96.7 % | 100.0 % | 100.0 % | 0.0 % |
| M3 retry, structured (v1) | 65.0 % | 30.0 % | 50.0 % | 0.0 % | 0.0 % |
| M3 retry, structured imperative | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 0.0 % |
| M5 deterministic repair | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 0.0 % |
| M6 ground-and-complete | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 0.0 % |

**Rule stated in the prompt** (success; 'corrupted' is the share of delete_dep tasks that ended with a broken graph)

| mode | 3B all | 3B delete_dep | 7B all | 7B delete_dep | 7B corrupted (delete_dep) |
|---|---|---|---|---|---|
| M0 unconstrained | 53.3 % | 10.0 % | 53.3 % | 36.7 % | 63.3 % |
| M1 constrained | 55.0 % | 13.3 % | 53.3 % | 36.7 % | 63.3 % |
| M2 gate only | 41.7 % | 3.3 % | 46.7 % | 33.3 % | 0.0 % |
| M3 retry, vague prose | 51.7 % | 16.7 % | 50.0 % | 40.0 % | 0.0 % |
| M3 retry, detailed prose | 53.3 % | 26.7 % | 56.7 % | 53.3 % | 0.0 % |
| M3 retry, structured (v1) | 46.7 % | 13.3 % | 60.0 % | 60.0 % | 0.0 % |
| M3 retry, structured imperative | 48.3 % | 16.7 % | 60.0 % | 60.0 % | 0.0 % |
| M5 deterministic repair | 51.7 % | 23.3 % | 56.7 % | 53.3 % | 0.0 % |
| M6 ground-and-complete | 95.0 % | 90.0 % | 73.3 % | 76.7 % | 0.0 % |

* **Pre-registered reading:** no rule, `delete_dep`, blind apply (M1): 30 of 30 corrupted (95 % interval 88.6 to 100). By the rule fixed in advance, scale alone did not fix it on this task. Every gated mode: 0 of 60 corrupted in both regimes.
* Structured v1 retry fell from 30 % to 0 % (no rule). Cause: in 30 of 30 retries the 7B returned only the edge deletes and dropped `DEL[r,c]` of the service; the gate admits that and the service stays. The imperative wording ("Keep DEL[r,c]") is 30 of 30.
* Not predicted, rule regime: the 7B uses the rule better on `delete_dep` (blind apply right 36.7 %, 3B 10 to 13 %) but over-applies it on the 12 `delete_leaf` tasks (0 of 12 exact, all 12 carry extra edge deletes; the gate admits 7 of them because the extra deletes name real edges of other services). All 16 M6 misses contain such a delete and none of the 44 M6 successes does. M6 is 73.3 % (3B 95.0 %). The gate checks legality, not whether the change is the one requested.
* Scorecard, 7B, `delete_dep`, no rule / rule: blind apply corrupts 100 % / 63.3 % (3B 100 % / 76.7 %); proposals naming things not in the graph 0 % / 26.7 % (3B 0 % / 36.7 %); proposals leaving dangling edges 100 % / 66.7 % (3B 100 % / 76.7 %).
* Post-hoc, not a result: dropping every edge delete that does not touch a service deleted in the same delta turns the saved 7B outputs into 60 of 60 under M6 (46 deletes dropped, rule regime). Needs a fresh seed and a task kind where an edge delete alone is correct.

## Latency of the modes, from the 7B runs, and the long-horizon test (8 Oct 2026)
* **Measured, no new run** (`LATENCY_7B.md`): on the 120 saved 7B runs the gate check is a few microseconds, the diagnosis a few more, the M5 repair about 15 to 20 and the M6 repair about 35 to 60, against about 1 s (no rule) to 2.5 s (rule) for one constrained model call; check plus diagnosis is about a hundred-thousandth of one call. M2, M5 and M6 add no model call; an M3 retry adds one (2 to 5 times the time on the rejected tasks). Model seconds per correct task: M6 0.97 (no rule) and 3.39 (rule); the best retry wording 2.30 and 6.57. Grammar-constrained decoding gave the same text as unconstrained in 120 of 120 tasks at a time ratio of 0.99 to 1.01. CPU numbers are one machine and vary by about 1.7 times between runs.
* **Long horizon, run 8 Oct** (`LONG_HORIZON_TEST.md`, files `docs/results/axi_long_*`, Qwen2.5-7B 4-bit, Colab T4): 30-step episodes with a fixed request script, drift = distance to an oracle's state. **No rule, 12 episodes:** blind apply was corrupted in 12 of 12 episodes, each at its first delete that has dependents, and ended with 14.1 dangling edges on average; the gate alone never corrupted but its state moved to distance 23.3 (blind 15.0) because it refused all 95 such deletes; gate plus M6 was exactly the intended state on 358 of 360 steps (the two off are one legal add_edge written from the wrong source service). **Rule stated, 8 episodes:** gate and M6 0 corrupted again; blind apply corrupted in 7 of 8 (6 of 8 by step 10, predicted at least 80 %: missed by one episode); M6 exact on only 55 % of steps, all 34 wrong items real edges the model deleted unasked (legal, so the gate admits them), 30 of which later vanished when the oracle deleted a service they belonged to. On the same 8 request scripts M6 was exact on 238 of 240 steps without the rule and 132 with it. **Chat kept (history mode, no rule, 4 episodes):** the held state stayed exact (105 of 105 steps), time per step was 5.1 times higher in the last quarter than the first (state mode 0.89 times; about 7.8 times as much per step near step 19), the raw proposals did not change (0 of 24 delete-with-dependents admitted, and 0 of 95 in state mode), and two of four episodes ran out of GPU memory at about 5,100 and 5,500 tokens. Nine of ten pre-registered rows held; one missed.

## Scope test: keep only the changes the request names (run 8 Oct 2026; `SCOPE_TEST.md`, `tests/test_scope_result.py`)
Qwen2.5-7B 4-bit, Colab T4, greedy, fresh seed 7; prediction and reading rules fixed in the page before the run; files `docs/results/axi_scope_{single,long}_{rule,norule}_Qwen2.5-7B_seed7.json`. All 13 reading-rule lines came out yes and all 6 predictions held.
* **One request at a time (140 tasks per regime, 7 kinds):** rule stated: blind apply 94, gate 87, repair_m6 **118**, scope_m7 **134**, scope_delta 80 (helped 16 tasks, harmed 0; old deletes 24 of 40 against 40 of 40; delete_leaf 5 of 20 against 20 of 20). No rule: blind 99, gate 99, repair_m6 135, scope_m7 135 (identical graphs on 140 of 140), scope_delta 80. The post-hoc baseline scope_delta solved 0 of 60 per regime on lone edge deletes, service-plus-edge deletes and implicit-target deletes. All 11 misses of scope_m7 are `delete_weight` ("remove every edge with weight w", no service named): the filter has nothing to read, and the model deleted a real edge of another weight in 9 of the 11.
* **30 requests in a row (12 episodes rule stated, 8 no rule):** rule stated: repair_m6 exact on 263 of 360 steps (73.1 %, 11 of 12 episodes went off), scope_m7 on 354 (98.3 %, 1 episode, one event: an edge written from the wrong source, which the filter dropped, leaving it missing). No rule: both 237 of 240 (98.8 %), the same one event (a wrong service coordinate). Neither policy was ever corrupted.
* **Not predicted:** with the rule stated the 7B wrote 3,975 tokens against 2,040 without it on the same requests (317 s against 193 s); the filter removes the extra deletes from the state but they are generated first. No rule with plain M6 (135 of 140) did as well as rule plus scope_m7 (134). The filter costs about 11 microseconds per request and adds no model call.
* **Limits:** requests name targets by coordinate (the one thing the filter reads); one model; the two failure modes written down beforehand (an extra edge between two named services; a request naming a service it does not touch) never occurred, so they are untested; 12 and 8 episodes.

## Bigger-model test, Qwen2.5-14B (run 9 Oct 2026; 14 of 15 pre-registered reading-rule lines yes)
Files: `docs/results/axi_14b_{single_rule,single_norule,long_rule,long_norule}_Qwen2.5-14B_seed7.json` (Qwen2.5-14B-Instruct 4-bit, Colab T4, greedy, seed 7; the same 140 tasks and the same episode seeds as the 7B scope test; Part 1 and Part 2 came from two Colab sessions, no file mixes sessions). Prediction and reading rules were fixed before the run in `SCALE_TEST_14B.md`, which has the full Result below them. `tests/test_bigger_model_result.py` recomputes every number below from the saved rows. No output was unparsed (0 of 280 single-step outputs, 0 of 900 recorded policy steps).

* **Verdict by the wording fixed in advance:** all 10 [guard] lines yes; 4 of 5 [need] lines yes. The one no: with no rule, blind apply was right on 11 of 20 delete_dep requests and the line was 10 or fewer (7B: 0 of 20). It missed by one task. Reading: at 14B the raw model needs the framework less than the 7B did, and the framework is a guard with a smaller, measured gain.
* **One request at a time (140 tasks per regime), 14B against 7B:** no rule: blind 111 (99), gate 107 (99), repair_m6 135 (135), scope_m7 136 (135). Rule stated: blind 100 (94), gate 93 (87), repair_m6 117 (118), scope_m7 133 (134). Graphs left corrupted by blind apply 24 (36) and 17 (24); by gate, M6 and scope_m7 0 in both regimes. scope_m7 over M6 with the rule stated: helped 16 tasks, harmed 0 (7B: the same 16). Without the rule: helped 1, harmed 0 (a stray edge delete the filter dropped).
* **30 requests in a row, 14B (7B on the same first episodes):** rule stated, 6 episodes: blind 76 of 180 exact steps (42.2 %), repair_m6 152 (84.4 %; 7B 124, 68.9 %), scope_m7 180 (100 %; 7B 180). No rule, 4 episodes: blind 8 of 120 (6.7 %), repair_m6 and scope_m7 120 of 120 (7B 117 each). Blind apply was corrupted in every one of the 10 episodes; gate-based policies in none.
* **Gain of the framework:** M6 over blind apply (no rule, single) 24 tasks, 17.1 points (7B 36, 25.7); scope_m7 over blind apply (rule stated, single) 33 tasks, 23.6 points (7B 40, 28.6); scope_m7 over M6 over 30 steps (rule stated) 28 of 180 steps, 15.6 points (7B 56, 31.1), shrunk because M6 got better, not because scope_m7 changed.
* **Not changed by size:** the service delete with a separate edge delete (delete_pair) 4 of 20 for the raw model and for the gate in both regimes (M6 and scope_m7 20 of 20); with the rule stated, stray edge deletions on 16 tasks (the 7B 16, 13 of them the same); all 11 misses of scope_m7 are `delete_weight`, where the request names no service (the 7B also 11).
* **Not predicted:** the rule costs the 14B less than the 7B but still costs: 3,510 tokens against 2,685 without it (1.3 times; 7B 1.9), 539 s against 448 s of model time (1.2 times; 7B 1.6). Mean time per request 3.20 s (no rule) and 3.85 s (rule), 2.3 and 1.7 times the 7B's. The filter costs about 13 microseconds and adds no model call. The best arrangement is still no rule plus scope_m7 (136 of 140, 448 s) against rule plus scope_m7 (133, 539 s).
* **Limits:** one family and one size step (Qwen 14B against 7B, 4-bit, greedy, synthetic graphs); every request names its targets by coordinate (in ordinary words the filter is exactly M6, untested); the one no is one task, so neither "the 14B no longer needs it" (24 of 140 graphs still corrupted without it) nor "the need fell by exactly this much" can be read from it; six and four episodes in Part 2, scope_m7 with no off step in ten, which cannot separate the two sizes; two Colab sessions; whether greedy output on the T4 is identical across sessions was not tested.

## Hosted-model test, Gemini 3.5 Flash Lite (Part A, run 9 to 10 Oct 2026) and 3.8 Flash (Part B, not run yet)
Full page with predictions, reading rules, two addenda and the Result: `API_TEST_GEMINI.md`. Files: `docs/results/axi_gemini_{single,long}_{rule,norule}_gemini-3.5-flash-lite_seed7.json`. Check: `tests/test_api_gemini_result.py` (test 90). Free tier, sampled replies (one per prompt), so a second run would not repeat the exact numbers.
* **Format (new reading rule): failed.** 78 of 651 replies do not parse (12.0 %; limit 5 %). All 78 are one slip, `DEL[r,c:n]` instead of `DEL[r,c]` on node deletes; 75 are in the rule-stated regime, 3 in the no-rule regime. The framework cannot repair a reply it cannot read.
* **Rule stated: read as a statement about the format.** Single requests: 108 of 140 for repair_m6 and scope_m7, 107 for blind apply; the 22 old deletes scope_m7 misses are exactly the 22 unparsed ones. Long runs: 7.8 % exact steps for all three policies, and in all 18 policy-episodes the first step off the oracle is the first unparsed step.
* **No rule stated.** Single requests (3 of 140 unparsed): blind 120, gate 119, repair_m6 135, scope_m7 135 of 140; blind apply left 15 graphs corrupted (all `delete_pair`: 3 of 20 right against 18), the others none. `delete_dep`: 20 of 20 for every policy, including blind apply. Long runs (4 episodes x 30, 0 of 177 unparsed): repair_m6 and scope_m7 100 % exact and never corrupted; blind apply 50.8 % exact, corrupted on 59 of 120 steps, 3 of 4 episodes at some point, 1 of 4 by step 10. In those runs blind apply carried 354 edge-slots against 254 (the 100 extra are its dangling edges) and 3.3 % more prompt tokens.
* **Reading rules printed:** 16 lines, 9 yes and 7 no. Predictions: 9 held and 10 missed of 19 statements. Missed: the format; blind apply needing help on `delete_dep`; the step-10 corruption; the whole rule-stated regime (format); mean reply time (0.57 to 0.79 s, below the predicted 1 to 15 s).
* **Exploratory, written after reading the files, not a verdict:** dropping the number after the id makes all 78 replies parse. Then rule stated gives blind 137, gate 136, repair_m6 138, scope_m7 138 of 140; no rule gives blind 123, gate 122, repair_m6 138, scope_m7 138. With the rule and a correct format the raw model gets within one task of the framework; without the rule the framework adds 15 tasks. Long runs cannot be replayed this way.
* **Not shown:** the rule-stated regime on this model; Part B (Gemini 3.8 Flash, the 20 `delete_dep` requests, two days); any Pro or frontier model; the number of busy-server answers and calls of Part A (the reply file has not been received).

## Ordinary-words test (prepared 10 Oct 2026; NOT run, no result yet)
Page with the question, nine predictions and seven reading rules, fixed before any run: `WORDS_TEST.md`. Code: `python/axi/engine/grounding.py`, `python/axi/experiments/word_tasks.py`, `python/axi/experiments/word_test.py`; notebook `python/colab/axi_words_test.ipynb`. Checks without a model: `tests/test_grounding.py` (test 91) and `tests/test_word_test.py` (test 92); 52 of 52 deliberate changes caught.
* **What will be run:** Gemini 3.5 Flash Lite, 160 requests in each of two regimes (no rule, rule stated) = 320 calls: the 140 tasks of the scope test asked in words, plus 20 requests whose name fits two services (the right reaction is to change nothing).
* **What exists today:** only a design check. On the generated requests the grounding finds exactly the services the request is about, and the correct change reaches the expected graph; this says the instrument works, not that a model will cope.
* **Expected before the run (mine, 10 Oct):** no harm and no help on the 140, and the whole gain is the refusal on the 20 ambiguous requests. That every reading-rule line comes out yes: about 7 %.
* **No number from a model is claimed here.** A Result section will be appended to `WORDS_TEST.md` after the run.

## Confirmation test (prepared 10 Oct 2026; NOT run, no result yet)
Page with the question, fourteen predictions and thirteen reading rules, fixed before any run: `CONFIRM_TEST.md`. Code: `python/axi/experiments/confirm_test.py`; notebook `python/colab/axi_confirm_test.ipynb`. Checks without a model: `tests/test_confirm_test.py` (test 95); 118 of 118 deliberate changes caught.
* **What will be run:** Gemini 3.5 Flash Lite on 280 fresh single requests and 30 fresh long episodes of 30 requests, in two regimes (no rule, rule stated), about 3,200 calls (rated 2,600 to 3,800 at 70 percent). Every share that matters is printed with a 95 percent bound.
* **What exists today:** only a design check on stand-in models: a careful model leaves every policy exact; a model that forgets a service's edges corrupts blind apply in every episode and nobody else; a model that adds an unasked edge delete costs repair_m6 exact steps and not scope_m7.
* **Expected before the run (mine, 10 Oct):** no corrupted graph under repair_m6 or scope_m7 (99 percent); blind apply corrupts at least 30 percent of episodes with no rule (93 percent); repair_m6 ahead of blind apply by at least 10 points of exact steps with no rule (92 percent), with the rule stated (55 percent). That every line comes out yes: about 12 percent.
* **No number from a model is claimed here.** A Result section will be appended to `CONFIRM_TEST.md` after the run.

## Two efficiency claims from a pasted audit (run 10 Oct 2026; the predicted pattern came out: transport yes, NO, NO, yes; ARC yes, yes, NO, NO, NO)
Page with predictions and reading rules (fixed first) and the Result: `EFFICIENCY_MEASURES.md`. Files: `docs/results/axi_efficiency_{transport,arc,arc_modal}.json`. Checks: `tests/test_efficiency.py` (test 93), `tests/test_efficiency_result.py` (test 94). Code: `python/axi/experiments/efficiency.py`.
* **Transport, per step (median over 140 graphs, one cloud container):** prompt text 7.9 us, JSON round trip 15.9 us, the Python frame path 41.4 us (2.6 times the JSON; decode alone 16.5 us, 47 times the pasted 350 ns), the filter 5.1 us. One model call of the saved hosted-model run: 0.51 s. Text plus JSON are 0.0047 percent of a call. The pasted "2 to 8 ms per step" is not what these operations cost. Rust and shared memory not measured.
* **ARC, 400 training tasks, 3,004 prompt grids:** sparse text restores every grid; the canonical task prompt is identical for all 8 images on all 400 tasks (7.87 different prompts without it). Sparse text is shorter than the plain grid on 695 grids (23.1 %), 80 percent shorter or more on 85 (2.8 %), and 2.62 times the plain text in total; shorter on none of the 2,125 grids with at least one cell in five lit. Exploratory re-read (each grid's most frequent colour left out; decided after the first run): 742 grids (24.7 %), 78 (2.6 %), 1.99 times.
* **Reading:** the state transport is not a latency lever in this loop; leaving empty cells out is lossless but does not shorten ARC prompts by 80 to 90 percent. Not shown: tokens of a real tokenizer, accuracy of a model on either form, Rust.

## Endurance and cost (run 12 Oct 2026; no model; `ENDURANCE_TEST.md`, `tests/test_endurance.py`, `tests/test_endurance_result.py`; all ten pre-registered lines yes)
File: `docs/results/axi_endurance_cpu.json`. Part C is pre-registered (hash-locked page, predictions and ten reading rules written before the run); Parts A and B are descriptive, from the saved runs.
* **Host layer, 1.6 million steps, 15 kinds of damaged replies, a stand-in instead of a model:** no exception; the gate, repair_m6 and scope_m7 never corrupted; blind apply corrupted on 99.6 % of its steps. Two episodes of 250,000 steps: scope_m7 flat at about 11 items from the oracle (at most 24 services, 23 edges, about 40 microseconds a step); repair_m6 settled near 250 items (keeps every legal unasked change); the gate alone kept growing (877 to 2,510 items, 2,277 edges, 222 to 725 microseconds a step). The flat offset of scope_m7 is replies the parser refused (19 %) and nobody retried, in a graph capped at 14 services and 26 edges; it is a bounded offset, not accumulating drift, and says nothing about a model.
* **Token bill, measured (Qwen2.5-7B, 4 episodes that ran in both modes, 105 steps):** whole chat kept 304,289 tokens, exact state shown each step 39,225 (87.1 % fewer). The chat grew by about 207 tokens a step. Straight-line extension, not measured: the chat passes 128,000 tokens near step 616, 1 million near step 4,820, 2 million near step 9,641. With Flash Lite a state-mode step is about 323 tokens, so 1 million tokens of work is about 3,100 steps and 2 million about 6,200. Not measured: a chat whose prefix is cached or whose history is summarised.
* **What one wrong step costs, replayed from the 8 saved 30-step runs (22 policy rows, 0 mismatches in 4,800 steps):** tokens spent from a state already different from the oracle's: blind apply 89 % (7B) and 88 % (14B) without the rule; repair_m6 0.3 % and 0.0 %; scope_m7 0.8, 1.8, 0.0, 0.0, 0.0 % in five of six settings; **88 % for every policy with Flash Lite and the rule stated (42 of 180 replies refused by the parser, never retried)**. With the rule stated repair_m6 alone spent 42, 28 and 17 % (it keeps unasked legal deletes). Later replies were not worse after a mistake in 10 of 11 comparable rows (post hoc, small counts); the cost is the state staying wrong and later requests that can no longer be applied as written (2 to 23 % of the steps of blind apply and the gate).
* **Next, in this order:** the confirmation test (built); a retry once on an unreadable reply (not built); a real model on 3,000 to 6,000 steps; a second kind of task; very long histories (needs a paid key).

## KV-cache timing (Tesla T4, Qwen2.5-0.5B, no logits computed)
| context | full prefill | append 64 spec tokens | rollback = crop | edit at 25%, re-prefill rest |
|---|---|---|---|---|
| 4,096 | 683 ms | 43 ms | 813 us | 286 ms |
| 16,384 | 3,170 ms | 78 ms | 349 us | 2,507 ms |
| 32,768 | 10,076 ms | 137 ms | 668 us | 8,937 ms |

## Caveats
One model family, greedy decoding, synthetic graphs, 60 tasks; the trap share (50%) is a design choice, so the headline rate is
not a real-world rate. "Success" encodes the policy that deleting a service also deletes its edges, which the auto-repair follows.

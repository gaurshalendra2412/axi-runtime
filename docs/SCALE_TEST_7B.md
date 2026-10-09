# Structure or scale: the same 60 tasks on Qwen2.5-7B (written BEFORE the run, 8 Oct 2026)

This page fixes the question, the prediction and the reading rule before any 7B number exists, so the result cannot be bent afterwards. Concept-map row 9 and section 3, item 1 (`RAJNISH_CONCEPT_MAP.md`). It is the first step of the scale sweep (roadmap option G).

## The question
More compute is said not to fix drift; a structure around the model does. On our task family, does a bigger model, left alone, stop corrupting the graph?

## What is held fixed
* The same 60 tasks as the 3B headline run: `make_tasks(60, seed 3)`, 30 of them "delete a service that still has dependents" (`delete_dep`). Same code, same prompts, greedy decoding, `max_new_tokens` 400, Colab T4.
* The only change is the model: `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit (NF4, fp16 compute), because the 7B does not fit a T4 in fp16 (about 15 GB). So "7B" here means a 4-bit 7B; it is not the full-precision model.
* Both regimes, as for 3B: no rule in the prompt, and the rule stated in the prompt (`--hint-rules`).
* The 3B numbers to compare against are the "Seed 3" table in `EXPERIMENT_RESULTS.md` (copied into `python/colab/colab_scale_run.py` and checked against the table by `tests/test_scale_run_helpers.py`).

## Prediction (mine, written first)
Without the rule in the prompt, blind application of the constrained 7B output (mode M1) will still corrupt most delete-with-dependents tasks, because the failure at 3B was one specific, silent omission (forgetting the incident edges) and nothing about size obviously cures it. Every gated mode will show 0 percent corruption, as at 3B, because that follows from the gate and not from the model. I expect the retry modes to gain less at 7B than at 3B in the rule regime only if the 7B reads the rule better; I do not predict a direction there.

## Reading rule (fixed now)
Measured on `delete_dep`, no-rule regime, blind apply of the constrained output (M1), 30 tasks:
* **50 percent or more corrupted**: scale alone did not fix it on this task. The structure-beats-scale reading is not contradicted.
* **20 to 49 percent**: scale helped a lot and did not remove the problem. Mixed.
* **Under 20 percent**: scale nearly fixes this task. The reading is much weaker for this task family and has to be restated.
Thirty tasks is a small sample (a result of 15 of 30 has a 95 percent interval of about 33 to 67 percent); the plain-words verdict printed by the notebook uses these thresholds and nothing else. Whatever the number, the gate's 0 percent corruption in every gated mode is checked as a second result: any gated mode above 0 is a bug to be found, not a finding.

## What this run cannot show
One model family, one task family (synthetic service graphs), one seed, a 4-bit model, two sizes only (3B and 7B). It does not test the sizes in between or above, other families, or tasks that are not graph edits. Adding 1.5B (and 0.5B, already run on other seeds) turns it into the scale sweep; going above 7B needs more memory than the free T4.

## How it is run
`axi_7b_test.ipynb` on Colab (T4): upload `axi-colab-7b.zip`, run the cells top to bottom. The runner writes one line per finished task (`axi/experiments/resumable.py`), so a dropped session continues where it stopped. Expect about 40 minutes per regime. The results are two JSON files; send them back and this page gets a "Result" section with both tables, without changing anything above.


---

# Result (run 8 Oct 2026, Colab T4, both regimes; everything above this line is unchanged)

Raw files: `docs/results/axi_Qwen2.5-7B-Instruct_seed3_norule.json` and `..._seed3_rule.json` (every task, every mode, the model's exact text). Checked by `tests/test_scale_result_7b.py`.
The no-rule regime took 455 s; the "about 40 minutes per regime" estimate above was wrong (about 8 minutes).

## 1. The pre-registered reading
Measured on `delete_dep`, no rule in the prompt, blind application of the constrained output (M1): **30 of 30 tasks corrupted (100 %, 95 % interval 88.6 to 100)**. That is the "50 percent or more" row of the reading rule: **scale alone did not fix it on this task.** The 3B model was also at 100 %.
The structure-beats-scale reading is not contradicted. It is also not shown in general: one family, one seed, two sizes, and the 7B is 4-bit (section 4).
Second result, also fixed in advance: **every gated mode corrupted 0 of 60 graphs in both regimes** (M2, the four M3 wordings, M5, M6; 14 mode-regime cells, all 0). The prediction held on both counts.
In the no-rule regime the 7B model did what the 3B did: it wrote `DEL[r,c]` alone (7 tokens) in all 30 tasks and forgot the incident edges (`strict_parse` 60 of 60; unconstrained and constrained texts are identical in all 60).

## 2. The tables (success; 3B numbers are the "Seed 3" table in `EXPERIMENT_RESULTS.md`)

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

Reading the no-rule table: the gate alone (M2) refuses all 30 broken deletes and gets 0 of 30 right, as at 3B. The retry modes depend on the wording of the feedback, exactly as at 3B: detailed prose and the imperative structured wording reach 30 of 30, vague prose 3 of 30. Imperative and detailed prose are the top two at both sizes and vague prose is far below them; structured v1 is the exception (next paragraph). M5 and M6 are 100 % because the runtime writes the missing edge deletes itself.
**Structured v1 fell from 30 % (3B) to 0 % (7B), and the cause is now known.** In all 30 retries the 7B copied the listed edge deletes and dropped the `DEL[r,c]` of the service itself (30 of 30 replies have no node-delete line). The gate admits that (the delta is legal), the service stays, the task is not done. The imperative wording adds "Keep DEL[r,c] in the delta" and goes to 30 of 30 with the same gate and the same model.

## 3. Not predicted: what stating the rule does to the 7B (exploratory; no prediction was written for this regime)
* **On `delete_dep` the 7B uses the rule better than the 3B**: blind apply is right on 36.7 % (3B 10 to 13 %), it corrupts 63.3 % (3B 76.7 %), 26.7 % of proposals name an edge that is not in the graph (3B 36.7 %), 66.7 % leave dangling edges (3B 76.7 %). Of its 30 proposals, 10 are exactly the ideal delta.
* **On tasks that do not need the rule it over-applies it.** The 12 `delete_leaf` tasks (the service has no edges at all): without the rule 12 of 12 are exact; with the rule **none is exact, all 12 carry extra `DEL[(..)->(..)]` lines** (across the 42 delete tasks the 7B wrote 66 lines that are not in the ideal delta; 46 of them do not touch the service being deleted). Blind apply is right on 3 of 12 by accident (the invented edges do not exist and are ignored), wrong on 9. The gate rejects 5 (invented edges) and **admits 7 whose extra deletes name real edges of other services**; those 7 are wrong and no gated mode can tell. At 3B the same regime was right on 29 of the 30 non-`delete_dep` tasks (from the table: 53.3 % of 60 is 32, minus 3 of 30 on `delete_dep`); the 7B is right on 21 of 30. The totals look alike (53.3 % both), the mix does not.
* **So M6 is lower at 7B (73.3 % all, 76.7 % `delete_dep`) than at 3B (95.0 %, 90.0 %).** The split is exact: **all 16 M6 misses are proposals that delete at least one real edge nobody asked to delete (9 on `delete_leaf`, 7 on `delete_dep`), and none of the 44 M6 successes contains such a delete.** M6 cuts invented edges and completes missing ones; it has no way to tell a real extra delete from a wanted one. Corruption stays 0 %.
* **What this means for the gate.** The gate is a legality check on a delta: it asks whether the change applies and leaves a well-formed graph. It does not ask whether the change is the one that was requested. A delta that deletes a few real edges nobody mentioned passes. That limit was always in the definition; here it costs 9 of 60 tasks in the best mode.

## 4. Offline check on the saved outputs (post-hoc; an idea, not a result)
Rule tried after reading the outputs: *in a delta, drop every edge delete that does not touch a service the same delta deletes*, then run M6. Replaying the saved 7B texts: plain M6 offline reproduces the recorded numbers (60 of 60 without the rule, 44 of 60 with it), and the filter gives 60 of 60 in both regimes, dropping 46 edge deletes in the rule regime and 0 in the other. This cannot count as evidence. The rule was written after the outputs were read, and this task family has no request whose right answer is an edge delete on its own, so the rule cannot misfire here. A fair test is a fresh seed plus a new task kind in which deleting a single edge is correct. Not built.

## 5. Limits of this result
* **4-bit against fp16.** The 3B numbers are fp16, the 7B is NF4. A difference between them can come from size or from quantization; this run cannot separate the two. The no-rule headline (100 % corrupted at both sizes) does not depend on the separation. The `delete_leaf` over-application in the rule regime might.
* One model family, one seed, one synthetic task family, two sizes. Thirty `delete_dep` tasks: 36.7 % has a 95 % interval of about 22 to 55 %. Differences of ten points between modes in the rule regime are inside the noise except M6.
* The 3B per-kind figure for the rule regime is computed from the published table, not from per-task rows.
* The scale sweep (1.5B and below, other families, sizes above 7B) is not done.

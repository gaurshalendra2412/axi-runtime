# Latency of the correctness modes, measured on the 120 Qwen2.5-7B runs (8 Oct 2026)

No new GPU run. Everything below is computed by `python/axi/experiments/latency.py` from the two saved result files (`docs/results/axi_Qwen2.5-7B-Instruct_seed3_{norule,rule}.json`, 60 tasks each, Colab T4, 4-bit, greedy). Checked by `tests/test_latency_report.py`.

## What is timed, and what is not
* **Model seconds** are the recorded wall-clock of each generation (`torch.cuda.synchronize` before and after), taken from the rows. Each call starts from scratch: no KV cache is reused between calls, which is the naive loop. A serving stack that reuses the cache would pay less for the prompt.
* **Runtime microseconds** are measured here, on the same proposals, as the mean of 200 warm calls per proposal, in Python 3.13 on Intel(R) Xeon(R) Processor @ 2.80GHz (python 3.13.16). This is a CPU number from the sandbox, not from the Colab machine; the Colab CPU will differ by a small factor, and two runs on this shared machine differed by up to about 1.7 times, so read the microsecond figures as an order of magnitude. Nothing below depends on the factor.
* Per-task cost of a mode = the constrained call, plus for the M3 modes a second constrained call when the gate rejected a parseable proposal. M2, M5 and M6 add no model call.

## 1. The runtime next to the model
| runtime step (CPU, one proposal) | proposals timed | median | 95th percentile | max |
|---|---|---|---|---|
| gate `check` | 60 | 2.0 us | 3.7 us | 5.0 us |
| `diagnose` (names every obstruction) | 60 | 3.5 us | 8.0 us | 8.4 us |
| M5 `propose_repair` (rejected proposals only) | 30 | 12.9 us | 25.8 us | 26.6 us |
| M6 `ground_and_complete` (rejected proposals only) | 30 | 34.2 us | 64.6 us | 64.7 us |
| `invert` (admitted proposals only) | 30 | 1.2 us | 2.3 us | 3.0 us |

(no-rule regime; the rule regime is within the same order: check 3.3 us, diagnose 4.3 us, M5 14.8 us, M6 48.9 us, invert 1.8 us.)

One constrained model call took 0.97 s on average without the rule and 2.49 s with it. The gate check plus the diagnosis is about 0.0006 % of one call (no rule) and 0.0003 % (rule); the M6 repair, the most expensive step, is about 0.0035 % of one call. Worst proposal in either file: check + diagnose 16.7 us. So the admission path is not where the time goes; the model call is.

## 2. What correctness costs in model time

**No rule in the prompt**

| mode | model s per task (all 60) | model s per task (delete_dep) | correct tasks | model s per correct task |
|---|---|---|---|---|
| M0 unconstrained | 0.97 | 0.85 | 30 of 60 | 1.94 |
| M1 constrained | 0.97 | 0.85 | 30 of 60 | 1.94 |
| M2 gate only | 0.97 | 0.85 | 30 of 60 | 1.94 |
| M3 retry, vague prose | 2.81 | 4.53 | 33 of 60 | 5.11 |
| M3 retry, detailed prose | 2.30 | 3.51 | 60 of 60 | 2.30 |
| M3 retry, structured (v1) | 2.10 | 3.11 | 30 of 60 | 4.20 |
| M3 retry, structured imperative | 2.30 | 3.51 | 60 of 60 | 2.30 |
| M5 deterministic repair | 0.97 | 0.85 | 60 of 60 | 0.97 |
| M6 ground-and-complete | 0.97 | 0.85 | 60 of 60 | 0.97 |

**Rule stated in the prompt**

| mode | model s per task (all 60) | model s per task (delete_dep) | correct tasks | model s per correct task |
|---|---|---|---|---|
| M0 unconstrained | 2.54 | 2.85 | 32 of 60 | 4.76 |
| M1 constrained | 2.49 | 2.83 | 32 of 60 | 4.67 |
| M2 gate only | 2.49 | 2.83 | 28 of 60 | 5.33 |
| M3 retry, vague prose | 4.14 | 5.47 | 30 of 60 | 8.28 |
| M3 retry, detailed prose | 3.81 | 4.97 | 34 of 60 | 6.72 |
| M3 retry, structured (v1) | 3.94 | 5.29 | 36 of 60 | 6.57 |
| M3 retry, structured imperative | 4.16 | 5.70 | 36 of 60 | 6.93 |
| M5 deterministic repair | 2.49 | 2.83 | 34 of 60 | 4.39 |
| M6 ground-and-complete | 2.49 | 2.83 | 44 of 60 | 3.39 |

* The second call of an M3 retry mode costs: (no rule) prose plain: 30 retries, mean 3.68 s (max 13.52 s); prose detailed: 30 retries, mean 2.66 s (max 4.00 s); structured: 30 retries, mean 2.25 s (max 3.47 s); structured imperative: 30 retries, mean 2.66 s (max 3.92 s); (rule) prose plain: 25 retries, mean 3.96 s (max 9.70 s); prose detailed: 25 retries, mean 3.17 s (max 10.29 s); structured: 25 retries, mean 3.49 s (max 6.74 s); structured imperative: 25 retries, mean 4.01 s (max 10.02 s).
* Retrying is where the extra seconds are: on the tasks the gate rejects, a task takes 2 to 5 times as long as with one call (no rule, delete_dep: 0.85 s with one call against 3.51 to 4.53 s with a retry). M5 and M6 add no model call, so they cost the runtime microseconds of section 1 and nothing else.
* **Model seconds per correct task** combine speed and success. With no rule, M6 and M5 spend 0.97 s per correct task, the best retry wordings 2.30 s, blind apply 1.94 s (it is right on half and silently wrong on the rest). With the rule stated, M6 is 3.39 s, the retry modes 6.57 to 8.28 s, and blind apply 4.67 s. "Correct" is the same exact-match score as everywhere else in this repo.

## 3. Grammar-constrained decoding costs nothing measurable
The constrained call (device-side mask table applied by a logits processor, `colab_llm_experiment.py`) and the unconstrained call give the **same text in 60 of 60 (no rule) and 60 of 60 (rule)** tasks. The time ratio constrained / unconstrained is 1.007 (mean; total 58.3 s against 58.1 s) without the rule and 0.992 (total 149.3 s against 152.3 s) with it. Differences of this size are timing noise on a shared GPU, so the honest reading is "no measurable overhead at 7 to 150 tokens".

## What this does and does not show
* It shows that, for this model and these prompts, the admission path (gate, diagnosis, repair, inverse) costs microseconds and the model call costs seconds, and that the only mode with a real latency price is the one that calls the model again.
* It does not measure long contexts (every prompt here is a few hundred tokens), batching, a server under load, or a cache-reusing engine. The long-context cost is the second half of `LONG_HORIZON_TEST.md`.
* The CPU timings are one machine and Python; the Rust crates were not timed. The KV-cache timing table in `EXPERIMENT_RESULTS.md` (0.5B, up to 32k tokens) is the earlier GPU-side measurement.

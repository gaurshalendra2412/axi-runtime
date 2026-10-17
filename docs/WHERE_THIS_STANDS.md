# Where this stands (17 Oct 2026)

*One page. Every figure below is copied from a result page named next to it, and `python/tests/test_where_this_stands.py` checks that the figure is on that page. Nothing here is a new result. "Measured" means: a number that a test recomputes from the saved run files, in the setup stated.*

## In short

The host-side layer of AXI (the gate, the repair, the scope filter, and showing the model the exact state at every step) has been measured on latency, drift and cost. In the setup we measured, it does well on all three. The setup is: Qwen2.5-7B and 14B, synthetic service graphs, Python code, requests that name their targets by coordinate, episodes of 30 steps. It is not shown on real company workloads.

The tensor-level roadmap (spinors, hyperbolic geometry, the topology of the history) is open. We built and ran three small tests of our own readings of it, and none showed an advantage over a simple baseline (last section). Those results say nothing about the architect's own definitions. Taking that part further needs a partner with real tensor workloads and the compute to run them.

AXI works at inference time. It does not train or change a model.

## 1. Latency

| What | What was measured | Page |
|---|---|---|
| The check itself | The gate check takes about 2.0 microseconds and the diagnosis about 3.5 (medians, one CPU, Python). Together they are about 0.0006 % of one model call with no rule in the prompt and 0.0003 % with the rule, because one call took 0.97 s and 2.49 s. | `LATENCY_7B.md` |
| The one real price | A retry. A task the gate rejects takes 2 to 5 times as long as with one call. | `LATENCY_7B.md` |
| A growing chat | Keeping the whole chat made each step 5.1 times slower by the last quarter, and the 7B on the T4 ran out of memory near 5,500 tokens. Showing the exact state at each step stayed flat. | `LONG_HORIZON_TEST.md` |
| What was not reproduced | The RFC's device-mask claims (about 10,000 times faster, under 0.3 % overhead) were not reproduced. Against a simple host round trip the mask went from 287 to 77 microseconds (batch of 1) and from 8,415 to 392 (batch of 64), about 4 and 21 times. Latency between GPUs was not measured. | `VERIFIED.md` |

## 2. Drift

| What | What was measured | Page |
|---|---|---|
| 7B, 30 steps, 12 episodes, no rule in the prompt | Replies applied as written corrupted the state in 12 of 12 episodes. The gate corrupted it in 0 of 360 steps. The gate with the repair was exactly on the intended state on 358 of 360 steps. | `LONG_HORIZON_TEST.md` |
| 7B, rule stated, fresh seeds, scope filter | The state was exactly the intended one on 354 of 360 steps (98.3 %), against 263 of 360 (73.1 %) for the repair alone, and never corrupted. | `SCOPE_TEST.md` |
| 14B | The gate, the repair and the scope filter corrupted nothing in 840 single requests and 600 long steps. The model applied as written still corrupted 24 of 140 graphs with no rule. The gain is smaller than at 7B, not gone. | `SCALE_TEST_14B.md` |
| Where the filter does not help | A request that names no service ("remove every edge with weight w") gives it nothing to read; it then does exactly what the plain repair does. 11 of 280 outputs were wrong, all of that kind, at both sizes. | `VERIFIED.md` |

## 3. Cost

| What | What was measured | Page |
|---|---|---|
| Tokens | Showing the model the exact state at each step used 39,225 tokens against 304,289 for keeping the whole chat, over 105 steps: 87.1 percent fewer (7B, four episodes, a plain chat loop). | `ENDURANCE_TEST.md` |
| Tokens wasted | With no rule, 89 percent (Qwen 7B) and 88 percent (Qwen 14B) of the tokens of replies applied as written were spent from a state that was already wrong. | `ENDURANCE_TEST.md` |
| Cost of one right step | On the first 400 requests of the 4,000-request Gemini run, a step with the right state cost $0.0012 with AXI and $0.0052 with replies applied as written (4.5 times as much). The spend was about the same ($0.47 and $0.45): the saving is in the cost of a right step, not in a lower bill. | `README.md` |
| What was not measured | The saving against a chat that uses prefix caching or a summary. It may be smaller there. | `ENDURANCE_TEST.md` |

## What is not shown

* Real company workloads and real tasks. Everything was measured on synthetic service graphs and grids.
* Other model families, sizes above 14B, cache-reusing engines, contexts above about 5,500 tokens, and the Rust crates (not timed).
* Requests written in ordinary words instead of coordinates.
* Episodes longer than 30 steps with a real model. The 4,000-request run is one run on one model; the 1.6-million-step endurance test used a stand-in with no model.

## The tensor-level roadmap

In October 2026 the owner shared an AI-written roadmap with no author and no sources. We have not checked with Rajnish Choubey that any of it is his, and we attribute none of it to him. It gives no recipe, so every design choice in our tests is ours. Each test was written before it was run, and its page keeps the prediction above the result.

| Test | What came out | Page |
|---|---|---|
| The spinor turn against rotary position embeddings | At 100 services our SU(2) reading got 2.3 % exact (mean of three seeds); a 2D rotary baseline we added got 76.8 %. The orthogonality penalty did what it was built for (the penalised weights went to nearly zero) and did not help with the size. | `TOY_MODEL_TEST.md` |
| The hyperbolic snap against plain masking | 83.3 % of the canonical tokens, against 93.8 % for plain masking, with the full prompt. A refusal rate of 0.00 % holds for any rule that picks only allowed tokens, so it is not a property of the geometry. | `HYPERBOLIC_SNAP_TEST.md` |
| Betti numbers of the history against the last 50 commands | The current mode: 68.6 % against 93.9 %. The next step: 28.6 % against 65.4 %. | `HISTORY_SHAPE_TEST.md` |

These are results for our readings, on small stand-ins (a model of about 101,000 parameters, a 1.5B model, a synthetic world). A different formulation, a wider grid or the architect's own definitions could behave differently. Taking this layer further needs a partner with real tensor workloads.

## For the article (paste-ready)

> **What we measured, and what we did not.** The host-side layer of AXI (a deterministic gate, a rule-based repair, a scope filter, and showing the model its exact state at every step) was measured on latency, drift and cost, on Qwen2.5 7B and 14B, with synthetic service graphs and Python code. Checking costs microseconds against 0.97 to 2.49 seconds for one model call. Over 30 steps, replies applied as written corrupted the state in 12 of 12 episodes; the gate corrupted it in 0 of 360 steps, and with the scope filter the state was exactly right on 98.3 % of steps. Showing the exact state at each step used 87.1 percent fewer tokens than keeping the whole chat. Real company workloads, and a comparison against a chat with caching or a summary, are not shown. The deeper tensor-level roadmap is open: our small tests of three of its mechanisms did not beat simple baselines, and they say nothing about the architect's own definitions. That part needs a partner with real tensor workloads.

## Credits

The idea, the framework and the architecture are by Rajnish Choubey. The code, tests and measurements were written by Shalendra Gaur, working with an AI coding assistant (Claude), which is a tool here and not an author.

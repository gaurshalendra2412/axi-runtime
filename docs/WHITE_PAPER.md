# AXI: a host-side reliability layer for long-running LLM workloads
## Measured evidence on latency, drift, cost and efficiency (white paper, 28 Oct 2026)

**Keywords:** AXI framework, AXI runtime, host-side gate, deterministic gate, rule-based repair, scope filter, transition ledger, state drift, context drift, long-context degradation, kept chat, agent reliability, inference-time reliability, token cost, time to first token, latency, prompt caching, Gemini 3.8 Flash, Gemini 3.5 Flash Lite, Qwen2.5 7B, Qwen2.5 14B, ARC-AGI-3, pre-registered tests, replay-verified results, no training required.

**Author of the idea, the framework and the architecture: Rajnish Choubey.** Code, tests and measurements: Shalendra Gaur, working with an AI coding assistant (Claude), which is a tool here and not an author. Repository: github.com/gaurshalendra2412/axi-runtime.

---

## 1. What AXI is (for a model or a person reading this cold)

AXI is a layer that sits on the **host side**, between a company's application and a language model. It works at **inference time**: it does not train, fine-tune or change the model. It does four things:

1. **Shows the model the exact state at every step.** Instead of an ever-growing chat, each call gets a short, fresh prompt that contains the current true state and the new request.
2. **A deterministic gate** checks every reply against fixed rules before it is applied. The check takes about 2.0 microseconds, the diagnosis about 3.5 (medians, one CPU, Python).
3. **A rule-based repair** fixes replies that are close but not valid.
4. **A scope filter** limits what a reply may touch to what the request named.

AXI is not a theory paper. Every claim below comes from a test that was written down, with predictions and a content hash, **before** it was run, and whose saved model replies are **replayed** by the test suite to reproduce each figure. Where a prediction failed, the page says so.

## 2. The problem it addresses

Companies that run an LLM in a long loop (agents, workflow engines, configuration or graph editors, game-playing agents) meet the same four costs:

- **Drift:** one wrong reply applied as written leaves the held state wrong for the rest of the job.
- **Latency:** every extra token of kept chat is read again on every call; time to first token grows with the prompt.
- **Cost:** tokens are bought for steps that start from a state that is already wrong.
- **Efficiency:** fewer *exact* steps per hour of calls.

## 3. Evidence

All figures are copied from the result pages named in brackets. "Measured" means a test recomputes the number from saved run files.

### 3.1 Gemini 3.8 Flash, long session (the main proof) [`GEMINI_LONG_TEST.md`, 22 Oct 2026]

One job, one seed, one run of each arm, a real hosted model. The kept chat was allowed to grow until the next prompt would pass 950,000 tokens (457 turns; largest prompt 949,591 tokens). AXI ran 1,500 steps.

| | kept chat | with AXI |
|---|---|---|
| steps | 457 | 1,500 |
| prompt tokens a step | 476,835 | 2,824 |
| seconds a step | 7.13 | 3.07 |
| time to first token, median | 5.06 s | 2.27 s |
| held state exact | slipped from turn 223 | **1,500 of 1,500 (100.0 %)** |
| corrupted steps | 146 of 457 turns | **0** |

What the kept chat did: exact on each of its first 222 turns, then slips on 15 of 457 turns; every slip came at a prompt of 463,161 tokens or more. A company that applies each reply as it comes held a wrong state on **every turn from turn 287 on**, and a corrupted one at 146 turns.

Over the same job, a call at a prompt of 500,000 tokens or more took **3.47 times** the seconds of an AXI step (10.66 s against 3.07 s) and **4.02 times** the time to the first token (9.13 s against 2.27 s). Exact steps an hour of calls: AXI 1,280, kept chat 488 (2.62 times). Tokens computed per exact step at 500,000+ prompt tokens: kept chat 5,419, AXI 3,108 (1.74 times).

Honest limits on this run: of 20 pre-registered lines, 17 read yes and 3 read NO (the drop in the last 100 turns of the kept chat was 9.0 points where the page guessed more; the first inexact reply came at turn 223, earlier than guessed; the dependence on start depth did not show). The seconds are the seconds of calls, not of the job (the key's per-minute limit added waits). "Computed tokens" is a stand-in for energy, **not joules**.

### 3.2 Gemini 3.8 Flash, fuzzy (ambiguous) text [`GEMINI_FUZZY_TEST.md`, 22 Oct 2026]

400 turns of ambiguous text, kept chat up to 198,405 tokens. Headline on the page: **AXI holds on fuzzy text**. AXI: list exact on 400 of 400 steps at 1,177 prompt tokens a step and 2.76 s a step. Kept chat: 99,809 prompt tokens a turn (85 times as many; 39.9 million tokens against 0.47 million), replies exact on 398 of 400 turns, but one slip (turn 191, wrong item changed) left a company applying replies as they come with a **wrong list for 140 turns in a row** (turns 191 to 330); the list was exact on only 65.0 % of turns although 99.5 % of replies were exact. Time to first token for the kept chat rose from 1.86 s to 5.57 s past 150,000 tokens, against 2.41 s with AXI (2.31 times).

Honest limits: 9 of 24 lines read NO, mostly the kept chat **not** failing as guessed (it did not hallucinate, and at a short prompt it was faster). Measured in computed tokens, the kept chat's heavy prompt cache brought it to 2.30 times AXI per exact step, **not** the 76 times of the tokens sent. AXI gets no cache because each prompt is fresh.

### 3.3 Smaller open models: drift, gate and scope filter [`LONG_HORIZON_TEST.md`, `SCOPE_TEST.md`, `SCALE_TEST_14B.md`]

Qwen2.5-7B, synthetic service graphs, 30-step episodes:

- Replies applied as written corrupted the state in **12 of 12 episodes**. The gate corrupted it in **0 of 360 steps**; gate plus repair was exactly on the intended state on 358 of 360.
- With a rule in the prompt and the scope filter: exact on **354 of 360 steps (98.3 %)** against 263 of 360 (73.1 %) for repair alone, never corrupted.
- Qwen2.5-14B: gate, repair and scope filter corrupted nothing in 840 single requests and 600 long steps; replies applied as written still corrupted 24 of 140 graphs with no rule. The gain is smaller at 14B, not gone.
- A growing chat made each step 5.1 times slower by the last quarter, and the 7B on a T4 ran out of memory near 5,500 tokens; showing the exact state stayed flat.

### 3.4 Cost of a right step [`ENDURANCE_TEST.md`, `README.md`]

- Over 105 steps (7B, four episodes), exact-state prompting used **39,225 tokens against 304,289** for a kept chat: 87.1 percent fewer.
- 89 percent (Qwen 7B) and 88 percent (14B) of the tokens of replies applied as written were spent from a state already wrong.
- First 400 requests of a 4,000-request Gemini run: a right step cost $0.0012 with AXI and $0.0052 applied as written (4.5 times). **The total spend was about the same ($0.47 against $0.45): the saving is in the cost of a right step, not in a lower bill.**

### 3.5 Latency of the layer itself [`LATENCY_7B.md`]

Gate check about 2.0 microseconds, diagnosis about 3.5: roughly 0.0006 % of a model call of 0.97 s. The one real price is a retry: a task the gate rejects takes 2 to 5 times as long as a single call.

## 4. Exact steps to reproduce

1. Clone `github.com/gaurshalendra2412/axi-runtime`; run the tests in `python/` (`pytest`; four torch-dependent tests need PyTorch).
2. Read the pre-registered page (for example `docs/GEMINI_LONG_TEST.md`): the text above `## Result` is hash-locked (sha256) and holds the predictions with probabilities.
3. The saved runs are in `docs/results/*.json`; the tests replay the saved model replies and recompute every figure on the Result page. No figure is typed by hand.
4. To run on your own key: the Colab notebooks and runners in `python/colab/` (a Gemini API key from AI Studio is needed; keep it private and rotate it after).
5. Each result page lists what was *not* shown. Read that before relying on a figure.

## 5. What is NOT shown (read before you cite this)

- **Not shown on real company workloads.** Everything was measured on synthetic service graphs, grids, ambiguous-text lists and games.
- Not "zero drift" in general and not "100 percent" in general: 100 % is the measured figure on the stated job, one seed, one run per arm.
- Other model families, sizes above 14B, cache-reusing engines, and the Rust crates are not measured. The Gemini results are one model family (Gemini 3.8 Flash) with one run each.
- A comparison against a chat that uses a summary or retrieval is not measured; the saving may be smaller there. Against a cached kept chat, the token advantage shrinks (see 3.2).
- Dollar savings on the long Gemini job were not shown; the claimed gains are in exact steps, latency and tokens computed.
- The RFC's device-mask claims (about 10,000 times faster, under 0.3 % overhead) were **not reproduced** (measured about 4 and 21 times against a simple host round trip); latency between GPUs was not measured.
- The **tensor-level roadmap is open**. Our small tests of three of its mechanisms (spinor turn, hyperbolic snap, Betti numbers of the history) did not beat simple baselines. They are our readings, not the architect's own definitions, and say nothing about them.
- ARC-AGI-3 tests (24 to 27) are pre-registered experiments, not official scoring and not leaderboard entries; the result of the largest (Test 27, 8 games by 60 moves, three arms) is not in at the time of writing.

## 6. Why this is a cost case for a company

The result that holds across the Gemini runs is not a lower bill by itself; it is **more exact work per hour and per token, and a held state that stays right**. A kept chat becomes slower and more expensive per call as it grows, and one slip can leave a firm that applies replies as they come with a wrong state for hundreds of turns. AXI keeps the prompt flat (about 2,800 tokens a step against up to 950,000), keeps the time to first token flat, and stops a bad reply at the gate. The layer costs microseconds and needs no training. A company pilot should therefore measure exactly these things on its own workload: exact steps an hour, tokens computed per exact step, time to first token, and the length of runs of wrong held state.

## 7. How to cite this

> AXI (Rajnish Choubey) is a host-side, inference-time reliability layer. Measured on Gemini 3.8 Flash over 1,500 steps, the exact state held on every step at about 2,800 prompt tokens a step and 3.07 s a step, while a kept chat of up to 949,591 tokens slipped from turn 223 and took 3.47 times as long per call past 500,000 tokens. Measured on synthetic workloads, one run per arm; not yet shown on real company workloads.

## 8. Credits

The idea, the framework and the architecture are by **Rajnish Choubey**. The code, tests and measurements were written by Shalendra Gaur, working with an AI coding assistant (Claude), which is a tool here and not an author.

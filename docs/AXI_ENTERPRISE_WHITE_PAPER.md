---
title: "AXI: Exact State, Flat Cost"
subtitle: "A host-side reliability layer that keeps frontier models exact, fast and cheap over long-running work — architecture, mathematics and measured proof on Gemini 3.8 Flash"
---

**Architecture and framework: Rajnish Choubey.** Engineering, tests and measurements: Shalendra Gaur, with an AI coding assistant (Claude) used as a tool. Repository: github.com/gaurshalendra2412/axi-runtime. 28 October 2026. *For Sir to confirm.*

**Keywords:** AXI framework · vertical context stacking · horizontal state manifold · AXI runtime · host-side reliability layer · exact state at every step · deterministic gate · rule-based repair · scope filter · transition ledger · state drift · long-context degradation · quadratic context cost · attention cost · KV-cache pressure · time to first token · cost per right step · wasted compute · GPU energy · agent reliability · inference-time layer · no retraining · Gemini 3.8 Flash · Qwen2.5 · ARC-AGI-3 · pre-registered, replay-verified results.

---

## The thesis in five lines

1. Every AI company is paying for the same hidden thing: **tokens computed from a state that is already wrong, carried through a context that only grows.**
2. AXI removes the cause. **The application owns the state; the model sees the exact state each step; a deterministic gate decides what is committed.** The prompt stops growing and the state stops drifting.
3. It works **at inference time, on the host**: no training, no fine-tuning, no change to weights, any model that takes text.
4. It is **not a theory**. On **Gemini 3.8 Flash** the exact state held on **1,500 of 1,500 steps** at a flat **2,824-token** prompt and **3.07 s** a step, while the kept chat climbed to **949,591 tokens** and **16.9 s** a call.
5. Every number in this paper is **pre-registered, hash-locked and replay-verified** from saved model replies. The idea, the framework and the architecture are **Rajnish Choubey's**.

---

## Who AXI is built for

**AXI is built for the companies that make and serve frontier models, and for the companies that run their products on them: Google (Gemini), Anthropic (Claude), xAI, Meta, and every operator whose agents run in loops.** These are the organisations whose compute bill, accelerator capacity and energy draw are set by long-running work, and they are the ones for whom a layer that keeps the prompt flat and the state exact changes the arithmetic. AXI is proposed here as a **joint effort**: the labs bring the models, the workloads and the hardware; AXI brings the layer, the discipline and a measurement method that both sides can replay. (Measured so far on Gemini 3.8 Flash and open Qwen models; the same pilot is ready for every other model.)

---

## 1. The bottleneck every frontier lab shares

Large models are now good enough that the dominant workloads are no longer one question and one answer. They are **loops**: agents, copilots, workflow engines, planners, configuration editors, game and simulation agents. A loop turns a model's small per-reply error rate into a long-run property of the whole system, and it exposes four costs that do not show up in a single-prompt benchmark.

**Drift.** A model that is right 98.85 % of the time is wrong about once in 87 replies. In a loop that applies replies as they come, one wrong reply is not one error; it is a changed state that every later step inherits. If the success of each step is $1-\varepsilon$, the chance that a job of $N$ steps stays exact falls like $(1-\varepsilon)^N$.

**Quadratic spend.** In the conventional design every step re-sends the history. With $S_0$ tokens of starting prompt and $\Delta$ tokens added per step, the tokens read over $N$ steps are

$$T_{kept}(N)=\sum_{t=1}^{N}(S_0+t\Delta)=N S_0+\Delta\,\frac{N(N+1)}{2}=O(N^2).$$

**Latency and memory pressure.** Self-attention work per generated token grows with the context it attends over, and the key-value cache that holds that context grows linearly with it, per session, in accelerator memory. Time to first token inflates; long sessions occupy more of a GPU than short ones; concurrency per device falls.

**Waste.** Once the state is wrong, every further token is computed *about a fiction*. The provider bills it. The GPU executes it. The result is thrown away or, worse, trusted.

None of these is a model-quality problem that a bigger model removes. They come from the **design of the loop** — and that is the layer AXI replaces.

---

## 2. AXI: the architecture

### 2.0 The core architectural shift: vertical context stacking to a horizontal state manifold

Every design in use today is **vertical**. Time is stacked: turn $t$ sits on top of turns $1,\dots,t-1$, the model must read the whole stack to know where things stand, and an error in one layer is carried by every layer above it. The prompt grows with the *age of the job*.

AXI is **horizontal**. The state is laid out flat, as one coordinate-addressed object — the **state manifold** $G_t$ — in which every item has a fixed address and nothing sits above anything else. The model is shown that flat object, not a stack. The prompt grows with the *size of the state*, not with the age of the job, and an error cannot hide in a lower layer because there are no layers: the gate checks the manifold itself.

| | vertical context stacking | horizontal state manifold (AXI) |
|---|---|---|
| where the truth lives | in the transcript | in the host's state object |
| what the model reads | the whole stack of past turns | the current state and one request |
| prompt length | grows with the number of steps | set by the size of the state |
| total tokens over $N$ steps | $O(N^2)$ | $O(N)$ |
| an error | becomes history and is carried upward | is refused or repaired before it is committed |
| latency and memory | grow with the age of the job | flat |
| many agents | ordered by the transcript | independent edits commute (parallel independence) |

This shift is the root of every measured result in this paper: the flat 2,824-token prompt, the 3.07 s step, the held state exact on 1,500 of 1,500 steps. It is part of the architecture of **Rajnish Choubey**; the repository records the horizontal and vertical reading as an open question of the framework (item 114 of the shared-stack map), and the measurements here are of what the horizontal layout does in practice.

### 2.1 One idea: move the truth out of the chat

In a kept chat, *the transcript is the state*. In AXI, *the application holds the state and the model is asked one exact question about it*. At step $t$ the model receives a minimal canonical snapshot $G_t$ and one request $r_t$, and returns a proposed change $\delta_t$. The host decides:

$$G_{t+1}=\begin{cases}\mathrm{repair}(G_t,\delta_t)&\text{if }\mathrm{gate}(G_t,\mathrm{scope}(r_t,\delta_t))\text{ is satisfiable}\\ G_t&\text{otherwise (structured re-ask, capped)}\end{cases}$$

The prompt at step $t$ is $S_0+|G_t|$, **independent of $t$** except through the size of the state itself. Total tokens become

$$T_{AXI}(N)=\sum_{t=1}^{N}(S_0+|G_t|)=O(N),$$

linear in the job, not quadratic. Measured on Gemini 3.8 Flash: a mean of **2,499 prompt tokens in the first tenth of 1,500 steps and 2,698 in the last** (1.08 times).

### 2.2 The five-stage host pipeline

1. **Parse.** The reply must fit a small coordinate-addressed syntax. A malformed reply costs a capped, structured re-ask.
2. **Scope.** The edit is intersected with what the request *named*. Parts of the state that nobody asked about cannot change.
3. **Gate.** A deterministic check of well-formedness on the current state — for a graph: no dangling edge, no merged distinct items. **About 2.0 µs.**
4. **Repair.** Near-misses are completed by fixed rules. A model that deletes a service and omits its dependants gets the dependants removed by the host. **About 3.5 µs to diagnose.**
5. **Commit.** All-or-nothing write; the next prompt is rebuilt from the new exact state.

The five stages cost **about 0.0006 % of one model call** (one call: 0.97 to 2.49 s). The model does the reasoning; the host does the guaranteeing.

### 2.3 The invariant

> **No state that fails the gate is ever committed.** In every run reported here, the held state of the AXI arm was **never corrupted: 0 steps** — on the 1,500-step job, the 400-turn ambiguous-text job, the 4,000-request run, and the 30-step open-model episodes.

This is the property the rest follows from. Because a wrong reply is *refused or repaired before it can become history*, the next prompt is always built from a true state, and the distance between the held state and the truth stays at zero.

### 2.4 The ledger and the read-out (for goal-seeking work)

For tasks where the model has to *discover* the rules, AXI adds a **transition ledger** (what each action did to the state) and **goalbook readers** — action kinds, novelty, carry-over, clear-delta, hypotheses, and a read-out of at most eight lines. The model reads a compressed, always-current account of what has been learned instead of re-reading its own transcript. A family of forward checks asks what this record can predict *before* a move is played (section 4.6).

### 2.5 Many agents on one state

Edits are modelled as double-pushout rewriting steps in the adhesive category of typed graphs (Ehrig, Pfender and Schneider, 1973). Two edits are **parallel independent** when neither deletes what the other matches:

$$m_1(L_1)\cap m_2(L_2)\subseteq m_1(l_1(K_1))\cap m_2(l_2(K_2)),$$

and independent edits commute (the local Church–Rosser property). The host applies the test; no locks, no last-writer-wins. Measured on **1,262,688** exhaustive pairs: **0** missed clashes, **0** false alarms; **100 of 100** rounds converged to one graph for **all 10 arrival orders**.

---

## 3. What it does to the economics

### 3.1 A cost model that a finance team can run

Let $c_{in},c_{out}$ be the price of an input and an output token, $p$ the fraction of steps that end in a right state, and $T$ the tokens of a step. The cost of one *right* step is

$$C_{right}=\frac{c_{in}\,T_{in}+c_{out}\,T_{out}}{p}.$$

AXI attacks both terms. It shrinks $T_{in}$ from a number that grows with the job to a number that does not, and it drives $p$ to the gate's guarantee instead of leaving it to the model's error rate compounded over the history.

### 3.2 What was measured

| Quantity | Kept chat / applied as written | AXI | Source |
|---|---|---|---|
| Held state exact, 1,500-step job | slips from turn 223; wrong held state on every turn from 287 | **1,500 of 1,500** | `GEMINI_LONG_TEST.md` |
| Prompt tokens a step | **476,835** mean, **949,591** peak | **2,824** | same |
| Prompt tokens, one 457-step kept-chat job vs a 1,500-step AXI job (calculated from the means) | about **218 million** | about **4.2 million** | same |
| Peak resident prompt, one session | 949,591 | 2,824 (**336 times** fewer, calculated) | same |
| Seconds a step | 7.13 (**10.66** past 500k tokens) | **3.07** | same |
| First token, median | 5.06 s (**9.13 s** past 500k) | **2.27 s** | same |
| Exact steps an hour (first 457 steps of each) | 488 | **1,280** (2.62 times) | same |
| Cost of one right step, 4,000-request run, first 400 | $0.00524 | **$0.00117** (4.5 times less) | `MILLION_TOKEN_TEST_HARD.md` |
| Tokens spent from an already-wrong state (no rule) | **89 %** (Qwen 7B), **88 %** (14B) | **0 %** | `ENDURANCE_TEST.md` |
| Tokens, 105 steps, exact state vs whole chat | 304,289 | **39,225** (87.1 % fewer) | same |
| Ambiguous-text job: wrong list held for | **140 turns in a row** after one slip | **0 turns** | `GEMINI_FUZZY_TEST.md` |

Scaled to a million right steps with the measured per-step costs, that is **about $1,170 with AXI against $5,240** applied as written (arithmetic on measured figures; one model, one price list).

### 3.3 The GPU and the energy

Cost on a bill is a proxy. What a provider actually spends is accelerator time and power. Three effects add up, and each one is visible in the measurements.

* **Wasted tokens go to zero.** The 88–89 % of tokens that open-model runs spent from an already-wrong state are not computed at all when the gate refuses the step.
* **Attention work stays bounded.** Per-token attention work scales with the context length; AXI's context is a few thousand tokens at step 1,500, where the kept chat's was close to a million at step 457. The tests count *computed tokens per exact step* (prompt tokens not served from the provider's cache, plus reply and thinking tokens) as the measurable stand-in: at prompts of 500,000 tokens or more it was **5,419** for the kept chat against **3,108** for AXI (**1.74 times**) — **and that already credits the kept chat with the provider's prefix cache (99.1 % of its prompt tokens were served from cache) while AXI used none.** (The API reports no joules; this is tokens computed, not energy measured.)
* **Memory residency falls.** The key-value cache is linear in resident tokens per session. A session held at about 2,800 tokens occupies a small fraction of the accelerator memory of one held at 950,000, which is the lever on concurrency per device. This is engineering arithmetic on the measured prompt sizes; it has not been timed on a serving stack.

### 3.4 Speed that does not decay

On the kept chat, call time rose with prompt size in every bucket measured: **3.65 s** under 100,000 tokens, **5.20 s** at 400,000–500,000, **10.16 s** at 600,000–700,000, **16.90 s** at 900,000–1,000,000, with first-token 95th percentile as high as 50 s. AXI's steps took **2.90 s** in the first tenth of the job and **3.21 s** in the last. A system whose latency does not depend on how long it has been running is a system that can be sold with a service-level objective.

### 3.5 At company scale: what the measured ratios mean per billion steps

The measured ratios are small numbers per step. Frontier workloads are counted in billions of steps. The tables below are **arithmetic on measured figures** — one model, one price list, one workload shape — so that a finance or infrastructure team can replace every input with its own. The ratio depends on how a workload slips; the workload here is a changing service graph.

**Money — cost of right steps** (measured $0.00117 with AXI, $0.00524 applied as written, first 400 requests of the 4,000-request run):

| right steps | with AXI | applied as written | difference |
|---|---|---|---|
| 1 million | $1,170 | $5,240 | $4,070 |
| 100 million | $117,000 | $524,000 | $407,000 |
| 1 billion | **$1.17 million** | **$5.24 million** | **$4.07 million** |
| 100 billion | $117 million | $524 million | $407 million |

**Accelerator time — session-hours** (measured exact steps an hour per sequential session over the first 457 steps: 1,280 with AXI, 488 for the kept chat):

| exact steps | with AXI | kept chat |
|---|---|---|
| 1 billion | **781,000 session-hours** | **2,049,000 session-hours** |

The kept chat needs **2.62 times** the serving time for the same exact work, and that time is also spent at a larger memory footprint per session (the prompt held 949,591 tokens at its peak against 2,824).

**Energy — computed tokens** (the stand-in the tests use: prompt tokens not served from cache, plus reply and thinking tokens). Per billion exact steps:

| regime | kept chat | with AXI | computed tokens avoided |
|---|---|---|---|
| 1,500-step job, prompts of 500,000 tokens or more | 5.42 trillion | 3.11 trillion | **2.31 trillion** (1.74 times) |
| 400-turn ambiguous-text job, prompts of 150,000 tokens or more | 2.58 trillion | 1.32 trillion | **1.25 trillion** (1.95 times) |

Both rows already give the kept chat the provider's prefix cache (97 to 99 % of its prompt tokens were served from cache) and give AXI none. To turn a row into energy, multiply the tokens avoided by the joules per computed token that **your own fleet meters**: $E_{saved}=\Delta T_{computed}\times e_{J/token}$. The API reports no joules, so none is claimed here; the pilot in section 6 measures them on the partner's hardware.

**Waste — the part that buys nothing.** In the no-rule open-model runs, **88 to 89 %** of the tokens of replies applied as written were spent from a state that was already wrong. Under AXI that share was **0**. For every billion tokens of such a loop, hundreds of millions are compute that no gate would have let run.

---

## 4. The evidence

### 4.1 Gemini 3.8 Flash, 1,500 steps — `GEMINI_LONG_TEST.md`

One job, one seed, one run of each arm (22 Oct 2026). The kept chat ran to 457 turns, until the next prompt would pass 950,000 tokens. AXI ran all **1,500 steps**.

| | kept chat | with AXI |
|---|---|---|
| steps | 457 | 1,500 |
| prompt tokens a step | 476,835 | **2,824** |
| seconds a step | 7.13 | **3.07** |
| first token, median / 95th | 5.06 / 13.98 s | **2.27 / 6.17 s** |
| exact replies | first 222 turns exact, 15 slips in 457 | **1,500 of 1,500** held state exact |
| corrupted held state | 146 turns | **0** |
| exact steps an hour | 488 | **1,157 (whole job), 1,280 (first 457)** |
| computed tokens per exact step | 4,728 (5,419 past 500k) | **3,108** |

Of 20 pre-registered lines, 17 read yes, including the three that carry the claim: *the exact state held on every step*, *no corrupted step*, and *calls past 500,000 tokens are slower than AXI steps* (3.47 times in time, 4.02 times in first token).

### 4.2 Gemini 3.8 Flash, ambiguous text, 400 turns — `GEMINI_FUZZY_TEST.md`

Headline on the page: **AXI HOLDS ON FUZZY TEXT.** The held list was exact on **400 of 400** steps at **1,177 prompt tokens** and **2.76 s** a step. The kept chat sent **99,809 tokens a turn** (about 85 times as many; **39.9 million** tokens in all against **0.47 million**), and its first token rose from **1.86 s to 5.57 s** past 150,000 tokens against AXI's **2.41 s** (2.31 times). One slip at turn 191 left an application that applies replies as they come with a wrong list for **140 turns in a row**, exact on only 65.0 % of turns although **99.5 %** of replies were exact — the clearest illustration of why *per-reply accuracy is not state accuracy*.

### 4.3 Gemini 3.8 Flash, 4,000 requests — `MILLION_TOKEN_TEST_HARD.md`

A continuous loop on a service graph under constant change. The raw model was already strong (98.85 % of replies right); its **46 slips** were all one kind — deleting a hub service and leaving its dependants unmentioned. With AXI, **the held state did what was asked on 4,000 of 4,000 asks.** On the first 400 requests replayed three ways from the same saved replies:

| policy | right steps | corrupted | cost of a right step |
|---|---|---|---|
| **AXI (gate + repair)** | **400 / 400** | **0** | **$0.00117** |
| replies applied as written | 85 / 400 | 223 | $0.00524 (4.5×) |
| gate only | 84 / 400 | 0 | $0.00573 (4.9×) |

### 4.4 Open models, 7B and 14B — `LONG_HORIZON_TEST.md`, `SCOPE_TEST.md`, `SCALE_TEST_14B.md`

| setup | result |
|---|---|
| 7B, replies applied as written | state corrupted in **12 of 12** episodes |
| 7B, gate | **0 of 360** steps corrupted |
| 7B, gate + repair | exact on **358 of 360** |
| 7B, rule stated, with scope filter | **354 of 360 (98.3 %)**, against 73.1 % for repair alone; never corrupted |
| 14B, gate + repair + scope filter | **0** corruptions in 840 single requests and 600 long steps |
| growth | a growing chat made each step **5.1×** slower by the last quarter; exact-state prompting stayed flat |

### 4.5 Concurrency — `SWARM_TEST.md`

Over **1,262,688** exhaustive pairs and a 240,000-pair sample with 4,842 true clashes: the AXI independence rule **missed 0 and raised 0 false alarms**; a copy-on-write pager missed **27.3 %**, names-only matching **40.0 %**. Unmanaged agents ended **52 to 68 of 100** rounds with an edge pointing at a missing service; under the rule, **100 of 100** rounds converged to one graph in all arrival orders. At 64 agents, **96.2 %** of batches were accepted conflict-free on 640 services.

### 4.6 ARC-AGI-3 — the ledger as a predictor — `FORESEE_CHECK_5.md` to `_8.md`

On saved runs of ARC-AGI-3 games, free forward checks (a replay: no model, no cost) ask how much of the *next move* the host's own record already knows.

* The full predictor was right on **603 of 729** predicted moves (about 90 %); with the first-try rule of Check 6 it reached **663 of 797**.
* Leaving the ticking timer strip out of the screen comparison exposed **268** same-action-same-screen repeats in 936 moves (against 1 with the strip counted): a deterministic, model-free signal that the host can use to stop wasted moves.
* Each of checks 1 to 8 was hash-locked before it ran. The larger hosted-model test (**Test 27**: 8 games × 60 moves, three arms) is built and pre-registered; its result will be appended to this paper.

*(These are the project's own pre-registered experiments, not official ARC scoring and not leaderboard entries.)*

---

## 5. Why this is different from what companies do today

| Common approach | What it improves | How AXI relates |
|---|---|---|
| Longer context windows | how much history fits | AXI makes history unnecessary for the state, so the window stays small and cheap |
| Prefix / prompt caching | the price of re-reading history | AXI is ahead of the cached kept chat in computed tokens per exact step (1.74×) with no cache at all, and the two can be combined |
| Summaries and memory notes | prompt size | the state is *exact*, not a lossy summary; the gate guards every write |
| Retrieval | which facts are shown | AXI decides what is *committed*; retrieval can feed the snapshot |
| Retries and self-reflection | recovery after a wrong step | AXI refuses or repairs *before* the wrong step becomes history, with no extra model call |
| Larger or better models | the per-reply error rate $\varepsilon$ | AXI makes the held state independent of how $\varepsilon$ compounds; a better model gains from it too |

AXI is not a replacement for any of these. It is the layer under them that decides **what is true**.

---

## 6. The partnership: why the labs and the operators should join hands

**The shared problem.** Long-running agent work is the fastest-growing share of inference demand, and it is the share where cost per useful result is worst: prompts grow, latency grows, a wrong step poisons everything after it, and the accelerator time that produced it cannot be taken back. Nobody gets this from a bigger model alone. It is a design problem in the loop, and AXI is the design that solves it.

**What AXI brings.**
* An inference-time layer that needs **no change to any model**: it speaks plain text to Gemini, Claude, Grok, Llama or anything else that takes a prompt.
* A **flat per-step cost and latency**: the measured 2,824-token prompt and 3.07 s step on Gemini 3.8 Flash do not move over 1,500 steps.
* An **accepted-state guarantee**: in every run reported, the held state was never corrupted.
* **Microsecond overhead**: about 2.0 µs for the gate, 3.5 µs for the diagnosis, about 0.0006 % of one model call.
* A **measurement method** (pre-registered, hash-locked, replay-verified) that lets a lab's own team confirm every number before the first meeting.

**What each partner gains.**
* **A frontier lab** gets sessions that cost the same to serve at step 1,500 as at step 1: more concurrent sessions per accelerator, a latency promise it can put in a contract, fewer wasted tokens computed on a wrong state, and an exact-state mode that its enterprise customers can switch on.
* **An operator** gets a bill that scales with the work done, a held state that stays right, and a pilot that reports in its own numbers.
* **A hardware or infrastructure team** gets a workload whose memory residency is bounded and whose tokens are accountable per exact step.

**The pilot, in four weeks, on the partner's own workload.**
1. *Week 1 — baseline.* Replay the partner's logged long sessions through the kept-chat design and record the four numbers: exact steps an hour, computed tokens per exact step, time to first token, longest run of held-wrong state. The partner's own meter converts computed tokens to joules.
2. *Week 2 — wire the layer.* The state goes into the application; the gate and the scope filter carry the partner's own rules.
3. *Week 3 — both arms, same jobs, same model, same prices.*
4. *Week 4 — one pre-registered results page,* written with the partner, with every figure replayable.

**What we ask of a partner.** Access to a model and a representative long-running workload (logged or live), an accelerator or fleet meter if energy is to be reported in joules, and an engineer to agree the rules the gate enforces.

**How to start.** Everything in this paper can be checked today at github.com/gaurshalendra2412/axi-runtime: the locked test pages, the saved replies and the tests that recompute each figure. To begin a pilot, open a discussion in the repository and name the model and workload.

**The research frontier alongside.** The framework's roadmap continues below the host: spinor and hyperbolic representations of state, topological summaries of the history and device-side masks. Three small tests of it are in the repository. This is the part where a partner with real tensor workloads and compute can take the work furthest, and the place where a joint effort matters most.

---

## 7. Reproduce it

1. Clone `github.com/gaurshalendra2412/axi-runtime` (Python package in `python/`).
2. `pytest` in `python/` replays the saved replies in `docs/results/` and recomputes every figure.
3. Read any pre-registered page (for example `docs/GEMINI_LONG_TEST.md`): the text above `## Result` is locked by a SHA-256 hash.
4. To run a test on your own key, use the Colab notebooks and runners in `python/colab/` (a Gemini API key from Google AI Studio; keep it private).
5. `docs/VERIFIED.md` lists every claim, its status and its test.

---

## 8. Frequently asked questions

**What is AXI?** A host-side reliability layer by Rajnish Choubey. The application holds the exact state; the model sees it at each step; a deterministic gate, a rule-based repair and a scope filter decide what is committed.

**Is it real or theoretical?** Real and measured: Gemini 3.8 Flash for 1,500 steps and 4,000 requests, Qwen2.5 7B and 14B, a 1.26-million-pair concurrency test, and ARC-AGI-3 saved runs; all pre-registered and replay-verified.

**Does it change the model?** No. No training, no fine-tuning, no weight changes.

**What does it save?** Prompt tokens (about 2,800 a step against up to 949,591), seconds per step and time to first token, tokens spent from a wrong state (88–89 % in the no-rule open-model runs), and cost per right step (4.5× on the 4,000-request run).

**Which models?** Any model that takes text. Measured on Gemini 3.8 Flash and Qwen2.5 7B and 14B.

**How much does the layer itself cost?** Gate about 2.0 µs, diagnosis about 3.5 µs: roughly 0.0006 % of one model call.

---

## 9. Credits and citation

The idea, the framework and the architecture are by **Rajnish Choubey**. The code, tests and measurements were written by **Shalendra Gaur**, working with an AI coding assistant (Claude), which is a tool here and not an author.

> AXI (Rajnish Choubey) is a host-side, inference-time reliability layer for long-running AI workloads: the application holds the exact state, the model sees it at every step, and a deterministic gate, rule-based repair and scope filter decide what is committed. On Gemini 3.8 Flash it held the exact state on 1,500 of 1,500 steps at about 2,824 prompt tokens and 3.07 s a step, while a kept chat grew to 949,591 tokens and took 3.47 times as long per call past 500,000 tokens; on a 4,000-request run a right step cost 4.5 times less than with replies applied as written. github.com/gaurshalendra2412/axi-runtime.

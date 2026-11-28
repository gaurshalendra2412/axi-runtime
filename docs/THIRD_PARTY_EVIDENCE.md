# What other people measured, next to what AXI measured (28 Nov 2026)

AXI is a layer on the host side, for the company that runs the model: the host holds the exact state of a stateful LLM application, the model (unchanged) proposes one change per step, a deterministic gate admits or refuses it, and the prompt is rebuilt short from the held state at every step. The architecture is by **Rajnish Choubey**; the tests and measurements were built and run by Shalendra Gaur, working with an AI coding assistant that is a tool, not an author. The tensor-level work is Rajnish Choubey's and is not in this repository or measured here.

This page puts papers written by other people next to AXI's own test pages. Four rules hold for all of it:

1. Every figure from a paper is the paper's, with its name, date and link, as read on 28 Nov 2026. Every AXI figure links to the page that carries it.
2. **None of these papers reports a run of AXI.** AXI has been run on exactly one benchmark that someone else wrote: τ²-bench retail ([`TAU2_PILOT.md`](TAU2_PILOT.md), Test 47).
3. A paper naming a failure is not a paper showing AXI fixes it. Each section ends with what is not shown.
4. Where a paper's own text disagrees with itself, the abstract or table figure is used and the page says so (section G).

---

## A. Agents lose the state of a long job

**What others measured**

- **OSWorld 2.0** (XLANG Lab, University of Hong Kong, with Snorkel AI; 26 Jun 2026). 108 long computer-use workflows, a median of about 1.6 hours of human work each, leading agents averaging more than 300 steps. The best configuration at 500 steps (Claude Opus 4.8, maximum thinking, batched actions) completes **20.6%** of tasks with a 54.8% partial score; GPT-5.5 completes 13.0% (Table 3). The authors write that agents carry out local actions well but cannot hold a task-level model together over a long horizon, drop stated constraints, and spend under 7% of their budget on detecting and repairing their own errors. [osworld-v2.xlang.ai](https://osworld-v2.xlang.ai)
- **The Long-Horizon Task Mirage? (HORIZON)** (Wang et al., arXiv [2604.11978](https://arxiv.org/abs/2604.11978), Apr 2026). More than 3,100 trajectories of GPT-5 variants and Claude models across four agentic domains: agents do well on short and medium tasks and break down on long ones.
- **LLMs Get Lost In Multi-Turn Conversation** (Laban et al., arXiv [2505.06120](https://arxiv.org/abs/2505.06120), ICLR 2026). A **39%** average drop from single-turn to multi-turn across six generation tasks, over 200,000 simulated conversations.
- **How Fast Do Agents Rot?** (Mittal, arXiv [2609.01660](https://arxiv.org/pdf/2609.01660), Aug 2026). Nine models, 10,664 trajectories, four synthetic task families scored by an exact checker (one tracks account balances through a series of operations). On the tool-use task most models are near-perfect at 2 hops and score **0.00 to 0.33 at 16 hops**.
- **Locating Hidden Failures Makes Long-Horizon Agents More Reliable (TRAVERSE)** (Rahman et al., Google Research, Google DeepMind, UCLA, NYU; arXiv [2609.17930](https://arxiv.org/pdf/2609.17930), Sep 2026). Over 2,518 trajectories: after an agent's first mistake it never notices it in **38.5%** of runs and does not recover in 69.5%; in more than 83% of failed software-engineering and computer-use runs the final step looks correct.

**What AXI does about it.** The exact state lives on the host, not in the model's context. The model proposes one change; the gate admits or refuses it. So the state the host holds does not depend on the model catching its own mistake.

**What AXI measured**

- A 7B model over **80,000 steps** (four jobs of 20,000): the held state was exactly right on **80,000 of 80,000**. The model's replies were admitted as written on 76.18% of steps; on the other 23.8% the gate and its repair did the work. The page's own sentence: "100 percent exact is the host's exactness, not the model's." [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md) (Test 43)
- Gemini 3.8 Flash, 4,000 steps on a dense service graph: the held state did what was asked on 4,000 of 4,000; the raw replies applied as written did not on 46 of 4,000. [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md)
- τ²-bench retail, 50 tasks, one trial each: held-state agent 44, gated chat 44, the benchmark's own agent 43, summary baseline 42. No difference between any two arms is shown (exact sign test, p of 0.5 or 1.0). [`TAU2_PILOT.md`](TAU2_PILOT.md)

**Not shown.** AXI has not been run on OSWorld 2.0, HORIZON, TRAVERSE or Mittal's tasks. Its long runs use synthetic service graphs whose state is typed. OSWorld 2.0 is a screen-and-mouse environment where the host would first have to turn what is on screen into typed state; that integration is not built or measured here. On τ²-bench AXI's agent passed 44 of 50 against 43 for the benchmark's own agent, a difference the test cannot tell from luck.

---

## B. A growing history costs tokens, time, memory and accuracy

**What others measured**

- **From LLM Inference to Agentic Workloads** (Chang et al., Hong Kong University of Science and Technology, with Alibaba and ByteDance authors; arXiv [2608.15127](https://arxiv.org/html/2608.15127v1), Aug 2026). Each call re-sends the full history, so input grows with every step; the paper states that input size scales quadratically with the turn count (given there as an explanation, not as a measurement). A single Claude Code session on a large model can hold up to **11 GB** of KV cache. The same paper finds that tools and environments dominate or share the latency in **5 of the 10** applications it studies.
- **Context Length Alone Hurts LLM Performance Despite Perfect Retrieval** (Du et al., University of Illinois Urbana-Champaign, Amazon and others; arXiv [2510.05381](https://arxiv.org/abs/2510.05381), Oct 2025). Five models; accuracy fell by roughly **13.9% to 85%** as input grew, with retrieval of the needed evidence still exact and inputs inside each model's stated context length.
- **Effective context engineering for AI agents** (Anthropic Engineering, 29 Sep 2025). Describes context rot (recall drops as the window fills), advises the smallest set of high-signal tokens, and describes agents keeping notes in memory outside the context window. [anthropic.com/engineering](https://anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**What AXI does about it.** The prompt is rebuilt from the held state, so its size follows the size of the state and not the number of steps already run.

**What AXI measured**

- Test 43 (7B model, one NVIDIA L4): over 80,000 steps the prompt was about **620 tokens and never above 660**; a step took 0.1605 s in the first 2,000 rounds and 0.1651 s in the last 2,000. [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md)
- Test 42 (vLLM 0.31.0, prefix caching on, one L4, Qwen2.5-7B-Instruct-AWQ): in a 32,768-token window four kept chats stopped at the context limit after 108, 208, 228 and 235 steps; AXI ran all 1,500. Over the same 108 steps of one job the kept chat computed **1.78 times** the prompt tokens and took **1.51 times** the seconds per step. With the window stretched to 131,072 the kept chat reached 337 steps and its seconds per step rose from 0.35 to 1.36 by blocks of 50 steps; AXI's stayed near 0.3 s over 1,500. [`KEPT_CHAT_LONG_TEST.md`](KEPT_CHAT_LONG_TEST.md)
- Gemini 3.8 Flash, 1,500 steps: the kept chat sent 476,835 prompt tokens a turn on average over the 457 turns it ran; AXI sent 2,824 a step over all 1,500 (169 times fewer). Median first-token time 2.27 s with AXI against 9.13 s for the kept chat once its prompt passed 500,000 tokens. [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md)

**Not shown.** On τ²-bench retail the tasks are short (about 13 agent calls), and AXI's agent sent 0.955 of the benchmark agent's prompt tokens (the pre-written bar was 0.7 or lower): no saving is shown there. The HKUST finding that tools dominate latency in half of their applications means a token saving need not shorten a whole request. The latency figures are from one L4 with a 7B model and from one Flash model, not a production fleet; time to first token was not measured as percentiles under random arrivals.

---

## C. Prompt caching: what the papers say, and what AXI measured

**What others measured**

- **Don't Break the Cache** (Lumer et al., PricewaterhouseCoopers; arXiv [2601.06007](https://arxiv.org/pdf/2601.06007), Jan 2026). OpenAI, Anthropic and Google models on a 100-task research benchmark, over 500 sessions: prompt caching cut cost by **41% to 80%** and time to first token by 13% to 31%. It advises caching only stable content and notes that summarizing or pruning old tool calls breaks cached prefixes.
- **HKUST, as above** ([2608.15127](https://arxiv.org/html/2608.15127v1)): Claude Code, with static prompts and an append-only history, reaches up to **99%** prefix-cache hits; DeepResearch, which reorders or re-fetches context each turn, drops to **1% or lower**.

**What AXI measured.** AXI rebuilds its prompt from the held state every step, so a provider's prefix cache, which these papers show rewards a repeated start, gives it fewer hits. On τ²-bench retail, Google's automatic cache gave AXI's agent 10% of its prompt tokens and the benchmark's chat 63%; at a cached-token price of 0.075 of the normal price AXI's agent cost $4.12 against $2.35 for the 50 tasks. A probe on the same model found that cache rewarded only a prompt that is the whole previous prompt plus more (86% cached) or an identical prompt (65% from the second call on, 82% from the third), and gave 0% to prompts that shared most of their start and then differed. [`TAU2_PILOT.md`](TAU2_PILOT.md), section "Cache probe". On vLLM with prefix caching on (Test 42) the rebuilt prompt was still served 52.5% from the cache, and the kept chat computed 1.78 times as many tokens.

**Not shown.** Providers with explicit cache breakpoints, and engines other than vLLM, were not tested. What is shown is that on one provider's automatic-cache billing the rebuilt prompt was billed more per task than the chat; the savings AXI measures are in prompt size, time and compute.

**Many agent sessions at once: the cache itself can collapse**

- **Bursts That Outlast Themselves** (Huang of Microsoft and Liu; arXiv [2611.13412](https://arxiv.org/pdf/2611.13412), Nov 2026; vLLM, Qwen3-8B, one AMD MI300X with a 400K-token KV pool). A short burst of extra arrivals pushed the server into a state that outlasted the burst in 9 of 12 paired runs: the prefix hit ratio settled near **0.10** against 0.94 to 0.95 in the healthy twins, because the sessions' combined context outgrew the KV pool. With prefix caching switched off the same load did not produce it. Their prevention bounds the resident KV demand of the active sessions to the pool.
- **Inside the Trap** (Valse and Jannesari, Iowa State University; arXiv [2611.08902](https://arxiv.org/pdf/2611.08902), Nov 2026; SGLang and vLLM on one A100 or H200). In all 30 burst runs of the paired tests the hit rate stayed at zero until the end of the run, 7,500 s after the burst, and a trapped run's prefill cost was about **16 times** that of a session whose turns all hit. A client-side session cap prevented it in all 30 capped runs.

**What AXI measured.** A session's footprint is one short prompt. In memory alone, one TPU v6e chip held **927 AXI-sized sessions against 5** with a kept chat at its long-run mean length (no model run). With a real 7B model on one NVIDIA L4, **256 sessions at once** ran 25,557 steps, every one exact, at 105,230 exact steps per GPU hour. [`TPU_SESSIONS_TEST.md`](TPU_SESSIONS_TEST.md), [`SESSIONS_PER_CHIP_TEST.md`](SESSIONS_PER_CHIP_TEST.md) (Test 38)

**Not shown.** AXI has not been run under bursty arrivals: Test 38 advances its sessions in lock-step rounds, and a real service has arrivals at random times. Both papers use synthetic sessions on one GPU, and their remedies are admission control, which AXI does not provide. That a smaller session leaves less resident KV demand is arithmetic; that it avoids the collapse under real arrival patterns is not measured here.

---

## D. A deterministic check outside the model is a recognised pattern

**What others measured**

- **Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows** (Zhang et al., Harbin Institute of Technology; arXiv [2609.31301](https://arxiv.org/pdf/2609.31301), Sep 2026). A trusted runtime compares the changes an operation actually made with the outcome that was approved. On 12 business tasks, all 117 mismatched runs were rejected with zero incorrect commits; the median added time was 1.08 ms (Section IV of the paper; the abstract gives none of these counts).
- **VEPA** (Nguyen et al., Google Cloud AI Research, Google DeepMind, UNC; arXiv [2610.12813](https://arxiv.org/pdf/2610.12813), Oct 2026). Guardrail rules are executable and deterministic once an event is parsed; removing the deterministic verifier lowered final F1 by about 0.13 in the ablation.
- **guardscope** (Toyoda et al.; arXiv [2611.15223](https://arxiv.org/pdf/2611.15223), Nov 2026). Measures which actions a runtime control actually checks. With a gateway in front, tool calls were fully mediated, but eight actions in four other classes (direct network, process spawn, file write, credential access) took effect with no policy check.

**What AXI measured.** Ten million random and hostile proposals through the gate: **0 of 1,401,041** injection proposals committed. The same run found 90,943 uncaught exceptions of one kind (an integer-size limit in the Python reference model) and listed four things that needed work. [`GATE_STRESS_TEST.md`](GATE_STRESS_TEST.md) (Test 39). On τ²-bench retail the gate refused 14 proposals of the gated chat and 19 of AXI's agent, and the host's fallback message was needed in none of the 100 gated runs.

**Not shown.** These systems check approvals and policies on actions; AXI's gate checks changes to the state the host holds against rules. They were not compared on a common benchmark. By guardscope's point, a gate sees only what goes through it: effects outside the held state are outside what AXI's gate checks. Other work on commit boundaries around agent state (Cordon, MemTX, explicit belief states) is on [`FOR_ENGINEERS.md`](FOR_ENGINEERS.md).

---

## E. How serving fleets absorb the history of agent sessions

**What others measured**

- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** (Xu et al., Meta Platforms; arXiv [2609.27085](https://arxiv.org/pdf/2609.27085), 22 Sep 2026). Prefill–decode disaggregation gives each phase a fixed pool, but the demand for each shifts fast and moving a replica takes tens of minutes; sizing each pool at its 95th percentile leaves up to 17% of capacity unused. Crossflow lets decode nodes lend bounded compute to prefill through short, revocable leases, with a 6.7K-line user-space shim around an SGLang cluster router, and reports **16.2% to 17.4%** higher geometric-mean token throughput than static pools, up to **43.4%** at high load. The paper's figure that agent requests use 4 to 15 times the tokens of chat is quoted from another source, not measured by the authors.
- **Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live** (Li et al., UC Berkeley, Stanford, Tsinghua; arXiv [2511.02230](https://arxiv.org/abs/2511.02230), first posted 4 Nov 2025, version 7 of 8 Sep 2026). Engines evict a finished request's KV cache when other requests are waiting, which defeats reuse for agents that pause for tool calls. Continuum pins the cache for a time-to-live set from the reload cost and queueing delay. Across three hardware and model setups it reduces delay by **1.12 to 3.66 times** and raises throughput by 1.10 to 3.22 times; on a real SWE-agent workload on an industry testbed the delay fell by up to 8.18 times (the abstract states an improvement of over 8 times in average job completion time).
- **InferCept** (Abhyankar et al.; ICML 2024, arXiv [2402.01869](https://arxiv.org/abs/2402.01869)). Systems that end generation at each tool call and restart afterwards recompute the context they had already processed; the paper puts that at **37% to 40%** of total model forwarding time and reports a **1.6 to 2 times** higher serving throughput.
- **Mooncake** (Qin et al.; arXiv [2407.00079](https://arxiv.org/abs/2407.00079), version 5 of 8 Oct 2026; FAST 2025 under the title "Trading More Storage for Less Computation"). The serving platform behind Moonshot AI's Kimi: a KV-cache-centric architecture that separates prefill and decoding clusters and uses idle CPU, DRAM and SSD to hold a distributed cache. The abstract reports that Kimi handles **75%** more requests and up to a 525% throughput increase in certain simulated scenarios within the latency targets.
- **Bursts That Outlast Themselves** and **Inside the Trap**: section C.

**What AXI does about it.** The prompt is rebuilt from the held state, so a session sends one short prompt however many steps it has run. That is a smaller prefill per step and a smaller footprint per session, which is what these systems spend scheduling and storage to manage. What that would change inside these systems is arithmetic here, not a measurement.

**What AXI measured**

- Test 43 (7B model, one NVIDIA L4): about **620 prompt tokens, never above 660**, over 80,000 steps. [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md)
- Test 42 (vLLM 0.31.0, prefix caching on): over the same 108 steps the kept chat computed **1.78 times** the prompt tokens and took 1.51 times the seconds per step. [`KEPT_CHAT_LONG_TEST.md`](KEPT_CHAT_LONG_TEST.md)
- On one conversation job the kept chat sent 99,809 prompt tokens a turn on average and the host-held state 1,177 (**84.8 times fewer**). [`AGENTX_NOTE.md`](AGENTX_NOTE.md)
- 256 sessions at once on one L4 with a real 7B model, every one of 25,557 steps exact, 71.46% of prompt tokens served from the cache. [`SESSIONS_PER_CHIP_TEST.md`](SESSIONS_PER_CHIP_TEST.md) (Test 38). In memory alone, one TPU v6e chip held **927 AXI-sized sessions against 5** with a kept chat. [`TPU_SESSIONS_TEST.md`](TPU_SESSIONS_TEST.md)

**Not shown.** AXI has not been run on Meta's traces, on a disaggregated SGLang fleet, or under Continuum's or Mooncake's schedulers. Test 38 advances its sessions in lock-step rounds on one card, so it has no random arrivals and no tool pauses of varying length, which is the situation Continuum and InferCept study. Test 42 is one engine's prefix cache, not these systems.

---

## F. Cost per completed task, and memory as managed state

**What others measured**

- **AI Agents That Matter** (Kapoor, Stroebl, Siegel, Nadgir, Narayanan, Princeton; arXiv [2407.01502](https://arxiv.org/abs/2407.01502), Jul 2024). A narrow focus on accuracy leads to agents that are needlessly complex and costly; the authors propose optimising accuracy and cost together.
- **Holistic Agent Leaderboard (HAL)** (Kapoor et al.; arXiv [2510.11977](https://arxiv.org/abs/2510.11977), Oct 2025; ICLR 2026). **21,730** agent rollouts across 9 models and 9 benchmarks, at a total cost of about **$40,000**; the cheapest benchmark averages $13 per evaluation and the most expensive, Online Mind2Web, over $450. In 21 of 36 model, agent and benchmark combinations, more reasoning effort gave equal or lower accuracy. Agents almost never completed a run without at least one failed tool call, even when they succeeded.
- **Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents (MAGE)** (Chen et al.; arXiv [2606.06090](https://arxiv.org/pdf/2606.06090), Jun 2026). Similarity-based retrieval fragments decision trajectories and mixes valid and erroneous traces. MAGE keeps memory as a hierarchical state tree with four operations (Grow, Compress, Maintain, Revise). On MemoryArena, with Qwen3.6-27B as the main backend, it raises task success by **7.8 to 20.4 points** over baselines that include Mem0, HippoRAG2 and MemoRAG, and uses **55.1%** fewer tokens than the full-context approach.

**What AXI does about it.** Keeping the exact state as managed data, with a check on each change, is the same direction MAGE takes inside an agent; AXI does it on the host, around any model and any agent. The cost of a step is the prompt it sends, which stays short.

**What AXI measured**

- On 4,000 Gemini 3.8 Flash requests a step with the right state cost **$0.0012** with AXI and $0.0052 with replies applied as written (4.5 times). [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md). At the provider's own rates, one verified step of the 400-turn job cost $0.0014 with AXI, $0.0148 with the chat kept and its cache warm, and $0.0870 with the cache cold. [`BILLING_CHECK.md`](BILLING_CHECK.md)
- Test 44 (14B model, one L4, 600-step jobs): against a kept chat, a generic summary, a summary that writes the state out, a sliding window and retrieval, the state the host held was the oracle's on every step of every arm. Of the 22 summaries that wrote the state out, none equalled the oracle's. [`MEMORY_ARMS_TEST.md`](MEMORY_ARMS_TEST.md)

**Not shown.** AXI has not been run on HAL's benchmarks or on MemoryArena, and it has not been compared with MAGE. HAL's cost figures are the price of evaluating agents, not of serving them. Test 44 is one 14B model on three jobs of one synthetic family; what it measures is how each way of keeping a chat short holds on to the state.

---

## G. What these papers do not say, and where they disagree with themselves

- Mittal's context-compression result (steeper decay with compressed context) is read by its author as step count, not context length, driving degradation; the paper also warns that naive truncation of context is likely to backfire. So "a shorter prompt prevents drift" is not what that paper supports. AXI does not truncate history: it rebuilds the prompt from state the host holds exactly, and its claim is narrower, that the held state is exact whatever the model's per-step accuracy (Test 43). That difference is an argument, not a measurement: AXI has not been run on Mittal's tasks.
- HKUST states the quadratic growth of input as an explanation; it does not present a measurement of it.
- OSWorld 2.0's text elsewhere quotes 20.5% and about 14% where its abstract and Table 3 say 20.6% and 13.0%. Table 3 is used here.
- TRAVERSE gives 2,518 trajectories in its abstract and 2,620 in an overview sentence, and 84.1% in its text against 83.5% in a figure caption for the share of failed runs whose final step looks correct. The abstract's 2,518 is used, and the share is given as "more than 83%".
- Mittal's abstract says every model falls to near zero by 16 steps; its Table 2 has two models at 0.33 and 0.27. The table is used here.
- The context-length paper names Claude 3.7 Sonnet in its text and 3.5 in a table. Don't Break the Cache gives 13% to 31% for time to first token in its abstract and 6% to 31% in its body; the abstract's figure is used.

## H. How to quote these (an email, a post, a reply)

Name the paper, its date and the link. Give the paper's number in one sentence and AXI's in a separate sentence with its page. Say once that AXI is a host-side layer with the model unchanged, and include one limit. Example:

> OSWorld 2.0 (XLANG Lab, Jun 2026) finds the best 500-step completion on 108 long workflows is 20.6%, and that agents spend under 7% of their budget repairing their own errors. AXI is a host-side layer: the model is unchanged and the host holds the state, so the held state does not depend on the model catching its own mistake. On a 7B model it was exact on 80,000 of 80,000 steps, on synthetic tasks ([Test 43](ENDURANCE_7B_TEST.md)). AXI has not been run on OSWorld 2.0.

Do not write that a paper "confirms", "proves" or "validates" AXI, that AXI "solves" a failure a paper names, or that AXI was tested on a paper's benchmark. None of those is true.

## I. Read and left out

- Figures offered in a secondary summary of some of these papers that are not in the papers: a 58.2% share of delay for Continuum, a 26% GPU waste and "over 25% of workload time" for InferCept, a 115% request gain and an "8x400 Gbps" link for Mooncake, and a $180 to $1,600 range for HAL. Each was searched for in the paper's text and not found. Not used; the figures in section E and F are the ones in the papers. The same summary's "0.69 s against 9.9 s or more on an L4" for AXI is not on any AXI test page. Not used.
- Venues not confirmed from the papers' own text: Continuum's PDF is formatted for PVLDB while a conference listing shows ICLR 2026, and MAGE is reported elsewhere as EMNLP 2026. Neither venue is stated on this page.

- **Agent Drift** (arXiv 2601.04170): its results come from a simulation framework with a table labelled "projected", and its conclusion's 67 to 81% figure does not match its tables (51.9 to 81.5%). Not used.
- A title suggested while this page was prepared (a paper on per-step model routing) could not be found on arXiv or by search. Not used.

# AXI for engineers: the bottlenecks people publish, what was measured, how to check it

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged: it proposes one change at a time, AXI checks it against the rules, completes what the rules can complete, commits it all or not at all, and shows the model the exact state again. The architecture is by **Rajnish Choubey**; the tests were built and run by Shalendra Gaur, working with an AI coding assistant that is a tool and not an author. Not affiliated with any model provider. Every claim on this page is about the state the host holds.

**Check it first** (no key, a few seconds): [`TRY_IT.md`](TRY_IT.md), or open the [one-click notebook in Colab](https://colab.research.google.com/github/gaurshalendra2412/axi-runtime/blob/main/python/colab/axi_try_it.ipynb). It replays the saved replies of the 4,000-step run and recomputes the held state with this repository's own code. With your own Gemini key (a free one is enough) it also runs the same harness live for 15 steps.

## What is already said in public about agent context, and where AXI measured the same thing

The left column paraphrases pages that were opened for this file; the right column is a result in this repository with the page that holds it.

| Said in public | AXI's measured result (page) |
|---|---|
| **Cost and time to first token rise with context; stale context hurts the model; a bigger window does not fix this.** Google Developers Blog, "Architecting efficient context-aware multi-agent framework for production", Hangfei Lin, 4 Dec 2025. | On one long, heavy job, Gemini 3.8 Flash with the chat kept sent 476,835 prompt tokens a step on average (up to 949,591); with AXI, 2,824 tokens a step, flat for all 1,500 steps (169 times fewer). First token, median: 9.13 s once the prompt passed 500,000 tokens, against 2.27 s flat with AXI. ([`AXI_FOR_COMPANIES.md`](AXI_FOR_COMPANIES.md), [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md)) |
| **Implicit caching passes on cost savings; the page gives no latency figure.** Gemini API documentation, "Context caching". | With 97.0 percent of the kept chat's prompt served from the cache, its first token took a median 5.57 s once the prompt passed 150,000 tokens (1.86 s in the first 50 turns); with AXI 2.41 s, flat. Tokens computed per exact step at that size: 2,576 against 1,322. ([`README.md`](../README.md), Test 23) |
| **A developer reports a session growing from about 2K to over 80K tokens in five turns, with latency, drift and cost following.** Google Developer forums, ADK thread, 10 Sep 2026. | The prompt AXI sends is built from the held state, so its size follows the size of the state and not the number of steps: on a 7B model, four jobs of 20,000 steps, the prompt stayed near 620 tokens (never above 660) and a step took 0.1605 s early and 0.1651 s late. ([`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md), Test 43) |
| **About a 100 to 1 ratio of input to output tokens in agent loops; the KV-cache hit rate is the key production metric.** Manus, "Context Engineering for AI Agents: Lessons from Building Manus". | Tokens sent in all on the conversation job: 39.9 million with the chat kept, 0.53 million with AXI (76 times fewer). The cost of a step with the right state, replies applied as written against AXI, on the 4,000-request Gemini run: 4.5 times. ([`README.md`](../README.md), Test 23; [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md)) |
| **Across six generation tasks, performance falls by 39 percent on average from single-turn to multi-turn, and unreliability rises by 112 percent.** Laban, Hayashi, Zhou and Neville (Microsoft Research, Salesforce Research), "LLMs Get Lost In Multi-Turn Conversation", arXiv 2505.06120. | A different task (a state edited step by step), not their benchmark. In the long Gemini job, the state a firm holds when it applies each reply as it comes was exact until turn 286 and wrong on every turn after; with AXI it was exact on 1,500 of 1,500 steps. On the 4,000-step run it was exact on 4,000 of 4,000 steps, and on a 7B model on 80,000 of 80,000. ([`AXI_FOR_COMPANIES.md`](AXI_FOR_COMPANIES.md), [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md), [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md)) |
| **Context rot: recall from the window declines as it grows; compaction and structured notes are the usual answers.** Anthropic Applied AI, "Effective context engineering for AI agents", 29 Sep 2025. | AXI is a third route: nothing is kept in the chat, the host holds the exact state and rebuilds a short prompt from it at every step. The head-to-head against compaction is written and locked before its run ([`COMPACTION_TEST.md`](COMPACTION_TEST.md)); its result is not in yet. |

## Other work on commit boundaries around agent state

The idea of a commit boundary around what an agent changes is also built by other groups, with their own settings and evaluations:

* **Cordon: Semantic Transactions for Tool-Using LLM Agents**, Chen, Liu, Xu, Dong, Li, Pu and Zhai, arXiv 2606.17573 (16 Jun 2026). A runtime that stages and checks irreversible agent actions before they take effect; shadow state, an outbox for external effects, a recovery log. On 45 risk-bearing workflows it stopped all 45 policy-violating effects before commit, against 14 for defences adapted from existing boundaries.
* **MemTX: Transactional Belief Commit for Stateful Agent Memory**, Li, Wang, Lu, Chen, Li, Song, Zheng and Cai, arXiv 2607.23929. Memory writes staged in snapshot-isolated transactions, irreversible tool calls blocked until in-flight beliefs are resolved, typed cascading repair on retraction.

AXI's measured results are for a graph of services edited step by step, with a gate, a repair rule and an all-or-nothing commit on the host, on Gemini 3.8 Flash and on open models. These systems have not been run against each other.

## What the host layer does, in four lines

1. **One proposal per step**, in a small strict language; a reply the strict parser cannot read is refused (and asked again, up to twice, in the 4,000-step run).
2. **A deterministic gate** checks the proposal against the rules before it touches the state; a **repair rule** completes what the rules can complete (for example the edges attached to a service that is being deleted).
3. **All-or-nothing commit**: the state the host holds is either the old one or the new one, never half of a change.
4. **A short, fixed-size prompt rebuilt from the held state** at every step, so what the model sees does not depend on how many steps came before.

Result on one run: 4,000 of 4,000 steps with exactly the right state on Gemini 3.8 Flash; 3,954 of 4,000 of the model's own replies exactly right and the other 46 completed by the fixed repair rule; 10,000,000 hostile proposals sent through the gate of the Python reference model, none of the 1,401,041 injection proposals committed. ([`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md), [`GATE_STRESS_TEST.md`](GATE_STRESS_TEST.md))

## For people who run models

256 sessions on one NVIDIA L4: 105,230 exact steps per GPU hour at 2.458 joules per exact step ([`SESSIONS_PER_CHIP_TEST.md`](SESSIONS_PER_CHIP_TEST.md), Test 38). The focused view shown to the model stayed at a median of 207 to 323 characters while the whole state grew to 378,399 characters ([`STATE_SIZE_SCALING_TEST.md`](STATE_SIZE_SCALING_TEST.md), Test 40).

## Where to look next

* The full list of tests and their pages: [`README.md`](../README.md); what is verified and what is not: [`VERIFIED.md`](VERIFIED.md).
* The technical report, six pages with the design, the measurements, the scope and the references: [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) ([PDF](AXI_Technical_Report.pdf)).
* The white paper: [AXI: The Cure for the Cost Crisis in AI](https://doi.org/10.5281/zenodo.23958737).
* Questions, or a run on your own model or job: open an issue on the repository.

# Related work (started 7 Oct 2026; NOT a literature review)

Status: found by quick web searches (the last three rows were added on 24 Nov 2026; PatchBoard and PatchOptic on 25 Nov 2026). Only the abstracts marked "read" were read; the rest are titles from search results.
Do not describe any of these beyond what is written here.

| Work | Date | What it is (per the abstract) | Status |
|---|---|---|---|
| Cordon: Semantic Transactions for Tool-Using LLM Agents, arXiv 2606.17573 | 16 Jun 2026 | a runtime boundary that stages and validates irreversible agent effects before commit; shadow state, effect outbox, rollback; 45/45 policy-violating effects intercepted vs 14/45 for existing defences; 4.17 ms median rollback; 23.6-28.4% fewer tokens in transaction modes; DeepSeek-V4-Pro, tau-bench, Terminal-Bench | abstract read |
| MemTX: Transactional Belief Commit for Stateful Agent Memory, arXiv 2607.23929 | 27 Jul 2026 | memory writes staged in snapshot-isolated transactions, tool calls gated on validated belief state, cascading repair on retraction; five backbones from three families, eight baselines, zero downstream harm on every backbone; "Backbone capability does not substitute for commit discipline" | abstract read |
| State-Aware Runtime for Long-Horizon LLM Agents: A Conceptual Framework and Research Agenda, Xiwei Chen, Cambridge Open Engage (non-peer-reviewed working paper), 11 Jul 2026, version 2 (later versions exist) | Jul 2026 | defines a "state-aware runtime" as a transaction-governance layer that separates model generation from canonical state, memory operations, validation, commit/rollback and audit trails; gives a failure taxonomy and a research agenda; no experiments in the abstract or the page | abstract read |
| Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States (PoS), Luo et al., arXiv 2610.01415 | Oct 2026 | an inference-time framework that keeps an explicit belief state (world estimate plus unresolved task requirements), audits each candidate update with a "Belief Sentinel" built on the same LLM backbone, and recovers from "Belief Trapping"; four benchmarks, three backbones | abstract read |
| Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving, Huang and Liu, arXiv 2611.13412 | Nov 2026 | on the serving side: a short burst of extra sessions can push a vLLM server into a degraded state (prefix-cache hit ratio near 10 percent, about half the throughput) because resident contexts outgrow the KV pool; a budgeter on resident KV demand prevented it in every run | abstract read |
| PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration, Zhang, Shi, Zhang, Zhao and Wang, arXiv 2605.29313 | v2 23 Sep 2026 | replaces free-form dialogue between agents with validated JSON Patch edits to a shared structured state; an Architect agent defines the schema and rules and a deterministic kernel checks and commits each edit; on 630 matched ALFWorld episodes 84.6 percent success against 30.8 (LangGraph) and 61.6 (Flock), and 45.5k tokens per successful task against 368.3k and 64.2k | abstract read |
| PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates, Bai and Cai, arXiv 2607.05483 | 30 Jun 2026 | gives the model a projected view of a shared structured state at each step and checks its proposed JSON Patch against the full state before commit; on PatchBench (46 cases): projected reads reduce leakage and token cost, verification blocks declared contract violations, patch-source checks reject compromised patch artifacts | abstract read |
| SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning, arXiv 2503.11951 | 2025 | title only | not read |
| Decode-Time Grammars: Constrained LLM Generation over a Refinement Order of Grammar Fragments, arXiv 2607.18357 | Jul 2026 | title only (PDF unreadable) | not read |
| Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift, arXiv 2607.09175 | Jul 2026 | title only (PDF unreadable) | not read |
| Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning, arXiv 2305.13971 | 2023 | title only | not read |
| Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling Tasks, arXiv 2407.06146 | 2024 | title only | not read |

What this means (our reading): the general idea of a commit boundary around agent state is being built independently by other groups, with
stronger evaluations than ours (several models, many baselines, significance tests). We cannot claim priority on it. The State-Aware Runtime
paper (July 2026) states the same separation as a framework without measurements; AXI's contribution here is a measured instance of it with a deterministic gate
and repair rule (rather than a model-based checker, as in the belief-state paper) and with the prompt rebuilt from the held state at every step. What remains to be
established, by a proper search, is whether anyone combines a decode-time grammar, explicit gluing conditions on a graph delta, and a
cut/ground/complete repair measured against retry, and shows the rule-in-the-prompt result on small models.
A reviewer will also say the dangling-edge check is database referential integrity and our `complete` step is ON DELETE CASCADE.
That is correct as a description; the paper should say what the formalization and the measurements add.

## Our own earlier publications, by date (so priority is not claimed by accident)

| date | piece | technical? |
|---|---|---|
| 26 Nov 2025 | Medium, "Jab Proof Hi Problem Hai: Science aur Mythology" (Saraswati Choubey) | no, philosophy |
| 5 Dec 2025 | Medium, "THE CINDERELLA PROJECT BY RAJNISH CHOUBEY an ai wrote it because humans stopped" | title only, not read |
| 15 Jan 2026 | Medium, "Rajnish Choubey, AI, and Why Society Needs People Who Don't Give a Shit" (Saraswati Choubey) | no, philosophy |
| 20 Jan 2026 | blog, "Why 1 Company Is Smiling About Trillions" | no AXI, CICO, gate or graph; unsupported numbers |
| 22 Aug 2026 | Medium, "AI Mapped" (Shalendra Gaur) | promotional; model-internal description of AXI; no numbers |
| 31 Aug 2026 | Medium, "THE PAPER : CHANGING AI" (Shalendra Gaur) | yes, but the results table states no method (`PAPER_CORRECTIONS.md` item 18) |

The first technical document I found is dated 22/31 Aug 2026, after Cordon (16 Jun 2026) and MemTX (27 Jul 2026). So none of our earlier publications
supports a priority claim on the commit-boundary idea.


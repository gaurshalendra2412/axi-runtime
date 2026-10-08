# Related work (started 7 Oct 2026; NOT a literature review)

Status: found by two quick web searches. Only the abstracts marked "read" were read; the rest are titles from search results.
Do not describe any of these beyond what is written here.

| Work | Date | What it is (per the abstract) | Status |
|---|---|---|---|
| Cordon: Semantic Transactions for Tool-Using LLM Agents, arXiv 2606.17573 | 16 Jun 2026 | a runtime boundary that stages and validates irreversible agent effects before commit; shadow state, effect outbox, rollback; 45/45 policy-violating effects intercepted vs 14/45 for existing defences; 4.17 ms median rollback; 23.6-28.4% fewer tokens in transaction modes; DeepSeek-V4-Pro, tau-bench, Terminal-Bench | abstract read |
| MemTX: Transactional Belief Commit for Stateful Agent Memory, arXiv 2607.23929 | 27 Jul 2026 | memory writes staged in snapshot-isolated transactions, tool calls gated on validated belief state, cascading repair on retraction; five backbones from three families, eight baselines, zero downstream harm on every backbone; "Backbone capability does not substitute for commit discipline" | abstract read |
| SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning, arXiv 2503.11951 | 2025 | title only | not read |
| Decode-Time Grammars: Constrained LLM Generation over a Refinement Order of Grammar Fragments, arXiv 2607.18357 | Jul 2026 | title only (PDF unreadable) | not read |
| Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift, arXiv 2607.09175 | Jul 2026 | title only (PDF unreadable) | not read |
| Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning, arXiv 2305.13971 | 2023 | title only | not read |
| Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling Tasks, arXiv 2407.06146 | 2024 | title only | not read |

What this means (our reading): the general idea of a commit boundary around agent state is being built independently by other groups, with
stronger evaluations than ours (several models, many baselines, significance tests). We cannot claim priority on it. What remains to be
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


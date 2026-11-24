---
title: "AXI Runtime: a host-side layer between a model and the state it edits"
subtitle: "Technical report: design, and measurements on Gemini 3.8 Flash and open models"
author:
  - "Rajnish Choubey (architecture)"
  - "Shalendra Gaur (tests and measurements)"
date: "November 2026"
---

**Abstract.** AXI is a layer on the host side, between a model and the state it edits. The model is unchanged. At every step it proposes one change in a small strict language; the host checks the proposal against the rules, completes what the rules can complete, commits it all or not at all, and shows the model the exact state again in a short prompt rebuilt from what the host holds. We report what this measured. On Gemini 3.8 Flash over 4,000 consecutive steps of a dense service-graph job, the state held by the host was exactly the intended state on 4,000 of 4,000 steps, while the same model's replies applied as written left the right state on 85 of the first 400 steps; a step with the right state cost 4.5 times as much when the replies were applied as written. On a long job of 1,500 steps the prompt AXI sent was 2,824 tokens a step against 476,835 on average for the same model with the conversation kept (169 times fewer), and the time to first token stayed at 2.27 s where the kept conversation reached 9.13 s past 500,000 tokens of prompt. On a 7B open model, four jobs of 20,000 steps were exact on 80,000 of 80,000 steps with a prompt that stayed near 620 tokens. Each figure comes from a page written before its run, with the saved files and code in the repository, and the first can be checked in seconds with no key.

**Keywords:** LLM agents, agent state, context engineering, context window, KV cache, prompt caching, time to first token, long-horizon agents, structured output, deterministic validation, atomic commit, cost per correct step.

Repository: <https://github.com/gaurshalendra2412/axi-runtime>. Check it yourself: `docs/TRY_IT.md`. White paper: doi:10.5281/zenodo.23958737.

# 1. The problem, as it is stated in public

Teams that build agents on hosted models describe the same cluster of problems. We quote only pages we opened.

* **Cost and time to first token rise with the size of the context; stale context hurts; a larger window does not remove this.** Lin, "Architecting efficient context-aware multi-agent framework for production", Google Developers Blog, 4 December 2025 [1].
* **Caching passes on a saving in cost, not a latency figure.** The Gemini API documentation on context caching describes implicit caching and the saving passed on; it gives no latency figure [2].
* **Sessions grow fast.** A developer on the Google developer forums reports an agent session growing from about 2,000 to over 80,000 tokens in five turns, with latency, drift and cost following [3].
* **Agent loops are input-heavy.** The Manus team reports a ratio of about 100 to 1 between input and output tokens in agent loops, and calls the KV-cache hit rate the most important production metric [4].
* **Multi-turn reliability falls.** Laban et al. measure, across six generation tasks, a 39 percent average drop from single-turn to multi-turn performance and a 112 percent rise in unreliability [5].
* **Context rot.** Anthropic's applied-AI team describes recall declining as the window grows, and names compaction and structured notes as the usual answers [6].

These are statements about conversations that carry their own history. AXI takes a different route: it does not carry the history. The state lives on the host, and the prompt is rebuilt from it.

# 2. The design

A step has four parts.

1. **One proposal.** The model is shown the exact state (or, for a large state, a focused view of it) and one request, and replies with one change in a small strict language. A reply the strict parser cannot read is refused and asked again, up to twice in the 4,000-step run.
2. **A deterministic gate.** The proposal is checked against the rules before it touches the state. The gate is ordinary code; it never calls the model.
3. **A repair rule** completes what the rules can complete. In the service-graph job, deleting a service also deletes the edges attached to it.
4. **An all-or-nothing commit.** The state the host holds is either the old state or the new one, never half of a change. The next prompt is built from the committed state.

Because the prompt is built from the state and not from the steps taken, its size follows the size of the state, not the number of steps. Nothing about the model is changed: it is the same model, with the same weights and the same settings, in every arm of every comparison below.

In the code the policies are named `scope_m7` (AXI: scope filter, gate, repair, commit), `blind` (replies applied as written), `gate` (applied only if the gate accepts, otherwise nothing) and `repair_m6` (the repair alone, without the scope filter).

# 3. Measurements

Every number below is about the state the host holds and is taken from a page in the repository that was written before its run. Each page names its saved files and the test that recomputes them, records its own scope, and says which of its pre-written reading lines read NO. For Tests 38 to 40 and 42 to 45 the raw per-step files stayed on the machines that ran them; the compact files in `docs/results` carry the figures and the sha256 of the raw files.

## 3.1 Gemini 3.8 Flash, 4,000 steps, dense job, strict rules

*Page: `docs/MILLION_TOKEN_TEST_HARD.md`. Written 13 October 2026, before any call.*

The job is a service graph of up to 24 services and 48 edges (17 to 22 services on average), edited by requests that each name their targets by coordinate, including deletes of services that have four to eight dependants. The rule that dependants must go too is not stated to the model. One run, one stream seed (20261031), sampled replies at the model's default temperature and thinking, 4,000 steps, 3,661,209 tokens as the service counted them. The reading rules were fixed before the run.

| | Result |
|---|---|
| Held state exactly the intended state | **4,000 of 4,000 steps**; largest distance from the oracle 0; no state ever malformed |
| Replies with exactly the ideal items | 3,954 of 4,000 (98.85 percent); 20 of these had the items in another order |
| Replies completed by the repair rule | 46 of 4,000 (1.15 percent), all of one kind: "delete the service and also remove this edge", where the model left out the service's dependants |
| Replies refused by the strict parser | 0 (the re-ask was not used) |
| Prompt, first tenth to last tenth | 1.16 times, following the size of the graph |
| Time in calls | 10,198 s, 2.55 s a call on average, 1.82 s median |

On the first 400 requests of the same stream, four ways of keeping the state were run with the same model and settings:

| First 400 requests | Steps with exactly the right state | Dollars per step with the right state | Times AXI |
|---|---|---|---|
| AXI (`scope_m7`) | 400 | $0.00117 | 1.0 |
| replies applied as written (`blind`) | 85 | $0.00524 | 4.5 |
| applied only if the gate accepts (`gate`) | 84 | $0.00573 | 4.9 |
| the repair alone (`repair_m6`) | 400 | $0.00117 | 1.0 |

The spend was within 5 percent across the arms; what differed is how many steps ended with the right state. One slip in a replies-as-written state stays until the stream removes it: the blind arm was off the intended state in stretches of 89, 36, 150 and 40 steps. Prices are the ones typed in by the person who ran it ($0.75 per million input tokens, $3.75 per million output tokens, thinking counted as output) and are not a result of the page; the whole 4,000-step run came to $4.95 at those prices.

On this run, of the three things AXI adds to a plain apply, only the repair acted: the scope filter removed 0 items and the re-ask was never needed. The page says so, and the figures above are what the repair alone bought.

## 3.2 One long, heavy job against the same model with the chat kept

*Pages: `docs/AXI_FOR_COMPANIES.md`, `docs/GEMINI_LONG_TEST.md`. One run of each arm, 22 October 2026.*

The job is a service graph edited request by request, 1,500 requests with AXI, each request accompanied by about 40 lines of routine operations log. Without AXI the whole chat was kept and sent again at every turn until it reached the model's window (457 turns, 949,591 tokens).

| | Chat kept | AXI |
|---|---|---|
| Prompt tokens per step | 476,835 on average | 2,824, flat for all 1,500 steps (169 times fewer) |
| Tokens sent in all | 217.5 million in 457 turns | 4.6 million in 1,500 steps |
| State if each reply is applied as it comes | exact until turn 286, wrong on every turn after | exact on 1,500 of 1,500 steps |
| First token, median | 2.23 s at best; 9.13 s once the prompt passes 500,000 tokens | 2.27 s, flat (2.15 to 2.35 s in every tenth) |
| Seconds a step | 7.13 on average | 3.07 |
| Exact steps an hour | 488 | 1,157 |
| Tokens computed per exact step, prompt of 500,000 or more | 5,419 | 3,108 |

## 3.3 A conversation in plain, ambiguous sentences

*Page: `docs/GEMINI_FUZZY_TEST.md` (Test 23). Gemini 3.8 Flash, 400 turns, chat kept against AXI.*

Tokens sent in all: 39.9 million with the chat kept, 0.53 million with AXI (76 times fewer); prompt tokens per step 99,809 on average against 1,177. The list the kept-chat arm holds if it applies each reply as it comes was exact on 65.0 percent of the turns: one wrong change at turn 191 stayed for 140 turns. With AXI the list was exact on 400 of 400 steps. Once the prompt passed 150,000 tokens the kept chat's first token took a median 5.57 s (1.86 s in the first 50 turns) against 2.41 s flat with AXI, with 97.0 percent of the kept chat's prompt served from the cache; tokens computed per exact step were 2,576 against 1,322. The kept chat did not drift on ambiguous text at this length (398 of 400 replies exact); the effect measured is that one slip stays in the state until something overwrites it.

## 3.4 A 7B open model, four jobs of 20,000 steps

*Page: `docs/ENDURANCE_7B_TEST.md` (Test 43).* Real steps of a 7B model (Qwen2.5-7B-Instruct-AWQ) served with vLLM 0.31.0 on one NVIDIA L4, greedy decoding, no retry: the held state was exact on 80,000 of 80,000 steps; the prompt stayed near 620 tokens and never passed 660; a step took 0.1605 s early and 0.1651 s late.

## 3.5 The gate under hostile input

*Page: `docs/GATE_STRESS_TEST.md` (Test 39).* Ten million proposals (legal ones, mutated ones, numeric extremes, injection texts, replays of old ones) were sent through the gate of the Python reference model on a CPU. 3,007,418 committed and 6,901,639 were refused; none of the 1,401,041 injection proposals committed, and the graph and the set of accepted changes matched an independent reference at every comparison. The run also found four things to harden, which the page lists, among them an integer-size limit that raised an exception on 90,943 proposals.

## 3.6 Capacity and large states

*Pages: `docs/SESSIONS_PER_CHIP_TEST.md` (Test 38), `docs/STATE_SIZE_SCALING_TEST.md` (Test 40).* On one L4 with a 7B model, 256 sessions ran at 105,230 exact steps per GPU hour, every step exact. At 20,001 items the whole state is 378,399 characters while the focused view shown to the model stayed at a median of 207 to 323 characters.

# 4. What the numbers are about

The claims are about the state the host holds. They are not about the model's accuracy in general, about any other model or job, or about any provider's servers. Each result is one run with sampled replies. The gate and the repair keep what is true and cut what is false; a reply that is readable, inside the request's scope and wrong (a wrong weight, a wrong endpoint) is applied, and AXI does not stop that. The 4,000-step run used a repair setting (delete up to eight dependants) fixed before any call and written into the header of every file; the framework's default is unchanged. The head-to-head against compaction is written and locked before its run (`docs/COMPACTION_TEST.md`); its result is not yet in.

# 5. Related work

A commit boundary around what an agent changes is also built by other groups. Cordon (Chen et al., arXiv 2606.17573 [7]) stages and checks irreversible agent actions before they take effect, with shadow state, an effect outbox and a recovery log; on 45 risk-bearing workflows it intercepted all 45 policy-violating effects against 14 for defences adapted from existing boundaries. MemTX (Li et al., arXiv 2607.23929 [8]) stages memory writes in snapshot-isolated transactions, blocks irreversible tool calls until in-flight beliefs are resolved, and repairs by typed cascade on retraction. Their settings and evaluations differ from ours and these systems have not been run against each other; we do not claim priority on the idea of a commit boundary (`docs/RELATED_WORK.md`). What AXI's pages add is a particular combination, a gate, a repair rule and an all-or-nothing commit on the host with a prompt rebuilt from the held state, and the measurements above.

# 6. Reproduction

```
git clone https://github.com/gaurshalendra2412/axi-runtime
cd axi-runtime/python
python3 quickstart_replay.py        # no key, no model, a few seconds
```

This replays the saved real replies of the 4,000-step run through the repository's own code and recomputes the held state step by step; it stops with an error if one replayed step differs from the saved one. With a Gemini key (a free one is enough), `python3 live_check.py` runs the same harness live for 15 steps and prints, for the live steps, how many held states are exact and how many would have broken had the replies been applied as written, and the prompt size at the first and last steps. A one-click Colab notebook is `python/colab/axi_try_it.ipynb`. The long-run and 7B numbers are recomputed from the saved files by the tests listed on their pages.

# 7. Credit and disclosure

The architecture is by **Rajnish Choubey**. Shalendra Gaur built and ran the tests and measurements. An AI coding assistant was used as a tool in writing code and these pages; it is not an author, and the disclosure is in the repository README. This work is not affiliated with any model provider. AXI here is a host-side runtime and is unrelated to the ARM AMBA AXI bus protocol.

# References

1. H. Lin. Architecting efficient context-aware multi-agent framework for production. Google Developers Blog, 4 December 2025. <https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/>
2. Context caching. Gemini API documentation. <https://ai.google.dev/gemini-api/docs/caching>
3. Best practices for handling context compaction and token blowup in recursive ADK agent tool loops? Google developer forums, 10 September 2026. <https://discuss.google.dev/t/best-practices-for-handling-context-compaction-and-token-blowup-in-recursive-adk-agent-tool-loops/396215>
4. Context Engineering for AI Agents: Lessons from Building Manus. <https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus>
5. P. Laban, H. Hayashi, Y. Zhou, J. Neville. LLMs Get Lost In Multi-Turn Conversation. arXiv:2505.06120.
6. Effective context engineering for AI agents. Anthropic, 29 September 2025. <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
7. Chen, Liu, Xu, Dong, Li, Pu, Zhai. Cordon: Semantic Transactions for Tool-Using LLM Agents. arXiv:2606.17573, 16 June 2026.
8. Li, Wang, Lu, Chen, Li, Song, Zheng, Cai. MemTX: Transactional Belief Commit for Stateful Agent Memory. arXiv:2607.23929.
9. R. Choubey, S. Gaur. AXI: The Cure for the Cost Crisis in AI. Zenodo, 4 November 2026. doi:10.5281/zenodo.23958737.

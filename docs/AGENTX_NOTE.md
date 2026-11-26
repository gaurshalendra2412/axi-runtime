# AXI and the AgentX numbers: where a kept chat reaches 99,809 prompt tokens a step, a host-held state sends 1,177 (84.8 times fewer)

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model, not for end users. The model is unchanged, and everything below is about the state the host holds. It fits any LLM application that holds state, short jobs and long, and it applies from the first step. Not affiliated with SemiAnalysis, WEKA, Nebius, Google, Anthropic, xAI, Meta or Microsoft.*

**In one paragraph.** This autumn's public serving benchmarks measure what it costs a host to replay long agent histories. SemiAnalysis's AgentX replays 393 Claude Code sessions, whose median request carries about 142k input tokens. A WEKA and Nebius test run under WEKA's Inference Economics Scorecard measured how much of the prefill work an HBM-only cluster repeats. Both are about the same cost: the history is the memory, so every step pays for all of it. On one job whose state the host can hold as data, the same arithmetic reads: 99,809 prompt tokens a step on average for the kept chat (97.0 percent of them served from the cache), 1,177 for the host-held state. This page sets the two side by side, names which number belongs to whom, and says what each one measures.

## What the two public sources report
| source | what it measured | figure |
|---|---|---|
| SemiAnalysis, AgentX v1.0 trace profile (inferencex.semianalysis.com/infx/agentx) | 393 Claude Code sessions replayed against inference engines | median input 142k tokens and median output 444 tokens per request |
| WEKA with Nebius, under WEKA's Inference Economics Scorecard (HPCwire, WEKA press item) | one 8-GPU NVIDIA HGX B300 node, DeepSeek-V4-Pro, 512 active sessions drawn from more than 5,000 real agentic coding sessions, 8 hours per configuration | HBM-only serving recomputed 60 percent of its prefill work; with WEKA's NVMe-backed Augmented Memory Grid, 6.8 percent. Under 5 seconds to first token for 44 percent of the successfully completed requests with HBM only, 97 percent with the grid |

Read on 26 Nov 2026. The 60 percent, 6.8 percent, 44 percent and 97 percent are WEKA's and Nebius's, not SemiAnalysis's; the 142k is SemiAnalysis's and is a median per request over the AgentX traces.

## What AXI measured, on the same kind of quantity
The job is the 400-step state edit of [`PRICE_CHECK.md`](PRICE_CHECK.md) and [`CACHE_PAIR_TEST.md`](CACHE_PAIR_TEST.md), run on Gemini 3.8 Flash. The prompt tokens are Google-billed counts (to about 1.3 percent, [`BILLING_CHECK.md`](BILLING_CHECK.md)).

| | host-held state (AXI) | kept chat, cold | kept chat, warm |
|---|---|---|---|
| mean prompt tokens a step | 1,177 | 97,353 | 99,809 |
| largest prompt of one step | 1,390 | 193,411 | 198,405 |
| share of prompt tokens read from the cache | 0 | 0 | 97.0 percent |
| times AXI's mean prompt | 1 | 82.7 | 84.8 |

The kept chat of this job lives in the same range as the AgentX median (142k): 99.8k on average, 198k at the longest step. A warm cache served 97.0 percent of its prompt tokens and the step still cost 6.5 to 19.5 times the host-held step at Claude, Grok and Gemini list prices ([`PRICE_CHECK.md`](PRICE_CHECK.md)), because cached tokens are still read and still billed. The kept chat also verified fewer steps: 260 of 400 warm and 328 of 390 cold, against 400 of 400 for the host-held state.

## What this does and does not compare
The percentages in the first table (60, 6.8, 44, 97) belong to coding sessions on DeepSeek-V4-Pro and are not set against any figure of ours. Our job edits a state; a coding session's context is code, tool output and notes, and only the part of it that is state can be held by the host as data. What the host holds as data is not replayed, not cached, and not paid for again at the next step. The size of that part in a coding session is the open question, and the one these two sources can answer from their traces.

## Check it
The AXI figures are recomputed from the saved result files by `python/tests/test_agentx_note.py`, with no model call. The two public figures are read from the pages named above.

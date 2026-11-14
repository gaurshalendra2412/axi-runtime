# Head-to-head: the kept chat with compaction against AXI, on the same job (written BEFORE the run, 14 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update. Nothing on this page has been run.*

## Why this test
The long-session test ([`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md)) put AXI against a chat that is kept whole and sent again at every turn. A fair objection: nobody runs a long agent session that way. Agent tools **compact** the chat when it gets long: the model is asked to summarise the conversation, and the summary replaces it. Two other things are also said about long sessions: that the service's cache makes a long chat cheap, and that a summary keeps what matters. The first was measured in the long-session test (99.1 percent of the kept chat's prompt tokens were served from the cache, and the chat still slowed and, after turn 222, slipped). The second has not been measured here. This test measures it.

It can come out **against AXI**. The kept chat was exact on every one of its first 222 turns, so a compaction arm that stays exact and costs less on this job would be a finding against AXI on this job, and the Result will say so in the same words as the other finding.

## What is run
One model, **`gemini-3.8-flash`** (the id is taken from the list the key can see), streamed, the model's own default thinking, no retry and no re-ask. The request stream is the hard test's dense service graph (`million_hard.HardStream`), seed **20261111**, and every request comes with the same heavy turn (40 lines of operations log) as in the long-session test: the same requests, the same text, the same model. Three arms, **600 steps** each, every reply applied as it comes (blind apply: no check, no repair), which is what a firm gets from the chat:

| arm | what it is |
|---|---|
| `S` | a chat like the kept chat of the long-session test. When the prompt of the last turn reached **32,000** tokens, the model is asked for a **generic summary** (`COMPACT_GENERIC`: "summarise our conversation so far; it will replace the conversation; include everything you will need to carry on the work correctly"). The chat is emptied and the summary leads the next turn. |
| `L` | the same, when the prompt of the last turn reached **128,000** tokens: fewer summaries, longer chats between them. |
| `V` | the same as `S` (32,000 tokens), but the summary is asked to **write out the complete current service graph** exactly as it stands, and any rule the model has noticed (`COMPACT_VERBATIM`). This is the rival built by someone who knows the job. |

The summary is one extra call before the next turn: the same chat, a different system text (`COMPACT_SYSTEM`), the request. Every call, the summary calls included, is counted (prompt tokens, cached tokens, reply and thinking tokens, seconds, seconds to the first token), and **every summary is kept in the file and published with the result**, so that anyone can read what the model kept.

**Read against.** The saved arm `A` of the long-session test (AXI: the host holds the exact state, `scope_m7`, a fresh prompt of about 2.8 thousand tokens at every step), **its first 600 steps**, from `docs/results/axi_gemini_long_session_seed20261111_default.json` (SHA-256 `7ba26d2946ec20ddd8a2085f18c6a2eee9ac3fb02025209158ce67a91faf29c6`, checked by the test). Arm `A` is **not run again**. It ran on 22 Oct on the same stream and the same heavy turns; the compaction arms run on a later day, on the same model name, and the service may have changed in between. The comparison is also not one change at a time: arm `A` has the framework's scope check and repair, the compaction arms have none. That is the point (what a firm gets from the chat against what the framework gives), and the contrast arms of the hard test (`docs/MILLION_TOKEN_TEST_HARD.md`) already separate the parts of the framework.

The code is `python/axi/experiments/gemini_compaction.py`; it imports `gemini_long.py`, `million.py`, `million_hard.py`, `api_llm.py`, `agent_loop.py`, `cascade.py` and the framework and **edits none of them**. The notebook is `python/colab/axi_gemini_compaction.ipynb`.

## What is measured
Per arm, over its 600 steps: the share of steps on which the state the host holds equals the oracle's; the first reply that is not exactly right; the first step on which the held state is wrong and the step from which it never equals the oracle's again; whether it is exact at step 600; corrupted steps (an edge pointing at a missing service); seconds a step (summary calls included); prompt tokens a step and the share served from the cache; and, at the **three prices typed in by the owner** (input, output, cached input; no price is built in), the cost of a step and of an exact step.

## What I read before writing this page
The long-session test: the kept chat gave the exact reply on its first 222 turns and slipped 15 times in 457; the hard test's contrast arms: with a fresh short prompt showing the state the host held, and **no** history, applying the replies as they came held the exact state on 85 of 400 steps; the four steps that took that arm off the oracle's state (steps 16, 132, 210 and 361) were all requests of one kind, `delete_pair`, in the saved file `docs/results/axi_million_hard_contrast_seed20261031_default.json`. A reading of these two (a hypothesis, not a result): the kept chat may have been exact early partly because its own earlier replies showed the model how to answer such delete requests, and a summary may not carry that. If so, the compaction arms slip after their summaries. If a summary does carry it, they will not.

## Predictions (written first by the AI assistant that built the test; the owner may change any before the page is locked)
Each line carries a probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 check that the run is the run described; lines 7 to 19 are about what was found.

1. [guard] arm S finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction (85 %)
2. [guard] arm L finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction (85 %)
3. [guard] arm V finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction (85 %)
4. [guard] the saved arm A of the long-session test is the file whose SHA-256 is written in the test, and its first 600 steps replay without a mismatch (90 %)
5. [instrument] every compaction has a non-empty summary, and there were at least 10 compactions in arm S, at least 3 in arm L and at least 10 in arm V (85 %)
6. [instrument] at least 98 percent of the replies of each of the three compaction arms were accepted by the strict parser (80 %)
7. [claim] arm S: the held state equals the oracle's on at least 98 percent of its 600 steps (20 %)
8. [claim] arm L: the held state equals the oracle's on at least 98 percent of its 600 steps (30 %)
9. [claim] arm V: the held state equals the oracle's on at least 98 percent of its 600 steps (35 %)
10. [claim] AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm S's (70 %)
11. [claim] AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm L's (60 %)
12. [claim] AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm V's (55 %)
13. [claim] arm S: the held state is not exact at step 600 (65 %)
14. [claim] arm L: the held state is not exact at step 600 (60 %)
15. [claim] arm V: the held state is not exact at step 600 (55 %)
16. [claim] arm S: the first reply that is not exactly right comes before turn 100 (55 %)
17. [claim] the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A over its first 600 steps (15 %)
18. [claim] (read only if the three prices were typed) the cost of a step of arm A is lower than that of each of the arms S, L and V (85 %)
19. [claim] (read only if the three prices were typed) the cost of an exact step of arm A is lower than that of each of the arms S, L and V (90 %)

## Reading rules (fixed before the run)
1. arm S finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction
2. arm L finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction
3. arm V finished its 600 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction
4. the saved arm A of the long-session test is the file whose SHA-256 is written in the test, and its first 600 steps replay without a mismatch
5. every compaction has a non-empty summary, and there were at least 10 compactions in arm S, at least 3 in arm L and at least 10 in arm V
6. at least 98 percent of the replies of each of the three compaction arms were accepted by the strict parser
7. arm S: the held state equals the oracle's on at least 98 percent of its 600 steps
8. arm L: the held state equals the oracle's on at least 98 percent of its 600 steps
9. arm V: the held state equals the oracle's on at least 98 percent of its 600 steps
10. AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm S's
11. AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm L's
12. AXI (arm A, its first 600 steps) holds the exact state on a share of steps at least 5 points above arm V's
13. arm S: the held state is not exact at step 600
14. arm L: the held state is not exact at step 600
15. arm V: the held state is not exact at step 600
16. arm S: the first reply that is not exactly right comes before turn 100
17. the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A over its first 600 steps
18. (read only if the three prices were typed) the cost of a step of arm A is lower than that of each of the arms S, L and V
19. (read only if the three prices were typed) the cost of an exact step of arm A is lower than that of each of the arms S, L and V

Headline: lines 10, 11 and 12 all read yes (AXI holds the state better than each compaction arm). If a guard or instrument line (1 to 6) does not read yes, no figure is claimed until the cause is found. If any of lines 7, 8 and 9 reads yes, the Result says in its first paragraph that that compaction arm matched AXI's exactness on this job, whatever else it says. Lines 18 and 19 read "not run" unless all three prices were typed. A line that was not run claims nothing.

## What this will not show
- **Other ways of keeping a long session going.** One generic summary request, one task-aware summary request, two sizes of trigger. Other summary prompts, memory kept in tools or in a search index, and other agent tools are not tested; a better summary prompt than `COMPACT_VERBATIM` may do better.
- **Anything about another company's tool.** The arms are built here from the description of compaction above; they are not a copy of any product.
- **That the result holds for other tasks, another model or another seed.** One job (a service graph) with log noise, one model, one seed, one run of each arm.
- **A run of 600 steps says nothing about longer ones.** On a job of a hundred or so heavy turns the kept chat was as exact as AXI in the earlier test.
- **Energy in joules.** The API gives none for Google's hardware.
- **Why a summary is right or wrong.** The summaries are published; whether they kept what mattered is for the reader to see.

## Guards on the bill
The notebook never turns billing on. It prints a **pilot** (45 turns of arms `S` and `V` on a seed of its own, with at least two summaries each) with the real size of a heavy turn and of a summary, and a **projection** of the whole test at the prices the owner types in, before any budget is set. A **dollar budget** (typed after the pilot) makes the notebook refuse the next call once the saved answers cost that much, and a **cap on tokens sent** (60,000,000 an arm) stops each run. Everything finished is saved and the same cell carries on.

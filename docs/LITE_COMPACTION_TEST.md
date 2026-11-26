# Test 46: the kept chat with compaction against AXI, on Gemini 3.5 Flash Lite, inside the free quota (written BEFORE the run, 25 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. AXI is for the company that runs the model, not for end users. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update. Nothing on this page has been run.*

## Why this test
[`COMPACTION_TEST.md`](COMPACTION_TEST.md) (14 Nov) asks whether a chat that is shortened by a summary, which is what agent tools do when a chat gets long, holds the state as well as AXI. It asks it on Gemini 3.8 Flash, and that model gives a free AI Studio key 20 calls a day, so the test needs a paid key and has not been run. The owner's key (read in AI Studio on 25 Nov 2026, Default Gemini Project, free tier, billing not set up) gives **Gemini 3.5 Flash Lite 15 calls a minute, 250,000 input tokens a minute and 500 calls a day**. This test asks the same question on that model, so it costs nothing. It does not replace that page: it is the same question on a smaller model, and the Result says "Flash Lite" wherever it states a figure.

Like that page, it can come out **against AXI**. A summary that carries the state well, on a job where AXI's gate has little to catch, would be a finding against AXI on this job, and the Result will say so in the same words as any other finding.

## What is run
One model, **`gemini-3.5-flash-lite`** (the id is taken from the list the key can see), streamed, the model's own default thinking, no retry other than the re-ask of arm A. The job is the one of test 22 and of `COMPACTION_TEST.md`: the dense service graph (`million_hard.HardStream`), seed **20261111**, and every request comes with the same heavy turn (40 lines of operations log, `gemini_long.ticket`). Three arms, **300 steps** each. In `S` and `V` every reply is applied as it comes (blind apply: no check, no repair), which is what a firm gets from the chat.

| arm | what it is |
|---|---|
| `S` | a chat like the kept chat of test 22. When the prompt of the last turn reached **32,000** tokens, the model is asked for a **generic summary** (`COMPACT_GENERIC`, the same text as `COMPACTION_TEST.md`): the chat is emptied and the summary leads the next turn. |
| `V` | the same trigger (**32,000** tokens), but the summary is asked to **write out the complete current service graph** and any rule the model has noticed (`COMPACT_VERBATIM`, the same text as `COMPACTION_TEST.md`). The rival built by someone who knows the job. |
| `A` | AXI: the host holds the exact state (`scope_m7`, up to 2 re-asks, the repair deleting up to 8 dependants) and shows it at every step with the same heavy turn; nothing is kept in the chat. Run again here, on the same model, because the saved arm A of test 22 was run on Gemini 3.8 Flash. |

The arm `L` of `COMPACTION_TEST.md` (a summary at 128,000 tokens) is not run: a prompt of that size is over the key's 250,000 tokens a minute at the call rate the key allows.

Every call, the summary calls included, is counted (prompt tokens, cached tokens, reply and thinking tokens, seconds, seconds to the first token), and **every summary is kept in the file and published with the result**, so that anyone can read what the model kept. The code is `python/axi/experiments/gemini_lite_compaction.py`; it imports `gemini_compaction.py` and `gemini_long.py` and **edits neither**. The notebook is `python/colab/axi_gemini_lite_compaction.ipynb`.

## What is measured
Per arm, over its 300 steps: the share of steps on which the state the host holds equals the oracle's; the first reply that is not exactly right; the first step on which the held state is wrong and the step from which it never equals the oracle's again; whether it is exact at step 300; corrupted steps (an edge pointing at a missing service); prompt tokens a step (summary calls included) and the share served from the cache; seconds a step (summary calls included). No dollar figure is claimed: there is no price in the code and the key is free.

## What I read before writing this page
Test 22 on Gemini 3.8 Flash: the kept chat gave the exact reply on every one of its first 222 turns and slipped 15 times in 457, every slip at a prompt of 463,161 tokens or more; with AXI the held state was exact on 1,500 of 1,500 steps at about 2,824 prompt tokens a step. The page `COMPACTION_TEST.md`: the contrast arms of the hard test, a short fresh prompt and no history with every reply applied as it comes, held the exact state on 85 of 400 steps. A reading (a hypothesis, not a result): a chat that has been shortened to a summary has lost most of what made the kept chat exact, so a generic summary arm may fall away early, a summary that writes out the graph may hold longer, and a smaller model may slip earlier than 3.8 Flash did. No Flash Lite run of this job exists.

## Predictions (written first by the AI assistant that built the test; the owner may change any before the page is locked)
Each line carries a probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 5 check that the run is the run described; lines 6 to 14 are about what was found.

1. [guard] arm S finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction (85 %)
2. [guard] arm V finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction (85 %)
3. [guard] arm A finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask (85 %)
4. [instrument] every compaction has a non-empty summary, and there were at least 8 compactions in arm S and at least 8 in arm V (85 %)
5. [instrument] at least 98 percent of the replies of each of the arms S and V were accepted by the strict parser, and at most 2 percent of the steps of arm A ended with a reply the strict parser refused after the re-asks (80 %)
6. [claim] arm A: the held state equals the oracle's on at least 98 percent of its 300 steps (65 %)
7. [claim] arm S: the held state equals the oracle's on at least 98 percent of its 300 steps (10 %)
8. [claim] arm V: the held state equals the oracle's on at least 98 percent of its 300 steps (35 %)
9. [claim] AXI (arm A) holds the exact state on a share of steps at least 5 points above arm S's (80 %)
10. [claim] AXI (arm A) holds the exact state on a share of steps at least 5 points above arm V's (55 %)
11. [claim] arm S: the held state is not exact at step 300 (85 %)
12. [claim] arm S: the first reply that is not exactly right comes before turn 100 (85 %)
13. [claim] the mean prompt tokens a step of arm S, summary calls included, are at least 3 times those of arm A (90 %)
14. [claim] the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A (50 %)

## Reading rules (fixed before the run)
1. arm S finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction
2. arm V finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction
3. arm A finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask
4. every compaction has a non-empty summary, and there were at least 8 compactions in arm S and at least 8 in arm V
5. at least 98 percent of the replies of each of the arms S and V were accepted by the strict parser, and at most 2 percent of the steps of arm A ended with a reply the strict parser refused after the re-asks
6. arm A: the held state equals the oracle's on at least 98 percent of its 300 steps
7. arm S: the held state equals the oracle's on at least 98 percent of its 300 steps
8. arm V: the held state equals the oracle's on at least 98 percent of its 300 steps
9. AXI (arm A) holds the exact state on a share of steps at least 5 points above arm S's
10. AXI (arm A) holds the exact state on a share of steps at least 5 points above arm V's
11. arm S: the held state is not exact at step 300
12. arm S: the first reply that is not exactly right comes before turn 100
13. the mean prompt tokens a step of arm S, summary calls included, are at least 3 times those of arm A
14. the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A

Headline: lines 6, 9 and 10 all read yes. If a guard or instrument line (1 to 5) does not read yes, no figure is claimed until the cause is found. A line that was not run claims nothing. A summary that carries the state well is a finding, and is written up as one.

## What this will not show
- **Anything about Gemini 3.8 Flash** (`COMPACTION_TEST.md` asks that) or about any other model: this is one small model on one day.
- **Anything about a company's own compaction code.** The summaries here are asked for by the repo's two prompts, which are the kind agent tools use; they are not ADK's, Anthropic's, OpenAI's or xAI's code.
- **That the result holds for other jobs.** It is one job (a service graph) with log noise; the claim is about this job.
- **Cost in dollars, or energy.** No price is in the code, and the API gives no joules.
- **Latency inside the service**, apart from what the streamed call shows: seconds to the first token includes the network from the notebook's machine.

## Guards on the bill
The key stays a free key: the notebook never turns billing on, has no price and no budget, and nothing in it can charge. The notebook spaces its calls 6 seconds apart (10 a minute, under the key's 15) and stops by itself after 480 calls in a day (the key gives 500; a failed call counts too). Everything finished is saved to Google Drive and the same cell carries on after the quota resets at midnight Pacific time. A cap on tokens sent stops each arm (12,000,000 for `S` and `V`, 3,000,000 for `A`). About 940 calls are needed in all (300 for each arm, about 20 summary calls in each of `S` and `V`, some re-asks in `A`): about two days of the free quota.

# Test 47: AXI around a public tool-agent benchmark (tau2-bench, retail), ten tasks, three arms, one model (written BEFORE the run, 26 Nov 2026)

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model, not for end users; the model is unchanged. This page is written and committed before any call; the Result goes below the line at its end, in a later update that changes nothing above it.*

## Why this test
Most of the repository's tests use jobs it defined itself. Sierra's tau2-bench is a public benchmark that someone else defined: a customer-service agent changes records in a database (orders, users) through tool calls, under a written policy, and an evaluator compares the final database with the expected one. It has the shape AXI is built for (a state that is edited, and rules that can be broken), and a result on it can be read by people who never saw this repository. This is a ten-task first look at what the pattern does to the tokens and to the pass count there; it is not a leaderboard entry.

A paper of 10 November 2026 (Sungbin Im, arXiv 2611.14581v1) compared ways of scoring agents on tau2-bench airline and retail, and tested three gates in front of writes as experimental conditions. Its gates did not significantly reduce forbidden writes, and its traces suggest that the gates' evidence summaries dropped details, so valid requests were rejected. Arms G and A below carry the same risk in a different form: the gate here is deterministic code, not a model call, but a rule that is too strict refuses a valid request in the same way.

## What is run
tau2-bench at commit `768f3d8fe8528b4f32a5096f9ed008ae50f48ead` (20 Nov 2026), domain `retail`, the first ten tasks of the `base` split (ids **0, 1, 2, 3, 4, 6, 7, 8, 10, 11**; the split has 114 tasks and skips 5 and 9), **one trial each**, seed 300, the benchmark's own user simulator and evaluator (reward 1.0 is a pass), at most 200 steps. The agent model and the user simulator are both `gemini/gemini-3.8-flash` through the benchmark's LLM layer, with the benchmark's default temperature of 0.0 unless the smoke test shows that the model needs another setting; both arms and the user simulator then use the same one and the Result says so. The benchmark's default user simulator is a different model, so nothing here is comparable with its leaderboard. The key used for the run needs billing on. On 26 Nov 2026 the smoke test ran on a free-tier key and Google answered that the free tier allows 20 requests a day for this model ("Generate content requests per day limit exceeded on Free Tier"); one conversation takes dozens of calls, so a free key cannot finish the 30 runs, and the runner stops at once on that error (it does not retry the conversation) and says so. The runner always asks again for the one call that failed with a rate-limit or "high demand" (503) error, because the benchmark would otherwise restart the whole conversation; for a key that has a per-minute cap, `--rpm N` (the notebook's `RPM`) also waits so that no more than N model calls leave in any minute. A failed call returns nothing and uses no tokens, so the conversations, prompts and token counts are the same as without the wait; only the time changes. The Result says which setting was used.

| arm | what it is |
|---|---|
| `B` | the benchmark's own agent (`llm_agent`): the whole chat is kept and sent again at every step. |
| `G` | the same chat, kept, with the host's deterministic gate in front of every tool call (below). |
| `A` | AXI: the host holds the state as data and rebuilds a short prompt at every step (below), with the same gate. The chat is not kept. |

**The gate** (`python/axi/experiments/tau2_retail_core.py`, plain Python, tested without any model): at most one tool call per message (a second is dropped); the customer is authenticated before any user or order tool; a user id or order must be the authenticated customer's; an order is read before it is changed; cancel and modify tools need a `pending` order and return and exchange tools a `delivered` one; item modification and exchange once per order; a cancel reason is one of the two the policy allows; and every write needs a fresh explicit customer confirmation (a "yes" after the details were listed, used up by the write). A refused call is sent back to the model with the reason, at most twice; after that the host sends the customer a plain message instead (a request to confirm, or the reason). A message that holds a tool call and text keeps the call and drops the text (the benchmark rejects such a message and ends the run in an agent error; arm B keeps that behaviour, as it is the benchmark's own agent). An empty reply is asked again, at most twice, and then the host sends a plain message; these are counted.

**The held state** (arm A): the policy; the customer id once found; the latest result of each distinct read (a re-read replaces the earlier copy, a write replaces the order with the updated order the tool returns); the list of changes made; the whole dialogue as text (the customer's and the agent's messages, without tool calls or results); and the one new event (the customer's message or the tool's result). No tool call or result of the past is replayed. The system prompt of arm A is the benchmark's own plus a note of 121 words that describes these parts, and the one new event is shown once (the newest result is not also listed among the held reads).

The code is `python/axi/experiments/tau2_retail_core.py` (held state and gate), `tau2_retail_agents.py` (the glue to the benchmark's agent interface) and `tau2_pilot_run.py` (the run); the notebook is `python/colab/axi_tau2_pilot.ipynb`. The benchmark's files are not edited.

## What is measured
For every task and arm: the benchmark's reward, the number of agent calls, prompt and completion tokens as the model layer reports them (summed over re-asks), refusals by the gate and what they were, and whether the host sent a fallback message. Dollars are the token counts at Google's list prices for this model (0.75 input, 3.75 output a million tokens; this price takes no discount for cached tokens, so it is an upper bound for every arm alike, and it is the price figure of reading 8). Cached prompt tokens are also read from the model's response where it carries them, and the Result gives the price with cached tokens at 0.075 a million as well; if that changes a reading, the Result says so.

## What this can and cannot show
Ten tasks and one trial each: a difference of one or two passes is within chance. The page reports every task. Token counts are a measurement; pass counts are a first look. A submission of an arm like `A` to the benchmark's leaderboard would be a "custom" one (the host changes the control flow and the prompt), not a standard one.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error.

1. [guard] all 30 runs end with a termination reason that is not an infrastructure error, and every task of every arm has a reward on file (75 %)
2. [claim] the mean prompt tokens of one agent call are lower in arm A than in arm B (90 %)
3. [claim] the total agent prompt tokens over the ten tasks of arm A are at most 0.7 of arm B's (55 %)
4. [claim] arm A passes at least as many of the ten tasks as arm B (50 %)
5. [claim] arm G passes at least as many of the ten tasks as arm B (50 %)
6. [claim] the gate refuses at least one proposed call in arm G or arm A (80 %)
7. [claim] the host sends a fallback message instead of the model's in at most 3 of the 20 gated runs (60 %)
8. [claim] the cost per passed task of arm A is at most that of arm B, if both pass at least one task (55 %)

What would count against AXI here: arm A passing three or more fewer tasks than arm B; prompt tokens per call not lower in arm A; a gate that refuses valid requests so often that the host's fallback message replaces the model's in more than 3 runs. Each is reported as it is.

## Reading rules
Pass counts are read as "k of 10". A difference of fewer than 3 tasks between arms is not a result on pass rate. Lines 2, 3 and 8 are read from the token and price columns of the Result. Line 6 and 7 are read from the gate columns.

## Not covered
One domain, ten tasks, one trial, one model, one user simulator. Retail's policy has rules the gate does not check (for example refund destinations and the gift-card balance). Whether the pattern holds on the other domains of the benchmark, on longer conversations, on other models, and on the full task set is not tested here.

# Test 47: AXI around a public tool-agent benchmark (tau2-bench, retail), the benchmark's frozen 50-task set, four arms, one model (written BEFORE the run, 26 Nov 2026; task set and rules changed 27 Nov 2026, also before any run, see below)

*The architecture is Rajnish Choubey's; the code, the tests and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model, not for end users; the model is unchanged. This page is written and committed before any call; the Result goes below the line at its end, in a later update that changes nothing above it.*

## Why this test
Most of the repository's tests use jobs it defined itself. Sierra's tau2-bench is a public benchmark that someone else defined: a customer-service agent changes records in a database (orders, users) through tool calls, under a written policy, and an evaluator compares the final database with the expected one. It has the shape AXI is built for (a state that is edited, and rules that can be broken), and a result on it can be read by people who never saw this repository. This is a ten-task first look at what the pattern does to the tokens and to the pass count there; it is not a leaderboard entry.

A paper of 10 November 2026 (Sungbin Im, arXiv 2611.14581v1) compared ways of scoring agents on tau2-bench airline and retail, and tested three gates in front of writes as experimental conditions. Its gates did not significantly reduce forbidden writes, and its traces suggest that the gates' evidence summaries dropped details, so valid requests were rejected. Arms G and A below carry the same risk in a different form: the gate here is deterministic code, not a model call, but a rule that is too strict refuses a valid request in the same way.

## Changes made before the run
On 27 Nov 2026, before any run of any arm on any task (the only calls made so far were the 26 Nov smoke test, which a free-tier key's daily quota stopped; it left no result), three things above were changed, and the page was edited in place; the commit history shows the earlier text.
1. **The task set.** It was the first ten tasks of the `base` split, a set I picked. It is now the benchmark's own frozen, stratified 50-task retail subset (`data/tau2/task_subsets/retail_50.json`, drawn by the benchmark's maintainers with seed 42 from the 114 retail base tasks; ids below). A set I did not choose cannot be accused of being chosen to suit the result, and 50 tasks can show a difference that 10 cannot.
2. **Arm S** was added (row `S` below).
3. **The rules that counted tasks** (reading 7, the guard, "what would count against AXI" and the reading rules) were scaled from ten tasks to fifty, and a paired exact sign test was added as the rule for calling a pass-rate difference a result. The probabilities in the Predictions section were set for ten tasks and are left as I wrote them; they were not re-set after seeing anything.

**A fourth change, 28 Nov 2026, after a first attempt that was stopped.** The first full run (arm B, 4 conversations in parallel) was started and stopped by hand after 16 tasks of arm B had finished (cost: under $5 at list prices; the Result gives the exact figure). Its files are not used; the run that is reported starts again from the first task under a new tag. Reason: 16 of the 50 tasks (ids 16, 19, 21, 36, 38, 39, 40, 47, 59, 63, 68, 70, 89, 104, 105, 107) carry written assertions ("natural-language assertions") that the benchmark checks with an LLM judge whose default model is an OpenAI one, and this run has no key for it, so those tasks failed on authentication after their conversations had been paid for. The judge is now the same model as the agent and the user simulator (`gemini/gemini-3.8-flash`, temperature 0.0), with the benchmark's own judge prompt and its own rule (every assertion must hold); the runner only removes a ```json fence from the reply, if the model adds one, so that the benchmark's `json.loads` can read it. The other 34 tasks have no written assertion, so their reward is the benchmark's database check alone, with no judge. The Result therefore reports every pass count and every paired test twice: on all 50 tasks, and on those 34 tasks. The judge is a deviation from the benchmark's own setting and is a second reason, next to the user simulator, why nothing here is comparable with its leaderboard. Before the stop I had seen only the running average reward of arm B on the first tasks (about 0.9), not any arm A, G or S result; the task set, arms and rules were not changed after that sight.

## What is run
tau2-bench at commit `768f3d8fe8528b4f32a5096f9ed008ae50f48ead` (20 Nov 2026), domain `retail`, the benchmark's frozen 50-task subset `retail_50` (ids **6, 8, 11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 30, 32, 36, 38, 39, 40, 42, 47, 49, 50, 52, 53, 56, 58, 59, 63, 64, 65, 66, 68, 70, 71, 74, 78, 79, 80, 83, 85, 86, 89, 97, 102, 104, 105, 107, 112**; run through `--taskset retail_50`, which the runner compares with the benchmark's own file before it starts), **one trial each**, seed 300, the benchmark's own user simulator and evaluator (reward 1.0 is a pass), at most 200 steps. The agent model, the user simulator and the judge of the written assertions (see the fourth change above) are all `gemini/gemini-3.8-flash` through the benchmark's LLM layer, with the benchmark's default temperature of 0.0 unless the smoke test shows that the model needs another setting; both arms and the user simulator then use the same one and the Result says so. The benchmark's default user simulator is a different model, so nothing here is comparable with its leaderboard. The key used for the run needs billing on. On 26 Nov 2026 the smoke test ran on a free-tier key and Google answered that the free tier allows 20 requests a day for this model ("Generate content requests per day limit exceeded on Free Tier"); one conversation takes dozens of calls, so a free key cannot finish the 200 runs (50 tasks, four arms), and the runner stops at once on that error (it does not retry the conversation) and says so. The runner always asks again for the one call that failed with a rate-limit or "high demand" (503) error, because the benchmark would otherwise restart the whole conversation; for a key that has a per-minute cap, `--rpm N` (the notebook's `RPM`) also waits so that no more than N model calls leave in any minute. A failed call returns nothing and uses no tokens, so the conversations, prompts and token counts are the same as without the wait; only the time changes. The Result says which setting was used. The first ten tasks of the `base` split (ids 0, 1, 2, 3, 4, 6, 7, 8, 10, 11) are kept in the runner as `--taskset first10` for the smoke test only.

| arm | what it is |
|---|---|
| `B` | the benchmark's own agent (`llm_agent`): the whole chat is kept and sent again at every step. |
| `G` | the same chat, kept, with the host's deterministic gate in front of every tool call (below). |
| `A` | AXI: the host holds the state as data and rebuilds a short prompt at every step (below), with the same gate. The chat is not kept. |
| `S` | **added on 27 Nov 2026, before any run:** the benchmark's own agent with the chat rolled up by the model itself. No gate, no held state. When the chat holds more than 24 messages, everything but the last 10 is written out and sent to the same model with one instruction; its reply replaces those messages as one message at the top of the chat, and the summary call is counted in the tokens and the cost. This is the plain compaction that agent frameworks and chat products offer, built here from that description (it is not a copy of any product). It has no reading line below: the Result reports it next to `B`, `G` and `A`. |

**The gate** (`python/axi/experiments/tau2_retail_core.py`, plain Python, tested without any model): at most one tool call per message (a second is dropped); the customer is authenticated before any user or order tool; a user id or order must be the authenticated customer's; an order is read before it is changed; cancel and modify tools need a `pending` order and return and exchange tools a `delivered` one; item modification and exchange once per order; a cancel reason is one of the two the policy allows; and every write needs a fresh explicit customer confirmation (a "yes" after the details were listed, used up by the write). A refused call is sent back to the model with the reason, at most twice; after that the host sends the customer a plain message instead (a request to confirm, or the reason). A message that holds a tool call and text keeps the call and drops the text (the benchmark rejects such a message and ends the run in an agent error; arm B keeps that behaviour, as it is the benchmark's own agent). An empty reply is asked again, at most twice, and then the host sends a plain message; these are counted.

**The held state** (arm A): the policy; the customer id once found; the latest result of each distinct read (a re-read replaces the earlier copy, a write replaces the order with the updated order the tool returns); the list of changes made; the whole dialogue as text (the customer's and the agent's messages, without tool calls or results); and the one new event (the customer's message or the tool's result). No tool call or result of the past is replayed. The system prompt of arm A is the benchmark's own plus a note of 121 words that describes these parts, and the one new event is shown once (the newest result is not also listed among the held reads).

The code is `python/axi/experiments/tau2_retail_core.py` (held state and gate), `tau2_retail_agents.py` (the glue to the benchmark's agent interface) and `tau2_pilot_run.py` (the run); the notebook is `python/colab/axi_tau2_pilot.ipynb`. The benchmark's files are not edited.

**Arm S, in detail.** `python/axi/experiments/tau2_retail_agents.py`, class `SummaryLLMAgent`, registered as `summary_chat`; the cut never separates a tool call from its result (`summary_cut` in `tau2_retail_core.py`, tested without any model). Same model, same temperature, same user simulator and evaluator as the other arms; the cached-token counts of the summary calls are added to the cached count of the run. Why it is here: the arms above compare AXI with a chat that is kept whole, and the next thing anyone who runs agents asks is how it compares with a chat that is summarised. If arm S passes as many tasks as `A` for less money, the Result says so in its first paragraph. Run it with `--arms S`; the other arms and their eight readings are unchanged.

## What is measured
For every task and arm (on all 50 tasks, and again on the 34 tasks without written assertions): the benchmark's reward, the number of agent calls, prompt and completion tokens as the model layer reports them (summed over re-asks), refusals by the gate and what they were, and whether the host sent a fallback message. Dollars are the token counts at Google's list prices for this model (0.75 input, 3.75 output a million tokens; this price takes no discount for cached tokens, so it is an upper bound for every arm alike, and it is the price figure of reading 8). Cached prompt tokens are also read from the model's response where it carries them, and the Result gives the price with cached tokens at 0.075 a million as well; if that changes a reading, the Result says so.

## What this can and cannot show
Fifty tasks and one trial each. The user simulator is a model, so running the same arm twice can pass a different set of tasks; this run does not measure that spread. A pass-rate difference is called a result only by the sign-test rule below. The page reports every task. Token counts are a measurement; pass counts are a first look. A submission of an arm like `A` to the benchmark's leaderboard would be a "custom" one (the host changes the control flow and the prompt), not a standard one.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error.

1. [guard] all 150 runs of arms B, G and A end with a termination reason that is not an infrastructure error, and every task of every arm has a reward on file (75 %)
2. [claim] the mean prompt tokens of one agent call are lower in arm A than in arm B (90 %)
3. [claim] the total agent prompt tokens over the fifty tasks of arm A are at most 0.7 of arm B's (55 %)
4. [claim] arm A passes at least as many of the fifty tasks as arm B (50 %)
5. [claim] arm G passes at least as many of the fifty tasks as arm B (50 %)
6. [claim] the gate refuses at least one proposed call in arm G or arm A (80 %)
7. [claim] the host sends a fallback message instead of the model's in at most 15 of the 100 gated runs (60 %; 15 % of the gated runs, the same share as 3 of 20 on ten tasks)
8. [claim] the cost per passed task of arm A is at most that of arm B, if both pass at least one task (55 %)

What would count against AXI here: arm A passing fewer tasks than arm B by a difference the sign-test rule below calls a result; prompt tokens per call not lower in arm A; a gate that refuses valid requests so often that the host's fallback message replaces the model's in more than 15 of the 100 gated runs. Each is reported as it is.

## Reading rules
Pass counts are read as "k of 50". Lines 4 and 5 are read on the raw counts. Any statement that one arm passes more or fewer tasks than another needs the paired test: take the tasks that exactly one of the two arms passed (X only, Y only), and the difference is a result only if the exact two-sided sign test gives p < 0.05 (for example 6 against 0, 8 against 1 or 10 against 2; 9 against 2 is not enough). Anything closer is written "no difference shown". The runner prints these pairs for A vs B, G vs B, S vs B and A vs S. With one trial per task and a model-simulated user the test is a loose bar, and the Result says so beside any p it reports. Lines 2, 3 and 8 are read from the token and price columns of the Result. Line 6 and 7 are read from the gate columns.

## Not covered
One domain, the 50 tasks of `retail_50`, one trial, one model, one user simulator. Retail's policy has rules the gate does not check (for example refund destinations and the gift-card balance). Whether the pattern holds on the other domains of the benchmark, on longer conversations, on other models, and on the 64 retail tasks outside `retail_50` is not tested here.

---

## Result (28 Nov 2026, run tag `t47r50`; everything above this line is unchanged)

**What this test is.** It runs the host-side layer in front of an unchanged model, over a provider's API, on a benchmark someone else wrote. The tensor-level work, which is Rajnish Choubey's, is not in this repository, and this test does not measure it.

**In one paragraph.** On the benchmark's 50 retail tasks, one trial each, the held-state agent (A) passed 44, the gated chat (G) 44, the benchmark's own agent (B) 43 and the summary baseline (S) 42. No pass-rate difference between any two arms is shown (exact sign test, p = 1.0 for A vs B, G vs B and S vs B; p = 0.5 for A vs S), so the gate and the held state did not cost passes here, and did not add any that this test can tell from luck. The host's fallback message was needed in none of the 100 gated runs. The measures that belong to the host-side layer itself: A sent 4.41 million agent prompt tokens against B's 4.62 million (0.955, so the 0.7 or lower of reading 3 did not come out) and 6,740 tokens per call against 6,882; at list prices the agent side cost $4.41 for A and $4.33 for B, which is $0.1003 and $0.1006 per passed task. These tasks are short (about 13 agent calls), so there is little chat for the rebuilt prompt to remove, and no saving is shown. **One provider's billing, kept apart from those:** if cached prompt tokens are priced at 0.075, A costs more than B ($4.12 against $2.35 for the 50 tasks), because Google's automatic prompt cache gave A 10 % of its prompt tokens against B's 63 %. A probe run afterwards on the same model (section below) shows what that cache rewards: a prompt that is the whole previous prompt plus more, and nothing else. It gave no hits to prompts that repeated about 90 % of the previous prompt's start and then differed. So that figure is how this provider's automatic cache bills today; it is not a property of the held state. Arm S passed fewer tasks than A, so the arm S paragraph's rule (S to be named here if it passed as many as A for less money) does not apply; at list price S cost more than A ($4.58 against $4.41), and at the cached price less ($2.72 against $4.12).

### How it was run
Model `gemini/gemini-3.8-flash` for the agent, the user simulator and the judge of the written assertions, temperature 0.0, seed 300, one trial per task, at most 200 steps, 4 conversations in parallel, no `--rpm` wait, a paid key. Code at `main` after PR #32 (commit `612b837`), tau2-bench at `768f3d8`. All 200 runs (50 tasks, four arms) ended with termination reason `user_stop`; none was an infrastructure error, no reward was missing, and the notebook's output shows no authentication error and no retried task. The summary file with every task's row is `docs/results/tau2_pilot.json` (written by the runner's `summary` command; the per-task rows were read off the notebook's own printout of that file and checked by rebuilding the file with the runner's own functions: the rebuilt file has the same length, 76,656 characters, and the same totals, tests and readings). The full conversation files stay on the notebook's disk and are not in this repository.

**Money.** Agent side at list price: B $4.33, G $4.88, A $4.41, S $4.58 ($18.20 together). User simulator at the same prices: $1.88. The judge's calls on the 16 judged tasks were not metered by the runner. The first attempt that was stopped (fourth change above) had no usage count kept, since its files were discarded; its cost was under $5 at list prices, as said above. The key's billing page has the exact total.

### Totals

| arm | passed of 50 | passed of the 34 without a judge | agent calls | prompt tokens | completion tokens | cached share of prompt | prompt per call | list $ | cached-price $ | list $ per pass | cached-price $ per pass | gate refusals | runs with the host's fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 43 | 31 | 671 | 4,617,534 | 230,024 | 63 % | 6,882 | 4.33 | 2.35 | 0.1006 | 0.0547 | 0 | 0 |
| G | 44 | 32 | 708 | 5,162,414 | 268,445 | 65 % | 7,292 | 4.88 | 2.62 | 0.1109 | 0.0596 | 14 | 0 |
| A | 44 | 33 | 654 | 4,407,984 | 294,903 | 10 % | 6,740 | 4.41 | 4.12 | 0.1003 | 0.0935 | 19 | 0 |
| S | 42 | 31 | 696 | 4,753,235 | 271,940 | 58 % | 6,829 | 4.58 | 2.72 | 0.1092 | 0.0647 | 0 | 0 |

Arm S made 30 summary calls in 25 of its 50 runs (counted in its tokens and dollars).

### Paired comparisons (tasks that exactly one of the two arms passed; exact two-sided sign test; a difference counts only at p < 0.05)

| pair | on all 50: X only / Y only / both / neither | p | on the 34 without a judge: X only / Y only / both / neither | p |
|---|---|---|---|---|
| A vs B | 2 / 1 / 42 / 5 | 1.00 | 2 / 0 / 31 / 1 | 0.50 |
| G vs B | 2 / 1 / 42 / 5 | 1.00 | 1 / 0 / 31 / 2 | 1.00 |
| S vs B | 0 / 1 / 42 / 7 | 1.00 | 0 / 0 / 31 / 3 | 1.00 |
| A vs S | 2 / 0 / 42 / 6 | 0.50 | 2 / 0 / 31 / 1 | 0.50 |

The largest gap anywhere is two tasks against none (A vs S) or two against one. The rule needed 6 against 0 or 8 against 1. With one trial per task and a model playing the customer, the test is a loose bar: it cannot tell a small real difference from luck, so "no difference shown" is not "no difference".

### The eight readings, against the probabilities I wrote first

| # | line | my probability | reading | detail |
|---|---|---|---|---|
| 1 | [guard] 150 runs on file, none an infrastructure error, every reward present | 75 % | **YES** | 150 runs |
| 2 | mean prompt tokens per agent call lower in A than in B | 90 % | **YES** | A 6740 vs B 6882 |
| 3 | total agent prompt tokens of A at most 0.7 of B's | 55 % | **NO** | A 4,407,984 / B 4,617,534 = 0.955 |
| 4 | A passes at least as many tasks as B | 50 % | **YES** | A 44 vs B 43 of 50 |
| 5 | G passes at least as many tasks as B | 50 % | **YES** | G 44 vs B 43 of 50 |
| 6 | the gate refuses at least one call in G or A | 80 % | **YES** | G 14, A 19 refusals |
| 7 | the host's fallback message replaces the model's in at most 15 of the 100 gated runs | 60 % | **YES** | 0 of 100 runs |
| 8 | cost per passed task of A at most that of B | 55 % | **YES** | A $0.1003 vs B $0.1006 per pass (list price, agent side) |

Seven of eight are yes; reading 3, the 30 % cut in total prompt tokens, is no, and it is the main finding. Of the three things the page said would count against AXI (fewer passes than B by a result, prompt per call not lower, more than 15 of 100 gated runs ending in the host's fallback), none happened. Reading 8 is yes only at the list price, which is how the plan page defined it. With cached tokens at 0.075 it would be no (A $0.0935 against B $0.0547 per passed task); the plan page said the Result would say so if that changed a reading.

### What this does and does not show
- **Shown:** in 100 gated runs (50 with G, 50 with A) on a benchmark that someone else defined, a deterministic check in front of every write, with the model unchanged, did not lower the pass count and never needed the host's fallback message. The paper cited at the top reported that gates in front of writes rejected valid requests; here that did not appear as a drop in passes (not proof that it never happened, see the refusals below).
- **Not shown:** a saving. On these short conversations (about 13 agent calls a task) the prompt per call is 6,740 tokens for A and 6,882 for B, 2 % apart. I did not measure how much of each prompt is the fixed part (policy and tool list), but with so little chat to remove it is likely most of it, so AXI's rebuilt prompt has little to take out. That is what the page's "Not covered" line on longer conversations left open; this run does not test it.
- **Cost, and what it depends on:** at list price A and B cost the same per passed task ($0.1003 and $0.1006). A's agent side is $0.08 higher in total ($4.41 against $4.33): A wrote 28 % more completion tokens than B (294,903 against 230,024; output is priced at five times input), which more than offsets its 4.5 % lower prompt tokens. With cached tokens at 0.075, A costs more ($4.12 against $2.35 for the 50 tasks). That figure rests on one provider's automatic cache, which gave A few hits (see the probe below); it is that provider's billing for this prompt shape, not a measure of the held state.
- **The gate's refusals:** G refused 14 calls in 14 runs and A 19 calls in 17 runs; every one was the rule that a write needs a fresh explicit customer confirmation. The summary rows record the rule and the tool, not the conversation, so I cannot say from them how many of those refusals were the model skipping the customer's yes and how many were a yes the gate should have counted. I have not read those conversations.
- **Tasks that differ:** B passed 107 alone (G, A and S failed it); G passed 18 and 104 where B failed them; A passed 52 and 64 where B failed them; S passed nothing that B failed. Tasks 19, 38 and 105 failed in all four arms. With one trial per task, any of these can flip on a rerun; this run does not measure that spread.
- **The judge:** 16 tasks are scored with the written assertions, judged by `gemini/gemini-3.8-flash` instead of the benchmark's default OpenAI model, so the 50-task counts are not comparable with the leaderboard. The 34-task columns carry no judge and show the same picture (B 31, G 32, A 33, S 31; no pair differs).

### Cache probe (28 Nov 2026, run after the result, same model and paid key)
Arm A's cached share (10 %) against the chat's (63 %) needed an explanation, and the cost line above depends on it. Five scripted shapes of prompt, six calls each, were sent to `gemini/gemini-3.8-flash` (temperature 0, made-up filler of Test 47's size: about 3,000 tokens of policy and two reads of 1,500), and the cached-token count was read from each reply (`cachedContentTokenCount`). Each shape starts with its own random line, so no shape could feed another's cache. Code: `python/axi/experiments/gemini_cache_probe.py` (standard library only; `python3 -m axi.experiments.gemini_cache_probe`; this run used about 486,000 prompt tokens, about $0.29 at list price).

| shape | what it is | prompt tokens per call | share of the prompt that repeats the previous prompt's start | cached share, call 2 on |
|---|---|---|---|---|
| `same` | the identical prompt every time | 11,114 | 100 % | 65 % (calls 1 and 2: 0 %; calls 3 to 6: 82 %) |
| `append` | chat shape (arms B and G): each prompt is the previous prompt plus more | 15,066 to 21,116 | 94 to 95 % | 86 % (80 % already on call 2) |
| `tail` | the same 15,000 tokens, only the last few characters change | 15,070 | about 100 % | 0 % |
| `rebuild` | arm A shape: the held reads, the dialogue, then the new event, rebuilt every step | 15,807 to 20,872 | 89 to 90 % | 0 % |
| `rebuild2` | the same with the dialogue first (the order tried in arm A2, not run) | 15,808 to 20,873 | 41 to 46 % | 0 % |

- **What it shows:** in this run the cache gave a discount when each prompt was the whole previous prompt plus more (`append`), and for an identical prompt from the third call on (`same`). It gave none in any of the 18 calls of the three shapes that share most of their start with the previous prompt but differ before their end (`tail`, `rebuild`, `rebuild2`). Arm A rebuilds its prompt every step, which is the `rebuild` shape; its 10 % against the chat's 63 % is consistent with that. Moving the dialogue to the front (`rebuild2`) did not change it.
- **Whose behaviour it is:** the provider's. The same prompts a host sends are billed differently by how the provider's cache matches them, and the host chooses the prompt's shape. This run does not test a held-state arm built for what the cache rewards.
- **Limits:** one key, one model, one day, six calls per shape. The provider's caching is best effort and the counts move (the identical prompt got no hit on its first two calls). The counts above were read off the notebook's printout; the probe's own JSON was not saved. Other providers, and providers that let the host mark what to cache, were not tested.

### Every task

✓ passed, ✗ failed; "J" marks the 16 tasks scored with written assertions; calls are agent calls; A and B prompt tokens are in thousands; refusals are the gate's (G / A).

| task | J | B | G | A | S | calls B / G / A / S | prompt k, B / A | refusals G / A |
|---|---|---|---|---|---|---|---|---|
| 6 |  | ✓ | ✓ | ✓ | ✓ | 12 / 15 / 15 / 12 | 79 / 104 | 0 / 0 |
| 8 |  | ✓ | ✓ | ✓ | ✓ | 14 / 12 / 15 / 12 | 99 / 114 | 0 / 1 |
| 11 |  | ✓ | ✓ | ✓ | ✓ | 11 / 13 / 13 / 11 | 66 / 84 | 1 / 1 |
| 13 |  | ✓ | ✓ | ✓ | ✓ | 9 / 10 / 10 / 11 | 47 / 54 | 1 / 0 |
| 14 |  | ✓ | ✓ | ✓ | ✓ | 10 / 13 / 11 / 14 | 55 / 68 | 1 / 1 |
| 16 | J | ✓ | ✓ | ✓ | ✓ | 13 / 16 / 15 / 13 | 85 / 105 | 1 / 1 |
| 17 |  | ✓ | ✓ | ✓ | ✓ | 7 / 7 / 8 / 7 | 33 / 40 | 0 / 0 |
| 18 |  | ✗ | ✓ | ✗ | ✗ | 10 / 12 / 10 / 10 | 57 / 56 | 0 / 0 |
| 19 | J | ✗ | ✗ | ✗ | ✗ | 10 / 10 / 8 / 9 | 62 / 41 | 0 / 0 |
| 21 | J | ✓ | ✓ | ✓ | ✓ | 20 / 20 / 18 / 19 | 161 / 129 | 0 / 0 |
| 22 |  | ✓ | ✓ | ✓ | ✓ | 15 / 16 / 16 / 15 | 98 / 112 | 1 / 1 |
| 23 |  | ✓ | ✓ | ✓ | ✓ | 20 / 22 / 20 / 20 | 176 / 164 | 1 / 0 |
| 25 |  | ✓ | ✓ | ✓ | ✓ | 12 / 11 / 10 / 10 | 72 / 56 | 0 / 0 |
| 26 |  | ✓ | ✓ | ✓ | ✓ | 10 / 10 / 10 / 10 | 55 / 55 | 0 / 0 |
| 30 |  | ✓ | ✓ | ✓ | ✓ | 21 / 20 / 19 / 25 | 163 / 132 | 0 / 0 |
| 32 |  | ✓ | ✓ | ✓ | ✓ | 20 / 19 / 20 / 22 | 139 / 139 | 0 / 1 |
| 36 | J | ✓ | ✓ | ✓ | ✓ | 16 / 17 / 16 / 18 | 127 / 116 | 0 / 0 |
| 38 | J | ✗ | ✗ | ✗ | ✗ | 16 / 18 / 3 / 18 | 120 / 10 | 0 / 0 |
| 39 | J | ✓ | ✓ | ✓ | ✓ | 10 / 11 / 10 / 11 | 55 / 55 | 0 / 0 |
| 40 | J | ✓ | ✓ | ✓ | ✓ | 9 / 9 / 9 / 9 | 45 / 46 | 0 / 0 |
| 42 |  | ✓ | ✓ | ✓ | ✓ | 18 / 19 / 17 / 16 | 132 / 144 | 1 / 3 |
| 47 | J | ✓ | ✓ | ✓ | ✓ | 16 / 17 / 14 / 12 | 105 / 85 | 1 / 0 |
| 49 |  | ✓ | ✓ | ✓ | ✓ | 14 / 14 / 14 / 17 | 96 / 94 | 0 / 0 |
| 50 |  | ✓ | ✓ | ✓ | ✓ | 3 / 3 / 3 / 7 | 12 / 10 | 0 / 0 |
| 52 |  | ✗ | ✗ | ✓ | ✗ | 11 / 10 / 10 / 12 | 66 / 56 | 0 / 0 |
| 53 |  | ✓ | ✓ | ✓ | ✓ | 10 / 10 / 10 / 10 | 55 / 55 | 0 / 0 |
| 56 |  | ✓ | ✓ | ✓ | ✓ | 14 / 14 / 14 / 14 | 91 / 91 | 0 / 0 |
| 58 |  | ✓ | ✓ | ✓ | ✓ | 13 / 13 / 13 / 13 | 100 / 100 | 1 / 1 |
| 59 | J | ✓ | ✓ | ✓ | ✓ | 10 / 11 / 11 / 11 | 55 / 70 | 0 / 1 |
| 63 | J | ✓ | ✓ | ✓ | ✓ | 12 / 14 / 14 / 15 | 76 / 89 | 0 / 0 |
| 64 |  | ✗ | ✗ | ✓ | ✗ | 13 / 14 / 12 / 14 | 81 / 71 | 0 / 0 |
| 65 |  | ✓ | ✓ | ✓ | ✓ | 6 / 6 / 6 / 6 | 26 / 27 | 0 / 0 |
| 66 |  | ✓ | ✓ | ✓ | ✓ | 10 / 10 / 10 / 10 | 54 / 54 | 0 / 0 |
| 68 | J | ✓ | ✓ | ✓ | ✓ | 10 / 10 / 10 / 10 | 51 / 51 | 0 / 0 |
| 70 | J | ✓ | ✓ | ✓ | ✓ | 14 / 14 / 11 / 14 | 94 / 65 | 0 / 0 |
| 71 |  | ✓ | ✓ | ✓ | ✓ | 19 / 18 / 18 / 19 | 153 / 133 | 0 / 0 |
| 74 |  | ✓ | ✓ | ✓ | ✓ | 14 / 16 / 15 / 14 | 101 / 120 | 1 / 1 |
| 78 |  | ✓ | ✓ | ✓ | ✓ | 16 / 18 / 17 / 15 | 113 / 125 | 1 / 1 |
| 79 |  | ✓ | ✓ | ✓ | ✓ | 14 / 14 / 13 / 14 | 106 / 86 | 0 / 0 |
| 80 |  | ✓ | ✓ | ✓ | ✓ | 11 / 11 / 10 / 10 | 67 / 57 | 0 / 0 |
| 83 |  | ✓ | ✓ | ✓ | ✓ | 12 / 12 / 12 / 12 | 72 / 80 | 0 / 1 |
| 85 |  | ✓ | ✓ | ✓ | ✓ | 15 / 15 / 15 / 15 | 97 / 96 | 0 / 0 |
| 86 |  | ✓ | ✓ | ✓ | ✓ | 16 / 16 / 15 / 16 | 108 / 106 | 0 / 1 |
| 89 | J | ✓ | ✓ | ✓ | ✓ | 12 / 12 / 12 / 11 | 78 / 80 | 0 / 0 |
| 97 |  | ✓ | ✓ | ✓ | ✓ | 12 / 13 / 12 / 12 | 79 / 83 | 1 / 1 |
| 102 |  | ✓ | ✓ | ✓ | ✓ | 19 / 20 / 19 / 23 | 148 / 128 | 0 / 0 |
| 104 | J | ✗ | ✓ | ✗ | ✗ | 23 / 27 / 25 / 25 | 221 / 214 | 1 / 1 |
| 105 | J | ✗ | ✗ | ✗ | ✗ | 11 / 14 / 11 / 14 | 72 / 65 | 0 / 0 |
| 107 | J | ✓ | ✗ | ✗ | ✗ | 15 / 18 / 15 / 14 | 118 / 120 | 1 / 1 |
| 112 |  | ✓ | ✓ | ✓ | ✓ | 23 / 24 / 20 / 25 | 197 / 172 | 0 / 1 |

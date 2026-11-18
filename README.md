<h1 align="center">AXI Runtime</h1>

<p align="center"><b>The model was right 98% of the time. Applied as written, only 21% of its steps ended with the right state.<br>With AXI between the model and the state: every one of them.</b></p>

<p align="center"><sub>First 400 requests of the same stream, three ways of keeping the state. The whole run was 4,000 requests.</sub></p>

<p align="center">AXI is a layer on the host side, between a model and the state it edits. The model is unchanged.<br>Architecture by <b>Rajnish Choubey</b>; tests built and run by Shalendra Gaur.<br>White paper: <a href="https://doi.org/10.5281/zenodo.23958737">AXI: The Cure for the Cost Crisis in AI</a> (Zenodo, 4 November 2026)</p>

<p align="center"><a href="https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/ci.yml/badge.svg"></a></p>

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/img/headline-numbers-dark.svg">
<img alt="Three numbers. 4,000 of 4,000 steps with exactly the right state, in one run on Gemini 3.8 Flash with 3.7 million tokens. A step with the right state cost 4.5 times as much when the replies were applied as written, on the first 400 requests. 46 of 46 slips, where the model left out a service's connections, repaired by a fixed rule." src="docs/img/headline-numbers-light.svg" width="600">
</picture>
</p>

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/img/headline-distance-dark.svg">
<img alt="Line chart over the first 400 requests. AXI is at zero items off the right state on every step. Replies applied as written go wrong at steps 16, 132, 210 and 361 and are off the right state on 315 of 400 steps. A gate that only refuses bad replies is never corrupted but is off the right state from step 210 to the end." src="docs/img/headline-distance-light.svg" width="600">
</picture>
</p>

**Why a model that is right 98% of the time can leave the state right on only 21% of the steps:** a slip that nothing catches stays in the state, and every later step builds on it. In these 400 requests the model slipped 7 times (393 replies right, 98%). Four of those slips, marked on the chart, each started a stretch in which the state was wrong, and together those stretches covered 315 of the 400 steps. The gate refuses bad replies, so it never held a broken state, but its state was off the right one from step 210 to the end. AXI's line is flat at zero.

The same idea was also tested on open models we host ourselves, which lets the host read cost, capacity and hostile input directly: [ten engineering questions, Tests 35 to 45](#ten-engineering-questions-answered-on-open-models-tests-35-to-45). Those runs add to the Gemini runs on this page; they do not replace them.

---

## What AXI is

A small runtime between an AI agent and the state it edits. The model proposes one change at a time. AXI checks the change against the rules, completes what the rules can complete, commits it all or not at all, and then shows the model the exact state again. The point is to stop small errors from compounding over long, multi-step agent runs.

The idea, the framework and the architecture are **Rajnish Choubey's**. This repository holds the code, tests and measurements that implement and check them. See [Credits](#credits).

## The run, in plain words

One model (Gemini 3.8 Flash), one job (editing a service graph of 17 to 22 services on average), one run of **4,000 requests in a row**, 3.7 million tokens sent in all.

* **With AXI keeping the state, it was exactly right at 4,000 of 4,000 steps** and never corrupted. The prompt stayed about the same size (1.16 times from the first tenth of the run to the last) and the speed did not fall (0.94 times).
* **The model itself is good:** its reply was right 3,954 times out of 4,000 (98.85 percent). All 46 slips were of one kind. Asked to "delete a service and also remove this edge", it left out the service's other connections. A fixed, rule-based repair completed all 46.
* **The same slips, applied as written, did real damage.** On the first 400 requests of the same stream, the replies applied as written left the state exactly right on 85 steps of 400 (21 percent) and corrupted it on 223 (56 percent). Once a slip got in, the state stayed wrong for 36 to 150 steps at a time (the last stretch was still going when the 400 steps ended).
* **A gate that only refuses bad replies** kept the state well formed (never corrupted) but did not make it right: exactly right on 84 of 400, and off the right state on every step from 210 to 400.

## What it cost

The spend was about the same for all three ways of keeping the state. What differed was how many steps ended with the right state, so what one right step cost differed too:

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/img/cost-per-right-step-dark.svg">
<img alt="Bar chart of what one step with the right state cost over the first 400 requests: AXI 0.0012 dollars, replies applied as written 0.0052 dollars (4.5 times as much), gate only 0.0057 dollars (4.9 times as much)." src="docs/img/cost-per-right-step-light.svg" width="600">
</picture>
</p>

| First 400 requests | Steps with the right state | Spent | Cost of one step with the right state |
|---|---|---|---|
| AXI (repair + scope filter) | 400 | $0.47 | $0.0012 |
| Replies applied as written | 85 | $0.45 | $0.0052 (4.5 times as much) |
| Gate only (refuses bad replies) | 84 | $0.48 | $0.0057 (4.9 times as much) |

Priced at $0.75 per million tokens in and $3.75 out (thinking counted as out), every call counted. The cost column was worked out from the saved files after the results were read, so it is not one of the pre-written reading rules.

**What this does not show.** It is one sampled run on one model and one job; a repeat would give different numbers, and the second run has not been made. The repair rule (a deleted service takes its dependants with it) is the job's own rule, and the model was never told it, so this shows the layer works when the rule is known, not that it can find rules. We did not measure an agent that reads its state back and corrects itself, or energy (a chat that keeps its history is measured in the section below).

Full page, every number, and what it does not show: [`docs/MILLION_TOKEN_TEST_HARD.md`](docs/MILLION_TOKEN_TEST_HARD.md).

### Earlier results on smaller models

* Qwen2.5-3B, four seeds, 60 tasks each, no rule in the prompt: applying the model's (grammar-constrained) output as written corrupted the graph on 120 of 120 "delete a service that still has dependants" tasks. With the gate, corruption was 0. Deterministic repair got every task right on all four seeds (240 of 240).
* Qwen2.5-7B, 30 requests in a row: applying as written corrupted the state in 12 of 12 episodes. The gate corrupted it in 0 of 360 steps.

Both are on synthetic service graphs, in Colab. The tables and what is still open are in [`docs/VERIFIED.md`](docs/VERIFIED.md) and [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md).

<!-- long-session:start -->
## The long session, with and without AXI (22 Oct 2026)

The same model (Gemini 3.8 Flash), a long job with heavy turns (every request came with about 40 lines of operations log), once with AXI holding the state (1,500 steps) and once with the whole chat kept (457 turns, up to 949,591 tokens).

| | Without AXI (chat kept) | With AXI |
|---|---|---|
| Prompt tokens sent per step | 476,835 | **2,824** (169 times fewer) |
| Tokens sent in all | 217.5 million (457 turns) | **4.6 million** (1,500 steps; 47 times fewer) |
| State held, exact | until turn 286; wrong on every turn after | **1,500 of 1,500 steps, 0 corrupted** |
| Distance of the held state from the true state | up to 17 items | **0 at every step** |
| First token, median | 2.23 s at best, 9.13 s past 500,000 tokens | **2.27 s, flat** |
| Seconds a step | 7.13 | **3.07** (2.3 times faster) |
| Exact steps an hour | 488 | **1,157** (2.4 times more) |
| Thinking tokens a step | 77 early, 239 late | **no climb: 210 early, 247 late** |

One run of each arm on one job. The page for a company that runs the model: [`docs/AXI_FOR_COMPANIES.md`](docs/AXI_FOR_COMPANIES.md). The full page with its pre-written reading rules and every table: [`docs/GEMINI_LONG_TEST.md`](docs/GEMINI_LONG_TEST.md).
<!-- long-session:end -->

<!-- fuzzy-and-arc:start -->
## For a company that runs the model: three more runs, and the whole picture (22 to 24 Oct 2026)

Gemini 3.8 Flash again, on three jobs that are not the dependency graph of the long session: a **400-turn conversation in plain, ambiguous sentences** (Test 23), **four ARC-AGI-3 games** (Test 24) and the same games **with a host-owned explorer** (Test 25). Each job was run once with the chat kept and once with AXI holding the state. The pages with their pre-written reading lines are in [`docs/`](docs/); every figure below is recomputed from the saved files by `python/tests/test_arc_fuzzy_results.py`. The full technical account is [`docs/AXI_TECHNICAL_REFERENCE.md`](docs/AXI_TECHNICAL_REFERENCE.md).

### Where AXI changes the numbers

| What a company counts | Without AXI (chat kept) | With AXI | Run |
|---|---|---|---|
| Prompt tokens sent a step, conversation job | 99,809 on average, 188,806 by the end | **1,177** (85 times fewer) | Test 23 |
| Tokens sent in all, conversation job | 39.9 million | **0.53 million** (76 times fewer) | Test 23 |
| Tokens sent in all, four ARC games | 25.7 million | **1.95 million** (13.2 times fewer) | Test 24 |
| The list a firm holds if it applies each reply as it comes | exact on 65.0 % of the turns: one wrong change at turn 191 stayed for **140 turns** | **exact on 400 of 400 steps**, distance 0 | Test 23 |
| First token, median, once the prompt passes 150,000 tokens | 5.57 s (1.86 s in the first 50 turns) | **2.41 s, flat** (2.31 times lower) | Test 23 |
| Seconds a step at that prompt size | 5.70 | **2.76** (2.07 times faster) | Test 23 |
| Tokens computed per exact step at that prompt size | 2,576 | **1,322** (1.95 times fewer) | Test 23 |
| Replies that could not be used, four ARC games | 13 of 240 (5.4 %) | **0 of 240** (asked again up to 2 times) | Test 24 |
| Moves that changed the screen, four ARC games | 87.5 % | **95.0 %** | Test 24 |
| Tokens sent, ARC games with the explorer | 25,689,461 (kept chat) | **1,790,904** (14.3 times fewer) | Test 25 |

### Where it does not, and what the runs say instead

| What a company counts | What was measured | Run |
|---|---|---|
| Does the kept chat drift or hallucinate on ambiguous text at this length? | **No.** 398 of 400 replies were exact (99.5 %), none was corrupted, and a fact changed 50 or more turns earlier was answered right 13 times of 14. The risk is not the rate of slips but that **one** slip stays in the state until something overwrites it. | Test 23 |
| Speed at a short prompt | The kept chat was the faster arm in the first 50 turns (1,609 exact steps an hour against 1,129). From turns 121 to 160 on it was slower; over all 400 steps AXI made 1.33 times the exact steps an hour (the page asked for 1.5). | Test 23 |
| Speed on ARC-AGI-3 | **AXI was slower**: 28.73 s a move against 15.87, 119 effective moves an hour against 200. The model thought 3,863 tokens a move with AXI and 1,714 without. | Test 24 |
| Levels completed on ARC-AGI-3 | **AXI did not complete more**: 1 level against 2 (60 moves a game, four games). The explorer did not add one (1). | Tests 24, 25 |
| Computed tokens on ARC-AGI-3 | About level (9,194 for the kept chat at a prompt of 100,000 tokens or more, 8,550 for AXI): the service served 94.8 % of the kept chat's prompt from its cache. | Test 24 |
| The explorer's conductor | **Never fired** (0 moves played by the framework): the model was never stuck by the rule. The object view and the untried list gave 95.4 % effective moves against 95.0 % and 22 % fewer thinking tokens a call, below the page's thresholds. | Test 25 |

### How to read the two tables together

AXI is a layer for the company that runs the model, and the runs show what it is for: **the state is kept by the host, so it is exact at every step, the prompt does not grow with the job, and the first token and the step time do not grow with it.** What it costs the model on a hard open-ended task is thinking time, and the ARC-AGI-3 runs show that cost; they do not show a gain in play. Both facts are measured and both are on the pages. A long job where a wrong state is expensive is the case the first table speaks to; a short, open-ended job like these ARC games is the case the second table speaks to.

*Scope: Gemini 3.8 Flash, three jobs, one run of each arm, one seed (four of the 25 public ARC-AGI-3 games, 60 moves each). Joules are not measured; tokens computed and seconds of calls are the stand-in. The ARC pages are not a leaderboard entry. Nothing here is a statement about any company's service.*
<!-- fuzzy-and-arc:end -->

---

## Ten engineering questions, answered on open models (Tests 35 to 45)

The headline runs above used a paid model (Gemini 3.8 Flash) through an API, which shows what a model does on a long, hard job. These runs complement them: they used open models that we serve ourselves with vLLM on one NVIDIA L4, so every number can be read from the host's side: what state the host holds, what a step costs it, how many sessions a chip carries, and what a hostile input does. In every run the model is unchanged and the state is the host's. The framework is **Rajnish Choubey's**; the tests were built and run by Shalendra Gaur.

| Question an engineer asks | What was measured | Page |
|---|---|---|
| Does the host's state stay exact over a long job? | 80,000 of 80,000 steps exact on a 7B model, in four jobs of 20,000 steps. The prompt stayed near 620 tokens and never passed 660; a step took 0.1605 s early and 0.1651 s late (+2.8 percent). | [Test 43](docs/ENDURANCE_7B_TEST.md) |
| What does keeping the whole chat cost the host instead? | The kept chat computed 1.24 to 4.11 times the prompt tokens per step on 11 models, and 1.78 times the tokens and 1.51 times the seconds on a like-for-like pair (7B). | [Tests 35](docs/MODEL_LADDER_TEST.md), [37](docs/MODEL_FAMILIES_TEST.md), [41](docs/MODEL_LADDER_COMPLETE_TEST.md), [42](docs/KEPT_CHAT_LONG_TEST.md) |
| How long can a kept chat go? | It ended after 108 to 235 steps in a 32,768-token window, and after 337 steps in a 131,072-token window, where a step slowed from 0.35 s to 1.36 s. The exact-state jobs ran 1,500 of 1,500 steps in every run, at a flat 0.28 to 0.31 s. | [Test 42](docs/KEPT_CHAT_LONG_TEST.md) |
| How many sessions does one chip carry? | 8, 32, 128 and 256 sessions on one L4 (7B): 100 percent exact in every cell; exact steps per GPU hour 51,134 at 8 sessions, 88,635 at 32, 105,043 at 128 and 105,230 at 256; joules per exact step 5.035 at 8 sessions and 2.458 at 256. The card is full at about 128 sessions. | [Test 38](docs/SESSIONS_PER_CHIP_TEST.md) |
| Does it hold on other models? | 100.00 percent exact in the main cell on 10 of 11 model runs, from 1.5B to 32B, with 0 corrupted steps in every cell. The eleventh, a DeepSeek-R1 distill, is explained below. | [Tests 35](docs/MODEL_LADDER_TEST.md), [37](docs/MODEL_FAMILIES_TEST.md), [41](docs/MODEL_LADDER_COMPLETE_TEST.md) |
| What about requests written in ordinary words? | When the host reads the sentence itself, the exact share rose on every one of the 11 models, for example 47.72 to 90.72 percent on the 14B, 97.91 to 100.0 on the 32B and 41.53 to 64.0 on the 7B. | [Tests 36](docs/MODEL_LADDER_WORDS_TEST.md), [37](docs/MODEL_FAMILIES_TEST.md), [41](docs/MODEL_LADDER_COMPLETE_TEST.md) |
| Does the way the chat is remembered matter? | On a 14B, six ways of keeping context all ended with the exact state (100.0 percent) once the host's check was in the loop. They differ in cost: the exact state took 0.30 s and 21.8 joules a step, summaries, a window and retrieval 0.55 to 0.66 s and 39.4 to 48.0 joules, the kept chat 1.72 s and 123.1 joules. | [Test 44](docs/MEMORY_ARMS_TEST.md) |
| Does it work for state that is not a service graph? | Four worlds (a table, a schedule, source code, tool calls) on a 7B and a 14B: with the world's own rule check, the held state broke no rule on any of the steps of that arm (four worlds, two models, 450 steps each). On the 7B the share of steps right went from 57.11 to 79.11 percent as written to 94.0 to 99.78 with the check. | [Test 45](docs/STATE_KINDS_TEST.md) |
| What does a hostile input do? | 10,000,000 proposals against the reference model, 1,401,041 of them injection attempts: 0 of the injection proposals committed and 0 state mismatches. The run also found four things to harden (one is an integer-size limit that raised an exception on 90,943 proposals); the page lists them. | [Test 39](docs/GATE_STRESS_TEST.md) |
| What happens when the state gets big? | At 20,001 items the whole state is 378,399 characters on an even graph (at least 985 times its size at 20 items), while the focused view the model is shown stayed at a median of 207 to 323 characters. | [Test 40](docs/STATE_SIZE_SCALING_TEST.md) |

### Does the host's state stay exact over a very long job?

Yes, in the runs here. On a 7B model, four jobs of 20,000 steps each ended with the state the host held equal to the true state on every one of 80,000 steps. The prompt the model was given stayed at about 620 tokens (largest 660), because the host shows the exact state, not the history. In Test 42 the exact-state jobs ran 1,500 steps each, 9,000 of 9,000 exact across three cells ([Test 42](docs/KEPT_CHAT_LONG_TEST.md)).

### What does it cost a host to keep the chat instead?

The kept chat in these tests was exact too, because it shows the whole state in every message. What it costs is compute and time. Over the same steps it computed 1.78 times the prompt tokens and took 1.51 times the seconds per step ([Test 42](docs/KEPT_CHAT_LONG_TEST.md)); in a 131,072-token window, 1.48 times the tokens and 2.76 times the seconds, with a last prompt of 129,759 tokens. On a 14B, the kept chat took 123.1 joules a step against 21.8 for the exact state ([Test 44](docs/MEMORY_ARMS_TEST.md)).

### How many sessions does one chip carry?

On one L4 with a 7B model, exact steps per GPU hour rose from 51,134 at 8 sessions to 105,043 at 128 and stayed there at 256 (105,230), with every step exact and the median round at 8.49 s for 256 sessions ([Test 38](docs/SESSIONS_PER_CHIP_TEST.md)). In this setup the card is full at about 128 sessions.

### What if the model gets a step wrong?

The host completes or refuses it. In the exact-state arm of Test 44 the model's own reply was right on 81.89 percent of steps (21.63 percent on deletes of a service with edges), and the state the host held was exact on 100.0 percent, because the gate checked each change and the repair completed what the rules could complete. Replies applied with no check were exact on 0.39 to 0.69 percent of steps in the same runs ([Test 44](docs/MEMORY_ARMS_TEST.md)). On the DeepSeek-R1 distill, every reply to a delete ran to the 96-token cap and could not be read; the host dropped those replies, so its state stayed well formed (0 corrupted steps) but fell behind the true state, which is why that one model is at 1.53 percent exact ([Test 37](docs/MODEL_FAMILIES_TEST.md)).

### What does a hostile input do?

Test 39 sent 10,000,000 proposals (legal ones, mutated ones, numeric extremes, injection texts, replays of old ones and more) through the gate of the Python reference model on a CPU. 3,007,418 committed, 6,901,639 were refused and 90,943 raised an uncaught exception of one kind. None of the 1,401,041 injection proposals committed, and the graph and the set of accepted changes matched an independent reference at every comparison (0 state mismatches). The run also found four things to harden, and the page lists them: a number of 4,300 or more digits makes Python's own limit raise an error instead of a refusal; two services whose coordinates land on one page can unmap each other's page; a crash in the middle of a commit leaves the pager failing its audit; and without a lock, 32 threads raise errors ([Test 39](docs/GATE_STRESS_TEST.md)).

### What happens when the state gets big?

At 20,001 items the whole state is 378,399 characters on an even graph (371,793 with hubs), about 124,000 to 126,000 tokens at 3 characters per token, which nearly fills a 128,000-token window. The focused view the model is shown stayed at a median of 207 to 323 characters from 20 to 20,000 items; a request that names the biggest hub shows 355 items (5,553 characters, 1.8 percent of the state). The gate's check does grow: 3.9 microseconds at 201 items and 257 at 20,001 (297 with hubs) ([Test 40](docs/STATE_SIZE_SCALING_TEST.md)).

### Reproduce Test 40 on your own machine

It needs only Python, takes about 3 seconds and writes one JSON file into the folder you name:

```bash
mkdir -p /tmp/axi_out
cd python
python3 -m axi.experiments.state_size_scaling /tmp/axi_out
```

It prints the tables and the thirteen reading lines of the page; the sizes and the 11 YES and 2 NO readings are the page's, while the microsecond timings vary by machine.

*Scope: open models on vLLM 0.31.0 and one NVIDIA L4 on Colab, greedy decoding, no retry; synthetic service graphs in Tests 35 to 38 and 41 to 44, four synthetic worlds in Test 45. The 32B ran with 10 GB of its weights offloaded to the CPU, so its seconds and steps per hour are not comparable with the others. Test 38's sessions run in lock-step. Tests 39 and 40 ran on a CPU against the Python reference model, not the Rust crates. Joules are the card's board power read during the run; the Gemini runs above measured none. The per-step raw files of Tests 37, 38 and 42 to 45 stayed on the Colab machines: the compact files in `docs/results` carry the figures and the sha256 of the raw files, and Tests 35 and 36 are printed lines only. Each page says which of its readings were NO. Nothing here is a statement about any company's servers.*

---

## How the tests are done

* **Written before the run.** Each page for a model run fixes the questions, the predictions and the reading rules before any number exists.
* **Locked, then appended.** The result goes below a line at the end of the page, and the text above the line is not changed. For the 4,000-step page and several others, a test checks this by hash.
* **A NO is printed as a NO.** The headline run had fourteen reading rules: twelve passed, one read NO and says so on the page, and one was not run.
* **Pages are recomputed.** For the result pages, a test rebuilds every number from the raw run files and fails if the page and the files disagree. The pictures on this page are drawn from the same files, and a test redraws them and compares.
* **Tests are tested.** Many are also run against deliberately broken copies of the code ("mutants"); `docs/VERIFIED.md` gives the number caught where this was done.

---

## Run it yourself

```bash
git clone https://github.com/gaurshalendra2412/axi-runtime
cd axi-runtime
```

**Check the headline numbers in a few seconds** (no key, no model, no GPU: it replays the saved replies of the 4,000-request run with this repository's own code and prints 4,000 of 4,000, 3,954 of 4,000, 85 of 400 and the 4.5 times above):

```bash
cd python
python3 quickstart_replay.py
```

**Python tests** (needs Python 3 and numpy; takes about three minutes; the four tests that need PyTorch or a GPU are skipped):

```bash
PYTHONPATH=python python3 python/tests/run_all.py
```

**Recompute the headline result** from the saved run files (about 20 seconds):

```bash
cd python
PYTHONPATH=. python3 tests/test_million_hard_result.py
```

**Redraw the pictures** on this page from the saved run files:

```bash
PYTHONPATH=python python3 python/tools/readme_figures.py docs/img
```

**Rust crates:** `cargo test --workspace` (this is what the CI badge above runs).

**Re-running the model experiments** needs your own API key and costs real money (the 4,000-step run cost $4.95 at the prices above). The Colab notebooks are in `python/colab/`. Keep the key in Colab's secrets and never put it in the repository.

---

## What is where

| Folder | What it holds |
|---|---|
| `crates/` | Rust prototype, early stage. The `axi-ir` crate includes four small modules (coordinate tuples, schema, router, positional encoding). The graph-rewrite check (`axi-dpo`) and the shadow pager are written but not yet included in a crate, and `axi-cache` is a placeholder. The gate and pager that are tested and measured are the Python ones below. |
| `python/axi/engine/` | The gate, the diagnosis and repair, the scope filter, the strict parser for the change language, the pager, the end-to-end runtime |
| `python/axi/experiments/` | The experiment code, including the 4,000-step run and the code that rebuilds its result page |
| `python/tools/` | The code that draws the pictures on this page from the saved run files |
| `python/tests/` | The test files; most say in their header what they show and what they do not |
| `python/colab/` | Colab notebooks used for the model runs |
| `docs/` | Every test page (written before its run) and the honest status table |
| `docs/results/` | The run files: for the earlier runs every request and every model reply; for Tests 35 to 45 the drivers and compact exports (see the scope line under the ten questions) |
| `docs/img/` | The pictures on this page |

**A few words used above.** *State:* the graph of services and the edges between them. *Change* (or *delta*): one proposed edit, written in a small strict language. *Gate:* the deterministic check that accepts or refuses a change. *Dangling edge:* an edge left pointing at a service that was deleted.

## Start reading here

1. [`docs/AXI_TECHNICAL_REFERENCE.md`](docs/AXI_TECHNICAL_REFERENCE.md): the whole of AXI in one technical document: the architecture as the code has it, the measured results of the last four Gemini runs, the method, the limits.
2. [`docs/COST_OF_BEING_WRONG.md`](docs/COST_OF_BEING_WRONG.md): what a wrong state costs a company that runs the model, counted from the saved files of the long-session and fuzzy-state runs: the turns that ended wrong and what they used, the cost of a step through the job, the cache sensitivity, what is not measured.
3. [`docs/MILLION_TOKEN_TEST_HARD.md`](docs/MILLION_TOKEN_TEST_HARD.md): the 4,000-step run.
4. [`docs/WHERE_THIS_STANDS.md`](docs/WHERE_THIS_STANDS.md): one page on latency, drift and cost as measured, what is not shown, and where the tensor-level roadmap stands.
5. [`docs/VERIFIED.md`](docs/VERIFIED.md): what is measured, what is only checked by tests, and what is **not** measured.
6. [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md): the earlier Qwen runs.
7. [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md): other groups working on neighbouring ideas, and what we cannot claim against them.
8. [`docs/RFC-001.md`](docs/RFC-001.md): the original design note.

## Articles and forum posts (November 2026)

The same results, written up for engineers in the words they search with: drift, token cost, latency, context window, sessions per GPU, prompt injection, model size, memory and tool calls. Each article gives its numbers and its limits and links back to the test pages here. AXI is the host-side layer by **Rajnish Choubey**: the model is unchanged and every claim is about the state the host holds. The tests were built and run by Shalendra Gaur.

- [AXI by Rajnish Choubey: a host-side layer that keeps LLM-edited state exact. Questions, answers and measured evidence](https://dev.to/shalendra_gaur_ecd0567a49/axi-by-rajnish-choubey-a-host-side-layer-that-keeps-llm-edited-state-exact-questions-answers-and-13cp)
- [AI agent drift: what it is, how to measure it, and the host-side fix that held 4,000 of 4,000 steps](https://dev.to/shalendra_gaur_ecd0567a49/ai-agent-drift-what-it-is-how-to-measure-it-and-the-host-side-fix-that-held-4000-of-4000-steps-4kg4)
- [LLM agent token cost: why a kept chat grows with the square of the steps, and the measured fix (prompt caching counted)](https://dev.to/shalendra_gaur_ecd0567a49/llm-agent-token-cost-why-a-kept-chat-grows-with-the-square-of-the-steps-and-the-measured-fix-1021)
- [Why LLM latency grows with conversation length, and the measured fix: a flat 0.1605 s to 0.1651 s step over 20,000 steps](https://dev.to/shalendra_gaur_ecd0567a49/why-llm-latency-grows-with-conversation-length-and-the-measured-fix-a-flat-01605-s-to-01651-s-2oi6)
- [How many LLM agent sessions can one GPU serve? Measured: 8 to 256 sessions on one NVIDIA L4, every step exact](https://dev.to/shalendra_gaur_ecd0567a49/how-many-llm-agent-sessions-can-one-gpu-serve-measured-8-to-256-sessions-on-one-nvidia-l4-every-1edn)
- [Host-held state for LLM agents: five questions to ask LangGraph, Letta, smolagents, PydanticAI and AXI, with measured answers](https://dev.to/shalendra_gaur_ecd0567a49/host-held-state-for-llm-agents-five-questions-to-ask-langgraph-letta-smolagents-pydanticai-and-1b43)
- [Prompt injection and LLM agent state: 0 of 1,401,041 injection proposals committed at the host's gate (10 million tested)](https://dev.to/shalendra_gaur_ecd0567a49/prompt-injection-and-llm-agent-state-0-of-1401041-injection-proposals-committed-at-the-hosts-2fbj)
- [Does model size matter for LLM agent reliability? 11 open models, 1.5B to 32B: host-held state exact on 10 of 11](https://dev.to/shalendra_gaur_ecd0567a49/does-model-size-matter-for-llm-agent-reliability-11-open-models-15b-to-32b-host-held-state-2c83)
- [LLM agent memory compared: whole chat, summary, sliding window, retrieval and host-held state on a 14B model](https://dev.to/shalendra_gaur_ecd0567a49/llm-agent-memory-compared-whole-chat-summary-sliding-window-retrieval-and-host-held-state-on-a-371i)
- [What happens when an LLM agent's chat outgrows the context window? Measured: 108 to 235 steps in 32k, 337 in 131k](https://dev.to/shalendra_gaur_ecd0567a49/what-happens-when-an-llm-agents-chat-outgrows-the-context-window-measured-108-to-235-steps-in-2fah)
- [LLM tool calls and structured edits: on a 7B model 57 to 79 percent of steps right as written, 94 to 99.8 with a rule check](https://dev.to/shalendra_gaur_ecd0567a49/llm-tool-calls-and-structured-edits-on-a-7b-model-57-to-79-percent-of-steps-right-as-written-94-om5)

The same articles are also on [Hashnode](https://axi-runtime.hashnode.dev) and on the [WordPress.com blog](https://axiruntimetests.wordpress.com). Two measurement posts with open questions are on the Google AI Developers Forum: [Gemini 3.8 Flash long agent loops: cost per correct step and first-token time](https://discuss.ai.google.dev/t/gemini-3-8-flash-long-agent-loops-measured-cost-per-correct-step-and-why-first-token-time-still-rises-with-about-97-of-the-prompt-cached/201990) and [Gemini 3.8 Flash, a 4,000-step test: compound delete requests left out dependants](https://discuss.ai.google.dev/t/gemini-3-8-flash-4-000-step-test-compound-delete-requests-left-out-dependants-in-46-of-97-replies-0-of-472-on-plain-deletes/202025).

## Status

A reference prototype and a set of measurements, not a product. The working, tested parts are in Python; the Rust crates are a partly wired sketch of the same design. Everything was measured on synthetic service graphs and grids, on the models named in `docs/`. The goals in the original design note (speed, cost, constant-time swaps) are what the design is meant to achieve; `docs/VERIFIED.md` says which of them have been measured here.

## Credits

* The idea, the framework, the architecture and the structural maps of AXI are by **Rajnish Choubey**.
* The code, tests and measurements in this repository implement and check his framework. They were written by **Shalendra Gaur**, working with an AI coding assistant (Claude). The assistant is a tool here: it is not an author and it owns nothing in this work. Shalendra Gaur is responsible for the code, the tests and every claim made from them.
* The mappings from Rajnish's writing to code (`docs/RAJNISH_CONCEPT_MAP.md`) are readings we propose. They are his to accept or reject, and a row he rejects is removed.
* Prior work we build on is cited, not claimed: double-pushout graph rewriting (Ehrig, Pfender and Schneider, 1973), adhesive categories (Lack and Sobocinski, 2004), grammar-constrained decoding (for example Outlines, Willard and Louf, 2023) and paged KV caches (vLLM). More in [`docs/CREDITS.md`](docs/CREDITS.md).

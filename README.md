<h1 align="center">AXI Runtime</h1>

<p align="center"><b>The model was right 98% of the time. Applied as written, only 21% of its steps ended with the right state.<br>With AXI between the model and the state: every one of them.</b></p>

<p align="center"><sub>First 400 requests of the same stream, three ways of keeping the state. The whole run was 4,000 requests.</sub></p>

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
| `docs/results/` | The raw run files: every request and every model reply |
| `docs/img/` | The pictures on this page |

**A few words used above.** *State:* the graph of services and the edges between them. *Change* (or *delta*): one proposed edit, written in a small strict language. *Gate:* the deterministic check that accepts or refuses a change. *Dangling edge:* an edge left pointing at a service that was deleted.

## Start reading here

1. [`docs/MILLION_TOKEN_TEST_HARD.md`](docs/MILLION_TOKEN_TEST_HARD.md): the 4,000-step run.
2. [`docs/WHERE_THIS_STANDS.md`](docs/WHERE_THIS_STANDS.md): one page on latency, drift and cost as measured, what is not shown, and where the tensor-level roadmap stands.
3. [`docs/VERIFIED.md`](docs/VERIFIED.md): what is measured, what is only checked by tests, and what is **not** measured.
4. [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md): the earlier Qwen runs.
5. [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md): other groups working on neighbouring ideas, and what we cannot claim against them.
6. [`docs/RFC-001.md`](docs/RFC-001.md): the original design note.

## Status

A reference prototype and a set of measurements, not a product. The working, tested parts are in Python; the Rust crates are a partly wired sketch of the same design. Everything was measured on synthetic service graphs and grids, on the models named in `docs/`. The goals in the original design note (speed, cost, constant-time swaps) are what the design is meant to achieve; `docs/VERIFIED.md` says which of them have been measured here.

## Credits

* The idea, the framework, the architecture and the structural maps of AXI are by **Rajnish Choubey**.
* The code, tests and measurements in this repository implement and check his framework. They were written by **Shalendra Gaur**, working with an AI coding assistant (Claude). The assistant is a tool here: it is not an author and it owns nothing in this work. Shalendra Gaur is responsible for the code, the tests and every claim made from them.
* The mappings from Rajnish's writing to code (`docs/RAJNISH_CONCEPT_MAP.md`) are readings we propose. They are his to accept or reject, and a row he rejects is removed.
* Prior work we build on is cited, not claimed: double-pushout graph rewriting (Ehrig, Pfender and Schneider, 1973), adhesive categories (Lack and Sobocinski, 2004), grammar-constrained decoding (for example Outlines, Willard and Louf, 2023) and paged KV caches (vLLM). More in [`docs/CREDITS.md`](docs/CREDITS.md).

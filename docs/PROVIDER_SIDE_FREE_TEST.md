# Test 20: the serving side a seventh time, requests in ordinary words (written BEFORE the run, 21 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before the run has been made. Code: `python/axi/experiments/provider_side_free.py` (it imports `provider_side_long.py`, `provider_side_typed.py` and the files they import, `engine/grounding.py` and `word_tasks.py`, and it edits none of them); instrument checks `python/tests/test_provider_side_free.py`; notebook `python/colab/axi_provider_side_free.ipynb`. Concept-map item 113. It follows `PROVIDER_SIDE_LONG_TEST.md` (item 112) and the Result at the end of that page. The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.

## Why this test
1. **The sixth run found what it could not show.** On jobs 192 to 255, `LN` (the grammar that names the coordinates) held the right state on 6,400 of 6,400 steps and `L0` (the capped expression) on 6,244 (97.6 %). In `LN` the host reads the request, builds the expression from the coordinates in it, and the model writes only the numbers (and, in a delete, the node alone). The sixth run says the state holds when the host owns the coordinates; it does not say what happens when the request does not carry them in a form the host can read.
2. **`L0` is already a partial answer.** In `L0` the model chose the coordinates itself from the sentence and was right on 97.6 % of the steps. But the sentence was one of three fixed shapes with the coordinates written out ("Delete the service at (2,1)."). A person writes "Remove the billing service." This test changes only that: every service has a name, and the request says the name.
3. **The framework already has a way to read a name** (`engine/grounding.py`, item 87, hash-locked and never run on a model): the host keeps a table of service names and aliases, and a name written as whole words is bound to its service; a phrase that fits two services is ambiguous and applies nothing. It is not language understanding and says so. This test uses it unchanged, and asks two separate questions with it: can the **model** read a name and write the coordinates, and what does a **host reader** add.

## What changes and what does not
**Does not change:** the model (`Qwen/Qwen2.5-7B-Instruct-AWQ`), the engine (vLLM 0.31.0), the GPU assumed (a Colab T4), the cache fixed at **80,000 tokens**, `max_model_len` 8,192, greedy decoding, at most 96 new tokens, prefix caching on, the seeds (job e has seed 20261017 x 1000 + e), the exact state shown at every step, the policy `repair_m6`, no retry, the lockstep rounds, the measures, the capped expression of the third run (sha256 `6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245`), the pinned files. **The jobs are the sixth run's: 192 to 255, 100 steps each**, with the same graphs, the same kinds of request, the same oracle replies. Only the sentence changes, so each cell can be put next to the sixth run's `L0` and `LN` job by job.
**Changes:**
* **Every service has a name.** The host's table lists a name (and for some an alias) for every service of the oracle's graph at every step; a service keeps its name while it exists, a new service takes the next unused name of a shuffled pool (the 22 names and the 7 aliases of `word_tasks.py`, edited by nothing). The table is the oracle's graph's, never the held state's.
* **The prompt shows the names** in a line "Names: billing = (2,1); email (also mailer) = (1,1); ..." after the state, and the system message gets the two lines of the words test about names (`NOTE` in the code). The same two additions are in all three cells.
* **Three cells**, in this order, on jobs 192 to 255 (64 jobs, 100 steps), no retry, about 35 minutes of GPU:

| cell | request | expression | what it asks |
|---|---|---|---|
| `C0` | with coordinates, as in the sixth run | capped | the control: do the two added lines alone change anything? Read against the sixth run's `L0` (6,244 of 6,400) |
| `W0` | in words | capped | **can the model read a name and write the coordinates?** Read against `C0` |
| `WN` | in words | the host reads the sentence (`read_request`) and the expression names what it read; where the host reads nothing, the capped expression | **what does a host reader add?** Read against `W0` (same prompts, only the expression differs) |

* **The wordings.** Each request is written from the oracle's coordinate request with a sentence chosen by a seed from two pools (`wording`), written once and not changed after the reader had run on them (see "What the reader did on the wordings" below). A name is replaced by its alias in one mention of five where the service has one. Directions are always "from A to B". The first pool (three quarters of the requests) has one service in the sentence for a delete, two for an edge and none for an added service (which keeps its coordinates and takes a new name, as in the words test). The second pool (one quarter) mentions one more service in the sentence, as a person does ("the billing service stays").
  * delete, first pool:
  * `Remove the {n} service.`
  * `Decommission the {n} service.`
  * `Please retire the {n} service.`
  * `Take the {n} service out.`
  * `Drop {n} from the graph.`
  * `We no longer need the {n} service, so delete it.`
  * delete, second pool (`m` is another service of the table):
  * `Remove the {n} service; the {m} service stays.`
  * `The {m} service is fine, but retire the {n} service.`
  * added service, first pool (`x` is the new name):
  * `Add a new service called {x} at ({r},{c}) with value {v}.`
  * `Create a service named {x} at ({r},{c}), value {v}.`
  * `Bring up {x} at ({r},{c}) with value {v}.`
  * `New service {x}: place it at ({r},{c}) with value {v}.`
  * added service, second pool:
  * `Add {x} at ({r},{c}) with value {v}; it will sit next to the {m} service.`
  * edge, first pool:
  * `Add a dependency edge from the {a} service to the {b} service with relation {rel} and weight {w}.`
  * `Connect the {a} service to the {b} service: relation {rel}, weight {w}.`
  * `Link {a} to {b} using relation {rel} and weight {w}.`
  * `Please add an edge from {a} to {b}. Relation {rel}, weight {w}.`
  * `Wire {a} -> {b} (relation {rel}, weight {w}).`
  * edge, second pool:
  * `Add a dependency edge from the {a} service to the {b} service with relation {rel} and weight {w}; the {m} service is not involved.`
  * `The {m} service is unaffected, but connect {a} to {b} with relation {rel} and weight {w}.`
* **The host reader (`read_request`) is small and refuses what is not clear.** It finds the names and aliases of the table in the sentence with `grounding.ground` (whole words, longest first; a phrase that fits two services is ambiguous), and takes the kind from the numbers the sentence carries: `relation` and `weight` for an edge (the first service named is the source, the second the target), one coordinate pair and `value` for an added service, none of the three for a delete. A sentence that names two or more services where one is wanted, a name that fits several, a kind it cannot tell, or a service not in the table is **not read**: the expression of that request is the capped one, so that nothing is refused and the model must read it as in `W0`. It never guesses, and it is not language understanding: "Remove the billing service; the auth service stays" names two services and it does not decide which one is meant.
* **What the reader did on the wordings (computed without a model, before this page was locked).** On the 6,400 requests of the 64 jobs (2,158 added services, 2,059 edges, 2,183 deletes) it read **5,380 right, 0 wrong and 1,020 not at all**. All 1,020 are second-pool sentences that name a second service (546 deletes, 474 edges); every added service is read (the coordinates carry it) and every first-pool sentence is read. The second pool is 1,548 requests (24.2 %), so the share not read is the share of my second pool that names a second service, not a rate of anything in the world: the numbers say that the reader is safe (it never read a request wrong here) and brittle (it does not read a sentence that mentions a second service), and nothing more. One defect was found and fixed in this first run of the reader over the wordings, before anything else was done with them: it looked for "relation" and "weight" in lower case and one first-pool edge sentence begins "Relation"; the reader now ignores case. No wording was changed.
* **A pre-flight check that can fail** runs before the engine is loaded: that the jobs are 192 to 255 with 100 steps, that the sixth run's file is the one named here, that the imported files (including `grounding.py` and `word_tasks.py`) are the pinned ones, that the expressions have the sha256 below, that the oracle's reply is inside the expression of every one of the 19,200 requests of the three cells, that the reader reads the 6,400 requests as stated above, that every sentence says what the oracle's request says (by a plain search, not by the reader), that the first request of each cell is built, and that the GPU is a T4 with less than 500 MiB in use. A RED line stops the notebook.

## What I read before writing this page
* **The sixth run's file** (`axi_provider_long.json`, sha256 `a0ad579961d006e9...`): `L0` 6,244 of 6,400 steps exact; its 14 first slips are all replies outside the named language; `LN` 6,400 of 6,400; 0.100 and 0.101 seconds per step.
* **The words test (item 87)** was built and checked on stand-in models and never run on a model; `grounding.py` says it over-refuses a paraphrase outside its stop list. Here the stop list is not tuned to the wordings, and the reader's refusals are measured, not hidden.
* **The fingerprints** (sha256 of the sorted set of expressions of a cell, recomputed from its jobs and the reader): `C0` and `W0` are the capped expression `6f4e5a99...`; `WN` `f8854dd780c79290e1807ce655770de597068290c1ab2f4f534aa36a34056c60` (775 different expressions: 774 named ones and the capped one, for the requests the host did not read).

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Lines 1 to 6 are checks that the run is the run I describe (lines 3, 4 and 6 are also checked before the run, so they stand at 99 %); lines 7 to 16 are about what the model does.

1. [instrument] the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run (99 %)
2. [instrument] the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens, and every reply is in the language of its own request (92 %)
3. [guard] the requests in words say what the oracle's request says: in every sentence of W0 and WN the name (or an alias) of each service the oracle's request is about is written, in the order of the oracle's request, the numbers are the oracle's, and a sentence of the second pool names exactly one service more (checked by plain search, not by the reader) (99 %)
4. [guard] the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs and the reader, and the oracle's own reply is inside the expression of every request (99 %)
5. [guard] every job of every cell ran all 100 steps; none ended early (96 %)
6. [guard] what the reader read, as recorded in WN, is what the reader gives on the same sentences when it is run again: the same outcome (right, wrong, miss) for every one of the 6,400 requests, and not one request read wrong (99 %)
7. [claim] the lines that tell the model the names do not hurt: the exact share of C0 is at most one point below that of the sixth run's L0 on the same 64 jobs (6,244 of 6,400) (65 %)
8. [claim] the model reads the names: the exact share of W0 over 100 steps is at least 95 percent (60 %)
9. [claim] the words cost little: the exact share of W0 is at most three points below that of C0 (65 %)
10. [claim] the host's reading does not hurt: the exact share of WN is at least that of W0 (90 %)
11. [claim] with the host reading what it can, the exact share of WN over 100 steps is at least 99 percent (55 %)
12. [claim] on the steps whose request the host read (right), the exact share of WN is at least 99.5 percent (50 %)
13. [claim] on the steps whose request the host did not read (miss, the model reads it as in W0), the exact share of WN is at least 95 percent (65 %)
14. [claim] the second pool of wordings (another service mentioned in the sentence) does not confuse the model more than three points: in W0 the exact share on the steps of that pool is at least that on the steps of the first pool minus three points (55 %)
15. [claim] the seconds per step of WN are at most 15 percent above those of W0 (both in this session) (88 %)
16. [claim] the exact steps per GPU hour of WN are higher than those of W0 (both in this session) (75 %)

**Headline** (lines 8 and 11): can the 7B model read a name and write the coordinates by itself, and does a host that reads what it can bring the state to at least 99 percent. Lines 2 to 6 all yes: about **85 %**. Lines 8 and 11 both yes: about **40 %**. All 16 yes: about **3 %**.

Why I think so. *Line 7 (65 %):* the added lines change the prompt but not the task; the sixth run's `L0` had 156 inexact steps of 6,400, and one point is 64 steps. *Line 8 (60 %):* for a delete the model has to look a name up in a list and copy the pair, for an edge it does so twice; a 7B model does this well but not always, and a wrong lookup stays in the state. *Line 9 (65 %):* if line 8 is yes this is likely. *Line 10 (90 %):* the host's reading only narrows the model on requests it read right, and it read none wrong. *Line 11 (55 %):* 84 percent of the requests are read by the host and cannot be misread; the other 16 percent are deletes and edges that mention a second service, where the model may take the wrong one; a slip persists. *Line 12 (50 %):* the steps whose request was read are exact unless an earlier unread request left a slip in the state. *Line 13 (65 %):* these are the hard sentences. *Line 14 (55 %):* the second pool is a distractor by design. *Line 15 (88 %):* the sixth run's 835 expressions cost +0.6 %; this one has 775. *Line 16 (75 %):* it follows from line 10 and the same speed.

**Also written down, not lines (read afterwards as descriptions).** (a) `W0` holds the right state on 90 to 98 percent of its 6,400 steps: 70 %. (b) More than half of the first slips of `W0` are on edge requests: 60 %. (c) In the second pool the model's wrong lookups take the second service named in the sentence rather than a service not in the sentence: 60 %. (d) The seconds per step of `WN` are within 5 percent of those of `W0`: 80 %. (e) The three cells take 30 to 45 minutes in all, loading excluded: 65 %. (f) The thing I do not know and would like to: whether the aliases (one mention in five for 7 of the 22 names) are read as well as the names.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers.
* *Lines 2 to 6 yes, lines 8 and 11 yes:* on the sixth run's 64 jobs, with the requests written with names in the sentences of this page, the 7B model read the names itself on at least 95 percent of the steps and the host's reading brought the state to at least 99 percent. The sentence is the shares, the seconds per step and the exact steps per GPU hour, with the setup beside them, and **in the same sentence that the second pool of wordings was refused by the host and read by the model**.
* *Line 8 NO:* the model did not read the names well enough by itself. Read the first misses of `W0` (printed): which kind, which lookup. That is the finding, and it is the case where a host reader or a bigger model has something to add.
* *Line 8 yes, line 9 NO:* the model reads names well but not as well as it reads coordinates; read by how many points, and in which kind.
* *Line 10 NO:* the host's reading hurt: read the steps where it read and the model's reply differs from the oracle's. It must not happen with a reader that read none wrong; if it does, the cause is in the interplay with the persisting slips.
* *Line 11 NO with line 10 yes:* the host reader helps but not to 99 percent in this setup; read the unread requests (line 13).
* *Line 13 NO:* the model reads the second pool badly where the host does not read. This is the case a better reader (or a bigger model) is for.
* *Line 14 NO:* a mentioned second service confuses the model by more than three points; read which of the two it chose.
* *Line 7 NO:* the added lines alone changed the exact share by more than a point; then `W0` against `C0` is the comparison and not `W0` against the sixth run's `L0`.
* *Lines 15 or 16 NO:* the 775 expressions cost seconds; read the rounds with new expressions.
* *Line 3 NO:* a sentence does not say what the oracle's request says: a mistake in the wordings, read first. *Line 6 NO:* the reader read differently in the run than offline, or read a request wrong: read first. *Line 4 NO:* a cell used other expressions than the ones this test defines. *Line 2 NO:* the constraint did not hold in some request. *Line 1 NO, the pre-flight RED, or the safety cell says STOP:* nothing is run or read.
* *A cell that did not run:* its lines read "not run", never NO.

## Reading rules (fixed now; the printout prints exactly these)
The 16 lines above are the reading rules; `provider_side_free.verdict_free` computes them from this run's cells. Line 1 uses every request of every cell; line 2 every reply (the share the strict parser accepted, the replies that used all 96 new tokens, and the replies that fully match the expression carried by their own request); line 3 checks, for every sentence of `W0` and `WN`, by a plain search of the table's names and aliases and of the numbers (not by the reader), that the sentence names the services of the oracle's request in its order, carries the oracle's numbers, and names exactly one more service in the second pool and none in the first; line 4 recomputes the sha256 of each cell's set of expressions from its jobs and the reader and checks that the oracle's reply is inside the expression of every request; line 5 that every job of every cell ran all 100 steps; line 6 that the reader, run again on the recorded sentences, gives the recorded outcome (right, wrong, miss) for every request and none wrong. Lines 7 to 16 compare shares of exact steps ("the held state equals the oracle's after the step"): line 7 `C0` against 6,244/6,400 minus one point; line 8 `W0` at least 95 percent; line 9 `W0` at least `C0` minus three points; line 10 `WN` at least `W0`; line 11 `WN` at least 99 percent; line 12 `WN` on the steps whose request the host read right, at least 99.5 percent; line 13 `WN` on the steps whose request the host did not read, at least 95 percent; line 14 `W0` on second-pool steps at least that on first-pool steps minus three points; line 15 `WN` seconds per step at most 1.15 times `W0`'s; line 16 `WN` exact steps per GPU hour above `W0`'s.

**The rule that decides what the Result may claim.** Lines 2, 3, 4, 5 and 6 all read yes: the figures of the cells are claimed as read. What the model does with names is claimed from lines 7, 8, 9 and 14, what the host reader adds from lines 10 to 13, a statement about time or cost only from lines 15 and 16 (cells of one session). Line 2, 3, 4, 5 or 6 reads NO: no claim is made from this run until the cause is found.

**How the result will be worded.** On the sixth run's 64 jobs, with the requests written in the sentences of this page, on one Tesla T4, one 7B model and the engine written next to the figures: the exact share over 100 steps for the three cells, the seconds per step, the exact steps per GPU hour, the share by pool and by what the host read; and in the same sentence that the names are the table's (given, not learned), that the host reader reads a sentence only where one service is named for a delete, two for an edge, or an added service with its coordinates, and refuses the rest, and that the sentences are those of this page and not requests in the words of any user. Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never "100 percent" for a grammar, never "proven" (one model, one GPU, one run, 64 jobs), never a statement about any provider, never a statement about requests in the words of a user, never a statement about a model that was not run. A "NO" on a claim line is a finding about this setup and goes into the article as such.

## What this does not show
Requests in the words of a user (the sentences are the ones listed above, written by me, in two pools); names the host did not give (the table is given, not learned; a paraphrase such as "the thing that charges cards" is outside it); a language other than English; tasks of more than 100 steps, let alone days; real company tasks; another model, size, GPU, engine or cache size (a bigger model is the next test, with its own page); a kept chat; a retry on top of a grammar; dollars; a capacity in jobs per GPU; energy; hallucination in free text. Nothing here is proven.

## How it is run
Open `python/colab/axi_provider_side_free.ipynb` in Colab with a T4 and run the cells one at a time: it gets the code from GitHub, installs vLLM 0.31.0, connects Drive, fixes the output stream, runs the **pre-flight check**, loads the model with the per-request constraint, runs the safety cell, then the three cells in order. Each finished cell is written as `axi_provider_f_<label>.json` and copied to Drive; a cell that was cut off is run again from its start (the longest takes about 13 minutes). The last cell puts the cells together, prints the table, the split by pool and by what the host read, the first misses and the 16 lines, and offers the files for download. The Result goes below the line at the end of this page, in a later update with a test of its own.

---


# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_free.py` and again by `tests/test_provider_side_free_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_free.ipynb`: 19,200 steps in three cells, 50.4 minutes of generation, vLLM 0.31.0, `Qwen/Qwen2.5-7B-Instruct-AWQ`, Tesla T4, a cache of 80,000 tokens. The merged file is `docs/results/axi_provider_free.json` (sha256 `4ec1ce81239b4b18...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*

**What the page expected, and what came out.** **Line 2 reads NO**, so by the page's rule no claim is made until the cause is found; the cause is found below (four replies of the capped expression that ran to the token limit) and the figures are reported as read, with that NO beside them. Lines 3, 4, 5 and 6 read yes. Of the 16 lines **9 read yes and 7 read NO** (2, 8, 9, 11, 12, 13 and 14); no line is "not run". **Both headline lines read NO:** the model alone did not read the names well enough (line 8, 60 %), and the host reading what it can did not bring the state to 99 percent (line 11, 55 %). The page put both yes at about 40 %.

**In one paragraph.** With the request given as coordinates plus the list of names, the exact state was held on 6,400 of 6,400 steps. With the request in ordinary words and the model left to read the name, it was held on 39.8 %; with the host reader reading the sentences it can read, 61.5 %. The host reader was never wrong (5,380 requests read right, 0 wrong, 1,020 not read) and none of the 1,536 steps it read before a job's first slip was a slip. The slips come from the model's own lookups of a name on edge requests, and from the requests the host did not read; a slip stays in the held state until a later request removes the wrong item, so the shares over 100 steps are much lower than the rate of new mistakes. The serving numbers (seconds per step, tokens per step) did not move with the words or with the reader; the names list in the prompt has a cost in tokens that the sixth run did not have.

## The 16 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 99 % | cached tokens reported on all 19,200 requests (5,523,456 of 8,394,619 prompt tokens served from the cache) | yes |
| 2 | 92 % | 19,196 of 19,200 replies accepted by the strict parser (99.98 %); **4 replies used all 96 new tokens** (3 in `W0`, 1 in `WN`, none in `C0`) and are outside the language of their own request | NO |
| 3 | 99 % | all 12,800 sentences of `W0` and `WN` name the services of the oracle's request in its order, with its numbers, and the second pool names exactly one service more (checked by plain search, not by the reader) | yes |
| 4 | 99 % | the recorded sha256 of every cell equals the one recomputed from its jobs and the reader; `WN` has 775 expressions; the oracle's reply is inside the expression of all 19,200 requests | yes |
| 5 | 96 % | all 192 jobs of the three cells ran 100 steps | yes |
| 6 | 99 % | the reader, run again on the recorded sentences, gives the recorded outcome at all 6,400 requests: right 5,380, wrong 0, not read 1,020 | yes |
| 7 | 65 % | `C0` 100.0 % (6,400 of 6,400 steps) against 97.6 % for the sixth run's `L0` (6,244 of 6,400) | yes |
| 8 | 60 % | `W0` 39.8 % (2,546 of 6,400 steps; 0 of 64 jobs exact to the end) | NO |
| 9 | 65 % | `W0` is 60.2 points below `C0` | NO |
| 10 | 90 % | `WN` 61.5 % against `W0` 39.8 % | yes |
| 11 | 55 % | `WN` 61.5 % (3,936 of 6,400 steps; 2 of 64 jobs exact to the end) | NO |
| 12 | 50 % | 63.0 % of the 5,380 steps whose request the host read right | NO |
| 13 | 65 % | 53.7 % of the 1,020 steps whose request the host did not read | NO |
| 14 | 55 % | `W0`: second pool 36.5 % of 1,548 steps, first pool 40.8 % of 4,852; the difference is 4.3 points | NO |
| 15 | 88 % | `WN` 0.157 s per step, `W0` 0.155: +1.3 % | yes |
| 16 | 75 % | `WN` 14,059 exact steps per GPU hour, `W0` 9,215 | yes |

Lines 7, 10, 15 and 16 are the ones that read yes among the claim lines (the names list does not hurt the exact share; the host's reading does not hurt; the 775 expressions cost no seconds; `WN` does more exact steps per hour than `W0` because it holds the state on more steps, at the same speed). Line 14 reads NO by 1.3 points more than the three the page allowed.

## The cells

| cell | request given as | steps | exact steps | jobs exact to the end | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |
|---|---|---|---|---|---|---|---|---|
| `C0` | coordinates + names list | 6,400 | 100.0 % | 64 of 64 | 0.160 | 156.2 | 442.7 | 22,513 |
| `W0` | words, capped expression | 6,400 | 39.8 % | 0 of 64 | 0.155 | 144.1 | 433.1 | 9,215 |
| `WN` | words, expression built by the host reader | 6,400 | 61.5 % | 2 of 64 | 0.157 | 148.4 | 435.8 | 14,059 |

Exact share and exact steps per hour count every step after a slip as a miss, because the held state stays wrong; they are the page's measures and they are used as written. The next table separates the new mistakes.

## The serving numbers, as a company that runs the model counts them

The page's measures for a company that runs a model are seconds per step, computed prompt tokens per step and exact steps per GPU hour. Energy was not measured. The sixth run's cells were run in another Colab session, so the comparison with them is in the table but the size of any difference cannot be separated from the session.

| | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |
|---|---|---|---|---|
| sixth run `L0` (capped, coordinates only) | 0.100 | 94.6 | 331.0 | 35,113 |
| sixth run `LN` (named, coordinates only) | 0.101 | 94.9 | 331.2 | 35,764 |
| this run `C0` (coordinates + names list) | 0.160 | 156.2 | 442.7 | 22,513 |
| this run `W0` | 0.155 | 144.1 | 433.1 | 9,215 |
| this run `WN` | 0.157 | 148.4 | 435.8 | 14,059 |

* **Speed.** `WN` ran +1.3 % against `W0` with 775 different expressions in one run; building an expression for each request cost no seconds that this run can see (line 15).
* **Tokens.** The names list and the longer sentences put 112 more tokens in each prompt than the sixth run's `L0` (34 %), and 62 more computed prompt tokens per step (65 %), at 0.160 s per step against 0.100. That is a cost of putting the names in the prompt, in another session; it is not a cost of the grammar.
* **Exact steps per hour.** `C0` 22,513 against 35,113 for `L0` and 35,764 for `LN`: the same exact share as `LN`, fewer steps per hour because each step is slower and heavier.
* **Drift.** There was none with the request given as coordinates (`C0`). With the model reading the name, the drift is the first slip below, and it is a matter of reading a sentence, which the serving side does not do when the request arrives as coordinates.

## What the page's rule lets this Result claim

Lines 3, 4, 5 and 6 read yes. Line 2 reads NO: 4 of 19,200 replies used all 96 new tokens and were refused by the strict parser (the replies below). The cause is the capped expression, which allows an item list of any length: in these four replies the model kept appending `ADD[...]` and `DEL[...]` items until the token limit. They are `W0` job 193 step 81, `W0` job 205 step 95, `W0` job 243 step 63 and `WN` job 222 step 70, the last on a request the host did not read (edge: not two services), so it was under the capped expression too. `C0`, where the request is given as coordinates, has none in 6,400. The figures are therefore claimed as read for this setup (one model, one GPU, one run, 64 jobs, the sentences of the page), with the NO of line 2 beside them. The names are the table's (given, not learned); the host reader reads a sentence only where one service is named for a delete, two for an edge, or an added service with its coordinates, and refuses the rest, and the sentences are those of the page and not requests in the words of a user.

## Where the first slips are

The first inexact step of a job is the clean measure, because until then the state was exact. `W0`: 64 of 64 jobs slipped, median step 11, in the three thirds of the job 60 / 3 / 1. `WN`: 62 of 64 jobs slipped, median step 16, thirds 43 / 13 / 6. The table counts, for each kind of request, the first slips and the steps that were run exact up to and including the first slip.

| cell | request kind | pool | host read | first slips | steps run before the first slip | rate |
|---|---|---|---|---|---|---|
| `W0` | add_edge | first | - | 29 | 205 | 14.1 % |
| `W0` | add_edge | second | - | 24 | 68 | 35.3 % |
| `W0` | add_node | first | - | 0 | 207 | 0.0 % |
| `W0` | add_node | second | - | 2 | 68 | 2.9 % |
| `W0` | delete | first | - | 4 | 225 | 1.8 % |
| `W0` | delete | second | - | 5 | 93 | 5.4 % |
| `WN` | add_edge | first | right | 0 | 435 | 0.0 % |
| `WN` | add_edge | second | miss | 53 | 140 | 37.9 % |
| `WN` | add_node | first | right | 0 | 478 | 0.0 % |
| `WN` | add_node | second | right | 0 | 149 | 0.0 % |
| `WN` | delete | first | right | 0 | 474 | 0.0 % |
| `WN` | delete | second | miss | 9 | 184 | 4.9 % |

* The model's weak point is the edge: 29 of 205 (14.1 %) in the first pool and 24 of 68 (35.3 %) in the second. A delete is 4 of 225 (1.8 %) and 5 of 93 (5.4 %); a new service 0 of 207 (0.0 %) and 2 of 68 (2.9 %). The usual slip is a wrong pair of coordinates or the two ends swapped.
* In `WN` **none of the 1536 steps whose request the host read right was a first slip**; every first slip is on a request the host did not read (53 of 140 (37.9 %) for edges, 9 of 184 (4.9 %) for deletes). So the host's reading is safe where it reads, and the loss is the 1,020 requests it refuses. The page's line 12 reads NO (63.0 %) only because later steps of a job that slipped on an unread request stay wrong.
* In the second pool, 10 of the 28 first slips of `W0` carry the coordinates of the extra service the sentence mentions; the other 18 do not (prediction (c) did not come out).
* Aliases (prediction (f)): in `W0`, first slips at 12 of 66 edge and delete requests that use an alias, against 50 of 525 that do not (18.2 % against 9.5 %). The numbers are small; read it as "the aliases were not read better than the names" and nothing more.

## The predictions written beside the lines, read afterwards

| written first | probability | measured | came out |
|---|---|---|---|
| (a) `W0` holds the right state on between 90 and 98 percent of its 6,400 steps | 70 % | 39.8 % | NO |
| (b) more than half of the first slips of `W0` are on edge requests | 60 % | 53 of 64 | yes |
| (c) in the second pool the wrong lookups take the second service named in the sentence | 60 % | 10 of 28 second-pool first slips carry the extra service's coordinates; the other 18 do not | NO |
| (d) the seconds per step of `WN` are within 5 percent of those of `W0` | 80 % | +1.3 % | yes |
| (e) the three cells take 30 to 45 minutes in all, loading excluded | 65 % | 50.4 minutes | NO |
| (f) whether the aliases are read as well as the names (not a line) | - | in `W0`, first slips at 12 of 66 edge and delete requests that use an alias, against 50 of 525 that do not: 18.2 % against 9.5 % | read below |

## What this means, and what it does not

The serving side did what it was built for where the request arrives as coordinates: with the names list and no more, the exact state was held at every step. The new question was the front of the pipeline, a sentence in ordinary words, and for the 7B model on this T4 the answer is that it is not good enough by itself (39.8 %), and that a fixed host reader that refuses what it cannot read helps (61.5 %) without being wrong once, but stops at the requests it refuses. The finding is a limit of this model and this reader, not a failure of the serving numbers.

## What this does not show
The words of a real user (the sentences are the ones on the page, written by me, in two pools); names the host did not give; another language; tasks of more than 100 steps; real company tasks; another model, size, GPU, engine or cache size (a bigger model is the next test, with its own page); a retry on top of a grammar; a reader that handles the second pool; energy; dollars; a capacity in jobs per GPU; hallucination in free text. Nothing here is proven.

## What comes next
A test on a bigger model on a GPU of the L4 class or a v6e-1 TPU, with its own page locked first, measuring drift, seconds per step, latency per request and tokens (and energy where the hardware reports it), written from the side of a company that runs the model. The cost in compute units is told before anything is run.

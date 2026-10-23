# AXI: the technical reference

*24 Oct 2026. One document that says what AXI is, how the code in this repository does it, what has been measured and what has not. The architecture is **Rajnish Choubey's**. The code, the tests and the measurements are the owner's, with an AI assistant as a tool; the assistant is not an author and owns nothing here. Every table in section 4 is generated from the saved run files by `python/tests/test_arc_fuzzy_results.py` (and, for the long session, `python/tests/test_gemini_long_result.py`); a test fails if this page and the files disagree. Where a sentence is an arithmetic illustration, or a design intent, or something the repository has not shown, it says so in the sentence.*

## Contents

1. What AXI is for
2. The architecture, as the code has it
3. How the claims are tested
4. What has been measured
5. The company side: what it adds up to
6. The realities: where AXI does not help
7. What is not shown
8. What comes next
9. Reproduce it
10. Credits

---

## 1. What AXI is for

A company that runs a language model for others, or inside its own products, counts a small number of things: whether the model **drifts or hallucinates** over a long task, how many **tokens** each step computes, how long a step takes (**speed**), how long a request waits for its first word and for its last (**latency**), how steady those are from the first request of a job to the last, and what a wrong step costs to find and to undo. Joules are the real unit of the energy bill, and the API gives none; this repository counts **tokens computed** and **seconds of calls** as the stand-in and says so wherever it uses them.

The usual way to give a model a long task is to keep the chat: every request and every reply stay in the prompt and are sent again at the next step. That has three properties that the runs below measure:

* **The prompt grows with the job.** The tokens sent in all grow with the square of the number of steps, and the time to the first token and the time of a step grow with the prompt.
* **The state lives in the chat.** Whatever the model said last is what the job holds. A wrong change that nothing checks stays in the state until a later message happens to overwrite it.
* **Nothing outside the model checks a step.** A reply the model writes with a wrong item, an impossible value or a dangling reference is applied as it is written.

AXI changes where the state lives. **The host owns the state.** The model is asked for one proposed change at a time, in a small strict language; the host parses the proposal, checks it against the exact state, repairs what a rule can repair, commits all of it or none of it, and shows the model the exact state again at the next step, in a prompt that does not grow with the job. In the repository's words, the model is a proposer and the host is the keeper.

AXI works at inference time. It does not train or change a model. It was built for the party that runs the model (the supplier side); it is not a prompt trick for the party that only calls it.

### 1.1 An arithmetic illustration of compounding (not a measurement)

If each step were right independently with probability *p*, the chance that *n* steps in a row are all right is *p* to the power *n*. With *p* = 0.9885 (the share of replies that were right from a correct state in the 4,000-step run: 3,954 of 4,000) and *n* = 100, that is about **31 percent**; with *n* = 400 it is below **1 percent**. Steps are not independent in a real job, so the number shows the shape of the effect and not a rate, and the runs in section 4 do not show a drift of that size for the kept chat at the lengths tried: in the 400-turn conversation 398 of 400 replies were exact. What the runs do show is the other half of the arithmetic: **one** slip that nothing catches stays in the state, and it stayed for 140 turns (section 4.3). The page uses the formula to say why a single slip matters, and does not use it to claim a rate.

---

## 2. The architecture, as the code has it

The code lives in `python/axi/engine/` (the Python reference that the tests run), `python/axi/experiments/` (the harnesses of the runs) and `crates/` (a Rust sketch of the same design, section 2.12). Section numbers below follow the order a proposal travels.

### 2.1 The state

The state is a **typed graph**: nodes `{(r, c): value}` and edges `{(u, v, relation): weight}`, with an incidence index so that "which edges touch this node" is a lookup (`engine/gate.py`, class `Graph`). Nodes are named by a **coordinate** `(r, c)`, not by a position in a token stream. That is the design's central choice: the state is addressed by what it is, so editing one item changes nothing about how the others are addressed, and the same state can be shown to the model in any order. The module `engine/cico_attention.py` contains the corresponding model-side idea (attention scores from coordinates, with no sequence position anywhere); the repository tests that it is exactly permutation-equivariant and states that it says nothing about whether a trained model drifts less.

Services in the harness jobs are graph nodes; a dependency is an edge with a relation name and a weight. The same shape holds a list of items with facts (Test 23 keeps the list in a plain dictionary of items to facts, with its own check, section 2.11) and a record of moves in a game (Tests 24 and 25).

### 2.2 The delta language (CICO)

A proposal is a **delta**: a list of items in a strict language (`engine/cico_parser.py`).

```
ADD[r,c:val]                      node add     (val may be negative)
DEL[r,c]                          node delete
ADD[(r,c)->(r,c):rel#w]           edge add     (rel = [A-Za-z0-9_]+, w may be negative)
DEL[(r,c)->(r,c):rel#w]           edge delete
```

Items are separated by exactly one space or newline, with one optional trailing separator. **Anything else raises `CICOParseError`; nothing is silently skipped.** The state can be shown to a model as `<CICO_STATE> NODES: [r,c:v] ... EDGES: (a,b)->(c,d):rel#w ... </CICO_STATE>` (`CICOParser.serialize_state`). The language has no in-place update: changing a value is a delete and an add, and the module `engine/inverse.py` records that SET is not part of the language yet.

Why strict: a parser that accepts "roughly right" text makes the next stage guess. Here the parse either yields a delta that means exactly one thing or it refuses, and the refusal is a stage name (`parse`) the host can count.

### 2.3 Constrained decoding (the model-side half of "strict")

The strict parser can also be run **before** the text exists, as a constraint on what the model may emit. `engine/cico_decoder.py` builds a byte-level DFA that recognises exactly the strings the parser accepts, and a `TokenMasker` that lifts it to a real token vocabulary: a token is allowed in a DFA state if feeding all of its bytes never reaches the reject state; masks are found by walking a vocabulary trie in step with the DFA and cached per state. `engine/gpu_mask.py` precomputes, once, `allowed[state, token]` and `next[state, token]` tables so that a decode step on a device is a masked fill and an index (no host round trip per token); for a 37-state grammar and a 50,000-token vocabulary the tables are about 5.6 megabytes, and the table grows with the number of DFA states, so the method suits small grammars like this one.

Status, honestly: the equivalence of the table and the host masker is tested; the RFC's device-mask speed claims (about 10,000 times faster, under 0.3 percent overhead) were **not reproduced**. Against a simple host round trip the mask went from 287 to 77 microseconds (batch of 1) and from 8,415 to 392 (batch of 64), about 4 and 21 times (`docs/VERIFIED.md`). The experiments in section 4 do not use constrained decoding for the Gemini runs (an API model cannot be masked from outside); they use the strict parser on the reply text and re-ask.

### 2.4 The gate

`engine/gate.py` is a DPO-style (double-pushout graph rewriting) gluing check. A delta is admitted **if and only if** all four conditions hold on the current graph:

| condition | what it refuses |
|---|---|
| **match** | a DEL of a node that is not in the host, or of an edge that is not there or has another weight |
| **identification** | an ADD of a node or edge that already exists; an item repeated within the delta; the same node or edge both added and deleted |
| **dangling edge** | deleting a node while an edge that touches it is not also deleted in the same delta |
| **well-formedness** | adding an edge whose endpoint will not exist after the delta |

The check is exact and takes about 2 microseconds in Python (median, one CPU; `docs/LATENCY_7B.md`), against 0.97 to 2.49 seconds for one model call in the same measurement. `apply` then performs an admitted delta in a fixed order: delete edges, delete nodes, add nodes, add edges. The gate stops at the first obstruction and names it (`GateResult.reason`).

What the gate is **not**: it answers "is this change legal on this state", not "is this the change that was asked". A legal change can still be the wrong one (a valid day that is not the true day); sections 2.6 and 6 come back to that.

### 2.5 Diagnostics and repair

`engine/diagnostics.py` goes beyond the first obstruction.

* **`diagnose`** collects *all* obstructions of a delta against a graph, each typed: `dangling`, `match_node`, `match_edge`, `match_weight`, `ident_node`, `ident_edge`, `ident_conflict`, `malformed`. A dangling obstruction carries the edges still attached.
* **`normalize`** drops only items that are provably redundant: exact duplicates, and an ADD of something already present with the identical value or weight.
* **`propose_repair`** repairs dangling-edge obstructions deterministically and **only** those: *primal* (add a DEL for every undeleted incident edge, used when the node has at most 4) or *dual* (keep the node by dropping its DEL, used above that threshold). Anything that is not a dangling obstruction is reported as not repairable. It never guesses.
* **`ground_and_complete`** is the larger repair, in three steps: *cut* (drop operations that name things not in the real graph, duplicates, an add-then-delete of an existing node), *ground* (correct an edge weight to the real one when the edge is named correctly and only the number is wrong), *complete* (close the boundary with the primal or dual rule). It never invents a node operation and never adds an edge the model did not propose, except the DELs of real incident edges. It returns `None` for an already-admissible delta, so a clean proposal passes through untouched, and it re-runs the gate on its own result before it hands it back.

The repair is a rule written for the job (a deleted service takes its dependants with it). The model is not told the rule in the 4,000-step run; the layer works when the rule is known and does not find rules.

### 2.6 Scope: keeping only what the request is about

`engine/scope.py` handles a failure the gate cannot see: a legal delta that touches things nobody asked it to touch (a model told the rule deletes real edges nobody named). The scope filter reads the **request** for the things it names and keeps only changes on them:

* a node ADD or DEL is in scope if the node is named;
* an edge ADD needs both endpoints named;
* an edge DEL needs both endpoints named, or one endpoint that is a named node the same delta deletes (the edges that must go with it).

Everything else is dropped and reported. With the request naming nothing ("remove every edge with weight 4") there is no scope to apply and the path is plain ground-and-complete. After the filter, an admissible delta is applied; an inadmissible one is grounded and completed; if nothing is left, nothing is applied (`admit_scoped`).

"Named" in the base version is a coordinate written in the text. `engine/grounding.py` extends it to **words**: the graph carries a table of canonical names and aliases, and the request is read against it (whole-word, longest match first, no overlapping spans; leftover words that are part of exactly one service's name name that service). A phrase that fits several services makes the request *ambiguous* and a phrase that fits none makes it *unresolved*; in both cases **nothing is applied**. The module says plainly that this is exact and token-overlap matching against a table the host supplies, not language understanding.

### 2.7 The runtime pipeline

`engine/runtime.py` is the end-to-end path of one proposal, and it is short enough to quote:

1. **Parse** the raw model text strictly. A failure is `Outcome(False, "parse", reason)`.
2. **Begin a transaction** on the pager and perform **speculative page writes** for every node the delta adds and every node an added edge touches (the stand-in for "prefill the new state into fresh cache blocks"). If there are no free blocks the transaction is rolled back (`"memory"`).
3. **Check** the delta with the gate. On failure, roll back and return the stage `gate`, the reason, and, if a deterministic repair exists, the **full repaired proposal** as CICO text, ready to resubmit.
4. **Unmap** the pages of deleted nodes, then **commit** the transaction. A conflict (another commit touched the same pages since) is `"commit"`.
5. **Apply** the delta to the graph.

So a proposal is either committed whole, or nothing changes and the outcome says at which stage and why. The harnesses of section 4 wrap this idea in a loop: parse, check, if refused ask again up to twice (with the refusal and, where there is one, the repair), and if still refused apply nothing.

### 2.8 The pager and the atomic commit

`engine/pager.py` is the pure-Python model of a **copy-on-write pager**. Physical blocks are in exactly one of three places: free, mapped, or held by a live transaction. `audit()` checks `free + mapped + in_flight == capacity` with no overlaps; a leak or a block in two places fails it. `begin` is O(1), `write_page` O(1), `commit` and `rollback` O(*t*) in the blocks the transaction touched, and `commit` is a short locked update that raises `Conflict` if a page's mapping changed since the transaction read it. `prepare` is the first phase of a two-phase commit. A page number comes from a node's coordinate (`vpn_of(n) = r * 65536 + c`), so "the page of this node" does not depend on where the node sits in any stream.

`engine/dist_commit.py` is the reference for **all-or-nothing commit across tensor-parallel ranks**: every rank makes speculative writes, each votes (allocation ok, prepare ok), the consensus is the gate's verdict AND every vote (an all-reduce MIN over a byte), and then all ranks commit or all roll back. A negative control, `NaiveIndependentRanks`, shows what happens when each rank decides alone.

What this does and does not show: the reference proves the *bookkeeping* (no block lost or doubled, atomic outcome, conflicts caught). It is not a measurement of a GPU cache; the harness runs in section 4 do not touch a GPU cache at all, because they run against an API.

### 2.9 Undo

`engine/inverse.py`: `invert(g, d)` returns the delta that undoes an admitted delta `d` (ADD node becomes DEL node; DEL node becomes ADD with the value it had; ADD edge becomes DEL with the weight added; DEL edge becomes ADD with the weight matched). Read as signed counts per item, the inverse is the negative of the delta, so a delta followed by its inverse adds up to zero. It must be called with the graph *before* the delta is applied, because a deleted node's value exists only there. Together with the gate this is the basis of an audit trail: every admitted change has a computable, checkable undo.

### 2.10 The wire format

`wire.py` implements the reference codec of RFC-001's axi-wire: a 32-byte big-endian header (magic `AXI\x01`, mode, compression, schema, transaction id, node count, edge count, payload length), a payload padded to eight bytes, and a **CRC-32C** trailer (Castagnoli; the module notes that `zlib.crc32` is a different checksum). The payload is either a coordinate-tuple list (16 bytes each: source, destination, relation, flags, weight) or a bit matrix; `choose_mode` picks between them by density. This is the format for moving a state between hosts. The harness runs do not use it.

### 2.11 How the harnesses use the pieces

The four Gemini runs in section 4 share one pattern and differ in what "the state", "the check" and "the record" are. The pattern: **arm K** is the chat kept (every request and reply stays in the prompt; each reply is applied as it comes, no check, no re-ask) and **arm A** is the host-held state (nothing is kept in the chat; each step the prompt is the exact current state plus the request; the reply goes through a check; up to two re-asks; a reply that still fails changes nothing). The same model, the same seed, the same script of requests, and a record of every call (prompt tokens, tokens served from the cache, reply and thinking tokens, seconds, seconds to the first token).

| run | the state | the check of arm A | other arms |
|---|---|---|---|
| Test 22 (long session) | the dependency graph; 40 lines of routine log with every request | the gate, the scope filter and the repair of sections 2.4 to 2.6 (the `scope_m7` policy of the 4,000-step run) | D: a depth probe with a perfect history |
| Test 23 (conversation) | a list of items with facts (owner, day, amount, status), built by a script, with 30 sentences of office chatter around every message | **a check written for this list in the test module** (the item must be there for a change and absent for an add; the field and the value must be in the allowed set); the framework's gate is not used | D: depth probe |
| Test 24 (ARC-AGI-3) | the game's own screen plus a Record the framework keeps from the game's frames | a legality check (the action must be one the game offers); the re-ask | none |
| Test 25 (ARC-AGI-3, explorer) | as Test 24, plus an object view of the screen and a list of untried moves | as Test 24 | the rows of Test 24 for A and K |

This table is the main thing to read before quoting a result: **the DPO gate is what runs in the dependency-graph jobs; the other jobs use the same pattern (host-held state, check before commit, re-ask, nothing kept in the chat) with a check that fits their state.** The Test 23 and Test 24 results therefore say something about the pattern, and nothing about the gate's four conditions on those states.

Every reply in every run is marked by an **exact match with what the script knows** (the oracle's state, the game's own screen), never by another model's opinion.

### 2.12 Around the core: what else is in the repository

* **Block-independence masks** (`engine/block_mask.py`): attention masks in which a block sees its own block plus interface keys, with a tiny model to test that editing one block does not disturb the others.
* **Symmetry canonicalization** (`engine/c4.py`, `engine/d4.py`): C4 and D4 canonical coordinates for 2-D grids. D4 (the eight rotations and mirrors) is a tested rewrite of a blueprint idea: a prompt built from a canonical task is byte-identical for all eight orientations, and one transform per *task* (not per pair) is needed for orientation-dependent rules. The module states that it does **not** show a model solves more ARC tasks.
* **The Rust crates** (`crates/`): a partly wired sketch of the same design. `axi-ir` includes four small modules (coordinate tuples, schema, router, positional encoding); the graph-rewrite check (`axi-dpo`) and the shadow pager are written but not yet included in a crate; `axi-cache` is a placeholder. The gate and the pager that the tests run are the Python ones. The Rust code is not timed.

---

## 3. How the claims are tested

The method matters as much as the numbers, because it is what lets a reader trust a table without trusting the author.

1. **The page comes first.** For each model run a page fixes the question, the **predictions** (with a probability on each reading line), and the **reading rules** (exact thresholds) before any call is made. The page is hashed (SHA-256) and the hash is written into the test that guards it.
2. **The Result goes below the line.** After the run, the result is appended below a marker at the end of the page. The text above the marker is not changed; a test checks that its hash is the hash taken before the run, and that the page equals *the locked text plus the render of the saved file*. Nothing in a Result is typed by hand.
3. **A NO is printed as a NO.** A reading line that comes out "NO" is a finding. Of the 24 lines of Test 23, 9 read NO; of the 19 of Test 24, 8; of the 17 of Test 25, 6 and one was not read. The pages say so.
4. **Every figure is recounted from the rows.** The result test re-derives each quoted number from the saved rows by an independent count (the slips, the run of turns a wrong list was held, the unusable replies, the levels, the sums of tokens), and checks that the 24, 19 and 17 lines recomputed from the rows are the ones the notebooks saved.
5. **Every saved reply is replayed.** The fuzzy runs replay every saved reply of every arm through the stream and the policies and require every saved flag to come out the same (guard lines 1 and 2). The ARC replays need the games of the ARC toolkit, which the test environment does not have; the notebooks ran them and saved the flags, and the test reads the saved flags.
6. **Guards before claims.** Lines 1 to 6 of each page check that the run *is* the run the page describes (it finished, it is on file, the instrument saw the cache). If a guard or an instrument line reads NO the page says no claim is made until the cause is found. In Test 24 line 6 read NO (13 of 240 replies of the kept chat were not usable); the Result gives the cause from the saved rows.
7. **The wording rules.** No money figure on the newer pages, no multiple of capacity, no count of devices saved, no claim that a grammar is complete or perfect, no claim of proof, no statement about any company's service, no statement about the words of a real user, and energy only as the stand-in. A test enforces the list on each Result.
8. **Tests are tested.** Many tests are also run against deliberately broken copies of the code (mutants); `docs/VERIFIED.md` gives the number caught where this was done. For the newest tests (135 to 145) the mutation pass has not been run yet; that is listed in section 8.

`docs/VERIFIED.md` is the status table: what is measured, what is only checked by tests, and what is **not** measured.

---

## 4. What has been measured

### 4.1 Earlier results (smaller models, synthetic jobs)

Every figure in this subsection is on the page named beside it (`docs/WHERE_THIS_STANDS.md` collects them, and a test checks that each figure is on its page).

| What | Result | Page |
|---|---|---|
| Qwen2.5-3B, grammar-constrained, no rule in the prompt, 4 seeds | Applied as written, the output corrupted the graph on 120 of 120 "delete a service that still has dependants" tasks; with the gate, 0; the deterministic repair got every task right (240 of 240) | `VERIFIED.md`, `EXPERIMENT_RESULTS.md` |
| Qwen2.5-7B, 30 requests in a row, 12 episodes | Applied as written, the state was corrupted in 12 of 12 episodes; the gate corrupted it in 0 of 360 steps; the gate with the repair was exactly on the intended state on 358 of 360 steps | `LONG_HORIZON_TEST.md` |
| Qwen2.5-7B, rule stated, fresh seeds, with the scope filter | Exactly the intended state on 354 of 360 steps (98.3 percent) against 263 of 360 (73.1 percent) for the repair alone; never corrupted | `SCOPE_TEST.md` |
| Qwen2.5-14B | The gate, the repair and the scope filter corrupted nothing in 840 single requests and 600 long steps; applied as written still corrupted 24 of 140 graphs with no rule. The gain is smaller than at 7B, not gone | `SCALE_TEST_14B.md` |
| Check cost | The gate takes about 2.0 microseconds and the diagnosis about 3.5, against 0.97 to 2.49 seconds for one call | `LATENCY_7B.md` |
| The one real price | A task the gate rejects and the model must retry takes 2 to 5 times as long as one call | `LATENCY_7B.md` |
| Growing chat, 7B | Keeping the whole chat made each step 5.1 times slower by the last quarter; the 7B on the T4 ran out of memory near 5,500 tokens; showing the exact state each step stayed flat | `LONG_HORIZON_TEST.md` |
| Tokens, 7B, 105 steps | Showing the exact state each step used 39,225 tokens against 304,289 for keeping the whole chat (87.1 percent fewer) | `ENDURANCE_TEST.md` |
| The 4,000-step run, Gemini 3.8 Flash | With AXI the state was exactly right at **4,000 of 4,000** steps, never corrupted, at about the same prompt size from the first tenth to the last (1.16 times) and the same speed (0.94 times). The model's reply was right 3,954 times (98.85 percent); all 46 slips were of one kind (a service deleted without its other connections) and a fixed repair completed 46 of 46. On the first 400 requests, replies applied as written were exactly right on 85 steps of 400 and corrupted on 223; a gate that only refuses bad replies was exactly right on 84 and off the right state from step 210 to the end | `MILLION_TOKEN_TEST_HARD.md` |
| Tensor-level readings (spinor turn, hyperbolic snap, Betti numbers of the history) | Our small readings of an unattributed roadmap did **not** beat simple baselines (for example the spinor turn 2.3 percent exact at 100 services against 76.8 percent for a rotary baseline). Those results say nothing about the architect's own definitions | `TOY_MODEL_TEST.md`, `HYPERBOLIC_SNAP_TEST.md`, `HISTORY_SHAPE_TEST.md` |

### 4.2 The four Gemini 3.8 Flash runs of 22 to 24 October 2026

The model is `gemini-3.8-flash`, streamed, default thinking, in all four. One run of each arm, one seed (Test 22: 20261111; Test 23: 20261122; Tests 24 and 25: game seed 0). The tables below are generated from the saved files in `docs/results/`.

What the columns mean: *seconds a step* is the mean seconds of the calls (the waits the key's per-minute limit made the notebook take are not in it); *first token* is the seconds from the request to the first streamed token, median and 95th percentile, read at the notebook; *served from the cache* is the share of prompt tokens the service reports as cached; *computed tokens per exact step* is the prompt tokens not served from the cache, plus reply and thinking tokens, per step that was exactly right. **Computed tokens are the stand-in for energy, not joules.**

<!-- measured:start -->
### Test 22, the long session on a dependency graph (1,500 steps with AXI, 457 turns of the kept chat)

| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| kept chat, all turns | 457 | 7.13 | 505 | 488 | 5.06 | 13.98 | 476,835 | 99.1 % | 160 | 4,728 |
| kept chat, prompt of 500,000 tokens or more | 217 | 10.66 | 338 | 316 | 9.13 | 17.49 | 725,061 | 99.3 % | 279 | 5,419 |
| with AXI, all steps | 1500 | 3.07 | 1,171 | 1,157 | 2.27 | 6.17 | 2,824 | 0.0 % | 226 | 3,108 |
| with AXI, the first 457 steps | 457 | 2.79 | 1,292 | 1,280 | 2.21 | 5.80 | 2,779 | 0.0 % | 198 | 3,023 |

### Test 23, the conversation in ambiguous text (400 turns)

| arm | steps | seconds a step | steps an hour | exact steps an hour | first token median | first token 95th | prompt tokens a step | served from the cache | thinking tokens a step | computed tokens per exact step |
|---|---|---|---|---|---|---|---|---|---|---|
| kept chat, all turns | 400 | 3.64 | 989 | 984 | 2.93 | 6.55 | 99,809 | 97.0 % | 3.3 | 3,035 |
| kept chat, turns 1 to 50 | 50 | 2.24 | 1,609 | 1,609 | 1.86 | 3.71 | 13,094 | 51.8 % | 1.8 | 6,338 |
| kept chat, prompt of 150,000 tokens or more | 99 | 5.70 | 631 | 625 | 5.57 | 7.43 | 174,384 | 98.6 % | 11.3 | 2,576 |
| with AXI, all steps | 400 | 2.76 | 1,306 | 1,306 | 2.41 | 3.75 | 1,177 | 0.0 % | 111.5 | 1,322 |
| with AXI, steps 1 to 50 | 50 | 3.19 | 1,129 | 1,129 | 2.35 | 3.62 | 959 | 0.0 % | 109.3 | 1,102 |

| turns | kept chat: exact reply | kept chat: held list exact | kept chat: prompt a turn | kept chat: seconds a turn | AXI: held list exact | AXI: prompt a step | AXI: seconds a step | AXI: first token median |
|---|---|---|---|---|---|---|---|---|
| 1 to 40 | 100.0 % | 100.0 % | 10,600 | 1.98 | 100.0 % | 946 | 3.39 | 2.46 |
| 41 to 80 | 100.0 % | 100.0 % | 30,416 | 2.82 | 100.0 % | 979 | 2.38 | 2.25 |
| 81 to 120 | 100.0 % | 100.0 % | 50,246 | 2.44 | 100.0 % | 1,040 | 2.84 | 2.59 |
| 121 to 160 | 100.0 % | 100.0 % | 70,112 | 3.05 | 100.0 % | 1,033 | 2.80 | 2.65 |
| 161 to 200 | 97.5 % | 75.0 % | 89,978 | 2.92 | 100.0 % | 1,149 | 2.98 | 2.77 |
| 201 to 240 | 100.0 % | 0.0 % | 109,674 | 3.20 | 100.0 % | 1,276 | 2.72 | 2.52 |
| 241 to 280 | 100.0 % | 0.0 % | 129,526 | 3.71 | 100.0 % | 1,335 | 2.47 | 2.29 |
| 281 to 320 | 100.0 % | 0.0 % | 149,423 | 4.40 | 100.0 % | 1,340 | 2.74 | 2.50 |
| 321 to 360 | 100.0 % | 75.0 % | 169,306 | 5.51 | 100.0 % | 1,339 | 2.57 | 2.32 |
| 361 to 400 | 97.5 % | 100.0 % | 188,806 | 6.39 | 100.0 % | 1,332 | 2.69 | 2.41 |

### Test 24, four ARC-AGI-3 games, 60 moves each

| arm | moves | seconds a move | moves an hour | effective moves an hour | first token median | first token 95th | prompt tokens a move | served from the cache | thinking tokens a move | computed tokens per effective move |
|---|---|---|---|---|---|---|---|---|---|---|
| kept chat, all moves | 240 | 15.87 | 227 | 200 | 5.90 | 52.14 | 110,948 | 94.8 % | 1,714 | 8,452 |
| kept chat, prompt of 100,000 tokens or more | 110 | 18.26 | 197 | 165 | 5.96 | 61.65 | 174,928 | 96.8 % | 2,039 | 9,194 |
| with AXI, all moves | 240 | 28.73 | 125 | 119 | 21.49 | 52.77 | 4,256 | 0.2 % | 3,863 | 8,550 |

| game | levels to win | arm K: levels | arm K: effective | arm K: largest prompt | arm A: levels | arm A: effective | arm A: largest prompt |
|---|---|---|---|---|---|---|---|
| ls20-9607627b | 7 | 0 | 86.7 % | 251,033 | 0 | 100.0 % | 4,631 |
| g50t-5849a774 | 7 | 0 | 71.7 % | 253,440 | 0 | 85.0 % | 4,665 |
| ka59-38d34dbb | 7 | 1 | 91.7 % | 230,894 | 0 | 95.0 % | 4,545 |
| sp80-589a99af | 6 | 1 | 100.0 % | 93,478 | 1 | 100.0 % | 2,030 |

### Test 25, the same games with the host-owned explorer

| arm | moves | played by the framework | effective moves | repeat no-ops | screens never seen before | per 1,000 computed tokens | levels | thinking tokens a call | seconds a move | effective moves an hour | computed tokens | tokens sent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B (explorer) | 240 | 0 | 95.4 % | 0 | 221 | 0.124 | 1 | 2,773 | 23.96 | 143 | 1,784,208 | 1,790,904 |
| A (Test 24) | 240 | 0 | 95.0 % | 0 | 227 | 0.116 | 1 | 3,566 | 28.73 | 119 | 1,949,418 | 1,951,121 |
| K (Test 24, kept chat) | 240 | 0 | 87.5 % | 1 | 208 | 0.122 | 2 | 1,719 | 15.87 | 200 | 1,698,881 | 25,689,461 |

### The arithmetic used in section 5 (from the rows of Test 23)

The prompt of the kept chat grew by about 495 tokens a turn from 849 tokens at turn 1. A straight line with that growth gives 39,850,800 prompt tokens over 400 turns; the rows add up to 39,923,458 (0.2 % apart). With AXI the prompt a step was 1,177 tokens on average. The sum of a growing prompt grows with the square of the number of turns: the same line run on to 800 turns gives 158,922,051 tokens, 4.0 times as many for twice the turns, while a step that does not grow gives twice as many. This second figure is arithmetic on a straight line, not a run.
<!-- measured:end -->

### 4.3 What the tables say, run by run

**Test 22, the long session (dependency graph, 40 lines of log with every request).** With AXI the state was exact on all 1,500 steps at a flat prompt of about 2,800 tokens a step; the kept chat (457 turns, up to about 935,000 tokens) was exact for its first 222 turns, then slipped on 15 turns, its held state was corrupted from turn 287 on, and its seconds a call and time to the first token grew with its prompt (the first token went from 2.2 seconds at the smallest prompts to 9.1 seconds past 500,000 tokens, against 2.3 seconds with AXI). The depth probe, with a perfect history, was exact everywhere: the slips of the kept chat are not shown to be caused by length alone. The company page with the comparison table is `docs/AXI_FOR_COMPANIES.md`; the Result is on `docs/GEMINI_LONG_TEST.md`.

**Test 23, the conversation in ambiguous text.** The objection this test answers: the state of Test 22 is a rigid graph, and real conversational work has a state that is ambiguous text. With AXI the held list was exact on all 400 steps at about 1,200 prompt tokens a step (the prompt grows only with the list). The kept chat did **not** fail the way Test 22 predicted: 398 of 400 replies were exact, none was corrupted, and answers about old facts were within the page's threshold of answers about new ones. What it did show is the following. At turn 191 the reply of the kept chat applied a change to the wrong item (`"the last one we added"` resolved to the wrong item); nothing checked it, and a firm that applies each reply as it comes **held a wrong list for 140 turns in a row** (turns 191 to 330), at most two cells from the true list, until a later message happened to overwrite the cell. The list was exact on 65 percent of the turns although 99.5 percent of the replies were exact. The reply accuracy and the state accuracy are different numbers, and the second is the one a company's product depends on. On time: at a short prompt the kept chat was the faster arm (1,609 exact steps an hour in the first 50 turns against 1,129), and from about turn 120 on it was slower; past 150,000 tokens its first token took 5.6 seconds against 2.4 with AXI. Over the whole job AXI made 1.33 times the exact steps an hour (the page asked for 1.5, and line 21 reads NO).

**Test 24, four ARC-AGI-3 games, 60 moves each.** ARC-AGI-3 is an interactive benchmark: the model sends one of the actions a game offers and sees the next screen. The test is of what the host-held Record and the legality check do to the cost and steadiness of the model on a hard game, and the page says beforehand that it is not a leaderboard entry. AXI sent 13.2 times fewer tokens, ended no move with an unusable reply (the kept chat: 13 of 240, nine of them empty replies after long thinking, two cut at the token limit, two with the wrong wording), and had a higher share of moves that changed the screen (95.0 against 87.5 percent). It was **slower** (28.7 seconds a move against 15.9) and completed **fewer levels** (1 against 2): with the current screen and a short Record in front of it the model thought more (3,863 tokens a move against 1,714). The run does not say why.

**Test 25, the same games with the host-owned explorer.** Notes shared with the owner named further structural layers (an invariant address frame, typed operators, an outer-loop conductor, a progress score, boundary completion). Whether those names match the architect's own writing is for the architect to confirm; this page claims no match. The test took the parts that fit a game that offers only its own actions and no target grid: an **object view** the framework computes from the frames (same-colour cells that touch: colour, size, box, one cell inside it, at most 12) and the exact box of what the last move changed; a **list of untried moves** from this exact screen; and a **conductor** that plays an untried move itself, with no call to the model, when the last 4 moves reached no new screen or the screen has been seen 3 times. The framework's core was not changed. Result: 95.4 percent effective moves against 95.0, the same number of levels as arm A (1, won in a different game), 22 percent fewer thinking tokens a call, 1.2 times the effective moves an hour, 6 fewer screens never seen before: all below the page's thresholds (6 lines NO, headline MIXED). The conductor **never fired**: the model was never stuck by the rule, so the part that acts on the model's behalf is untested.

---

## 5. The company side: what it adds up to

This section collects what a company that runs the model gets from the pattern, in the terms the company counts, using only the tables above. Where a number is arithmetic and not a run, it says so.

**5.1 State safety.** With the host holding the state and checking each step, the held state was exact at every step of Tests 22 and 23 and at the 4,000 steps of the hard run (distance from the true state 0). Without the host's state the kept chat's state was exact until its first uncaught slip and wrong from then on until something overwrote it (Test 22: from turn 287, for the rest of the job; Test 23: for 140 turns). The number a product depends on is the share of the time the *state* is right, and it can be far below the share of replies that are right (99.5 percent against 65 percent in Test 23).

**5.2 Tokens, and the shape of the bill.** The prompt of the kept chat grows with the job, so the tokens sent in all grow with the square of the number of steps (section 4.2 gives the straight-line arithmetic: for the Test 23 growth rate, twice the turns gives four times the tokens). With AXI the prompt is the size of the state and a step costs the same at step 1 and at step 1,500. Measured: 85 times fewer prompt tokens a step and 76 times fewer tokens sent in all over 400 turns (Test 23); 169 times fewer a step and 47 times fewer in all over the long session (Test 22); 13 times fewer over the ARC games (Test 24). **The caveat that belongs with these numbers** is the cache: the service served 97 percent of the kept chat's prompt tokens from its cache, so the kept chat's *computed* tokens are far below its *sent* tokens. Counted in computed tokens per exact step the ratios are 2.3 (Test 23, whole job) and 1.7 (Test 22, past 500,000 tokens), not 76 or 169. Which of the two a company's bill follows depends on how the company prices cached tokens, and this repository does not price them.

**5.3 Latency and its steadiness.** For a service-level target the useful property is that the time to the first token and the time of a step do not move as a job gets longer. With AXI the median first token was flat (2.4 seconds in Test 23, between 2.3 and 2.8 in every tenth; 2.3 seconds in Test 22) while the kept chat's rose with its prompt (1.9 seconds in the first band to 5.6 in the last in Test 23; in Test 22 2.2 seconds in the first band and 9.1 over all the turns past 500,000 tokens), and its 95th percentile was higher (7.4 seconds against 3.8 in Test 23). The caveat: at a short prompt the kept chat was as fast or faster, because AXI's step carries the model's thinking over the state; the advantage begins where the prompt gets large (in Test 23, past turn 120 or about 70,000 prompt tokens).

**5.4 Governance and audit.** Every change to the state passes a deterministic check with a typed reason when it fails; a failure leaves the state untouched; an admitted change has a computable inverse (section 2.9); the commit is all-or-nothing (section 2.8). For a company that runs agents for customers this is the part that does not depend on how well the model behaves on a given day: the same proposal on the same state gets the same verdict. What the pattern does not do is decide whether a *legal* change was the *right* one (section 6).

**5.5 Fewer re-runs and clean-ups.** In the runs, the AXI arm used no re-asks in Tests 22 and 23 and a few in Tests 24 and 25 (20 and 7 moves asked again), and no reply was left unusable. A state that is right at every step does not need a clean-up job, a re-run or a support case; that saving is indirect and the repository does not put a number on it.

---

## 6. The realities: where AXI does not help

These are measured, they are on the pages, and a company should weigh them equally with section 5.

* **The kept chat did not drift at 400 turns of ambiguous text** (Test 23), and the model's reply accuracy was 99.5 percent. If the job is short enough, or the model good enough, a kept chat may be as exact as AXI. In Test 22 the kept chat was exact for its first 222 heavy turns. The finding is about what happens after the first uncaught slip, and about cost and latency at length, not that the kept chat is bad from the start.
* **On ARC-AGI-3, AXI was slower and won fewer levels.** The state-holding arm made the model think more per move (3,863 tokens against 1,714), took 28.7 seconds a move against 15.9, and completed 1 level against 2 in 240 moves. Neither Test 24 nor Test 25 shows that AXI makes the model better at a hard open-ended game; the explorer did not change the level count.
* **The checks can only refuse what is not allowed.** The gate and the Test 23 check cannot know that a valid value is the wrong one. Arm A made no wrong-but-valid reply in Test 23, so that run does not show what the check does with one; a legal wrong change would be committed.
* **Cache pricing changes the token story.** See 5.2. A company with a cache that serves 97 percent of a growing prompt pays much less than the sent tokens suggest; the saving against a chat that uses prefix caching or a summary may be smaller than the sent-token ratio. Section 4.2 gives both figures.
* **Thinking is not free.** In Tests 24 and 25 the arm that holds the state used more thinking tokens a move than the kept chat; in Test 23 it used 112 tokens a step where the kept chat used 3.3. The tokens saved in the prompt can be partly spent in thinking.
* **The explorer's conductor is untested.** It never fired. The object view and the untried list changed the outcome by little.
* **The repair is the job's rule.** The 4,000-step result (all 46 slips completed) needed a repair rule that the job defines and the model was not told. The layer works when the rule is known.
* **The size of the effect depends on the job.** The long session shows large effects because the turns are heavy and the job is long; the ARC games are short and the effect is on cost and unusable moves, not on play.

---

## 7. What is not shown

* **Joules.** The API gives no energy figure. Tokens computed and seconds of calls are the stand-in; they are not a measure of the energy a step takes.
* **Any other model.** All four 2026 runs are one model (Gemini 3.8 Flash); the earlier runs are Qwen2.5 3B, 7B and 14B. Nothing is said about other models, sizes above 14B, or any provider's service.
* **Any other seed or a repeat.** Every run is one sampled run of each arm. The 4,000-step run was not repeated. A repeat would give different numbers; whether the 140-turn stretch of Test 23 would recur is not known.
* **Real company workloads.** Everything was measured on synthetic service graphs, a script-written conversation, and four of the 25 public ARC-AGI-3 games at 60 moves each. The private ARC-AGI-3 games and longer plays are not tried.
* **The job time with the waits.** The seconds are the seconds of the calls; the waits the key's per-minute limit imposed on the notebook are not included.
* **A cache-aware or summary-based baseline.** The kept chat is the plain kept chat; a chat with a rolling summary, or a retrieval step, is not measured.
* **A trained model.** AXI is inference-time. Nothing here shows a model trained with these layers drifts less.
* **GPU-level behaviour.** The pager, the distributed commit and the device masks are reference models and microbenchmarks; the Rust crates are not timed; latency between GPUs is not measured.
* **That the layers in the shared notes are the architect's.** Whether the names in those notes match the architect's own writing is for him to confirm.
* **Requests in ordinary words on larger jobs.** The grounding of words to services is tested on small tables; it is exact matching against a table, not language understanding.
* **An agent that reads its own state back and corrects itself.**

---

## 8. What comes next

* **A stall rule that also fires on a repeated action.** In Test 24 one game saw one action sent 34 times of 60; the conductor of Test 25 fires on "no new screen in 4 moves" or "this screen seen 3 times" and never fired. A rule on the same action repeated is a separate test with its own page, not a change to Test 25.
* **A longer conversation.** The kept chat in Test 23 stopped at 400 turns and about 200,000 tokens; in Test 22 its first slip came near 460,000 tokens. A Test 23 run to about a million tokens would put the ambiguous-text state where Test 22 saw the slips.
* **A repeat of the 4,000-step run and a second seed for Tests 22 to 25,** to see how much of each difference is the run.
* **A baseline with a cache price and a summary.** So that the token comparison can be read in the terms a company is billed in.
* **The mutation pass on tests 135 to 145,** to find how many broken copies of the code the new tests catch.
* **A partner with real tensor workloads and the compute** for the tensor-level part of the roadmap (spinors, hyperbolic geometry, the topology of the history), where our small readings did not beat their baselines and the architect's own definitions are not tested.

---

## 9. Reproduce it

```bash
git clone https://github.com/gaurshalendra2412/axi-runtime
cd axi-runtime
PYTHONPATH=python python3 python/tests/run_all.py          # about three minutes; four tests that need PyTorch or a GPU are skipped
cd python
PYTHONPATH=. python3 tests/test_arc_fuzzy_results.py        # recount Tests 23, 24 and 25 from the saved files
PYTHONPATH=. python3 tests/test_gemini_long_result.py       # recount Test 22
```

The saved files are in `docs/results/`: `axi_gemini_long_session_seed20261111_default.json`, `axi_gemini_fuzzy_seed20261122_default.json`, `axi_gemini_arc_default.json`, `axi_gemini_arc_b_default.json`. Re-running a model experiment needs your own API key and costs real money; the Colab notebooks are in `python/colab/` (the key stays in Colab's secrets, never in the repository, and each notebook prints a pilot with a projected cost and takes a budget before it spends anything).

---

## 10. Credits

The idea, the framework, the architecture and the structural maps of AXI are by **Rajnish Choubey** (the Cinderella Project and the Sovereign Manifesto). They are his to accept or reject. This repository holds the code, the tests and the measurements that implement and check them; they were written by Shalendra Gaur, working with an AI assistant (Claude), which is a tool here and not an author. See `docs/CREDITS.md`.

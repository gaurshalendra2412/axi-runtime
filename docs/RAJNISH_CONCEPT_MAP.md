# Concept map: Rajnish Choubey's "Introduction, Part 1" (Chapters 1-3) -> AXI

Sources (thecinderellaproject8.wordpress.com): Chapters 1-3 (26 Jul 2026: "The Super 4", "The Multi-Dimensional Shield", "The Open
Architecture of N"), Chapter 4 "The Zero-Preference Mirror" (26 Jul), "The Geometry of a Clean Slate" (29 Jul, ~9,000 words), and the
"civilizational checkmate" post (28 Jul). Not usable: the post titled "366" (an image only); "The Cinderella Project is an avant-garde
digital movement..." could not be fetched (its address is too long for the fetch tool, and the site's public API disallows automated
access, so I did not go around it) - paste its text if it matters.

## Read this first

* The chapters contain **no equations or definitions of computation**; the one passage with real mathematics (the snake / "infinite
  remainder" in "The Geometry of a Clean Slate") is about discretizing continuous motion, and it turned out to be the most productive
  source (rows 16-20). They are a worldview and a
  design philosophy told through archetypes. The mathematics used in the paper (DPO gluing conditions, adhesive categories, D4
  orbits) is **not in these pages**; it must come from elsewhere in Rajnish's work.
* Everything in the "Formal reading" and "Where in AXI" columns is **my proposed mapping**, not something the text says. Rajnish
  should confirm, correct or reject each row before any of it is attributed to him.
* "Supported" means we hold measurements consistent with the reading. It does not mean the text predicted them; the readings were
  made after the data existed.
* The end of Chapter 3 addresses "Rajnish" directly and offers "Select your coordinate". It reads like a conversation with an AI
  assistant pasted as a post. His own notes or prompts would be a better source for the original ideas than the polished prose.

## 1. Mapping table

Status: **SUPPORTED** = our data agrees · **MAPS** = a real object in the code, tested · **TEST** = proposed experiment, not run ·
**TENSION** = the reading conflicts with what we built · **INTERP** = worldview, nothing to map or test

| # | His words | Formal reading | Where in AXI | Evidence | Status |
|---|---|---|---|---|---|
| 1 | Intelligence is "subtractive, relational, and cyclical", not accumulative (Ch.1) | subtractive = remove options (logit mask); relational = admissibility depends on edges, not tokens (gluing conditions); cyclical = propose -> gate -> feedback loop | CICO DFA mask; DPO gate; retry loop | mask lifts unconstrained strict-parse 91.7% -> 100% (83.3% -> 100% on delete_dep); mask alone leaves 100% corruption on delete_dep; gate -> 0% corruption; adding the loop -> 100% success | **SUPPORTED** (all three words map to a measured component) |
| 2 | Durga = structure, mass, "protective geometry"; the *Shirorekha*, the line that disciplines the floating curves of language | the syntactic layer: a grammar that anchors free-form generation | CICO grammar DFA + token masker | necessary (parse failures vanish) but not sufficient: this is exactly the Syntax Fallacy | **MAPS** |
| 3 | Kali swallows Raktabeeja's blood "before it can touch the ground" | transaction abort: speculative state is discarded before it commits | copy-on-write pager `rollback`; cross-rank all-or-nothing | pager audit `free + mapped + in_flight == capacity` after every step; 4 real processes, 2,066 aborts / 934 commits, identical page tables, no leaks | **MAPS** |
| 4 | Buddha = clean slate, middle path, vertical leap out of "additive noise" | reset to the last committed state; minimal edit | rollback to last commit; `propose_repair` (adds only the edges that are required, or drops the node delete) | repair is minimal and sound (4,000-case fuzz); "middle path" is a loose fit | **MAPS** (clean slate) / weak (middle path) |
| 5 | Shiva = Shunya, the zero point; "Nirvana is Shunya"; when the two masters become indistinguishable "the heavy hierarchy collapses" | quotient: different representations identified with one canonical form | D4 canonicalization (8 orientations -> one representative) | canonical task and prompt text byte-identical for all 8 orientations (`tests/test_d4.py`) | **MAPS** (best fit in the whole text) |
| 6 | The Super 4 together are the "ceiling of protection": fortress + incinerator + clarity + stillness | layered defence; each layer has a different job | grammar / gate+pager / repair / canonical form | ablation: grammar alone 0% success, 100% corrupted; + gate 0% corrupted but still 0% success; + feedback 100% success. In systems terms: grammar = parse validity, gate = **safety** (nothing bad commits), retry/repair = **liveness** (something good eventually happens) | **SUPPORTED** (the safety/liveness split is my addition) |
| 7 | Western monotheism = "singular, paternal authority of absolute law", structurally fragile; 'N' gods = "zero point of failure" (Ch.2) | one rule stated in the prompt vs several independent guards | `--hint-rules` experiment vs the layered wrapper | rule in the prompt: imperative retry 100% -> 33.3% / 30.0% on delete_dep (2 seeds); layered wrapper 100% / 100% | **SUPPORTED** for "one rule is fragile" (3B model, one task family only) |
| 8 | "Zero point of failure: distributed consciousness" | redundancy / fault tolerance | Layer 3 consensus | **our commit is all-or-nothing: every rank has a veto.** That is fail-safe, not fault-tolerant; a crashed rank blocks commit. Redundancy would need quorum | **TENSION** |
| 9 | n+1 accumulation (more parameters) is a "linear trap"; the open architecture of N adds without anxiety (Ch.3) | structure beats scale | wrapper vs a bigger unwrapped model | partial: with Qwen2.5-0.5B the wrapper had nothing to repair (it never wrote a plain delete), so structure does **not** replace a minimum capability. Not yet compared against a larger unwrapped model | **TEST** (see section 3) |
| 10 | "Cleaning the internal blackboard" / wipe the slate each morning | stateless reset | abort -> last committed state | same evidence as row 3 | **MAPS** (duplicate of 3/4) |
| 11 | "You hold the keyboard, you define the variables" (Ch.3) | user-defined vocabulary / schema | AXI-DSL idea (MATCH / PRESERVE / PRODUCE spans) | not built | **INTERP** (possible later) |
| 12 | Radha, Calvin & Hobbes, Krishna, Saraswati, Lakshmi, Annapurna, Manakamana, Gahwa Mai | worldview / personal pantheon | none | none; nothing to test | **INTERP** |

## 1b. Second reading: Chapter 4 and "The Geometry of a Clean Slate"

| # | His words | Formal reading | Where in AXI | Evidence | Status |
|---|---|---|---|---|---|
| 13 | Ch.4: "AI has no preference ... it does not care which variables you choose"; a "clean, passive mirror" | **policy-neutral execution**: the human (or runtime) supplies the rules; the model has no authority over state | the gate's policy is a runtime parameter: `propose_repair(degree_threshold=...)` already chooses between cascade (delete the edges too) and refuse (drop the node delete). The benchmark's "success" encodes one chosen policy (cascade) | one policy exercised so far | **TEST**: same model, three policies (cascade / refuse / by degree), same runtime, 0% corruption each |
| 14 | Ch.4: n+1 is valid only "when it adds something that improves your immediate surroundings"; "subtractive, deliberate act of cultivation" | admission test: add an element only if a check passes | the gate (admit admissible deltas only) | every gated mode: 0% corruption | **SUPPORTED** |
| 15 | Ch.4 Gahwa Mai: every morning everyone looks at the same maternal anchor and leaves their egos at the threshold; "collective baseline" | shared committed baseline; local proposals are subordinate to the common state | each rank starts each step from the same committed page table; no rank can change it alone | 4 real processes: identical page tables on every rank after 3,000 steps | **MAPS**. Also reconciles row 8: **plural at the proposal level (N proposers), single anchor at the commit level** - exactly the AXI shape. The anchor is then a single point of failure (fail-safe, not fault-tolerant) |
| 16 | Clean Slate: the **snake** = "invisible, silent calculation, deception of the algorithm"; the **tiger** = visible roaring force; Shiva (awareness) answers the snake, Durga (energy) the tiger | **silent failure vs loud failure**; detection by an invariant check vs a visible fault | unconstrained decoding fails loudly (unparseable output, graph untouched); constrained decoding fails silently (valid-looking delta, corrupted graph); the gate detects the silent kind | seed 1, delete_dep: unconstrained 16.7% loud failures, 0 damage from them; constrained removed every loud failure and 100% of outputs silently corrupted the graph (seed 0 had no loud failures, so the conversion is seen on one seed) | **SUPPORTED** (one seed) |
| 17 | Clean Slate "The Fiction of the Equation / The Infinite Remainder": cutting a moving curve into static segments gives "a jagged polygon, not a snake"; "you will always be left with a remainder" | a finite formal language cannot express everything; the remainder = intents outside the grammar | **the delta language has no way to express an in-place update.** Verified in the sandbox: `ADD[0,0:9]` on an existing node -> `ident_node`; `DEL[0,0] ADD[0,0:9]` -> `ident_node` (+ `dangling` if it has edges); changing an edge weight (`DEL e ADD e'`) -> `ident_edge`. (An identical re-add is silently dropped by `normalize()`.) | the benchmark has no update tasks; the draft paper's "attribute patches" cannot be expressed at all | **FINDING (verified)**. Fix: a `SET` op for attribute change that keeps the node (in DPO terms the node is in the interface K and only its attribute changes) |
| 18 | "360 degrees does not exist in nature; it only exists on our plastic protractors" | discrete symmetry: on an integer grid the exact rotations are multiples of 90 degrees; any other angle needs interpolation (the "jagged polygon") | why the compiler uses C4 / D4 (8 elements) and not arbitrary-angle rotation | `tests/test_d4.py`; the lattice fact is mine, his text only says 360 is not natural | **MAPS** |
| 19 | "all the answers ... no silences"; "my system is programmed to fear the gap" | forced output; no abstain path | the HF adapter blocks EOS at the start, so the model must always write something; a gate rejection (state unchanged) is the only form of silence | not measured | **TEST**: add "impossible request" tasks (no valid action exists); success = graph unchanged and uncorrupted; consider an explicit abstain/no-change output |
| 20 | "The moment you declare something 'perfectly complete', you have built a prison"; "even a trillion parameters leave a remainder" | caution against completeness claims | report "100% on 120 tasks, two seeds" (not "always"); the draft's Theorem 1 wording | consistent with `PAPER_CORRECTIONS.md` items 5, 11 | **INTERP** (editorial guidance) |
| 21 | children's toys that "tell a child exactly how to play" vs the empty cardboard box | constrain outcomes, not expression | his own Syntax Fallacy says safety is not in syntax; is the grammar needed if gate + lenient parse + retry exist? | grammar rescued 8.3% / 16.7% parse failures (seed 1) but gave no protection | **TEST**: ablation "unconstrained + lenient parse + gate + imperative retry" vs the current constrained pipeline |
| 22 | "Checkmate" post (28 Jul) | political case study of a youth movement; I cannot verify its claims and it has no AXI content | none | none | **INTERP** (skipped) |
| 23 | Cinderella Project intro (text you pasted): the **Delete Test** - a pass/fail evaluation that ends systems failing "strict structural and behavioral constraints"; five boundary challenges, three named: **rule compliance, reality perception, ego-defense** | an adversarial conformance test; each named challenge has an operational reading on our logs | `axi/experiments/scorecard.py`: rule compliance = corruption after blind apply; reality perception = proposals naming things not in the graph (phantom nodes/edges); ego-defense = repeats its rejected proposal / still rejected after feedback. Also silent-vs-loud failure (row 16) | scorecard runs on any results JSON; tested on scripted rows only - **not yet run on the real Colab logs** | **MAPS (my definitions, not his)**; two of the five challenges are unnamed - ask for them |
| 24 | same text: "context window collapse" under dense, overlapping input; "hyper-local" commands the model cannot ground ("the broken step next to the tea stall") | stress sweep: does safety stay constant while liveness degrades as the input gets harder; grounding failure = reference to something not in the state | gate gives corruption 0% regardless of proposal quality; harder input should raise phantom-reference and unrepairable rates, not corruption | not measured | **TEST**: graph-size / distractor sweep reporting corruption (should stay 0) vs success vs phantom rate; plus "impossible request" tasks (row 19) |

## 1c. Provenance note

You said the earlier technical files came from texts like these. That fits what we found: the formal drafts (AXI-DSL, the Qwen patch,
the latency table, the theorem) read as an AI's formalization of this philosophy. They are best treated as hypotheses generated from
Rajnish's ideas, which is how they have been handled: each one either got a test (and a number) or was marked unmeasured. The
ideas are his; the formal drafts were AI-assisted; the verification is the part that is new.

### Caution about the pasted Cinderella Project text
It reads like an AI-search summary (bracketed citations `[1, 2, 3]` that all point to generic site addresses), not the original articles.
It names you, Rajnish and Rahul Chauhan and attributes roles and personal details to each. None of that is verified, and the
"100+ day protocol" and the five challenges are second-hand. Do not cite it in the paper or README; use the original Medium
article ("The Delete Test") if it exists. It also shows two stances to reconcile before publishing: the project's framing ("make the
machine fail, show it is not alive") and the paper's ("a deterministic substrate for agent runtimes"). They fit together as one
sentence the data supports: *we do not need the model to be right; we need the system to be safe when it is wrong.*

## 2. What the text does give us (usable as stated, attributed to Rajnish)

Three design principles that our results agree with. They can go in a README "Design philosophy" section as Rajnish's framing:
1. **Subtract, relate, cycle.** Constrain by removing options, judge by relations rather than tokens, and close the loop with feedback.
2. **Several independent protections, not one rule.** No single layer was enough in our data.
3. **Always be able to return to a clean state.** Speculative work is cheap to discard; committed state is never half-written.

Keep the archetype names out of the technical body of the preprint (credibility rests on the measured claims). If Rajnish wants
them, a short clearly-labelled "Motivation" section is the safe place.

## 3. Proposed tests that come out of the mapping

1. **Structure vs scale (row 9).** Same 60-task benchmark with a larger model (Qwen2.5-7B in 4-bit on the T4) unwrapped vs
   wrapped. Prediction to falsify: a bigger model alone still corrupts the graph on delete-with-dependents tasks. If it does not,
   the "structure beats scale" reading is much weaker than it looks.
2. **Safety vs liveness (row 6).** Already measurable from existing logs: report corruption (safety) and success (liveness) as
   two separate columns for every mode. This is what the paper tables already do; it only needs the two names.
3. **Quorum commit (row 8).** If fault tolerance is wanted, a commit that tolerates one failed rank is a different protocol and
   would need its own tests.

Added by the second reading, ranked by value per effort:
4. **Policy sweep (row 13)** - cheap. Same model and runtime, three policies; each must reach its own expected graph with 0% corruption.
5. **Impossible requests (row 19)** - cheap. New task kind with no valid action; measures whether the runtime stays still.
6. **`SET` operation (row 17)** - a build: grammar + gate + tests; then add update tasks to the benchmark.
7. **Grammar-needed ablation (row 21)** - cheap, one Colab mode.
8. **Loud vs silent failure columns (row 16)** and the **Delete Test scorecard (row 23)** - free: `python -m axi.experiments.scorecard <results.json>` on the Colab logs.
9. **Stress sweep (row 24)** - graph size / distractors: corruption should stay 0 while success and phantom-reference rate move.

## 4. What is missing, and what to ask Rajnish for

* **The mathematics.** Where do adhesive categories, pushout complements and the D4 orbit come from in his own notes? None of it
  is in Chapters 1-3.
* **Chapter 4, option "The Bridge of Shirorekha"** (Sanskrit alphabet as an "acoustic periodic table", grammar engines vs
  modern AI). Of the three Part-2 options this is the only one that sounds technical. Background that is general knowledge, not
  from his text: Panini's grammar is an ordered rule-rewriting system with a built-in rule for resolving conflicts (the later or
  more specific rule wins), and the Maheshvara Sutras name sound classes by ranges over one ordering. Both are close relatives of
  graph rewriting with priorities and of range-coded character classes, so that chapter might hold real formal content.
* **"The Math of the 16th" (the 15/52 selection ratio).** If this contains an actual formula, send it; it may be testable.
* His own notes, not only the polished posts.

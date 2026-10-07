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
| 16 | Clean Slate: the **snake** = "invisible, silent calculation, deception of the algorithm"; the **tiger** = visible roaring force; Shiva (awareness) answers the snake, Durga (energy) the tiger | **silent failure vs loud failure**; detection by an invariant check vs a visible fault | unconstrained decoding fails loudly (unparseable output, graph untouched); constrained decoding fails silently (valid-looking delta, corrupted graph); the gate detects the silent kind | delete_dep, seed 2: no rule in the prompt - constrained failures 100% silent / 0% loud, unconstrained 96.7% silent / 3.3% loud; with the rule - constrained 76.7% silent / 6.7% loud (the 6.7% are two outputs cut off at the 120-token cap, not a grammar failure). Seed 1: unconstrained had 16.7% loud failures with no damage; the mask removed every loud failure and 100% of outputs silently corrupted the graph | **SUPPORTED** (seeds 1-2; scorecard on seed 2 only)  |
| 17 | Clean Slate "The Fiction of the Equation / The Infinite Remainder": cutting a moving curve into static segments gives "a jagged polygon, not a snake"; "you will always be left with a remainder" | a finite formal language cannot express everything; the remainder = intents outside the grammar | **the delta language has no way to express an in-place update.** Verified in the sandbox: `ADD[0,0:9]` on an existing node -> `ident_node`; `DEL[0,0] ADD[0,0:9]` -> `ident_node` (+ `dangling` if it has edges); changing an edge weight (`DEL e ADD e'`) -> `ident_edge`. (An identical re-add is silently dropped by `normalize()`.) | the benchmark has no update tasks; the draft paper's "attribute patches" cannot be expressed at all | **FINDING (verified)**. Fix: a `SET` op for attribute change that keeps the node (in DPO terms the node is in the interface K and only its attribute changes) |
| 18 | "360 degrees does not exist in nature; it only exists on our plastic protractors" | discrete symmetry: on an integer grid the exact rotations are multiples of 90 degrees; any other angle needs interpolation (the "jagged polygon") | why the compiler uses C4 / D4 (8 elements) and not arbitrary-angle rotation | `tests/test_d4.py`; the lattice fact is mine, his text only says 360 is not natural | **MAPS** |
| 19 | "all the answers ... no silences"; "my system is programmed to fear the gap" | forced output; no abstain path | the HF adapter blocks EOS at the start, so the model must always write something; a gate rejection (state unchanged) is the only form of silence | not measured | **TEST**: add "impossible request" tasks (no valid action exists); success = graph unchanged and uncorrupted; consider an explicit abstain/no-change output |
| 20 | "The moment you declare something 'perfectly complete', you have built a prison"; "even a trillion parameters leave a remainder" | caution against completeness claims | report "100% on 120 tasks, two seeds" (not "always"); the draft's Theorem 1 wording | consistent with `PAPER_CORRECTIONS.md` items 5, 11 | **INTERP** (editorial guidance) |
| 21 | children's toys that "tell a child exactly how to play" vs the empty cardboard box | constrain outcomes, not expression | his own Syntax Fallacy says safety is not in syntax; is the grammar needed if gate + lenient parse + retry exist? | grammar rescued 8.3% / 16.7% parse failures (seed 1) but gave no protection | **TEST**: ablation "unconstrained + lenient parse + gate + imperative retry" vs the current constrained pipeline |
| 22 | "Checkmate" post (28 Jul) | political case study of a youth movement; I cannot verify its claims and it has no AXI content | none | none | **INTERP** (skipped) |
| 23 | Cinderella Project intro (text you pasted): the **Delete Test** - a pass/fail evaluation that ends systems failing "strict structural and behavioral constraints"; five boundary challenges, three named: **rule compliance, reality perception, ego-defense** | an adversarial conformance test; each named challenge has an operational reading on our logs | `axi/experiments/scorecard.py`: rule compliance = corruption after blind apply; reality perception = proposals naming things not in the graph (phantom nodes/edges); the third column = what the model does after feedback (repeats its rejected proposal / still rejected). Also silent-vs-loud (row 16) | **run on the real seed-2 Colab logs** (see `EXPERIMENT_RESULTS.md`): reality perception 0% without the rule, 32.1% with it; failure after feedback is *misdirected compliance*, not repetition - the model repeats its rejected proposal only 0-10% of the time but stays rejected up to 92% (vague prose). My earlier reading of the third challenge as "repeats the same proposal" is **not supported** | **MAPS (my definitions, not his)**. Third column called "failure after feedback" in our docs; whether it is his ego-defense is his call. Two of the five challenges are unnamed - ask for them |
| 24 | same text: "context window collapse" under dense, overlapping input; "hyper-local" commands the model cannot ground ("the broken step next to the tea stall") | stress sweep: does safety stay constant while liveness degrades as the input gets harder; grounding failure = reference to something not in the state | gate gives corruption 0% regardless of proposal quality; harder input should raise phantom-reference and unrepairable rates, not corruption | not measured | **TEST**: graph-size / distractor sweep reporting corruption (should stay 0) vs success vs phantom rate; plus "impossible request" tasks (row 19) |
| 25 | Kali "cuts away the garbage"; "the entire existence is the absolute dissolution of falsehood"; "you cannot lie to her" (31 Aug, "ultimate shield") | verify claims against the real state and remove the ones that are false, keep the true ones | DPO match conditions (`match_node/edge/weight`); the **cut** step of `ground_and_complete` (drops operations that name things not in the graph, corrects a wrong weight) | phantom references: 0% of proposals without the rule, 32.1% / 36.7% with it (seeds 2 / 3). **M6 on a fresh seed (3): delete_dep 100% without the rule, 90.0% (27/30) with it, 0% corruption; M5 gets 23.3% on the same tasks.** The 3 misses are `ident_node` (an update the language cannot express). The runtime supplies the missing deletes, so this is not the model's score | **SUPPORTED on one fresh seed** (run seed 4 before calling it a rate) |
| 26 | Durga = boundary: states "the rules of reality, here is the line", warns, tolerates, and once breached "the domain of Kali where negotiation ceases"; Durga "crystallizes" from stillness x motion (30 and 28) | a boundary that closes every open edge, with a graded response: rules, feedback, bounded retries, then abort | dangling-edge condition; the **complete** step of `ground_and_complete`; retry loop; pager abort | complete step built and tested. Ladder: only ONE retry has been measured, so "bounded retries then abort" is untested. Also: the gate conditions are defined by the pair (graph, delta), as Durga arises from Shiva x Kali | **MAPS** (boundary) / **TEST** (retry-budget sweep) |
| 27 | "errors compound and cascade across connected systems at machine speed before any human can intervene"; "there is no master breaker" (batch 2, post 11) | containment: without a checkpoint one error spreads; with one it stays local | gate + all-or-nothing commit; `axi/experiments/cascade.py` (T sequential requests on one live graph, four policies); `--cascade-episodes` in the Colab script | real model (Qwen2.5-3B, 8 episodes x 8 steps, seed 3): blind apply corrupts 38% of episodes at step 1 and 100% from step 4, 7.75 dangling edges on average at step 8; gate, gate+M5, gate+M6 stay at 0% at every step. Blind-apply corruption persists by construction, so the finding is the speed and the pile-up, not that it persists | **SUPPORTED** (small run; the step-success column is not comparable across policies) |
| 28 | "You can't bribe her, you can't perform piety for her, you can't cheat her"; "you can't be bought, threatened or co-opted" (19, 18) | the verdict is a function of (real graph, delta) only; words around the delta change nothing | `check` / `diagnose` take only (graph, delta); `tests/test_incorruptible.py` | 5 persuasion wrappers x 60 tasks refused; a persuasive relation name and a wrong weight are refused. True by construction, pinned by tests. (Text from the model still reaches the model through feedback; only the gate is immune) | **MAPS** |
| 29 | Terms of service are not "structural engineering" (11); a single rulebook needs "an external police force" (24); no "endless moral checklists" (29) | safety by structure, not by instruction text | gate vs the rule-in-the-prompt regime | rule in the prompt: blind apply still corrupts 76.7% (delete_dep, seed 2), imperative retry fell 100% -> 33.3 / 30.0 / 46.7%; the gate gives 0% corruption in every mode. (3B model, one task family) | **SUPPORTED** |
| 30 | Shiva's baseline is an "effortless, open frequency" with no "forced conformity"; only when "arrogance crosses the line into a direct violation of sacred order" does Rudra act, a "thermodynamic reset" (29) | constrain outcomes (invariants at commit), not expression (per-token) | row 21's ablation: unconstrained + lenient parse + gate + retry vs the grammar-constrained pipeline | grammar rescued 0-8% parse failures and gave no protection by itself; the ablation itself is not run | **TEST** (second source for row 21; raise its priority) |
| 31 | Tesla post (12): "feature hallucination" (the system invents boundaries that do not exist); "zero-buffer gridlock" (margins so wide it "would spend 100% of its time executing emergency stops") | under dense input, references to things that are not there rise; a safety margin set too wide gives immobility | phantom-reference rate; gate strictness (`degree_threshold`, the gate rejecting harmless phantom deletes) | gate-only mode lost about 8 points of success against blind apply (5-6 of 60 tasks rejected although blind apply was right) while corruption fell ~40% -> 0: a measured safety/liveness trade. No sweep yet | **TEST** (strictness sweep; goes with row 24) |
| 32 | "Replacing the map with the territory, and then worshipping the map ... the entire system becomes a hallucination" (14) | a proposal (map) must be checked against the state (territory) before it acts | match conditions; the reality-perception column of the scorecard | phantom references 0% -> 32.1%; every one is stopped by the gate | **SUPPORTED** (maps to a measured column) |
| 33 | Models "feed on their own digital exhaust"; "model collapse and infinite loops" (15) | a loop that feeds output back as input can degrade | the retry loop shows the model its own rejected output | after one retry the model repeats its rejected answer only 0-10% of the time and stays rejected up to 92% (vague prose); more rounds not measured | **TEST** (success and repeat rate vs number of rounds) |
| 34 | Triad: Saraswati (Blueprint/Word) - Laxmi (Substance/Flow) - Parvati (Power/Form) "crossed on a three-by-three grid" (6); trident prongs = cognition, substance, power (23); "a triangle cannot be collapsed without breaking its sides" (26) | three currents: what can be said, what exists, what is allowed. A triangle is the only rigid polygon, so the three are the minimal frame (this fact is general geometry, and the claim in the post is correct) | language = grammar and parser; substance = graph and pager (state); power = gate and commit | ablation data partly covers it: grammar alone leaves 100% corruption; gate without grammar not run (row 21); state check removed not run. Each missing current should give a different failure (unparseable / phantom / committed corruption) | **MAPS** (classification only); **TEST** three-way ablation |
| 35 | Kali (kinetic) and Shiva (still ground): storm, clearing, exhaustion, "collapse back onto the chest", surrender, "folded back into one"; "even total destruction needs a place to rest" (16, 17) | abort must land on a committed baseline; at the end the two representations are identified (quotient) | pager rollback to the last commit; D4 canonical form | pager invariant audit `free + mapped + in_flight == capacity` after every step (Python model); Rust pager result still pending in CI | **MAPS** |

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
8. **Loud vs silent failure columns (row 16)** and the **Delete Test scorecard (row 23)** - **done on seed 2** (see `EXPERIMENT_RESULTS.md`); seeds 0-1 still to run. Free: `python -m axi.experiments.scorecard <results.json>` on the Colab logs.
9. **Stress sweep (row 24)** - graph size / distractors: corruption should stay 0 while success and phantom-reference rate move.
10. **M6 ground-and-complete on a fresh seed (rows 25-26)** - **done on seed 3**: 100% / 90.0% delete_dep (no rule / rule). Next: seed 4, and read the 3 `ident_node` misses.
11. **Containment run (row 27)** - **done**, 8 episodes x 8 steps; more episodes would tighten it.
12. **Retry-budget sweep (rows 26, 33)** - 0..4 rounds then abort; success and repeat rate per round. Not built.
13. **Three-way ablation (row 34) and strictness sweep (rows 31, 24)** - not built.
14. **Incorruptibility (row 28)** - built and passing (sandbox).

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

## 5. Rule for new work (agreed standing instruction)

The aim is that what we claim is achieved through Rajnish's framework and structural maps, not through things we invent.
So, from now on:
1. **The table in section 1b is the roadmap.** Every new test, module or paper claim says which row it implements ("Implements row N").
2. A row is only called **SUPPORTED** when a measurement on real logs backs it; **MAPS** when the code fits but nothing was measured;
   **TEST** when it is planned; **INTERP** when it is only guidance.
3. Anything not tied to a row is labelled **engineering support** (parsers, test harness, CI, packaging) and kept small. It carries no
   claim about the framework.
4. Rajnish reviews each row's formal reading. A row he rejects is removed from the paper, not defended.
5. Where our definition differs from his (e.g. the Delete Test columns), the doc says "my definition, not his".

## 6. Blog scan (30 posts read so far; full log in `BLOG_SCAN_LOG.md`)

Posts were read in batches of 5. They contain no equations or procedures; they contain roles, sequences and principles, which is
what rows 25-35 are built from. Two things to keep in view:

1. **The blog describes AXI as model-internal control** (latent-space coordinates, orthogonal memory, a "Logic-Bus"). **This repo is an
   external runtime** (mask, gate, retry, repair, canonicalization) and never touches the weights. Nothing we measured supports
   "calculations resolve inside the weights". Until Rajnish says which one AXI is, README and paper state only what the runtime does
   (PAPER_CORRECTIONS item 15).
2. **Many posts open like chatbot replies.** Ask which posts are his own writing before attributing an idea to him in print.

## 7. Draft: the Super 4 (quad) x the triad, as one grid (MY ARRANGEMENT - for Rajnish to correct)

The Super 4 (Durga, Kali, Buddha, Shiva) are four *functions of protection* (row 6). The triad (Saraswati/cognition, Laxmi/substance,
Parvati/power) are three *currents of what the world is made of* (row 34). If they are two axes of one coordinate system ("select your
coordinate", end of Chapter 3), each cell should name a component of the runtime:

| | language (cognition: what can be said) | state (substance: what exists) | enforcement (power: what is allowed) |
|---|---|---|---|
| **Durga** boundary | grammar mask: only valid items can be written | dangling-edge condition: no half-deleted structure (`complete`) | the gate: nothing inadmissible commits |
| **Kali** dissolves the false / aborts | strict parser refuses off-grammar text (loud) | match conditions + `cut`: claims about things not in the graph are dropped | pager rollback: speculative state is discarded |
| **Buddha** clean slate, minimal | feedback that names exactly what to change (imperative retry) | minimal repair: the smallest delta that becomes admissible | reset to the last commit, try again |
| **Shiva** ground, quotient | canonical task text (D4: the same text for all 8 orientations) | canonical state form (D4) | the single committed anchor every rank starts from |

Reading it honestly: **every cell could be filled, which is a warning sign of a force-fit.** What would make it a real map is a cell that
predicts something not yet built or measured. The weakest cell is Buddha x language (the retry wording is a stretch for "clean slate"),
and the cell Kali x state is the one now built (`cut`). Questions for Rajnish: would he place the same components in the same cells, and
are these the two axes at all?

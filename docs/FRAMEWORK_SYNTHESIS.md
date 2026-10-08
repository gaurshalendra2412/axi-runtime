# What the framework can be: a synthesis after thirteen scans (8 Oct 2026)

You asked me not to go by the words, but to say what Rajnish Choubey's code and structural framework can be underneath: geometric, mathematical, "quad-core" type structure for mapping and architecture. This file is that answer. It is **my reading**, built from 406 distinct posts, 29 Medium pieces and everything we tested and measured. Rajnish should accept, change or reject each part. The page-by-page evidence is in `STRUCTURE_EXTRACTION.md`, `STRUCTURE_EXTRACTION_2.md`, `STRUCTURE_EXTRACTION_3.md`, `STRUCTURE_EXTRACTION_4.md` and the older extraction files; section 9 is what the tenth scan added, section 10 the eleventh and section 11 the twelfth; the row numbers refer to `RAJNISH_CONCEPT_MAP.md`.

## 0. The short answer
Underneath the mythology and the manifestos, the structure that keeps coming back is this:

> **A small symmetric board holds a labelled graph. Changes to it are written as signed edits (add, delete, leave alone). A local check, which has no memory and does not care about tone, decides whether each edit may be applied. A refusal comes with a reason and the loop tries again. The same state in any of 8 orientations is one state (for relations without a direction; for a directed relation only the 4 rotations, section 9).**

That is a **relational language over named slots with a symmetry group around it and a checked loop**. It is mathematical (a group acting on a grid, graph rewriting, orbit counting), it is partly geometric (the board and D4), and almost everything in it is now tested. What it is **not yet** is a *geometric* language in Rajnish's strong sense: the board, its symmetries and a correct-mapping check are outside the language, not inside it (section 5). Whether "geometric structural language is the deepest point of AI" is true is a hypothesis; nothing here proves or disproves it.

## 1. The picture (our arrangement of rows 6, 30, 38, 48, 58 and 76)

```
 a model proposes text
      |   grammar mask: only well-formed items can be written          Durga: the boundary
      v
 a delta:   ADD[r,c:v]   DEL[r,c]   ADD/DEL[(r,c)->(r,c):rel#w]
      |   the gate: match | identification | dangling edge | well-formed Kali: dissolves the false
      |      admit  -> apply (delete edges, delete nodes, add nodes, add edges)
      |      refuse -> a named reason (one of four) -> repair and retry  Buddha: the clean slate, minimal
      v
 the state: a labelled graph on 9 slots (3x3); meaning lives on edges
      |   canonical form: the 8 views of the board are one state        Shiva: the ground, the quotient
      v
 the next prompt: the canonical state, plus the reason for any refusal
```
The four-stage names are Rajnish's Super 4 (row 6); which stage is which is my arrangement (section 7 of the concept map warns that every cell of that grid can be filled, which is a force-fit warning).

## 2. The mathematical objects, and whether each is tested

| Object | Mathematical name | Where in the code | Status |
|---|---|---|---|
| The board | the 3x3 grid with the dihedral group D4 (8 elements) acting | `engine/d4.py` | tested: cell orbits 1 + 4 + 4; Burnside 24 / 8 = 3 |
| The state | a labelled graph over slots, values on nodes, typed weighted edges | `engine/gate.py` `Graph` | tested |
| The edit | a signed vector over slots (+1 add, -1 delete, 0 untouched), plus edges | `cico_parser.py`, `engine/inverse.py` | tested: invertible, balanced (rows 58, 59) |
| The admissibility check | double-pushout style conditions: match, identification, dangling edge, well-formedness | `engine/gate.py` | tested: frame independent, item-order independent, total, stateless, no lockout (rows 79, 80, 91) |
| The loop | propose, check, repair, retry | `python/axi/experiments/agent_loop.py`, `python/colab` | **measured** (below) |
| The quotient | the orbit space of D4 | `engine/d4.py` `canonical` | tested: 362,880 labelings become 45,360 classes (row 64) |
| Which symmetries count as "the same" | D4 (8) for undirected relations, only the 4 rotations when a relation has a direction | `engine/d4.py` uses all 8 | tested as counts (row 99); **a design choice, not yet made** (section 9) |
| A correct mapping | a map that keeps the relations (a homomorphism) | **not built** | calendar test is the worked example (row 90) |
| The verdict | admit or refuse; **a third value, unknown, is proposed** | not built | row 98 |
| A ring, a circle, a cuboid | a cycle graph (1 loop), its contraction to one place with a self-line, and the cube graph (5 loops) | composite deltas through `engine/gate.py` and `engine/inverse.py` | tested, with inverses (row 133, thirteenth scan) |
| A dependency chain with a ground, closed into a cycle | a path whose last place is identified with an earlier one (a quotient); loops by cyclomatic number | merge as one composite delta | tested (row 134) |
| Independent two-valued axes | the hypercube Q4 (16 states, 32 lines, 384 symmetries, even closed walks) | the 2 x 2 board with the gate's single-cell deltas | tested (row 135) |
| Headroom for a safe write | copy-on-write: rewrite everything iff at most half full | `engine/pager.py` | tested by brute force at capacity 64 (row 136) |
| A grammar's language against N^L | the transfer matrix of a DFA | `engine/cico_decoder.py` | tested: 2, 2, 2, 2, 22, 240 valid prefixes (row 138) |

**Measured** (Colab T4, Qwen2.5-3B, seed 3, 60 tasks, 30 deletion-with-dependency tasks; `EXPERIMENT_RESULTS.md`). No gate corrupts 100 percent (no rule given) and 76.7 percent (rule given). The gate alone corrupts 0 and completes 0.0 and 3.3 percent. The gate with feedback and repair (M6) corrupts 0 and completes 100 and 90.0 percent. Letting a model's edits through blindly corrupts 38, 75, 88 and 100 percent of 8 episodes after steps 1 to 4. Four processes agree on the commit: 934 commits and 2,066 aborts in 3,000 steps with identical page tables. These are one model and one seed; they say the mechanism works, not how well it generalizes.

## 3. The numbers the board forces (all brute-force checked)

| Fact | Value | Test |
|---|---|---|
| Symmetries of the board | 8 = 4 rotations + 4 reflections | `d4.py`; `test_d4_orbits.py` |
| Orbits of cells | 3: the centre (1), four edge cells (4), four corners (4) | `test_d4_orbits.py` |
| Pairs of cells ("a line between two points") | 36 = 5 distance kinds = **8 D4 classes** (8, 8, 4, 4, 4, 4, 2, 2) | `test_board_vocabulary.py` |
| Pairs on a straight line of the board | 28 of 36; the 8 knight-move pairs are not | same |
| Straight lines of three | 8; each cell lies on 4 (centre), 3 (corner) or 2 (edge) | same |
| Four lines (2 vertical, 2 horizontal) | 4 crossings, 9 fields, bounded by 4 / 3 / 2 lines | same |
| Arrangements of nine distinct labels | 362,880, in **45,360** classes, 15.47 bits | `test_nine_labelings.py`, `test_shape_capacity.py` |
| Four labels on a square | 24 arrangements, **3 classes**, 1.585 bits (which two are opposite) | `test_shape_capacity.py` |
| Three labels on a triangle | 1 class, 0 bits | same |
| A mirror-symmetric board | k^6 of the k^9 states | `test_mirror_axis.py` |
| "Figures are rows" in a 3x3 table | survives 4 of the 8 transforms (a Klein four-group); the relations survive all 8 | `test_table_axes.py` |
| Placements of the nine entries as rows or columns | 2,592 of 362,880; 324 classes | same |
| Four labels on a square, by how much is named | 3 classes (nothing named, 1.585 bits); 6 (axes named: Klein four-group, 2.585 bits); 24 (poles named too, 4.585 bits) | `test_shape_capacity.py` |
| Rotations only versus all eight symmetries | triangle 2 vs 1 classes; square 6 vs 3; 3x3 90,720 vs 45,360 (one bit: handedness) | `test_symmetry_direction.py` |
| Forgetting the frame | canonical form keeps 45,360 of 362,880; averaging over the 8 symmetries keeps **167**; the global mean keeps 1 | `test_average_vs_canonical.py` |
| Cell orbits of an n x n board | triangular numbers 1, 3, 3, 6, 6, 10, 10 for n = 2 to 8; a centre exists iff n is odd; lines of n cells 2n + 2 | `test_board_scaling.py` |
| Smallest rigid body | 3 points in the plane (2n - 3 bars), 4 points in space (3n - 6); volume needs 4 points | `test_rigidity_counts.py` |
| A wall of 100 fragments | 4,950 possible strings; 2^4,950 walls; 100^98 spanning trees | `test_wall_counts.py` |
| Four different names, by the shape they sit in | undirected ring 3 classes; directed ring 6; undirected path 12; directed path 24; a fork 12 until its branches have different relation names or weights, then 24 (n! over the symmetries) | `test_shape_of_four.py` |
| Four arms into one node | 24 symmetries unnamed, 1 named; on the board's plus shape the 24 namings fold to 3 classes under the 8 symmetries (6 under rotations) | `test_convergence_star.py` |
| Distance on the 3x3 board, by neighbourhood | 12 / 16 / 20 / 36 pairs give farthest 4 / 4 / 2 / 1 and sum of distances 72 / 62 / 52 / 36; one diagonal per square keeps 4 of 8 symmetries, both keep 8 | `test_distance_collapse.py` |
| Order of a log of inverses | replayed newest first it restores in every episode; any other order is refused by the gate most of the time and rarely lands in a wrong state | `test_summary_order.py` |

## 4. Why "four" keeps appearing, and what that does and does not mean
The corpus uses the word four for: the Super 4 wrapper layers; the four Devis; four intersecting lines that carve nine fields; the 1 Sep "full quadrant" of four ways to build AI; and the four yugas. Our four-process commit is a sixth use, but it is **ours**, not his. No page states a rule that joins them.

What the mathematics offers, as a reading: **four is the natural unit of this board.** The symmetry group is 4 rotations plus 4 reflections; the cells fall into one centre plus 4 edge cells plus 4 corners; of the 11 orbits of cells and pairs, 6 have size 4; the four lines make 4 crossings; and the subgroup that preserves "figures are rows" is the Klein four-group. If "quad-core" means four units arranged around a core, the 3x3 board is exactly that twice over (four gates and four corners around one centre), and the post of 13 Sep says "the center holds the core and the eight surrounding gates face the quarters" (row 48; under D4 those eight gates are two classes of four, edges and corners, which is a point for Rajnish to confirm). So the board's own symmetry may be the reason four recurs, rather than any single doctrine.

Two cautions. (1) **This is a numerology risk** (concept map section 6, item 6): small numbers match each other all the time, so the reading counts only if it predicts something. The prediction on offer is that once Rajnish names the axes of the 1 Sep quadrant and the rows of the 13 Sep table, the cells of the 3x3 can be filled without leftovers. If they cannot, the reading is wrong. (2) A square layout of four named things carries only log2(3) = 1.585 bits (which two are opposite), and none unless the axes are named. A quad is a *structure* only when the axes are given.

The tenth scan adds three further places where "four" is mathematics rather than doctrine, and one where it is not. (a) **Four is two independent yes/no switches** (the Klein four-group, Z2 x Z2): a quadrant with its two axes named has exactly that symmetry group and carries 2.585 bits (6 classes), against 1.585 with nothing named and 4.585 with the poles named too; the strand algebra (`reverse`, `complement`, their product) and the "figures are rows" subgroup are the same group (rows 87, 88, 108). (b) **The smallest rigid body in space has four points** (a tetrahedron: 6 bars for 4 points); in the plane it has three (row 100). (c) A centre exists exactly when the board's side is odd, so the "four around one core" picture is a feature of the 3x3 and gone at 4x4 (row 101). The one that is **not** mathematics: the four ages 4:3:2:1 have cumulative shares 40, 70, 90, 100 percent, within 5 points of our measured containment run (37.5, 75, 87.5, 100), which is a coincidence at this resolution and must not be written down as a finding (`test_vedic_arithmetic.py`, PAPER_CORRECTIONS 40).

## 5. What it is not yet (the honest gaps)
1. **The gate has no board.** `ADD[99,99:7]` is admitted on a 3x3 graph. The geometry lives in the canonicalizer and the task, not in the language (`GEOMETRY_EXTRACTION.md` section 4; PAPER_CORRECTIONS 25).
2. **Transformations are not operators.** D4 acts around the model (to canonicalize), never inside a delta. There is no `ROT` or `FLIP`.
3. **There is no mapping checker.** "Correct mapping" has an exact meaning (a homomorphism that keeps the relations), and it needs to be told which relation to keep: the date-to-tithi map keeps order and not successor (row 90).
4. **Two verdicts only.** There is no "unknown" (rows 86, 98).
5. **No derivations anywhere on the pages.** Every structural statement is asserted. The formula promised on one page is not shown (question 71). The mathematics we use (graph-rewriting conditions, orbit counting) is not derived on any page I read; its source is the RFC-001 text and our own work, and the concept map (section 4) asks Rajnish where it comes from.
6. **The "years" cannot be seen on the blog.** The earliest dated text is 21 Jan 2026; the manuscript is said to run from 2019. Dated notes would settle priority (questions 65, 70).
7. **One model, one seed.** The measured results above are one 3B model.
8. **The frame group is undecided for directed relations.** The gate is covariant under all 8 symmetries, but a meaning that refers to clockwise versus anticlockwise does not survive a mirror; whether the frame is D4 or the four rotations changes the board's capacity from 45,360 to 90,720 classes (row 99).
9. **The scorer exists, the experiments do not.** The 13-hour-clock scorer is tested; no model has been run on it, and the Period Test has no harness (row 107).
10. **The gate does not check task rules.** A relation with one value, a value a page forbids, or a column that is a function of two others (rows 124, 125, 130) are each about ten lines in a test and nowhere in the engine; whether they belong in the gate or in a task-level checker is question 94.

## 6. A dictionary: Rajnish's words, the formal object, the status

| His words | Formal object | Row | Status |
|---|---|---|---|
| "subtractive, relational, cyclical" | logit mask, gate on edges, propose-check-retry loop | 1 | SUPPORTED |
| Durga / Kali / Shiva = +1 / -1 / 0 | signed slots, inverse, balance | 58 | TEST |
| gods are state, goddesses are operators | slots versus deltas | 72 | TEST + MAPS |
| the loom: text becomes a pattern on a lattice | a one-dimensional stream edits a two-dimensional state | 76 | INTERP |
| waste isolation, Line 10 / Line 20 | precondition, effect, cleanup in one act | 77 | MAPS |
| "count without losing" | parser returns every item; ledger never drifts | 78 | TEST |
| two frames, Rahi's Law | a verdict independent of the observer's labels | 79 | TEST |
| rules joined, not ranked | conditions joined by AND; always an answer | 80 | TEST (by construction) |
| the mirror and its axis | fixed cells; symmetric states | 81 | TEST |
| "if one goes to zero, or one overpowers the rest" | no gate, gate alone, gate plus repair (M1, M2, M6) | 83 | SUPPORTED (narrowly) |
| "A bare number is a linguistic ghost" | every value sits in a slot | 86 | MAPS |
| the nine entries of the 3x3 | nine labelled cells; meaning on edges | 88 | TEST |
| the full quadrant | four labels on a square: 3 classes, 1.585 bits | 87 | TEST |
| point, line, crossing | cell, pair of cells, crossing of lines; 8 pair classes | 89 | TEST |
| a mapping from one calendar to another | a map that keeps order, not successor | 90 | TEST |
| "a true tool evaluates inputs on logic, not emotional hygiene" | a stateless, tone-blind check with no lockout | 91 | TEST |
| help versus appearing to help | an approval proxy; a check outside the model | 92 | SUPPORTED (narrowly), diagnosis is known literature |
| the Known (1) and the Unknown (0) | a third verdict, unknown | 98 | INTERP |
| "the universal threefold symmetry of the recycling and radiation symbols" | three rotations if the figure is directed, six symmetries if not | 99 | TEST |
| "three points ... make a structure truly unyielding" | rigidity: 2n - 3 bars in the plane, 3n - 6 in space; volume needs four points | 100 | TEST (corrects the page) |
| "abandon human scale" (3x3 to 4x4) | no cliff; the centre disappears (parity) | 101 | TEST |
| "averaging ten thousand quotes collapses every contradiction" | averaging forgets the arrangement (167 results); canonical form forgets only orientation (45,360) | 102 | TEST |
| "a hundred fragments of evidence", "a few intentional strings" | a sparse graph chosen from 4,950 possible edges | 103 | TEST (counts) |
| "the blackboard that never has to be erased" | an append-only log of inverses | 104 | TEST |
| "DNA strands mirroring each other" | a half-turn (from memory); the strand algebra is a Klein four-group | 108 | TEST (algebra only) |
| "the third observer" | the check outside the loop of model and state | 109 | MAPS / INTERP |
| "the N+1 container of the cosmos" | the board as an object (N cells and the board) | 110 | INTERP |
| "a stark question mark on your wall" | the third verdict, unknown | 111 | INTERP |
| "if Brahma / Vishnu / Shiva had been at the chisel" | blind apply, repair, gate alone | 113 | INTERP (force-fit warning) |
| "The West broke itself on the line. The East holds itself in the cycle." | a path closed by one edge is a cycle (n! orderings become (n-1)!) | 123 | TEST |
| "The 8th engine ... would simply be Do again" | the eighth place is the first: one delta, delete edge, delete node, add edge | 123 | TEST |
| a page that says two things about one logo and breaks its own colour rule | a functional relation and a forbidden value, as guards outside the gate | 124, 130 | TEST |
| a table with empty cells and its numbers one column left | a derived column: one of 36 pairings is consistent | 125 | TEST |
| "inspectable in their limits, stable in their character, accountable in their failures" | named checks and reasons (two of three); drift (the third) unmeasured | 127 | MAPS + hypothesis |

## 7. How to use this honestly in the article and the preprint
The spine that every part of this supports:
1. **His problem statement.** Its earliest wording in the corpus is January 2026 ("AI Can't Be Corrected. It Can Be Managed.", 26 Jan), before any mechanism; the 11 Sep pages give the approval form of it: a system optimized on approval cannot tell helping from appearing to help.
2. **The known literature** it restates: Goodhart's law, reward over-optimization, sycophancy (to be cited from primary sources, checked before print).
3. **The formal object**: a graph on a small symmetric board, signed edits, a local check (sections 1 and 2).
4. **Tested properties**: stateless, tone-blind, frame independent, always answers, no lockout, invertible (section 2 and 3).
5. **Measured remedy** with its limits: M1, M2, M6; one model, one seed (section 2).
6. **Limits stated first**: no board in the gate, no derivations, the name and the collision with an unrelated project of the same acronym, authorship (the idea and architecture are Rajnish Choubey's; the code and tests are ours).

Two housekeeping facts the preprint needs: "AXI" is a placeholder and a project called axi (kunchenguid/axi, first commit 2026-03-18) already exists, so Rajnish should choose a coined name; and pages written in an AI voice (02 and 24 of the ninth scan; 13 and 23 of the tenth, which also have no byline; the undated Vedic-cosmos page has no byline either; and others in `PAPER_CORRECTIONS.md` item 27) must not be quoted as his.

## 8. What to build next (your choice, none started)
| Option | What it does | Size | Why it matters |
|---|---|---|---|
| A. Board and bounds in the language | a declared shape; `ADD[99,99:7]` refused on 3x3 | small, testable | makes "geometric" literally true of the gate; the board as an object is the best reading of "N+1" (row 110); it must also declare the neighbourhood (4, 8 or both), which decides distances and symmetries (row 117) |
| B. `ROT` and `FLIP` operators | D4 inside a delta, guarded by the equivariance test | medium | moves D4 from around the model to inside the language; directed relations give it a measured reason (row 99) |
| C. A mapping checker | takes two graphs and a proposed map, lists the relations that break | medium | lets every blog-to-code row be tested instead of argued; the calendar test is its first case; order is one more relation it must be told to keep (row 115) |
| D. A third verdict, unknown | admit, refuse, or unknown | small | ties to the needle test, "Not withheld. Not known." and the question mark on the wall (row 111) |
| E. The A/B harness | plain versus structured prompt; picture versus list; the Period Test; the 13-hour clock (its scorer exists) | one Colab script | the only way to measure a claim about models |
| F. A birth rule (an idea only) | every added node must come with an edge from a node that already exists; isolated ADDs are refused | small, testable | "life only comes from life" (rows 121, 114); the gate today admits an isolated node; needs Rajnish's answer first (question 91) |
| G. A scale sweep (an idea only) | the test of `RAJNISH_CONCEPT_MAP.md` section 3 item 1 (row 9: a larger model, with and without the wrapper), repeated at several model sizes on the same seeded tasks; records corruption without the gate, refusals, completion with the gate and feedback (M6), model calls per completed task and wall-clock per task | one Colab script on the existing experiment code | the only direct test of "more compute will not fix it" for one task family; says nothing about long-conversation drift or production latency; the A/B harness (E) is what answers \"better than what\" for the rest |

My suggestion would be A then C, because A makes the language geometric and C makes the mapping claims testable, and both are small. If the next step should be a measurement rather than a build, it is G (a run on a Colab T4, the first time a larger model meets the wrapper) before E. It is your decision and Rajnish's.

## 9. What the tenth scan added (25 pages, 9 of them re-reads, so 16 new posts; `STRUCTURE_EXTRACTION_2.md`)
Nothing on the pages is derived; what the scan gave is nine more things that can be counted.
1. **Which symmetries count is a choice.** A figure with a direction (a fan, a ring of arrows, an edge of our language) keeps only the rotations, half of the group; a board's capacity doubles from 45,360 to 90,720 classes if the frame is the four rotations instead of D4 (row 99). This is the measured reason for option B.
2. **Two ways to forget the frame.** Canonical form forgets only the orientation (45,360 of 362,880 arrangements survive); averaging over the symmetries forgets the arrangement (167 survive); the global mean forgets everything (1). Shiva is the canonical form, not the mean (row 102).
3. **A quad is two binary axes.** Naming them takes a square's capacity from 3 classes to 6 to 24 (`STRUCTURE_EXTRACTION_2.md` section 2.1b). "Four" in the corpus is easiest to read as Z2 x Z2 plus a name for each axis.
4. **Parity, not a threshold.** Between 3x3 and 4x4 nothing jumps; the centre disappears (row 101).
5. **Determined means constrained enough.** Rigidity counts (2n - 3, 3n - 6) are the bookkeeping of "unyielding"; the page's "three points give volume" is off by one (row 100). Our gate has no bars; this is an analogy for counting freedoms and constraints.
6. **The observer outside the loop** appears on five pages (the third observer, the Common Man in the corner, the perimeter and the man at the centre, the second-hand map, the hundred-fragment wall). In our system it is the gate: stateless, tone-blind, silent but for a reason (row 109, MAPS).
7. **A ladder of undo is a log.** Paper keeps crossed-out words; a blackboard erases and the teacher remembers; in our language `invert` computed before each step is the log, and it was tested end to end (row 104).
8. **A wall is a sparse graph** chosen from 4,950 possible edges (row 103).
9. **The page with the most formulas in this scan** has exact year-count arithmetic and a speed-of-light line that misses by 6.4 percent; a tempting match with our own data is recorded as a coincidence (row 105, PAPER_CORRECTIONS 36, 40).
And one correction to the bookkeeping: nine of the 25 pages were re-reads, so the corpus read is 245 distinct posts, not 253.

## 10. What the eleventh scan added (25 pages, all new, posts 253 to 277; `STRUCTURE_EXTRACTION_3.md`)
A thinner round, and it should be read that way: most pages are one paragraph of rhetoric and five are almost empty. Nothing on the pages is derived; five things could be counted.
1. **What names an arm is an edge.** Four arms into one node have 24 symmetries until they are named and then 1; a fork has a swap symmetry until its branches carry different relation names or weights; four different names on a ring, a path or a fork carry 3, 6, 12 or 24 arrangements. In our language the information that breaks a symmetry is the `rel` and the `#w` of an edge (rows 114, 116).
2. **Two triads lie on one trident.** The page about the primordial jungle assigns past, present and horizon to the three prongs through three kin figures, with the fourth (the brother) as the owner. Post 23 gave the same prongs cognition, substance and power. No page says which prong carries what: a 3x3 table of triads by prongs with its alignments missing (row 114, question 88).
3. **"Spectrum" is a fork, not a line**, if Post 48 is what the author means (row 116, question 89).
4. **Order is a relation a mapping must be told to keep**, and the gate is a good net for a wrong order (refused about 87 percent of the time in the logged distribution) but not a perfect one (row 115).
5. **Distance is edges, and rigid and symmetric pull apart.** One diagonal per small square makes the board rigid with 16 bars but keeps 4 of 8 symmetries; both diagonals keep all 8 with 20 bars (row 117, question 92).
Recorded without use: a binary-string title with isolated ones (row 118), a second ladder of writing surfaces (row 119), addition against subtraction (row 120), "life only comes from life" as a possible birth rule (row 121, option F), and two dogs with no shared word (row 122). Count: 270 distinct posts.

## 11. What the twelfth scan added (30 pages, 20 of them drawn at random, 28 new posts, numbers 278 to 305; `STRUCTURE_EXTRACTION_4.md`)
Again nothing on the pages is derived. Three things could be counted, and one thing about the blog itself could be measured.
1. **A line closed by one edge is a cycle, and "the eighth is Do again" is an identification.** A directed path of n names carries n! orderings and a directed cycle (n-1)!; the loss is log2 n bits and the start and the end disappear. In our language closing is one `ADD`; the octave is `DEL` an edge, `DEL` a node, `ADD` an edge in one delta that the gate admits only whole. Independent loops: 0 for a line, 1 for a ring, 2 for a ring with a chord, 4 for the plain board (row 123, question 93).
2. **A page can contradict itself and the gate as built will not notice.** One page of about 1,950 words says two things about Google's logo, two about Claude's, and breaks its own "no red, no yellow" rule. A functional relation ("one logo, one shape") and a forbidden value ("no red") are two different guards; each is about ten lines in a test; neither is in the engine, which checks that a delta applies, not that the task's rules hold (rows 124, 130, question 94). One page is an example, not a rate.
3. **A derived column is a checkable relation.** A table with its cells one column off has a third quantity that is a ratio of the other two; of 36 pairings exactly one is consistent, so the arithmetic alone recovers the layout (row 125, question 95). A first case for option C.
4. **What the blog looks like when pages are drawn at random.** The archive has about 4,585 slots (month pages of 7: August 371 pages, September 284). Of 20 random pages 14 are one paragraph with nothing countable, 4 have explicit structure (about 8 to 42 percent as a 95 percent interval), and 2 produced a test; the other three tests came from long pages I chose. So the archive probably holds more than the 298 posts read, in a minority of pages, and long, structured pages are the better target than a plain draw. This is a sample of 20.
**On drift, latency and cost.** The pages that use those words (rows 126, 127, 129) assert and measure nothing. What this repository has measured is one 3B model on one family of graph edits (corruption 100 and 76.7 percent without the gate, 0 percent with it; M6 completes 100 and 90 percent of the dependent deletes) and the gate's own check at a few microseconds per tiny delta in the sandbox. That structure also lowers latency and cost, and that a larger model alone would not fix the corruption, are **hypotheses with a test each**: option G for scale, the A/B harness for the rest. Count: 298 distinct posts.

## 12. What the thirteenth scan added (100 pages in 20 batches of 5, plus 8 neighbour pages, all new, numbers 306 to 413; `STRUCTURE_EXTRACTION_5.md`)
The instruction was to look under the words, in patterns and across pages, and to look again at the earlier material. Again nothing on the pages is derived. What could be counted, and what a passing test means: **my count of a structure is right; no claim about AI is thereby true.**
1. **Rectangle, circle, cuboid are a ring and what you do to it.** A ring of n places has one loop at every n; contracting it to one place keeps the loop count and changes only the symmetry count; "4 = 1" survives that and "1 = 0" does not. The cuboid frame has 5 loops, by four independent routes, and contracts to one place with five self-lines (row 133).
2. **"X is borrowed from Y" is a chain with a ground; "the four is the circle" closes exactly one loop and removes the ground and the order.** "=" as identity merges places; "=" as dependence keeps them. The page uses both, so which one the equation means is open (rows 134, 139; questions 99 to 101).
3. **Four pairs are the 4-cube** (16 states, 32 lines, 384 symmetries); the repository's 2 x 2 board is one. Closed walks on it have even length, so a five-step loop must include a no-op (row 135).
4. **A container with most of its space empty is the shape of the copy-on-write pager.** A store can rewrite everything in one transaction only while it is at most half full (brute-forced). The pages' 1,024 / 240 / 784 are the blog host's storage dashboard, so they are a quota and not a design number (row 136; PAPER_CORRECTIONS 61).
5. **The grammar, not the alphabet, sets how many strings are valid.** The delta language admits 2, 2, 2, 2, 22, 240 valid prefixes at lengths 3 to 8, not 26^L or 49^L (row 138).
6. **Structure that no single page contains.** An equation re-posted four times in two hours, each keeping every term; a 4 x 2 table of traditions in three posts published within six minutes; a shared order of four words on two pages. The author keeps fixed-order lists and reuses them, so tables appear across posts (rows 139, 140).
7. **A declared count can be checked, and the gate does not check it.** Two log pages declare five fields and list five; a title says "three-step" and lists four (row 141).
8. **A refusal can be a code, a footprint can be a number, a village can be a star** (rows 144 to 147): every gate refusal is one of four codes and carries no voice; the admission core is 363 lines; a star of a home and four orientations plus four "friendship" lines is a wheel with the grid's four loops.
**What did not survive:** a pattern I saw in three pages of the equation (only one-name rungs equated) is contradicted by the fourth; a numerological match of 784 and 1,024 was a subtraction of two dashboard figures. **What the mutation checks found in the engine** (not in the pages): the strict parser had an untested gap and two gate lines never decide anything (both recorded).
**The shape of the blog, again.** Of 50 random pages 9 (18 percent; about 10 to 31) have explicit structure and 5 (10 percent; about 4 to 21) are the main subject of a test; pooled with scan 12, 13 of 70 and 7 of 70. 20 of the 50 chosen pages gave tests (40 percent), which is a property of the choice. Rajnish's own ten best posts would replace sampling (BLOG_SCAN_LOG question 15).
**On drift, latency and cost.** The pages that use those words assert and measure nothing (`STRUCTURE_EXTRACTION_5.md` section 6). What this repository has measured is unchanged: one 3B model on one family of graph edits, and the gate's own check at a few microseconds on a tiny delta in the sandbox; new this round is only the size of our code (363 lines for the admission core). That structure lowers latency and cost, and that a larger model alone would not fix the corruption, are hypotheses with a test each: option G for scale, the A/B harness (which could start with a counting eval) for the rest. Count: 406 distinct posts.

**Addendum (8 Oct, a pasted AI summary of Medium pages; `SHARED_STACK_MAP.md`).** The five pages named could not be read (410 Gone), so this is secondhand. What could be counted was checked (`tests/test_grid_13.py`): the 13 x 13 board has 28 symmetry classes, one centre, and a 3 x 3 seed has exactly one symmetric place in it; a 12 x 12 world has none, because 13 - 12 is odd. The five-layer stack (source, ground, grid, operators, conduct) matches the order of this code and mostly repeats rows already mapped from the blog (Shiva 0, Durga +1, Kali -1; the four Devis; the read-only witness). Two disagreements with the blog are open (Bhairavi against Durga; Laxmi as output against resource). Claims of higher reasoning density or lower drift are unmeasured. Questions 109 to 114.

## 12b. What the fourteenth scan added (30 pages read as ordinary text, all new, numbers 414 to 443; `STRUCTURE_EXTRACTION_6.md`)
The instruction was to read plain pages the way they are and see how they hold structure. Five questions were put to each paragraph: the items, the relations, the cell that is absent, what may be swapped, what is thrown away. A passing test shows that the count or identity is right; it does not run a model.
1. **Ordinary text builds lists of three and closes them.** Of 30 pages 10 have an explicit list or numbered parts; none has four items, eight have three, and 8 of the 10 end on one restating line (3 + 1). At least 13 name an opposition pair. Four pages are AI replies pasted as posts and three of those carry a numbered list or a diagram. This is what a parser should expect first (row 166).
2. **The clock sentence is exact (rows 157, 158).** The multipliers of a 12-mark clock that lose nothing are {1, 5, 7, 11}, the Klein four-group; every prime above 3 sits on one of those four marks; on 13 marks the units are one cycle of 12 and only the multiplier 0 loses anything. "0 is 0, no matter what stood behind it" is a many-to-one map: twelve lives cannot all keep their own score on a 0-to-10 scale.
3. **A table with a missing cell (row 159).** The barks-and-bites page names three of the four cells of loud/quiet x acts/idle and never the fourth. The four forms of a conditional are two meanings under the Klein four-group; the page's "inverse" is not the textbook one, and the textbook one is about the cell the page leaves out.
4. **No first moment means a loop or a loss (row 160).** In a finite system "no start", "nothing lost" and "every state on a cycle" are one property (brute force on all 4-node digraphs and all rules on up to 5 states). The gate, taken with its inverses, has no start. The theorem also allows an infinite set; which one the page means is question 116.
5. **Two lines of one shape (row 161).** The Product and Tool diagrams are isomorphic, three of four labels differ, and a line has no feature that marks a "gate": it needs a weight on an edge.
6. **13 as 12 plus a hub (row 162).** A hub joined to a 12-ring keeps the 24 symmetries of the 12; a 13-ring has 26 (question 117, refining 110).
The next step for these results is practical: the five questions are a scoring rule for the A/B harness (does a model's answer to a plain paragraph recover the items, the relations and the missing cell?). Nothing here has been run on a model.

## 13. Questions that decide the most (from `STRUCTURE_EXTRACTION.md`, `STRUCTURE_EXTRACTION_2.md`, `STRUCTURE_EXTRACTION_3.md`, `STRUCTURE_EXTRACTION_4.md` section 7 and `STRUCTURE_EXTRACTION_5.md` section 10)
66 (rows or columns, and the centre), 67 (the axes of the quadrant), 68 (one rule behind "four"), 71 (the missing formula), 75 (which mappings must be checked, and which relation each must keep), 76 (rotation only or rotation and mirror), 78 (what N and the +1 are), 84 (did the Period Test and the 13-hour clock ever run, with what output). The new ones from the eleventh scan: 88 (which prong carries which triad), 89 (a line or a fork), 92 (4, 8 or both neighbours). From the twelfth: 93 (a cycle, and may a line be closed), 94 (are the one-value, forbidden-value and derived-column rules in the gate or in a checker), 98 (drift, latency, cost: which first and against what baseline). From the thirteenth: 99 (which version of the equation is final, and is "=" identity or "is borrowed from"), 102 (are the four pairs independent two-valued axes), 103 (is the empty space headroom for a safe commit, and how much), 106 (are the "NN FIELDS" logs a schema to be checked). The full list of unanswered questions is now 1 to 108.

New from the shared text of 8 Oct: 109 (are the five pieces still published, whose accounts), 110 (what the 13th is), 111 (Bhairavi: space or operator), 112 (Laxmi: output or resource), 113 (the rule that expands a 3 x 3 seed), 114 (horizontal, vertical and zero leverage as checkable rules); see `SHARED_STACK_MAP.md` section 6.

New from the fourteenth scan: 115 (the unstated cell of the barks-and-bites table), 116 (loop or unending line), 117 (hub or ring for the thirteenth), 118 (gate and lever as readable rules), 119 (are the nine steps three groups of three); see `STRUCTURE_EXTRACTION_6.md` section 6.

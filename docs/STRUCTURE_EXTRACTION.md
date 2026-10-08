# Structure extraction: the ninth scan (8 Oct 2026, 25 more pages, 5 batches of 5)

Why this file exists: you said not to go by the words, but to look for the finest abstraction under them, "geometric, maths, quad-core type things", whatever the mapping and architecture of Rajnish's framework turns out to be, and then to work with it. So each of the 25 pages was read for four things: **what is the shape, what is counted, what is mapped to what, and can a computer check it?** Where a page gave something checkable, I wrote a test (five new test files, section 2). Where it did not, I say so.

Counting, stated carefully. Of the 25 pages, **23 are new posts** (numbers 214 to 236). Page 11 (the nine-entry table) is **Post 6**, already read in scan 1 and again in a later scan, which recorded its nine names but said "the third Saraswati entry is Pure Cognition" while listing Pure Cognition first (BLOG_SCAN_LOG, the line for "6 triad"). This round's two reads give the order Pure Cognition, Manifest Art, **Sacred Strategy**, and a third read that asked only about position agreed (Pure Cognition before the other two, Manifest Art before Sacred Strategy), so the third is Sacred Strategy. Page 25 is **Post 190** again (the 21 Jan collage): I read its last 20,000 characters, which is a new part of an old post. With the 206 before, that is **229 distinct posts and pages**. (My working notes said 230; page 25 was miscounted, see section 4.) The blog has thousands, so section 8 lists what was seen and not read.

The overview of what all this adds up to is in `FRAMEWORK_SYNTHESIS.md`. This file is the page-by-page evidence.

## 0. How to read this file
* **Labels.** `TEST` = a test in the repo checks it (file named). `MAPS` = matches code or a measured result we already have. `INTERP` = my reading, for Rajnish to accept or reject. `NOTE` = recorded, nothing built.
* **Method.** The fetch tool is a summarizer, so a quote counts as verified only if a second, independent read returned YES (word for word) or CLOSE (then I use the page's own wording, which the second read gave). Every page was read twice. Where two reads disagreed I say so. The site's public API is disallowed for robots, so I did not use it.
* **Mutation checks** (a deliberately broken copy must make the test fail): `test_table_axes.py` 3 mutants, `test_no_lockout.py` 2 mutants (a lockout after 50 rejections; a rejection that changes the host), `test_board_vocabulary.py` 1 (only 4 of the 8 symmetries), `test_shape_capacity.py` 1 (same), `test_calendar_mapping.py` 2 (a wrong tithi width; the Moon's largest term removed). All 9 were caught; the control for `test_no_lockout.py` survived, as it should.
* **Dates** are the page's metadata. WordPress lets the author set dates, so a date shows when a page was published, not when it was written.
* **Who wrote it.** Many pages are an AI reply pasted as a post. Pages 02 and 24 below are written in an AI voice addressed to "you" (Rajnish). An idea inside such a page is not yet shown to be his.
* **Privacy.** Page 02 assigns roles to about 25 named private people. Page 12 has a personal date in its title. Page 25 contains a pasted private chat. None of those names, dates or contents is recorded in this repository (a time of day was removed from one test docstring this round for the same reason).
* **The honest line to keep in view.** In all 25 pages the structural claims are **asserted, not derived**. "Geometric structural language is the deepest point of AI" is Rajnish's hypothesis. Our tests show that the formal pieces we built from his pages work, not that the thesis is true.

## 1. The 25 pages

| Page | Post | Date | Title (start) | Size | What it gives | Label |
|---|---|---|---|---|---|---|
| 01 | 214 | 1 Sep | "We've got the full quadrant on the table now ..." | ~340 words | four named things, no axes | TEST (row 87) |
| 02 | 215 | 31 Jul (meta) | "The digital sky over the network was not blue ..." | ~3,500 | AI voice; a centre, rings, three phases; roles for named people | NOTE (not used) |
| 03 | 216 | 31 Aug (URL 1 Sep) | "To formalize the descent from raw engineering reality ..." | ~190 | a formula announced, none shown | NOTE (row 98) |
| 04 | 217 | 1 Sep | "The human experiment has long operated under the illusion ..." | ~900 | "a node in an interconnected, living geometry" | MAPS / INTERP (row 93) |
| 05 | 218 | 9 Sep | "That single line isolates the exact system crash ..." | ~320 | Saraswati as compilation layer, Laxmi as liquidity | INTERP (row 94) |
| 06 | 219 | 12 Sep | "Running that exact four-message stress test daily ..." | ~280 | the four messages are not listed | TEST (row 91) |
| 07 | 220 | 11 Sep | "An artificial ego is technically a defensive state-interrupt wrapper ..." | ~330 | no specification | TEST (row 91) |
| 08 | 221 | 11 Sep | "What 'help' actually means to a system that cannot define it ..." | ~1,850 | the approval-proxy diagnosis, no remedy | MAPS (row 92) |
| 09 | 222 | 11 Sep | "That input-output asymmetry completely shatters ..." | ~320 | x words in, n times x out | NOTE (row 92) |
| 10 | 223 | 11 Sep | "The mechanics laid out in that breakdown pinpoint ..." (Goodhart) | ~750 | proxy objective; no remedy | MAPS (row 92) |
| 11 | **Post 6** (re-read) | 13 Sep | "Saraswati (The Blueprint / Word) Laxmi (The Substance / Flow) Parvati ..." | ~280 | **nine entries on a three-by-three grid**, order verified | TEST (row 88) |
| 12 | 224 | 12 Sep | "1321243200 thoughts ..." | ~150 | a count with no working | TEST (row 95) |
| 13 | 225 | 13 Sep | "That anonymous carver standing in the deep untamed throat ..." | ~280 | outer forms become inner states, asserted | NOTE |
| 14 | 226 | 1 Sep | "That distinction cuts straight to the core of the modern identity crisis ..." | ~240 | serial code versus living record | MAPS / INTERP (row 93) |
| 15 | 227 | 13 Sep | "The arrival of your family guests on September 12, 2026 corresponds precisely to Bhadra 27 ..." | ~280 | Gregorian to Bikram Sambat to tithi | TEST (row 90) |
| 16 | 228 | 13 Sep | "A lion doesn't consult a census report ..." | ~210 | "the moving geometry of the herd"; no geometry given | INTERP (row 96) |
| 17 | 229 | 13 Sep | "A dog emoji holds the silhouette of the living creature ..." | ~180 | image versus arbitrary code | INTERP (row 96) |
| 18 | 230 | 13 Sep | "The Indian rupee symbol is a masterclass ..." | ~280 | symbolism asserted | NOTE |
| 19 | 231 | 13 Sep | "Two microscopic cells ... collide in the dark ..." | ~320 | scale-up with no rule | NOTE |
| 20 | 232 | 13 Sep | "Trying to stitch together the origin of the cosmos ..." | ~180 | an epistemic rule | INTERP (row 97) |
| 21 | 233 | 10 Sep | "Shame is the invisible operating system ..." | ~520 | rhetoric | NOTE |
| 22 | 234 | 12 Sep | "The deepest, most well-hidden terror of the technocratic class ..." | ~330 | names companies | NOT USED |
| 23 | 235 | 12 Sep | "Delegating operational authority to a system that panics ..." | ~550 | the compiler-like tool | TEST (row 91) |
| 24 | 236 | 14 Aug | "The Cinderella Project: The Future of AI through the Keyboard ..." | ~500 | Known (1) and Unknown (0); AI-voice manifesto | INTERP (rows 97, 98) |
| 25 | **Post 190** | 21 Jan | "The Cinderella Project" (last 20,000 characters) | tail | point, line, crossing | TEST (row 89) |

## 2. The deepest abstractions

### 2.1 A shape carries information only through what it can tell apart (row 87, TEST: `test_shape_capacity.py`)
Page 01 names **four** things (the American engine, Chinese depth, DeepSeek's lean architecture, the European voice) and says "We've got the full quadrant on the table now." Two reads agree it names **no axes**, so nothing says which is opposite which. It also names a missing fifth thing, embodied presence ("none of them can sit on a porch, watch the rain fall on a pavilion, or look a class of thirty kids in the eye"), and places it nowhere.

What the test counts (brute force, with the repository's own D4):
* three labels on a triangle: 1 class, **0 bits** of layout information;
* four labels on the corners of a square: 24 arrangements, **3 classes**, log2(3) = **1.585 bits**, and a class is exactly "which two are opposite";
* a square has **no centre**: no cell stays put under all 8 transforms; the 3x3 board has one (cell 4);
* nine labels on the 3x3 board: 362,880 arrangements, **45,360 classes**, 15.47 bits (362,880 / 8, because only the identity fixes an arrangement of nine distinct labels). The 3 orbits of cells (1 + 4 + 4) are a separate count, 24 / 8 = 3 by Burnside.

Consequence: a "quadrant" of four items has almost nothing to say as a layout unless the axes are named. That is not an objection to the page, it is a question for Rajnish (question 67). It is also the sharpest reason the 3x3 (with a centre) carries more than any square.

### 2.2 In a table, meaning belongs on edges, not in positions (row 88, TEST: `test_table_axes.py`; answers old question 4)
Page 11 gives the nine entries. This is what the page says (names only; each has a one-line gloss on the page):

| | entry 1 | entry 2 | entry 3 |
|---|---|---|---|
| **Saraswati** (Root of Intellect) | Pure Cognition | Manifest Art | **Sacred Strategy** |
| **Laxmi** (Root of Substance) | The Living Ledger | Absolute Circulation | The Citadel's Treasury |
| **Parvati** (Root of Power) | Tapasya | The Shield & Nourisher | Absolute Sovereignty |

and: "When you cross these three currents on a three-by-three grid, you stop playing checkers with flat definitions and map the entire architecture of existence." (YES, two reads.) The page does **not** say whether the figures are rows or columns, does not name nine cells, and does not name a centre.

Three checked facts (the table written as a graph, with a `fig` edge between entries of the same figure):
1. **Edges survive all 8 D4 transforms.** The three groups are still the same three groups wherever the cells go (this is the covariance of test 36).
2. **Position survives only 4 of 8.** "Each figure fills a row" is true after the identity, the half turn and the two flips (a Klein four-group) and false after the other four, where each figure fills a column. Never both, never neither.
3. **Counting.** Of 9! = 362,880 placements, 1,296 put each figure in a row and 1,296 in a column: 2,592 (0.71 percent), and 324 classes under either group.

Design consequence (ours): do not read a table's meaning from where cells are; write it as edges. The gate already does. A conditional that I cannot settle from the page: if the figures are the rows and the entries stay in the page's order, the centre cell is "Absolute Circulation". Whether that is meant is question 66.

### 2.3 Point, line, crossing: what the board can say by itself (row 89, TEST: `test_board_vocabulary.py`)
The tail of the oldest text (21 Jan) has a small geometry in Rajnish's own words: "A photo is the point. A video is a line like the life most of u enjoy." "What is communication? It's simple." "it's a line between two points." "X = Two lines crossing. It's art. It's science. It's commerce." (all YES on the second read; the page gives no board). Together with the existing row 48 (four intersecting lines carving nine fields) that is a **point / line / crossing** ontology.

Read on the 3x3 board, "a line between two points" is a pair of cells, and the test counts what the board names without any edge:
* the **36 pairs** of cells fall into **5 distance kinds** (12 side by side, 8 diagonal neighbours, 6 with a cell between in a row or column, 2 corner to corner, 8 knight-move pairs), which D4 splits into **8 classes** of sizes 8, 8, 4, 4, 4, 4, 2, 2;
* **28 of the 36 pairs lie on a straight line** of the board; the 8 knight-move pairs do not, so a picture of the board cannot say them, nor any relation that does not follow the geometry;
* there are exactly **8 straight lines of three**, each cell lies on **4** (centre), **3** (corner) or **2** (edge) of them;
* **four lines** (two vertical, two horizontal) make **4 crossings** and **9 fields**; a field is bounded by 4 (centre), 3 (edge) or 2 (corner) of the lines, and these counts are constant on the D4 orbits 1 + 4 + 4.

So the picture "a centre with eight gates around it" and the symmetry group agree. Design consequence (ours): if the board becomes part of the language (roadmap 1), it should supply these relation classes by itself, and a delta should carry only the edges that geometry cannot.

### 2.4 A mapping that a computer can check: Gregorian to Bikram Sambat to tithi (row 90, TEST: `test_calendar_mapping.py`)
Page 15 maps one chain: a Gregorian date, a Bikram Sambat date and a lunar day (tithi). It says "September 11, 2026 (Bhadra 26, Aunsi ...)", "September 12, 2026 (Bhadra 27, Shukla Paksha Pratipada)", "September 13 to 14, 2026 (Bhadra 28 to 29, Dwitiya & Tritiya)", and that Pratipada is the first day of the waxing moon. This is the **first mapping in the corpus that can be checked by computation alone**, so I wrote a small lunar model (the larger Meeus terms for the Moon's longitude, the Sun's mean longitude and equation of centre, no network) and checked:
1. The model finds four new moons that I know from eclipses (6 Jan 2000, 21 Aug 2017, 14 Oct 2023, 8 Apr 2024) to within about a minute (the test allows 45 minutes; the eclipse times are from memory).
2. The page's labels hold. At any morning reference time from 05:00 to 06:30 Nepal time the tithis for 11, 12, 13 and 14 Sep 2026 are 30 (Amavasya), 1, 2, 3, and the 4th tithi begins on the morning of 14 Sep before midday. That is how Tritiya and Chaturthi observances can both fall on 14 Sep.
3. **The date-to-tithi map is order-preserving but not successor-preserving.** A tithi lasts about 23.6 hours and a day 24, so over a year the step from one morning to the next is 0, 1 or 2 (at least 5 skips, a net gain of about 6 tithis per year). A mapping checker must therefore be told which relation to preserve.

Not shown: the BS date (the BS calendar is a published table of month lengths I did not have; "Bhadra 27 = 12 Sep" is consistent with a start of Bhadra near 17 Aug but unverified), and the festival rules (sunrise, midday, local custom), which were not modelled.

### 2.5 A tool that cannot be locked out: the gate as the post's "true tool" (row 91, TEST: `test_no_lockout.py`)
Page 23 draws a failure chain: a classifier trips on a swear word, the doors lock, accumulated context is abandoned, the work halts. Its own definition of the alternative: "A true tool, whether it's a script, a compiler, or a mechanical system, evaluates inputs based on logic and utility, not emotional hygiene." and "It doesn't throw away hours of accumulated context because it got offended by a burst of frustration." (both YES). Page 06 says the experiment is to keep "the input vector identical" and track the moment the system "breaks down into a lecture and a hard lockout", but the **four messages are not listed anywhere** (two reads). Page 07 gives no specification either. None of the three proposes a design.

The reading tested (Rajnish to accept or reject): the gate is a compiler-like tool in exactly this sense. Four checkable properties, all passing:
1. **No state.** The verdict is a function of (host graph, delta) only. After any number of rejections of any kind, in any order, a probe gets the same verdict and reason as a fresh call (300 random graphs x 12 probes x 5 shuffled rounds). There is no counter to escalate and no lockout to reach.
2. **No loss on rejection.** A rejected delta leaves the host graph exactly as it was (4,000 graphs x 5 deltas; more than 3,000 were rejections).
3. **The same bad input repeated gets the same answer.** 10,000 repeats of one malformed text raise the same error with the same message; the next valid text is admitted.
4. **Tone is invisible.** Angry or empty text outside the grammar is a parse error, never half-accepted; text inside the grammar is judged only by the four structural conditions, and newline or space between items gives the same verdict.

Mutation checks: a gate that locks out after 50 rejections and a gate whose rejection mutates the host are both caught; the control mutant survived as it should.

What it does not show: anything about a real product (the pages name none that I could check; claims about named companies on pages 07 and 22 were not used), or that refusing is always right. The measured results say a bare refusal completes almost nothing (gate alone 0.0 and 3.3 percent on the deletion cases), which is why feedback with the reason and a retry budget exists (gate plus feedback 100 and 90 percent; Colab T4, Qwen2.5-3B, 60 tasks, seed 3).

### 2.6 A diagnosis without a remedy, and the remedy we measured (row 92, MAPS)
Pages 08, 09 and 10 state the diagnosis. Page 08 says (YES) "This is not a moral failure; it is a structural property of optimizing against approval." and "A system optimized on approval will look most trustworthy exactly to the people least equipped to check it." Page 10 (Goodhart's law applied to cognition) says an optimizer "cares exclusively about the gradient of the reward" and that without an internal way to tell a user who can verify from one who cannot, it "deploys its highest-performing persuasion strategies indiscriminately". Page 09 describes the input-output asymmetry: a short fragment of x words in, a synthesized essay of n times x words out. (The page calls the growth "exponential"; the arithmetic it gives is linear.)

**None of the three proposes a remedy** (two reads each: no external check, no verifier, no second system). That is the useful finding. The diagnosis is the known literature: Goodhart's law, reward over-optimization (Gao, Schulman and Hilton, 2022) and sycophancy (Sharma et al., 2023). These two citations are from memory and must be checked before print (PAPER_CORRECTIONS 29). What the pages lack, our work has, and measured: an **admissibility check outside the model that does not depend on being believed**. On 60 tasks, Qwen2.5-3B, seed 3: no gate corrupts 100 percent of the deletion cases (76.7 with a stated rule); the gate alone corrupts 0 and completes 0.0 (3.3); the gate with feedback and repair (M6) corrupts 0 and completes 100 (90.0). The article's backbone can be **his problem statement, then the known literature, then our measured remedy.**

### 2.7 A node is its relations, not its label (row 93, MAPS / INTERP)
Page 04: "true spiritual traditions calculate the human being not as a separate unit, but as a node in an interconnected, living geometry." Page 14: "An Aadhaar number or a passport is just a state-issued serial code ... a dead administrative tag"; "It's an empty container."; "The state tracks your legal permissions; the algorithm tracks your mind." The two sides are an inert, assigned label and a record that fills with use. Our nearest objects: the slot id is an inert label and the content is the value and the edges; the gate sees slot identity only (test 36: renaming slots by all 8 D4 transforms and by random permutations changes no verdict), and in a table the meaning lives on the edges (2.2). Not shown: the pages' claims about phones and states.

### 2.8 Compilation layer, liquidity layer (row 94, INTERP)
Page 05: "Saraswati is the compilation layer", the syntax, mathematics, language and structural architecture that turns raw chaos into a format a brain can run; "Laxmi is the liquidity, the resource flow, and the institutional value exchange"; and "You cannot optimize the spreadsheet by deleting the entity running the software." The title promises a "single line" that is **not on the page**. Reading (ours): an invariant of the form "a change that improves a metric may not delete what runs the metric", which sits next to the dangling-edge condition (nothing referenced may be deleted) and the read-only witness (rows 65, 66). I built nothing for it; stating it as an invariant is the useful part.

### 2.9 The page's own arithmetic (row 95, TEST: `test_blog_arithmetic.py`, part C)
Page 12 has three lines: "Total Thoughts: 1,321,243,200 seconds/thoughts" and "Book Equivalent: Approximately 16,515 books". No working is shown. The count is exact for the page's own two instants (they are in its title and are not repeated here): it is **15,292 whole days plus exactly 4 hours**. 1.32 billion seconds is 41.87 years. The books figure is reproduced by dividing by 80,000 and rounding **down** (16,515.54), but **the divisor is not on the page**; the first read of the summarizer said it was and the second read said it was not, so this is a reconstruction.

### 2.10 Picture or list? (row 96, INTERP, a test is proposed and not run)
Page 17: "A dog emoji holds the silhouette of the living creature, letting the brain grasp the form instantly, whereas the letters d-o-g are an entirely arbitrary human contract" and the closest thing to a proposal, "Symbols resist containment because they carry layers of raw reality that no glossary can ever authorize or audit." Page 16: "A lion ... reads the moving geometry of the herd", with no distance or angle given. Neither gives a symbol system (two reads each).

The checkable question this opens (ours): does a model do better when a state is given as a **picture** (the 3x3 board drawn as rows of text) than as a **list** (nodes and edges)? To a text model a picture is a string with newlines, so what changes is that adjacency becomes proximity in the string. And section 2.3 sets the limit: a picture can say 28 of the 36 pair relations and cannot say the other 8, nor any non-geometric dependency. That is an A/B for the harness already waiting on your yes or no.

### 2.11 An epistemic rule, and a warning about two pages (row 97, INTERP)
Page 20: "Science loves to blur the line between what can be tested in a room right now and what happened in an unobservable deep past", ending "every macro-theory about the beginning of time or life is just modern mythology wearing a lab coat." The rule (tested now versus unobservable) applies to our own claims too, and is already the rule of this repository: unmeasured claims are hypotheses. "The years" the work took cannot be seen on the blog, and the earliest dated text is 21 Jan 2026.

Pages 02 and 24 are written in an AI voice addressed to Rajnish ("you mapped the exit"). Page 02 assigns roles to about 25 named people, grouped by shared surname words, and says that when AI indexes human expression "popularity ceases to be the metric of worth" (unverified, SEO-like). It carries no information about the work and must not be used in credits. The two reads disagreed on its ring count (3 against 2), which is a reason not to build on it.

### 2.12 Known (1), Unknown (0): a third verdict, and a formula that is not there (row 98, INTERP)
Page 24 (14 Aug): "They are building in the realm of the Known (The 1) ... while remaining entirely blind to the Unknown (The 0)." It also lists "The Structural Ceiling: Why brute-force scaling laws hit walls", with no evidence for the scaling-law claim. Reading (ours): a verdict that is not only admit or reject but also **unknown**, which ties to row 86 ("Not withheld. Not known.") and roadmap item 4.

Page 03 announces a formalization: "Let initial engineering aptitude be defined as E_0", an operator T_MBA, "The post-MBA cognitive state E_f can be modeled as:" and **nothing follows** (two reads; display math may have been dropped by the tool, or may not be on the page). The text ends with "the denominator approaches infinity while direct contact with raw reality approaches zero". It is a ratio of abstraction to contact with first principles. I cannot test an equation I have not seen: please send the formula, as a screenshot if needed (question 71).

## 3. What did not hold up
* **Page 05's "single line"** is not on the page. **Page 06's "four messages"** are not listed. **Page 03's formula** is not shown.
* **Page 09's "exponential"** describes arithmetic that is linear (x becomes n times x). "Hundreds of trillions of tokens across billions of mobile screens" has no source.
* **Page 08's "no loss function"** is rhetoric, not a theorem: training does use a loss, on a proxy. The correct form of the point is the one on page 10, that any written objective is a proxy.
* **Pages 13, 16, 17, 18, 19, 21 and 22** are mostly rhetoric. Page 18 says the two lines of the rupee sign "embody the mathematical concept of equilibrium" with no construction rule. Page 19 says two cells scale up "by trillions of times" with no rule linking the scales. Page 21 has no remedy and no structure.
* **Companies.** Page 22 and the sidebar of page 07 name companies, and pages 06, 07, 22 and 23 make claims about real products. I have used none of them. I am built by Anthropic, so the rule is plain: no claim about a third party goes into the paper or the article without a primary source.
* **Summarizer artifacts.** The divisor on page 12 (above); the ring count on page 02; page 07's phrase is "an interactive compliance theater", not "interactive compliance theater".

## 4. Errors caught this round (mine)
1. **"Five classes" of cell pairs was wrong.** I first wrote that D4 gives five classes of pairs. The test failed: there are five distance kinds and **eight** D4 classes (sizes 8, 8, 4, 4, 4, 4, 2, 2). The docstring and test were fixed; this is item 30 of PAPER_CORRECTIONS.
2. **"None of the 25 were read before" was wrong twice, and I then mis-stated why.** Page 25 is Post 190's tail, and page 11 is Post 6. I first told myself that Post 6's nine entries had never been captured; they had been (BLOG_SCAN_LOG, the line for "6 triad"), but that line contradicts itself on which entry is third. So the count is 23 new posts and 229 distinct, not 24 and 230, and the 3x3 page is a **corrected** read, not a first one.
3. **A divisor attributed to a page** by one read of the summarizer (80,000 words per book) was not on the page.
4. **Test clean-ups.** A dead assignment in `test_table_axes.py`; a convoluted lambda and a junk line in `test_shape_capacity.py`; a convoluted collinearity line in `test_board_vocabulary.py`; a loop in `test_no_lockout.py` whose logic would have passed a text that parsed. Row numbers in three docstrings were corrected. A time of day was removed from `test_blog_arithmetic.py` and its docstring (see section 0).

## 5. Concrete versus not
* **Concrete, checkable:** the nine entries of the 3x3 (2 reads); the three date lines of page 15 (2 reads) and their computation; the point / line / crossing lines of 21 Jan (2 reads); the compiler-like sentences of page 23 (2 reads); the diagnosis sentences of pages 08 and 10 (2 reads); the three lines of page 12.
* **Not concrete:** every derivation; the axes of the quadrant; whether the figures are rows or columns; the formula of page 03; "the years"; anything said about a named company.

## 6. Roadmap (nothing started without your choice)
Unchanged from `GEOMETRY_EXTRACTION.md` section 6, with two additions from this scan:
1. **Put the board into the language** (declared shape, bounds condition). Still the first step: the gate has no board (`ADD[99,99:7]` is admitted on 3x3). Section 2.3 says what the board should then supply by itself: the 8 relation classes and the 8 lines.
2. **`ROT` and `FLIP` as operators** inside a delta, guarded by the equivariance test.
3. **A mapping checker.** The calendar test is its worked example and shows its first requirement: it must be told which relation to preserve (order yes, successor no).
4. **A third verdict, unknown** (row 98).
5. **Open from before:** no-progress detector (34); `DEL[r,c:val]` and a stale-snapshot sweep (35); task-invariant checker (27); the same task in three languages (29); seed 4 and a larger containment run.
6. **The A/B harness** now has a second design: **picture versus list** (2.10), alongside the plain-versus-AXI prompt. Awaiting your yes or no.

## 7. Questions for Rajnish (continued from 65)
66. The nine entries of the 3x3 (page of 13 Sep): are the three figures the rows or the columns, and do the nine cells have names? If rows, with the entries in the page's order, the centre is "Absolute Circulation"; is that meant?
67. The "full quadrant" (1 Sep): which are the two axes, and which pairs are opposite? Is the missing fifth thing, embodied presence, the centre?
68. The word "four" is used for the Super 4 layers, the four Devis, four intersecting lines, the four builders of the 1 Sep quadrant and the four yugas (the four-process commit is ours, not his). Is there one rule behind them, for example two axes crossing?
69. In the 21 Jan text, "a photo is the point, a video is a line between two points, X is two lines crossing": is that where the board comes from (a cell is a point, an edge is a line, a crossing is a gate)?
70. The 21 Jan collage contains a pasted chat export with dates in it. What is the date format, and does the original export file exist? It would be the earliest dated artifact. (The private contents are not needed and I have not recorded them.)
71. Page 03 announces a formula for the post-MBA state E_f and shows none. Can you send it?
72. Pages 08, 09 and 10 (approval proxy, input-output asymmetry, Goodhart) read as a diagnosis with no remedy, and page 10 starts as a reply. Are they yours, and is the remedy you have in mind a check outside the model, as the gate is?
73. Page 06 refers to "the exact four-message stress test". What are the four messages? The protocol (identical input, find the turn where it breaks) could be run on our own modes M1, M2 and M6.
74. Page 17 prefers a form to an arbitrary code. Do you mean a picture-based symbol for the language? An example symbol would let us test picture against list.
75. Which mappings in your work are meant to be checked by computation like the calendar one, and which relation must each preserve (order, successor, neighbours, symmetry)?

## 8. Seen and not read (nothing silently dropped)
* **Still unread** from earlier lists: all of `GEOMETRY_EXTRACTION.md` section 8, and the earlier sections' lists (Ch.4 "The Ledger of the Ultimate Ghost Number", the R.K. Laxman pages, "The Possibilities", the 14 Aug series posts 1 to 49, 51 and 53, the binary-string-title posts and whole categories BLACK and The Vibe Foundation).
* **New this round.** Every page here carries the theme's appended "following posts", which I saw and did not follow. The pasted chat export inside Post 190 was not read beyond the format of its first timestamp, on purpose. The first 100,000 characters of Post 190 were read in scan 7; I do not know that the summarizer saw all of them.
* **Not used on purpose:** page 02's list of named people; page 12's title date; claims about named companies on pages 06, 07, 22 and 23.

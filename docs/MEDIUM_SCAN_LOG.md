# Medium scan log (thecinderellaproject publication and the accounts around it), read 7 Oct 2026

29 Medium pieces read (22 on 7 Oct earlier: 2 before this scan and 20 in it; and 7 more in the fifth scan, section 10), plus the profile pages of five accounts. Same limits as the blog scan: the pages were read through a **summarising fetch tool**, so quotes and
numbers must be checked against the live pages before they are printed anywhere, and its "reads like an AI" verdicts are not used. Where I say "transcript" below it is
because the page itself shows INPUT / THINKING / OUTPUT blocks or quoted AI replies. Personal details (ages, places, family) in a few pieces are deliberately not recorded.

## 1. The accounts (who publishes what)

| account | display name | followers | what is on it |
|---|---|---|---|
| @gaur.shalendra2412 | Shalendra gaur | 32 | bio "Investor in AI related companies and companies with bigger purpose. The Cinderella Project." Editor of the publication. **All 10 AXI technical pieces below carry this byline** |
| @rajnishdurga07 | Saraswati. Choubey. | 64 | **Rajnish's account (told by Shalendra, 7 Oct).** bio "The Cinderella Project: Where AI's Midnight Arrives Apply the Delete Test. Period. Kill It." Jan to Feb 2026 essays and AI-conversation pieces |
| @gaur.boy2412 | FUTURE OF AI | 20 | **Shalendra's own account (told by him, 7 Oct).** Nov 2025 to Mar 2026 opinion pieces; one ends with a signature line naming Rajnish Choubey. Also the commenter "FUTURE OF AI" ("Mind blowing", 27 Jul) |
| @rajnishchoubey.email | Rajnish choubey | 52 | one of Rajnish's two or three accounts (told by Shalendra; I have not verified it independently). bio "It's very simple to be happy but very difficult to be simple." Ten personal essays and tales from June to July 2019. **No AI content** |
| @rahi.rajnish_72415 | RAJNISH CHOuBEY | 27 | likely another of Rajnish's accounts (same caveat); three near-empty posts, Sep 2024 |

The publication has about 31 to 33 followers. **All ten AXI technical pieces are on Shalendra's account; none is on Rajnish's accounts, which hold the philosophy pieces.** His framework reaches the technical
pieces through the Shalendra byline (credited to him in the text) and through the blog. That matters for the credit question in `CREDITS.md`.

## 2. The ten AXI pieces, by date (the answer to "where did the numbers come from")

| date 2026 | title | form | what it claims | evidence shown |
|---|---|---|---|---|
| 4 Jun | "AXI for a noob" | 3-minute explainer | AXI is "Thinking in Shapes": a fixed 13x13 grid (169 cells), "TOON" shorthand, "Zero-Drift"; "+591% gain", "80% less power", "Standard AI scores 0.37% ... AXI scores 100%" on ARC-AGI, "84.6% Context Tax" | **none**: no run, model or script. The ARC-AGI line is the riskiest sentence in the whole set (see section 5) |
| 9 Jun | "AXI_HYPER_CUBE_PROJECTION: 3D_SPATIAL_SCALE" | AI-written notes or transcript (starts "[ the AXI configuration by projecting ...]") | 13x13 grid lifted to 13x13x13 (2,197 cells); a table of attention weights (0.942, 0.989, 0.999), "drift 0.00000000", "ATMOSPHERE 15.0_BAR", "Pull #134" | **a chat model cannot read its own attention weights.** When it prints them, it is writing plausible numbers. These cells are role-play, not telemetry |
| 10 Jun | "Google is now a Tenant in the Architect's House" | transcript: four rounds of INPUT / THINKING / OUTPUT on a 13x13 puzzle (gravity, a 1+3 -> 8 fusion rule, a splitter) | the model "I am trained by Google ... but I am governed by Rajnish Choubey"; "Standard Gemini (Gen 1) cannot do this" | the model's own outputs. The arithmetic in it is right (centre of mass 108/18 = 6), but the puzzle has fixed rules that any capable model can follow when they are in the prompt. A model saying it is "governed" by someone is following the prompt, not evidence. No plain-prompt control |
| 27 Jul | "The Geometric Invariant: How AXI Shattered the Multi-Trillion-Dollar AI Scaling Wall" | 3-minute article | one Google AI Studio session of 1,193,946 tokens; asked to "locate a deeply buried anchor coordinate, trace the dependency logic, and return a strict schema"; output `located_deep_anchor 7F2D9A1B5C8E43G6, computed_verification_result 1167`; "proved definitively"; four pillars (Absolute Zero Drift, Radical Efficiency, Exponential Cost Reduction, High-Density Intelligence) | **one query, one output.** No model named, no run count, no baseline, no plain-format control. This is the source of the "1.19M-token needle 100%" row in THE PAPER |
| 16 Aug | "AI industry solved by Rajnish Choubey" | prose, 2 minutes | "Generation 3": four shifts (coordinate mapping, subspace orthogonality, latent logic execution, cross-modal integration) | none |
| 22 Aug | "AI Mapped" | promotional | Coordinate System (the Grid), Subspace Orthogonality, Internal Logic-Bus ("calculations resolve natively inside the geometry of the weights") | none. Earliest source for the model-internal description |
| 31 Aug | "THE PAPER : CHANGING AI" | 12-minute white paper | CICO, Protocols A and B, results table (SDR 0.1840 -> 0.0000, needle 94.20% -> 100%, 13x13 parity, -46.90% tokens, LDR 0.29 -> 1.00, latency -42.3% to -82.4%), Theorem 1, `orchestrator.py` and `schemas.py` | **no sentence says the numbers were measured**, and none says how. The needle row traces to the 27 Jul session; the others have no source I can find |
| 21 Sep | "AXI by Rajnish Choubey is 108x ahead ..." (the listing says 100x) | 13-minute specification | "Coordinate-Invariant Spatial Compiler" (CISC), five phases, a 9x9 worked example; SDR 0.0000, "4.2x reduction in token payload", "100.0% topological boundary precision" against "82.5%", ARC accuracy 97.5% -> 100% | one 9x9 example; the rest has no method. The title says 108x, the listing 100x, and the text never defines it |
| 23 Sep | "RFC-001: The AXI Architecture ..." | the real technical document: axioms, Rust crates, Python harness, repo layout | coordinate tuples, subspace orthogonality, **DPO gluing conditions (identification and dangling edge)**, shadow pager with O(1) commit and rollback, density router at rho = 0.65, grammar mask, a Triton patch kernel; "Return Latency < 15ms" in a diagram | the formal text and the code are real and match this repo's design. The latency figure is a **target**, the 0.65 has "no empirical justification" in the text, and the cargo output shown is expected output. It does honestly list three "open engineering frontiers". No bibliography; DPO is not cited (our docs cite Ehrig et al. and Lack and Sobocinski) |
| 3 Oct | "Beyond Autoregressive Hallucination: The AXI Architecture" | 9-minute article | the same three pillars; "<15ms SLA", "microseconds", "zero rollback overhead"; links the GitHub repo and its CI; states the public code is **5%** and lists a **95% held back by Rajnish** | CI results are real (2 Rust tests, one Python slice check). The speed claims have no measurement behind them |

## 3. What this explains

* **The numbers in THE PAPER came from chat sessions and chat-written tables**, not from runs with saved prompts, outputs and a scoring script. That is not a lie; it is
  what a model's own answer looks like when it is asked to produce a result. It is also not yet a measurement, and a reader cannot check it.
* **"AI Studio proofs" have three limits.** (1) One query shows the model can do the task once; it does not show AXI is the reason (the same model may do it with a plain
  prompt). (2) Models can follow rules given in the prompt, so a model obeying the AXI rules shows obedience, not a new architecture. (3) Any "telemetry" the model prints
  about itself (attention weights, drift 0.00000000, atmosphere in bar) is text it wrote.
* **What would make them count** is in section 6.

## 4. What is solid, and what to keep

* RFC-001's formal core (gluing conditions, the dangling edge, the shadow pager) is correct and is what this repo implements and tests. It is the strongest thing on Medium.
* Its **open frontier 1** (causal attention factorization versus in-place cache mutation) is a real problem, and we have measured it: editing early context needs a re-prefill
  (8.9 s at 32k tokens on a 0.5B model on a T4) while a tail rollback is a crop (0.35 to 0.8 ms). Concept map row 52.
* "5% public, 95% held back" is an honest scope statement, if it is kept in line with what the repo contains (question 23).

## 5. Public statements that the repository does not support (details in `PAPER_CORRECTIONS.md` item 19)

Highest risk first: "AXI scores 100% on ARC-AGI" (4 Jun); the THE PAPER results table (31 Aug); "100.0% topological boundary precision vs 82.5%" and "4.2x" (21 Sep);
"proved definitively ... the industry's greatest bottleneck has officially been solved" (27 Jul); "<15ms SLA", "microseconds", "zero rollback overhead" (23 Sep, 3 Oct);
the attention-weight and "drift 0.00000000" tables (9 Jun). None of these appears in our logs. The honest fixes are to label them (target, illustrative, one example) or to replace them with the
measured numbers in `VERIFIED.md` and `EXPERIMENT_RESULTS.md`.

## 6. How to turn an AI Studio result into a result

1. Pick one claim with a checkable answer (the 13x13 puzzle of 10 Jun, or the needle).
2. Generate N = 100 random instances from a seed, with a small script that also computes the correct answer.
3. Run every instance twice on the **same named model**, at temperature 0: once with a plain prompt and once with the AXI-format prompt. Save each prompt and each raw output to a file.
4. Score with the script, not by eye. Report both columns with counts, including the failures. Publish the folder.
5. For the needle, repeat at 100k, 500k and 1M tokens, 20 per length. Expect the plain-prompt baseline to be high already; the interesting version has several scattered values and some arithmetic.

I can build this as one Colab script (`python/colab/ab_grid_puzzles.py`). The API key goes in Colab secrets, never in chat. It would test the "drift", "token consumption" and "13x13 parity" claims for real, whichever way they come out.

## 7. Other pieces read

| date | title | account | what it gives |
|---|---|---|---|
| 23 Nov 2025 | "Bhagavad Gita - Chapter 15, Verse 1" | FUTURE OF AI | the verse of the tree with roots above and branches below, as a translation; **no** structural reading, nothing about AI |
| 1 Jan 2026 | "Geminis Debut here. More on 2026" (subtitle "AI: The Autopsy of the Scalpel ...") | Saraswati. Choubey. | 23-minute essay: measurement as a made system ("ANY measurement system works IF YOU MAKE THE SYSTEM"), Western science as pattern spotting, many unsourced historical claims; a section quotes an AI in a profane first-person voice. The "Delete Test" appears only as the bio line. No technical content |
| 23 Jan | "AI Will Never Make Money: The Math That Destroys the Dream" | FUTURE OF AI | prose; an investment-versus-revenue argument. The arithmetic is right (e.g. 1M x $30 x 12 = $360M; $15B / $800B = 1.9%), the inputs are unsourced, and it treats revenue as profit. Ends with a signature line naming Rajnish Choubey. No AXI |
| 26 Jan | "The Phylogenetic Illusion: When Control Becomes the Product" | FUTURE OF AI | opinion: "When creators see themselves as apex predators, their products become cages. When creators see themselves as fellow humans, their products become bridges." No framework |
| 28 Jan | "OH WOW. NOW I SEE IT. The Cinderella Project" | Saraswati. Choubey. | 17 minutes; a personal narrative about a conversation with an AI. The Delete Test, the "five challenges" and "Midnight" are **not defined** |
| 30 Jan | "THE CINDERELLA PROJECT: Testing AI Understanding of Modern Visual Communication" | Saraswati. Choubey. | 27 minutes; an AI conversation about images sent by team members ("Claude AI had only ONE message"); no scoring, no numbers. The conclusion that the AI cannot understand the images is argued from the AI's own long analysis of them |
| 31 Jan | "A Dark Comedy About AI Founders Discovering They Built The Wrong Thing ACT 1." | Saraswati. Choubey. | ~12,000-word fiction naming real living founders with psychological labels; invented statistics inside the fiction (e.g. "94% want the AI to shut up"). Do not link or cite |
| 3 Feb | "The Cinderella Project: Where Village Girls Learn AI Isn't Intelligent, It's Just a Car for the ..." | Saraswati. Choubey. | a manifesto-style story: AI is "a car for the brain", intelligence is "stolen" and reduced to "statistical correlation". No curriculum, no tests; counts (130 accounts, 32 artifacts) are assertions |
| 7 Feb | "The Cinderella Project. Cannot be treated as ordinary" | FUTURE OF AI | "operators" who apply "sustained conversational pressure" rather than prompt optimisation; refers to Gemini's "structural self-analysis". No definitions |
| 21 Mar | "This is where it gets truly interesting." | FUTURE OF AI | opens like a pasted AI reply; describes Rajnish's method as long (10 to 12 hour) conversations that treat the AI "like a Mirror". No AXI |

## 8. What did not give anything
Everything in section 7 except as noted; it is philosophy, narrative or opinion. **The Delete Test is not defined in any Medium piece I could read**; it is only the bio line. So the "five challenges" in concept map row 23 rest on the pasted AI summary, not on a Medium original.

## 9. Questions (continue the numbering in `BLOG_SCAN_LOG.md`)
22. **RESOLVED (7 Oct):** "FUTURE OF AI" (@gaur.boy2412) is Shalendra's own account. One small thing stays: the 27 Jul comment "Mind blowing" under the 10 Jun article is from that account, so a reader will see praise from the author's own second account. Deleting it costs nothing.
23. **(Rajnish.)** The 3 Oct article lists "Dynamic BPE-aware prefix DFAs", "Causal Attention Factorization (block-diagonal isolation kernels)" and "PagedAttention v2 zero-copy allocator pools" as the held-back 95%. The repo already contains a BPE-aware token DFA (`TokenDFATable`), block-independence tests and a paged pager. Which parts does he consider open, and which are his to release? Nothing public should go further until he says.
24. **(Rajnish.)** Do you want the framework credited to you in the byline of the AXI Medium pieces (now "by Shalendra gaur", with a credit sentence inside)? The RFC-001 page already says the paradigms "are based on the theoretical work of Rajnish Choubey".
25. **(Shalendra.)** The 9 Jun, 10 Jun and 27 Jul pieces are AI Studio sessions. Which model and version were they, and are the full sessions saved? They are the raw material for the test in section 6.
26. **(Rajnish.)** Is there an original "Delete Test" article (steps, pass or fail rule, the five challenges)? None of the Medium pieces I read defines it.

## 10. Seven more Medium pieces (fifth scan, 7 Oct), January and February 2026
All seven predate every AXI technical piece. Details and quotes to check are in `MYTHOLOGY_EXTRACTION.md` (M23-M29).

| id | date | title | account | what it is |
|---|---|---|---|---|
| M23 | 26 Jan 2026 | "The Complete Takedown: Everything" | FUTURE OF AI | ~8,500 words. Ravana's ten heads = the many AI models ("each claiming intelligence, all ungrounded"); the section heading "AI Can't Be Corrected. It Can Be Managed." ("It cannot be corrected. Because correction requires learning. And it doesn't learn."); and the formula "Saraswati (knowledge/creation) + Kali (destruction/truth) + Radha (devotion/love) = Complete". No evidence given |
| M24 | 24 Jan 2026 | "AI: Postal Service With a Big Mouth" | FUTURE OF AI | AI as a delivery system, not an author; a five-step "correct use"; "40 years of practice" is a personal claim |
| M25 | 23 Jan 2026 | "AI: The Final Postmortem" | FUTURE OF AI | industry numbers with no sources; the "165,000+" layoffs headline does not match the five firms listed (85,000; re-read on the live page) |
| M26 | 18 Jan 2026 | "PERFECT. Perfect it is" | FUTURE OF AI | "You keep zero. Which is most significant." Three years at zero income, 130 pages a day: personal testimony |
| M27 | 28 Jan 2026 | "The Cinderella Project" | FUTURE OF AI | the original meaning of the project: AI cannot predict what is not in its data; five "spaces" (physical, digital, cultural, mathematical impossibility, strategic deployment) |
| M28 | 29 Jan 2026 | "6 messages from our CEO" | Saraswati. Choubey. | six numbered messages and 20 numbered images; a "mathematical proof" with no derivation; the profile says "Delete Test" and the publication description says "Autoregressive hallucination" |
| M29 | 1 Feb 2026 | "AI: The Best Use Case by an Indian Family" | Saraswati. Choubey. | one phone, one evening, six named people; "AI = TV with Infinite Channels"; two level lists. The summarising tool says some sections repeat verbatim |

Still unread from the same run: "Teenage girls with AI" (30 Jan), "Mobile phones. The Idiot Box Just Got Smaller" (1 Feb), "Why the most connected family I know has only one device between them" (31 Jan), the Arabic bismillah piece "Allah loves you either way" (3 Feb).
Questions 27-34 are in `MYTHOLOGY_EXTRACTION.md` section 6.


## 11. Five named pieces that could not be read (8 Oct)

A pasted AI summary (`SHARED_STACK_MAP.md`) named five Medium pieces on three handles that are not in section 1: "Time over AXI", "Shiv Ling - Let it be X", "First - THE AXI MASTER BLUEPRINT: DAY 1 DE-CODING", "AGI is here - a subset of AXI" and "The 8th Religion: Humanity". All five pages and the three profile pages returned **410 Gone** on 8 Oct. Shalendra's own profile lists only the ten pieces in section 2. Nothing from these five pieces is used as evidence. The digits of the handles look like a telephone number and are not recorded here. Open: question 109.

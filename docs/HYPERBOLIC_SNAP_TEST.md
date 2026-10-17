# Does snapping a hidden vector to the nearest allowed token in a hyperbolic ball do anything that masking does not? A hyperbolic-snap test (written BEFORE the run, 17 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before any hidden vector of a language model has been collected for this test. Code: `python/axi/experiments/hypsnap.py`; instrument checks `python/tests/test_hypsnap.py`. Concept-map item 104. Stage 1 needs a GPU (the free Colab T4 is enough) and the model `Qwen/Qwen2.5-1.5B-Instruct` (downloaded in the notebook); stage 2 is numpy on a CPU.

## Where the idea comes from, and how it is treated
On 17 Oct 2026 the owner pasted an AI-written roadmap (no author, no sources given; I have not checked with Rajnish Choubey that anything in it is his, so I attribute none of it to him). Its third section, "Validation of the Continuous-to-Discrete Functor (H^n -> Lat)", says: extract "the final un-normalized hidden activation vectors from an open-weights model just before token generation"; project them into a Euclidean space and "apply standard Softmax thresholding"; project them into the Poincare ball and apply a "hyperbolic Voronoi metric snapping, F(h) = argmin d_B(pi_B(h), p_k)"; then "measure the exact syntax slip rate (like the DEL[2,0:7] error). The empirical data will show that standard Softmax sampling yields a >= 5% syntax failure rate due to token overlapping, whereas the Hyperbolic Functor mathematically guarantees a 0.00% syntax failure rate because the continuous vector is physically trapped inside a discrete Voronoi cell corresponding to an admissible Double Pushout rule."

What I can and cannot say. The roadmap gives no model, no task, no definition of the prototypes p_k, no map pi_B, and no curvature. I had to choose all of them, and **every choice below is mine, not Rajnish's**. Two things follow from the sentence itself, and I state them before the run so that they are not read as findings afterwards. First: a nearest-prototype rule that is restricted to the tokens the grammar allows can only ever output an allowed token. So a 0.00% syntax failure rate **is true by construction** for any such rule, Euclidean, cosine or hyperbolic, and so it is for plain masking (take the allowed token with the highest logit). The ball does not "trap" anything; the restriction to the allowed set does. This test reports that figure for every arm, and it checks the code, not the geometry. Second: the only thing the geometry can change is *which allowed token* is chosen, and that can be measured against the canonical reply. That is what the lines below test.

## What is tested, and what is not
**Tested:** a small open model (`Qwen/Qwen2.5-1.5B-Instruct`) reads a prompt (a service graph and a request: the repository's own prompt) and then the canonical reply, teacher forced, one token at a time. At every position of the reply the program stores the last hidden vector (after the final norm, the input of the output head), the token the model itself would pick over the whole vocabulary, the set of tokens the repository's grammar allows there (the same `TokenDFATable` that the grammar mask uses) and the model's logit for each of them. 400 tasks, two conditions: `full` (the repository's prompt, which describes the syntax) and `bare` (the graph and the request only; no description of the syntax). Then, from the stored arrays alone, nine ways of choosing the token ("arms"; table below) are scored against the canonical token.

**Not tested here:** free generation (the reply is the canonical one, fed in; after a slip a real generation would go somewhere else), a model larger than 1.5B, sampling at a temperature (the model's pick is the greedy one), the roadmap's "Double Pushout rule" (there is no code in the repository that maps hidden vectors to rewrite rules; the allowed token set of the grammar is what stands in for "admissible"), a model trained for the ball, and anything about the "token overlapping" that the roadmap names as the cause of slips.

## The arms
Let h be the stored hidden vector, A the tokens the grammar allows at this position, u_t the row of the output head of token t.
| arm | how it chooses | free numbers |
|---|---|---|
| `free` | the highest logit over the whole vocabulary: the model's own pick (the roadmap's "Euclidean projection and softmax", and the only arm that can break the syntax) | none |
| `masked` | the highest logit among A: what the repository's grammar mask does | none |
| `euclid_nn` | the token of A whose u_t is nearest to h in Euclidean distance; h and the rows u_t are each divided by their median norm (taken over the calibration half) | none |
| `euclid_tuned` | the same with h multiplied by alpha, alpha from {0.25, 0.5, 1, 2, 4}, chosen on the calibration half | 1 |
| `cosine_nn` | the token of A with the highest cosine between h and u_t | none |
| `hyper_nn` | the token of A nearest in the Poincare ball: exponential map at the origin of the (normalised) h and u_t with curvature c, c from {0.01, 0.1, 1, 10, 100} chosen on the calibration half; the geodesic distance | 1 |
| `lin_dot`, `lin_eu`, `lin_hyp` | a linear map of h to 32 numbers and one 32-number prototype for each token of A (started from the map of the rows u_t); the score is a dot product, minus a Euclidean distance, minus a Poincare distance; trained 25 epochs (Adam, learning rate 0.01, batch 256, seed 5, cross entropy over A, analytic gradients, checked in the tests against finite differences) on the calibration half | 32 x D + K x 32 (and one temperature for the two distance heads) |
The nine arms are *the roadmap's two arms and seven controls that I think a fair reading needs*: without `euclid_tuned` and `cosine_nn`, "hyperbolic beats Euclidean" could just mean "a tuned scale beats an untuned one"; without the three trained heads, the question whether the geometry of the *prototypes* matters when they are learned is not asked. The roadmap's comparison is `free` against `hyper_nn`; every other arm is mine.

The calibration half is the even-numbered tasks, the test half the odd-numbered ones (a position where the grammar refuses the canonical token, if any, is left out of both); every figure of an arm is on the test half. A learned head is therefore trained on about 200 replies and tested on another 200.

## What is measured
* **Slip rate of `free`**: the share of replies in which the model's own pick is a token the grammar refuses at some position, and the share of positions at which it is. Both are measured on all 400 replies, in both conditions. This is the roadmap's "syntax failure rate of Softmax" under teacher forcing: the model has been shown the right reply so far, so the figure is the rate of a first slip, not of an accumulated one.
* **Canonical**: the share of test positions at which the arm picks the canonical token. Also reported for the positions where the model's own pick is refused ("on slips") and where it is not ("off slips").
* **Allowed**: the share of test positions at which the arm's token is allowed (100 % for every arm but `free`, by construction).

## What I looked at before locking this page (and what I did not)
I looked at the code of the stage 2 on made-up data: the exponential map, the distance, the gradients of the three heads (finite differences), a planted signal that every arm must find and a signal-free case in which every arm must be at chance, and the fixed arms against a plain loop. **I did not run the model.** I have not seen a hidden vector of Qwen2.5-1.5B-Instruct, nor the logits, nor any figure of how often it breaks the grammar on this prompt, with or without teacher forcing. What I do know that bears on my probabilities: the repository's earlier tests found format slips in the replies of a hosted model (the `DEL[2,0:7]` for `DEL[2,0]` of the Gemini Flash Lite test, 12 % of replies), and a small model without the syntax description in its prompt is, I expect, likely to slip more; I have no figure for this model.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. The first three lines are about the instrument; a "NO" there means the rest cannot be read.

1. [guard] both conditions were collected for all 400 tasks, every position of every reply was walked, and the output-head rows were stored for every token that is allowed anywhere (97 %)
2. [instrument] at every position of both conditions the canonical token is one the grammar allows (the grammar admits the replies the test expects) (99 %)
3. [instrument] every arm except `free` chose a token the grammar allows at 100 % of the test positions of both conditions (true by construction: this checks the code and is not a finding) (99 %)
4. [claim] condition full (the repository's prompt): the model's own pick over the whole vocabulary is a token the grammar refuses in at least 5 % of the replies (35 %)
5. [claim] condition bare (no description of the syntax): the model's own pick is a token the grammar refuses in at least 5 % of the replies (90 %)
6. [claim] condition full: the model's own pick is refused by the grammar at at least 5 % of the positions (25 %)
7. [claim] condition full: `masked` picks the canonical token at at least 95 % of the test positions (75 %)
8. [claim] condition full: `hyper_nn` picks the canonical token at least as often as `masked` minus 2 points (10 %)
9. [claim] condition full: `hyper_nn` beats `euclid_tuned` by at least 2 points of the canonical token (25 %)
10. [claim] condition full, at the test positions where the model's own pick is refused: `masked` picks the canonical token at least as often as `hyper_nn` (85 %)
11. [claim] condition full: `lin_hyp` beats `lin_eu` by at least 1 point (35 %)
12. [claim] condition full: `lin_hyp` beats `lin_dot` by at least 1 point (25 %)
13. [claim] condition full: `lin_dot` is within 2 points of `masked` or above it (50 %)

**Headline** (lines 5 and 8): the roadmap's claim. If both read "yes", the model's own pick breaks the syntax at least one reply in twenty even when the syntax is not described, and the hyperbolic snap picks the canonical token within two points of what masking does: the claim in its fair form is "snapping is as good as masking and the model alone is not enough". My probabilities are 90 % and 10 %; my single number is **8 %**.

Why I think so, line by line. *Lines 1 to 3 (97, 99, 99 %):* the collection walks every reply; the canonical reply is built by the repository's own serializer and the grammar was written for it; the restriction to A makes line 3 true. A "NO" would be a bug. *Line 4 (35 %):* with the syntax in the prompt and the right reply shown so far, a 1.5B instruction model should mostly pick a legal token; I put it a little over a third that at least 5 % of 400 replies contain one illegal pick. *Line 5 (90 %):* without any description of the syntax, a first token of the reply that is chatty (a sentence, a bullet, a code fence) is refused by the grammar, which is a slip on every reply at once. *Line 6 (25 %):* a share of positions is much smaller than a share of replies (a reply is tens of tokens); I put it low. *Line 7 (75 %):* `masked` is the model's own best token among the allowed ones, and the canonical reply is what the model was asked for; I expect it right most of the time with the syntax described. *Line 8 (10 %):* the nearest row of the output head to the hidden vector is not what the model computes (the model takes a dot product, and a norm that differs between tokens changes who is nearest); I expect all of the nearest-prototype arms, in any geometry, to fall clearly below masking. The headline needs this to be wrong. *Line 9 (25 %):* `hyper_nn` against `euclid_tuned`: both are untrained geometries over the same vectors with one free number; the ball has the curvature to bend which are near, but I do not expect two points. *Line 10 (85 %):* on the positions where the model's own pick is refused, the model has ranked the allowed tokens by logit, and masking uses that ranking; the nearest-prototype rules do not. *Lines 11 and 12 (35, 25 %):* a hyperbolic head has room for a tree of tokens (Nickel and Kiela's argument for hierarchies), but the grammar's allowed sets are small flat lists, so I expect no hierarchy to exploit; 1 point is a low bar, and the head sizes are the same. *Line 13 (50 %):* a trained dot-product head starting from the model's own rows should be close to the model's own ranking, which `masked` is.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *Lines 5 and 8 both yes:* the roadmap's claim holds in its fair form on this model and prompt, and I would be wrong by a large margin; the first thing to check is the geometry of the calibration (the median norms) and whether the stored `h_post` is the one that goes through the output head. *Line 5 yes, line 8 NO:* the model alone breaks the syntax and the grammar needs a restriction; the restriction does the work and the ball adds nothing over masking, which is the plain reading of the construction. *Line 5 NO:* the model, even bare, does not break this grammar often enough for a restriction to matter at this size; the premise of the roadmap's figure ("at least 5 %") is not met by this model on this task. *Line 9 yes:* the ball beats a scaled Euclidean snap by more than the tuning of one number explains; this would be the first sign that the geometry matters, and would call for a second model. *Lines 11 and 12 both yes:* learned prototypes in the ball beat learned prototypes in the plane and a dot-product head of the same size; real, and small until a second seed and a second model say otherwise. *Line 13 NO:* even the trained head loses more than two points against the model's own logits, so a head trained on 200 replies cannot replace the output head, in any geometry.

## Reading rules (fixed now; the printout prints exactly these)
The thirteen lines above are the reading rules; `hypsnap.verdict_hypsnap` computes them from the saved result. Thresholds, as in the code: share of replies with a slip 5 %; share of positions with a slip 5 %; `masked` 95 %; the slack of lines 8 and 13 2 points; the gain of line 9 2 points, of lines 11 and 12 1 point. Line 10 needs at least one test position at which the model's own pick is refused; if there is none it reads NO. A rule about a condition the run does not hold reads "not run"; the real run holds both.

## What this cannot show
One model of 1.5 billion parameters, one task family (synthetic service graphs), one grammar, two prompts, teacher forcing, greedy picks. The model may be better or worse than a larger one at keeping to the syntax; the slip rate of a larger model needs a larger run. The prototypes are the rows of the model's own output head and a head I trained on 200 replies, not prototypes designed for the ball, and the map into the ball is the exponential map at the origin with one curvature, the simplest choice; a different map, a trained curvature per token, or a model trained with the ball in the loop could change the outcome, and this test does not say it would not. Nothing about "Dirichlet-Voronoi cells", "Double Pushout rules" or "token overlapping" is tested: the stand-in for an admissible rule is the grammar's allowed set. The sentence "the hyperbolic functor mathematically guarantees 0.00%" is not tested, because it needs no test: it is a property of restricting the choice to the allowed set, which `masked` has as well.

## Related work (from memory, not searched; check before citing)
Poincare embeddings for hierarchies (Nickel and Kiela, 2017); hyperbolic neural networks and the exponential map at the origin (Ganea, Becigneul and Hofmann, 2018); hyperbolic prototype learning and hyperbolic image classification heads; constrained decoding with a grammar mask or a finite-state machine over the vocabulary (Willard and Louf, 2023 "Efficient guided generation"; the grammar-constrained decoding of Geng et al., 2023; and others), which is the family `masked` belongs to. If these hold as I remember them, constrained decoding already gives a 0 % grammar failure rate by restriction, and the open question for a hyperbolic head is only the one measured here: whether it picks the right allowed token better than the model's own logits do.

## How it is run
Stage 1 (GPU): the notebook `python/colab/axi_hyperbolic_snap.ipynb` loads the model, collects both conditions (about 400 replies each) and saves two `.npz` files to Drive. Stage 2 (CPU, a few minutes): `python -m axi.experiments.hypsnap --full axi_hs_full.npz --bare axi_hs_bare.npz --out axi_hypsnap.json`. The result file is JSON; send it back and this page gets a "Result" section without changing anything above this line. A smoke cell in the notebook checks on five prompts that the stored shapes are right and that the canonical token is allowed before the long collection starts; if that cell fails nothing is collected.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_hypsnap.py` and again by `tests/test_hypsnap_result.py`). The run was made by the owner on 17 Oct 2026: stage 1 (the model, on a free Colab T4, notebook `python/colab/axi_hyperbolic_snap.ipynb`), then stage 2 (`python -m axi.experiments.hypsnap`, numpy, on the CPU of the same Colab) in 2,657.8 seconds; the library versions of that machine were not recorded. The result file is `docs/results/axi_hypsnap.json` (sha256 `3010c7a13944a3a2c9b6c05f87d9536a850204bf3891f31d354c2a87164329c5`). Every figure below is a count of the 4,270 test positions of a condition (the odd-numbered replies), or of all 9,795 positions where it says so; there is one run and one seed of the heads, so no range is shown. Differences are taken from the counts, so they can differ by 0.1 from the difference of the rounded figures shown.*

*Two things about the run itself, stated so that nobody has to wonder. (1) The first full run of the collection cell failed after the collection was done and saved: `unembed` in `torch_runner` called `.numpy()` on a tensor that requires grad. The owner ran a cell that redefines `unembed` with `.detach()` and `torch.no_grad()`, and ran the collection cell again, which loaded the saved arrays. The same fix is now in `hypsnap.py` (it touches only the copy of the output-head rows; no figure depends on it; the first hardware run of that function found it, which the stand-in tests could not). (2) Stage 2 took 44 minutes, not the "few minutes" the page and the notebook said: the grammar allows 32,837 different tokens somewhere, and every nearest-token rule compares each hidden vector with all of those rows; my timing on made-up data had used 300.*

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 97 % | both conditions: 400 and 400 replies, 9,795 and 9,795 positions, 0 and 0 broken replies; the head rows of 32,837 tokens stored | yes |
| 2 | 99 % | the canonical token was allowed at every position of both conditions (full 9,795, bare 9,795) | yes |
| 3 | 99 % | every arm but `free` chose an allowed token at 4,270 of 4,270 test positions in full and 4,270 of 4,270 in bare (eight arms, two conditions) | yes |
| 4 | 35 % | the model's own pick was refused in 63 of 400 replies (15.8 %; limit 5 %) | yes |
| 5 | 90 % | the model's own pick was refused in 400 of 400 replies (100.0 %; limit 5 %) | yes |
| 6 | 25 % | refused at 63 of 9,795 positions (0.6 %; limit 5 %) | NO |
| 7 | 75 % | `masked` 4,005 of 4,270 (93.8 %) (limit 95 %) | NO |
| 8 | 10 % | `hyper_nn` 3,558 of 4,270 (83.3 %) against `masked` 93.8 %: 10.5 points below (it had to be no more than 2 below) | NO |
| 9 | 25 % | `hyper_nn` 83.3 % against `euclid_tuned` 92.4 %: -9.1 points (it had to be at least +2) | NO |
| 10 | 85 % | on the 23 test positions where the model's own pick is refused: `masked` 23, `hyper_nn` 23 right (a tie) | yes |
| 11 | 35 % | `lin_hyp` 95.5 % against `lin_eu` 94.8 %: +0.7 points (28 positions; limit +1) | NO |
| 12 | 25 % | `lin_hyp` 95.5 % against `lin_dot` 94.6 %: +0.9 points (38 positions; limit +1) | NO |
| 13 | 50 % | `lin_dot` 94.6 % against `masked` 93.8 %: +0.8 points (it had to be at least -2) | yes |

**7 of the 13 lines read yes; lines 6, 7, 8, 9, 11 and 12 read NO. Of the two headline lines, 5 reads yes and 8 reads NO, so the headline did not come out.** The page said in advance what this combination means: *"Line 5 yes, line 8 NO: the model alone breaks the syntax and the grammar needs a restriction; the restriction does the work and the ball adds nothing over masking, which is the plain reading of the construction."* That is what came out. My probabilities were 90 % for line 5 and 10 % for line 8 (yes), so this outcome was the one I put at about 81 %: a confirmation of what I wrote down. The three first lines are the instrument and all read yes; line 3 is true by construction, as the page said, and checks the code only.

## Where my hand numbers were off
* **Line 4 (35 %) read yes.** In the full prompt the model's own pick is refused at 63 of 9,795 positions (0.6 %), which is rare, but a reply has about 24.5 positions, so 63 of 400 replies (15.8 %) hold at least one. I had put the chance that the reply share stays under 5 % at 65 %. Line 6 (25 %), the share of positions, read NO as I thought.
* **Line 7 (75 %) read NO by 1.2 points.** `masked` gets 93.8 % of the canonical tokens in the full prompt, short of the 95 % I asked. I had written "right most of the time"; it is, but not that much of the time. The model's own pick is the canonical token at 92.6 % of all positions.
* **Lines 8 and 9: the size of the miss.** I had written "clearly below masking" and put 25 % on a 2-point gain over a tuned flat rule. `hyper_nn` is 10.5 points below `masked` and 9.1 points *below* `euclid_tuned`.
* **Lines 11 and 12 (35 and 25 %): near misses, in the direction of the roadmap's idea.** `lin_hyp` is ahead of `lin_eu` by 0.7 points (28 positions of 4,270) and of `lin_dot` by 0.9 points (38 positions), under my bar of 1 point by 0.3 and 0.1. Whether that is more than noise I cannot say from this file: the per-position picks were not stored, and there is one seed of the heads. In the bare condition (not one of the 13 lines) the order is the other way: `lin_dot` 94.5 %, `lin_hyp` 92.8 %.
* **Line 10 (85 %) reads yes by a tie.** The test half of the full prompt holds only 23 positions where the model's own pick is refused, and every restricted arm gets all of them right; so many positions cannot separate the arms. The same comparison has power in the bare condition (not one of the 13 lines), with 512 such positions: `masked` gets 272 of them right (53.1 %), `hyper_nn` 266 (52.0 %).

## The tables
**Full prompt** (test half: 4,270 positions; "on slips" are the positions where the model's own pick is refused, "off slips" the others):

| arm | canonical | share | allowed | on slips | off slips |
|---|---|---|---|---|---|
| `free` | 3,982 | 93.3 % | 99.5 % | 0 of 23 (0.0 %) | 93.8 % |
| `masked` | 4,005 | 93.8 % | 100.0 % | 23 of 23 (100.0 %) | 93.8 % |
| `euclid_nn` | 2,233 | 52.3 % | 100.0 % | 23 of 23 (100.0 %) | 52.0 % |
| `euclid_tuned` | 3,945 | 92.4 % | 100.0 % | 23 of 23 (100.0 %) | 92.3 % |
| `cosine_nn` | 3,951 | 92.5 % | 100.0 % | 23 of 23 (100.0 %) | 92.5 % |
| `hyper_nn` | 3,558 | 83.3 % | 100.0 % | 23 of 23 (100.0 %) | 83.2 % |
| `lin_dot` | 4,038 | 94.6 % | 100.0 % | 23 of 23 (100.0 %) | 94.5 % |
| `lin_eu` | 4,048 | 94.8 % | 100.0 % | 23 of 23 (100.0 %) | 94.8 % |
| `lin_hyp` | 4,076 | 95.5 % | 100.0 % | 23 of 23 (100.0 %) | 95.4 % |

**Bare** (no description of the syntax in the prompt; test half: 4,270 positions, 512 of them slips):

| arm | canonical | share | allowed | on slips | off slips |
|---|---|---|---|---|---|
| `free` | 3,432 | 80.4 % | 88.0 % | 0 of 512 (0.0 %) | 91.3 % |
| `masked` | 3,704 | 86.7 % | 100.0 % | 272 of 512 (53.1 %) | 91.3 % |
| `euclid_nn` | 1,694 | 39.7 % | 100.0 % | 81 of 512 (15.8 %) | 42.9 % |
| `euclid_tuned` | 3,260 | 76.3 % | 100.0 % | 140 of 512 (27.3 %) | 83.0 % |
| `cosine_nn` | 3,565 | 83.5 % | 100.0 % | 266 of 512 (52.0 %) | 87.8 % |
| `hyper_nn` | 3,168 | 74.2 % | 100.0 % | 266 of 512 (52.0 %) | 77.2 % |
| `lin_dot` | 4,035 | 94.5 % | 100.0 % | 450 of 512 (87.9 %) | 95.4 % |
| `lin_eu` | 3,957 | 92.7 % | 100.0 % | 410 of 512 (80.1 %) | 94.4 % |
| `lin_hyp` | 3,961 | 92.8 % | 100.0 % | 415 of 512 (81.1 %) | 94.4 % |

**The calibration** (accuracy on the calibration half for each value in the grid I locked; the chosen value is the last column):

| condition, rule | 1st value | 2nd | 3rd | 4th | 5th | chosen |
|---|---|---|---|---|---|---|
| full, `euclid_tuned` scale | 28.9 % | 36.0 % | 47.4 % | 75.4 % | 90.4 % | 4 |
| full, `hyper_nn` curvature | 46.9 % | 42.2 % | 37.0 % | 28.8 % | 80.7 % | 100 |
| bare, `euclid_tuned` scale | 28.8 % | 31.2 % | 39.3 % | 57.1 % | 76.3 % | 4 |
| bare, `hyper_nn` curvature | 39.1 % | 38.0 % | 32.5 % | 28.8 % | 73.0 % | 100 |

(The scale grid is 0.25, 0.5, 1, 2, 4; the curvature grid 0.01, 0.1, 1, 10, 100. Median norm of the hidden vectors 144.5, of the head rows 1.10.)

**The three trained heads** (loss on the calibration half, epoch 1 and epoch 25; the temperature of the two distance heads is exp(logtau); `lin_dot` has none):

| condition | head | first loss | last loss | logtau |
|---|---|---|---|---|
| full | `lin_dot` | 1.519 | 0.102 | 0.00 |
| full | `lin_eu` | 1.282 | 0.094 | 0.45 |
| full | `lin_hyp` | 1.893 | 0.092 | 1.00 |
| bare | `lin_dot` | 1.569 | 0.111 | 0.00 |
| bare | `lin_eu` | 1.334 | 0.098 | 0.53 |
| bare | `lin_hyp` | 1.952 | 0.097 | 1.02 |

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The roadmap's comparison, as I built it, has one half that holds and one that does not.** The model's own pick does break the grammar: in every one of the 400 bare replies (9.8 % of positions), and in 15.8 % of the replies of the full prompt. The hyperbolic snap does not match masking: 83.3 % against 93.8 % in the full prompt, 74.2 % against 86.7 % in bare. Against the model's own pick, the roadmap's actual comparison, the snap is 9.9 points worse in the full prompt (`free` 93.3 %) and 6.2 points worse in bare (`free` 80.4 %).
2. **The restriction does real work where the syntax is not described.** In bare the model's own pick is refused at 961 of the 9,795 positions and is an allowed token at only 88.0 % of the test positions; masking lifts the canonical share from 80.4 % to 86.7 % (+6.4 points) and recovers 272 of the 512 slip positions (53.1 %). In the full prompt it adds 0.5 points (23 positions: all the slips). Plain masking is what the repository already does.
3. **"0.00 %" is 100 % allowed for all eight restricted arms in both conditions**, ball or no ball: the table's "allowed" column. The model alone is allowed at 99.5 % (full) and 88.0 % (bare).
4. **The nearest-token rules sit below masking, and the flat one climbs toward it.** Untuned `euclid_nn` gets only 52.3 % (the rows and the vectors have different norms, so "nearest" is not what the model computes); with a scale it gets 92.4 %, cosine 92.5 %, the ball 83.3 %; `masked`, which is the model's own dot product among the allowed tokens, gets 93.8 %. The chosen scale is the largest in my grid (accuracy on the calibration half rises with it: 90.4 % at 4), and a reason is plain algebra: a nearest-token rule ranks by 2·alpha·(h . u) minus the squared length of u, so as alpha grows the ranking turns into h . u, the model's own ranking, which is `masked`. So the tuned flat rule is approaching masking, not beating it. **The chosen curvature is also the largest in my grid** (80.7 % at 100 on the calibration half; the next best value, 0.01, gets 46.9 %), so `hyper_nn` may be under-tuned: a wider grid could raise it, and I cannot say by how much. The page locked this grid; a wider one is a follow-up and would have to be labelled exploratory.
5. **The three trained heads end within one point of each other in the full prompt, and above masking by 0.8 to 1.7 points** (`lin_dot` 94.6 %, `lin_eu` 94.8 %, `lin_hyp` 95.5 %; masking 93.8 %). In bare they are 6 to 8 points above masking (`lin_dot` 94.5 %, `lin_eu` 92.7 %, `lin_hyp` 92.8 %, masking 86.7 %) and on the slip positions they are right 80 to 88 % of the time (`lin_dot` 87.9 %, `lin_eu` 80.1 %, `lin_hyp` 81.1 %) against 53.1 % for masking. So a small head trained on about 200 replies recovers the canonical token much more often than the model's own ranking does when the prompt does not describe the syntax, **and the geometry-free dot head is the best of the three there.** What it learned is, most likely, the format of the canonical replies, which are formulaic; it is a supervised classifier on that format, a different kind of tool from a mask, and **AXI never trains**, so it is not something the repository would use. Nothing in these figures points at the ball as the reason for the gain.
6. **What the test does not give the roadmap.** No arm is helped by the ball in this run: untrained, it loses to the flat rule and to cosine; trained, it is ahead of the plane and the dot head by under a point in the full prompt (below the bar I set) and, in bare, level with the flat head and 1.7 points behind the dot head.

## What this does and does not show about the roadmap's third test
* **"The Hyperbolic Functor mathematically guarantees a 0.00 % syntax failure rate."** As the page said before the run, this is a property of restricting the choice to the allowed set, which every arm but `free` has; the table shows 100 % allowed for the flat rules, cosine, the ball, masking and the heads alike. It is not a property of the geometry.
* **"Softmax sampling yields a syntax failure rate of at least 5 %."** Met here for replies in both conditions under teacher forcing (15.8 % and 100.0 %), not for positions in the full prompt (0.6 %); one 1.5B model, greedy picks, the canonical reply fed in.
* **Not done and worth doing:** a wider grid for the scale and the curvature; other maps into the ball (a trained curvature, a per-token one); a second seed of the three heads (to say whether lines 11 and 12 are noise); a second and a larger model; free generation, where a slip changes what follows; a head trained for the ball from the start. None of these is implied by this result, either way.

## Reproduce
Stage 1 needs the GPU and the model (the notebook says 10 to 20 minutes on a T4 plus the download of about 3 GB; that is my estimate, the owner's time for it was not recorded); its two arrays (`axi_hs_full.npz`, `axi_hs_bare.npz`, about 150 MB each) are not in the repository. Stage 2 from those arrays: `python -m axi.experiments.hypsnap --full axi_hs_full.npz --bare axi_hs_bare.npz --out axi_hypsnap.json`, about 45 minutes on the Colab CPU. `tests/test_hypsnap_result.py` recomputes every figure of this Result from the saved result file.

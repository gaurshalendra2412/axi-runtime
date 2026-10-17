# Does a pair of Betti numbers of the recent history carry what the last 50 commands carry? A history-shape test (written BEFORE the run, 17 Oct 2026)

This page fixes the question, the design, the measures, the predictions and the reading rules before any rate from this test at full size exists. Code: `python/axi/experiments/histshape.py`; instrument checks `python/tests/test_histshape.py`. Concept-map item 105. It needs no GPU, no language model and no internet: numpy and scikit-learn on a CPU (Codespaces or a free Colab are enough), about ten minutes.

## Where the idea comes from, and how it is treated
On 17 Oct 2026 the owner pasted an AI-written roadmap (no author, no sources given; I have not checked with Rajnish Choubey that anything in it is his, so I attribute none of it to him). Its fourth section, "Validating Topological Cobordisms (The Time Invariant)", says: run an agent through a 10,000-step environment; Arm A gets "a sliding window of the last 50 text commands"; Arm B gets "only the current state and the Betti Numbers (beta0, beta1) representing the topological cobordism of its recent history". The test "will empirically measure whether the highly compressed topological invariants (beta0, beta1) allow the model to recognize cyclic behaviors and temporal momentum just as effectively as reading 50,000 tokens of chat history. If Arm B matches Arm A's task accuracy, you have proven that chronological intelligence is a geometric property, not a sequence-retention property."

What I can and cannot say. The roadmap gives no task, no definition of the complex whose Betti numbers are meant, and no accuracy to match. I had to choose all three, and **every choice below is mine, not Rajnish's**. "Cobordism" is not tested: no code in the repository builds one (I searched `python/axi` for "betti", "homolog" and "cobord" before writing the code of this test, and found nothing). What is tested is whether two integers computed from the graph of the cells the agent visited in the last 50 steps tell a predictor as much as the 50 commands themselves. The sentence "chronological intelligence is a geometric property" is a claim about intelligence that no accuracy on this test could prove; what a result can show is how much of two particular answers is in each summary of the past.

## What is tested, and what is not
**Tested (CPU, no language model):** an agent walks a 7 x 7 grid for 10,000 steps under a hidden mode (it goes round a loop, it keeps its direction, or it moves at random). At every step a predictor sees one summary of the past and is asked which mode the agent is in (MODE) and which command comes next (NEXT). The summaries are the roadmap's two arms and several controls (below). Each trained summary gets the same learner, so the differences are differences in what the summary holds.

**Not tested here:** a language model reading a chat history (the predictor is a gradient-boosted tree model, not a model that reads text), 50,000 tokens (50 bare commands are a few hundred tokens at most, see "What this cannot show"), cobordisms, persistent homology, Betti numbers of higher degree (the complex here is a graph, so there are beta0 and beta1 only), a real environment, or an agent that acts on what it predicts.

## The world
A grid of 7 x 7 cells with walls at the edge. Five commands: U, D, L, R (a move) and S (stay). A walk has 10,000 steps. It is cut into segments of 40 plus an exponential draw of mean 110 steps (about 150 on average); each segment has a mode, drawn at random among the two modes other than the segment before it, so every switch changes the mode.
* **tour**: at the start of the segment the walker takes a rectangle of 2 to 6 cells each way that has its cell on its perimeter (4 to 20 cells round), picks a direction and goes round it. At every step there is a chance 0.05 that the move fails: the command is logged, the cell does not change, and the walker tries the same move again next step.
* **drift**: the walker starts in a random direction and keeps its last command with chance 0.85, else picks any of the four moves; if its last move was into a wall (it stayed) its next command is the opposite one. This is the "momentum".
* **random**: any of the four moves with equal chance; a move into a wall leaves the walker where it is.
The log holds the command issued (also when it failed or hit a wall). The label of step t, `mode`, is the mode that chose command t.

## The views (the arms)
All views contain the current cell (its two coordinates). Besides it:
| view | what it adds | what it is for |
|---|---|---|
| `state` | nothing | the control: what the cell alone says |
| `B_betti` | beta0 and beta1 of the graph of the last 50 moves | **the roadmap's Arm B** |
| `B_betti3` | beta1 over the last 10, 25 and 50 moves | a more generous B (my extension) |
| `C_counts` | the number of distinct cells and of distinct links in the last 50 moves | beta1 is links - cells + beta0, so this holds the same news as `B_betti` in different words |
| `C_hist` | how often each of the five commands occurs in the last 50 | a five-number summary of the same length that is not topological |
| `D_short` | the last three commands | what a very short memory holds |
| `A_window` | the last 50 commands, one column each | **the roadmap's Arm A** |
| `A_eng` | 26 numbers from the last 50 commands: for k = 1 to 25 how often a command equals the one k steps earlier, and how often the command changes | the window read by someone who looks for repeats |
| `A_all` | the 50 commands and the 26 numbers | the strongest reader of the window that I have |
| `A_match` | (NEXT only, nothing is trained) the longest earlier repeat of the last commands inside the window (8 commands down to 2), and the command that followed the most recent such repeat; else the last command | a direct reader of the window for prediction |
The graph of the last 50 moves: its points are the cells the walker was in (51 states, from 50 steps ago to now), its links the distinct pairs of cells between which it moved (a failed move or a stay adds none). A walk is connected, so **beta0 is 1 on every window and carries no news at all**; beta1, the number of independent loops, is links - points + 1. So `B_betti` is, in effect, the cell and one small integer. That follows from the definition and is stated here before the run; it is the reason for `B_betti3`.

Every trained view uses one scikit-learn `HistGradientBoostingClassifier` per question with the fixed settings of `histshape.HGB` (at most 200 rounds, learning rate 0.1, 31 leaves, early stopping on a tenth of the training rows, the command columns of `A_window` and `D_short` read as categories). The learner is the same for every view and was not tuned. A gradient-boosted tree model reads 50 columns of commands less well than a model built to find repeats, which is why `A_eng`, `A_all` and `A_match` exist: the strongest reader of the window is the best of those, and the rules compare B with that best, not with `A_window` alone.

## What is measured
* **MODE**: accuracy of naming the mode of the step. **TOUR F1**: the F1 of the answer "tour" against all other modes (the roadmap's "recognize cyclic behaviours"). **NEXT**: accuracy of naming the command. The majority class of the training rows is reported as `majority`. NEXT is also reported by the true mode, since the modes differ a great deal in how predictable the next command is.
* The first 50 steps of each walk are not scored (the window is not full). A step in the first dozens of steps after a switch has a window that is mostly the earlier mode; this limits every view, and is the same for all of them.
* **Token cost**, as a fact of the design and not a measurement: a view holding 50 commands written one per word is of the order of 100 to 200 tokens, one holding the cell and two integers is of the order of 10.

## The experiment
Three seeds (1, 2, 3). In each: 20 walks of 10,000 steps (seeds `hist/<seed>/<i>`), the first 14 for training (139,300 scored steps) and the last 6 for testing (59,700 scored steps), so the test walks are walks the learners have not seen. Every view is trained on the training rows and scored on the test rows, for MODE and for NEXT. The reported figure of a view is the mean over the three seeds of its test accuracy; the per-seed figures are in the file. A digest of all the walks (the commands, the cells and the modes) is printed and should be the same on any machine.

## What I looked at before locking this page (and what I did not)
I looked at: (a) one walk (seed `hist/1/0`): the mode counts (3,551 steps of tour, 3,219 of drift, 3,230 of random in 74 segments), the first tour segment (a rectangle of 4 x 5 cells gone round, with failed moves repeated), and the share of steps on which the cell did not change (tour 4.9 %, drift 13.1 %, random 14.1 %: walls), to see that the walker does what the text says; (b) the time of the fits (a few seconds on 40,000 rows; no accuracy printed); (c) **a `--quick` run, to see that the program works: 3 + 2 walks of 600 steps, 1,650 training rows, which printed the accuracies of all views. I did see them** (MODE between 24 % and 64 %, NEXT between 22 % and 58 %); the sample is far too small to read (the two test walks hold about 1,100 steps), guard line 1 read NO in it simply because 600 steps rarely hold all three modes, and I did not change a setting of the world or of the learner because of them. I did change one bar: line 5 asked for 0.85 in my first draft of the code and asks for 0.80 now, because while writing the lines below I worked out by hand that about a third of all steps have a window that contains a switch, which caps every view; I cannot claim that having seen a best figure of 64 % played no part. I did **not** look at beta1 by mode, nor at any accuracy at the size of the experiment.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. The first four lines are about the instrument; a "NO" there means the rest cannot be read.

1. [guard] every walk ran all its steps, every walk held all three modes at least once, and the number of scored steps is walks x (steps - 50) (98 %)
2. [guard] beta0 was 1 on every scored step (a run of consecutive cells is connected: this checks the Betti code and is not a finding) (99 %)
3. [instrument] MODE: the current cell alone is within 12 points of the majority rate (the cell does not give the mode away) (70 %)
4. [instrument] the loops have teeth: the mean of beta1 in random mode is at least 1.0 above its mean in tour mode (93 %)
5. [claim] MODE: the best of the three readers of the last 50 commands (A_window, A_eng, A_all) reaches at least 0.80 (80 %)
6. [claim] MODE: B_betti beats the current cell alone by at least 10 points (90 %)
7. [claim] MODE: B_betti is within 5 points of the best reader of the last 50 commands (it matches) (30 %)
8. [claim] TOUR (is the agent going round a loop?), F1: B_betti reaches at least 0.9 of the best reader's F1 (45 %)
9. [claim] NEXT: the best of the readers of the last 50 commands (A_window, A_eng, A_all, A_match) beats B_betti by at least 10 points (95 %)
10. [claim] NEXT: B_betti is at most 2 points above the current cell alone (the pair adds almost nothing to prediction) (55 %)
11. [claim] MODE: B_betti beats C_hist (the five command counts, a summary that is not topological) (40 %)
12. [claim] MODE: B_betti3 (beta1 over three scales) beats B_betti by at least 3 points (65 %)
13. [claim] NEXT: B_betti is within 5 points of the best reader of the last 50 commands (it matches) (2 %)

**Headline** (lines 7, 8 and 13): the roadmap's claim. If all three read "yes", the two integers match the last 50 commands on both questions in this world: MODE within 5 points, the loop detector within a tenth, and NEXT within 5 points. My probabilities are 30 %, 45 % and 2 %; they are positively correlated, and my single number is **1 %**.

Why I think so, line by line. *Lines 1 and 2 (98, 99 %):* the walker is checked by eye on one walk, and beta0 is 1 by the definition of the graph; a `NO` would be a bug. *Line 3 (70 %):* the tour sits on rectangles anywhere, the drift bounces off walls and the random walker stays nearer the middle, so the cell says something about the mode, but I expect it to say little. *Line 4 (93 %):* a random walk of 50 steps on 49 cells goes back over its path and closes many small loops, a tour closes exactly one, and a drift along a line none; I have not looked at the numbers. *Line 5 (80 %):* the commands of a tour repeat with the length of the loop, the commands of a drift repeat the last one 85 % of the time, those of a random walk do not repeat; a reader that looks for repeats should do well except at a switch (a third of the steps have a window that contains one). *Line 6 (90 %):* beta1 separates random (many loops) from the others at least, and the cell adds the rest. *Line 7 (30 %):* tour (one loop), drift (none or few) and random (many) have different beta1, so I expect `B_betti` to do much better than the cell alone, but a loop that is gone round once and one that is closed by a drift that turns look alike, and a window that straddles a switch confuses both; I expect it to fall short of the best reader by more than 5 points. *Line 8 (45 %):* the loop detector is the question beta1 is best at, and it is the one the roadmap names; the F1 of "tour" against all other modes is more forgiving than the three-way accuracy. *Line 9 (95 %):* the next command depends on direction and phase, which a count of loops does not hold: in a tour it is given by the last command and the cell, in a drift it is the last command with chance 0.85; `A_match` and `D_short` see the last commands, `B_betti` does not. *Line 10 (55 %):* knowing the mode (through beta1) without knowing the last direction helps little for the next command; at a corner of a rectangle the cell narrows it to two moves, which the cell alone already says. *Line 11 (40 %):* the five command counts say a lot about the mode (a tour is balanced, a drift is not, a random walk is flat), perhaps as much as beta1; I put it a little below even. *Line 12 (65 %):* three scales of beta1 tell a loop that is closing from one that has closed, which should help. *Line 13 (2 %):* the reason for line 9.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *Lines 7, 8 and 13 all yes:* the roadmap's claim holds in this world, and I would be wrong by a large margin; the first thing to check would be the definition of the graph and the leak between the training and test walks. *Line 8 yes and line 7 NO:* the pair detects loops, which is what a "cyclic behaviour" is, and does not name the mode; this would be a real but narrow result. *Line 9 yes:* the summary does not hold the phase of the cycle or the direction, which is what a short or long window holds and what prediction needs; "temporal momentum", as the roadmap calls it, is not in the two integers. *Line 10 NO:* the pair adds to prediction through the mode. *Line 11 NO:* a non-topological five-number summary of the same size does as well or better, and the topology is not what carries the information. *Line 12 yes:* more scales help; the roadmap's pair is the weakest form of its own idea. *Line 4 NO:* the instrument has no signal in beta1, and lines 6 to 8 and 12 mean nothing.

## Reading rules (fixed now; the printout prints exactly these)
The thirteen lines above are the reading rules; `histshape.verdict_histshape` computes them from the saved result. Every figure is the mean over the three seeds of the test accuracy. "The best reader of the last 50 commands" is the largest of the means of `A_window`, `A_eng` and `A_all` for MODE and for TOUR F1, and of those and `A_match` for NEXT. Thresholds, as in the code: line 3: 12 points above the share of the commonest mode in the test rows; line 4: 1.0; line 5: 0.80; line 6: 10 points; line 7: 5 points; line 8: 0.9; line 9: 10 points; line 10: 2 points; line 12: 3 points; line 13: 5 points. A rule about a view the run does not hold reads "not run"; the real run holds every view.

## What this cannot show
The agent is a walker I wrote, the modes are three simple generators, and the numbers are about this instrument; the predictors are tree models. Nothing is measured on a language model reading a history; a model that reads text might use 50 commands better or worse than a tree model does, and might use two integers in a different way. Fifty bare commands are cheap: the roadmap's "50,000 tokens of chat history" would need about a thousand tokens per command, which is a different regime (long, wordy commands), and the cost saving of Arm B is not tested. The complex is a graph of the cells visited; a different complex (cells filled in, the commands themselves as points, a different window) would give different Betti numbers, and the roadmap's phrase "topological cobordism" does not tell which; `B_betti3` is my one step in that direction. The modes are my choice, as are their parameters (the loop's slip 0.05, the drift's 0.85, the segment length); the three modes also differ strongly in their loops by design, which favours B on MODE; the question of NEXT is the one on which B has no help from the design. The learner is one tree model; a neural network on the 50 commands could do better than `A_window`, which the strongest-reader rule is meant to cover but cannot guarantee. A switch cuts every view the same way and caps the MODE figures. The test walks are new walks, but they come from the same generator as the training walks.

## Related work (from memory, not searched; check before citing)
Delay embeddings of a signal (Takens, 1981) and the persistent homology of sliding windows as a detector of periodicity (Perea and Harer, 2015, "Sliding windows and persistence"); recurrence plots (Eckmann, Kamphorst and Ruelle, 1987). The older question "which summary of the past is enough to predict the future" is the one of sufficient statistics of the history: predictive state representations (Littman, Sutton and Singh, 2002), belief states of partially observed decision processes (Kaelbling, Littman and Cassandra, 1998), causal states in computational mechanics (Crutchfield and Young, 1989). If these hold as I remember them, a topological summary of a window is one candidate for such a statistic, and the test measures how much of two particular answers one candidate holds, not whether topology is the right language.

## How it is run
In `python/`: `python -m axi.experiments.histshape --out axi_histshape.json` (about ten minutes on two CPUs). `--quick` runs a few seconds of a small sample, to see that it works, not to read. It needs numpy and scikit-learn (both are in Colab and in Codespaces after `pip install scikit-learn`). The digest printed at the end should be the same on any machine; the accuracies can differ in the last digits between machines (the trees use floating point), the rules should not move. A Colab notebook, `python/colab/axi_history_shape.ipynb`, runs it from a zip with copy-paste cells. The result file is JSON; send it back and this page gets a "Result" section without changing anything above this line.

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_histshape.py` and again by `tests/test_histshape_result.py`). The run was made on 17 Oct 2026 in my workspace, not on the owner's machine: `python -m axi.experiments.histshape --out axi_histshape.json`, 977.1 seconds on two CPUs, Python 3.13.16, numpy 2.5.3, scikit-learn 1.9.1. The result file is `docs/results/axi_histshape.json` (sha256 `ebfa27391bddc3706fb5c1f7f29586c38e12c989075713d6a135b7bd8e8e1f17`; digest of the 60 walks `c30005542a6567da`). Every figure below is the mean over the three seeds of the test accuracy; the range over the seeds is in the first table. Differences are taken from the unrounded means, so they can differ by 0.1 from the difference of the rounded figures shown.*

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 98 % | 3 seeds x 20 walks of 10,000 steps, all three modes in every walk; 139,300 training and 59,700 test steps in each seed (= walks x (10,000 - 50)) | yes |
| 2 | 99 % | beta0 was 1 on every scored step of every walk (smallest 1, largest 1) | yes |
| 3 | 70 % | the cell alone 36.4 %, the majority rate 30.3 %: 6.1 points above (limit 12) | yes |
| 4 | 93 % | mean beta1: random 3.16, tour 1.41, drift 1.00; random minus tour 1.74 (limit 1.0) | yes |
| 5 | 80 % | best reader of the window 93.9 % (A_window; limit 80 %) | yes |
| 6 | 90 % | B_betti 68.6 % against the cell alone 36.4 %: 32.2 points above (limit 10) | yes |
| 7 | 30 % | B_betti 68.6 % against the best reader 93.9 %: 25.4 points short (limit 5) | NO |
| 8 | 45 % | F1 of B_betti 0.698 against 0.940 for the best reader: 0.74 of it (limit 0.90) | NO |
| 9 | 95 % | best reader of the window 65.4 % (A_window) against B_betti 28.6 %: 36.7 points above (limit 10) | yes |
| 10 | 55 % | B_betti 28.6 % against the cell alone 28.1 %: +0.6 points (limit 2) | yes |
| 11 | 40 % | B_betti 68.6 % against C_hist 56.3 %: +12.3 points | yes |
| 12 | 65 % | B_betti3 77.7 % against B_betti 68.6 %: +9.2 points (limit 3) | yes |
| 13 | 2 % | B_betti 28.6 % against the best reader 65.4 %: 36.7 points short (limit 5) | NO |

**10 of the 13 lines read yes; lines 7, 8 and 13, the three of the headline, read NO.** The headline did not come out: on this world, with these definitions, the pair (beta0, beta1) of the last 50 steps does not match the last 50 commands on either question. It falls short by 25.4 points on MODE and by 36.7 points on NEXT. My probability for the three together was 1 %; for lines 7, 8 and 13 separately it was 30, 45 and 2 %, so the outcome is the one I put at about 99 %. That makes it a confirmation of what I wrote down, not a surprise.

**The four first lines are the instrument, and all four read yes.** Line 2 (beta0 is 1 on every step) is true by the definition of the graph, as the page said. Line 4 says that the loop count has teeth: mean beta1 is 3.16 in random mode, 1.41 in tour mode, 1.00 in drift mode, so a gap of 1.74 between random and tour. A NO there would have emptied the test.

## Where my hand numbers were off
* **Drift is not loop-free (line 4's reason).** I wrote that a drift "along a line" closes none; the mean of beta1 in drift mode is 1.00, about one loop in a window of 50 steps, where I expected about zero. Tour is 1.41 against the one loop I wrote down (windows that straddle a switch, and failed moves, change it). The drift is a momentum walker that turns at walls and, 15 % of the time, picks a new direction; in a 7 x 7 grid that closes small loops often. I did not look at beta1 by mode before locking the page, as it said.
* **Line 11 (40 %): the five command counts did worse than the two loop integers.** `B_betti` 68.6 % against `C_hist` 56.3 % on MODE (12.3 points). I had written that the counts would say "perhaps as much as beta1". They say clearly less. I have not tested why.
* **Line 5 (80 %): the best reader of the window is better than my bar.** 93.9 % for `A_window`, against 80 % asked; I had put the cap lower because about a third of the steps have a window that holds a switch (segments of about 150 steps, windows of 50). The readers still name the mode on about nine steps in ten, so they must recognise a new mode after a handful of steps; I did not measure how many.
* **Line 7 (30 %) and its size.** I said B would fall short by more than 5 points. It falls short by 25.4.

## The tables
**All the views** (mean of the three seeds; the range over the seeds in the next column; the majority rate is that of the training rows; `A_match` is not trained and answers NEXT only):

| view | MODE | MODE, range over seeds | TOUR F1 | NEXT | NEXT, range over seeds |
|---|---|---|---|---|---|
| `majority` | 30.3 % | 27.7-31.7 | 0.161 | 25.2 % | 24.9-25.7 |
| `state` | 36.4 % | 35.6-37.0 | 0.355 | 28.1 % | 27.2-28.9 |
| `B_betti` | 68.6 % | 67.2-69.6 | 0.698 | 28.6 % | 27.7-29.7 |
| `B_betti3` | 77.7 % | 77.4-78.3 | 0.805 | 29.5 % | 29.3-29.8 |
| `C_counts` | 73.5 % | 72.6-74.3 | 0.756 | 28.1 % | 28.1-28.2 |
| `C_hist` | 56.3 % | 54.5-57.3 | 0.509 | 32.1 % | 31.9-32.5 |
| `D_short` | 69.4 % | 69.4-69.5 | 0.656 | 57.1 % | 55.3-60.3 |
| `A_window` | 93.9 % | 93.9-94.0 | 0.940 | 65.4 % | 63.3-68.8 |
| `A_eng` | 82.9 % | 82.6-83.1 | 0.813 | 28.9 % | 28.6-29.2 |
| `A_all` | 91.0 % | 90.7-91.6 | 0.902 | 65.2 % | 63.2-68.6 |
| `A_match` | - | - | - | 56.1 % | 54.3-58.6 |

**MODE by the true mode of the step** (the share of steps of that mode that were named right):

| view | tour | drift | random | all |
|---|---|---|---|---|
| `majority` | 33.3 % | 33.3 % | 33.3 % | 30.3 % |
| `state` | 41.7 % | 30.0 % | 38.0 % | 36.4 % |
| `B_betti` | 77.9 % | 46.7 % | 81.2 % | 68.6 % |
| `B_betti3` | 78.9 % | 79.9 % | 74.5 % | 77.7 % |
| `C_counts` | 70.4 % | 72.0 % | 77.9 % | 73.5 % |
| `C_hist` | 48.9 % | 52.4 % | 67.0 % | 56.3 % |
| `D_short` | 66.2 % | 68.3 % | 73.9 % | 69.4 % |
| `A_window` | 93.2 % | 93.5 % | 95.0 % | 93.9 % |
| `A_eng` | 77.0 % | 84.5 % | 86.8 % | 82.9 % |
| `A_all` | 87.0 % | 91.8 % | 94.1 % | 91.0 % |

**NEXT by the true mode of the step** (random mode picks one of four moves at random, so 25 % is chance there for any reader):

| view | tour | drift | random | all |
|---|---|---|---|---|
| `majority` | 25.8 % | 24.7 % | 25.1 % | 25.2 % |
| `state` | 34.8 % | 24.5 % | 25.1 % | 28.1 % |
| `B_betti` | 35.5 % | 25.7 % | 24.9 % | 28.6 % |
| `B_betti3` | 36.5 % | 27.4 % | 24.8 % | 29.5 % |
| `C_counts` | 33.8 % | 25.6 % | 25.1 % | 28.1 % |
| `C_hist` | 41.4 % | 30.2 % | 25.0 % | 32.1 % |
| `D_short` | 70.5 % | 75.9 % | 25.0 % | 57.1 % |
| `A_window` | 88.2 % | 83.4 % | 24.8 % | 65.4 % |
| `A_eng` | 34.7 % | 27.1 % | 24.9 % | 28.9 % |
| `A_all` | 87.6 % | 83.2 % | 25.1 % | 65.2 % |
| `A_match` | 82.9 % | 60.8 % | 25.0 % | 56.1 % |

**beta1 by mode, and the mode shares of the test steps**, per seed:

| seed | beta1 in tour | in drift | in random | test steps in tour / drift / random |
|---|---|---|---|---|
| 1 | 1.428 | 0.980 | 3.202 | 31.7 % / 30.8 % / 37.4 % |
| 2 | 1.416 | 1.005 | 3.148 | 33.3 % / 31.5 % / 35.2 % |
| 3 | 1.397 | 1.011 | 3.124 | 33.4 % / 38.9 % / 27.7 % |

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The roadmap's comparison, as I built it, reads NO on both of its questions.** Naming the mode: `B_betti` 68.6 %, the best reader of the 50 commands 93.9 %. Naming the next command: `B_betti` 28.6 %, the best reader of the window 65.4 %. Detecting a tour: F1 0.698 against 0.940.
2. **On NEXT the pair adds nothing.** `B_betti` 28.6 %, the cell alone 28.1 %, the majority command 25.2 %. Whatever the two integers say about the mode, they do not say which way the walker is going: they are the same number for a loop gone round clockwise and anticlockwise, and no information about the last move.
3. **What the pair holds is one scale of the loops.** It names random mode well (81.2 % of random steps) and a tour fairly (77.9 %), and it confuses drift (46.7 % of drift steps right; the cell alone gets 30.0 % there). The same quantity at three scales, `B_betti3`, names drift 79.9 % of the time and reaches 77.7 % on MODE: 9.2 points above the pair, 16.2 points below the best reader of the window. So the pair is the weakest form of its own idea, as line 12 asked, and a richer topological summary is not ruled out by this run: only the roadmap's two numbers are.
4. **Where prediction lives is the last few commands.** The last three commands alone (`D_short`) name the next command 57.1 % of the time, 87 % of what the best window reader gets (65.4 %); the whole window adds 8.3 points. In this world "temporal momentum" is the last few moves, which a window holds and no count of loops can.
5. **The loss on NEXT is not special to topology.** `A_eng`, the 26 repeat numbers of the window (how often a command equals the one k steps back), names the mode 82.9 % of the time and the next command 28.9 %, which is the cell alone again. Any summary of the window that forgets which command was which loses the direction; the Betti pair and the repeat numbers lose it alike.
6. **`A_match` and the window.** The direct reader of repeats (`A_match`) gets 56.1 % on NEXT (82.9 % in tour mode, 60.8 % in drift mode, 25.0 % in random mode, where nobody can beat a quarter); the tree model reading the 50 raw columns gets 65.4 % (88.2 % in tour mode). Adding the 26 repeat numbers to the window (`A_all`, 91.0 % on MODE) did not help the tree model. The worry on the page, that a tree model reads a raw window badly, did not bite here, though a model built for sequences could still do better than `A_window`.

## What this does and does not show about the roadmap's fourth test
* **"Topological cobordism", "persistent memory without a context window".** Nothing here builds a cobordism or tests memory in a model. What was compared is two integers of a graph I defined against 50 commands.
* **"If Arm B matches Arm A's task accuracy, you have proven that chronological intelligence is a geometric property."** Arm B did not match, so the "if" is not met on this world. The reverse does not follow: that two integers of one graph fall short does not show that no geometric summary matches a window.
* **Token cost.** Not measured. The window of 50 bare commands is a few hundred tokens at most; the roadmap's "50,000 tokens" is a different regime.
* **Not done and worth doing:** a summary that is learned and not chosen (a small network that compresses the window to two numbers, for the fair comparison with the pair); other complexes and other scales (`B_betti3` points that way); a world in which loops are what the task needs (returning to a start), where I would now expect the pair to do better; a language model reading the history; a second world; a second learner.

## Reproduce
In `python/`: `python -m axi.experiments.histshape --out axi_histshape.json` (about sixteen minutes on two CPUs; it needs numpy and scikit-learn). The printed digest should be **c30005542a6567da** whatever the machine; the accuracies may differ in the last digits between machines (the trees use floating point), the 13 lines should not move. `tests/test_histshape_result.py` recomputes every figure of this Result from the saved file, replays the 60 walks against the digest, and, with `AXI_HEAVY=1`, runs seed 1 again and compares it with the file.

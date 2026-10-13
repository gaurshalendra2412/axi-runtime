# Two efficiency claims from a pasted audit, measured on a CPU (written BEFORE any number was read, 10 Oct 2026)

A pasted AI audit (10 Oct) listed five "missing latency and efficiency modules". Two of its claims can be checked against real data right now, with no GPU and no model call, so they are checked here. The other three are answered on the concept-map item (89): the 65 percent density switch is already tested and contradicted (`tests/test_wire.py`), the mask already sits inside the decoding loop of the local runs (`CICOProc` in `colab_llm_experiment.py`; measured 77 us per token at batch 1, `VERIFIED.md`), and dropping a finished step from a model's memory needs a Colab experiment that does not exist yet. Code: `python/axi/experiments/efficiency.py`. Checks of the instrument without any measurement: `python/tests/test_efficiency.py`.

## The two claims, in plain words
**Transport (the paste's point 5).** The paste says the experiment loops lose "2 to 8 ms per step" to JSON, string splitting and regular expressions, and that a binary frame over shared memory would do the transfer in "under 350 ns". What we can time is the work our own step really does (turning the state into prompt text, and a JSON round trip of the same state) and our Python reference of the binary frame (`axi.wire`: build the frame, CRC-32C, decode), against one model call.

**Leaving empty cells out (the paste's point 4).** The paste says that writing only the non-empty cells of a grid, instead of all N x N cells, gives "often 80 to 90 percent" fewer tokens, and that collapsing the 8 rotations and mirrors to one canonical form removes the need for 8-fold test-time search. The repo already has both pieces and proves them lossless on random grids (`d4.compile_stream(keep_background=False)`, `d4.canonicalize_task`); nobody has measured them on real ARC grids.

## Design
* **Transport.** The 140 graphs of `make_scope_tasks(140, 7)` (seed 7; 8 services and about 8 edges each). For each graph, the smallest mean over 5 blocks of 200 calls of: the prompt text (`agent_loop._serialize`), a JSON dumps plus loads of the same state, the Python frame path (convert to COO tuples, `wire.encode` with CRC-32C, `wire.decode`, `unpack_coo`), the frame decode alone, and the filter (`repair_m6`) for reference. Reported: the median over the 140 graphs, in whole nanoseconds. The model-call reference is the median seconds of the 140 saved single requests of the hosted-model run (`docs/results/axi_gemini_single_norule_gemini-3.5-flash-lite_seed7.json`). Only orders of magnitude are read; the timings depend on the machine and will not repeat to the digit.
* **ARC.** The 400 public training tasks of `fchollet/arc-agi` at commit 399030444e0ab0cc8b4e199870fb20b863846f34. For each task, the grids a prompt shows: every training input, every training output and the first test input. Plain text = the digits, one row per line (the layout of `d4.task_prompt`). Sparse text = a header `HxWtT:` and one `i,j=colour` per non-empty cell, in the canonical orientation, joined by `;`. **Colour 0 is taken as empty**, which is true of most ARC tasks and not of all. Lengths are counted in **characters**, a proxy for tokens: a digit or a punctuation mark is about one token for the digit-by-digit tokenisers this repo targets, and a real tokenizer may merge some punctuation, which would shorten both texts. For each task also: how many different plain prompts its 8 rotated/mirrored images give, and how many different prompts after `canonicalize_task`.

## Predictions (mine, written first; the number is how sure I am)
1. Transport [claim]: the prompt text and the JSON round trip each take less than 2 ms per step. (97 %) I expect tens of microseconds, not milliseconds.
2. Transport [claim]: the Python frame path is **not** faster than the JSON round trip. (85 %) Reason: the CRC-32C in `axi.wire` is a byte-by-byte Python loop; the Rust crate is not in this repo copy and is not timed.
3. Transport [claim]: decoding a frame does **not** take under 350 ns in this Python reference. (99 %)
4. Transport [scale]: text plus JSON are under 1 percent of one model call. (99 %) The saved hosted-model run has a median of about 0.6 s.
5. ARC [guard]: sparse text restores every one of the grids exactly. (99 %; it is lossless by construction and was tested on random grids)
6. ARC [guard]: the canonical task prompt is identical for all 8 images of each of the 400 tasks. (96 %)
7. ARC [claim]: fewer than one grid in four is 80 percent shorter or more with empty cells left out. (75 %) A cell written as a tuple costs about 6 to 8 characters against about 1 for a digit, so sparse text wins only when fewer than roughly one cell in six is lit.
8. ARC [claim]: sparse text is shorter than plain text on **fewer than half** of the grids. (70 %)
9. ARC [claim]: summed over all grids, sparse text is **longer** than plain text. (80 %)
10. Not a reading-rule line. The median share of empty cells over all grids is between 0.5 and 0.8. (60 %) The mean number of different plain prompts among the 8 images of a task is at least 7.5. (75 %) The median text path is between 5 and 200 microseconds. (80 %)

The pattern I expect: transport yes, NO, NO, yes; ARC yes, yes, NO, NO, NO. That this exact pattern comes out, I rate about **30 %**. If it does, the finding is: the transport costs are millions of times smaller than one model call so the shared-memory frame cannot matter for latency in this loop, and leaving empty cells out helps only the minority of ARC grids that are mostly empty.

## Reading rules (fixed now; the program prints exactly these)
Transport, four lines:
1. [claim] prompt text and the JSON round trip each take less than 2 ms per step (the paste: 2 to 8 ms)
2. [claim] the Python binary-frame path (convert, encode with CRC-32C, decode) is faster than the JSON round trip
3. [claim] decoding a frame takes under 350 ns in this Python reference (the paste: under 350 ns for the whole transfer)
4. [scale] prompt text plus the JSON round trip together are under 1 percent of one model call (the saved hosted-model run)

ARC, five lines:
5. [guard] writing a grid as sparse tuples and reading it back restores every grid exactly
6. [guard] the canonical task prompt is identical for all 8 rotated/mirrored images of every task
7. [claim] at least one grid in four is 80 percent shorter or more when empty cells are left out (the paste: often 80 to 90 percent)
8. [claim] leaving empty cells out gives a shorter text for at least half of all grids
9. [claim] summed over all grids, the sparse text is shorter than the plain grid text

**How the result will be worded.** A [guard] line "no" is a finding about the framework and is looked into before anything is claimed. A [claim] line "no" means that the pasted number is not reproduced here; the measured number is reported next to it, and nothing is inferred about code that was not run. A [scale] line "no" would mean that transport is a real share of a step and would be worth a closer look. A line that comes out "NO" is a finding, not an error.

## What this cannot show
Transport: a Python reference only. The Rust crates are not in this repo copy and are not timed, and shared memory itself is not used (no `shm_open`, no second process): the frame is built and decoded in one process, so nothing here says what a Rust implementation over a shared ring would cost. One machine (a cloud container); small graphs only (8 services); the model-call reference is a hosted model, network included. ARC: lengths in characters, not tokens of any particular model; colour 0 taken as empty; the training set only; prompt grids only. **A shorter text says nothing about whether a model solves the task better**: a model may read a list of tuples worse than a grid, and that was not tested. Nothing here is about equivariant attention (not built) or about accuracy under the canonical form; the canonical prompt being identical only means that a rotated copy of a task looks the same to a model.

## How it is run
```
PYTHONPATH=python python3 -m axi.experiments.efficiency transport --model-run docs/results/axi_gemini_single_norule_gemini-3.5-flash-lite_seed7.json --out docs/results/axi_efficiency_transport.json
git clone --depth 1 https://github.com/fchollet/arc-agi /tmp/arc && git -C /tmp/arc checkout 399030444e0ab0cc8b4e199870fb20b863846f34
PYTHONPATH=python python3 -m axi.experiments.efficiency arc --arc-dir /tmp/arc/data/training --out docs/results/axi_efficiency_arc.json
```

---

# Result

Run on 10 Oct 2026 in the cloud container (Python 3.13.16), after the page above was fixed (its text above this line is byte for byte what was written before the run, checked by a hash in `tests/test_efficiency.py`). Files: `docs/results/axi_efficiency_transport.json` (sha256 `f54ab354c33fabecc79b595804c84c08365e694f7552ed586d2e3fca3aad920e`), `docs/results/axi_efficiency_arc.json` (`aa544ec77a9ae7119d18887eb8c7691033f8d8af7e99ef6db3818b5fdf441f95`), `docs/results/axi_efficiency_arc_modal.json` (`d89ced6b1f57b44432360f21ce57ab46452c5143e646bf5ba9c9734d758e915f`; the exploratory re-read of section 4). Check of every number below against the files: `python/tests/test_efficiency_result.py`. Concept-map item 89.

## 1. The reading-rule lines
Transport: yes, NO, NO, yes. ARC: yes, yes, NO, NO, NO. This is exactly the pattern predicted (transport yes, NO, NO, yes; ARC yes, yes, NO, NO, NO; rated 30 % beforehand). Every numbered prediction held, and so did the three numbers of prediction 10; the measurement was cheap and most of it is arithmetic, so a result without surprises is what it is.

## 2. Transport: what a step costs (median over the 140 graphs, whole nanoseconds, one cloud container)
* Prompt text 7,893 ns (7.9 us); JSON dumps plus loads 15,885 ns (15.9 us); the Python frame path (convert, encode with CRC-32C, decode) 41,378 ns (41.4 us), of which decoding alone 16,486 ns; the filter (`repair_m6`) 5,071 ns. One model call (median of the saved hosted-model run): 510,625,839 ns, about 0.51 s.
* **Text plus JSON together are 0.0047 percent of one model call** (about 21,500 times smaller). The paste's "2 to 8 ms per step" is not what these operations cost: 0.008 ms and 0.016 ms. Sizes: 262 characters of text, 308 bytes of JSON, 164 bytes of frame.
* The Python frame path is 2.6 times slower than the JSON round trip, and its decode is 47 times the paste's 350 ns. The cause is the byte-by-byte CRC-32C in Python; the Rust crate is not in this repo copy and was not timed, so **nothing here says how fast a Rust frame over shared memory would be**. It says that the state transport is not where a step's time goes: taking text and JSON out of a step altogether would shorten it by 0.0047 percent.
* Machine note: the filter measured 5,071 ns here and about 13 microseconds on the Colab runs in `VERIFIED.md`. Timings move with the machine; only the orders of magnitude are read.

## 3. ARC: leaving empty cells out (400 training tasks, 3,004 prompt grids, colour 0 empty)
* **Lossless and canonical, on real data:** the sparse text restores 3,004 of 3,004 grids exactly, and the canonical task prompt is identical for all 8 rotated/mirrored images on 400 of 400 tasks. Without canonicalising, the 8 images of a task give 7.87 different prompts on average; with it, 1. This is the part of the paste's point 4 that holds, and it holds on real tasks.
* **The length claim does not hold.** The median grid is 0.650 empty. The sparse text is shorter than the plain grid on 695 of 3,004 grids (23.1 %); 50 percent shorter or more on 301 (10.0 %); **80 percent shorter or more on 85 (2.8 %)**, against "often 80 to 90 percent". Summed over all grids the sparse text is 2.62 times the plain text (1,077,079 against 411,386 characters).
* **Where it does help:** only on grids with fewer than about one cell in five lit. Of the 879 grids with under 20 percent of cells non-empty, the sparse text is shorter on 695; of the 2,125 grids with 20 percent or more lit it is shorter on none. (A tuple costs about 6 to 8 characters, a plain cell about 1.)

## 4. Exploratory re-read (decided after seeing section 3; no prediction was written for it)
Colour 0 as "empty" is the weakest reading for tasks with a coloured background, so each grid's most frequent colour was left out instead (the header then names it). The sparse text is shorter on 742 of 3,004 grids (24.7 %), 80 percent shorter or more on 78 (2.6 %), and 1.99 times the plain text in total. Lossless on 3,004 of 3,004. So the length claim fails under the friendliest reading too.

## 5. What this means
* The state transport is not a latency lever in this loop: tens of microseconds against about half a second. The shared-memory frame may still be worth building for other reasons (a defined, checksummed format between processes), but not to make steps faster in this loop: the Python path already bounds the transport at 0.0047 percent of a step, and a faster implementation could only lower that.
* Dropping empty cells is a lossless re-encoding, and the canonical form collapses a task's 8 images into one prompt on all 400 real tasks. It is **not** a way to shorten ARC prompts by 80 to 90 percent: for most grids the tuples are longer than the grid.
* Not tested: whether a model reads a canonical prompt better or worse, whether tokens (not characters) change the ranking, and anything about the Rust code. Characters are a proxy for tokens; colour 0 or the most frequent colour are two of several ways to say "empty"; the training set only.

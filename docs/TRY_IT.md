# Check AXI yourself

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged. Architecture by **Rajnish Choubey**; tests built and run by Shalendra Gaur.

Prefer a browser? [Open the notebook in Colab](https://colab.research.google.com/github/gaurshalendra2412/axi-runtime/blob/main/python/colab/axi_try_it.ipynb) and run the cells: check 1 needs no key. The published bottlenecks these results map to, with sources, are in [`FOR_ENGINEERS.md`](FOR_ENGINEERS.md).

Three checks, each one command, from the shortest to the longest. Start in a copy of the repository:

```bash
git clone https://github.com/gaurshalendra2412/axi-runtime
cd axi-runtime/python
```

## 1. Three seconds. No key, no model, no GPU.

```bash
python3 quickstart_replay.py
```

It replays the saved real replies of the 4,000-step run (Gemini 3.8 Flash) through this repository's own code and recomputes the held state step by step. It prints:

| Line | What you should see |
|---|---|
| held state exactly the right one | 4,000 of 4,000 steps |
| the model's replies exactly the ideal reply | 3,954 of 4,000 (98.85%) |
| first 400 requests, replies applied as written | 85 of 400 steps with the right state |
| first 400 requests, the AXI run | 400 of 400 |
| cost of a step with the right state, as written against AXI | 4.5 times |

The last line says whether every figure matches the published ones. The script stops with an error if one replayed step differs from the saved one.

## 2. A few minutes. Your own key.

This runs the same harness live, on your own Gemini key (a free key from [Google AI Studio](https://aistudio.google.com/apikey) is enough):

```bash
export GEMINI_API_KEY=your_key
pip install google-genai numpy
python3 live_check.py
```

The first 200 steps are answered by the program itself (nothing is sent), so the graph has already grown; then Gemini 3.8 Flash is asked for 15 steps. At the end it prints, for those live steps:

* on how many the held state is exactly the oracle's (AXI step) and on how many it is broken;
* how many of the model's replies were exactly the ideal reply, and how many steps the same replies would have left off the right state if they had been applied as written;
* the prompt size at the first live steps and at the last (the host rebuilds the prompt from the held state, so it does not grow with the step), the seconds a step took, and the tokens sent.

Every reply is saved the moment it arrives (`python/live_check_out/`). Running the same command again calls the model for nothing and prints the same table. If the free daily limit stops it, the same command carries on the next day. The key is held in memory; it is never printed and never written to a file.

## 3. Longer, or another model.

```bash
python3 live_check.py --steps 100 --daily-limit 0 --min-interval 0.3
python3 live_check.py --model gemini-3.5-flash-lite
```

The first line is for a paid key: the 4,000-step run cost $4.95 at the prices in the README, so 100 steps cost about 12 cents. The second names another model; the script looks up the id your key can see. `--report-only` prints the table of the steps already on file without calling a model. The 4,000-step numbers themselves are the model and the settings of check 1.

## What these checks show

* The held state is checked against an oracle after every step, so "exactly the right one" is a count you can read, not a claim.
* The same reply is run two ways from the same starting state: applied as written, and through the AXI step (gate, repair, commit whole or not at all). The difference between the two is what the host layer adds; the model is the same in both.
* The prompt is built from the held state at every step, so its size follows the size of the state, not the number of steps taken.

## Files

* `python/quickstart_replay.py`: check 1.
* `python/live_check.py`: checks 2 and 3. Its test, with a stand-in model that makes the same slip Gemini made, is `python/tests/test_live_check.py` (no key, no network).
* `docs/MILLION_TOKEN_TEST_HARD.md`: the page of the 4,000-step run, written before it was run.
* `docs/results/`: the saved replies.

Not affiliated with Google. AI assistance in writing the code and these pages is disclosed in the README.

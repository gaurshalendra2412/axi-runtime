# AXI for companies that run models

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged. It is for the company that runs the model, not for end users, and it fits any LLM application that holds state: agents, chat, workflows, short jobs and long ones, from the first step. The architecture is by **Rajnish Choubey**; the tests and measurements were built and run by Shalendra Gaur, working with an AI coding assistant that is a tool, not an author.

The host holds the exact state and builds a short prompt from it at every step. The model proposes one change; the host checks it against the rules, completes what the rules can complete, and commits it all or not at all. So drift, cost, compute, latency, speed and energy are addressed by the same mechanism, and each has a measured number.

## What a company counts, and what AXI measured

| What a company counts | Measured | Test page |
|---|---|---|
| **Drift** (is the held state the true state?) | The state the host held was exactly right on **4,000 of 4,000 steps** on Gemini 3.8 Flash and on **80,000 of 80,000 steps** on a 7B model. Replies applied as written left the state exactly right on 85 of 400 steps. | [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md), [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md) (Test 43) |
| **Cost** (tokens sent) | 99,809 prompt tokens a turn with the chat kept, 1,177 with AXI: **85 times fewer** over 400 turns. On a 1,500-step job the kept chat sent 476,835 tokens a turn on average over the 457 turns it ran, AXI 2,824 a step over all 1,500: **169 times fewer a step**. A step with the right state cost $0.00117 with AXI and $0.00524 with replies applied as written (4.5 times). | [`GEMINI_FUZZY_TEST.md`](GEMINI_FUZZY_TEST.md) (Test 23), [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md), [`MILLION_TOKEN_TEST_HARD.md`](MILLION_TOKEN_TEST_HARD.md) |
| **Cost** (dollars, against the provider's own bill) | At the provider's own rates (an output token 5.000 times an input token, a cached token 0.100 of one), one verified step of the 400-turn job cost **$0.0014 with AXI, $0.0870 with the chat kept and its cache cold (61 times) and $0.0148 with the chat kept and its cache warm (10.4 times)**; counting every step, right or wrong, 51.4 and 6.8 times. On the day of the cache pair test, both arms together, the provider's bill came within 1.3 percent of the logged total ($29.45 against $29.09). Per million verified steps: about $1,422 with AXI, $14,811 warm and $86,957 cold (the measured cost of a step times a million; the warm figure is worked out from its tokens, not a bill line). | [`BILLING_CHECK.md`](BILLING_CHECK.md) |
| **Compute** (sessions per chip) | **256 sessions at once on one NVIDIA L4 with a real 7B model, every step exact: 105,230 exact steps per GPU hour.** In memory alone (no model run), **one TPU v6e chip held 927 AXI-sized sessions against 5 with the chat kept** (at the kept chat's long-run mean of 476,835 tokens). | [`SESSIONS_PER_CHIP_TEST.md`](SESSIONS_PER_CHIP_TEST.md) (Test 38), [`TPU_SESSIONS_TEST.md`](TPU_SESSIONS_TEST.md) (Test 30) |
| **Latency** (time to first token) | Median **2.27 s with AXI** over all 1,500 steps, against **9.13 s** for the kept chat once its prompt passed 500,000 tokens (4.02 times). On the 7B model a step took 0.1605 s in the first 2,000 rounds and 0.1651 s in the last 2,000 (four jobs of 20,000 steps), with a prompt of about 620 tokens. | [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md), [`ENDURANCE_7B_TEST.md`](ENDURANCE_7B_TEST.md) |
| **Speed** (steps an hour) | At a prompt of 500,000 tokens or more the kept chat took 10.66 s a call; AXI took 3.07 s a step (**3.47 times faster**). Exact steps an hour over the first 457 steps of each: **1,280 with AXI against 488** (2.62 times). | [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md) |
| **Efficiency** (energy, compute) | **2.458 joules per exact step** at 256 sessions on one L4. On a 14B model over 600-step jobs, **413.7 prompt tokens a step** with the exact state shown against 8,228.8 with the chat kept (which reached its 16,000-token limit near step 400); the host's check kept the held state exact in every arm. | [`SESSIONS_PER_CHIP_TEST.md`](SESSIONS_PER_CHIP_TEST.md), [`MEMORY_ARMS_TEST.md`](MEMORY_ARMS_TEST.md) (Test 44) |
| **Hostile input** | 10,000,000 proposals, 1,401,041 of them injection attempts: **0 committed**. | [`GATE_STRESS_TEST.md`](GATE_STRESS_TEST.md) (Test 39) |

## Why these come together

The prompt is built from the held state, so its size follows the state and not the number of steps already run; cost, time to first token and compute per step stay flat from step 1. The state is exact because the host commits only changes that pass its rules, so a slip never reaches it.

## What other groups measured on the same failures

On OSWorld 2.0 (XLANG Lab, University of Hong Kong, 26 Jun 2026: 108 long computer-use workflows, leading agents averaging more than 300 steps) the best 500-step configuration completes 20.6% of tasks, and the authors report that agents cannot hold a task-level model together over a long horizon and spend under 7% of their budget on repairing their own errors. AXI keeps that state on the host, checked by the gate, so the model does not have to hold it: 80,000 of 80,000 steps exact on a 7B model ([Test 43](ENDURANCE_7B_TEST.md)). [`THIRD_PARTY_EVIDENCE.md`](THIRD_PARTY_EVIDENCE.md) sets twenty-one papers and one engineering post by other groups beside the AXI test that measures the same thing, with what each shows and does not show; none of the papers uses AXI.

## Check it yourself

No key, no GPU, about 3 seconds. It replays the saved replies of the 4,000-step run and recomputes the held state with the repository's own code.

```
git clone https://github.com/gaurshalendra2412/axi-runtime
cd axi-runtime/python
python3 quickstart_replay.py
```

Or open the [one-click Colab notebook](https://colab.research.google.com/github/gaurshalendra2412/axi-runtime/blob/main/python/colab/axi_try_it.ipynb). The one-page summary with every table is [`AXI_ONE_PAGE.md`](AXI_ONE_PAGE.md); the white paper is [doi.org/10.5281/zenodo.23958737](https://doi.org/10.5281/zenodo.23958737) and the technical report is [doi.org/10.5281/zenodo.24508592](https://doi.org/10.5281/zenodo.24508592).

*Every figure is scoped on its test page and can be recomputed from the saved run files. Architecture by Rajnish Choubey; tests and measurements by Shalendra Gaur.*

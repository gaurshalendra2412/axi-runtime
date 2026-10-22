# What AXI does for a company that runs the model

*Gemini 3.8 Flash on one long, heavy job, with and without AXI. The architecture is Rajnish Choubey's; the code, the tests and the measurements are the owner's. Every figure on this page is recomputed from `docs/results/axi_gemini_long_session_seed20261111_default.json` by `tests/test_gemini_long_result.py`; the full page with its pre-written reading rules is [`GEMINI_LONG_TEST.md`](GEMINI_LONG_TEST.md). Measured on one job, one run of each arm, 22 Oct 2026.*

## The job
A service graph edited request by request, 1,500 requests with AXI. Every request came with about 40 lines of routine operations log, so every turn is heavy. Without AXI the whole chat is kept and sent again at every turn, up to the model's window (457 turns, 949,591 tokens). With AXI the framework holds the exact state and shows it at every step; nothing is kept in the chat.

## The comparison

| What a company counts | Without AXI (the chat kept) | With AXI | What it means |
|---|---|---|---|
| Prompt tokens sent per step | 476,835 on average | 2,824, flat for all 1,500 steps | **169 times fewer tokens a step** |
| Tokens sent in all | 217.5 million in 457 turns | 4.6 million in 1,500 steps | **47 times fewer, for 3.3 times as many steps** |
| State safety (the state held if each reply is applied as it comes) | exact until turn 286, wrong on every turn after; 146 turns corrupted | **exact on 1,500 of 1,500 steps; 0 corrupted** | the state is safe at every step, however heavy the turns |
| Drift of the held state (items from the true state) | up to 17 items off | **0 at every step** | zero after all the heavy turns, and still zero at step 1,500 |
| Latency, first token (median) | 2.23 s at its best (a short chat); 9.13 s once the prompt passes 500,000 tokens | **2.27 s, flat** (2.15 to 2.35 s in every tenth of the job) | AXI keeps the best latency the model has and holds it for the whole job (4.0 times lower than the long chat) |
| Step time (seconds a call) | 7.13 s on average | **3.07 s** | **2.3 times faster a step** |
| Speed (steps an hour of calls) | 505 | **1,171** | **2.3 times more steps an hour** |
| Throughput (exact steps an hour) | 488 | **1,157** | **2.4 times more exact steps an hour** |
| Thinking tokens a step | 77 over the first 50 turns, 239 over the last 100 | **no climb: 210 in the first tenth of the job, 247 in the last** (between 173 and 334 in every tenth) | AXI shows the model a flat state, not a history that compounds, so its effort per step does not grow |
| Tokens computed per exact step, at a prompt of 500,000 tokens or more | 5,419 | **3,108** | **1.7 times fewer** |
| Re-asks and refused replies | not available (replies applied as they come) | **0 and 0** over 1,500 steps | nothing was paid for twice |

## Why it matters to a company that runs the model

* **Fewer tokens is less work done, directly.** A step with AXI sends 2,824 tokens where the kept chat sends 476,835. Tokens sent and seconds of calls are what the hardware time is made of, so 169 times fewer tokens a step and 2.3 times less time a step leave room for more steps on the same hardware.
* **A flat step is a steady step.** The time to the first token stays at 2.27 s from the first step to the last, and the thinking a step stays between 173 and 334 tokens without climbing. A company can plan for a step that costs the same at step 1 and at step 1,500; it cannot plan for one that slows from 2.2 s to 9 s as the chat grows.
* **A right step is a step nobody pays for twice.** The state was exact on 1,500 of 1,500 steps with no re-ask, no refused reply and no repair by hand. Every wrong state a long job avoids is a clean-up, a re-run or a support case that does not happen: the saving there is indirect, and it adds to the saving in tokens.
* **Heavy turns do not wear it down.** The log noise in every request did not change the state, the distance from the true state (0), or the speed of the steps. AXI keeps the model on the state, not on the noise around it.
* **More exact steps an hour.** 1,157 exact steps an hour of calls against 488: 2.4 times the useful work out of the same hour.

## In one paragraph (for the article)

On one long, heavy job, Gemini 3.8 Flash with AXI held the exact state on all 1,500 steps, with zero distance from the true state at every step, while sending 2,824 tokens a step instead of 476,835 (169 times fewer; 47 times fewer in all). Its time to the first token stayed at 2.27 s, the best the model showed, from the first step to the last, its step time was 2.3 times shorter, and it did 2.4 times as many exact steps an hour. Its thinking per step did not climb, and it needed no re-ask. For a company that runs the model, that is fewer tokens, steadier latency, more exact steps for the same hour of hardware, and a state that is safe at every step: AXI turns a long task from something the model has to remember into something the framework holds.

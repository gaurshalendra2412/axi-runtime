# What is verified, what is not (honest status)

| Claim / component | Status | How it was checked |
|---|---|---|
| Strict CICO parser == byte-level DFA | **verified** | 30,000-string mutation fuzz, no disagreement; real GPT-2 vocab on Colab |
| Grammar-masked sampling always parses | **verified** | 3,000 numpy samples; 500 real-BPE samples; 128-row GPU decode loop (Colab) |
| Gate: match + gluing conditions (indexed) | **verified** | unit tests + 4,000-proposal stress; diagnose() == gate on 4,000 fuzz cases |
| Parse -> gate -> pager commit/rollback, no leaks | **verified (Python model)** | invariants audited after every step; Rust pager: see CI |
| Block independence (strict mask), any depth | **verified (toy)** | bit-exact zero change in numpy and on T4; pasted RFC mask leaks (max change 95 on GPU) |
| Layer 1 repair deltas are sound & minimal | **verified (symbolic)** | 4,000-case fuzz; closed loop through Runtime. *Does an LLM recover faster with them? NOT measured* |
| Layer 2 device mask table == CPU masks | **verified (numpy + T4)** | 37 states x 50,257 tokens identical; 128-row GPU decode, all finished rows parse |
| Layer 2 device mask speed | **measured (T4, Colab)** | vs a simple host round-trip: 287 -> 77 us (B=1, ~4x), 8,415 -> 392 us (B=64, ~21x). The RFC's ~10,000x / <0.3% overhead is NOT reproduced; mask costs ~36-54% of one softmax+sample step and can be optimised |
| Layer 3 all-or-nothing commit across ranks | **verified (Python model + 4 real processes)** | 6,000 fault-injected steps in the model; torch.distributed all_reduce(MIN), 3,000 steps, page tables identical on every rank. NVLink latency NOT measured |
| Layer 4 C4 attention exactly rotation-equivariant | **verified (numpy)** | exact per-cell match under 90/180/270 rotations; odd-grid centre bug found and fixed |
| Layer 5 frame: roundtrip, corruption, truncation | **verified (Python reference)** | every single-bit flip detected; garbage never decodes |
| Off-the-shelf LLM + AXI wrapper: parse rate, corrupted-graph rate, recovery after a gate rejection | **pending** (`colab/colab_llm_experiment.py`; logic tested with a scripted stand-in only) | no weights are trained |
| Rollback vs re-prefill on a real KV cache | **pending** (same script, Part 2) | tail rollback = crop; editing earlier context re-prefills the suffix on a stock model |
| Any other speed claim in microseconds / nanoseconds | **NOT verified** | only numbers measured by the benchmarks in this repo count |
| Strict block mask / C4 attention inside a model | **out of scope (needs training)** | kept as optional research; the runtime itself never trains |
| Multi-GPU NVLink consensus latency | **NOT measured** | needs a multi-GPU machine |

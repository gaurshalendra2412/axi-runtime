# What is verified, what is not (honest status)

| Claim / component | Status | How it was checked |
|---|---|---|
| Strict CICO parser == byte-level DFA | **verified** | 30,000-string mutation fuzz, no disagreement; real GPT-2 vocab on Colab |
| Grammar-masked sampling always parses | **verified** | 3,000 numpy samples; 500 real-BPE samples; 128-row GPU decode loop (Colab) |
| Gate: match + gluing conditions (indexed) | **verified** | unit tests + 4,000-proposal stress; diagnose() == gate on 4,000 fuzz cases |
| Parse -> gate -> pager commit/rollback, no leaks | **verified (Python model)** | invariants audited after every step; Rust pager: see CI |
| Block independence (strict mask), any depth | **verified (toy)** | bit-exact zero change in numpy and on T4; pasted RFC mask leaks (max change 95 on GPU) |
| Layer 1 repair deltas are sound & minimal | **verified (symbolic)** | 4,000-case fuzz; closed loop through Runtime. *Does an LLM recover faster with them? NOT measured* |
| Layer 2 device mask table == CPU masks | **verified (numpy)**; GPU timing: run `colab/colab_layers.py` | table/loop equivalence tests |
| Layer 3 all-or-nothing commit across ranks | **verified (Python model, 6,000 fault-injected steps)**; real `torch.distributed` run: `colab/colab_layers.py` | negative control (no consensus) diverges |
| Layer 4 C4 attention exactly rotation-equivariant | **verified (numpy)** | exact per-cell match under 90/180/270 rotations; odd-grid centre bug found and fixed |
| Layer 5 frame: roundtrip, corruption, truncation | **verified (Python reference)** | every single-bit flip detected; garbage never decodes |
| Any speed claim in microseconds / nanoseconds | **NOT verified** | only numbers measured by the benchmarks in this repo count |
| Trained model keeps accuracy under strict block mask / C4 attention | **NOT measured** | needs a training experiment |
| Rollback is cheaper than re-prefill on a real LLM | **NOT measured** | needs a timing experiment |
| Multi-GPU NVLink consensus latency | **NOT measured** | needs a multi-GPU machine |

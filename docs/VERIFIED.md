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
| Layer 4 D4 grid canonicalization (ARC-style), task-level transform | **verified (numpy)** | `engine/d4.py`: canonical form invariant over all 8 transforms incl. non-square (blueprint's byte-compare was not); exact stream round trip; canonical task + prompt byte-identical for all 8 orientations; per-pair canonicalization loses the common rule on orientation-dependent tasks (40/40), task-level 0/40. *Does it improve ARC accuracy? NOT measured* |
| Layer 5 frame: roundtrip, corruption, truncation | **verified (Python reference)** | every single-bit flip detected; garbage never decodes |
| Off-the-shelf LLM + AXI wrapper (no training) | **measured on 3 seeds (Colab T4)**, see `docs/EXPERIMENT_RESULTS.md` | Qwen2.5-3B, 60 tasks/seed, no rule in prompt: constrained blind apply corrupted the graph on 90/90 delete-with-dependents tasks; gate -> 0% corruption; imperative retry and deterministic repair recovered 100% on all three seeds (90/90 delete_dep, 180/180 overall). Seeds 1-2 ran on unchanged code after the wording was fixed. One model family, greedy, synthetic graphs |
| Retry feedback wording | **measured on 3 seeds** | delete_dep success (s0/s1/s2): vague prose 16.7/20.0/6.7%, structured v1 36.7/46.7/33.3%, detailed prose 93.3/86.7/93.3%, structured imperative 100/100/100%. Same ordering on every seed |
| Rule stated in the prompt helps? | **measured: no (3B model), 3 seeds** | imperative retry 100% -> 33.3/30.0/46.7% on delete_dep (noisy; report a range). Failure mode changes: 30/30 pure `dangling` without the rule, only 7/30 with it (seed 2); the rest are phantom edges, add-then-delete, added edges touching the deleted node. Not shown to be attention entropy |
| Repair of `match_edge` / `ident_node` obstructions | **NOT implemented** | reported but not repaired; this is why deterministic repair stalls at ~50% in the rule-in-prompt regime |
| Rollback vs re-prefill on a real KV cache | **measured (T4, 0.5B model)** | tail rollback (crop) < 1 ms at 4k-32k; full prefill 683 ms / 3.2 s / 10.1 s; editing at 25% of the context re-prefills 286 ms / 2.5 s / 8.9 s. Tail rollback is cheap in any engine; the early-edit cost is what block independence would need model support to avoid |
| Any other speed claim in microseconds / nanoseconds | **NOT verified** | only numbers measured by the benchmarks in this repo count |
| Strict block mask / C4 attention inside a model | **out of scope (needs training)** | kept as optional research; the runtime itself never trains |
| Multi-GPU NVLink consensus latency | **NOT measured** | needs a multi-GPU machine |

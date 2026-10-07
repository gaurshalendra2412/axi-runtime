# Triage of the three additional design files (Rajnish Choubey)

Legend: **KEEP** = used as is · **BUILT** = rewritten as tested code in this repo · **FIX** = idea is good, text/code has a specific defect · **LATER** = needs a GPU experiment or more design

## 1. Ideas worth keeping

| Item | Status | Where / next step |
|---|---|---|
| "Syntax Fallacy": valid syntax is not relational admissibility | **KEEP** | Our data supports it (grammar decoding: 100% parse, 100% corrupted on delete-with-dependents). Use Proposition 1 in `PAPER_CORRECTIONS.md`. |
| Invariants belong in the runtime, not in the prompt | **KEEP** | Supported: rule in the prompt did not rescue the 3B model (imperative retry 100% -> 33.3%). Mechanism (attention entropy) is a hypothesis. |
| Reactive imperative steering: tell the model exactly what operations to add | **BUILT** | `agent_loop.feedback_text("structured_imperative")` and `diagnostics.propose_repair`. The wording result (16.7% / 93.3% / 36.7% / 100%) is the strongest finding so far. |
| Obstruction -> deterministic repair delta ("diagnostic inversion") | **BUILT** | `engine/diagnostics.py` (primal = add edge DELs, dual = drop the node DEL). The sketch `AxiImperativeInverter` is replaced by this; see section 3. |
| D4 canonical orbit for ARC grids, store transform id, invert on the answer | **BUILT** | `engine/d4.py`, 12 tests. Fixes in section 2. |
| AXI-DSL as spans: MATCH (L) / PRESERVE (K) / PRODUCE (R) | **LATER (good)** | A second grammar for the same decoder would let the gate check the identification condition against K *before* commit. Needs a new DFA + gate input type. Worth doing after seed-1 results. |
| Milestones: (1) kernel hook, (2) ARC runner, (3) preprint | **KEEP** | Order suggested: preprint with corrected numbers first, then ARC orientation-invariance experiment, then the kernel hook. |

## 2. ARC D4 compiler — what changed from the blueprint

| Blueprint | Defect (reproduced in `tests/test_d4.py`) | Now |
|---|---|---|
| canonical form = min of `grid.tobytes()` | shape not part of the key: `[[1,2]]` and its rotation `[[1],[2]]` have equal bytes, so the two orientations got **different** "canonical" grids | key = (shape, values) |
| `inv_map = {0:0,1:3,2:2,3:1,4..7:self}` | correct, but was assumed | derived by brute force and asserted equal to the blueprint's table |
| drops colour 0 | 0 is a real ARC colour (black); fine only if h, w are known | header carries (h, w, t); background cells optional; exact round trip tested incl. all-zero grids |
| same phase repeated on every token | redundant | one header per grid |
| ties for symmetric grids | transform id not unique | `stabilizer()` lists all; any is valid |
| canonicalize each grid on its own | valid only for D4-**equivariant** rules (recolour, fill). For orientation-dependent rules (gravity) every pair gets a different transform: in the test 40/40 tasks lost a common rule | `canonicalize_task`: ONE transform for all demos + test input; 0/40 lost the rule |
| — | — | canonical task and its **prompt text are byte-identical for all 8 orientations**, so any deterministic model gives an orientation-invariant answer after `restore_answer` |

Not shown (and not claimed): that a model solves more ARC tasks. Canonicalization removes orientation variance; it does not add capability. A 3B model's zero-shot ARC accuracy is expected to be very low. The measurable experiment is: accuracy of the same model on the 8 orientations of each task, raw vs canonical (raw should vary, canonical must be identical by construction), plus the overall accuracy for honest context.

## 3. `AxiImperativeInverter` sketch

* `failed_cmd.split(",")[1]` is the **column**, not a node id; the delta language addresses nodes by (row, col).
* `DEL_EDGE[u,v]` and `RETRACT_ADD[...]` are not in the grammar the decoder enforces; our edge delete form is `DEL[(r,c)->(r,c):rel#w]`.
* It needs `host_graph.get_incident_edges`; ours is `Graph.incident`.
* The working equivalent was already measured at 100% (seed 0).

## 4. Qwen monkey-patch (`patch_qwen.py`, `test_unified_loop.py`) — not usable as written

Reasons it would fail on Colab (none of this can be run in this sandbox: no torch/GPU):
1. old HF API (`self.attn.num_heads`, `rotary_emb(value_states, seq_len=...)`, `get_usable_length`) — changed in current transformers;
2. writes only the last token's K/V into the pool, so prefill (q_len > 1) breaks;
3. calls `dynamic_block_sparse_attention`, which does not exist (the kernels are `block_sparse_attention` / `_train`), and the decode kernel has no causal mask;
4. no GQA handling (Qwen2.5-3B: 16 query heads, 2 KV heads);
5. return signature does not match the decoder layer;
6. `from axi.pager import TorchShadowPager` does not exist; `np` used without import;
7. does not use the strict interface mask, and a pretrained Qwen was never trained under block independence, so output quality with a changed attention pattern is unknown.

What it would take to make this real: an attention wrapper that keeps HF's own prefill path and only intercepts decode steps, tested first for *exact equality with stock attention on an all-visible mask*, then for the strict mask on a toy model. That is a separate GPU project and it is outside the "never trains" scope; the runtime wins we measured (gate, mask table, consensus) do not depend on it.

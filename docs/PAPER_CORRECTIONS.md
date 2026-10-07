# Paper draft ("The Syntax Fallacy ...") — what is solid, what to fix before it is posted

Architecture, mathematics, the "Syntax Fallacy" framing and the AXI-DSL formalism: **Rajnish Choubey**.
Implementation, tests and the Colab experiments that produced the numbers: Shalendra Gaur, with AI-assisted coding.

Every row below was checked against `docs/EXPERIMENT_RESULTS.md`, the Colab output files, or the test-suite.

## A. Solid — keep as written

| Claim in draft | Evidence |
|---|---|
| Main results table, both panels, all 8 modes x 2 regimes (n=60, delete_dep n=30) | matches the seed-0 Colab output cell for cell |
| Syntax Fallacy as a phenomenon: grammar-valid output, 100% corrupted graphs on delete-with-dependents | measured: unconstrained and constrained both emitted `DEL[0,1]`; gate obstruction = `dangling` |
| Reactive steering wording matters (prose 16.7% -> detailed 93.3% -> imperative 100%) | measured, same 30 tasks |
| Rule-in-prompt did **not** help the 3B model (imperative 100% -> 33.3%) | measured |
| Gate drives corruption to 0% in every mode | measured |

## B. Must be changed (factually wrong or unmeasured)

| # | Draft says | Problem | Replace with |
|---|---|---|---|
| 1 | "NVIDIA A100-SXM4-40GB" | The runs were on a **Tesla T4** (Colab). | "NVIDIA Tesla T4 (16 GB), Google Colab" |
| 2 | Latency table: axi-logits 1.2 us, axi-dpo 11.4 us, axi-sync 2.8 us, axi-cache 0.08 us, "10,000x" | None of these were measured. The only measurements are in section D below. | Replace the table with section D (measured numbers) and say the rest is "not yet measured". |
| 3 | "axi-cache 0.08 us vs 8.9 s re-prefill" | Compares two different operations. Rolling back the **tail** is a `crop` (0.35-0.8 ms measured) and is cheap in every engine. The 8.9 s figure is the cost of **editing early context** (25% into a 32k context) and re-computing the rest. | State both numbers, side by side, for what they are. |
| 4 | "`DEL[0,1]` ... `DEL[u,v]`" and "`ADD[0,1:9]` adds an edge with payload :9" | In the CICO language `DEL[0,1]` deletes the **node at row 0, column 1**. `ADD[0,1:9]` adds a **node** at (0,1) with value 9. `ident_node` arises because that node already exists. | Fix both sentences and the proof of Theorem 1. |
| 5 | Theorem 1: `R_corrupt(M1) = 1.0` whenever rho(G) > 0 | (a) rho > 0 only says some edge exists, not that the deleted node has one — the premise needs deg(v) > 0. (b) 1.0 is a measured rate for one model's proposals, not something grammar decoding can prove. | Use the Proposition below (statement that *is* provable) and report 100% as a measurement. |
| 6 | "inflates attention entropy" / "phase dispersion" explains the rule-stuffing drop | Never measured. Seed-1 transcripts suggest a simpler cause: with the rule the model tries to delete the incident edges but lists phantom or mis-stated edges (`match_edge`), or writes `ADD[r,c:v] DEL[r,c]` (`ident_node`). | Describe the observed behaviour; keep entropy as an untested hypothesis. |
| 7 | "100% strict-parse for grammar-constrained decoding" and "unconstrained 100% strict parse" | Constrained: 100% with no rule in the prompt on every seed, but with the rule 98.3 / 100 / 96.7% overall (96.7 / 100 / 93.3% delete_dep); cause unknown (check `tokens` on failing rows: truncation at 120?). Unconstrained: 100 / 91.7 / 98.3% (no rule), 88.3 / 88.3 / 78.3% (rule). | Report ranges over seeds. The mask removing parse failures while the proposals it rescues all corrupt the graph is itself a clean Syntax Fallacy illustration. |
| 8 | M5 "100%" presented as an AI result | M5 is a deterministic repair and the success criterion is "graph equals cascade-delete". It shows the pipeline works, not that a model got smarter. | Say so in one sentence; headline the M3 numbers (a real second LLM call). |
| 9 | Layer 4 as D4 with per-cell tau(i,j) = min over g of g.(i,j) | Verified code is **C4** per-cell phase attention (`engine/c4.py`) and, new, **D4 per-grid** canonicalization (`engine/d4.py`). Per-cell lexicographic minimum is not what either does. | Describe D4 as grid-level canonicalization + one stored transform id. |
| 10 | "Complete Hardware Acceleration Stack" / "warp-level radix trie" / "NVLink 2.8 us on 8 GPUs" | Not built or measured here. What exists: device mask table (`TokenDFATable`), 4-process gloo consensus. | Use the wording in section D. |
| 11 | "100% on 60 tasks" | **Replicated on two further seeds (unchanged code)**: imperative 100% on all 180 tasks and 90/90 delete_dep; M5 100%; same ordering of retry wordings. | Report all three seeds (seed 0 = development, seeds 1-2 = confirmation) and "90/90" with the interval, not "always". Rule-in-prompt: report a range (33.3 / 30.0 / 46.7%), not 33.3%. |
| 12 | Affiliation "AXI Research Initiative", email `r.choubey@axi-runtime.org` | Not verifiable from the repo. | Both authors confirm real affiliation and a contact address they actually own. |
| 13 | "global suite = additions, attribute patches, topology removals" | The generator has four kinds: `delete_dep`, `delete_leaf`, `add_node`, `add_edge` (5:2:1:2). | Describe the real mix. Note: "attribute patches" cannot be expressed in the current delta language at all (verified: `ADD` of an existing node -> `ident_node`; edge-weight change -> `ident_edge`); see `RAJNISH_CONCEPT_MAP.md` row 17. |
| 14 | Synthetic-benchmark scope | One model family, greedy decoding, synthetic graphs, 50% trap tasks by design; "success" encodes the policy that deleting a node also deletes its edges. | Add a Limitations paragraph. |

## C. Drop-in replacement for Theorem 1 (provable)

> **Proposition 1 (syntactic validity does not imply admissibility).**
> Let G = (V, E) be a graph and v in V a node with at least one incident edge. The delta delta = `DEL[v]`
> lies in the language L of the CICO grammar, but it violates the dangling-edge condition of the DPO gluing
> conditions on G. Hence L intersected with Admissible(G) is a strict subset of L whenever E is non-empty,
> and a decoder whose only guarantee is membership in L can emit an inadmissible delta. Applying such a delta
> without a gate leaves edges whose endpoint no longer exists.
>
> *Proof.* `DEL[v]` matches the grammar by construction. Removing v with an incident edge e leaves e without a
> source or target in the result, so the gluing condition (no dangling edges) fails. QED.
>
> **Empirical corollary (measured, not proved).** On the 30 delete-with-dependents tasks, Qwen2.5-3B-Instruct
> (greedy) proposed `DEL[r,c]` without the incident edge deletions in 30/30 cases, so blind application produced a
> corrupted graph 30/30 times, with and without grammar constraint.

## D. Measured numbers to put in the performance section (all Colab T4)

| What | Measured |
|---|---|
| Device mask table vs host round trip, GPT-2 vocab | 287 -> 77 us (B=1, ~4x); 8,415 -> 392 us (B=64, ~21x). Mask = 36-54% of one softmax+sample step. |
| Mask table size, Qwen vocab | 37 states x 151,665 tokens = 16.8 MB |
| Tail rollback (crop), Qwen2.5-0.5B | 813 us (4k), 349 us (16k), 668 us (32k) |
| Full prefill | 683 ms (4k), 3,170 ms (16k), 10,076 ms (32k) |
| Edit at 25% of context, re-prefill the rest | 286 ms (4k), 2,507 ms (16k), 8,937 ms (32k) |
| Block independence, strict mask vs RFC mask | 0.0 vs 95.4 max change downstream |
| Cross-rank all-or-nothing commit (4 gloo processes) | 934 commits / 2,066 aborts / 3,000 steps, identical page tables |
| axi-wire reference codec (Python, no CRC), 10,000 edges | pack 34 us vs json.dumps 9,994 us; unpack 1.9 us vs json.loads 8,279 us |

## E. Checks still open

1. `--seed 1` (and `--seed 2`), with and without `--hint-rules`; the headline numbers are the fresh-seed ones.
2. Why constrained strict-parse is 98.3% in the hint regime (cell below).
3. The 4 non-trap tasks where blind apply was right but the gate rejected (hint regime).

```python
import json
r = json.load(open('/content/axi_llm_results.json'))
rows = r['rows'] if isinstance(r, dict) and 'rows' in r else r
bad = [x for x in rows if not x['M1_constrained'].get('strict_parse', True)]
print(len(bad), "constrained outputs that did not parse")
for x in bad[:5]: print(x['kind'], repr(x['M1_constrained'].get('text'))[:200])
```
(If the key names differ, `print(rows[0].keys())` and adjust.)

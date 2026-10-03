RFC-001: The AXI Architecture
An Axiomatic, Coordinate-Invariant State-Transition Runtime for Foundation Models
� � �
1. Problem Statement & Motivation
Autoregressive transformers serialize multi-dimensional state graphs into 1D token streams governed by monotonic Rotary Positional Embeddings (RoPE). Under long-horizon enterprise workloads, this introduces three systemic failure modes:
Coordinate and Relational Drift: Spatial, relational, and tabular topologies degrade under 1D serialization, causing coordinate hallucinations and off-by-one errors.
The "Reasoning Token Tax": Conventional Chain-of-Thought (CoT) prompting spends hundreds of discursive natural language tokens to compute basic state transitions, inflating P99 inference latency and inference costs.
Stochastic State Corruption: Post-hoc schema validation (e.g., JSON schema regex masks) fails to enforce categorical relational invariants, risking downstream database corruption.
The AXI framework demotes the foundation model to a heuristic transition proposer while delegating verification to a deterministic category-theoretic gatekeeper.
2. Core Architectural Pillars
I. Invariant Coordinate Tuples (axi-ir)
Entities and relationships are represented as permutation-invariant tuples:
Nodes: $\tau_v = (\mathbf{z}_v : \mathbf{x}_v)$
Edges: $\tau_e = (\mathbf{z}_u, \mathbf{z}v, \mathcal{R} : \mathbf{w}{uv})$
Topological identifiers $\mathbf{z}_v$ are derived via normalized graph Laplacians or $k$-step Random Walk Positional Encodings (RWPE), eliminating dependence on sequential token positions.
II. Category-Theoretic Double Pushout (DPO) Gates (axi-dpo)
State transitions are evaluated as categorical rewrite spans $L \stackrel{l}{\hookleftarrow} K \stackrel{r}{\hookrightarrow} R$ over an adhesive category of typed graphs:
Identification Condition: Verifies that distinct pattern elements mapping to the same host element are preserved in interface $K$.
Dangling Edge Condition: Rejects node deletions if external edges remain attached in the host graph without being matched in $L$.
Pushout Complement ($D$): Excises consumed state elements only when boundary conditions hold, blocking invalid state commits before database execution.
III. Topological Paged Memory Management (axi-cache)
Decouples physical GPU KV-cache slots from sequence position:
Topological Block Table: Maps invariant coordinates $\mathbf{z}_v$ to physical VRAM page IDs.
Copy-on-Write Shadow Paging: Candidate transitions write to transient shadow blocks. Passing DPO checks triggers an $\mathcal{O}(1)$ atomic pointer swap; rejections drop shadow blocks with zero rollback recomputation.
3. Active Research Frontiers
This codebase is an architectural reference prototype. Full multi-node production deployment requires addressing the following systems-level challenges:
Causal Attention Factorization: Developing block-diagonal attention masks to allow in-place KV-cache patching without corrupting historical causal attention dependencies.
BPE Tokenizer Prefix Tries: Compiling byte-level Deterministic Finite Automata (DFAs) over Byte-Pair Encoding vocabularies to handle multi-character token merges.
Serving Runtime Integration: Mapping virtual coordinate pages directly into contiguous PagedAttention block structures within vLLM and TensorRT-LLM runtimes.
4. Attribution & Intellectual Lineage
The theoretical origin of this work is Rajnish Choubey, through The Cinderella Project and The Sovereign Manifesto.
5. Build & Test
cargo test --workspace
pip install -e .
python python/tests/test_end_to_end_pipeline.py
"""Block-independence attention masks (numpy reference) + a tiny multi-layer model to test them.

rule 'pasted'  : causal & (same_block | key_is_interface)            (from the RFC kernel)
rule 'strict'  : interface queries see ONLY interface keys (causal, any block);
                 internal queries see own block (causal) + interface keys of earlier blocks.
"""
import numpy as np


def make_mask(block_ids, is_iface, rule):
    b = np.asarray(block_ids); f = np.asarray(is_iface).astype(bool); n = len(b)
    causal = np.tril(np.ones((n, n), bool))
    same = b[:, None] == b[None, :]
    kif = f[None, :]
    if rule == "pasted":
        return causal & (same | kif)
    if rule == "strict":
        q_if = f[:, None]
        return causal & np.where(q_if, kif, same | kif)
    raise ValueError(rule)


class TinyModel:
    """L layers of single-head attention + tanh MLP, residual. No positions (CICO-style), so edits
    never shift anything; only the mask decides who sees whom."""
    def __init__(self, d=16, layers=3, seed=0):
        r = np.random.default_rng(seed); s = d ** -0.5
        self.L = [dict(q=r.normal(0, s, (d, d)), k=r.normal(0, s, (d, d)), v=r.normal(0, s, (d, d)),
                       o=r.normal(0, s, (d, d)), w1=r.normal(0, s, (d, 2 * d)), w2=r.normal(0, s, (2 * d, d)))
                  for _ in range(layers)]
        self.d = d

    def forward(self, x, mask):
        h = x.copy()
        for p in self.L:
            q, k, v = h @ p["q"], h @ p["k"], h @ p["v"]
            sc = np.where(mask, q @ k.T / np.sqrt(self.d), -np.inf)
            sc = sc - sc.max(1, keepdims=True)
            w = np.exp(sc); w /= w.sum(1, keepdims=True)
            h = h + (w @ v) @ p["o"]
            h = h + np.tanh(h @ p["w1"]) @ p["w2"]
        return h

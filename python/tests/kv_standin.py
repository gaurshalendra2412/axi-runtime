"""A small real transformer in numpy (float64), used to test the KV-cache logic of axi/experiments/kv_drop.py without torch and without a GPU. Not a test file (run_all.py skips it).
It is a genuine causal transformer, not a mock: RMS norm, grouped-query attention with rotary position embeddings in the half-split layout that Hugging Face uses, an MLP, tied embeddings,
and a key/value cache whose rows carry the rotation of their position id. Its weights are random, so it has no task skills; what it has is the property the tests need: its output depends on
every visible token and on the positions, so a cache that is subtly wrong or stale gives different logits. The cache implements the protocol described in kv_drop.py."""
import numpy as np

from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker
from axi.engine.gpu_mask import TokenDFATable
from axi.experiments.kv_drop import PromptKit, rope_tables

EOS = 256


def byte_kit(hint_rules: bool = False) -> PromptKit:
    """A tokenizer with one token per byte (token 256 = end of reply): a piece of text is always exactly the concatenation of its parts."""
    return PromptKit(lambda s: list(s.encode("utf-8")), lambda ids: bytes(int(i) for i in ids if i < 256).decode("utf-8", "replace"), EOS, hint_rules)


def byte_table() -> TokenDFATable:
    vocab = [bytes([i]) for i in range(256)] + [b""]; dfa = CICOGrammarDFA(); table = TokenDFATable(dfa, TokenMasker(dfa, vocab, EOS))
    table.allowed[table.start, EOS] = False; table.next[table.start, EOS] = -1          # at least one operation, as in build_hf_llm
    return table


def _rms(x): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def _rot_half(x): h = x.shape[-1] // 2; return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class TinyEngine:
    tol, tol_keys = 1e-8, 1e-8

    def __init__(self, vocab=257, d=32, heads=4, kv_heads=2, layers=3, seed=0, theta=10000.0, eos_bonus=3.0):
        rng = np.random.default_rng(seed); self.d, self.H, self.KV, self.hd, self.layers, self.vocab = d, heads, kv_heads, d // heads, layers, vocab
        self.emb = rng.normal(0, 0.5, (vocab, d)); self.eos_bonus = eos_bonus; self.force = None; self.k = 0; s = 3.0 * d ** -0.5
        self.w = [dict(wq=rng.normal(0, s, (d, heads * self.hd)), wk=rng.normal(0, s, (d, kv_heads * self.hd)), wv=rng.normal(0, s, (d, kv_heads * self.hd)),
                       wo=rng.normal(0, s, (heads * self.hd, d)), w1=rng.normal(0, s, (d, 2 * d)), w2=rng.normal(0, (2 * d) ** -0.5, (2 * d, d))) for _ in range(layers)]
        self.inv_freq = 1.0 / theta ** (np.arange(0, self.hd, 2) / self.hd); self.cache_class = TinyCache; self.calls = 0; self.fail_after = None

    def new_cache(self): return self.cache_class(self)

    def sync(self): pass

    def reset_peak(self): pass

    def peak_mb(self): return 0.0


class TinyCache:
    def __init__(self, e): self.e, self.K, self.V, self.pos = e, [np.zeros((e.KV, 0, e.hd)) for _ in range(e.layers)], [np.zeros((e.KV, 0, e.hd)) for _ in range(e.layers)], []

    @property
    def n(self) -> int: return len(self.pos)

    def forward(self, tokens, positions):
        e = self.e; e.calls += 1
        if e.fail_after is not None and e.calls > e.fail_after: raise RuntimeError("session dropped")
        tokens, positions = list(tokens), [int(p) for p in positions]; L, n0 = len(tokens), self.n; assert L == len(positions) and L > 0
        x = e.emb[tokens]; cos, sin = rope_tables(positions, e.inv_freq)
        for li, w in enumerate(e.w):
            h = _rms(x); q = (h @ w["wq"]).reshape(L, e.H, e.hd); k = (h @ w["wk"]).reshape(L, e.KV, e.hd); v = (h @ w["wv"]).reshape(L, e.KV, e.hd)
            q = q * cos[:, None] + _rot_half(q) * sin[:, None]; k = k * cos[:, None] + _rot_half(k) * sin[:, None]
            self.K[li] = np.concatenate([self.K[li], k.transpose(1, 0, 2)], axis=1); self.V[li] = np.concatenate([self.V[li], v.transpose(1, 0, 2)], axis=1)
            mask = np.arange(n0 + L)[None, :] <= (n0 + np.arange(L))[:, None]; out = np.zeros((L, e.H, e.hd))
            for hh in range(e.H):
                g = hh // (e.H // e.KV); sc = q[:, hh] @ self.K[li][g].T / np.sqrt(e.hd); sc = np.where(mask, sc, -np.inf); sc -= sc.max(-1, keepdims=True); p = np.exp(sc); p /= p.sum(-1, keepdims=True)
                out[:, hh] = p @ self.V[li][g]
            x = x + out.reshape(L, -1) @ w["wo"]; h = _rms(x); x = x + np.tanh(h @ w["w1"]) @ w["w2"]
        self.pos += positions; logits = _rms(x[-1]) @ e.emb.T * 2.0; logits[EOS] += e.eos_bonus
        if e.force is not None:                                              # tests only: make the reply the scripted one (the request ends in a newline; every later call is one token)
            e.k = 0 if L > 1 and tokens[-1] == 10 else e.k + 1
            if e.k < len(e.force): logits[e.force[e.k]] += 100.0
        return logits

    def forward_blocks(self, blocks, positions, visible=None):
        """Read several blocks side by side (the protocol of kv_drop.py): each block sees the first `visible` rows of the cache (default: all rows) and itself, no other block; then all are appended in order."""
        e, v0, k0 = self.e, self.n if visible is None else int(visible), self.e.k; new_k, new_v, new_pos = [[] for _ in self.K], [[] for _ in self.V], []
        for toks, pos in zip(blocks, positions):
            tmp = TinyCache(e); tmp.K, tmp.V, tmp.pos = [k[:, :v0].copy() for k in self.K], [v[:, :v0].copy() for v in self.V], list(self.pos[:v0]); tmp.forward(toks, pos)
            for li in range(len(self.K)): new_k[li].append(tmp.K[li][:, v0:]); new_v[li].append(tmp.V[li][:, v0:])
            new_pos += [int(p) for p in pos]
        e.k = k0                                                             # tests only: the scripted-reply counter is not moved by reading blocks
        self.K = [np.concatenate([k] + nk, axis=1) for k, nk in zip(self.K, new_k)]; self.V = [np.concatenate([v] + nv, axis=1) for v, nv in zip(self.V, new_v)]; self.pos += new_pos

    def select(self, rows):
        rows = list(rows); self.K = [k[:, rows] for k in self.K]; self.V = [v[:, rows] for v in self.V]; self.pos = [self.pos[i] for i in rows]

    def rerotate(self, new_pos):
        new_pos = [int(p) for p in new_pos]; assert len(new_pos) == self.n; cos, sin = rope_tables(np.asarray(new_pos) - np.asarray(self.pos), self.e.inv_freq)
        self.K = [k * cos[None] + _rot_half(k) * sin[None] for k in self.K]; self.pos = new_pos

    def nbytes(self) -> int: return sum(k.size * 8 + v.size * 8 for k, v in zip(self.K, self.V))

    def keys(self, layer): return self.K[layer].copy()

    def value_deviation(self, other, rows):
        d = [np.linalg.norm(a[:, :rows] - b[:, :rows]) / max(np.linalg.norm(b[:, :rows]), 1e-300) for a, b in zip(self.V, other.V)]; return float(np.mean(d))

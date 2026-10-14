"""The PyTorch / Hugging Face side of the KV test (docs/KV_DROP_TEST.md). Everything that decides what the test means lives in `kv_drop.py`, which has no torch in it and is tested on a numpy model;
this file only gives the same small cache protocol to a real model. It cannot run without torch, so the notebook first runs `kv_drop.preflight` on the real model (read in two pieces = read at once;
keys turned by N positions = keys read at those positions; cut rows leave no trace) and stops if any line fails.

Design choices that keep it independent of the Hugging Face version:
  * the cache is OUR list of (keys, values) per layer, each [1, kv_heads, rows, head_dim], plus the list of position ids. For every forward call a fresh `DynamicCache` is filled through its public
    `update` method and the result is read back; nothing inside the Hugging Face cache is edited.
  * the base model (`model.model`) is called with explicit `position_ids`; only the last token goes through `lm_head` (no [rows x vocabulary] logits in memory).
  * turning keys is the same formula as Hugging Face's `apply_rotary_pos_emb` (`k * cos + rotate_half(k) * sin`), done in float32 and stored back in the cache's dtype; the angles come from the model's
    own `inv_freq` if it has one."""
import numpy as np

from axi.experiments.kv_drop import KV_MODEL, Item, PromptKit, rope_tables


def _tensors(cache, i):
    layers = getattr(cache, "layers", None)
    if layers is not None and hasattr(layers[i], "keys") and layers[i].keys is not None: return layers[i].keys, layers[i].values
    if hasattr(cache, "key_cache"): return cache.key_cache[i], cache.value_cache[i]
    kv = cache[i]; return kv[0], kv[1]


class TorchCache:
    def __init__(self, engine): self.e, self.kv, self.pos = engine, None, []

    @property
    def n(self) -> int: return len(self.pos)

    def _hf(self):
        from transformers import DynamicCache
        c = DynamicCache()
        if self.kv is not None:
            for i, (k, v) in enumerate(self.kv): c.update(k, v, i)
        return c

    def forward(self, tokens, positions):
        t, e = self.e.torch, self.e; ids = t.tensor([list(tokens)], device=e.device); pos = t.tensor([[int(p) for p in positions]], device=e.device); c = self._hf()
        with t.no_grad():
            out = e.model.model(input_ids=ids, position_ids=pos, past_key_values=c, use_cache=True); logits = e.model.lm_head(out.last_hidden_state[:, -1])
        c = out.past_key_values if getattr(out, "past_key_values", None) is not None else c
        self.kv = [_tensors(c, i) for i in range(e.layers)]; self.pos += [int(p) for p in positions]
        return logits[0].float().cpu().numpy()

    def select(self, rows):
        rows = list(rows); t = self.e.torch
        if rows == list(range(len(rows))): m = len(rows); self.kv = [(k[:, :, :m], v[:, :, :m]) for k, v in self.kv]
        else: ix = t.tensor(rows, device=self.e.device); self.kv = [(k.index_select(2, ix), v.index_select(2, ix)) for k, v in self.kv]
        self.pos = [self.pos[i] for i in rows]

    def rerotate(self, new_pos):
        new_pos = [int(p) for p in new_pos]; assert len(new_pos) == self.n, (len(new_pos), self.n)
        delta = np.asarray(new_pos) - np.asarray(self.pos)
        if not delta.any(): return
        t, e = self.e.torch, self.e; cos, sin = rope_tables(delta, e.inv_freq); cos = t.from_numpy(cos).float().to(e.device)[None, None]; sin = t.from_numpy(sin).float().to(e.device)[None, None]; h = cos.shape[-1] // 2

        def rot(k):
            kf = k.float(); return (kf * cos + t.cat((-kf[..., h:], kf[..., :h]), dim=-1) * sin).to(k.dtype)
        self.kv = [(rot(k), v) for k, v in self.kv]; self.pos = new_pos

    def nbytes(self) -> int: return 0 if self.kv is None else sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.kv)

    def keys(self, layer): return self.kv[layer][0][0].float().cpu().numpy()

    def value_deviation(self, other, rows):
        d = []
        for (_, a), (_, b) in zip(self.kv, other.kv):
            a, b = a[:, :, :rows].float(), b[:, :, :rows].float(); d.append(float((a - b).norm() / b.norm().clamp_min(1e-30)))
        return sum(d) / len(d)


class TorchEngine:
    tol, tol_keys = 0.5, 0.05           # fp16 with 4-bit weights. tol: allowed largest difference of a raw logit (they run to about 30). tol_keys: allowed size of the difference of the keys relative to the size of the keys

    def __init__(self, model):
        import torch
        self.torch, self.model, self.device, self.layers = torch, model, next(model.parameters()).device, model.config.num_hidden_layers
        rot = getattr(model.model, "rotary_emb", None); inv = getattr(rot, "inv_freq", None)
        if inv is not None:
            assert float(getattr(rot, "attention_scaling", 1.0)) == 1.0, "this model scales its rotary embedding; the key rotation here does not"
            self.inv_freq = inv.detach().float().cpu().numpy()
        else:
            cfg = model.config; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]; self.inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))

    def new_cache(self): return TorchCache(self)

    def sync(self): self.torch.cuda.synchronize()

    def reset_peak(self): self.torch.cuda.reset_peak_memory_stats()

    def peak_mb(self) -> float: return self.torch.cuda.max_memory_allocated() / 1e6


def make_hf_kit(tok, hint_rules: bool = False) -> PromptKit:
    """The prompt kit for a ChatML tokenizer (Qwen2.5), checked against the tokenizer's own chat template."""
    kit = PromptKit(lambda s: tok.encode(s, add_special_tokens=False), lambda ids: tok.decode(list(ids), skip_special_tokens=True), tok.eos_token_id, hint_rules)
    instr = "Delete the service at (0,0)."
    want = tok.apply_chat_template([{"role": "system", "content": kit.system}, {"role": "user", "content": "<CICO_STATE> [0,0:1] </CICO_STATE>\n\nRequest: " + instr}], add_generation_prompt=True, tokenize=False)
    got = kit.whole_text([Item(("n", (0, 0)), 1)], instr)
    assert want == got, "this tokenizer's chat template is not the ChatML text the test builds; first difference at character %d" % next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
    return kit


def build_kv(name: str = KV_MODEL, load_4bit: bool = True, device: str = "cuda"):
    """(engine, tokenizer, grammar table, model). The grammar table is the one `build_hf_llm` builds (an end token is not allowed before the first item)."""
    from colab.colab_llm_experiment import build_hf_llm
    llm, model = build_hf_llm(name, device, load_4bit); return TorchEngine(model), llm.tok, llm.table, model

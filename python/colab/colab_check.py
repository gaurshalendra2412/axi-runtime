"""Colab checks (needs internet + torch; GPU optional).  Run from repo root:  python python/colab/colab_check.py
A) CICO decoder against a REAL BPE vocabulary (GPT-2 byte-level BPE via transformers)
B) Block-independence mask in PyTorch (GPU if present): edit a scratch token, compare other blocks
"""
import os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from axi.engine.cico_parser import CICOParser
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker, sample_delta


def bytes_to_unicode():
    """GPT-2's reversible byte<->unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def vocab_bytes_from_tokens(token_strings, eos_id=None):
    """token string -> raw bytes via GPT-2 table. Unmappable/special tokens become b'' (never allowed)."""
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = []
    for i, t in enumerate(token_strings):
        try:
            out.append(b"" if i == eos_id else bytes(inv[ch] for ch in t))
        except KeyError:
            out.append(b"")
    return out


def check_real_tokenizer(name="gpt2"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    n = len(tok)
    strings = [""] * n
    for t, i in tok.get_vocab().items():
        if i < n: strings[i] = t
    eos = tok.eos_token_id
    vocab = vocab_bytes_from_tokens(strings, eos)
    usable = sum(1 for v in vocab if v)
    print(f"[A] vocab={n} usable_byte_tokens={usable} eos={eos}")
    dfa = CICOGrammarDFA(); t0 = time.time(); m = TokenMasker(dfa, vocab, eos)
    print(f"[A] trie built in {time.time()-t0:.2f}s")
    t0 = time.time(); sizes = {}
    for q in dfa.states(): sizes[q] = len(m.allowed_token_ids(q))
    print(f"[A] all {len(sizes)} state masks in {time.time()-t0:.2f}s; START allows {sizes['START']}, NA_r allows {sizes['NA_r']}")
    # trie masks must equal naive masks on a real vocab (sample of states to bound time)
    for q in ["START", "AO", "NA_r", "E_rel", "SEP", "E_w"]:
        assert m.allowed_token_ids(q) == m.allowed_token_ids_naive(q), q
    print("[A] trie == naive masks: OK")
    # real tokenizer's own tokenization of valid strings must be allowed token by token
    valid = ["ADD[1,2:3]", "DEL[4,5]", "ADD[(1,2)->(3,4):owns#5] DEL[(0,0)->(1,1):dep_on#-2]", "ADD[12,345:-678]\nDEL[7,8]"]
    for s in valid:
        st = "START"
        for tid in tok.encode(s):
            assert tid in m.allowed_token_ids(st), (s, tid, tok.decode([tid]))
            st = m.advance(st, tid)
        assert dfa.is_accepting(st)
    print("[A] real BPE encodings of valid deltas are all allowed: OK")
    rng = random.Random(0); okc = 0
    for _ in range(500):
        s, ids = sample_delta(m, rng, max_tokens=30)
        CICOParser.parse_transition_delta(s); okc += 1
    print(f"[A] {okc}/500 sampled deltas parse with the strict parser: OK")
    blocked = [w for w in ["Sure", "The", "```", "{"] if tok.encode(w)[0] not in m.allowed_token_ids("START")]
    print("[A] prose/JSON openers blocked at START:", blocked)
    return m


def check_block_independence_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[B] device={dev}")
    nb, bs, d, layers = 8, 32, 64, 6
    N = nb * bs
    blk = torch.arange(nb).repeat_interleave(bs)
    pos = torch.arange(bs).repeat(nb)
    iface = (pos >= bs - 4)                                 # last 4 tokens of every block = interface
    ar = torch.arange(N)
    causal = ar[:, None] >= ar[None, :]
    same = blk[:, None] == blk[None, :]
    kif = iface[None, :]
    masks = {"pasted": causal & (same | kif),
             "strict": causal & torch.where(iface[:, None], kif, same | kif)}
    g = torch.Generator().manual_seed(0)
    W = [{k: torch.randn(d, d, generator=g) * d ** -0.5 for k in "qkvo"} |
         {"w1": torch.randn(d, 2 * d, generator=g) * d ** -0.5, "w2": torch.randn(2 * d, d, generator=g) * (2 * d) ** -0.5} for _ in range(layers)]
    W = [{k: v.to(dev) for k, v in w.items()} for w in W]

    def fwd(x, mask):
        h = x
        for w in W:
            q, k, v = h @ w["q"], h @ w["k"], h @ w["v"]
            s = (q @ k.T / d ** 0.5).masked_fill(~mask, float("-inf"))
            h = h + (torch.softmax(s, -1) @ v) @ w["o"]
            h = h + torch.tanh(h @ w["w1"]) @ w["w2"]
        return h

    x = torch.randn(N, d, generator=g).to(dev); res = {}
    for rule, mask in masks.items():
        mask = mask.to(dev); a = fwd(x, mask)
        worst = 0.0
        for edit in (3, 20, bs + 5, 3 * bs + 10):           # scratch tokens in blocks 0,0,1,3
            x2 = x.clone(); x2[edit] += torch.randn(d, generator=g).to(dev) * 5
            diff = (fwd(x2, mask) - a).abs()
            later = (blk.to(dev) > blk[edit].item())
            worst = max(worst, diff[later].max().item())
        res[rule] = worst
        print(f"[B] {rule:6s}: max change in LATER blocks after editing a scratch token = {worst:.3e}")
    assert res["strict"] == 0.0, "strict rule must be bit-exact"
    assert res["pasted"] > 1e-3, "pasted rule was expected to leak"
    print("[B] strict rule bit-exact on", dev, "; pasted rule leaks: OK")


if __name__ == "__main__":
    try:
        check_real_tokenizer()
    except ImportError as e:
        print("[A] skipped:", e, "-> pip install transformers")
    check_block_independence_torch()

import random
from colab.colab_check import bytes_to_unicode, vocab_bytes_from_tokens
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker


def test_bytes_to_unicode_is_bijection_and_roundtrips():
    t = bytes_to_unicode(); assert len(t) == 256 and len(set(t.values())) == 256
    inv = {c: b for b, c in t.items()}
    s = "ADD[1,2:3] DEL[(0,0)->(1,1):x#-2]\n"
    enc = "".join(t[b] for b in s.encode())            # GPT-2 style: space -> 'Ġ', newline -> 'Ċ'
    assert "Ġ" in enc and "Ċ" in enc
    assert bytes(inv[c] for c in enc).decode() == s


def test_gpt2_style_vocab_gives_correct_masks():
    t = bytes_to_unicode()
    enc = lambda s: "".join(t[b] for b in s.encode())
    toks = [enc(x) for x in ["ADD[", "DEL[", " ADD[", "\nDEL[", "12", "Sure", "the", "],", ")->(", "#-"]] + [t[i] for i in range(256)]
    toks.append("<|endoftext|>")                       # special: contains chars outside the table -> b''
    vocab = vocab_bytes_from_tokens(toks, eos_id=len(toks) - 1)
    assert vocab[-1] == b"" and vocab[2] == b" ADD["
    dfa = CICOGrammarDFA(); m = TokenMasker(dfa, vocab, len(toks) - 1)
    allowed = {vocab[i] for i in m.allowed_token_ids("START") if vocab[i]}
    assert b"ADD[" in allowed and b"Sure" not in allowed and b" ADD[" not in allowed   # no leading space at START
    after = {vocab[i] for i in m.allowed_token_ids("END") if vocab[i]}
    assert b" " in after and b"\n" in after
    for q in dfa.states(): assert m.allowed_token_ids(q) == m.allowed_token_ids_naive(q)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

import random
import numpy as np
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker, build_vocab
from axi.engine.cico_parser import CICOParser
from axi.engine.gpu_mask import TokenDFATable, decode_numpy

DFA = CICOGrammarDFA(); VOCAB, EOS = build_vocab(); M = TokenMasker(DFA, VOCAB, EOS); T = TokenDFATable(DFA, M)


def test_table_equals_cpu_masks_and_transitions():
    for name, s in T.idx.items():
        assert set(np.flatnonzero(T.allowed[s])) == set(M.allowed_token_ids(name)), name
        for t in M.allowed_token_ids(name):
            if t != EOS: assert T.names[T.next[s, t]] == M.advance(name, t)
        assert ((T.next[s] >= 0) == T.allowed[s]).all()
    print("   table:", len(T.names), "states x", T.V, "tokens =", T.nbytes() // 1024, "KiB")


def test_decode_loop_identical_to_cpu_masker_loop():
    rng = np.random.default_rng(0); B, steps = 16, 45
    L = rng.normal(size=(steps, B, T.V)); L[:, :, EOS] += 4.0              # nudge EOS so rows finish
    toks, state = decode_numpy(T, VOCAB, lambda t: L[t], steps, B)
    for b in range(B):
        st, seq, done = "START", [], False
        for t in range(steps):
            if done: seq.append(EOS); continue
            allowed = M.allowed_token_ids(st); row = np.full(T.V, -np.inf); row[list(allowed)] = L[t, b, list(allowed)]
            tok = int(row.argmax()); seq.append(tok)
            if tok == EOS: done = True
            else: st = M.advance(st, tok)
        assert seq == list(toks[b]), b


def test_every_finished_row_parses_and_unfinished_rows_are_valid_prefixes():
    rng = np.random.default_rng(1); B, steps = 64, 50
    L = rng.normal(size=(steps, B, T.V)); L[:, :, EOS] += 5.0
    toks, state = decode_numpy(T, VOCAB, lambda t: L[t], steps, B); fin = 0
    assert (state >= 0).all()
    for b in range(B):
        ids = list(toks[b]); done = EOS in ids
        s = b"".join(VOCAB[t] for t in ids if t != EOS).decode()
        if done: CICOParser.parse_transition_delta(s); fin += 1
        else: assert DFA.run(s.encode()) is not None                      # still a live prefix, can be completed
    assert fin > 10


def test_table_is_small():
    assert T.nbytes() < 5_000_000


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

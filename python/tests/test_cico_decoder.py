import random
from axi.engine.cico_parser import CICOParser, CICOParseError
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker, build_vocab, sample_delta

DFA = CICOGrammarDFA()
VOCAB, EOS = build_vocab()
MASK = TokenMasker(DFA, VOCAB, EOS)

VALID = ["", "ADD[1,2:3]", "ADD[0,0:-7]", "DEL[4,5]", "ADD[(1,2)->(3,4):owns#5]",
         "DEL[(0,0)->(1,1):dep_on#-2]", "ADD[1,2:3] DEL[4,5]\nDEL[(0,0)->(1,1):a#1]", "ADD[1,2:3] ", "DEL[1,1]\n"]
INVALID = ["ADD[1,2]", "DEL[1,2:3]", "ADD[1,2:3]]", "Sure! ADD[1,2:3]", "ADD[1,2:3]  DEL[1,1]", " ADD[1,2:3]",
           "ADD[(1,2)->(3,4):#5]", "ADD[(1,2)->(3,4):x#]", "ADD[1,2:--3]", "add[1,2:3]", "ADD[1,2:3]DEL[1,1]",
           "ADD[١,2:3]", "ADD[1,2:3]\n\n"]


def parses(s):
    try:
        CICOParser.parse_transition_delta(s); return True
    except CICOParseError:
        return False


def test_dfa_accepts_and_rejects():
    for s in VALID: assert DFA.accepts(s.encode()), s
    for s in INVALID: assert not DFA.accepts(s.encode()), s


def test_dfa_and_parser_agree_on_fixed_cases():
    for s in VALID + INVALID:
        assert DFA.accepts(s.encode("utf-8")) == parses(s), s


def test_dfa_vs_parser_fuzz():
    rng = random.Random(1)
    alpha = "ADEL[]()-:#>,0123456789 \nxy_"
    seeds = VALID[1:7]
    agree = acc = 0
    for _ in range(30000):
        s = rng.choice(seeds)
        s = list(s)
        for _ in range(rng.randint(0, 3)):
            op = rng.random(); i = rng.randrange(len(s) + 1)
            if op < .4: s.insert(i, rng.choice(alpha))
            elif op < .7 and s: s.pop(min(i, len(s) - 1))
            elif s: s[min(i, len(s) - 1)] = rng.choice(alpha)
        s = "".join(s)
        a = DFA.accepts(s.encode()); p = parses(s)
        assert a == p, repr(s)
        agree += 1; acc += a
    assert acc > 500, "fuzz never produced valid strings; test is vacuous"


def test_trie_masks_equal_naive_masks():
    for q in DFA.states():
        assert MASK.allowed_token_ids(q) == MASK.allowed_token_ids_naive(q), q


def test_mask_blocks_prose_at_start():
    allowed = {VOCAB[t] for t in MASK.allowed_token_ids("START") if t != EOS}
    assert b"Sure" not in allowed and b"```" not in allowed and b"the" not in allowed
    assert b"ADD[" in allowed and b"DEL[" in allowed and EOS in MASK.allowed_token_ids("START")


def test_every_state_can_finish():
    for q in DFA.states():
        assert DFA.distance_to_accept(q) is not None, q
        assert MASK.allowed_token_ids(q), q


def test_sampled_deltas_always_parse():
    rng = random.Random(7)
    nonempty = multi = 0
    for _ in range(3000):
        s, ids = sample_delta(MASK, rng)
        d = CICOParser.parse_transition_delta(s)  # must not raise
        assert DFA.accepts(s.encode())
        if s: nonempty += 1
        if len(d.node_deltas) + len(d.edge_deltas) > 1: multi += 1
    assert nonempty > 1000 and multi > 100


def test_forced_completion_from_every_state():
    for q in DFA.states():
        st = q; n = 0
        while not DFA.is_accepting(st):
            st = DFA.step(st, DFA.completion_byte(st)); n += 1
            assert n < 50
        assert n == DFA.distance_to_accept(q)


def test_any_tokenization_of_valid_string_is_allowed():
    rng = random.Random(3)
    by_bytes = {}
    for i, t in enumerate(VOCAB):
        if i != EOS: by_bytes.setdefault(t, i)
    for s in VALID[1:]:
        b = s.encode()
        for _ in range(50):
            i, st = 0, "START"
            while i < len(b):
                cands = [n for n in range(1, min(8, len(b) - i) + 1) if b[i:i + n] in by_bytes]
                n = rng.choice(cands)
                tid = by_bytes[b[i:i + n]]
                assert tid in MASK.allowed_token_ids(st), (s, b[i:i + n])
                st = MASK.advance(st, tid); i += n
            assert EOS in MASK.allowed_token_ids(st)


def test_advance_rejects_bad_token_and_early_eos():
    prose = VOCAB.index(b"Sure")
    for f in (lambda: MASK.advance("START", prose), lambda: MASK.advance("A1", EOS)):
        try: f()
        except ValueError: continue
        raise AssertionError("should have raised")


def test_logit_mask():
    m = MASK.logit_mask("START")
    assert len(m) == len(VOCAB)
    assert {i for i, x in enumerate(m) if x == 0.0} == set(MASK.allowed_token_ids("START"))


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)

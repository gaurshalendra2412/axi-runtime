"""Implements concept-map row 138 (thirteenth scan): the 5 Sep page with the numeric slug 4824 (Post 377, about 430 words, two reads; a WordPress post number, not explained on the
page). Quotes verified on the second read (YES): "English Alphabetic Base: 26 letters (5 vowels, 21 consonants)."; "Devanagari Varnamala Base: 49 core phonetic units (13 Swaras/vowels, 36
Vyanjanas/consonants)"; "N^L, where N is the character set size and L is word length"; "combinatorial ceiling (N^L ...)"; the counts for lengths 3, 4, 5: 17,576 vs 117,649; 456,976 vs
5,764,801; 11,881,376 vs 282,475,249 (all six are right, tests/test_blog_arithmetic_2.py). The page's claim is that a bigger set of units (N) gives a bigger ceiling (N^L) of strings.

The reading being tested (Rajnish to accept or reject): N^L counts EVERY string of length L, words or not; what a decoder can actually emit under a grammar is a tiny part of it, and the part is
set by the grammar, not by N. The repository has exactly such a grammar: the byte-level DFA for the delta language (`axi/engine/cico_decoder.py`). Counted by a transfer matrix over its
transitions (strings that never hit the reject sink), checked against explicit enumeration of the transition table and, for complete strings, against the real parser:
  1. Valid prefixes by length L = 1, 2, 3, ... 8 are 2, 2, 2, 2, 22, 240, 2,620, 28,600 against 256^L byte strings: at length 3 only "ADD" and "DEL" are valid (2 of 16,777,216, one in 8,388,608);
     that is fewer than the 17,576 three-letter strings of the page's 26-letter set and the 117,649 of its 49-unit set, whatever N is. Exhaustively, for every valid prefix up to length 6, each of the
     256 possible next bytes is accepted or rejected as the transition table says (`run()` and the table are the same code, so this checks the enumeration, not the grammar).
  2. COMPLETE strings (the DFA accepts them; the parser agrees on all of the ones listed here): none of length 1..7 (the empty string is accepted; the parser rejects every valid prefix up to length 7), 100 of length 8 ("DEL[r,c]" with one digit each: 10 x 10), 2,200 of length 9 (two-digit r or c, 2 x 1,000 = 2,000, or
     "DEL[r,c]" plus one separator, 100 x 2 = 200).
  3. The alphabet used is 74 byte values, all ASCII (largest 122, the letter z): 28.9% of the 256 values appear on any transition. The largest number of bytes allowed in any state is 64 (inside a relation
     name: 63 name characters and the '#' that ends it); the number of valid prefixes grows by a factor between 10 and 11.5 per byte from length 8 to 11 (digit positions), not by 256, 26 or 49.
What they do NOT show: that the page means a grammar when it says "ceiling" (it uses N^L to argue about letters and sounds); anything about a model's token choices (a BPE vocabulary changes the counts per
token, not the bytes; `test_vocab_bytes.py` and `test_cico_decoder.py` hold those); that fewer valid strings means a better language. Counting strings by transfer matrix is standard; constrained
decoding is cited work (CREDITS.md).
The fourth check is ENGINEERING SUPPORT (no page): the strict parser and the decoder agree on every one-byte extension of the 2,300 shortest complete strings, added after a mutation of the parser survived."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_decoder import CICOGrammarDFA
from axi.engine.cico_parser import CICOParser, CICOParseError


def parses(text):
    try:
        CICOParser.parse_transition_delta(text); return True
    except CICOParseError:
        return False


def counts(dfa, upto):
    """[(valid prefixes, complete strings) for L = 1..upto] by a transfer matrix over the DFA's transitions."""
    cnt, out = {dfa.START: 1}, []
    for _ in range(upto):
        nxt = {}
        for s, c in cnt.items():
            for b, t in dfa.T[s].items(): nxt[t] = nxt.get(t, 0) + c
        cnt = nxt
        out.append((sum(cnt.values()), sum(c for s, c in cnt.items() if s in dfa.accepting)))
    return out


def valid_prefixes(dfa, L):
    layer = [(b"", dfa.START)]
    for _ in range(L):
        layer = [(p + bytes([b]), t) for p, s in layer for b, t in dfa.T[s].items()]
    return layer


def test_valid_prefixes_by_length_match_explicit_enumeration_and_only_add_and_del_are_valid_at_length_3():
    dfa = CICOGrammarDFA()
    table = counts(dfa, 9)
    assert [v for v, _ in table[:8]] == [2, 2, 2, 2, 22, 240, 2620, 28600]
    for L in range(1, 9): assert len(valid_prefixes(dfa, L)) == table[L - 1][0]          # the transfer matrix agrees with listing every string
    assert {p for p, _ in valid_prefixes(dfa, 3)} == {b"ADD", b"DEL"}
    assert 256 ** 3 // 2 == 8_388_608 and 2 < 17_576 < 117_649                          # fewer than the page's three-letter strings, for either alphabet
    exhaustive = 0
    for L in range(0, 7):                                                                  # every possible next byte, tested against the real run()
        for p, s in valid_prefixes(dfa, L):
            ok = {b for b in range(256) if dfa.run(p + bytes([b])) is not None}
            assert ok == set(dfa.T[s]), (p, s)
            exhaustive += 256
    assert exhaustive == 256 * (1 + 2 + 2 + 2 + 2 + 22 + 240)
    two = [bytes([a, b]) for a in range(256) for b in range(256) if dfa.run(bytes([a, b])) is not None]
    assert sorted(two) == [b"AD", b"DE"]


def test_complete_strings_start_at_length_8_with_100_and_2200_at_length_9():
    dfa = CICOGrammarDFA()
    table = counts(dfa, 9)
    assert [c for _, c in table[:7]] == [0] * 7 and table[7][1] == 100 == 10 * 10 and table[8][1] == 2200 == 2 * 1000 + 100 * 2
    assert dfa.accepts(b"") and not dfa.accepts(b"DEL[0,0") and dfa.accepts(b"DEL[0,0]") and dfa.accepts(b"DEL[0,0] ") and dfa.accepts(b"DEL[0,0]\n")
    done = [p for p, s in valid_prefixes(dfa, 8) if s in dfa.accepting]
    assert len(done) == 100 and all(p.startswith(b"DEL[") and p.endswith(b"]") for p in done)       # the shortest complete strings are all node deletions
    assert all(parses(p.decode()) for p in done)                                         # the real parser accepts every one of the 100
    short = [p for L in range(1, 8) for p, _ in valid_prefixes(dfa, L)]
    assert len(short) == 2 + 2 + 2 + 2 + 22 + 240 + 2620 and not any(parses(p.decode()) for p in short)      # and none of the shorter valid prefixes is a complete delta
    assert min(len(b"ADD[0,0:0]"), len(b"DEL[0,0]")) == 8 and dfa.accepts(b"ADD[0,0:0]") and len(b"ADD[0,0:0]") == 10


def test_the_alphabet_is_74_ascii_bytes_the_widest_state_allows_64_and_growth_per_byte_is_about_11():
    dfa = CICOGrammarDFA()
    alphabet = set()
    for tr in dfa.T.values(): alphabet |= set(tr)
    assert len(alphabet) == 74 and max(alphabet) == 122 == ord("z") and all(b < 128 for b in alphabet)
    assert round(100 * 74 / 256, 1) == 28.9
    widest = max(len(tr) for tr in dfa.T.values())
    assert widest == 64 and [s for s, tr in dfa.T.items() if len(tr) == widest] == ["E_rel"]
    table = counts(dfa, 12)
    ratios = [table[i + 1][0] / table[i][0] for i in range(7, 11)]
    assert all(10 < r < 11.5 for r in ratios)
    assert table[11][0] < 2 * 64 ** 11 and table[11][0] / 256 ** 12 < 1e-20             # at length 12 the valid share of all byte strings is below one in 10^20


def test_the_parser_and_the_decoder_agree_on_every_one_byte_extension_of_every_short_complete_string():
    """Found by a mutation of the parser: a trailing-whitespace change in the node-DEL pattern survived all 60 test files, because no test appended a stray byte to a complete string. Here every complete string of length 8 and 9 (2,300
    strings) and nine hand-made ones of other shapes are extended by each of the 256 byte values; the strict parser accepts the result exactly when the byte-level DFA does."""
    dfa = CICOGrammarDFA()
    base = [p for L in (8, 9) for p, st in valid_prefixes(dfa, L) if st in dfa.accepting]
    assert len(base) == 100 + 2200
    base += [s.encode() for s in ["", "ADD[0,0:0]", "ADD[10,2:1]", "DEL[12,3]", "ADD[(0,0)->(0,1):r#1]", "DEL[(1,2)->(2,2):adj_x#3]", "ADD[0,0:1] DEL[1,1]", "ADD[0,0:1]\nDEL[1,1]", "ADD[0,0:1] "]]
    checked = accepted = 0
    for s in base:
        for b in range(256):
            t = s + bytes([b])
            text = t.decode("latin-1")
            p, d = parses(text), dfa.accepts(text.encode("utf-8"))
            assert p == d, (t, p, d)
            checked += 1; accepted += p
    assert checked == 256 * (2300 + 9)
    closed = [s for s in base if s == b"" or s[-1:] in (b" ", b"\n")]       # strings that already end in their one allowed separator (or are empty) accept no further byte
    assert len(closed) == 202 and accepted == 2 * (len(base) - len(closed)) # every other string takes exactly a space or a newline, nothing else
    assert not parses("DEL[0,0]\t") and not parses("DEL[0,0]\r") and not parses("DEL[0,0]\x0b") and not parses("DEL[0,0] \t")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

"""Implements concept-map rows 124 and 130 (twelfth scan): blog 7 Aug, "Here is a brief introduction to each of the seven AI apps, focusing on the depth of their logo design and the background of their CEOs"
(Post 298; categories brown, durga; about 1,900 to 2,000 words; byline rajnish choubey; reads as an AI assistant's reply, it opens "You are absolutely right", which proves nothing either way; two reads).
The second read, asked only about contradictions, returned YES three times, with these quotes (verbatim, each under 125 characters):
  * Google/Gemini's logo. Section "Symbolic anatomy", item 4: "It is a four-pointed sparkle, fully open and blasting energy in all directions." Section "Logo & Design Philosophy", item 3:
    "The iconic "G" now features brighter hues and a gradient design."
  * Claude/Anthropic's logo. "Symbolic anatomy", item 2: "Often represented by a clean, diagonal forward slash / or a soft, open crescent." "Logo & Design Philosophy", item 1:
    "Its logo is a simple typographic wordmark with a forward slash, which references "the code that underlies AI"." (The second read called this one "mildly" a contradiction: the two could be reconciled.)
  * Colour. Section "The Psychological Verdict": "Every single one of them avoided pure red (danger/error) and pure yellow (anxiety/chaos)." Yet Google's colours, "Symbolic anatomy" item 4:
    "Multi-color Gradient (Blue, Red, Yellow, Green)."
So one page of about 2,000 words says two different things about the same logo twice and breaks its own rule once. That is drift inside one document, the kind a reader of a long AI reply meets.

The same machinery also settles concept-map row 130 (Post 278, 31 Aug, BLACK, byline rajnish choubey, about 250 words, ONE read, so the quotes are not verified): "They want a neat little organizational
chart-mother here, wife there, daughter somewhere else" is the page's complaint about sorting Kali into family roles; she "is the mother because ...; she is the wife because ...; she is the terrifying destroyer
because ...". A chart gives each person ONE role (a functional relation `role`); the page wants one node to hold several relations to the same node at once. Both are expressible, and the gate admits the second
without any change, because an edge is keyed by (from, to, relation name): three relation names between one pair are three different edges (the last test below).

The reading being tested (Rajnish to accept or reject): once each claim is an edge (subject -> object, named by a relation), a contradiction is a checkable pattern, not a feeling.
  1. FUNCTIONAL relation: "a logo has one shape" means at most ONE outgoing `logo_shape` edge per node. Two such edges from one node are a conflict.
  2. FORBIDDEN value: the page's own rule "no app uses red or yellow" means no `colour` edge may point at Red or Yellow.
In OUR gate neither is checked. `check` tests only match, identification, dangling edge and well-formedness (axi/engine/gate.py), so both Google logo claims and the red/yellow claims are admitted
(this repeats PAPER_CORRECTIONS item 20: the gate checks that a delta applies, not that the task's rules hold). The tests below hand-encode the verified quotes as edges and show:
  * unguarded, the gate admits every claim, contradictions included;
  * with a functional-edge guard written here (about ten lines, NOT part of the engine), exactly the two later logo claims are refused, with a reason, and the refused claim leaves the graph unchanged;
  * the forbidden-value rule catches the red and yellow claims, which the functional guard does not: they are two different kinds of rule.
What they do NOT show: that a model can be made to emit such edges (the claims here are encoded by hand from quotes; extracting them from prose is a separate, unsolved step); how often models contradict
themselves (one page is an example, not a rate); that the Claude pair is a real contradiction (a human may call it a rephrasing: a functional rule over free-text values flags any two different strings, so
it is only as good as the extraction that makes values canonical); that the engine should get this guard (whether it should is the question for Rajnish)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, GateResult, check, apply
from test_signed_slots import etext, same, copy

FUNCTIONAL = {"logo_shape"}                                                # a logo has one shape
FORBIDDEN = {"colour": {"Red", "Yellow"}}                                    # the page's own rule: nobody uses pure red or pure yellow

# (section of the page, subject, relation, object), in the order the sections appear on the page. Hand-encoded from the quotes above.
CLAIMS = [
    ("Symbolic anatomy 2",       "Claude", "logo_shape", "slash_or_crescent"),
    ("Symbolic anatomy 4",       "Google", "logo_shape", "four_pointed_sparkle"),
    ("Symbolic anatomy 4",       "Google", "colour",     "Blue"),
    ("Symbolic anatomy 4",       "Google", "colour",     "Red"),
    ("Symbolic anatomy 4",       "Google", "colour",     "Yellow"),
    ("Symbolic anatomy 4",       "Google", "colour",     "Green"),
    ("Color palette",            "Claude", "colour",     "Warm_Coral_Burnt_Orange"),
    ("Logo and Design 1",        "Claude", "logo_shape", "typographic_wordmark_with_slash"),
    ("Logo and Design 1",        "Claude", "colour",     "Ivory_Sepia_Orange"),
    ("Logo and Design 3",        "Google", "logo_shape", "G_with_gradient"),
]


class Board:
    """Names -> cells, in order of first appearance (the gate has no board of its own)."""
    def __init__(self): self.cell = {}

    def of(self, name):
        if name not in self.cell: self.cell[name] = (len(self.cell) // 6, len(self.cell) % 6)
        return self.cell[name]


def claim_delta(g, board, s, rel, o):
    new = [n for n in (s, o) if board.of(n) not in g.nodes]
    parts = [f"ADD[{board.of(n)[0]},{board.of(n)[1]}:0]" for n in dict.fromkeys(new)]
    parts.append(etext("ADD", (board.of(s), board.of(o), rel), 1))
    return CICOParser.parse_transition_delta(" ".join(parts))


def functional_violations(g):
    out = {}
    for (u, v, rel) in g.edges:
        if rel in FUNCTIONAL: out.setdefault((u, rel), set()).add(v)
    return {k: vs for k, vs in out.items() if len(vs) > 1}


def forbidden_hits(g, board):
    names = {c: n for n, c in board.cell.items()}
    return sorted((names[u], names[v]) for (u, v, rel) in g.edges if v in {board.cell[x] for x in FORBIDDEN.get(rel, ()) if x in board.cell})


def guarded_step(g, delta, board, functional, forbidden):
    r = check(g, delta)
    if not r.ok: return r
    trial = copy(g)
    apply(trial, delta)
    if functional and len(functional_violations(trial)) > len(functional_violations(g)):
        return GateResult(False, "functional: a second value for a one-value relation")
    if forbidden and len(forbidden_hits(trial, board)) > len(forbidden_hits(g, board)):
        return GateResult(False, "forbidden: the page's own rule excludes this value")
    apply(g, delta)
    return r


def replay(functional=False, forbidden=False):
    g, board, refused = Graph(), Board(), []
    for i, (sec, s, rel, o) in enumerate(CLAIMS):
        d = claim_delta(g, board, s, rel, o)
        before = copy(g)
        r = guarded_step(g, d, board, functional, forbidden)
        if not r.ok:
            refused.append((i, sec, s, rel, o, r.reason))
            assert same(g, before)                                          # a refused claim leaves the graph exactly as it was
    return g, board, refused


def test_unguarded_the_gate_admits_every_claim_and_so_the_contradictions():
    g, board, refused = replay()
    assert refused == []                                                    # all 10 claims admitted
    assert len([e for e in g.edges if e[2] == "logo_shape"]) == 4          # Claude twice, Google twice
    assert set(functional_violations(g)) == {(board.cell["Claude"], "logo_shape"), (board.cell["Google"], "logo_shape")}
    assert forbidden_hits(g, board) == [("Google", "Red"), ("Google", "Yellow")]       # the page's own rule is broken by its own Google claim


def test_a_functional_edge_guard_refuses_exactly_the_two_later_logo_claims_with_a_reason():
    g, board, refused = replay(functional=True)
    assert [(r[2], r[3], r[4]) for r in refused] == [("Claude", "logo_shape", "typographic_wordmark_with_slash"), ("Google", "logo_shape", "G_with_gradient")]
    assert all(r[5].startswith("functional") for r in refused)
    assert functional_violations(g) == {}
    assert len([e for e in g.edges if e[2] == "colour"]) == 6              # many-valued relations are untouched: Google 4 colours, Claude 2 palettes
    assert len([e for e in g.edges if e[2] == "logo_shape"]) == 2
    assert forbidden_hits(g, board) == [("Google", "Red"), ("Google", "Yellow")]       # the functional guard does not see the colour rule


def test_the_forbidden_value_rule_catches_red_and_yellow_which_the_functional_guard_cannot():
    g, board, refused = replay(functional=True, forbidden=True)
    assert [(r[2], r[3], r[4]) for r in refused if r[3] == "colour"] == [("Google", "colour", "Red"), ("Google", "colour", "Yellow")]
    assert forbidden_hits(g, board) == [] and functional_violations(g) == {}
    only_forbidden = replay(forbidden=True)[2]
    assert [(r[3], r[4]) for r in only_forbidden] == [("colour", "Red"), ("colour", "Yellow")]      # alone it leaves the two logo conflicts in
    assert len(refused) == 4 and len(g.edges) == 6                          # two logo claims and two colour claims refused, 6 of 10 admitted


def test_one_pair_can_hold_three_relations_at_once_but_a_one_role_chart_refuses_the_second_role():
    global FUNCTIONAL
    keep = FUNCTIONAL
    try:
        board, g = Board(), Graph()
        for rel in ("mother_of", "wife_of", "destroyer_of"):                 # three relation names between the same two nodes
            d = claim_delta(g, board, "Kali", rel, "Shiva")
            assert check(g, d).ok
            apply(g, d)
        assert len(g.edges) == 3 and len({(u, v) for (u, v, _) in g.edges}) == 1       # three edges, one pair
        FUNCTIONAL = {"role"}                                                # the chart: one role per person
        g2, board2 = Graph(), Board()
        first = guarded_step(g2, claim_delta(g2, board2, "Kali", "role", "mother"), board2, True, False)
        second = guarded_step(g2, claim_delta(g2, board2, "Kali", "role", "wife"), board2, True, False)
        assert first.ok and not second.ok and second.reason.startswith("functional")
        assert len([e for e in g2.edges if e[2] == "role"]) == 1             # the chart holds one role; the pair of nodes held three
    finally:
        FUNCTIONAL = keep


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

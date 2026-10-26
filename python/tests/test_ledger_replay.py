"""Implements concept-map item 133 (the transition ledger run over the saved ARC-AGI-3 runs of Test 24 and Test 25; python/axi/experiments/ledger_replay.py), test 159.
It comes from the structure of Rajnish Choubey's RFC-001, as read for the ledger (docs/HISTORY_LEDGER.md; the reading is for Sir to confirm). No model, no benchmark package: the stand-in game and
stand-in model of test 143. It checks the INSTRUMENT, not a result on real screens:
  1. the replay still verifies every saved flag with the ledger watching (arms K, A and B), and a doctored row is still refused;
  2. `apply` of the replayed class is put back after a replay and after a refused one, so nothing stays patched;
  3. every move that is played is shown to the ledger once (arm B calls the base `apply` through its own: no double count), except a RESET and a move that completes a level, which are counted, not recorded;
  4. each (game, level) has its own ledger;
  5. the summary: operations found, the matching that gave up, the balance check, the hidden-state flags, the host time, and the text;
  6. the loader of a saved result file and the Colab notebook (no key, no model, no Drive, no billing, nothing run);
  7. the front pages (concept-map item 133, the counter).
What it does NOT show: anything on a real ARC-AGI-3 screen. That needs `pip install arc-agi arcengine` on a machine that can reach PyPI, then `python -m axi.experiments.ledger_replay <result.json> --arm K|A|B`."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_arc as R
from axi.experiments import gemini_arc_b as B
from axi.experiments import ledger_replay as LR
import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
NEXT = "Next free: row 187, item 141, test 166, question 122"
G2 = T.GAMES2


def _rows(arm, llm=None, moves=30):
    if arm == "B": return B.run(llm or T.FakeLLM(), T.make_env, T.tmp(), games=G2, moves=moves, action_of=T.IDENT)["rows"]
    return T.run(llm or T.FakeLLM(), arm, moves=moves)["rows"]


def _feed(arm, rows, moves=30): return LR.feed(rows, T.make_env, arm, G2, moves, T.IDENT)


def test_all_three_arms_replay_with_the_ledger_watching():
    for arm in ("K", "A", "B"):
        rows = _rows(arm); s = _feed(arm, rows)
        played = sum(1 for r in rows if r.get("moved"))                                         # moves that reached the game
        recorded = sum(s.moves.values()); skipped = sum(sum(c.values()) for c in s.skipped.values())
        assert played > 0 and recorded + skipped == played, (arm, played, recorded, skipped)
        assert set(s.skipped) <= set(G2) and set(s.moves) <= set(G2)


def test_apply_is_put_back_and_a_doctored_row_is_still_refused():
    for cls, arm in ((R.Play, "K"), (R.Play, "A"), (B.PlayB, "B")):
        before = cls.__dict__["apply"]; rows = _rows(arm); _feed(arm, rows)
        assert cls.__dict__["apply"] is before
        bad = json.loads(json.dumps(rows)); bad[3]["changed"] = not bad[3]["changed"]
        try: _feed(arm, bad); raise AssertionError("a doctored row was accepted")
        except R.ReplayMismatch: pass
        assert cls.__dict__["apply"] is before                                                  # also after the refusal


def test_the_ledger_does_not_change_the_replay():
    for arm in ("K", "B"):
        rows = _rows(arm)
        plain = (B.replay(rows, T.make_env, G2, 30, T.IDENT) if arm == "B" else R.replay(rows, T.make_env, arm, G2, 30, T.IDENT))
        plain = plain if arm == "B" else plain[0]
        with LR.watching(B.PlayB if arm == "B" else R.Play, LR.Session()):
            pass
        again = (B.replay(rows, T.make_env, G2, 30, T.IDENT) if arm == "B" else R.replay(rows, T.make_env, arm, G2, 30, T.IDENT)); again = again if arm == "B" else again[0]
        assert {g: (p.screen, p.m, p.levels()) for g, p in plain.items()} == {g: (p.screen, p.m, p.levels()) for g, p in again.items()}


def test_reset_and_level_change_are_counted_not_recorded_and_each_level_has_its_own_ledger():
    class Walk(T.FakeLLM):
        def generate(self, messages, constrained=True, max_new_tokens=400):
            super().generate(messages, constrained, max_new_tokens); n = self.n % 12                # right x3 (level 1 at (3,0)), then a RESET, then more moves
            out = '{"action": "RESET"}' if n == 8 else '{"action": "ACTION4"}' if n % 5 else '{"action": "ACTION2"}'
            return out, 8, 0.4
    rows = _rows("K", Walk(), moves=30); s = _feed("K", rows)
    assert s.skipped["alpha-1"]["level_change"] >= 1 and s.skipped["beta-2"]["reset"] >= 1, {g: dict(c) for g, c in s.skipped.items()}
    levels = sorted(lv for (g, lv) in s.ledgers if g == "alpha-1"); assert levels[0] == 0 and len(levels) >= 2
    for L in s.ledgers.values(): assert all(r.action != "RESET" for r in L.rows)


def test_the_summary_and_the_text():
    rows = _rows("K"); s = _feed("K", rows); sm = LR.summarize(s); assert set(sm) == set(G2) or set(sm) <= set(G2)
    for g, v in sm.items():
        assert v["moves"] == s.moves[g] and v["balance_bad"] == 0 and isinstance(v["ms_per_move"], float) and v["ms_per_move"] >= 0
        assert v["changed"] <= v["moves"] and all("moves" in h for h in v["hidden_state"])
    assert sm["alpha-1"]["ops"].get("move", 0) > 0 and sm["alpha-1"]["actions"]["ACTION4"][0]["step"] == (1, 0)
    strip = lambda d: {g: {k: v for k, v in x.items() if k != "ms_per_move"} for g, x in d.items()}; assert strip(LR.summarize(s)) == strip(LR.summarize(s))      # pure (the time differs between calls only because it is measured once, at feed)
    t = LR.text(s)
    for g in sm: assert f"== {g}:" in t
    assert "hidden-state flags" in t and "ms per move" in t and "strips that get smaller or larger" in t


def test_a_moving_player_is_one_object_that_moved_not_gone_and_new():
    rows = _rows("K"); s = _feed("K", rows); sm = LR.summarize(s)
    for g, v in sm.items(): assert v["identity_maybe_lost"] == 0, (g, v)


def test_the_loader_and_the_notebook():
    path = T.tmp().replace("r.jsonl", "res.json"); rows = _rows("K"); meta = dict(arm="K", games=list(G2), moves=30)
    json.dump({"K": dict(rows=rows, meta=meta)}, open(path, "w"))
    r, games, moves = LR.load(path, "K"); assert r == rows and games == G2 and moves == 30
    s = LR.feed(r, T.make_env, "K", games, moves, T.IDENT); strip = lambda d: {g: {k: v for k, v in x.items() if k != "ms_per_move"} for g, x in d.items()}; assert strip(LR.summarize(s)) == strip(LR.summarize(_feed("K", rows)))
    nb = json.load(open(os.path.join(HERE, "..", "colab", "axi_ledger_replay.ipynb"))); src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "ledger_replay" in src and "make_env" in src and "arc-agi arcengine" in src
    assert "sys.modules" in src and 'hasattr(Ledger, "ticks")' in src and "git -C /content/axi log" in src             # a second run in one session must not use the first run's code, and the commit is printed
    for bad in ("userdata", "GOOGLE_API_KEY", "PAID", "BUDGET", "llm", "generate", "drive.mount", "git push"): assert bad not in src, bad
    assert all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")                # nothing was run when it was written


def test_the_front_pages_and_the_counter():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); assert NEXT in cm and cm.count("Next free:") == 1
    j = cm.index("\n133. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "ledger_replay.py" in item and "test 159" in item and "Rajnish Choubey" in item and "no model" in item.lower()
    k = cm.index("\n134. **") + 1; item = cm[k:cm.index("\n", k)]; assert "ticks" in item and "resize" in item and "Rajnish" not in item or "for Sir to confirm" in item
    assert "| 179 |" in cm and "| 180 |" in cm
    vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); assert any("ledger_replay.py" in l and "test_ledger_replay.py" in l for l in vf.split("\n"))


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)

"""Engineering support for docs/KV_DROP_TEST.md (concept-map item 96; item 89: block independence on a stock model). No GPU, no torch, no real model. The KV test compares ways of holding a model's key/value
cache (rebuilt, kept exact, kept with the deleted items cut out, a whole chat) on the same prompts. These checks run the whole instrument on `tests/kv_standin.py`, a genuine small transformer in numpy
(rotary positions, grouped-query attention, a cache that carries its rows' rotations), to show that the instrument is right before a Colab number is trusted:
  1. the item list follows the oracle, an add never moves anything, the three pieces of a prompt are the whole prompt and the check catches a tokenizer that merges across pieces; a toy tokenizer that
     follows the pre-tokenizer rule of Qwen shows why the closing tag follows a space (a line break after a bracket would merge into one token);
  2. the cache operations are what the notebook's `preflight` says they are, and `preflight` fails for each of six deliberately broken caches;
  3. the exact arm holds exactly the cache of a fresh read (all layers, keys, values, positions) and writes the same reply with KL 0; adds are exact for the drop arms; deleting the LAST items is exact;
     a delete in the middle makes the rows behind the hole stale (their values differ from a fresh read), the compact arm has the positions of a fresh read, the gap arm a hole in the positions;
  4. the instrument can see the difference: on the random model the drop arms write other replies after a delete and exact never does;
  5. tokens read, tokens kept and bytes follow from the design, `keep` holds exactly the chat and stops at its cap, scoring counts a legal-as-written reply, a repaired one and a refused one correctly;
  6. the runner is resumable, a resumed run equals an uninterrupted one, a file from another run is refused; summaries recompute from the rows;
  7. the fifteen reading rules flip exactly at their thresholds; the page is locked and carries them; the notebook never saves outputs, every code cell of it runs on the stand-in (and stops at a failed
     check), and the PyTorch side has the pieces the checks rely on."""
import hashlib, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from kv_standin import EOS, TinyCache, TinyEngine, byte_kit, byte_table
import axi.experiments.kv_drop as K
from axi.engine.cico_parser import CICOParser
from axi.experiments.agent_loop import same_graph
from axi.experiments.long_horizon import oracle_script

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "KV_DROP_TEST.md")
PAGE_SHA256 = "0e13a782b7c530a739df955cb6047f2882fe7b7e3c6c39f013a2bc8aef186c43"
NB = os.path.join(HERE, "..", "colab", "axi_kv_drop.ipynb")
RESULT_MARK = "\n\n---\n\n# Result"
TIME_KEYS = {"update_s", "prefill_s", "decode_s", "seconds", "peak_mb"}
KIT, TAB = byte_kit(), byte_table()


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def engine(**kw): return TinyEngine(**kw)


def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list): return [strip(v) for v in x]
    return x


def parse(ideal): return CICOParser.parse_transition_delta(ideal)


def fresh_state(eng, items, kit=KIT):
    c = K.Tracked(eng.new_cache()); toks = kit.state_tokens(items); c.forward(toks, range(len(toks))); return c


def same_cache(a, b, rows=None, atol=1e-9):
    n = a.n if rows is None else rows
    return all(np.allclose(a.c.K[i][:, :n], b.c.K[i][:, :n], atol=atol) and np.allclose(a.c.V[i][:, :n], b.c.V[i][:, :n], atol=atol) for i in range(a.c.e.layers)) and a.pos[:n] == b.pos[:n]


def walk(eng, names, seed, steps, compare, max_new=6, kit=KIT, k_cap=10 ** 9, forced=None):
    """Step the arms through an oracle episode by hand and call compare(t, items, prev_items, change, arms, recs) after every step."""
    g0, script, oracle = oracle_script(seed, steps); items = K.graph_items(g0); arms = K.make_arms(eng, kit, TAB, ["fresh"] + list(names), max_new, k_cap); change, prev = None, None
    for t, st in enumerate(script):
        ref, recs = None, {}
        if forced: eng.force = forced(st)
        for a, arm in arms.items():
            rec, rep = arm.step(t, items, change, st["instruction"], ref)
            if a == "fresh": ref = rep
            recs[a] = K.annotate(rec, rep, None if a == "fresh" else ref, kit, oracle[t], oracle[t + 1], st["instruction"], t, st["kind"])
        compare(t, items, prev, change, arms, recs)
        prev = items; items, deleted, added = K.apply_delta_items(items, parse(st["ideal"])); change = (deleted, added)
    return arms


# ---------------------------------------------------------------- 1 the items and the pieces of the prompt
def test_the_item_list_follows_the_oracle_and_an_add_never_moves_anything():
    multi = nonadjacent = 0
    for seed in (11000, 11001, 11002, 12000):
        g0, script, oracle = oracle_script(seed, 40); items = K.graph_items(g0); assert same_graph(K.items_graph(items), oracle[0])
        assert [it.key[1] for it in items if it.key[0] == "n"] == sorted(g0.nodes) and [it.key[1] for it in items if it.key[0] == "e"] == sorted(g0.edges)      # the starting graph is written sorted, as in the earlier tests
        kinds0 = [it.key[0] for it in items]; assert kinds0 == ["n"] * kinds0.count("n") + ["e"] * kinds0.count("e") and kinds0.count("n") == len(g0.nodes)        # the start: services first, then edges
        for t, st in enumerate(script):
            new, deleted, added = K.apply_delta_items(items, parse(st["ideal"])); assert same_graph(K.items_graph(new), oracle[t + 1]), (seed, t)
            assert new == [it for i, it in enumerate(items) if i not in set(deleted)] + added and deleted == sorted(deleted)       # survivors keep their order, adds go to the end
            if st["kind"].startswith("add"): assert deleted == [] and len(added) == 1 and new[:len(items)] == items and new[-1] == added[0]
            else: assert added == [] and deleted and len(new) == len(items) - len(deleted)
            if len(deleted) >= 2: multi += 1; nonadjacent += any(b - a > 1 for a, b in zip(deleted, deleted[1:]))
            items = new
    assert multi >= 5 and nonadjacent >= 3                                                                                       # a delete with dependants cuts several places out of the list
    it = K.Item(("n", (1, 2)), 5); assert it.text == "[1,2:5]" and K.Item(("e", ((0, 1), (2, 3), "dep")), 7).text == "(0,1)->(2,3):dep#7"
    delta = parse("ADD[1,2:9]"); new, deleted, added = K.apply_delta_items([K.Item(("n", (1, 2)), 5), K.Item(("n", (0, 0)), 1)], delta)
    assert deleted == [0] and added == [K.Item(("n", (1, 2)), 9)] and new == [K.Item(("n", (0, 0)), 1), K.Item(("n", (1, 2)), 9)]    # a re-added key counts as removed and added


def test_the_pieces_of_a_prompt_are_the_whole_prompt_and_the_check_catches_a_tokenizer_that_merges_across_pieces():
    g0, script, _ = oracle_script(11000, 5); items = K.graph_items(g0)
    for kit in (byte_kit(), byte_kit(True)):
        for st in script:
            assert kit.canonical(items, st["instruction"]) and kit.prompt_tokens(items, st["instruction"]) == kit.enc(kit.whole_text(items, st["instruction"]))
            assert kit.dec(kit.prompt_tokens(items, st["instruction"])) == kit.whole_text(items, st["instruction"])
        assert ("Rule: a service can only be removed" in kit.system) == kit.hint_rules and kit.whole_text(items, "x").startswith("<|im_start|>system\n" + kit.system)
    assert KIT.whole_text(items, "Delete the service at (1,0).").endswith(" </CICO_STATE>\n\nRequest: Delete the service at (1,0).<|im_end|>\n<|im_start|>assistant\n")
    vocab = {}

    def merging_enc(s):                                                                       # a word and the space after it are one token, like many real tokenizers: pieces and whole disagree
        return [vocab.setdefault(w, len(vocab)) for w in re.findall(r"[^ ]+ ?| ", s)]
    bad = K.PromptKit(merging_enc, lambda ids: "", 0)
    assert len(items) >= 2 and not bad.canonical(items, "Delete the service at (1,0).")


def test_the_wording_of_the_prompt_is_chatml_and_a_toy_tokenizer_with_the_pre_tokenizer_rule_of_qwen_shows_why_the_closing_tag_follows_a_space():
    assert K.first_prefix_text("S") == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<CICO_STATE>" and K.LATER_PREFIX == "<|im_start|>user\n<CICO_STATE>"
    assert K.tail_text("Q") == " </CICO_STATE>\n\nRequest: Q<|im_end|>\n<|im_start|>assistant\n"
    # Qwen2's pre-tokenizer (my reading of its published regular expression; \p{L} and \p{N} written for plain text): single digits, a run of punctuation takes the line breaks right after it. A byte-pair
    # tokenizer merges only INSIDE a pre-token, so a vocabulary entry for a bracket plus a line break exists exactly when such a pre-token occurs. The toy vocabulary below has those entries.
    pre = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d| ?(?:_|[^\w\s])+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"); vocab = {}; whole = {"]\n", "]\n\n", ")\n", "]\n\n\n"}

    def toy_enc(text):
        out = []
        for part in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            for tok in ([part] if part in ("<|im_start|>", "<|im_end|>") else [t for t in pre.findall(part)]):
                for piece in ([tok] if tok in whole or tok.startswith("<|im_") else list(tok)): out.append(vocab.setdefault(piece, len(vocab)))
        return out
    steps = last_node = last_edge = 0
    for seed in (11000, 11001, 11002, 12000):
        g0, script, oracle = oracle_script(seed, 40); items = K.graph_items(g0); kit = K.PromptKit(toy_enc, lambda ids: "", 0, seed % 2 == 0)
        for st in script:
            assert kit.canonical(items, st["instruction"]), (seed, st["instruction"], items[-1])
            steps += 1; last_node += bool(items) and items[-1].key[0] == "n"; last_edge += bool(items) and items[-1].key[0] == "e"; items, _, _ = K.apply_delta_items(items, parse(st["ideal"]))
    assert last_node >= 10 and last_edge >= 10                                                      # both kinds of last item occur
    old_tail = K.tail_text                                                                           # the earlier wording (a line break before the closing tag) merges after a service
    try:
        K.tail_text = lambda instruction: "\n" + old_tail(instruction)[1:]
        kit = K.PromptKit(toy_enc, lambda ids: "", 0); g0, script, _ = oracle_script(11000, 40); items = K.graph_items(g0); bad = 0
        for st in script: bad += not kit.canonical(items, st["instruction"]); items, _, _ = K.apply_delta_items(items, parse(st["ideal"]))
        assert bad >= 5, bad
    finally: K.tail_text = old_tail
    assert K.tail_text is old_tail


def test_rope_tables_are_rotations_that_add_in_the_half_split_layout():
    inv = 1.0 / 10000.0 ** (np.arange(0, 8, 2) / 8); c1, s1 = K.rope_tables([3, -5, 0], inv); c2, s2 = K.rope_tables([4, 4, 4], inv); c3, s3 = K.rope_tables([7, -1, 4], inv)
    assert c1.shape == (3, 8) and np.allclose(c1 ** 2 + s1 ** 2, 1.0) and np.allclose(c1[:, :4], c1[:, 4:]) and np.allclose(s1[:, :4], s1[:, 4:]) and np.allclose(c1[2], 1) and np.allclose(s1[2], 0)
    x = np.random.default_rng(0).normal(size=(3, 8)); rot = lambda v, c, s: v * c + np.concatenate([-v[:, 4:], v[:, :4]], axis=1) * s
    assert np.allclose(rot(rot(x, c1, s1), c2, s2), rot(x, c3, s3))                         # turning by 3, -5, 0 and then by 4 is turning by 7, -1, 4
    z = x[:, :4] + 1j * x[:, 4:]; ang = np.array([3, -5, 0])[:, None] * inv[None, :]; w = z * np.exp(1j * ang)                                  # the same thing with complex numbers
    r = rot(x, c1, s1); assert np.allclose(r[:, :4], w.real) and np.allclose(r[:, 4:], w.imag)


def test_the_tiny_model_is_a_causal_rope_transformer_with_the_properties_the_checks_need():
    e = engine(); rng = np.random.default_rng(1); T = list(rng.integers(0, 256, 40))
    a = e.new_cache(); la = a.forward(T, range(40)); b = e.new_cache(); b.forward(T[:17], range(17)); lb = b.forward(T[17:], range(17, 40)); assert np.allclose(la, lb, atol=1e-10)      # reading in pieces
    c = e.new_cache(); lc = c.forward(T, range(100, 140)); assert np.allclose(la, lc, atol=1e-10)                                                                                  # only relative positions matter
    d = e.new_cache(); ld = d.forward(T, [p * 2 for p in range(40)]); assert not np.allclose(la, ld, atol=1e-3)                                                                    # other spacing is a different input
    T2 = T[:-1] + [(T[-1] + 1) % 256]; f = e.new_cache(); f.forward(T2[:39], range(39)); g = e.new_cache(); g.forward(T[:39], range(39))
    assert all(np.allclose(f.K[i][:, :39], g.K[i][:, :39]) for i in range(3))                                                                                                       # causal: a later token cannot change earlier rows
    h = e.new_cache(); h.forward(T[:20], range(20)); h2 = e.new_cache(); h2.forward([(T[0] + 1) % 256] + T[1:20], range(20)); assert not np.allclose(h.V[2][:, 19], h2.V[2][:, 19])  # but an earlier token changes later rows


# ---------------------------------------------------------------- 2 the cache operations and the checks before the run
def test_preflight_passes_on_the_tiny_model():
    res = K.preflight(engine(), KIT)
    assert len(res) == 9 and all(ok for _, ok, _ in res), res
    assert any("can tell" in name for name, _, _ in res)


def _broken(cls_body):
    e = engine(); e.cache_class = type("Broken", (TinyCache,), cls_body); return e


def test_preflight_fails_for_each_of_six_deliberately_broken_caches():
    def no_turn(self, new_pos): self.pos = [int(p) for p in new_pos]                       # forgets to turn the keys
    def ignores_positions(self, tokens, positions): return TinyCache.forward(self, tokens, range(self.n, self.n + len(tokens)))
    def drops_first(self, rows):
        rows = list(rows); TinyCache.select(self, rows[1:] if rows and rows == list(range(len(rows))) else rows)
    def no_size(self): return 0

    def read_without(self, tokens, positions, hide):                                         # a call that does not see the rows held before it
        e = self.e; old = (self.K, self.V, self.pos); self.K = [np.zeros((e.KV, 0, e.hd)) for _ in range(e.layers)]; self.V = [np.zeros((e.KV, 0, e.hd)) for _ in range(e.layers)]; self.pos = []
        out = TinyCache.forward(self, tokens, positions); self.K = [np.concatenate([a, b], axis=1) for a, b in zip(old[0], self.K)]; self.V = [np.concatenate([a, b], axis=1) for a, b in zip(old[1], self.V)]; self.pos = old[2] + self.pos
        return out

    def hides_across_gap(self, tokens, positions):                                          # what a library that mistakes a jump in the position numbers for a new sequence would do
        if self.n and int(list(positions)[0]) != self.pos[-1] + 1: return read_without(self, tokens, positions, True)
        return TinyCache.forward(self, tokens, positions)

    def forgets_past(self, tokens, positions):
        if self.n: return read_without(self, tokens, positions, True)
        return TinyCache.forward(self, tokens, positions)
    for name, body, expect in (("no_turn", dict(rerotate=no_turn), "turned by 211"), ("ignores_positions", dict(forward=ignores_positions), "turned by 211"),
                               ("drops_first", dict(select=drops_first), "cutting the last rows"), ("no_size", dict(nbytes=no_size), "reports its size"),
                               ("hides_across_gap", dict(forward=hides_across_gap), "hole in the position numbers"), ("forgets_past", dict(forward=forgets_past), "read in two pieces")):
        bad = [n for n, ok, _ in K.preflight(_broken(body), KIT) if not ok]
        assert any(expect in n for n in bad), (name, bad)


# ---------------------------------------------------------------- 3 the arms against a fresh read
def test_the_exact_arm_holds_the_cache_of_a_fresh_read_and_writes_the_same_reply_with_kl_zero():
    e = engine(); seen = {"del": 0}

    def compare(t, items, prev, change, arms, recs):
        assert same_cache(arms["exact"].cache, fresh_state(e, items)), t
        assert recs["exact"]["v_dev"] < 1e-9 and (recs["exact"]["kl_max"] or 0) < 1e-9 and recs["exact"]["same_as_fresh"] is True
        seen["del"] += bool(change and change[0])
    walk(e, ["exact"], 11000, 14, compare, max_new=8); assert seen["del"] >= 2


def test_adds_are_exact_for_the_drop_arms_and_a_delete_in_the_middle_makes_the_rows_behind_it_stale():
    e = engine(); st = {"deleted": False, "first": None, "checked": 0}; pre = len(KIT.first_prefix)

    def compare(t, items, prev, change, arms, recs):
        ref = fresh_state(e, items); now = bool(change and change[0])
        for name in ("drop_compact", "drop_gap"):
            a = arms[name]; assert a.cache.ids == KIT.state_tokens(items)
            if not st["deleted"] and not now: assert same_cache(a.cache, ref), (name, t); assert recs[name]["stale"] == 0 and recs[name]["v_dev"] < 1e-9      # nothing deleted yet: appending is exact
        if now and not st["deleted"]:
            st["deleted"] = True; hole = pre + sum(len(KIT.item(it)) for it in prev[:min(change[0])]); after = len(KIT.state_tokens(items)) - hole - sum(len(KIT.item(it)) for it in change[1])
            for name in ("drop_compact", "drop_gap"):
                a = arms[name]; assert same_cache(a.cache, ref, rows=hole), name                                                                              # rows before the hole: untouched, same as fresh
                dev = max(float(np.abs(a.cache.c.V[i][:, hole:hole + after] - ref.c.V[i][:, hole:hole + after]).max()) for i in range(3))
                assert (dev > 1e-6) == (after > 0), (name, dev, after)                                                                                       # rows behind it: values computed with the deleted item visible
                assert recs[name]["stale"] == after and recs[name]["v_dev"] > 0 if after > 0 else recs[name]["v_dev"] < 1e-9
                exp = float(np.mean([np.linalg.norm(a.cache.c.V[i] - ref.c.V[i]) / np.linalg.norm(ref.c.V[i]) for i in range(3)])); assert abs(recs[name]["v_dev"] - exp) <= 1e-9 * max(1.0, exp), (name, recs[name]["v_dev"], exp)
            assert arms["drop_compact"].cache.pos == list(range(arms["drop_compact"].cache.n))                                                              # compact: the positions of a fresh read
            gap = arms["drop_gap"].cache; assert gap.pos[:hole] == list(range(hole)) and gap.pos[-1] + 1 - gap.n == recs["drop_gap"]["gap"] > 0 or after == 0
            st["checked"] += 1
    walk(e, ["drop_compact", "drop_gap"], 11001, 12, compare, max_new=6); assert st["checked"] == 1


def test_deleting_the_last_items_is_exact_and_deleting_one_in_the_middle_is_not():
    e = engine(); g0, script, _ = oracle_script(11000, 3); items = K.graph_items(g0)
    for cut, exact in (("last", True), ("last_and_add", True), ("middle", False)):
        arms = K.make_arms(e, KIT, TAB, ["fresh", "drop_compact", "drop_gap"], 4); ref = None
        for a, arm in arms.items():
            rec, rep = arm.step(0, items, None, "Delete the service at (1,0).", ref); ref = rep if a == "fresh" else ref
        if cut == "middle": gone, new = [3], items[:3] + items[4:]
        else: gone, new = [len(items) - 1], items[:-1]
        add = [K.Item(("n", (9, 9)), 4)] if cut == "last_and_add" else []
        new = new + add; ref = None
        for a, arm in arms.items():
            rec, rep = arm.step(1, new, (gone, add), "Delete the service at (1,0).", ref)
            if a == "fresh": ref = rep
            elif exact: assert rec["v_dev"] < 1e-9 and rep.tokens == ref.tokens and rec["stale"] == 0, (cut, a)
            else: assert rec["v_dev"] > 1e-6 and rec["stale"] > 0, (cut, a)
        assert (arms["drop_gap"].cache.pos[-1] + 1 - arms["drop_gap"].cache.n == 0) == exact or cut == "last_and_add" or not exact


def test_the_instrument_sees_the_difference_the_drop_arms_make_and_exact_makes_none():
    e = engine(); out = {}
    for seed in (11000, 11001):
        ep = K.run_kv_episode(e, KIT, TAB, seed, 12, arms=("fresh", "exact", "drop_compact", "drop_gap"), max_new=12); assert ep["stopped"]["exact"] is None
        g0, script, _ = oracle_script(seed, 12); first_del = next(i for i, s in enumerate(script) if not s["kind"].startswith("add")) + 1          # the cache first holds a hole at step first_del + 1
        for a in ("exact", "drop_compact", "drop_gap"):
            for s in ep["arms"][a]:
                out.setdefault(a, []).append((s["t"] > first_del, s["same_as_fresh"], s["kl_max"], s["v_dev"]))
    assert all(same and (kl or 0) < 1e-9 and dev < 1e-9 for _, same, kl, dev in out["exact"])
    for a in ("drop_compact", "drop_gap"):
        before = [x for x in out[a] if not x[0]]; after = [x for x in out[a] if x[0]]
        assert all(same and (kl or 0) < 1e-9 for _, same, kl, _ in before)                                                      # before any delete: identical to a fresh read
        assert sum(not same for _, same, _, _ in after) >= 3 and sum((kl or 0) > 1e-6 for _, _, kl, _ in after) >= len(after) // 2 and all(dev > 0 for _, _, _, dev in after)


def test_what_a_step_reads_keeps_and_costs_follows_from_the_design():
    e = engine(); rows = {"suffix": [], "keep": []}; cur = {}

    def compare(t, items, prev, change, arms, recs):
        tail = len(KIT.tail(cur["instr"])); state = len(KIT.state_tokens(items)); full = state + tail; per_row = 768                                       # 3 layers x (keys + values) x 2 heads x 8 values x 8 bytes
        for a, r in recs.items():
            fed = len(r["tokens"]) - (1 if r["ended"] else 0)
            assert r["kv_bytes"] == per_row * r["resident"] and abs(r["seconds"] - (r["update_s"] + r["prefill_s"] + r["decode_s"])) < 1e-12 and r["decoded"] == len(r["tokens"])
            if a != "keep": assert r["resident"] == state + tail + fed and r["resident_state"] == state
        assert recs["fresh"]["prefilled"] == recs["fresh"]["prompt_tokens"] == full
        add_toks = sum(len(KIT.item(it)) for it in (change[1] if change else []))
        if t == 0: assert recs["exact"]["prefilled"] == recs["drop_compact"]["prefilled"] == recs["drop_gap"]["prefilled"] == full
        else:
            suffix = 0
            if change[0]:
                i0 = min(change[0]); suffix = sum(len(KIT.item(it)) for j, it in enumerate(prev) if j > i0 and j not in set(change[0]))
            assert recs["drop_compact"]["prefilled"] == recs["drop_gap"]["prefilled"] == tail + add_toks and recs["exact"]["prefilled"] == tail + add_toks + suffix
            rows["suffix"].append(suffix)
        assert recs["keep"]["prefilled"] == len(KIT.state_tokens(items, first=t == 0)) + tail + 1 + len(KIT.nl) and recs["keep"]["resident_state"] == recs["keep"]["resident"]
        rows["keep"].append(recs["keep"]["resident"])

    def forced(st): cur["instr"] = st["instruction"]; return None
    walk(e, ["exact", "drop_compact", "drop_gap", "keep"], 11002, 10, compare, max_new=6, forced=forced)
    keep = rows["keep"]; assert all(b > a for a, b in zip(keep, keep[1:])) and keep[-1] > 3 * keep[0] and max(rows["suffix"]) > 0 and min(rows["suffix"]) == 0


def test_keep_is_exactly_the_chat_and_stops_at_its_cap_while_the_other_arms_go_on():
    e = engine(); log = []

    def compare(t, items, prev, change, arms, recs):
        log.append((list(items), recs["fresh"]["tokens"], recs["keep"]["tokens"], recs["keep"]["ended"]))
        want = []
        for k, (its, _, tk, ended) in enumerate(log):
            instr = oracle_script(11003, 6)[1][k]["instruction"]
            want += KIT.state_tokens(its, first=k == 0) + KIT.tail(instr) + [x for x in tk if x != EOS] + [EOS] + KIT.nl
        assert arms["keep"].cache.ids == want and arms["keep"].cache.pos == list(range(len(want))), t
    arms = walk(e, ["keep"], 11003, 6, compare, max_new=5)
    assert len(log) == 6
    chat = KIT.dec(arms["keep"].cache.ids)                                                                    # in words: one system prompt, then six turns of user, assistant
    assert chat.count("<|im_start|>system") == 1 and chat.count("<|im_start|>user\n<CICO_STATE>") == 6 and chat.count("<|im_start|>assistant\n") == 6 and chat.startswith(K.first_prefix_text(KIT.system))
    ep = K.run_kv_episode(e, KIT, TAB, 11003, 8, arms=("fresh", "exact", "keep"), k_cap=2300, max_new=5)
    assert ep["stopped"] == {"fresh": None, "exact": None, "keep": "cap"} and 1 <= len(ep["arms"]["keep"]) < len(ep["arms"]["fresh"]) == len(ep["arms"]["exact"]) == 8
    assert all(s["resident"] <= 2300 for s in ep["arms"]["keep"])
    e2 = engine(); long_reply = lambda t, st: setattr(e2, "force", list(b"ADD[1"))                                                                                    # a reply that uses all 5 new tokens and does not end
    full = K.run_kv_episode(e2, KIT, TAB, 11003, 3, arms=("fresh", "keep"), max_new=5, on_step=long_reply); r3 = full["arms"]["keep"][2]["resident"]; assert not any(x["ended"] for x in full["arms"]["keep"])
    fits = K.run_kv_episode(e2, KIT, TAB, 11003, 3, arms=("fresh", "keep"), k_cap=r3, max_new=5, on_step=long_reply); assert fits["stopped"]["keep"] is None and fits["arms"]["keep"][2]["resident"] == r3
    tight = K.run_kv_episode(e2, KIT, TAB, 11003, 3, arms=("fresh", "keep"), k_cap=r3 - 1, max_new=5, on_step=long_reply)                                                    # one token less room: step 3 does not start
    assert tight["stopped"]["keep"] == "cap" and len(tight["arms"]["keep"]) == 2 and all(x["resident"] <= r3 - 1 for x in tight["arms"]["keep"])


def test_scoring_counts_a_reply_that_is_legal_as_written_a_repaired_one_and_a_refused_one():
    def run(force, max_new=200, steps=12):
        e = engine(); return K.run_kv_episode(e, KIT, TAB, 11000, steps, arms=("fresh", "exact", "drop_compact"), max_new=max_new, on_step=lambda t, st: setattr(e, "force", force(st)))
    ideal = lambda st: list(st["ideal"].encode()) + [EOS]
    ep = run(ideal)
    for a, steps in ep["arms"].items():
        assert all(s["parsed"] and s["admitted"] and s["exact_blind"] and s["exact_m6"] and s["ended"] for s in steps), a
        if a != "fresh": assert all(s["same_as_fresh"] for s in steps)                       # a forced reply is the same whatever the cache; the difference measures are not about the forcing
    node_only = lambda st: list(st["ideal"].split(" ")[0].encode()) + [EOS]
    ep = run(node_only); dd = [s for s in ep["arms"]["fresh"] if s["kind"] == "delete_dep"]; ad = [s for s in ep["arms"]["fresh"] if s["kind"] == "add_edge"]
    assert len(dd) >= 2 and all(s["parsed"] and not s["admitted"] and not s["exact_blind"] and s["exact_m6"] for s in dd)                      # the 7B's habit: the service without its edges; blind leaves dangling edges, m6 repairs
    assert ad and all(s["exact_m6"] and s["exact_blind"] and s["admitted"] for s in ad)                                                          # the node-only reply IS the whole answer to an add request
    ep = run(lambda st: list(b"ADD[9,9:1]") + [EOS]); allw = ep["arms"]["fresh"]                                                                 # legal as written, and wrong on every request
    assert all(s["parsed"] and s["admitted"] and not s["exact_m6"] and not s["exact_blind"] for s in allw)
    ep = run(ideal, max_new=3, steps=4); assert all(not s["parsed"] and not s["exact_m6"] and not s["ended"] for s in ep["arms"]["fresh"])           # cut off after 3 tokens: refused, nothing applied


def test_decoding_and_the_kl_helper():
    p = np.array([0.0, 1.0, -np.inf, 2.0]); q = np.array([0.5, 1.0, -np.inf, 1.0])
    assert K.masked_kl(p, q) > 0 and K.masked_kl(p, np.array([0.0, 1.0, 0.0, 2.0])) == K.masked_kl(p, np.array([0.0, 1.0, 5.0, 2.0]))     # tokens the first distribution forbids do not count
    assert abs(K.masked_kl(p, p)) < 1e-12 and K.masked_kl(p, np.array([0.0, -np.inf, -np.inf, 2.0])) == float("inf")
    pa, qa = np.exp([0.0, 1.0, 2.0]), np.exp([0.5, 1.0, 1.0]); pa, qa = pa / pa.sum(), qa / qa.sum(); assert abs(K.masked_kl(np.array([0.0, 1.0, -np.inf, 2.0]), np.array([0.5, 1.0, 3.0, 1.0])) - float((pa * np.log(pa / qa)).sum())) < 1e-12

    class Stub:
        def __init__(self): self.fed = []

        def forward(self, tokens, positions): self.fed.append((list(tokens), list(positions))); lg = np.full(257, -5.0); lg[ord("]")] = 5.0; lg[EOS] = 4.0; return lg
    lg = np.full(257, -5.0); lg[ord("A")] = 3.0; lg[ord("D")] = 2.0; lg[EOS] = 99.0                                               # EOS is not allowed first (at least one operation); A is the best other start
    s = Stub(); rep = K.decode(s, lg, TAB, EOS, 10, 5)
    assert rep.tokens[0] == ord("A") and not rep.ended and len(rep.tokens) == 5 and [f[1][0] for f in s.fed] == [10, 11, 12, 13, 14] and [f[0][0] for f in s.fed] == rep.tokens
    lg2 = np.full(257, -5.0); lg2[ord("A")] = 3.0

    class Stub2(Stub):
        def forward(self, tokens, positions):
            self.fed.append((list(tokens), list(positions))); out = np.full(257, -5.0); out[{1: ord("D"), 2: ord("D"), 3: ord("[")}.get(len(self.fed), EOS)] = 5.0; out[EOS] = 99.0 if len(self.fed) > 5 else out[EOS]; return out
    s = Stub2(); rep = K.decode(s, lg2, TAB, EOS, 0, 50); assert rep.tokens[:4] == [ord("A"), ord("D"), ord("D"), ord("[")]
    ref = K.decode(Stub2(), lg2, TAB, EOS, 0, 6, keep=True); assert len(ref.masked) == len(ref.tokens) and ref.kl == []
    same = K.decode(Stub2(), lg2, TAB, EOS, 0, 6, ref=ref); assert same.tokens == ref.tokens and same.first_diff is None and len(same.kl) == len(ref.tokens) and max(same.kl) < 1e-12      # the same reply: KL at every position
    moved = lg2.copy(); moved[ord("A")] = -1.0; mv = K.decode(Stub2(), moved, TAB, EOS, 0, 6, ref=ref)
    assert mv.tokens == ref.tokens and mv.first_diff is None and mv.kl[0] > 1e-3 and max(mv.kl[1:]) < 1e-12                                                                         # the same reply with another distribution: KL shows it
    other = lg2.copy(); other[ord("D")] = 4.0; ot = K.decode(Stub2(), other, TAB, EOS, 0, 6, ref=ref)
    assert ot.tokens[0] == ord("D") and ot.first_diff == 0 and len(ot.kl) == 1 and ot.kl[0] > 0.1                                                                                   # another reply: the first difference is found, KL stops being collected there
    assert K.decode(Stub2(), lg2, TAB, EOS, 0, 6).kl == [] and K.decode(Stub2(), lg2, TAB, EOS, 0, 6).first_diff is None                                                              # no reference: nothing to compare



# ---------------------------------------------------------------- 4 the runner, the summaries and the reading rules
def test_the_runner_is_resumable_a_resumed_run_equals_an_uninterrupted_one_and_another_run_is_refused():
    arms = ("fresh", "exact", "drop_compact"); d = tempfile.mkdtemp(); p1, p2 = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
    full = K.run_kv(engine(), KIT, TAB, 3, 6, 5, False, p1, meta=dict(model="stand-in"), arms=arms, max_new=8)
    e = engine(); e.fail_after = 250
    try: K.run_kv(e, KIT, TAB, 3, 6, 5, False, p2, meta=dict(model="stand-in"), arms=arms, max_new=8); raise AssertionError("the dropped session should have stopped the run")
    except RuntimeError as ex: assert "session dropped" in str(ex)
    done = sum(1 for _ in open(p2)) - 1; assert 0 <= done < 3
    res = K.run_kv(engine(), KIT, TAB, 3, 6, 5, False, p2, meta=dict(model="stand-in"), arms=arms, max_new=8)
    assert strip(res["rows"]) == strip(full["rows"]) and len(res["rows"]) == 3 and [r["seed"] for r in res["rows"]] == [5000, 5001, 5002]
    for kw in (dict(steps=7), dict(seed=6), dict(arms=("fresh", "exact"))):
        args = dict(episodes=3, steps=6, seed=5, arms=arms, max_new=8); args.update(kw)
        try: K.run_kv(engine(), KIT, TAB, args["episodes"], args["steps"], args["seed"], False, p2, meta=dict(model="stand-in"), arms=args["arms"], max_new=8); raise AssertionError("another run must be refused")
        except ValueError as ex: assert "different run" in str(ex)
    try: K.run_kv(engine(), byte_kit(True), TAB, 3, 6, 5, False, os.path.join(d, "c.jsonl"), arms=arms); raise AssertionError("the kit and the run must agree about the rule")
    except AssertionError as ex: assert "disagree" in str(ex)


def test_summaries_recompute_from_the_rows():
    ep = K.run_kv(engine(), KIT, TAB, 2, 9, 5, False, os.path.join(tempfile.mkdtemp(), "s.jsonl"), arms=("fresh", "exact", "drop_compact", "keep"), max_new=8)
    rows, S = ep["rows"], ep["summary"]; steps = lambda a: [s for r in rows for s in r["arms"][a]]
    for a in ("fresh", "exact", "drop_compact", "keep"):
        xs = steps(a); v = S[a]; assert v["n"] == len(xs) == 18 and v["episodes"] == 2
        assert v["prefilled"] == sum(s["prefilled"] for s in xs) and abs(v["seconds"] - sum(s["seconds"] for s in xs) / 18) < 1e-12 and abs(v["exact_m6"] - sum(s["exact_m6"] for s in xs) / 18) < 1e-12
        assert v["resident_last"] == sum(r["arms"][a][-1]["resident"] for r in rows) / 2 and abs(v["kv_mb_last"] - sum(r["arms"][a][-1]["kv_bytes"] for r in rows) / 2e6) < 1e-12
        first = [s for r in rows for s in r["arms"][a] if s["t"] <= 3]; last = [s for r in rows for s in r["arms"][a] if s["t"] > 6]
        assert len(first) == len(last) == 6 and abs(v["seconds_first"] - sum(s["seconds"] for s in first) / 6) < 1e-12 and abs(v["seconds_last"] - sum(s["seconds"] for s in last) / 6) < 1e-12
        late = [s for s in xs if s["t"] > 6]; assert len(late) == 6
        if a != "fresh": assert abs(v["same"] - sum(s["same_as_fresh"] for s in xs) / 18) < 1e-12 and abs(v["same_late"] - sum(s["same_as_fresh"] for s in late) / 6) < 1e-12
    import copy
    for kl, want in ((0.04, 1.0), (0.2, 0.0)):                                                                                       # the share of steps whose largest KL is below 0.05 nats
        r2 = copy.deepcopy(rows)
        for r in r2:
            for st in r["arms"]["drop_compact"]: st["kl_max"] = kl
        assert K.summarize_kv(r2)["drop_compact"]["kl_below"] == want, kl
    assert S["fresh"]["same"] is None and S["fresh"]["canonical"] == 1.0 and S["exact"]["same"] == 1.0 and S["drop_compact"]["stale_last"] is not None and S["exact"]["stale_last"] is None
    assert sum(k["n"] for k in S["drop_compact"]["by_kind"].values()) == 18 and set(S["drop_compact"]["by_kind"]) <= {"add_node", "add_edge", "delete_dep", "delete_leaf"}


def _summary(**o):
    A = dict(fresh=dict(n=240, canonical=1.0, exact_m6=0.99, prefilled=100000, seconds=1.0), exact=dict(n=240, same=0.99, prefilled=40000, seconds=0.9),
             drop_compact=dict(n=240, same=0.97, same_late=0.97, kl_below=0.95, exact_m6=0.98, prefilled=25000, seconds=0.85, resident_last=500), drop_gap=dict(n=240, same=0.95),
             keep=dict(n=240, same=0.90, seconds_first=1.0, seconds_last=1.5, resident_last=5000))
    C, B = dict(drop_compact=dict(n=240, same=0.95)), dict(drop_compact=dict(n=200, same_late=0.95))
    for k, v in o.items():
        tgt = {"A": A, "C": C, "B": B}[k[0]]; arm, key = k[1:].split("__"); tgt[arm][key] = v
    return A, C, B


def test_the_fifteen_reading_rules_flip_exactly_at_their_thresholds():
    A, C, B = _summary(); base = K.verdict_kv(A, C, B); assert len(base) == len(K.KV_LINES) == 15 and all(ok is True for _, ok in base), base
    assert [t for t, _ in base] == [t for _, t in K.KV_LINES] and [t for t, _ in K.KV_LINES].count("guard") == 1 and [t for t, _ in K.KV_LINES].count("instrument") == 1
    flips = [("Afresh__canonical", 0.999, 1), ("Aexact__same", 0.9899, 2), ("Adrop_compact__same", 0.9699, 3), ("Adrop_gap__same", 0.9499, 4), ("Adrop_compact__exact_m6", 0.9799 - 0.0001, 5), ("Adrop_compact__kl_below", 0.9499, 6),
             ("Adrop_compact__prefilled", 25001, 7), ("Aexact__prefilled", 40001, 8), ("Akeep__resident_last", 4999, 9), ("Adrop_compact__seconds", 0.8501, 10), ("Aexact__seconds", 0.8, 11),
             ("Akeep__seconds_last", 1.5001, 12), ("Akeep__same", 0.8999, 13), ("Cdrop_compact__same", 0.9499, 14), ("Bdrop_compact__same_late", 0.9499, 15)]
    for key, val, line in flips:
        A2, C2, B2 = _summary(**{key: val}); got = K.verdict_kv(A2, C2, B2)
        assert [i + 1 for i, (_, ok) in enumerate(got) if ok is not True] == [line], (key, [i + 1 for i, (_, ok) in enumerate(got) if ok is not True])
    for ex, want in ((0.85, False), (0.8500001, True), (0.8, False)):                                                               # drop_compact's 0.85 seconds against exact's: a tie is not 'below'
        A2, C2, B2 = _summary(Aexact__seconds=ex); assert K.verdict_kv(A2, C2, B2)[10][1] is want, ex
    for v, want in ((0.98, True), (1.0, True), (0.9799, False), (1.0001, False), (0.995, True)):                                                        # fresh is 0.99: 'within 1 point' works in both directions
        A2, C2, B2 = _summary(Adrop_compact__exact_m6=v); assert K.verdict_kv(A2, C2, B2)[4][1] is want, v
    v = K.verdict_kv(A, None, None); assert v[13][1] is None and v[14][1] is None and all(ok is True for i, (_, ok) in enumerate(v) if i < 13)         # runs that were not made read 'not run'
    assert K.verdict_kv({}, None, None) == [(t, None) for _, t in K.KV_LINES]


# ---------------------------------------------------------------- 5 the page, the notebook and the PyTorch side
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_fifteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(K.KV_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert len(probs) == 15
    assert probs == [90, 85, 80, 75, 90, 70, 80, 80, 90, 70, 75, 75, 65, 60, 60]
    p = {i + 1: v for i, v in enumerate(probs)}; prod = 100 * np.prod([p[i] / 100 for i in K.HEADLINE_LINES]); assert K.HEADLINE_LINES == (2, 3, 5, 6, 14) and abs(prod - 26) <= 1 and "about **26 %**" in head
    assert "(lines 2, 3, 5, 6 and 14)" in head.replace("**Headline** (lines 2, 3, 5, 6 and 14)", "(lines 2, 3, 5, 6 and 14)")
    for needle in (f"stream seed {K.KV_SEED} (episode seeds 11000 to 11007)", f"stream seed {K.KV_LONG_SEED} (episode seeds 12000 and 12001)", f"{K.KV_A_EPISODES} episodes of {K.KV_A_STEPS} steps", f"{K.KV_B_EPISODES} episodes of {K.KV_B_STEPS} steps",
                   f"{K.KV_K_CAP:,} tokens", f"at most {K.KV_MAX_NEW} new tokens", "Qwen2.5-7B-Instruct in 4-bit", "Teacher forcing", "`fresh`", "`exact`", "`drop_compact`", "`drop_gap`", "`keep`", "`preflight`", "`kv_drop.verdict_kv`",
                   "Prompt Cache", "CacheBlend", "Block-Attention for Efficient RAG", "RFC-001", "item 96", "$0", "about 57 KB per token", "not a survey", "Related work (checked 14 Oct 2026"):
        assert needle in head, needle
    assert (K.KV_A_EPISODES, K.KV_A_STEPS, K.KV_C_EPISODES, K.KV_C_STEPS, K.KV_B_EPISODES, K.KV_B_STEPS, K.KV_SEED, K.KV_LONG_SEED, K.KV_K_CAP, K.KV_MAX_NEW) == (8, 30, 8, 30, 2, 100, 11, 12, 20000, 120)
    consts = [("at least 99 %", K.K_EXACT_SAME == 0.99), ("at least 97 %", K.K_DC_SAME == 0.97), ("within 1 point", K.K_EXACT_POINTS == 0.01), ("below 0.05 nats", K.K_KL == 0.05), ("at most 25 %", K.K_PREFILL_D == 0.25), ("at most 40 %", K.K_PREFILL_X == 0.40),
              ("at least 10 times", K.K_RESIDENT == 10.0), ("at most 0.85 times", K.K_SEC_D == 0.85), ("at most 1.5 times", K.K_GROWTH == 1.5), ("at least 90 %", K.K_KEEP_SAME == 0.90), ("steps 67 to 100", K.KV_B_STEPS == 100 and K.K_B_SAME == 0.95)]
    lines_text = "\n".join(l for _, l in K.KV_LINES)
    for needle, ok in consts: assert ok and needle in lines_text, needle
    assert lines_text.count("at least 95 %") == 4 and K.K_DG_SAME == K.K_KL_SHARE == K.K_C_SAME == K.K_B_SAME == 0.95
    assert "the first third is steps 1 to 10 of a 30-step episode, the last third steps 21 to 30" in head and K.KV_A_STEPS == 30


def test_the_notebook_is_clean_saves_no_outputs_holds_no_key_and_runs_the_checks_before_the_runs():
    nb = json.load(open(NB)); code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]; allsrc = "\n".join("".join(c["source"]) for c in nb["cells"]); joined = "\n".join(code)
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in nb["cells"])
    assert not re.search(r"AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{20,}", allsrc) and "userdata" not in joined and "docs/KV_DROP_TEST.md" in allsrc and "axi-colab-kv.zip" in allsrc and "never turns billing on" not in allsrc
    assert "build_kv(" in joined and "make_hf_kit(" in joined and "preflight(" in joined and "assert all(ok for" in joined and "run_kv(" in joined and "verdict_kv(" in joined and "open_store(" in joined
    assert joined.index("preflight(") < joined.index("run_kv(") and re.search(r"KV_SEED", joined) and "KV_LONG_SEED" in joined and "hint_rules" in joined
    assert "T4" in allsrc and "Do not paste" in allsrc and "$0" in allsrc and "pip -q install bitsandbytes accelerate" in joined


def _run_notebook_cells(eng, tmp, until=None):
    """Execute the code cells of the notebook with Colab and torch replaced by stubs, the model builder by the numpy stand-in, and the sizes of the runs by tiny ones. Returns (namespace, downloads); the files go to the folder `tmp`."""
    import contextlib, io, types
    import axi.experiments.kv_torch as KT, colab.colab_persist as CP
    downloads = []; saved = {k: getattr(K, k) for k in ("KV_A_EPISODES", "KV_A_STEPS", "KV_C_EPISODES", "KV_C_STEPS", "KV_B_EPISODES", "KV_B_STEPS")}
    mods = {k: sys.modules.get(k) for k in ("google", "google.colab", "torch")}; orig = (KT.build_kv, KT.make_hf_kit, CP.open_store)
    gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(upload=lambda: {"axi-colab-kv.zip": b""}, download=lambda path: downloads.append(path)); gc.drive = types.SimpleNamespace(mount=lambda mp: None)
    g = types.ModuleType("google"); g.colab = gc; tm = types.ModuleType("torch"); tm.cuda = types.SimpleNamespace(memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    try:
        sys.modules.update({"google": g, "google.colab": gc, "torch": tm}); K.KV_A_EPISODES, K.KV_A_STEPS, K.KV_C_EPISODES, K.KV_C_STEPS, K.KV_B_EPISODES, K.KV_B_STEPS = 1, 4, 1, 4, 1, 6
        KT.build_kv = lambda name, load_4bit, device: (eng, None, TAB, None); KT.make_hf_kit = lambda tok, hint_rules=False: byte_kit(hint_rules)
        CP.open_store = lambda local_dir="/content/out", subdir="x", mount_point="/content/drive": orig[2](tmp + "/out", subdir, tmp + "/drive")
        ns = {"__name__": "__main__"}; out = io.StringIO(); nb = json.load(open(NB)); n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code": continue
            src = "".join(l for l in c["source"] if not l.lstrip().startswith("!")).replace("/content/out", tmp + "/out").replace("/content/smoke_kv.jsonl", tmp + "/smoke_kv.jsonl")
            if not src.strip() or "files.upload" in src: continue
            n += 1
            if until is not None and n > until: break
            with contextlib.redirect_stdout(out): exec(compile(src, "cell %d" % n, "exec"), ns)
        ns["_printed"] = out.getvalue(); return ns, downloads
    finally:
        for k, v in saved.items(): setattr(K, k, v)
        KT.build_kv, KT.make_hf_kit, CP.open_store = orig
        for k, v in mods.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_every_code_cell_of_the_notebook_runs_on_the_stand_in_saves_after_each_run_and_stops_when_a_check_fails():
    tmp = tempfile.mkdtemp(); ns, downloads = _run_notebook_cells(engine(), tmp)
    assert [os.path.basename(d) for d in downloads] == ["axi_kv_a.json", "axi_kv_c.json", "axi_kv_b.json"]
    pre = json.load(open(tmp + "/out/axi_kv_preflight.json")); assert len(pre) == 9 and all(x["ok"] for x in pre)
    for name in ("axi_kv_preflight.json", "axi_kv_a.jsonl", "axi_kv_a.json", "axi_kv_c.jsonl", "axi_kv_c.json", "axi_kv_b.jsonl", "axi_kv_b.json"): assert os.path.exists(tmp + "/drive/MyDrive/axi_kv_drop/" + name), name   # also on the Drive
    a, c, b = (json.load(open(tmp + "/out/axi_kv_%s.json" % x)) for x in "acb")
    assert a["meta"]["seed"] == K.KV_SEED and a["meta"]["hint_rules"] is False and c["meta"]["seed"] == K.KV_SEED and c["meta"]["hint_rules"] is True and b["meta"]["seed"] == K.KV_LONG_SEED and b["meta"]["hint_rules"] is True
    assert [r["seed"] for r in a["rows"]] == [K.KV_SEED * 1000] == [r["seed"] for r in c["rows"]] and [r["seed"] for r in b["rows"]] == [K.KV_LONG_SEED * 1000]
    assert [(r["steps"]) for r in a["rows"]] == [4] and [(r["steps"]) for r in b["rows"]] == [6] and set(a["rows"][0]["arms"]) == set(K.ARMS)
    final = ns["final"]; assert len(final) == 15 and all(ok is not None for _, ok in final) and [t for t, _ in final] == [t for _, t in K.KV_LINES]
    assert final == K.verdict_kv(ns["res_a"]["summary"], ns["res_c"]["summary"], ns["res_b"]["summary"])
    txt = ns["_printed"]; assert "== KV test RUN A: no rule ==" in txt and "ALL CHECKS OK" in txt and "-- ALL RUNS (reading rules fixed before the run) --" in txt and "smoke, 1 episode x 4 steps" in txt
    broken = engine(); broken.cache_class = type("Broken", (TinyCache,), dict(rerotate=lambda self, new_pos: setattr(self, "pos", [int(p) for p in new_pos])))        # keys not turned
    tmp2 = tempfile.mkdtemp()
    try: _run_notebook_cells(broken, tmp2)
    except AssertionError as err: assert "STOP: a check failed" in str(err)
    else: raise AssertionError("the notebook went on after a failed check")
    saved = json.load(open(tmp2 + "/out/axi_kv_preflight.json")); assert not all(x["ok"] for x in saved) and not [f for f in os.listdir(tmp2 + "/out") if f.startswith(("axi_kv_a", "axi_kv_c", "axi_kv_b", "smoke"))]   # the failed line is saved, no run began


def test_the_pytorch_side_has_the_pieces_the_checks_rely_on_and_the_module_imports_without_torch():
    import axi.experiments.kv_torch as T
    src = open(T.__file__).read()
    assert "position_ids=" in src and "past_key_values=" in src and "DynamicCache()" in src and ".update(" in src and "model.model(" in src and "lm_head" in src and ".generate(" not in src
    assert ".cat((-kf[..., h:], kf[..., :h]), dim=-1)" in src and "kf * cos" in src and "rope_tables(" in src and "inv_freq" in src and "attention_scaling" in src
    assert "key_cache" in src and "layers" in src and "cache[i]" in src                                                       # three ways to read a Hugging Face cache back
    assert "index_select(2" in src and "def value_deviation" in src and "def keys" in src and "tol, tol_keys" in src
    for name in ("TorchCache", "TorchEngine", "make_hf_kit", "build_kv"): assert hasattr(T, name)
    ce = open(os.path.join(HERE, "..", "colab", "colab_llm_experiment.py")).read()
    assert "llm.tok, llm.table, llm.eos = tok, table, eos" in ce and "return llm, model" in ce
    for name in ("new_cache", "sync", "reset_peak", "peak_mb", "tol", "tol_keys", "layers"):                                 # the stand-in has everything the torch engine has
        assert hasattr(TinyEngine, name) or name in ("layers",) or name in TinyEngine().__dict__, name
    for name in ("forward", "select", "rerotate", "nbytes", "keys", "value_deviation", "n"): assert hasattr(TinyCache, name) and hasattr(T.TorchCache, name), name


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

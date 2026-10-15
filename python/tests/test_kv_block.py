"""Engineering support for docs/KV_BLOCK_TEST.md (concept-map item 99): can a stock model read every item of its state in a block of its own? No GPU, no torch, no real model. The arms, the cache protocol, the
scoring and the runner are `kv_drop.py`; the large request script is `kv_large.py`; this file tests what is new, on the numpy stand-in (`tests/kv_standin.py`):
  1. `forward_blocks` (the new cache operation): a block sees the header and itself and no other block, at the positions it is given; a block added later sees the header only; a single block is a normal read;
  2. the seven checks the notebook runs on the real model (`preflight_blocks`) pass on the stand-in, and each fails on a cache that is broken in the way it guards against;
  3. `BlockArm`: what it reads, what it keeps, the kept cache equals a cache rebuilt from nothing at the same positions (and the check fails for a cache that closes the gap or lets a new block see the others);
  4. the summary, the fourteen reading rules (each flips exactly at its threshold) and the dry-run figures the page quotes; the page is locked;
  5. the notebook runs end to end on the stand-in (both tests), saves after each run, never saves outputs, and stops when a check fails."""
import hashlib, inspect, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from kv_standin import TinyCache, TinyEngine, byte_kit, byte_table
import axi.experiments.kv_drop as K
import axi.experiments.kv_large as L
import axi.experiments.kv_block as B
import axi.experiments.kv_layout as Y
from axi.engine.cico_parser import CICOParser

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "KV_BLOCK_TEST.md")
PAGE_SHA256 = "8f893996d9df330886651ac4cecdd957da76d41a501e5a72e5bd7ecaa38d6b04"
NB = os.path.join(HERE, "..", "colab", "axi_tensor_tests.ipynb")
RESULT_MARK = "\n\n---\n\n# Result"
TIME_KEYS = {"update_s", "prefill_s", "decode_s", "seconds", "peak_mb"}
KIT, TAB = byte_kit(), byte_table()
SMALL = L.make_script_fn(14, 20, (5, 5), 18, 30)            # a graph the numpy stand-in can read in a moment (about 900 byte tokens)
BYTES = lambda s: len(s.encode("utf-8"))


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list): return [strip(v) for v in x]
    return x


def parse(ideal): return CICOParser.parse_transition_delta(ideal)


def raises(exc, fn, *a, **k):
    try: fn(*a, **k)
    except exc as ex: return ex
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have raised {exc.__name__}")


def rel(a, b): return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))


# ---------------------------------------------------------------- 1 forward_blocks on the stand-in
def _tokens(seed, n): return [int(x) for x in np.random.default_rng(seed).integers(0, 256, n)]


def _blocks(e, hdr, parts, positions=None, visible=None, base=None):
    c = K.Tracked(e.new_cache()); c.forward(hdr, range(len(hdr))); off, pos = len(hdr), []
    for p in parts: pos.append(list(range(off, off + len(p))) if positions is None else positions[len(pos)]); off += len(p)
    c.forward_blocks(parts, pos, visible); return c


def _rows(c, lo, hi, e): return [c.keys(i)[:, lo:hi] for i in range(e.layers)], [c.c.V[i][:, lo:hi] for i in range(e.layers)]


def _dist(x, y): return max(rel(a, b) for a, b in zip(x, y))


def test_forward_blocks_reads_each_block_with_the_header_and_itself_and_no_other_block():
    e = TinyEngine(); h = 12; hdr = _tokens(1, h); A, A2, Bk, C = _tokens(2, 5), _tokens(2, 5)[::-1], _tokens(3, 6), _tokens(4, 4); assert A != A2
    x, y = _blocks(e, hdr, [A, Bk]), _blocks(e, hdr, [A2, Bk]); la, lb = h + 5, h + 11
    assert x.n == lb and x.pos == list(range(lb)) and x.ids == hdr + A + Bk and y.ids == hdr + A2 + Bk
    kb_x, vb_x = _rows(x, la, lb, e); kb_y, vb_y = _rows(y, la, lb, e); ka_x, _ = _rows(x, h, la, e); ka_y, _ = _rows(y, h, la, e)
    assert _dist(kb_x, kb_y) < 1e-12 and _dist(vb_x, vb_y) < 1e-12                                                                  # B is the same whatever A holds
    assert _dist(ka_x, ka_y) > 1e-3                                                                                                 # ...and the check can tell: A's own rows differ
    cx, cy = K.Tracked(e.new_cache()), K.Tracked(e.new_cache()); cx.forward(hdr + A + Bk, range(lb)); cy.forward(hdr + A2 + Bk, range(lb))
    assert _dist(_rows(cx, la, lb, e)[0], _rows(cy, la, lb, e)[0]) > 1e-3                                                           # in a normal read B DOES depend on A
    n = K.Tracked(e.new_cache()); n.forward(hdr, range(h)); n.forward(A, range(h, la)); assert _dist(_rows(x, 0, la, e)[0], _rows(n, 0, la, e)[0]) < 1e-12        # the first block after the header is a normal read of it
    one = _blocks(e, hdr, [A + Bk]); assert _dist(_rows(one, 0, lb, e)[0], _rows(cx, 0, lb, e)[0]) < 1e-12                           # one block holding everything is the normal read
    assert inspect.signature(K.Tracked.forward_blocks).parameters["visible"].default is None and x.c.forward_blocks(  [A], [list(range(x.n, x.n + 5))]) is None


def test_forward_blocks_keeps_the_positions_it_is_given_and_a_block_added_later_sees_the_header_only():
    e = TinyEngine(); h = 12; hdr = _tokens(1, h); A, Bk, C = _tokens(2, 5), _tokens(3, 6), _tokens(4, 4)
    pa, pb = list(range(h, h + 5)), list(range(h + 9, h + 15))                                                                      # a hole of four positions between the blocks
    x = _blocks(e, hdr, [A, Bk], [pa, pb]); assert x.pos == list(range(h)) + pa + pb
    ref = K.Tracked(e.new_cache()); ref.forward(hdr, range(h)); ref.forward(Bk, pb); assert _dist(_rows(x, h + 5, h + 11, e)[0], _rows(ref, h, h + 6, e)[0]) < 1e-12      # B sits at its own positions, as if read alone after the header
    nohole = _blocks(e, hdr, [A, Bk]); assert _dist(_rows(x, h + 5, h + 11, e)[0], _rows(nohole, h + 5, h + 11, e)[0]) > 1e-3                                                    # the positions matter
    pc = list(range(h + 15, h + 19)); x.forward_blocks([C], [pc], visible=h); together = _blocks(e, hdr, [A, Bk, C], [pa, pb, pc]); n = h + 11                                                           # rows before the block that is added
    assert x.n == together.n == n + 4 and x.ids == together.ids and x.pos == together.pos and _dist([x.keys(i) for i in range(e.layers)], [together.keys(i) for i in range(e.layers)]) < 1e-12
    y = _blocks(e, hdr, [A, Bk], [pa, pb]); y.forward_blocks([C], [pc]); assert _dist(_rows(y, n, n + 4, e)[0], _rows(together, n, n + 4, e)[0]) > 1e-3             # visible=None: it sees the others, and is not the independent block
    alone = _blocks(e, hdr, [C], [pc]); assert _dist(_rows(together, n, n + 4, e)[0], _rows(alone, h, h + 4, e)[0]) < 1e-12
    x.select(list(range(h)) + list(range(h + 5, x.n))); assert x.n == x.c.n == h + 10 and x.ids == hdr + Bk + C and x.pos == list(range(h)) + pb + pc


def test_the_request_reads_every_block_and_the_forced_reply_counter_is_not_moved():
    e = TinyEngine(); h = 12; hdr = _tokens(1, h); A, A2, Bk = _tokens(2, 5), _tokens(2, 5)[::-1], _tokens(3, 6); tail = _tokens(9, 7)
    x, y = _blocks(e, hdr, [A, Bk]), _blocks(e, hdr, [A2, Bk]); lx, ly = x.forward(tail, range(x.n, x.n + 7)), y.forward(tail, range(y.n, y.n + 7)); assert float(np.abs(lx - ly).max()) > 1e-3
    c = K.Tracked(e.new_cache()); c.forward(hdr, range(h)); e.force = [5, 6, 7]; e.k = 2; c.forward_blocks([A, Bk], [list(range(h, h + 5)), list(range(h + 5, h + 11))]); assert e.k == 2          # tests only: reading blocks does not move the scripted-reply counter


# ---------------------------------------------------------------- 2 the seven checks on the real model
def _broken(**methods):
    e = TinyEngine(); e.cache_class = type("Broken", (TinyCache,), methods); return e


def _leaky(self, blocks, positions, visible=None):
    for b, p in zip(blocks, positions): self.forward(b, p)                                                                           # each block sees the ones before it: a normal read


def _blind(self, blocks, positions, visible=None):
    TinyCache.forward_blocks(self, blocks, positions, 0)                                                                             # the blocks do not see the header either


def _contiguous(self, blocks, positions, visible=None):
    n, v = (self.pos[-1] + 1 if self.pos else 0), (self.n if visible is None else visible)
    for b in blocks: TinyCache.forward_blocks(self, [b], [list(range(n, n + len(b)))], v); n += len(b)                              # the positions given are ignored


def _sees_everything(self, blocks, positions, visible=None):
    TinyCache.forward_blocks(self, blocks, positions, None)                                                                          # `visible` is ignored: a block added later sees all rows


def _deaf_blocks(self, blocks, positions, visible=None):
    TinyCache.forward_blocks(self, [[1] * len(b) for b in blocks], positions, visible)                                              # every block holds the same token: the rows of A cannot change with A's tokens


def _deaf_forward(self, tokens, positions):
    v0, e, last, rows = self.n, self.e, None, []                                                                                    # a normal read in which no token sees another token of the same call
    for t, p in zip(tokens, positions):
        tmp = TinyCache(e); tmp.K, tmp.V, tmp.pos = [k[:, :v0].copy() for k in self.K], [v[:, :v0].copy() for v in self.V], list(self.pos[:v0]); last = tmp.forward([t], [p]); rows.append(tmp)
    self.K = [np.concatenate([k] + [r.K[li][:, v0:] for r in rows], axis=1) for li, k in enumerate(self.K)]; self.V = [np.concatenate([v] + [r.V[li][:, v0:] for r in rows], axis=1) for li, v in enumerate(self.V)]
    self.pos += [int(p) for p in positions]; return last


def _failed(res): return [i + 1 for i, (_, ok, _) in enumerate(res) if not ok]


def test_the_seven_checks_pass_on_the_stand_in_and_each_has_teeth():
    res = B.preflight_blocks(TinyEngine(), KIT, B.BLOCK_SEED, SMALL); assert len(res) == 7 and all(ok for _, ok, _ in res), res
    assert len({n for n, _, _ in res}) == 7 and all(isinstance(d, str) and d for _, _, d in res)
    seen = []
    def spy(seed, steps): seen.append((seed, steps)); return SMALL(seed, steps)
    spy.__name__ = "spy"; B.preflight_blocks(TinyEngine(), KIT, B.BLOCK_SEED, spy); assert seen == [(21000, 3)]                         # the first prompt of the script it was given, at the seed of episode 0
    calls = []
    class Spy(TinyCache):
        def forward_blocks(self, blocks, positions, visible=None): calls.append([len(b) for b in blocks]); return TinyCache.forward_blocks(self, blocks, positions, visible)
    e = TinyEngine(); e.cache_class = Spy; B.preflight_blocks(e, KIT, B.BLOCK_SEED, SMALL); items = K.graph_items(SMALL(21000, 3)[0]); whole = sum(len(KIT.item(it)) for it in items[:40])
    assert calls[0] == [whole] and whole > 20 * len(KIT.item(items[0])), "the first check reads the whole state (the first 40 items) as ONE block"
    for name, eng, want in (("blocks see each other", _broken(forward_blocks=_leaky), [2, 5]), ("blocks do not see the header", _broken(forward_blocks=_blind), [1]),
                            ("positions ignored", _broken(forward_blocks=_contiguous), [5]), ("`visible` ignored", _broken(forward_blocks=_sees_everything), [6]),
                            ("blocks ignore their tokens", _broken(forward_blocks=_deaf_blocks), [1, 3, 7]), ("a normal read in which tokens do not see each other", _broken(forward=_deaf_forward), [1, 4])):
        got = _failed(B.preflight_blocks(eng, KIT, B.BLOCK_SEED, SMALL)); assert got == want, (name, got, want)                                  # exactly the lines that guard against that fault


# ---------------------------------------------------------------- 3 the arms
def walk(arm, seed, steps, script_fn=SMALL, each=None):
    g0, script, _ = script_fn(seed, steps); items = K.graph_items(g0); change, out = None, []
    for t, st in enumerate(script):
        rec, rep = arm.step(t, items, change, st["instruction"], None); rec["t"] = t + 1; out.append(rec)
        if each: each(t, arm)
        items, deleted, added = K.apply_delta_items(items, parse(st["ideal"])); change = (deleted, added)
    return out


def test_the_block_arm_equals_a_cache_rebuilt_from_nothing_at_the_same_positions_and_the_check_has_teeth():
    arm = K.make_arms(TinyEngine(), KIT, TAB, ["fresh", "block"], max_new=4)["block"]; assert arm.name == "block" and arm.maintained and arm.group == 1 and K.BlockArm.CHECK_EVERY == 10
    a16 = K.make_arms(TinyEngine(), KIT, TAB, ["fresh", "block16"], max_new=4)["block16"]; its = K.graph_items(SMALL(31000, 3)[0]); toks, pos = a16._blocks(its)
    assert a16.name == "block16" and a16.group == 16 and not a16.maintained and len(its) > 32 and [len(b) for b in toks] == [sum(len(KIT.item(it)) for it in its[i:i + 16]) for i in range(0, len(its), 16)]
    assert [x for b in toks for x in b] == [x for it in its for x in KIT.item(it)] and [x for p_ in pos for x in p_] == list(range(len(KIT.first_prefix), len(KIT.first_prefix) + sum(len(b) for b in toks)))
    recs = walk(arm, 31000, 12); checked = [r["t"] for r in recs if "rebuilt_dev_v" in r]; assert checked == [1, 11]                       # steps 1, 11 and 21 of an episode (t counts from 1 in the record)
    assert all(r["rebuilt_dev_v"] < 1e-12 and r["rebuilt_dev_k"] < 1e-12 for r in recs if "rebuilt_dev_v" in r) and recs[-1]["gap"] > 0 and all(r["gap"] >= 0 for r in recs)
    assert any(st["kind"] in ("delete_dep", "delete_leaf") for st in SMALL(31000, 10)[1]) and any(st["kind"].startswith("add") for st in SMALL(31000, 10)[1])         # a delete and an add came before step 11: the check is not vacuous
    mono = []; walk(K.BlockArm(TinyEngine(), KIT, TAB, 4, group=1, maintained=True), 31000, 12, each=lambda t, a: mono.append(all(y > x for x, y in zip(a.cache.pos, a.cache.pos[1:])) and a.cache.pos[:len(KIT.first_prefix)] == list(range(len(KIT.first_prefix)))))
    assert len(mono) == 12 and all(mono), "every row has its own position, increasing, and the header is at 0 to h"
    # a cache that closes the gap after a cut is NOT the rebuild at the new positions (deeper layers were computed at the old distance): the check says so
    class Closing(K.BlockArm):
        def _update(self, items, change):
            r = super()._update(items, change)
            if change[0]: self.cache.rerotate(range(self.cache.n))
            return r
    bad = walk(Closing(TinyEngine(), KIT, TAB, 4, group=1, maintained=True), 31000, 12); assert max(r["rebuilt_dev_v"] for r in bad if "rebuilt_dev_v" in r) > 1e-3 and max(r["rebuilt_dev_k"] for r in bad if "rebuilt_dev_k" in r) > 1e-3
    # a new block that sees the other blocks is not an independent block: the check says so
    def see_all(t, a):
        if t == 0: orig = a.cache.forward_blocks; a.cache.forward_blocks = lambda blocks, positions, visible=None: orig(blocks, positions, None)
    bad = walk(K.BlockArm(TinyEngine(), KIT, TAB, 4, group=1, maintained=True), 31000, 12, each=see_all); assert max(r["rebuilt_dev_v"] for r in bad if "rebuilt_dev_v" in r) > 1e-3
    raises(AssertionError, K.BlockArm, TinyEngine(), KIT, TAB, 4, group=2, maintained=True)


def test_what_the_block_arms_read_is_what_the_page_says_and_their_rows_are_not_a_normal_read():
    e = TinyEngine(); res = B.run_block(e, KIT, TAB, 2, 12, 5, False, os.path.join(tempfile.mkdtemp(), "b.jsonl"), meta=dict(model="stand-in"), max_new=4, script_fn=SMALL); rows = res["rows"]
    assert set(rows[0]["arms"]) == set(B.BLOCK_ARMS) == {"fresh", "drop_compact", "block", "block16"} and res["meta"]["script"] == SMALL.__name__ and res["meta"]["arms"] == list(B.BLOCK_ARMS)
    for r in rows:
        f, dc, bk, b16 = (r["arms"][a] for a in ("fresh", "drop_compact", "block", "block16")); assert len(f) == len(dc) == len(bk) == len(b16) == 12
        assert [s["prefilled"] for s in bk] == [s["prefilled"] for s in dc]                                                           # block appends the same items and the same request and reads nothing again
        assert [s["prefilled"] for s in b16] == [s["prefilled"] for s in f]                                                           # block16 is rebuilt every step: it reads the whole prompt
        assert [s["resident"] for s in bk] == [s["resident"] for s in f] and [s["resident"] for s in b16] == [s["resident"] for s in f]   # the same rows are held
        assert all("gap" in s for s in bk) and not any("gap" in s for s in b16) and all("stale" not in s for s in bk) and all(s["v_dev"] > 1e-3 for s in bk + b16)                  # the rows are not those of a normal read (it is a different model of the state)
        assert all(s["canonical"] for s in f) and all(s["same_as_fresh"] is not None for s in bk + b16 + dc) and bk[0]["t"] == 1
    dry = L.dry_run(SMALL, [r["seed"] for r in rows], 12, BYTES); tot = lambda arm: sum(s["prefilled"] for r in rows for s in r["arms"][arm])
    assert dry["tokens"]["drop"] == tot("block") == tot("drop_compact") and dry["tokens"]["fresh"] == tot("fresh") == tot("block16")                                              # the dry run speaks for the real arms
    s = res["summary"]; assert s["block"]["rebuilt_checks"] == 4 and s["block"]["rebuilt_dev_v_max"] < 1e-12 and s["block"]["rebuilt_dev_k_max"] < 1e-12 and s["drop_compact"]["rebuilt_checks"] == 0 and s["drop_compact"]["rebuilt_dev_v_max"] is None
    calls = e.calls; again = B.run_block(e, KIT, TAB, 2, 12, 5, False, os.path.join(tempfile.mkdtemp(), "again.jsonl"), max_new=4, script_fn=SMALL)           # a new file: runs again, with the same result
    assert e.calls > calls and strip(again["rows"]) == strip(rows)
    assert B.block_dry_run(1, 3, count=BYTES) == L.dry_run(L.large_script, [21000], 3, BYTES) and inspect.signature(B.run_block).parameters["script_fn"].default is None
    raises(AssertionError, K.make_arms, TinyEngine(), KIT, TAB, ["fresh", "block32"])


# ---------------------------------------------------------------- 4 summaries, reading rules, the dry-run figures, the page
def _step(t, same=True, kl=0.01, **kw):
    d = dict(t=t, same_as_fresh=same, kl_max=kl, kl_mean=None if kl is None else kl / 2, exact_m6=True, exact_blind=True, admitted=True, parsed=True, prefilled=50, decoded=5, resident=100, resident_state=90, kv_bytes=1e6, peak_mb=10.0,
             seconds=1.0, update_s=0.1, prefill_s=0.2, decode_s=0.7, kind="add_node", v_dev=0.1); d.update(kw); return d


def test_the_summary_adds_the_share_below_half_a_nat_and_the_largest_difference_from_a_rebuild():
    steps = [_step(1, kl=0.2, rebuilt_dev_v=0.0, rebuilt_dev_k=0.0), _step(2, kl=0.5), _step(3, kl=0.49), _step(4, kl=1.5), _step(11, kl=0.1, rebuilt_dev_v=0.003, rebuilt_dev_k=0.04)]
    rows = [dict(seed=0, steps=5, stopped=dict(fresh=None, block=None), arms=dict(fresh=[_step(t, None, None, prompt_tokens=100 + t) for t in range(1, 6)], block=steps))]; s = B.summarize_block(rows)["block"]
    assert s["kl_below_b"] == 3 / 5 and s["rebuilt_checks"] == 2 and s["rebuilt_dev_v_max"] == 0.003 and s["rebuilt_dev_k_max"] == 0.04 and s["kl_max"] == 1.5                           # KL = 0.5 is not below 0.5
    f = B.summarize_block(rows)["fresh"]; assert f["kl_below_b"] is None and f["rebuilt_checks"] == 0 and f["rebuilt_dev_v_max"] is None and abs(f["prompt_tokens_mean"] - 103) < 1e-9
    base = L.summarize_large(rows)
    for a in base:
        for k, v in base[a].items(): assert B.summarize_block(rows)[a][k] == v, (a, k)                                                                                       # everything the large summary has is kept


def _summ(**o):
    A = dict(fresh=dict(n=90, prompt_tokens_mean=3800.0, canonical=1.0, exact_m6=0.90, prefilled=100000, seconds=10.0), drop_compact=dict(n=90, v_dev=0.02),
             block=dict(n=90, same=0.95, exact_m6=0.90, kl_below_b=0.95, rebuilt_dev_v_max=0.0, rebuilt_dev_k_max=0.0, v_dev=0.30, prefilled=5000, seconds=4.0), block16=dict(n=90, same=0.95))
    Bb = dict(fresh=dict(n=60, exact_m6=0.80, exact_blind=0.70), block=dict(n=60, exact_m6=0.78, exact_blind=0.68, same=0.75))
    for k, v in o.items():
        tgt = {"A": A, "B": Bb}[k[0]]; arm, key = k[1:].split("__"); tgt[arm][key] = v
    return A, Bb


def _bad(**o): A, Bb = _summ(**o); return [i + 1 for i, (_, ok) in enumerate(B.verdict_block(A, Bb)) if ok is not True]


def test_the_fourteen_reading_rules_flip_exactly_at_their_thresholds():
    A, Bb = _summ(); base = B.verdict_block(A, Bb); assert len(base) == len(B.BLOCK_LINES) == 14 and all(ok is True for _, ok in base) and [t for t, _ in base] == [t for _, t in B.BLOCK_LINES]
    assert [t for t, _ in B.BLOCK_LINES].count("instrument") == 3 and [t for t, _ in B.BLOCK_LINES].count("guard") == 1 and [t for t, _ in B.BLOCK_LINES].count("claim") == 10
    flips = [({"Afresh__prompt_tokens_mean": 2999.9}, [1]), ({"Afresh__prompt_tokens_mean": 3000.0}, []), ({"Afresh__canonical": 0.999}, [2]), ({"Ablock__rebuilt_dev_v_max": 0.0100001}, [3]), ({"Ablock__rebuilt_dev_v_max": 0.01}, []),
             ({"Ablock__rebuilt_dev_k_max": 0.0500001}, [3]), ({"Ablock__rebuilt_dev_k_max": 0.05}, []), ({"Afresh__exact_m6": 0.8999, "Ablock__exact_m6": 0.8699}, [4]), ({"Afresh__exact_m6": 0.90}, []),
             ({"Ablock__same": 0.8999}, [5]), ({"Ablock__same": 0.90}, []), ({"Ablock__exact_m6": 0.87}, []), ({"Ablock__exact_m6": 0.8699}, [6]), ({"Ablock__kl_below_b": 0.8999}, [7]), ({"Ablock__kl_below_b": 0.90}, []),
             ({"Ablock16__same": 0.8999}, [8]), ({"Ablock__v_dev": 0.02}, [9]), ({"Ablock__v_dev": 0.0201}, []), ({"Ablock__prefilled": 10000}, []), ({"Ablock__prefilled": 10001}, [10]),
             ({"Ablock__seconds": 5.0}, []), ({"Ablock__seconds": 5.0001}, [11]), ({"Bblock__exact_m6": 0.75}, []), ({"Bblock__exact_m6": 0.7499}, [12]), ({"Bblock__exact_blind": 0.65}, []), ({"Bblock__exact_blind": 0.6499}, [13]),
             ({"Bblock__same": 0.70}, []), ({"Bblock__same": 0.6999}, [14])]
    for o, want in flips: assert _bad(**o) == want, (o, _bad(**o), want)
    v = B.verdict_block(A, None); assert [ok for _, ok in v][11:] == [None, None, None] and all(ok is True for _, ok in v[:11])                      # run B2 not made: its three lines read 'not run'
    assert B.verdict_block({}, None) == [(t, None) for _, t in B.BLOCK_LINES] and B.verdict_block({"fresh": A["fresh"]})[2][1] is None and B.verdict_block({"fresh": A["fresh"]})[8][1] is None
    assert B.BLOCK_HEADLINE == (5, 6, 10, 11)


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_numbers_the_page_quotes_from_the_script_alone_are_the_ones_the_code_computes():
    head = _head(); a = B.block_dry_run(); b = B.block_dry_run(B.BLOCK_B2_EPISODES, B.BLOCK_B2_STEPS)
    assert round(a["prompt_first"], -2) == 3800 and round(100 * a["drop_over_fresh"], 1) == 4.5 and round(100 * b["drop_over_fresh"], 1) == 6.1 and a["prompt_first"] > B.L_MIN_PROMPT
    for needle in ("about 3,800 tokens", "reads **4.5 %** of what `fresh` reads over run B1 (3 x 30 steps, first builds included) and 6.1 % over run B2 (3 x 20)"): assert needle in head, needle
    assert a["drop_over_fresh"] <= B.B_PREFILL                                                                                                  # the line the dry run speaks to would read yes
    assert (B.BLOCK_SEED, B.BLOCK_B1_EPISODES, B.BLOCK_B1_STEPS, B.BLOCK_B2_EPISODES, B.BLOCK_B2_STEPS) == (21, 3, 30, 3, 20) and B.BLOCK_ARMS == ("fresh", "drop_compact", "block", "block16")
    assert [B.BLOCK_SEED * 1000 + e for e in range(3)] == [21000, 21001, 21002] and L.LARGE_SEED == L.LARGE_C_SEED == B.BLOCK_SEED


def test_the_page_is_locked_and_carries_the_fourteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(B.BLOCK_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [97, 95, 93, 95, 55, 65, 50, 65, 85, 95, 85, 40, 55, 40]
    p = {i + 1: v for i, v in enumerate(probs)}; prod = 100 * np.prod([p[i] / 100 for i in B.BLOCK_HEADLINE]); assert abs(prod - 29) <= 0.5 and "about 29 %" in head and "about **40 %**" in head and "**Headline** (lines 5, 6, 10 and 11)" in head
    for needle in ("Concept-map item 99", "(item 98)", "(item 96)", "(item 97)", "Qwen2.5-7B-Instruct, 4-bit NF4", "at most 120 new tokens", "seed 21, episode seeds 21000 to 21002", "**Run B1 (no rule):** 3 episodes x 30 steps", "**Run B2 (rule stated, optional):** 3 episodes x 20 steps",
                   "`fresh`", "`drop_compact`", "`block`", "`block16`", "`CHECK_EVERY` = 10", "`kv_block.preflight_blocks`", "`kv_block.verdict_block`", "`kv_block.block_dry_run`", "seven lines", "**Cost: $0.**", "Prompt Cache", "CacheBlend", "Block-Attention",
                   "I have not checked with Rajnish Choubey that it is his formulation; I do not attribute it to him", "axi_tensor_tests.ipynb", "test_kv_block.py", "Cells 9 and 10", "LAYOUT_TEST.md", "79.4 % against 82.5 %", "11 steps lost, 6 gained",
                   "93 % (Run A) and 92 % (Run C) of the state rows were stale", "4 of 52 delete replies exactly right", "recall 0.23", "14 of 52", "71.9 %", "37 % on the 71 delete steps", "checked 14 Oct 2026", "no pre-training required"):
        assert needle in head, needle
    assert (B.B_REBUILT_V, B.B_REBUILT_K, B.B_FRESH_M6, B.B_SAME, B.B_M6_POINTS, B.B_KL, B.B_KL_SHARE, B.B_PREFILL, B.B_SEC, B.B_B2_M6_POINTS, B.B_B2_BLIND_POINTS, B.B_B2_SAME) == (0.01, 0.05, 0.90, 0.90, 0.03, 0.5, 0.90, 0.10, 0.50, 0.05, 0.05, 0.70)
    lines = "\n".join(l for _, l in B.BLOCK_LINES)
    for needle in ("at least 3,000 tokens", "at most 0.01", "at most 0.05", "at least 90 %", "at most 3 points below", "below 0.5 nats", "at most 10 %", "at most 0.5 times", "at most 5 points below", "at least 70 %"): assert needle in lines, needle


# ---------------------------------------------------------------- 5 the notebook (both tests), end to end on the stand-in
def _run_notebook_cells(eng, tmp, until=None):
    """Execute the code cells of the notebook with Colab and torch replaced by stubs, the model builder by the numpy stand-in, and the sizes of the runs, the graphs and the problem set by tiny ones."""
    import contextlib, io, types
    import axi.experiments.kv_torch as KT, colab.colab_persist as CP
    downloads = []; names = {B: ("BLOCK_B1_EPISODES", "BLOCK_B1_STEPS", "BLOCK_B2_EPISODES", "BLOCK_B2_STEPS"), L: ("large_script",), Y: ("make_problems",)}; saved = {(m, k): getattr(m, k) for m, ks in names.items() for k in ks}
    mods = {k: sys.modules.get(k) for k in ("google", "google.colab", "torch")}; orig = (KT.build_kv, KT.make_hf_kit, CP.open_store)
    gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(upload=lambda: {"axi-colab-tensor.zip": b""}, download=lambda path: downloads.append(path)); gc.drive = types.SimpleNamespace(mount=lambda mp: None)
    g = types.ModuleType("google"); g.colab = gc; tm = types.ModuleType("torch"); tm.cuda = types.SimpleNamespace(memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    small_problems = Y.make_problems
    try:
        sys.modules.update({"google": g, "google.colab": gc, "torch": tm}); B.BLOCK_B1_EPISODES, B.BLOCK_B1_STEPS, B.BLOCK_B2_EPISODES, B.BLOCK_B2_STEPS = 1, 3, 1, 3; L.large_script = SMALL
        Y.make_problems = lambda *a, **k: small_problems(seed=5, graphs=2, deps=1, controls=2, nodes=14, edges=22, grid=(5, 5))
        KT.build_kv = lambda name, load_4bit, device: (eng, None, TAB, None); KT.make_hf_kit = lambda tok, hint_rules=False: byte_kit(hint_rules)
        CP.open_store = lambda local_dir="/content/out", subdir="x", mount_point="/content/drive": orig[2](tmp + "/out", subdir, tmp + "/drive")
        ns = {"__name__": "__main__"}; out = io.StringIO(); nb = json.load(open(NB)); n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code": continue
            src = "".join(l for l in c["source"] if not l.lstrip().startswith("!")).replace("/content/out", tmp + "/out").replace("/content/smoke_", tmp + "/smoke_")
            if not src.strip() or "files.upload" in src: continue
            n += 1
            if until is not None and n > until: break
            with contextlib.redirect_stdout(out): exec(compile(src, "cell %d" % n, "exec"), ns)
        ns["_printed"] = out.getvalue(); return ns, downloads
    finally:
        for (m, k), v in saved.items(): setattr(m, k, v)
        KT.build_kv, KT.make_hf_kit, CP.open_store = orig
        for k, v in mods.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_the_notebook_is_clean_saves_no_outputs_holds_no_key_and_runs_the_checks_before_the_runs():
    nb = json.load(open(NB)); cells = nb["cells"]; code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]; allsrc = "\n".join("".join(c["source"]) for c in cells); joined = "\n".join(code)
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in cells) and not re.search(r"AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{20,}", allsrc) and "userdata" not in joined
    for needle in ("docs/LAYOUT_TEST.md", "docs/KV_BLOCK_TEST.md", "axi-colab-tensor.zip", "T4", "Do not paste", "$0", "13 reading rules", "14 reading rules"): assert needle in allsrc, needle
    assert "build_kv(" in joined and "make_hf_kit(" in joined and "preflight(engine, kit_plain, BLOCK_SEED, large_script)" in joined and "preflight_blocks(engine, kit_plain, BLOCK_SEED, large_script)" in joined and "assert all(ok for" in joined
    assert "run_block(" in joined and "run_layout(" in joined and "verdict_block(" in joined and "verdict_layout(" in joined and "open_store(" in joined and "Projection:" in joined
    assert joined.index("preflight(") < joined.index("run_layout(") < joined.index("run_block(") and joined.index("preflight_blocks(") < joined.index("run_layout(") and "make_hf_kit(tok, hint_rules=False)" in joined and "make_hf_kit(tok, hint_rules=True)" in joined
    run_b = [c for c in code if "run_block(" in c and "BLOCK_B1" in c and "files.download" in c]; run_b2 = [c for c in code if "run_block(" in c and "BLOCK_B2" in c and "files.download" in c]; assert len(run_b) == 1 and len(run_b2) == 1 and "kit_plain" in run_b[0] and "False" in run_b[0] and "kit_rule" in run_b2[0] and "True" in run_b2[0]
    src = open(B.__file__).read(); assert "torch" not in src and "import transformers" not in src
    kt = open(os.path.join(HERE, "..", "axi", "experiments", "kv_torch.py")).read(); assert "def forward_blocks" in kt and "visible" in kt
    fb = kt[kt.index("def forward_blocks"):kt.index("def select", kt.index("def forward_blocks"))]; assert fb.count(".clone()") >= 2 and "new.append([(a[:, :, v0:].clone(), b[:, :, v0:].clone()) for a, b in" in fb and "v0 = self.n if visible is None else int(visible)" in fb and "self._hf(v0)" in fb and "position_ids=p" in fb


def test_every_code_cell_of_the_notebook_runs_on_the_stand_in_saves_after_each_run_and_stops_when_a_check_fails():
    tmp = tempfile.mkdtemp(); ns, downloads = _run_notebook_cells(TinyEngine(), tmp)
    assert [os.path.basename(d) for d in downloads] == ["axi_layout.json", "axi_block_b1.json", "axi_block_b2.json"]
    pre = json.load(open(tmp + "/out/axi_tensor_preflight.json")); assert len(pre) == 9 + 7 + 1 and all(x["ok"] for x in pre)
    for name in ("axi_tensor_preflight.json", "axi_layout.jsonl", "axi_layout.json", "axi_block_b1.jsonl", "axi_block_b1.json", "axi_block_b2.jsonl", "axi_block_b2.json"): assert os.path.exists(tmp + "/drive/MyDrive/axi_tensor/" + name), name
    lay, b1, b2 = (json.load(open(tmp + "/out/" + n)) for n in ("axi_layout.json", "axi_block_b1.json", "axi_block_b2.json"))
    assert lay["meta"]["kind"] == "kv_layout" and len(lay["rows"]) == 4 and set(lay["summary"]) == set(Y.CONDITIONS) and lay["meta"]["conditions"] == list(Y.CONDITIONS)
    assert b1["meta"]["seed"] == B.BLOCK_SEED and b1["meta"]["hint_rules"] is False and b2["meta"]["seed"] == B.BLOCK_SEED and b2["meta"]["hint_rules"] is True and b1["meta"]["arms"] == list(B.BLOCK_ARMS) == b2["meta"]["arms"]
    assert [r["seed"] for r in b1["rows"]] == [21000] == [r["seed"] for r in b2["rows"]] and [r["steps"] for r in b1["rows"]] == [3] and b1["meta"]["script"] == SMALL.__name__ and set(b1["rows"][0]["arms"]) == set(B.BLOCK_ARMS)
    assert "rebuilt_checks" in b1["summary"]["block"] and b1["summary"]["block"]["rebuilt_checks"] == 1
    fl, fb = ns["final_layout"], ns["final_block"]; assert len(fl) == 13 and len(fb) == 14 and [t for t, _ in fl] == [t for _, t in Y.LAYOUT_LINES] and [t for t, _ in fb] == [t for _, t in B.BLOCK_LINES]
    assert fb == B.verdict_block(ns["res_b1"]["summary"], ns["res_b2"]["summary"]) and fl == Y.verdict_layout(ns["res_layout"]["summary"])
    txt = ns["_printed"]; assert "ALL CHECKS OK" in txt and "== layout test" in txt and "== KV test" in txt and "-- LAYOUT TEST, ALL LINES" in txt and "-- BLOCK TEST, ALL LINES" in txt and "Projection:" in txt and "kept block cache against one rebuilt" in txt
    sm = ns["smoke_block"]; assert sm["rows"][0]["steps"] == 3 and sm["meta"]["seed"] == 99 and sm["meta"]["smoke"] is True
    broken = _broken(forward_blocks=_leaky); tmp2 = tempfile.mkdtemp()
    try: _run_notebook_cells(broken, tmp2)
    except AssertionError as err: assert "STOP: a check failed" in str(err)
    else: raise AssertionError("the notebook went on after a failed check")
    saved = json.load(open(tmp2 + "/out/axi_tensor_preflight.json")); assert not all(x["ok"] for x in saved) and not [f for f in os.listdir(tmp2 + "/out") if f.startswith(("axi_layout", "axi_block", "smoke"))]


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

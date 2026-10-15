"""Block independence on a STOCK model: what happens to a model's answers when every item of the state is read in its own block, with no block seeing any other? Page with the predictions written
BEFORE any run: docs/KV_BLOCK_TEST.md (concept-map item 99).

Where the idea comes from: a text the owner shared on 15 Oct 2026 (section 2, "sheaf-factorized block attention"; its table says "no pre-training required: applied as an attention mask at test time").
Nothing in the text says it was measured. The text's claim is tested here in the only form a stock open model allows without training: the key/value rows of every item are computed on their own
(the header in view, the other items NOT in view), then put side by side, and the request reads them all. If a stock model keeps its answers, then (a) deleting an item is EXACT (no stale rows: nothing
was computed in the item's presence) and (b) adding an item costs only that item. If it does not, the claim "no pre-training required" is false for this model and this task.

It is the test of docs/KV_DROP_TEST.md and docs/KV_LARGE_TEST.md (`kv_drop.py`: same arms machinery, same cache protocol, same prompt layout, same scoring, the large request script of `kv_large.py`) with
two new arms (`BlockArm` in `kv_drop.py`):
  block    every item its own block, the cache kept between steps (a cut leaves no stale rows; the positions keep their holes like drop_gap; `rebuilt_dev_v` and `rebuilt_dev_k` compare the kept cache with one
           rebuilt from nothing at the same positions);
  block16  blocks of 16 consecutive items, rebuilt every step (accuracy only: it shows whether bigger blocks lose less).
and `fresh` (a normal read; the reference) and `drop_compact` (the cut-out cache of the large test, for comparison), on the first steps of the same request scripts as Run A and Run C of the large test.
Run B1: no rule in the prompt (the replies are copies of the request; Run A of the large test); Run B2: the rule stated (the reply must find the edges of a service; Run C)."""
from typing import Callable, List, Optional, Tuple

import numpy as np

from axi.experiments.kv_drop import KV_MAX_NEW, Tracked, _fx, _ge, _le, _mean, _pc, graph_items, run_kv
from axi.experiments.kv_large import L_MIN_PROMPT, dry_run, large_script, print_large, summarize_large

BLOCK_SEED = 21                       # the same scripts as Run A and Run C of the large test (episode seeds 21000, 21001, 21002; the first steps of them)
BLOCK_B1_EPISODES, BLOCK_B1_STEPS = 3, 30
BLOCK_B2_EPISODES, BLOCK_B2_STEPS = 3, 20
BLOCK_ARMS = ("fresh", "drop_compact", "block", "block16")

# thresholds of the reading rules (docs/KV_BLOCK_TEST.md)
B_REBUILT_V, B_REBUILT_K, B_FRESH_M6, B_SAME, B_M6_POINTS, B_KL, B_KL_SHARE, B_PREFILL, B_SEC, B_B2_M6_POINTS, B_B2_BLIND_POINTS, B_B2_SAME = 0.01, 0.05, 0.90, 0.90, 0.03, 0.5, 0.90, 0.10, 0.50, 0.05, 0.05, 0.70
BLOCK_HEADLINE = (5, 6, 10, 11)
BLOCK_LINES = [
    ("instrument", "The graph is large: the prompt of `fresh` holds on average at least 3,000 tokens over the steps of run B1."),
    ("instrument", "The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run B1."),
    ("instrument", "The kept block cache equals one rebuilt from nothing at the same positions: at every step where this was checked (run B1) the mean relative difference of the values is at most 0.01 and that of the keys at most 0.05."),
    ("guard", "fresh: the share of steps on which repair_m6 turns the reply into the right next state is at least 90 % (run B1)."),
    ("claim", "block: the reply is identical to fresh's on at least 90 % of the steps of run B1."),
    ("claim", "block: the share of steps on which repair_m6 turns the reply into the right next state is at most 3 points below fresh's (run B1)."),
    ("claim", "block: on at least 90 % of the steps of run B1 the largest KL along the reply, against fresh's distribution, is below 0.5 nats."),
    ("claim", "block16: the reply is identical to fresh's on at least 90 % of the steps of run B1."),
    ("claim", "block: the mean relative difference of its values from fresh's is larger than drop_compact's (run B1): leaving out all other items is a bigger change than a stale tail."),
    ("claim", "block reads at most 10 % of the tokens that fresh reads (run B1, whole episodes, the first build of each included)."),
    ("claim", "block: mean seconds per step at most 0.5 times fresh's (run B1)."),
    ("claim", "run B2 (rule stated): the share of steps on which repair_m6 turns the reply into the right next state is, for block, at most 5 points below fresh's."),
    ("claim", "run B2 (rule stated): the share of replies that are exactly right when applied as written is, for block, at most 5 points below fresh's."),
    ("claim", "run B2 (rule stated): block's reply is identical to fresh's on at least 70 % of the steps."),
]


# ---------------------------------------------------------------- what the design alone fixes
def block_dry_run(episodes: int = BLOCK_B1_EPISODES, steps: int = BLOCK_B1_STEPS, seed: int = BLOCK_SEED, count=None) -> dict:
    """`kv_large.dry_run` for the scripts of this page. `drop_over_fresh` is also the share of tokens that the kept `block` arm reads (it appends the same items and the same request and re-reads nothing)."""
    kw = {} if count is None else dict(count=count)
    return dry_run(large_script, [seed * 1000 + e for e in range(episodes)], steps, **kw)


# ---------------------------------------------------------------- the checks the notebook runs on the real model first
def preflight_blocks(engine, kit, seed: int = BLOCK_SEED, script_fn: Callable = large_script) -> List[Tuple[str, bool, str]]:
    """Checks of `forward_blocks` on one real prompt. Each returns (name, ok, detail). Without them a mistake in how the blocks are read (a block seeing another block, a cut that leaves a trace, a
    block read at the wrong positions) would look like a finding about the model."""
    g0, script, _ = script_fn(seed * 1000, 3); items = graph_items(g0); kit_ = kit; hdr = kit_.first_prefix; h = len(hdr); res, tol, tolk = [], engine.tol, engine.tol_keys
    A, B, C = (kit_.item(items[i]) for i in (20, 21, 22)); tail = kit_.tail(script[0]["instruction"]); full = lambda: Tracked(engine.new_cache())
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300)); layers = sorted({0, engine.layers // 2, engine.layers - 1})
    rows = lambda c, lo, hi: [c.keys(i)[:, lo:hi] for i in layers]
    dist = lambda xs, ys: max(rel(a, b) for a, b in zip(xs, ys))

    def blocks(parts, positions=None, visible=None):
        c = full(); c.forward(hdr, range(h)); off, pos = h, []
        for p in parts: pos.append(list(range(off, off + len(p))) if positions is None else positions[len(pos)]); off += len(p)
        c.forward_blocks(parts, pos, visible); return c

    def causal(parts):
        c = full(); flat = [x for p in parts for x in p]; c.forward(hdr + flat, range(h + len(flat))); return c

    # 1 one block holding the whole state is a normal read
    state = [x for it in items[:40] for x in kit_.item(it)]; c1 = full(); c1.forward(hdr, range(h)); c1.forward_blocks([state], [list(range(h, h + len(state)))]); l1 = c1.forward(tail, range(h + len(state), h + len(state) + len(tail)))
    c2 = full(); c2.forward(hdr + state, range(h + len(state))); l2 = c2.forward(tail, range(h + len(state), h + len(state) + len(tail))); d = float(np.abs(l1 - l2).max())
    res.append(("one block that holds the whole state, read after the header, gives the same logits as reading the prompt at once", d <= tol, f"max difference {d:.2e} (allowed {tol:.0e})"))
    # 2 independence
    A2 = A[::-1]; assert A2 != A and len(A2) == len(A)
    x, y = blocks([A, B]), blocks([A2, B]); la, lb = h + len(A), h + len(A) + len(B)
    dB = dist(rows(x, la, lb), rows(y, la, lb)); dA = dist(rows(x, h, la), rows(y, h, la))
    res.append(("the rows of block B do not change when block A has other tokens (blocks read side by side)", dB <= tolk, f"largest relative difference of B's keys {dB:.2e} (allowed {tolk:.0e})"))
    res.append(("...and the check can tell: the rows of block A itself do change", dA > 3 * tolk, f"largest relative difference of A's keys {dA:.2e} (needs more than {3 * tolk:.0e})"))
    x, y = causal([A, B]), causal([A2, B]); dC = dist(rows(x, la, lb), rows(y, la, lb))
    res.append(("...and in a normal read the rows of B DO change when A changes", dC > 3 * tolk, f"largest relative difference of B's keys {dC:.2e} (needs more than {3 * tolk:.0e})"))
    # 3 a cut leaves no trace (same positions)
    x = blocks([A, B, C]); pa, pb, pc = list(range(h, h + len(A))), list(range(h + len(A), h + len(A) + len(B))), list(range(h + len(A) + len(B), h + len(A) + len(B) + len(C)))
    x.select(list(range(h)) + list(range(h + len(A), h + len(A) + len(B) + len(C)))); y = blocks([B, C], [pb, pc]); n = x.n
    d = dist([x.keys(i) for i in layers], [y.keys(i)[:, :n] for i in layers])
    res.append(("cutting block A out leaves exactly the cache of a build without A, at the same positions", d <= tolk and x.n == y.n, f"largest relative difference of the keys {d:.2e} (allowed {tolk:.0e})"))
    # 4 a block added later equals the same block built together with the others
    x = blocks([A, B]); pc2 = list(range(h + len(A) + len(B), h + len(A) + len(B) + len(C))); x.forward_blocks([C], [pc2], visible=h); y = blocks([A, B, C])
    d = dist([x.keys(i) for i in layers], [y.keys(i) for i in layers])
    res.append(("a block added afterwards (it sees the header only) equals the same block built together with the others", d <= tolk and x.n == y.n, f"largest relative difference of the keys {d:.2e} (allowed {tolk:.0e})"))
    # 5 the request reads all blocks
    x = blocks([A, B, C]); lx = x.forward(tail, range(x.pos[-1] + 1, x.pos[-1] + 1 + len(tail))); y = blocks([A2, B, C]); ly = y.forward(tail, range(y.pos[-1] + 1, y.pos[-1] + 1 + len(tail))); d = float(np.abs(lx - ly).max())
    res.append(("the request reads every block: other tokens in block A change the next-token logits", d > 4 * tol, f"largest logit difference {d:.2e} (needs more than {4 * tol:.0e})"))
    return res


# ---------------------------------------------------------------- summaries and the reading rules
def summarize_block(rows) -> dict:
    """`kv_large.summarize_large` plus: the share of steps whose largest KL is below B_KL (0.5 nats; the large page's 0.05 is for caches that are meant to be nearly exact) and, for the kept block arm, the
    largest difference between the kept cache and the rebuilt one over the steps where it was checked."""
    out = summarize_large(rows)
    for a, v in out.items():
        if not v.get("n"): continue
        steps = [s for r in rows for s in r["arms"][a]]; kls = [s["kl_max"] for s in steps if s.get("kl_max") is not None]
        v["kl_below_b"] = _mean(k < B_KL for k in kls) if kls else None
        rv = [s["rebuilt_dev_v"] for s in steps if s.get("rebuilt_dev_v") is not None]; rk = [s["rebuilt_dev_k"] for s in steps if s.get("rebuilt_dev_k") is not None]
        v["rebuilt_checks"] = len(rv); v["rebuilt_dev_v_max"] = max(rv) if rv else None; v["rebuilt_dev_k_max"] = max(rk) if rk else None
    return out


def verdict_block(a: dict, b: Optional[dict] = None) -> List[Tuple[str, Optional[bool]]]:
    """The fourteen reading rules, in the order of BLOCK_LINES. None = the run the line needs has not been made."""
    f, dc, bk, b16 = (a.get(n, {}) for n in ("fresh", "drop_compact", "block", "block16")); fb, bb = ((b or {}).get(n, {}) for n in ("fresh", "block")); ratio = lambda p, q: None if not p or not q else p / q
    below = lambda x, y, pts: None if x is None or y is None else x >= y - pts - 1e-12                  # 'at most pts below y': a tie at exactly pts below reads yes
    ok = [_ge(f.get("prompt_tokens_mean"), L_MIN_PROMPT), _ge(f.get("canonical"), 1.0),
          None if bk.get("rebuilt_dev_v_max") is None else bool(bk["rebuilt_dev_v_max"] <= B_REBUILT_V + 1e-12 and bk["rebuilt_dev_k_max"] <= B_REBUILT_K + 1e-12), _ge(f.get("exact_m6"), B_FRESH_M6),
          _ge(bk.get("same"), B_SAME), below(bk.get("exact_m6"), f.get("exact_m6"), B_M6_POINTS), _ge(bk.get("kl_below_b"), B_KL_SHARE), _ge(b16.get("same"), B_SAME),
          None if bk.get("v_dev") is None or dc.get("v_dev") is None else bk["v_dev"] > dc["v_dev"], _le(ratio(bk.get("prefilled"), f.get("prefilled")), B_PREFILL),
          _le(ratio(bk.get("seconds"), f.get("seconds")), B_SEC),
          below(bb.get("exact_m6"), fb.get("exact_m6"), B_B2_M6_POINTS) if b else None, below(bb.get("exact_blind"), fb.get("exact_blind"), B_B2_BLIND_POINTS) if b else None, _ge(bb.get("same"), B_B2_SAME) if b else None]
    return [(text, v) for (_, text), v in zip(BLOCK_LINES, ok)]


def run_block(engine, kit, table, episodes: int, steps: int, seed: int, hint_rules: bool, path: str, meta=None, progress=None, max_new: int = KV_MAX_NEW, script_fn: Optional[Callable] = None) -> dict:
    """`kv_drop.run_kv` with the four arms of this page and the large request script (or `script_fn`), and the summary of `summarize_block`. Resumable like `run_kv`."""
    res = run_kv(engine, kit, table, episodes, steps, seed, hint_rules, path, meta=meta, arms=BLOCK_ARMS, max_new=max_new, progress=progress, script_fn=script_fn or large_script)
    res["summary"] = summarize_block(res["rows"]); return res


def print_block(summary: dict, title: str = ""):
    """The tables of `kv_large.print_large` and, below them, what is particular to this test."""
    print_large(summary, title)
    bk = summary.get("block", {})
    if bk.get("n"):
        print(f"\nkept block cache against one rebuilt from nothing at the same positions ({bk.get('rebuilt_checks', 0)} checks): largest relative difference of the values {_fx(bk.get('rebuilt_dev_v_max'), 5)}, of the keys {_fx(bk.get('rebuilt_dev_k_max'), 5)}")
    print(f"{'arm':13s} {'KL<0.5':>8s} {'KL max':>8s} {'value dev. (mean)':>18s}")
    for arm, v in summary.items():
        if v.get("n") and arm != "fresh": print(f"{arm:13s} {_pc(v.get('kl_below_b')):>8s} {_fx(v.get('kl_max'), 3):>8s} {_fx(v.get('v_dev'), 4):>18s}")

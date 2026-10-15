"""KV-cache test: what happens to a stock model's answers when the interior tokens are DROPPED from its cache? Page with the predictions written BEFORE any run: docs/KV_DROP_TEST.md
(concept-map item 96; rows 3, 9 and item 89: block independence on a stock model was never measured).

One logical prompt per step, the same for every arm: [system and header] [the current state, one item after the other] [the request] -> the model answers with a delta. The state is the
ORACLE's (teacher forcing: the arms never feed their own mistakes back into the state), so the arms differ only in HOW THE KEY/VALUE CACHE FOR THAT PROMPT IS OBTAINED:
  fresh          the whole prompt is read again from nothing at every step. The reference: what the stock model says about this prompt.
  exact          the cache is kept between steps and made exact: the request and the reply are cropped; an added item is appended (exact: causal attention); after a delete everything
                 from the first deleted item on is read again. What a serving stack with prefix caching does.
  drop_compact   as exact, but a deleted item is CUT OUT of the cache and nothing after it is read again (the tokens behind the hole keep their old keys and values, computed while the
                 deleted item was visible: STALE). The keys behind the hole are turned back to the positions a fresh read would give them (RoPE rotation is exact), so the positions are right.
  drop_gap       as drop_compact, but the positions are left as they were (a hole in the position numbers). No rotation at all.
  keep           a chat: every step's state, request and raw reply stays in the cache; nothing is dropped. A different INPUT (it holds the whole past), exact for that input.
INTERFACE = the tokens of the live state (and the fixed header). INTERIOR = the request and reply tokens of a step (cropped by every cache arm: exact) and the tokens of an item the step
deleted (cut out by the two drop arms: this is the approximation). The cache is reached only through the small protocol below, so the logic is tested on a numpy model
(tests/test_kv_drop.py) and the PyTorch/Hugging Face side (`kv_torch.py`) is a thin shell that the notebook checks on the real model first (`preflight`).

Cache protocol (`engine.new_cache()` returns one):  .n tokens held;  .pos position ids of the rows;  .forward(tokens, positions) appends the rows and returns the raw logits of the LAST token
(numpy, float);  .forward_blocks(blocks, positions, visible=None) reads several blocks SIDE BY SIDE (docs/KV_BLOCK_TEST.md): each block sees the first `visible` rows of the cache (default all) and itself, no other block, then all are appended in order (returns nothing);
.select(rows) keeps those rows in that order;  .rerotate(new_positions) turns the keys to new position ids;  .nbytes();  .keys(layer) the keys of one layer (numpy, for the
checks only);  .value_deviation(other, rows) mean relative L2 difference of the values of the first `rows` rows against another cache (how far a patched cache is from a rebuilt one).  Engine: `new_cache()`, `sync()`, `reset_peak()`, `peak_mb()`, `tol` (allowed difference in logits), `tol_keys` (allowed difference in keys)."""
import json, os, time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check
from axi.experiments.agent_loop import RULES_HINT, SYSTEM, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script
from axi.experiments.resumable import _roundtrip, load_rows

KV_MODEL = "Qwen/Qwen2.5-7B-Instruct"
KV_SEED = 11                         # run A and run C: episode seeds 11000 to 11007 (the same request scripts, with and without the rule)
KV_LONG_SEED = 12                    # run B: episode seeds 12000 and 12001
KV_A_EPISODES, KV_A_STEPS = 8, 30
KV_C_EPISODES, KV_C_STEPS = 8, 30
KV_B_EPISODES, KV_B_STEPS = 2, 100
KV_K_CAP = 20_000                    # the keep arm stops before its cache would pass this many tokens (the model was trained on 32,768)
KV_MAX_NEW = 120
ARMS = ("fresh", "exact", "drop_compact", "drop_gap", "keep")
IM_START, IM_END, STATE_OPEN, STATE_CLOSE = "<|im_start|>", "<|im_end|>", "<CICO_STATE>", "</CICO_STATE>"

# thresholds of the reading rules (docs/KV_DROP_TEST.md)
K_CANON, K_EXACT_SAME, K_DC_SAME, K_DG_SAME, K_EXACT_POINTS = 1.0, 0.99, 0.97, 0.95, 0.01
K_KL, K_KL_SHARE, K_PREFILL_D, K_PREFILL_X, K_RESIDENT, K_SEC_D, K_GROWTH, K_KEEP_SAME, K_C_SAME, K_B_SAME = 0.05, 0.95, 0.25, 0.40, 10.0, 0.85, 1.5, 0.90, 0.95, 0.95
HEADLINE_LINES = (2, 3, 5, 6, 14)
KV_LINES = [
    ("instrument", "The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run A."),
    ("guard", "exact: the reply is identical to fresh's, token for token, on at least 99 % of the steps of run A."),
    ("claim", "drop_compact: the reply is identical to fresh's on at least 97 % of the steps of run A."),
    ("claim", "drop_gap: the reply is identical to fresh's on at least 95 % of the steps of run A."),
    ("claim", "drop_compact: the share of steps on which repair_m6 turns the reply into the right next state is within 1 point of fresh's (run A)."),
    ("claim", "drop_compact: on at least 95 % of the steps of run A the largest KL along the reply, against fresh's distribution, is below 0.05 nats."),
    ("claim", "drop_compact reads at most 25 % of the tokens that fresh reads (run A, whole episodes, the first build of each included)."),
    ("claim", "exact reads at most 40 % of the tokens that fresh reads (run A)."),
    ("claim", "keep: at the last step of run A its cache holds at least 10 times as many tokens as drop_compact's."),
    ("claim", "drop_compact: mean seconds per step at most 0.85 times fresh's (run A)."),
    ("claim", "drop_compact: mean seconds per step below exact's (run A)."),
    ("claim", "keep: mean seconds per step in the last third of the steps at most 1.5 times the first third (run A)."),
    ("claim", "keep: the reply is identical to fresh's on at least 90 % of the steps of run A."),
    ("claim", "run C (rule stated): drop_compact's reply is identical to fresh's on at least 95 % of the steps."),
    ("claim", "run B (episodes of 100 steps): drop_compact's reply is identical to fresh's on at least 95 % of steps 67 to 100."),
]


# ---------------------------------------------------------------- the logical state: items in the order they were added
@dataclass(frozen=True)
class Item:
    key: tuple          # ("n", (r, c)) or ("e", (u, v, rel))
    val: int

    @property
    def text(self) -> str:
        if self.key[0] == "n": r, c = self.key[1]; return f"[{r},{c}:{self.val}]"
        u, v, rel = self.key[1]; return f"({u[0]},{u[1]})->({v[0]},{v[1]}):{rel}#{self.val}"


def graph_items(g: Graph) -> List[Item]:
    return [Item(("n", n), v) for n, v in sorted(g.nodes.items())] + [Item(("e", k), w) for k, w in sorted(g.edges.items())]


def items_graph(items: Sequence[Item]) -> Graph:
    g = Graph()
    for it in items:
        if it.key[0] == "n": g.add_node(it.key[1], it.val)
    for it in items:
        if it.key[0] == "e": u, v, rel = it.key[1]; g.add_edge(u, v, rel, it.val)
    return g


def apply_delta_items(items: Sequence[Item], delta) -> Tuple[List[Item], List[int], List[Item]]:
    """(new item list, indices in the OLD list of the items the delta removed, the items it added in order). Removed items leave their place; added items go to the END, so an add never
    moves anything. The result describes the same graph as `gate.apply` (a re-added key counts as removed and added)."""
    kill = {("e", (d.u, d.v, d.relation)) for d in delta.edge_deltas if d.op == "DEL"} | {("n", (d.r, d.c)) for d in delta.node_deltas if d.op == "DEL"}
    added = [Item(("n", (d.r, d.c)), d.val) for d in delta.node_deltas if d.op == "ADD"] + [Item(("e", (d.u, d.v, d.relation)), d.weight) for d in delta.edge_deltas if d.op == "ADD"]
    kill |= {a.key for a in added}
    deleted = [i for i, it in enumerate(items) if it.key in kill]
    return [it for it in items if it.key not in kill] + added, deleted, added


# ---------------------------------------------------------------- the prompt, in pieces
def first_prefix_text(system: str) -> str: return f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{STATE_OPEN}"


LATER_PREFIX = f"{IM_START}user\n{STATE_OPEN}"


def tail_text(instruction: str) -> str:
    """The closing tag follows a SPACE, not a line break, on purpose: with Qwen's pre-tokenizer a closing bracket followed by a line break is ONE pre-token, so a line break right after the last item
    (which ends in a bracket when it is a service) would be merged with it into a single token by the tokenizer, and the pieces would no longer be the tokenizer's own tokens. A space starts a new pre-token."""
    return f" {STATE_CLOSE}\n\nRequest: {instruction}{IM_END}\n{IM_START}assistant\n"


class PromptKit:
    """Turns a logical state into tokens in three pieces (header, one piece per item, request), the same for every arm. `enc(text) -> ids` and `dec(ids) -> text` come from the tokenizer."""
    def __init__(self, enc: Callable[[str], List[int]], dec: Callable[[Sequence[int]], str], eos: int, hint_rules: bool = False):
        self.enc, self.dec, self.eos, self.hint_rules = enc, dec, eos, bool(hint_rules)
        self.system = SYSTEM + (RULES_HINT if hint_rules else "")
        self.first_prefix, self.later_prefix, self.nl = enc(first_prefix_text(self.system)), enc(LATER_PREFIX), enc("\n"); self._item, self._tail = {}, {}

    def item(self, it: Item) -> List[int]:
        if it not in self._item: self._item[it] = self.enc(" " + it.text)
        return self._item[it]

    def tail(self, instruction: str) -> List[int]:
        if instruction not in self._tail: self._tail[instruction] = self.enc(tail_text(instruction))
        return self._tail[instruction]

    def state_tokens(self, items: Sequence[Item], first: bool = True) -> List[int]:
        return (self.first_prefix if first else self.later_prefix) + [x for it in items for x in self.item(it)]

    def prompt_tokens(self, items: Sequence[Item], instruction: str) -> List[int]: return self.state_tokens(items) + self.tail(instruction)

    def whole_text(self, items: Sequence[Item], instruction: str) -> str:
        return first_prefix_text(self.system) + "".join(" " + it.text for it in items) + tail_text(instruction)

    def canonical(self, items: Sequence[Item], instruction: str) -> bool:
        """True if the piece-by-piece tokens are what the tokenizer gives for the whole text (a token never straddles two pieces)."""
        return self.enc(self.whole_text(items, instruction)) == self.prompt_tokens(items, instruction)


class Tracked:
    """A cache plus the list of token ids it holds, so that 'the cache holds exactly the logical prompt' can be asserted at every step."""
    def __init__(self, cache): self.c, self.ids = cache, []

    @property
    def n(self) -> int: return self.c.n

    @property
    def pos(self): return self.c.pos

    def forward(self, tokens, positions):
        self.ids += list(tokens); return self.c.forward(list(tokens), list(positions))

    def forward_blocks(self, blocks, positions, visible=None):
        blocks = [list(b) for b in blocks]; self.ids += [x for b in blocks for x in b]; self.c.forward_blocks(blocks, [list(p) for p in positions], visible)

    def keys(self, layer): return self.c.keys(layer)

    def select(self, rows):
        rows = list(rows); self.ids = [self.ids[i] for i in rows]; self.c.select(rows)

    def rerotate(self, new_pos): self.c.rerotate(list(new_pos))

    def nbytes(self) -> int: return self.c.nbytes()

    def value_deviation(self, other, rows: int) -> float: return self.c.value_deviation(other.c, rows)


def rope_tables(delta: Sequence[int], inv_freq: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """cos and sin of the angles delta[i] * inv_freq, shape [len(delta), 2 * len(inv_freq)], laid out like Hugging Face's rotary embedding (the angle list twice, the 'half-split' layout).
    Turning a key that already carries position p by `delta` gives the key of position p + delta: rotations add."""
    ang = np.asarray(delta, np.float64)[:, None] * np.asarray(inv_freq, np.float64)[None, :]; ang = np.concatenate([ang, ang], axis=-1)
    return np.cos(ang), np.sin(ang)


# ---------------------------------------------------------------- greedy decoding under the grammar, with the comparison to the reference as it goes
def masked_kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) in nats between the two softmaxes over the tokens the grammar allows (the same set while two replies agree)."""
    idx = np.flatnonzero(np.isfinite(p_logits)); a = p_logits[idx].astype(np.float64); b = q_logits[idx].astype(np.float64)
    if not np.isfinite(b).all(): return float("inf")
    a -= a.max(); b -= b.max(); la = a - np.log(np.exp(a).sum()); lb = b - np.log(np.exp(b).sum())
    return float((np.exp(la) * (la - lb)).sum())


@dataclass
class Reply:
    tokens: List[int]                 # the chosen tokens, the end-of-reply token included if the reply ended
    ended: bool
    masked: Optional[List[np.ndarray]] = None      # fresh keeps its masked logits for the other arms to be compared with
    kl: Optional[List[float]] = None               # KL against the reference at each position while the two agreed
    first_diff: Optional[int] = None               # first position at which this reply differs from the reference
    cache: object = None                           # fresh keeps the cache it read, so that the other arms' state rows can be compared with it


def decode(cache, logits, table, eos: int, start_pos: int, max_new: int, keep: bool = False, ref: Optional[Reply] = None) -> Reply:
    """Greedy decoding with the CICO grammar (`table`: TokenDFATable). `logits` are the raw logits after the prompt; every chosen token except the end token is fed to the cache."""
    V, state, toks, masked, kls, first_diff, aligned, lg, pos = table.V, table.start, [], [], [], None, ref is not None, logits, start_pos
    for j in range(max_new):
        m = np.full(lg.shape[0], -np.inf, np.float32); ok = table.allowed[state]; m[:V][ok] = np.asarray(lg[:V], np.float32)[ok]; tok = int(m.argmax())
        if aligned and j < len(ref.tokens):
            kls.append(masked_kl(ref.masked[j], m))
            if tok != ref.tokens[j]: aligned, first_diff = False, j
        if keep: masked.append(m)
        toks.append(tok)
        if tok == eos: return Reply(toks, True, masked or None, kls, first_diff)
        state = int(table.next[state, tok]); lg = cache.forward([tok], [pos]); pos += 1
    return Reply(toks, False, masked or None, kls, first_diff)


# ---------------------------------------------------------------- the arms
class _Arm:
    name = ""

    def __init__(self, engine, kit: PromptKit, table, max_new: int = KV_MAX_NEW):
        self.engine, self.kit, self.table, self.max_new = engine, kit, table, max_new

    def _now(self) -> float: self.engine.sync(); return time.perf_counter()

    def _rec(self, upd_s, pre_s, dec_s, prefilled, rep, resident, resident_state, cache, **kw) -> dict:
        return dict(update_s=upd_s, prefill_s=pre_s, decode_s=dec_s, seconds=upd_s + pre_s + dec_s, prefilled=prefilled, decoded=len(rep.tokens), resident=resident, resident_state=resident_state,
                    kv_bytes=cache.nbytes(), peak_mb=self.engine.peak_mb(), tokens=list(rep.tokens), ended=rep.ended, **kw)


class FreshArm(_Arm):
    name = "fresh"

    def step(self, t, items, change, instr, ref):
        toks = self.kit.prompt_tokens(items, instr); cache = Tracked(self.engine.new_cache()); self.engine.reset_peak()
        t0 = self._now(); logits = cache.forward(toks, range(len(toks))); t1 = self._now()
        rep = decode(cache, logits, self.table, self.kit.eos, len(toks), self.max_new, keep=True); t2 = self._now()
        rec = self._rec(0.0, t1 - t0, t2 - t1, len(toks), rep, cache.n, len(toks) - len(self.kit.tail(instr)), cache, canonical=self.kit.canonical(items, instr), prompt_tokens=len(toks))
        rep.cache = cache; return rec, rep


class _StateCacheArm(_Arm):
    """Arms that keep the state in a cache between steps (exact, drop_compact, drop_gap). The request and the reply are cropped after every step."""
    def __init__(self, *a, **kw): super().__init__(*a, **kw); self.cache = None; self.items: List[Item] = []

    def _first_build(self, items):
        self.cache = Tracked(self.engine.new_cache()); toks = self.kit.state_tokens(items); self.cache.forward(toks, range(len(toks))); return len(toks)

    def _update(self, items, change) -> int: raise NotImplementedError

    def _check_layout(self, items):
        want = self.kit.state_tokens(items)
        assert self.cache.ids == want, f"{self.name}: the cache does not hold the logical prompt ({len(self.cache.ids)} tokens, expected {len(want)})"

    def _extra(self) -> dict: return {}

    def step(self, t, items, change, instr, ref):
        self.engine.reset_peak(); t0 = self._now(); read = self._update(items, change) if self.cache is not None else self._first_build(items); self.items = list(items); t1 = self._now()
        self._check_layout(items); tail = self.kit.tail(instr); base = self.cache.n; p0 = self.cache.pos[-1] + 1
        v_dev = self.cache.value_deviation(ref.cache, base) if ref is not None and ref.cache is not None else None
        logits = self.cache.forward(tail, range(p0, p0 + len(tail))); t2 = self._now()
        rep = decode(self.cache, logits, self.table, self.kit.eos, p0 + len(tail), self.max_new, ref=ref); t3 = self._now()
        resident, nbytes_cache = self.cache.n, self.cache; extra = self._extra()
        rec = self._rec(0.0, t2 - t1, t3 - t2, read + len(tail), rep, resident, base, nbytes_cache, v_dev=v_dev, **extra)
        self.cache.select(range(base)); t4 = self._now(); rec["update_s"] = (t1 - t0) + (t4 - t3); rec["seconds"] = rec["update_s"] + rec["prefill_s"] + rec["decode_s"]
        return rec, rep


class ExactArm(_StateCacheArm):
    name = "exact"

    def _update(self, items, change) -> int:
        deleted, added = change; kit, old = self.kit, self.items; add_toks = [x for it in added for x in kit.item(it)]
        if deleted:
            i0, dset = min(deleted), set(deleted); start = len(kit.first_prefix) + sum(len(kit.item(it)) for it in old[:i0]); self.cache.select(range(start))
            toks = [x for j, it in enumerate(old) if j > i0 and j not in dset for x in kit.item(it)] + add_toks
        else: toks = add_toks
        if toks: self.cache.forward(toks, range(self.cache.n, self.cache.n + len(toks)))
        return len(toks)


class DropArm(_StateCacheArm):
    def __init__(self, *a, compact: bool, **kw):
        super().__init__(*a, **kw); self.compact, self.name, self.stale = compact, "drop_compact" if compact else "drop_gap", []

    def _first_build(self, items):
        n = super()._first_build(items); self.stale = [False] * n; return n

    def _update(self, items, change) -> int:
        deleted, added = change; kit, old = self.kit, self.items; dset = set(deleted); keep, off, first_hole = list(range(len(kit.first_prefix))), len(kit.first_prefix), None
        for j, it in enumerate(old):
            m = len(kit.item(it))
            if j in dset:
                if first_hole is None: first_hole = off
            else: keep.extend(range(off, off + m))
            off += m
        if deleted:
            self.stale = [self.stale[r] or r > first_hole for r in keep]; self.cache.select(keep)
            if self.compact: self.cache.rerotate(range(len(keep)))
        add_toks = [x for it in added for x in kit.item(it)]
        if add_toks:
            p0 = self.cache.pos[-1] + 1; self.cache.forward(add_toks, range(p0, p0 + len(add_toks))); self.stale += [False] * len(add_toks)
        return len(add_toks)

    def _extra(self) -> dict: return dict(stale=int(sum(self.stale)), gap=int(self.cache.pos[-1] + 1 - self.cache.n))

    def step(self, *a):
        rec, rep = super().step(*a); self.stale = self.stale[:self.cache.n]; return rec, rep


class KeepArm(_Arm):
    name = "keep"

    def __init__(self, *a, cap: int = KV_K_CAP, **kw): super().__init__(*a, **kw); self.cache, self.cap = None, cap

    def step(self, t, items, change, instr, ref):
        kit = self.kit; toks = kit.state_tokens(items, first=self.cache is None) + kit.tail(instr); close = [kit.eos] + kit.nl
        if self.cache is not None and self.cache.n + len(toks) + self.max_new + len(close) > self.cap: return None, None
        if self.cache is None: self.cache = Tracked(self.engine.new_cache())
        self.engine.reset_peak(); p0 = self.cache.pos[-1] + 1 if self.cache.n else 0
        t0 = self._now(); logits = self.cache.forward(toks, range(p0, p0 + len(toks))); t1 = self._now()
        rep = decode(self.cache, logits, self.table, kit.eos, p0 + len(toks), self.max_new, ref=ref); t2 = self._now()
        p1 = self.cache.pos[-1] + 1; self.cache.forward(close, range(p1, p1 + len(close))); t3 = self._now()
        rec = self._rec(t3 - t2, t1 - t0, t2 - t1, len(toks) + len(close), rep, self.cache.n, self.cache.n, self.cache)
        return rec, rep


class BlockArm(_StateCacheArm):
    """INDEPENDENT BLOCKS (docs/KV_BLOCK_TEST.md; the 'block independence at test time, no training' claim): the header is read first; then every block of `group` consecutive items is read on its own, seeing the
    header and itself and NO other block; the request that follows sees everything. A block sits at the positions it would have in a normal read, so what differs from `fresh` is that a block's rows were computed
    without the other blocks in view.
    maintained (group 1): the cache is kept between steps like `drop_gap`: a deleted item's rows are cut, NOTHING is turned (a row's deeper layers were computed at its position, with its distance to the header; turning
    only the keys would not be a rebuild at the new position), an added item is read as a new block after the last row (a hole in the position numbers, as in `drop_gap`). No block depended on the cut one, so the
    cache equals one rebuilt from nothing AT THE SAME POSITIONS: `CHECK_EVERY` steps this is measured (`rebuilt_dev_v`, `rebuilt_dev_k`).
    Otherwise (a coarser group): rebuilt from nothing every step with contiguous positions; this arm is for the accuracy question only, no speed is claimed for it."""
    CHECK_EVERY = 10

    def __init__(self, *a, group: int = 1, maintained: bool = False, **kw):
        super().__init__(*a, **kw); assert group >= 1 and (group == 1 or not maintained); self.group, self.maintained = group, maintained
        self.name = "block" if maintained else f"block{group}"

    def _blocks(self, items, positions=None):
        """Token lists and position lists of the blocks. `positions`: one position per item token, in order (default: contiguous after the header)."""
        kit, off, toks, pos, k = self.kit, len(self.kit.first_prefix), [], [], 0
        for i in range(0, len(items), self.group):
            b = [x for it in items[i:i + self.group] for x in kit.item(it)]; toks.append(b)
            if positions is None: pos.append(list(range(off, off + len(b)))); off += len(b)
            else: pos.append([int(x) for x in positions[k:k + len(b)]]); k += len(b)
        return toks, pos

    def _build(self, items, positions=None) -> Tracked:
        c = Tracked(self.engine.new_cache()); h = len(self.kit.first_prefix); c.forward(self.kit.first_prefix, range(h)); toks, pos = self._blocks(items, positions)
        if toks: c.forward_blocks(toks, pos)
        return c

    def _first_build(self, items):
        self.cache = self._build(items); return self.cache.n

    def _update(self, items, change) -> int:
        if not self.maintained: self.cache = self._build(items); return self.cache.n
        deleted, added = change; kit, old = self.kit, self.items; dset = set(deleted); keep, off = list(range(len(kit.first_prefix))), len(kit.first_prefix)
        for j, it in enumerate(old):
            m = len(kit.item(it))
            if j not in dset: keep.extend(range(off, off + m))
            off += m
        if deleted: self.cache.select(keep)
        add = [kit.item(it) for it in added]
        if add:
            off, pos = self.cache.pos[-1] + 1, []
            for b in add: pos.append(list(range(off, off + len(b)))); off += len(b)
            self.cache.forward_blocks(add, pos, visible=len(kit.first_prefix))                      # a new block sees the header only, not the blocks that are already there
        return sum(len(b) for b in add)

    def _extra(self) -> dict: return dict(gap=int(self.cache.pos[-1] + 1 - self.cache.n)) if self.maintained else {}

    def step(self, t, items, change, instr, ref):
        rec, rep = super().step(t, items, change, instr, ref)
        if self.maintained and self.CHECK_EVERY and t % self.CHECK_EVERY == 0:
            h = len(self.kit.first_prefix); chk = self._build(items, self.cache.pos[h:]); base = chk.n; L = self.engine.layers
            rel = lambda a, b: float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))
            rec["rebuilt_dev_v"] = self.cache.value_deviation(chk, base); rec["rebuilt_dev_k"] = max(rel(self.cache.keys(i)[:, :base], chk.keys(i)) for i in (0, L - 1)); del chk
        return rec, rep


def make_arms(engine, kit, table, arms: Sequence[str], max_new: int = KV_MAX_NEW, k_cap: int = KV_K_CAP) -> Dict[str, _Arm]:
    made = {"fresh": lambda: FreshArm(engine, kit, table, max_new), "exact": lambda: ExactArm(engine, kit, table, max_new), "drop_compact": lambda: DropArm(engine, kit, table, max_new, compact=True),
            "drop_gap": lambda: DropArm(engine, kit, table, max_new, compact=False), "keep": lambda: KeepArm(engine, kit, table, max_new, cap=k_cap),
            "block": lambda: BlockArm(engine, kit, table, max_new, group=1, maintained=True), "block16": lambda: BlockArm(engine, kit, table, max_new, group=16)}
    assert list(arms)[0] == "fresh" and all(a in made for a in arms), arms
    return {a: made[a]() for a in arms}


# ---------------------------------------------------------------- scoring
def score_reply(kit: PromptKit, tokens: Sequence[int], live: Graph, expected: Graph, instruction: str) -> dict:
    text = kit.dec([x for x in tokens if x != kit.eos]); d = strict_parse(text)
    blind = next_state("blind", live, d, instruction); m6 = next_state("repair_m6", live, d, instruction)
    return dict(text=text, parsed=d is not None, admitted=bool(d is not None and check(live, d).ok), exact_blind=same_graph(blind, expected), exact_m6=same_graph(m6, expected))


def annotate(rec: dict, rep: Reply, ref: Optional[Reply], kit: PromptKit, live: Graph, expected: Graph, instruction: str, t: int, kind: str) -> dict:
    """Add to a step record what only the whole step knows: the comparison with fresh's reply (ref=None for fresh itself) and the score of the reply on the oracle's state."""
    rec.update(t=t + 1, kind=kind, same_as_fresh=None if ref is None else rep.tokens == ref.tokens, first_diff=rep.first_diff, kl_max=max(rep.kl) if rep.kl else None, kl_mean=float(np.mean(rep.kl)) if rep.kl else None)
    rec.update(score_reply(kit, rep.tokens, live, expected, instruction)); return rec


# ---------------------------------------------------------------- one episode, and the resumable driver
def run_kv_episode(engine, kit: PromptKit, table, seed: int, steps: int, arms: Sequence[str] = ARMS, k_cap: int = KV_K_CAP, max_new: int = KV_MAX_NEW, on_step: Optional[Callable] = None,
                   script_fn: Optional[Callable] = None) -> dict:
    """`script_fn(seed, steps) -> (start graph, requests, oracle graphs)`: the request script; the default is the small one of docs/KV_DROP_TEST.md (`long_horizon.oracle_script`)."""
    g0, script, oracle = (script_fn or oracle_script)(seed, steps); items = graph_items(g0); arm = make_arms(engine, kit, table, arms, max_new, k_cap)
    out, stopped, change = {a: [] for a in arms}, {a: None for a in arms}, None
    for t, st in enumerate(script):
        assert same_graph(items_graph(items), oracle[t]), f"the item list left the oracle's state at step {t + 1}"
        if on_step: on_step(t, st)
        instr, ref = st["instruction"], None
        for a in arms:
            if stopped[a]: continue
            rec, rep = arm[a].step(t, items, change, instr, ref)
            if rec is None: stopped[a] = "cap"; continue
            if a == "fresh": ref = rep
            out[a].append(annotate(rec, rep, None if a == "fresh" else ref, kit, oracle[t], oracle[t + 1], instr, t, st["kind"]))
        items, deleted, added = apply_delta_items(items, CICOParser.parse_transition_delta(st["ideal"])); change = (deleted, added)
    return dict(seed=seed, steps=len(script), arms=out, stopped=stopped)


def run_kv(engine, kit: PromptKit, table, episodes: int, steps: int, seed: int, hint_rules: bool, path: str, meta=None, arms: Sequence[str] = ARMS, k_cap: int = KV_K_CAP,
           max_new: int = KV_MAX_NEW, progress=None, script_fn: Optional[Callable] = None) -> dict:
    """One JSON line per finished episode (all arms); run again with the same arguments to carry on after a dropped session. A `script_fn` other than the default is written into the meta
    (its `__name__`), so a file made with one script is refused by a run with another."""
    assert kit.hint_rules == bool(hint_rules), "the prompt kit and the run disagree about the rule"
    meta = dict(meta or {}, kind="kv_drop", episodes=episodes, steps=steps, seed=seed, hint_rules=bool(hint_rules), arms=list(arms), k_cap=k_cap, max_new=max_new)
    if script_fn is not None: meta["script"] = script_fn.__name__
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for e in range(len(rows), episodes):
            row = _roundtrip(run_kv_episode(engine, kit, table, seed * 1000 + e, steps, arms, k_cap, max_new, script_fn=script_fn)); f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(e + 1, episodes)
    return {"rows": rows, "summary": summarize_kv(rows), "meta": meta}


# ---------------------------------------------------------------- the checks the notebook runs on the real model before anything else
def preflight(engine, kit: PromptKit, seed: int = KV_SEED, script_fn: Optional[Callable] = None) -> List[Tuple[str, bool, str]]:
    """Checks of the cache operations on one real prompt (the first prompt of the script `script_fn`; default the small one). Each returns (name, ok, detail). A cache whose rotation or selection
    is wrong would make 'drop' look better or worse than it is."""
    g0, script, _ = (script_fn or oracle_script)(seed * 1000, 3); items = graph_items(g0); T = kit.prompt_tokens(items, script[0]["instruction"]); n, tol, res = len(T), engine.tol, []
    full = lambda: Tracked(engine.new_cache())
    c = full(); l_one = c.forward(T, range(n))
    a = n // 2; c = full(); c.forward(T[:a], range(a)); l = c.forward(T[a:], range(a, n)); d = float(np.abs(l - l_one).max())
    res.append(("a prompt read in two pieces gives the same logits as read at once", d <= tol, f"max difference {d:.2e} (allowed {tol:.0e})"))
    shift = 211; probe = T[-1]; c = full(); c.forward(T, range(shift, shift + n)); l_ref = c.forward([probe], [shift + n]); k_ref = [c.c.keys(i)[:, :n] for i in range(engine.layers)]
    c = full(); c.forward(T, range(n)); k_old = [c.c.keys(i) for i in range(engine.layers)]; c.rerotate(range(shift, shift + n))
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))                       # size of the difference relative to the size of the keys, per layer
    dk = max(rel(c.c.keys(i), k_ref[i]) for i in range(engine.layers)); dk0 = max(rel(k_old[i], k_ref[i]) for i in range(engine.layers))
    res.append(("keys turned by 211 positions equal the keys of a prompt read at those positions, in every layer", dk <= engine.tol_keys, f"largest relative difference {dk:.2e} (allowed {engine.tol_keys:.0e})"))
    res.append(("...and that check can tell: the same keys WITHOUT the turn are clearly different", dk0 > 3 * engine.tol_keys, f"largest relative difference {dk0:.2e} (needs more than {3 * engine.tol_keys:.0e})"))
    l_rot = c.forward([probe], [shift + n]); d = float(np.abs(l_rot - l_ref).max())
    res.append(("...and the next token's logits agree", d <= tol, f"max difference {d:.2e} (allowed {tol:.0e})"))
    m = n - 17; c = full(); c.forward(T[:m], range(m)); c.forward(T[m:], range(m, n)); c.select(range(m)); l = c.forward(T[m:], range(m, n)); d = float(np.abs(l - l_one).max())
    res.append(("cutting the last rows and reading them again gives the same logits", d <= tol, f"max difference {d:.2e} (allowed {tol:.0e})"))
    c = full(); c.forward(T, range(n)); c.select(range(n)); l = c.forward([probe], [n]); c2 = full(); c2.forward(T, range(n)); l2 = c2.forward([probe], [n]); d = float(np.abs(l - l2).max())
    res.append(("selecting every row changes nothing", d <= tol, f"max difference {d:.2e} (allowed {tol:.0e})"))
    alt = T[:a][::-1]; assert alt != T[:a]; c1 = full(); c1.forward(T[:a], range(a)); l1 = c1.forward(T[a:], range(a + 50, n + 50)); c2 = full(); c2.forward(alt, range(a)); l2 = c2.forward(T[a:], range(a + 50, n + 50)); d = float(np.abs(l1 - l2).max())
    res.append(("a hole in the position numbers does not hide the earlier rows (the logits change when the earlier rows are different)", d > 4 * tol, f"largest logit difference {d:.2e} (needs more than {4 * tol:.0e})"))
    c = full(); c.forward(T, range(n)); nb0 = c.nbytes(); c.select(range(n // 2)); res.append(("the cache reports its size and the size follows the rows", c.n == n // 2 and c.nbytes() < nb0 and len(c.pos) == c.n, f"{nb0} bytes for {n} rows, {c.nbytes()} for {c.n}"))
    res.append(("the three pieces of the prompt are the tokenizer's own tokens", kit.canonical(items, script[0]["instruction"]), "header + items + request against the whole text"))
    return res


# ---------------------------------------------------------------- summaries
def _mean(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None


def summarize_kv(rows) -> dict:
    out = {}
    for a in (rows[0]["arms"] if rows else {}):
        eps = [r["arms"][a] for r in rows]; steps = [s for e in eps for s in e]
        if not steps: out[a] = dict(n=0); continue
        first, last = [s for e in eps for s in e if s["t"] <= len(e) // 3], [s for e in eps for s in e if s["t"] > len(e) - len(e) // 3]
        late = [s for e in eps for s in e if s["t"] > 2 * len(e) // 3]; kls = [s["kl_max"] for s in steps if s["kl_max"] is not None]
        out[a] = dict(n=len(steps), episodes=len(eps), stopped=[r["stopped"][a] for r in rows],
                      same=_mean(s["same_as_fresh"] for s in steps if s["same_as_fresh"] is not None), same_late=_mean(s["same_as_fresh"] for s in late if s["same_as_fresh"] is not None),
                      kl_below=(_mean(k < K_KL for k in kls) if kls else None), kl_max=max(kls) if kls else None, kl_mean=_mean(s["kl_mean"] for s in steps),
                      exact_m6=_mean(s["exact_m6"] for s in steps), exact_blind=_mean(s["exact_blind"] for s in steps), admitted=_mean(s["admitted"] for s in steps), parsed=_mean(s["parsed"] for s in steps),
                      prefilled=sum(s["prefilled"] for s in steps), prefilled_mean=_mean(s["prefilled"] for s in steps), decoded_mean=_mean(s["decoded"] for s in steps),
                      resident_mean=_mean(s["resident"] for s in steps), resident_last=_mean(e[-1]["resident"] for e in eps if e), kv_mb_last=_mean(e[-1]["kv_bytes"] / 1e6 for e in eps if e),
                      peak_mb=max(s["peak_mb"] for s in steps), seconds=_mean(s["seconds"] for s in steps), update_s=_mean(s["update_s"] for s in steps), prefill_s=_mean(s["prefill_s"] for s in steps),
                      decode_s=_mean(s["decode_s"] for s in steps), seconds_first=_mean(s["seconds"] for s in first), seconds_last=_mean(s["seconds"] for s in last),
                      canonical=_mean(s.get("canonical") for s in steps) if a == "fresh" else None, v_dev=_mean(s.get("v_dev") for s in steps), v_dev_last=_mean(e[-1].get("v_dev") for e in eps if e), stale_last=_mean(e[-1].get("stale") for e in eps if e), gap_last=_mean(e[-1].get("gap") for e in eps if e),
                      by_kind={k: dict(n=len(v), same=_mean(s["same_as_fresh"] for s in v if s["same_as_fresh"] is not None), exact_m6=_mean(s["exact_m6"] for s in v), admitted=_mean(s["admitted"] for s in v))
                               for k in sorted({s["kind"] for s in steps}) for v in [[s for s in steps if s["kind"] == k]]})
    return out


def _ge(x, thr): return None if x is None else x >= thr - 1e-12


def _le(x, thr): return None if x is None else x <= thr + 1e-12


def verdict_kv(a: dict, c: Optional[dict] = None, b: Optional[dict] = None) -> List[Tuple[str, Optional[bool]]]:
    """The fifteen reading rules, in the order of KV_LINES. None = the run the line needs has not been made."""
    f, x, dc, dg, k = (a.get(n, {}) for n in ("fresh", "exact", "drop_compact", "drop_gap", "keep")); ratio = lambda p, q: None if not p or not q else p / q
    ok = [_ge(f.get("canonical"), K_CANON), _ge(x.get("same"), K_EXACT_SAME), _ge(dc.get("same"), K_DC_SAME), _ge(dg.get("same"), K_DG_SAME),
          None if dc.get("exact_m6") is None or f.get("exact_m6") is None else abs(dc["exact_m6"] - f["exact_m6"]) <= K_EXACT_POINTS + 1e-12, _ge(dc.get("kl_below"), K_KL_SHARE),
          _le(ratio(dc.get("prefilled"), f.get("prefilled")), K_PREFILL_D), _le(ratio(x.get("prefilled"), f.get("prefilled")), K_PREFILL_X), _ge(ratio(k.get("resident_last"), dc.get("resident_last")), K_RESIDENT),
          _le(ratio(dc.get("seconds"), f.get("seconds")), K_SEC_D), None if dc.get("seconds") is None or x.get("seconds") is None else dc["seconds"] < x["seconds"],
          _le(ratio(k.get("seconds_last"), k.get("seconds_first")), K_GROWTH), _ge(k.get("same"), K_KEEP_SAME),
          _ge((c or {}).get("drop_compact", {}).get("same"), K_C_SAME) if c else None, _ge((b or {}).get("drop_compact", {}).get("same_late"), K_B_SAME) if b else None]
    return [(text, v) for (_, text), v in zip(KV_LINES, ok)]


def _pc(x, k=1): return "-" if x is None else f"{x * 100:.{k}f}%"


def _fx(x, k=2): return "-" if x is None else f"{x:.{k}f}"


def print_kv(summary: dict, title: str = ""):
    print(f"\n== KV test {title} ==")
    print(f"{'arm':13s} {'steps':>5s} {'reply = fresh':>13s} {'KL<0.05':>8s} {'KL max':>8s} {'m6 exact':>9s} {'blind exact':>11s} {'raw legal':>10s} | {'read/step':>9s} {'cache/step':>10s} {'cache last':>10s} {'MB last':>8s}")
    for arm, v in summary.items():
        if not v.get("n"): print(f"{arm:13s} {0:5d}"); continue
        print(f"{arm:13s} {v['n']:5d} {_pc(v['same']):>13s} {_pc(v['kl_below']):>8s} {_fx(v['kl_max'], 3):>8s} {_pc(v['exact_m6']):>9s} {_pc(v['exact_blind']):>11s} {_pc(v['admitted']):>10s} | "
              f"{v['prefilled_mean']:9.1f} {v['resident_mean']:10.0f} {v['resident_last']:10.0f} {_fx(v['kv_mb_last'], 1):>8s}")
    print(f"\n{'arm':13s} {'seconds':>8s} {'update':>7s} {'prefill':>8s} {'decode':>7s} | {'first third':>11s} {'last third':>10s} {'ratio':>6s} | {'peak MB':>8s} {'stale rows':>10s} {'position gap':>12s} {'value dev.':>10s}")
    for arm, v in summary.items():
        if not v.get("n"): continue
        r = None if not v["seconds_first"] else v["seconds_last"] / v["seconds_first"]
        print(f"{arm:13s} {_fx(v['seconds']):>8s} {_fx(v['update_s']):>7s} {_fx(v['prefill_s']):>8s} {_fx(v['decode_s']):>7s} | {_fx(v['seconds_first']):>11s} {_fx(v['seconds_last']):>10s} {_fx(r):>6s} | {v['peak_mb']:8.0f} {_fx(v['stale_last'], 0):>10s} {_fx(v['gap_last'], 0):>12s} {_fx(v['v_dev_last'], 4):>10s}")
    kinds = sorted({k for v in summary.values() if v.get("n") for k in v["by_kind"]})
    print(f"\nreply = fresh, by kind:  " + "   ".join(f"{k}: " + " ".join(f"{arm[:6]} {_pc(v['by_kind'][k]['same'], 0)}" for arm, v in summary.items() if v.get("n") and arm != "fresh" and k in v["by_kind"]) for k in kinds))
    print("stopped early (cap):", {arm: v["stopped"] for arm, v in summary.items() if v.get("n") and any(v["stopped"])})


def print_verdict_kv(checks, title: str):
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, (text, ok) in enumerate(checks, 1): print(("  yes      " if ok else "  not run  " if ok is None else "  NO       ") + f"{i:2d}. {text}")

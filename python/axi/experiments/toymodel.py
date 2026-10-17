"""Engineering for docs/TOY_MODEL_TEST.md (concept-map item 103): a small transformer trained from scratch on graphs of 10 services and asked about graphs of up to 100, with four ways of telling it where a service is.

CPU only, numpy only (the forward pass, the backward pass and the optimiser are written here and checked against finite differences in tests/test_toymodel.py), no language model, no internet.
The task ("which parts of the graph go when the service at cell c is deleted"): a graph is a list of tokens, a query token Q first (it holds the cell c), then the services S (a cell and a value) and the edges E (the cell
of the source, the offset to the target, a weight) in a random order. For every S and E token the model says whether it is hit: a service is hit when it is at c, an edge when its source or its target is at c.
The cell of a token is its coordinate (x, y); the offset and the weight are its payload. Graphs of N services live on an N x N grid, so N = 10 is trained on and N = 100 is asked about: ten times the range of the
coordinates and ten times the number of tokens.
How the coordinate gets into the attention (the only difference between the arms; `pos` is the first half of every head, the other half, `sem`, is not turned):
  rope1d   the roadmap's "Arm A": the first half of each head is turned by the token's index in the sequence (standard rotary embedding); the coordinates enter only as learned tables added to the token
  rope2d   the first half of each head is turned by the coordinate: 4 frequencies for x, 4 for y (axial rotary embedding); no tables
  su2      the first half of each head is split into 4 blocks of C^2 and each block is turned by the SU(2) element exp(theta_b (x T1 + y T2)), T_a = -i sigma_a / 2 (my reading of "spinor embeddings"); no tables
  and an optional penalty (`lam`): the squared size of the blocks of W_Q and W_K that carry the payload part of the input into `pos`, and the structural part of the input into `sem` (my reading of
  ||P_coord W_Q P_payload||_F^2 -> 0).
Nothing here is the exact recipe of the roadmap, which gives none; see docs/TOY_MODEL_TEST.md."""
import copy, hashlib, json, math, os, random, sys, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- the task (fixed in docs/TOY_MODEL_TEST.md before the run)
OFFS = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]       # the 24 offsets of an edge
OFF_ID = {o: i for i, o in enumerate(OFFS)}
OFF_NONE = 24
K_Q, K_S, K_E = 0, 1, 2
PAY_TRAIN, PAY_ALL = 10, 20                                                                  # payload ids 0..9 are met in training (0 = none, 1..9 values and weights); 10..19 only in the payload-shift sets
GMAX = 100
EXTRA_EDGE_P, QUERY_EXISTS_P = 0.15, 0.85
TRAIN_N = 10
EVAL_SETS = (("N10", 10, 10, False), ("N20", 20, 20, False), ("N40", 40, 40, False), ("N70", 70, 70, False), ("N100", 100, 100, False), ("range100_n10", 10, 100, False),
             ("pay_shift_N10", 10, 10, True), ("pay_shift_N100", 100, 100, True))           # (name, services, grid side, payload ids never seen in training)
EVAL_GRAPHS = {"N10": 400, "N20": 400, "N40": 300, "N70": 200, "N100": 200, "range100_n10": 400, "pay_shift_N10": 400, "pay_shift_N100": 200}

# ---------------------------------------------------------------- the model (sizes fixed on the page)
D_MODEL, D_STRUCT, N_HEAD, D_HEAD, D_POS, D_FF, N_LAYER = 64, 32, 4, 32, 16, 128, 2
THETA = (1.0, 0.4, 0.16, 0.064)                                                              # the four frequencies, for the coordinates (rope2d, su2)
ARMS = {"A_rope1d": dict(code="rope1d", table=True, lam=0.0), "A_rope2d": dict(code="rope2d", table=False, lam=0.0), "B_su2": dict(code="su2", table=False, lam=0.05),
        "B_su2_free": dict(code="su2", table=False, lam=0.0), "A_rope2d_orth": dict(code="rope2d", table=False, lam=0.05)}
TRAIN = dict(steps=4000, batch=32, lr=2e-3, lr_end=0.1, clip=1.0, pos_weight=3.0, init_seed=11, data_seed=23, eval_graphs=300)
SEEDS = (1, 2, 3)


# ---------------------------------------------------------------- graphs
def make_graph(rng: random.Random, n: int, g: int, pay_shift: bool = False) -> dict:
    """n services grown as a blob on a g x g grid, edges between services at most 2 apart (a tree plus extra edges), a query cell; returns the tokens in a random order after the query."""
    start = (rng.randrange(g), rng.randrange(g)); order = [start]; cells = {start}
    while len(order) < n:
        c = None
        for _ in range(60):
            px, py = order[rng.randrange(len(order))]; dx, dy = OFFS[rng.randrange(24)]; cand = (px + dx, py + dy)
            if 0 <= cand[0] < g and 0 <= cand[1] < g and cand not in cells: c = cand; break
        if c is None:
            free = [(x + dx, y + dy) for (x, y) in order for dx, dy in OFFS if 0 <= x + dx < g and 0 <= y + dy < g and (x + dx, y + dy) not in cells]
            if not free: break
            c = rng.choice(free)
        order.append(c); cells.add(c)
    pidx = {}
    for i, c in enumerate(order):
        pidx[c] = i
    edges = set()
    for i in range(1, len(order)):                                                          # the tree: a service is joined to some earlier service that is at most 2 away
        near = [j for j in range(i) if max(abs(order[j][0] - order[i][0]), abs(order[j][1] - order[i][1])) <= 2]
        j = near[rng.randrange(len(near))] if near else 0
        if max(abs(order[j][0] - order[i][0]), abs(order[j][1] - order[i][1])) > 2: continue
        edges.add((j, i) if rng.random() < 0.5 else (i, j))
    for i in range(len(order)):
        for dx, dy in OFFS:
            c = (order[i][0] + dx, order[i][1] + dy); j = pidx.get(c)
            if j is not None and j > i and rng.random() < EXTRA_EDGE_P and (i, j) not in edges and (j, i) not in edges:
                edges.add((i, j) if rng.random() < 0.5 else (j, i))
    lo, hi = (PAY_TRAIN, PAY_ALL - 1) if pay_shift else (1, 9)
    if rng.random() < QUERY_EXISTS_P: q = order[rng.randrange(len(order))]
    else:
        free = [(x + dx, y + dy) for (x, y) in order for dx, dy in OFFS if 0 <= x + dx < g and 0 <= y + dy < g and (x + dx, y + dy) not in cells]
        q = rng.choice(free) if free else order[rng.randrange(len(order))]
    toks = [("S", c, 0, rng.randint(lo, hi), int(c == q)) for c in order]
    for (i, j) in sorted(edges):
        a, b = order[i], order[j]; toks.append(("E", a, OFF_ID[(b[0] - a[0], b[1] - a[1])], rng.randint(lo, hi), int(a == q or b == q)))
    rng.shuffle(toks)
    seq = [("Q", q, OFF_NONE, 0, 0)] + [(k, c, (OFF_NONE if k == "S" else o), p, y) for (k, c, o, p, y) in toks]
    return dict(tokens=seq, query=q, n=len(order), edges=len(edges))


def batch_of(graphs: Sequence[dict]) -> dict:
    """Pad to the longest; kind, off, pay, x, y, label, scored (a service or an edge, not the query, not padding), length."""
    B = len(graphs); T = max(len(g["tokens"]) for g in graphs)
    out = dict(kind=np.zeros((B, T), np.int64), off=np.full((B, T), OFF_NONE, np.int64), pay=np.zeros((B, T), np.int64), x=np.zeros((B, T), np.int64), y=np.zeros((B, T), np.int64),
               label=np.zeros((B, T), np.float64), scored=np.zeros((B, T), bool), length=np.array([len(g["tokens"]) for g in graphs]))
    for b, g in enumerate(graphs):
        for t, (k, c, o, p, y) in enumerate(g["tokens"]):
            out["kind"][b, t] = {"Q": K_Q, "S": K_S, "E": K_E}[k]; out["off"][b, t] = o; out["pay"][b, t] = p; out["x"][b, t], out["y"][b, t] = c; out["label"][b, t] = y; out["scored"][b, t] = k != "Q"
    return out


# ---------------------------------------------------------------- position codes
def pos_params(code: str, batch: dict, dtype=np.float32) -> dict:
    """What the turning needs, per token."""
    B, T = batch["x"].shape
    if code == "rope1d":
        ang = np.arange(T, dtype=np.float64)[None, :, None] * (10000.0 ** (-np.arange(8) / 8.0))[None, None, :]; ang = np.broadcast_to(ang, (B, T, 8))
    elif code == "rope2d":
        th = np.array(THETA); ang = np.concatenate([batch["x"][..., None] * th, batch["y"][..., None] * th], axis=-1).astype(np.float64)
    elif code == "su2":
        th = np.array(THETA); x, y = batch["x"].astype(np.float64), batch["y"].astype(np.float64); r = np.sqrt(x * x + y * y)[..., None]; a = th[None, None, :] * r / 2.0
        k = (th / 2.0)[None, None, :] * np.sinc(a / np.pi); u = k * (x[..., None] + 1j * y[..., None])                      # u = sin(a) (cos phi + i sin phi) with phi the direction of (x, y); smooth at the origin
        return dict(ch=np.cos(a).astype(dtype), u=u.astype(np.complex64 if dtype == np.float32 else np.complex128))
    else: raise KeyError(code)
    return dict(cos=np.cos(ang).astype(dtype), sin=np.sin(ang).astype(dtype))


def rotate(x: np.ndarray, pp: dict, code: str, inverse: bool = False) -> np.ndarray:
    """Turn the pos block x of shape (B, H, T, D_POS); `inverse` applies the transpose (which is the inverse, the turn being orthogonal)."""
    if code in ("rope1d", "rope2d"):
        c, s = pp["cos"][:, None], pp["sin"][:, None]
        if inverse: s = -s
        x1, x2 = x[..., 0::2], x[..., 1::2]; out = np.empty_like(x); out[..., 0::2] = x1 * c - x2 * s; out[..., 1::2] = x1 * s + x2 * c; return out
    B, H, T, _ = x.shape; xb = x.reshape(B, H, T, 4, 4); z1 = xb[..., 0] + 1j * xb[..., 1]; z2 = xb[..., 2] + 1j * xb[..., 3]; ch, u = pp["ch"][:, None], pp["u"][:, None]
    if inverse: w1 = ch * z1 + 1j * np.conj(u) * z2; w2 = 1j * u * z1 + ch * z2
    else: w1 = ch * z1 - 1j * np.conj(u) * z2; w2 = -1j * u * z1 + ch * z2
    out = np.stack([w1.real, w1.imag, w2.real, w2.imag], axis=-1).astype(x.dtype); return out.reshape(B, H, T, 16)


# ---------------------------------------------------------------- parameters
def init_params(table: bool, seed: int, dtype=np.float32) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed); n = lambda *s, sd: (rng.standard_normal(s) * sd).astype(dtype)
    HD = N_HEAD * D_HEAD; P = dict(emb_kind=n(3, D_STRUCT, sd=0.5), emb_off=n(OFF_NONE + 1, D_STRUCT, sd=0.5), emb_pay=n(PAY_ALL, D_MODEL - D_STRUCT, sd=0.5))
    if table: P["cx"] = n(GMAX, D_STRUCT, sd=0.5); P["cy"] = n(GMAX, D_STRUCT, sd=0.5)
    for l in range(N_LAYER):
        P.update({f"g1_{l}": np.ones(D_MODEL, dtype), f"b1_{l}": np.zeros(D_MODEL, dtype), f"Wq_{l}": n(D_MODEL, HD, sd=D_MODEL ** -0.5), f"Wk_{l}": n(D_MODEL, HD, sd=D_MODEL ** -0.5), f"Wv_{l}": n(D_MODEL, HD, sd=D_MODEL ** -0.5),
                  f"Wo_{l}": n(HD, D_MODEL, sd=HD ** -0.5), f"g2_{l}": np.ones(D_MODEL, dtype), f"b2_{l}": np.zeros(D_MODEL, dtype), f"W1_{l}": n(D_MODEL, D_FF, sd=D_MODEL ** -0.5), f"c1_{l}": np.zeros(D_FF, dtype),
                  f"W2_{l}": n(D_FF, D_MODEL, sd=D_FF ** -0.5), f"c2_{l}": np.zeros(D_MODEL, dtype)})
    P.update(gf=np.ones(D_MODEL, dtype), bf=np.zeros(D_MODEL, dtype), Wout=n(D_MODEL, 1, sd=D_MODEL ** -0.5), bout=np.zeros(1, dtype)); return P


def n_params(P) -> int: return int(sum(v.size for v in P.values()))


def ortho_masks() -> Tuple[np.ndarray, np.ndarray]:
    """The two blocks of a (D_MODEL, N_HEAD * D_HEAD) query or key matrix that the penalty shrinks: payload part of the input -> pos columns, structural part of the input -> sem columns."""
    cols_pos = np.zeros(N_HEAD * D_HEAD, bool)
    for h in range(N_HEAD): cols_pos[h * D_HEAD:h * D_HEAD + D_POS] = True
    rows_pay = np.zeros(D_MODEL, bool); rows_pay[D_STRUCT:] = True
    m = np.zeros((D_MODEL, N_HEAD * D_HEAD), bool); m[np.ix_(rows_pay, cols_pos)] = True; m[np.ix_(~rows_pay, ~cols_pos)] = True; return m, ~m


def ortho_penalty(P) -> float:
    m, _ = ortho_masks(); return float(sum(((P[f"{w}_{l}"].astype(np.float64)) ** 2)[m].sum() for l in range(N_LAYER) for w in ("Wq", "Wk")))


def offblock_share(P) -> float:
    m, _ = ortho_masks(); num = sum(((P[f"{w}_{l}"].astype(np.float64)) ** 2)[m].sum() for l in range(N_LAYER) for w in ("Wq", "Wk")); den = sum(((P[f"{w}_{l}"].astype(np.float64)) ** 2).sum() for l in range(N_LAYER) for w in ("Wq", "Wk"))
    return float(num / den)


# ---------------------------------------------------------------- forward and backward
def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); xc = x - mu; var = (xc * xc).mean(-1, keepdims=True); inv = 1.0 / np.sqrt(var + eps); xh = xc * inv; return xh * g + b, (xh, inv, g)


def _ln_back(dy, c):
    xh, inv, g = c; dxh = dy * g; dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)); return dx, (dy * xh).reshape(-1, xh.shape[-1]).sum(0), dy.reshape(-1, xh.shape[-1]).sum(0)


def embed(P, batch):
    xs = P["emb_kind"][batch["kind"]] + P["emb_off"][batch["off"]]
    if "cx" in P: xs = xs + P["cx"][batch["x"]] + P["cy"][batch["y"]]
    return np.concatenate([xs, P["emb_pay"][batch["pay"]]], axis=-1)


def forward(P, batch, code: str, pp: dict, keep: bool = False, taps: Optional[dict] = None):
    """Logits (B, T). With `keep` the pieces the backward pass needs are returned as well; with `taps` (a dict) the attention probabilities of each layer are stored in it."""
    x = embed(P, batch); B, T, _ = x.shape; dt = x.dtype; caches = []; causal = np.triu(np.ones((T, T), bool), 1)
    for l in range(N_LAYER):
        a, c1 = _ln(x, P[f"g1_{l}"], P[f"b1_{l}"]); q = (a @ P[f"Wq_{l}"]).reshape(B, T, N_HEAD, D_HEAD).transpose(0, 2, 1, 3); k = (a @ P[f"Wk_{l}"]).reshape(B, T, N_HEAD, D_HEAD).transpose(0, 2, 1, 3)
        v = (a @ P[f"Wv_{l}"]).reshape(B, T, N_HEAD, D_HEAD).transpose(0, 2, 1, 3)
        qr = q.copy(); kr = k.copy(); qr[..., :D_POS] = rotate(q[..., :D_POS], pp, code); kr[..., :D_POS] = rotate(k[..., :D_POS], pp, code)
        s = (qr @ kr.transpose(0, 1, 3, 2)) / math.sqrt(D_HEAD); s = np.where(causal, np.asarray(-1e30, dt), s); s = s - s.max(-1, keepdims=True); p = np.exp(s); p = p / p.sum(-1, keepdims=True)
        if taps is not None: taps[l] = p
        ctx = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, N_HEAD * D_HEAD); x1 = x + ctx @ P[f"Wo_{l}"]
        m, c2 = _ln(x1, P[f"g2_{l}"], P[f"b2_{l}"]); hid = m @ P[f"W1_{l}"] + P[f"c1_{l}"]; r = np.maximum(hid, 0); x = x1 + r @ P[f"W2_{l}"] + P[f"c2_{l}"]
        if keep: caches.append(dict(a=a, c1=c1, q=q, k=k, v=v, qr=qr, kr=kr, p=p, ctx=ctx, m=m, c2=c2, hid=hid, r=r))
    z, cf = _ln(x, P["gf"], P["bf"]); logit = (z @ P["Wout"])[..., 0] + P["bout"][0]
    return (logit, dict(layers=caches, cf=cf, z=z)) if keep else logit


def loss_and_grads(P, batch, code: str, pp: dict, lam: float, pos_weight: float):
    """Weighted cross-entropy over the scored tokens (positives weigh `pos_weight`) plus lam times the penalty; returns (loss, ce, grads)."""
    logit, C = forward(P, batch, code, pp, keep=True); sc = batch["scored"]; y = batch["label"].astype(logit.dtype); n = max(int(sc.sum()), 1)
    sp = lambda u: np.logaddexp(0, u); sig = lambda u: 1.0 / (1.0 + np.exp(-u))
    ce = float(((pos_weight * y * sp(-logit) + (1 - y) * sp(logit)) * sc).sum() / n); dz = ((-pos_weight * y * sig(-logit) + (1 - y) * sig(logit)) * sc / n).astype(logit.dtype)
    G = {k: np.zeros_like(v) for k, v in P.items()}; B, T = dz.shape
    G["bout"][0] = dz.sum(); G["Wout"] = (C["z"].reshape(-1, D_MODEL).T @ dz.reshape(-1, 1)).astype(P["Wout"].dtype); dx, G["gf"], G["bf"] = _ln_back(dz[..., None] * P["Wout"][:, 0], C["cf"])
    for l in range(N_LAYER - 1, -1, -1):
        c = C["layers"][l]; dF = dx
        G[f"W2_{l}"] = c["r"].reshape(-1, D_FF).T @ dF.reshape(-1, D_MODEL); G[f"c2_{l}"] = dF.reshape(-1, D_MODEL).sum(0); dr = dF @ P[f"W2_{l}"].T; dhid = dr * (c["hid"] > 0)
        G[f"W1_{l}"] = c["m"].reshape(-1, D_MODEL).T @ dhid.reshape(-1, D_FF); G[f"c1_{l}"] = dhid.reshape(-1, D_FF).sum(0); dm = dhid @ P[f"W1_{l}"].T
        dx1, G[f"g2_{l}"], G[f"b2_{l}"] = _ln_back(dm, c["c2"]); dx1 = dx1 + dx
        G[f"Wo_{l}"] = c["ctx"].reshape(-1, N_HEAD * D_HEAD).T @ dx1.reshape(-1, D_MODEL); dctx = (dx1 @ P[f"Wo_{l}"].T).reshape(B, T, N_HEAD, D_HEAD).transpose(0, 2, 1, 3)
        p = c["p"]; dp = dctx @ c["v"].transpose(0, 1, 3, 2); dv = p.transpose(0, 1, 3, 2) @ dctx; ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(D_HEAD)
        dqr = ds @ c["kr"]; dkr = ds.transpose(0, 1, 3, 2) @ c["qr"]; dq = dqr.copy(); dk = dkr.copy(); dq[..., :D_POS] = rotate(dqr[..., :D_POS], pp, code, inverse=True); dk[..., :D_POS] = rotate(dkr[..., :D_POS], pp, code, inverse=True)
        flat = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, N_HEAD * D_HEAD); da = flat(dq) @ P[f"Wq_{l}"].T + flat(dk) @ P[f"Wk_{l}"].T + flat(dv) @ P[f"Wv_{l}"].T
        a2 = c["a"].reshape(-1, D_MODEL); G[f"Wq_{l}"] = a2.T @ flat(dq).reshape(-1, N_HEAD * D_HEAD); G[f"Wk_{l}"] = a2.T @ flat(dk).reshape(-1, N_HEAD * D_HEAD); G[f"Wv_{l}"] = a2.T @ flat(dv).reshape(-1, N_HEAD * D_HEAD)
        dxa, G[f"g1_{l}"], G[f"b1_{l}"] = _ln_back(da, c["c1"]); dx = dxa + dx1
    ds_ = dx[..., :D_STRUCT]; dp_ = dx[..., D_STRUCT:]
    np.add.at(G["emb_kind"], batch["kind"], ds_); np.add.at(G["emb_off"], batch["off"], ds_); np.add.at(G["emb_pay"], batch["pay"], dp_)
    if "cx" in P: np.add.at(G["cx"], batch["x"], ds_); np.add.at(G["cy"], batch["y"], ds_)
    pen = 0.0
    if lam > 0:
        m, _ = ortho_masks()
        for l in range(N_LAYER):
            for w in ("Wq", "Wk"):
                W = P[f"{w}_{l}"]; pen += float(((W.astype(np.float64)) ** 2)[m].sum()); G[f"{w}_{l}"] = G[f"{w}_{l}"] + (2 * lam) * (W * m)
    return ce + lam * pen, ce, G


# ---------------------------------------------------------------- training
class Adam:
    def __init__(self, P, b1=0.9, b2=0.999, eps=1e-8): self.m = {k: np.zeros_like(v) for k, v in P.items()}; self.v = {k: np.zeros_like(v) for k, v in P.items()}; self.t = 0; self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, P, G, lr):
        self.t += 1; c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * G[k]; self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * G[k] * G[k]; P[k] -= (lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)).astype(P[k].dtype)


def clip_global(G, c):
    nrm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in G.values()))
    if nrm > c:
        for k in G: G[k] = G[k] * (c / nrm)
    return nrm


def predict(P, batch, code: str, taps=None):
    return forward(P, batch, code, pos_params(code, batch, P["Wout"].dtype), taps=taps)


def evaluate(P, graphs: Sequence[dict], code: str, chunk: int = 25) -> dict:
    """Exact (every scored token right), F1 of the positives, and the shares of graphs with a false alarm and with a miss."""
    ex = fp = fn = tp = fpt = fnt = 0; n = 0
    for i in range(0, len(graphs), chunk):
        b = batch_of(graphs[i:i + chunk]); pr = predict(P, b, code) > 0; y = b["label"] > 0.5; sc = b["scored"]
        wrong = (pr != y) & sc; ex += int((~wrong.any(1)).sum()); fp += int(((pr & ~y) & sc).any(1).sum()); fn += int(((~pr & y) & sc).any(1).sum()); tp += int(((pr & y) & sc).sum()); fpt += int(((pr & ~y) & sc).sum()); fnt += int(((~pr & y) & sc).sum()); n += len(graphs[i:i + chunk])
    f1 = 2 * tp / (2 * tp + fpt + fnt) if (2 * tp + fpt + fnt) else 0.0
    return dict(n=n, exact=ex / n, f1=f1, graphs_with_false_alarm=fp / n, graphs_with_miss=fn / n, tokens_tp=tp, tokens_fp=fpt, tokens_fn=fnt)


def eval_graphs(name: str, n: int, g: int, shift: bool, count: int, seed: int) -> List[dict]:
    rng = random.Random(f"toy-eval/{seed}/{name}"); return [make_graph(rng, n, g, shift) for _ in range(count)]


def train_arm(arm: str, seed: int, cfg: Optional[dict] = None, say: Callable = lambda *a: None) -> Tuple[dict, dict]:
    cfg = dict(TRAIN, **(cfg or {})); spec = ARMS[arm]; code = spec["code"]; P = init_params(spec["table"], cfg["init_seed"] * 1000 + seed); opt = Adam(P)
    rng = random.Random(f"toy-train/{cfg['data_seed']}/{seed}"); hist = []; t0 = time.time(); steps = cfg["steps"]
    for it in range(1, steps + 1):
        lr = cfg["lr"] * (cfg["lr_end"] + (1 - cfg["lr_end"]) * 0.5 * (1 + math.cos(math.pi * (it - 1) / max(steps - 1, 1))))
        b = batch_of([make_graph(rng, TRAIN_N, TRAIN_N) for _ in range(cfg["batch"])]); pp = pos_params(code, b, P["Wout"].dtype)
        loss, ce, G = loss_and_grads(P, b, code, pp, spec["lam"], cfg["pos_weight"]); clip_global(G, cfg["clip"]); opt.step(P, G, lr)
        if it % max(steps // 10, 1) == 0 or it == steps:
            hist.append(dict(step=it, ce=ce, loss=loss, penalty=ortho_penalty(P))); say(arm, seed, it, round(ce, 4), round(hist[-1]["penalty"], 5), round(time.time() - t0, 1))
    return P, dict(history=hist, seconds=round(time.time() - t0, 1), params=n_params(P), offblock_share=offblock_share(P))


def payload_sensitivity(P, code: str, graphs: Sequence[dict], seed: int = 0) -> float:
    """Mean total-variation distance between the attention rows of layer 1 of the same graph with its payload ids resampled (kinds, offsets and cells unchanged): how much the payload moves where the heads look."""
    rng = random.Random(seed); tot = cnt = 0.0
    for i in range(0, len(graphs), 25):
        gs = graphs[i:i + 25]; g2 = copy.deepcopy(gs)
        for g in g2: g["tokens"] = [(k, c, o, (p if k == "Q" else rng.randint(1, 9)), y) for (k, c, o, p, y) in g["tokens"]]
        b1, b2 = batch_of(gs), batch_of(g2); t1, t2 = {}, {}; predict(P, b1, code, taps=t1); predict(P, b2, code, taps=t2)
        for l in (0,):
            a1, a2 = t1[l].astype(np.float64), t2[l].astype(np.float64); tv = 0.5 * np.abs(a1 - a2).sum(-1); m = b1["scored"][:, None, :]; tot += float((tv * m).sum()); cnt += float(m.sum() * N_HEAD)
    return tot / cnt


# ---------------------------------------------------------------- the run
def _digest(ev: Dict[str, List[dict]]) -> str:
    dig = hashlib.sha256()
    for name, gs in ev.items(): dig.update(repr([g["tokens"] for g in gs]).encode())
    return dig.hexdigest()[:16]


def full_eval_digest() -> str:
    """The digest of the evaluation graphs of the full run, without training anything (a notebook prints it before the long run: a machine whose Python makes other graphs shows another number)."""
    return _digest({s[0]: eval_graphs(s[0], s[1], s[2], s[3], EVAL_GRAPHS[s[0]], 0) for s in EVAL_SETS})


def run_toymodel(seeds: Sequence[int] = SEEDS, arms: Sequence[str] = tuple(ARMS), cfg: Optional[dict] = None, sets: Sequence[tuple] = EVAL_SETS, counts: Optional[dict] = None, progress=lambda *a: None) -> dict:
    t0 = time.time(); counts = dict(EVAL_GRAPHS, **(counts or {})); cfgf = dict(TRAIN, **(cfg or {})); res = dict(seeds=list(seeds), arms=list(arms), cfg=cfgf, sets={s[0]: dict(n=s[1], g=s[2], shift=s[3], graphs=counts[s[0]]) for s in sets}, runs={})
    ev = {s[0]: eval_graphs(s[0], s[1], s[2], s[3], counts[s[0]], 0) for s in sets}; res["eval_digest"] = _digest(ev)
    res["baseline_exact"] = {name: float(np.mean([all(t[4] == 0 for t in g["tokens"]) for g in gs])) for name, gs in ev.items()}                         # the all-zero answer is right exactly when no token is hit
    res["tokens_mean"] = {name: float(np.mean([len(g["tokens"]) for g in gs])) for name, gs in ev.items()}
    for sd in seeds:
        res["runs"][str(sd)] = {}
        for arm in arms:
            P, info = train_arm(arm, sd, cfg, say=progress); code = ARMS[arm]["code"]; out = {name: evaluate(P, gs, code) for name, gs in ev.items()}
            info["payload_sensitivity"] = payload_sensitivity(P, code, ev["N10"][:100]); res["runs"][str(sd)][arm] = dict(info, eval=out)
    res["seconds"] = round(time.time() - t0, 1); return res


# ---------------------------------------------------------------- the reading rules
TM_LINES = [
    ("guard", "every arm in every seed was trained for all its steps with a finite loss, and every evaluation set was scored on all its graphs"),
    ("guard", "the evaluation sets hold the sizes they are named for (graphs of 10, 20, 40, 70 and 100 services, with the grid side equal to the size; the range set has 10 services on a grid of 100)"),
    ("instrument", "every arm learned the task where it was trained: exact accuracy on N10 at least 0.80, for each of the five arms, mean over the seeds"),
    ("instrument", "the all-zero answer is not what the arms learned: on N10 every arm has an F1 of the positive tokens above 0.5"),
    ("claim", "the roadmap's Arm A (rope1d) does not carry what it learned to the large graphs: its exact accuracy on N100 is at most half its exact accuracy on N10"),
    ("claim", "the roadmap's Arm B (su2 with the penalty) keeps what it learned: its exact accuracy on N100 is at least 0.8 of its exact accuracy on N10"),
    ("claim", "Arm B beats Arm A on N100 by at least 0.20 in exact accuracy"),
    ("claim", "the stronger standard baseline (rope2d, no penalty) keeps what it learned: its exact accuracy on N100 is at least 0.8 of its exact accuracy on N10"),
    ("claim", "Arm B beats the stronger baseline rope2d on N100 by at least 0.05 in exact accuracy"),
    ("claim", "the penalty does what it is built for: the share of the squared weight of W_Q and W_K that sits in the penalised blocks is under 1 % for the arms with the penalty and above 10 % for the arms without"),
    ("claim", "the penalty makes the attention less sensitive to the payload: the mean change of the attention rows of the first layer when the payload ids are resampled is smaller for su2 with the penalty than for su2 without"),
    ("claim", "the penalty helps when the payload is unfamiliar: on pay_shift_N10 su2 with the penalty has an exact accuracy at least 0.05 above su2 without it"),
    ("claim", "range alone is enough to break Arm A: on range100_n10 (10 services, coordinates up to 99) rope1d has an exact accuracy at most half its exact accuracy on N10"),
    ("claim", "size alone is not what breaks the rotation arms: on N40 rope2d and su2 each keep at least 0.8 of their exact accuracy on N10"),
]
TM_HEADLINE = (5, 6, 7)                                     # the roadmap's claim: the standard arm fails at N = 100, the spinor arm succeeds
TM_MIN_N10, TM_MIN_F1, TM_HALF, TM_KEEP, TM_B_OVER_A, TM_B_OVER_ROPE2D, TM_SHARE_LO, TM_SHARE_HI, TM_SHIFT_GAIN = 0.80, 0.5, 0.5, 0.8, 0.20, 0.05, 0.01, 0.10, 0.05
TM_SIZES = {"N10": (10, 10), "N20": (20, 20), "N40": (40, 40), "N70": (70, 70), "N100": (100, 100), "range100_n10": (10, 100)}


def _mean_arm(res: dict, arm: str, key: Callable[[dict], float]) -> Optional[float]:
    v = []
    for r in res["runs"].values():
        if arm not in r: continue
        try: v.append(key(r[arm]))
        except KeyError: return None                                                        # an evaluation set the run does not hold (a quick run): the rule that needs it reads "not run"
    return float(np.mean(v)) if v else None


def acc(res: dict, arm: str, name: str, field: str = "exact") -> Optional[float]: return _mean_arm(res, arm, lambda a: a["eval"][name][field])


def verdict_toymodel(res: dict) -> List[Tuple[str, Optional[bool]]]:
    out: List[Tuple[str, Optional[bool]]] = []; arms = res["arms"]; g = lambda arm, name: acc(res, arm, name)
    steps = res["cfg"]["steps"]
    out.append(("guard", bool(res["runs"]) and all(all(a["history"] and a["history"][-1]["step"] == steps and all(math.isfinite(h["ce"]) and math.isfinite(h["loss"]) for h in a["history"]) and all(e["n"] == res["sets"][nm]["graphs"] for nm, e in a["eval"].items()) for a in r.values()) for r in res["runs"].values())))
    out.append(("guard", all(res["sets"].get(nm, {}).get("n") == n and res["sets"][nm]["g"] == gg for nm, (n, gg) in TM_SIZES.items() if nm in res["sets"]) and all(nm in res["sets"] for nm in TM_SIZES) and all(res["tokens_mean"][nm] > 0 for nm in res["sets"])))
    have = [a for a in ARMS if a in arms]
    out.append(("instrument", None if len(have) < len(ARMS) else all(g(a, "N10") >= TM_MIN_N10 for a in ARMS)))
    out.append(("instrument", None if len(have) < len(ARMS) else all(acc(res, a, "N10", "f1") > TM_MIN_F1 for a in ARMS)))
    a1, a10 = g("A_rope1d", "N100"), g("A_rope1d", "N10"); out.append(("claim", None if a1 is None else a1 <= TM_HALF * a10))
    b1, b10 = g("B_su2", "N100"), g("B_su2", "N10"); out.append(("claim", None if b1 is None else b1 >= TM_KEEP * b10))
    out.append(("claim", None if a1 is None or b1 is None else b1 - a1 >= TM_B_OVER_A))
    r1, r10 = g("A_rope2d", "N100"), g("A_rope2d", "N10"); out.append(("claim", None if r1 is None else r1 >= TM_KEEP * r10))
    out.append(("claim", None if b1 is None or r1 is None else b1 - r1 >= TM_B_OVER_ROPE2D))
    sh = lambda arm: _mean_arm(res, arm, lambda a: a["offblock_share"]); pen_arms = [a for a in ARMS if ARMS[a]["lam"] > 0]; free_arms = [a for a in ARMS if ARMS[a]["lam"] == 0]
    out.append(("claim", None if not all(a in arms for a in ARMS) else all(sh(a) < TM_SHARE_LO for a in pen_arms) and all(sh(a) > TM_SHARE_HI for a in free_arms)))
    ps = lambda arm: _mean_arm(res, arm, lambda a: a["payload_sensitivity"]); out.append(("claim", None if "B_su2" not in arms or "B_su2_free" not in arms else ps("B_su2") < ps("B_su2_free")))
    p1, p2 = g("B_su2", "pay_shift_N10"), g("B_su2_free", "pay_shift_N10"); out.append(("claim", None if p1 is None or p2 is None else p1 - p2 >= TM_SHIFT_GAIN))
    q, q10 = g("A_rope1d", "range100_n10"), a10; out.append(("claim", None if q is None else q <= TM_HALF * q10))
    k = [(g(a, "N40"), g(a, "N10")) for a in ("A_rope2d", "B_su2")]; out.append(("claim", None if any(x is None for p in k for x in p) else all(x >= TM_KEEP * y for x, y in k)))
    return out


# ---------------------------------------------------------------- printing and the command line
def _pc(x: Optional[float]) -> str: return "   -  " if x is None else f"{100 * x:.1f}%"


def print_toymodel(res: dict) -> None:
    print(f"=== the toy model (seeds {res['seeds']}, {res['cfg']['steps']} steps of {res['cfg']['batch']} graphs, {res['seconds']} s, evaluation digest {res['eval_digest']}) ===")
    names = list(res["sets"]); print(f"{'arm':14s} " + " ".join(f"{n:>14s}" for n in names) + f" {'offblock':>9s} {'payload sens.':>13s}")
    print(f"{'all-zero':14s} " + " ".join(f"{_pc(res['baseline_exact'][n]):>14s}" for n in names))
    for arm in res["arms"]:
        row = " ".join(f"{_pc(acc(res, arm, n)) + ' F' + format(acc(res, arm, n, 'f1'), '.2f'):>14s}" for n in names)
        print(f"{arm:14s} {row} {_mean_arm(res, arm, lambda a: a['offblock_share']):9.4f} {_mean_arm(res, arm, lambda a: a['payload_sensitivity']):13.4f}")


def print_verdict_toymodel(checks) -> None:
    print("--- reading rules ---")
    for i, (kind, v) in enumerate(checks, 1): print(f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}" + ("   (headline)" if i in TM_HEADLINE else ""))
    head = [checks[i - 1][1] for i in TM_HEADLINE]; print("HEADLINE (the standard arm fails at N = 100, the spinor arm does not, and the spinor arm is clearly better):", "ALL YES" if all(head) else "NOT all yes")


def merge_results(parts: Sequence[dict]) -> dict:
    """Put the files of partial runs together (a seed, an arm of a seed, a few seeds): same settings, same evaluation graphs; no (seed, arm) twice; at the end every seed holds the same arms."""
    assert parts, "nothing to merge"; first = parts[0]; runs: Dict[str, dict] = {}
    for p in parts:
        for key in ("cfg", "sets", "eval_digest", "baseline_exact", "tokens_mean"): assert p[key] == first[key], f"the files disagree on {key}"
        assert bool(p.get("quick")) == bool(first.get("quick")), "a quick run cannot be merged with a full one"
        for sd, r in p["runs"].items():
            for arm, a in r.items():
                assert arm not in runs.setdefault(sd, {}), f"seed {sd}, arm {arm} is in two files"; runs[sd][arm] = a
    arms = [a for a in ARMS if any(a in r for r in runs.values())]
    for sd, r in runs.items(): assert set(r) == set(arms), f"seed {sd} lacks the arms {sorted(set(arms) - set(r))}"
    out = dict(first); out["runs"] = {sd: {a: runs[sd][a] for a in arms} for sd in sorted(runs, key=int)}; out["seeds"] = [int(s) for s in out["runs"]]; out["arms"] = arms; out["seconds"] = round(sum(p["seconds"] for p in parts), 1); return out


QUICK = dict(seeds=(1,), arms=("A_rope1d", "B_su2"), cfg=dict(steps=40, batch=8), counts={k: 8 for k in EVAL_GRAPHS}, sets=EVAL_SETS[:2] + EVAL_SETS[5:6])


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="the toy model (docs/TOY_MODEL_TEST.md)")
    ap.add_argument("--out", default="axi_toymodel.json"); ap.add_argument("--quick", action="store_true", help="a few seconds, small samples: to see that it runs, not to read")
    ap.add_argument("--merge", nargs="+", default=None, help="put the files of partial runs (a seed, an arm of a seed) together, no training, and print the reading rules"); ap.add_argument("--seeds", default=None, help="comma separated, default 1,2,3 (a run of one seed is one third of the work; send each file back)"); ap.add_argument("--arms", default=None, help="comma separated subset of the five arms")
    a = ap.parse_args(argv)
    if a.merge:
        res = merge_results([json.load(open(f)) for f in a.merge])
        with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
        print_toymodel(res); print_verdict_toymodel(verdict_toymodel(res)); print("saved", os.path.abspath(a.out)); return
    kw = dict(QUICK) if a.quick else {}
    if a.seeds: kw["seeds"] = tuple(int(s) for s in a.seeds.split(","))
    if a.arms: kw["arms"] = tuple(a.arms.split(","))
    res = run_toymodel(progress=lambda arm, sd, it, ce, pen, t: print(f"  {arm} seed {sd} step {it}: ce {ce} penalty {pen} ({t} s)", flush=True), **kw); res["quick"] = bool(a.quick)
    with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
    print_toymodel(res); print_verdict_toymodel(verdict_toymodel(res)); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()

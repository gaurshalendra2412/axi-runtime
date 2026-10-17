"""Engineering for docs/HYPERBOLIC_SNAP_TEST.md (concept-map item 104): what does it cost, and what does it buy, to turn the last hidden vector of a language model into a token by the nearest of the allowed tokens in a hyperbolic ball?

Two stages.
  STAGE 1 (a GPU; `collect`, `torch_runner`): an open model reads a prompt (a service graph and a request, the repository's own prompt) followed by the canonical reply, teacher forced. At every position of the reply the
  program stores the last hidden vector before the final norm (`h_pre`) and after it (`h_post`, the input of the output head), the token the model itself would pick over the whole vocabulary (`free_top`), the
  grammar state, the set of tokens the grammar allows there and the model's logit for each of them, and the canonical token (`gold`). The output-head rows of every token that is ever allowed are stored too.
  STAGE 2 (numpy, no torch; `analyze`): the decisions of the arms, from the stored arrays alone:
    free          the model's own pick over the whole vocabulary (the roadmap's "Euclidean projection and softmax")
    masked        the allowed token with the highest logit (what the repository's grammar mask does)
    euclid_nn     the allowed token whose output-head row is nearest to the hidden vector (both scaled to norm 1), no free parameter
    euclid_tuned  the same with the hidden vector scaled by one number chosen on half of the examples
    cosine_nn     the allowed token with the highest cosine
    hyper_nn      the allowed token nearest in the Poincare ball (exponential map at the origin, curvature chosen on half of the examples from a fixed list)
    lin_dot, lin_eu, lin_hyp   three heads of the same size trained on half of the examples (a linear map to 32 numbers and one 32-number prototype per allowed token): a dot product, a Euclidean distance, a
                               hyperbolic distance
All arms except `free` choose among the allowed tokens, so none of them can produce a token the grammar refuses: that is true by construction and says nothing about the geometry.
What differs between them is which allowed token they choose, and that is what is measured against the canonical reply."""
import hashlib, json, math, os, random, sys, time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- the design (fixed in docs/HYPERBOLIC_SNAP_TEST.md before the run)
CONDITIONS = ("full", "bare")
N_TASKS, TASK_SEED = 400, 31
GRID_ALPHA = (0.25, 0.5, 1.0, 2.0, 4.0)
GRID_C = (0.01, 0.1, 1.0, 10.0, 100.0)
HEAD_DIM = 32
HEAD = dict(epochs=25, batch=256, lr=0.01, seed=5)
ARMS = ("free", "masked", "euclid_nn", "euclid_tuned", "cosine_nn", "hyper_nn", "lin_dot", "lin_eu", "lin_hyp")
RESTRICTED = ARMS[1:]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SLIP_REPLY, SLIP_TOKEN_LINE = 0.05, 0.05
EPS = 1e-6


# ---------------------------------------------------------------- stage 1
def messages_for(task, condition: str) -> List[dict]:
    """`full`: the repository's prompt (system text with the item syntax and examples, the state, the request). `bare`: the state and the request only, no description of the syntax."""
    from axi.experiments.agent_loop import base_messages, _serialize
    if condition == "full": return base_messages(task, False)
    return [{"role": "user", "content": f"{_serialize(task.graph)}\n\nRequest: {task.instruction}"}]


def make_tasks(n: int = N_TASKS, seed: int = TASK_SEED):
    from axi.experiments.agent_loop import make_tasks as mk
    return mk(n, seed)


def check_walkable(table, encode: Callable, eos: int, tasks: Sequence) -> List[Tuple[int, int, int]]:
    """Before any model is run: the canonical replies as the tokenizer cuts them, walked through the grammar. Returns (task, position, token) of every token the grammar refuses (an empty list is what the page's line 2 needs)."""
    bad = []; V = table.V
    for i, task in enumerate(tasks):
        state = table.start
        for j, g in enumerate(list(encode(task.ideal)) + [eos]):
            if not (g < V and table.allowed[state, g]): bad.append((i, j, int(g))); break
            if g == eos: break
            state = int(table.next[state, g])
    return bad


def norm_identity_error(d: dict, apply_norm: Callable[[np.ndarray], np.ndarray], rows: int = 20) -> float:
    """Largest difference between the stored `h_post` and the final norm applied to the stored `h_pre`, over the first `rows` positions, as a share of the largest stored `h_post` value: a check that the hook read the right tensor."""
    pre = d["h_pre"][:rows].astype(np.float32); post = d["h_post"][:rows].astype(np.float32); again = np.asarray(apply_norm(pre), dtype=np.float32)
    return float(np.abs(again - post).max() / max(float(np.abs(post).max()), 1e-12))


def collect(runner: Callable, prompt_ids_fn: Callable, encode: Callable, table, eos: int, tasks: Sequence, condition: str, say: Callable = lambda *a: None) -> dict:
    """Walk the canonical reply of every task through the grammar and store what stage 2 needs.
    `runner(prompt_ids, reply_in_ids) -> dict(h_pre=[R, D], h_post=[R, D], logits=callable(j) -> 1-D array over the whole output vocabulary)` with R = len(reply_in_ids) + 1 (the reply and the end token);
    `prompt_ids_fn(messages) -> list of ids`; `encode(text) -> list of ids`."""
    rows = dict(ex=[], j=[], gold=[], state=[], free_top=[], free_logit=[], gold_logit=[], gold_in=[], h_pre=[], h_post=[]); a_ptr = [0]; a_ids: List[np.ndarray] = []; a_logit: List[np.ndarray] = []; broken = 0; V = table.V
    for i, task in enumerate(tasks):
        p_ids = prompt_ids_fn(messages_for(task, condition)); reply = list(encode(task.ideal)); gold = reply + [eos]; out = runner(p_ids, reply)
        assert out["h_pre"].shape[0] == len(gold) == out["h_post"].shape[0], (out["h_pre"].shape, len(gold)); state = table.start; stop = False
        for j, g in enumerate(gold):
            ok = bool(g < V and table.allowed[state, g]); ids = np.flatnonzero(table.allowed[state]).astype(np.int32)
            lg = np.asarray(out["logits"](j), dtype=np.float32)
            rows["ex"].append(i); rows["j"].append(j); rows["gold"].append(g); rows["state"].append(state); rows["free_top"].append(int(lg.argmax())); rows["free_logit"].append(float(lg.max())); rows["gold_logit"].append(float(lg[g]))
            rows["gold_in"].append(ok); rows["h_pre"].append(out["h_pre"][j].astype(np.float16)); rows["h_post"].append(out["h_post"][j].astype(np.float16)); a_ids.append(ids); a_logit.append(lg[ids]); a_ptr.append(a_ptr[-1] + len(ids))
            if not ok: broken += 1; break                                                    # the grammar does not allow the canonical token: the rest of this reply cannot be walked
            if g == eos: break
            state = int(table.next[state, g])
        if (i + 1) % 50 == 0: say(condition, i + 1, len(tasks), len(rows["ex"]))
    data = {k: np.array(v) for k, v in rows.items()}; data["a_ptr"] = np.array(a_ptr, np.int64); data["a_ids"] = np.concatenate(a_ids).astype(np.int32); data["a_logit"] = np.concatenate(a_logit).astype(np.float32)
    data["condition"] = condition; data["broken"] = broken; return data


def attach_unembedding(datasets: Sequence[dict], unembed: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The rows of the output head of every token that is allowed anywhere: returns the ids; each dataset gets `U_ids` and `U` (shared)."""
    ids = np.unique(np.concatenate([d["a_ids"] for d in datasets])); U = np.asarray(unembed(ids), dtype=np.float32).astype(np.float16)
    for d in datasets: d["U_ids"] = ids; d["U"] = U
    return ids


def torch_runner(model, tok, device: str = "cuda") -> Tuple[Callable, Callable, Callable]:
    """(runner, prompt_ids_fn, unembed) for a Hugging Face causal model. The last hidden vector before the final norm is read with a hook on the norm; the output head is applied to one position at a time."""
    import torch
    store = {}
    hook = model.model.norm.register_forward_pre_hook(lambda m, args: store.__setitem__("pre", args[0].detach()))

    def prompt_ids_fn(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True); return enc["input_ids"][0].tolist()

    def runner(p_ids, reply_in):
        ids = torch.tensor([list(p_ids) + list(reply_in)], device=device); n = len(p_ids)
        with torch.no_grad():
            out = model.model(input_ids=ids, use_cache=False); post = out.last_hidden_state[0, n - 1:]; pre = store["pre"][0, n - 1:]
        def logits(j):
            with torch.no_grad(): return model.lm_head(post[j:j + 1])[0].float().cpu().numpy()
        return dict(h_pre=pre.float().cpu().numpy(), h_post=post.float().cpu().numpy(), logits=logits)

    def unembed(ids):
        with torch.no_grad(): return model.lm_head.weight[torch.tensor(ids, device=model.lm_head.weight.device)].detach().float().cpu().numpy()
    runner.hook = hook; return runner, prompt_ids_fn, unembed


def save_data(path: str, data: dict) -> None:
    np.savez_compressed(path, **{k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in data.items()})


def load_data(path: str) -> dict:
    z = np.load(path, allow_pickle=False); out = {k: z[k] for k in z.files}
    for k in ("condition",): out[k] = str(out[k]); out["broken"] = int(out["broken"])
    return out


# ---------------------------------------------------------------- geometry
def expmap0(v: np.ndarray, c: float) -> np.ndarray:
    """Exponential map at the origin of the Poincare ball of curvature -c: tanh(sqrt(c) |v|) v / (sqrt(c) |v|); the norm of the result is kept below 1/sqrt(c)."""
    r = np.linalg.norm(v, axis=-1, keepdims=True); sc = math.sqrt(c); t = np.tanh(np.minimum(sc * r, 15.0)) / np.maximum(sc * r, 1e-12)
    out = v * t; n = np.linalg.norm(out, axis=-1, keepdims=True); cap = (1 - EPS) / sc
    return np.where(n > cap, out * (cap / np.maximum(n, 1e-12)), out)


def poincare_dist(u: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    """Pairwise distance between the rows of u [B, n] and the rows of w [K, n] in the ball of curvature -c: arcosh(1 + 2 c |u - w|^2 / ((1 - c|u|^2)(1 - c|w|^2))) / sqrt(c)."""
    uu, ww = (u * u).sum(-1)[:, None], (w * w).sum(-1)[None, :]; d2 = np.maximum(uu + ww - 2 * u @ w.T, 0.0)
    g = 1 + 2 * c * d2 / (np.maximum(1 - c * uu, EPS) * np.maximum(1 - c * ww, EPS)); return np.arccosh(np.maximum(g, 1.0)) / math.sqrt(c)


# ---------------------------------------------------------------- stage 2: the decisions
def allowed_mask(d: dict, idx: np.ndarray) -> np.ndarray:
    """Boolean [len(idx), K] over the columns of U, True where the token is allowed at that position."""
    col = np.full(int(d["a_ids"].max()) + 1, -1, np.int64); col[d["U_ids"]] = np.arange(len(d["U_ids"])); m = np.zeros((len(idx), len(d["U_ids"])), bool)
    for r, i in enumerate(idx): m[r, col[d["a_ids"][d["a_ptr"][i]:d["a_ptr"][i + 1]]]] = True
    return m


def gold_col(d: dict) -> np.ndarray:
    col = np.full(int(max(d["U_ids"].max(), d["gold"].max())) + 1, -1, np.int64); col[d["U_ids"]] = np.arange(len(d["U_ids"])); return col[d["gold"]]


def norms_for(d: dict, rows: np.ndarray, key: str = "h_post") -> Tuple[float, float]:
    """(median norm of the hidden vectors of `rows`, median norm of the output-head rows)."""
    return float(np.median(np.linalg.norm(d[key][rows].astype(np.float32), axis=1))), float(np.median(np.linalg.norm(d["U"].astype(np.float32), axis=1)))


def decide_fixed(d: dict, idx: np.ndarray, arm: str, param: Optional[float], mh: float, mu: float, key: str = "h_post", chunk: int = 1000) -> np.ndarray:
    """The token id chosen by a fixed-prototype arm at the positions `idx`."""
    U = d["U"].astype(np.float64) / mu; out = np.empty(len(idx), np.int64)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; H = d[key][ii].astype(np.float64) / mh; M = allowed_mask(d, ii)
        if arm == "euclid_nn": sc = -((H ** 2).sum(1)[:, None] + (U ** 2).sum(1)[None, :] - 2 * H @ U.T)
        elif arm == "euclid_tuned": Hs = H * param; sc = -((Hs ** 2).sum(1)[:, None] + (U ** 2).sum(1)[None, :] - 2 * Hs @ U.T)
        elif arm == "cosine_nn": sc = (H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)) @ (U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)).T
        elif arm == "hyper_nn": sc = -poincare_dist(expmap0(H, param), expmap0(U, param), param)
        else: raise KeyError(arm)
        out[s:s + chunk] = d["U_ids"][np.where(M, sc, -np.inf).argmax(1)]
    return out


def decide_masked(d: dict, idx: np.ndarray) -> np.ndarray:
    out = np.empty(len(idx), np.int64)
    for r, i in enumerate(idx): a, b = d["a_ptr"][i], d["a_ptr"][i + 1]; out[r] = d["a_ids"][a + int(d["a_logit"][a:b].argmax())]
    return out


def is_allowed(d: dict, idx: np.ndarray, tok: np.ndarray) -> np.ndarray:
    return np.array([bool((d["a_ids"][d["a_ptr"][i]:d["a_ptr"][i + 1]] == t).any()) for i, t in zip(idx, tok)])


def fit_scalar(d: dict, arm: str, rows: np.ndarray, grid: Sequence[float], mh: float, mu: float, key: str = "h_post") -> Tuple[float, Dict[float, float]]:
    """The value of the one free number of an arm that gives the highest accuracy on `rows` (ties: the first in the grid)."""
    acc = {g: float((decide_fixed(d, rows, arm, g, mh, mu, key) == d["gold"][rows]).mean()) for g in grid}; best = max(grid, key=lambda g: (acc[g], -grid.index(g))); return best, acc


# ---------------------------------------------------------------- stage 2: the learned heads
def pi_forward(z: np.ndarray) -> np.ndarray: return expmap0(z, 1.0)


def pi_backward(z: np.ndarray, du: np.ndarray) -> np.ndarray:
    """Gradient of a loss with respect to z, given its gradient du with respect to pi(z) = tanh(|z|) z / |z|."""
    r = np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9); t = np.tanh(np.minimum(r, 15.0)); zd = (z * du).sum(-1, keepdims=True)
    return (t / r) * du + (1 - t * t - t / r) * z * zd / (r * r)


class Head:
    """A linear map A (n x D) of the normalised hidden vector and one prototype Q_k (n numbers) for every allowed token; the score of token k is `dot`: Z Q_k, `eu`: -tau |Z - Q_k|^2, `hyp`: -tau d_B(pi(Z), pi(Q_k))."""
    def __init__(self, kind: str, D: int, K: int, seed: int, n: int = HEAD_DIM):
        rng = np.random.default_rng(seed); self.kind = kind; self.A = (rng.standard_normal((n, D)) / math.sqrt(D)).astype(np.float64); self.Q = (rng.standard_normal((K, n)) / math.sqrt(n)).astype(np.float64); self.logtau = 0.0

    def init_prototypes(self, U: np.ndarray) -> None: self.Q = (U @ self.A.T).astype(np.float64)

    def scores(self, H: np.ndarray, want_cache: bool = False):
        Z = H @ self.A.T; tau = math.exp(self.logtau)
        if self.kind == "dot": S = Z @ self.Q.T; cache = (Z,)
        elif self.kind == "eu": D2 = (Z * Z).sum(1)[:, None] + (self.Q * self.Q).sum(1)[None, :] - 2 * Z @ self.Q.T; S = -tau * D2; cache = (Z, D2)
        elif self.kind == "hyp": u, w = pi_forward(Z), pi_forward(self.Q); Dh = poincare_dist(u, w, 1.0); S = -tau * Dh; cache = (Z, u, w, Dh)
        else: raise KeyError(self.kind)
        return (S, cache) if want_cache else S

    def grads(self, H: np.ndarray, M: np.ndarray, gold: np.ndarray):
        """Cross-entropy of the gold column over the allowed columns M; returns (loss, dA, dQ, dlogtau)."""
        S, cache = self.scores(H, True); B = len(H); Sm = np.where(M, S, -np.inf); Sm = Sm - Sm.max(1, keepdims=True); P = np.exp(Sm); P = P / P.sum(1, keepdims=True)
        loss = float(-np.log(np.maximum(P[np.arange(B), gold], 1e-300)).mean()); dS = P.copy(); dS[np.arange(B), gold] -= 1.0; dS /= B; tau = math.exp(self.logtau); dlt = 0.0
        if self.kind == "dot": Z = cache[0]; dZ = dS @ self.Q; dQ = dS.T @ Z
        elif self.kind == "eu":
            Z, D2 = cache; dlt = float((dS * (-tau * D2)).sum()); G = dS * (-tau); dZ = 2 * (G.sum(1)[:, None] * Z - G @ self.Q); dQ = 2 * (G.sum(0)[:, None] * self.Q - G.T @ Z)
        else:
            Z, u, w, Dh = cache; dlt = float((dS * (-tau * Dh)).sum()); G = dS * (-tau)
            uu, ww = (u * u).sum(1)[:, None], (w * w).sum(1)[None, :]; al, be = np.maximum(1 - uu, EPS), np.maximum(1 - ww, EPS); d2 = np.maximum(uu + ww - 2 * u @ w.T, 0.0); gam = np.maximum(1 + 2 * d2 / (al * be), 1.0 + 1e-9)
            coef = G * (4.0 / (al * be)) / np.sqrt(gam * gam - 1.0)
            du = u * (coef * (1 + d2 / al)).sum(1)[:, None] - coef @ w; dw = w * (coef * (1 + d2 / be)).sum(0)[:, None] - coef.T @ u
            dZ = pi_backward(Z, du); dQ = pi_backward(self.Q, dw)
        return loss, dZ.T @ H, dQ, dlt


def train_head(d: dict, kind: str, rows: np.ndarray, mh: float, mu: float, cfg: Optional[dict] = None, key: str = "h_post") -> Tuple[Head, List[float]]:
    cfg = dict(HEAD, **(cfg or {})); D = d[key].shape[1]; K = len(d["U_ids"]); head = Head(kind, D, K, cfg["seed"]); head.init_prototypes(d["U"].astype(np.float32).astype(np.float64) / mu)
    gc = gold_col(d); rng = np.random.default_rng(cfg["seed"] + 1); mA, vA, mQ, vQ, mt, vt = (np.zeros_like(head.A), np.zeros_like(head.A), np.zeros_like(head.Q), np.zeros_like(head.Q), 0.0, 0.0); t = 0; curve = []
    for ep in range(cfg["epochs"]):
        order = rng.permutation(rows); tot = 0.0; nb = 0
        for s in range(0, len(order), cfg["batch"]):
            ii = order[s:s + cfg["batch"]]; H = d[key][ii].astype(np.float64) / mh; M = allowed_mask(d, ii); loss, dA, dQ, dl = head.grads(H, M, gc[ii]); t += 1; tot += loss; nb += 1
            for name, g in (("A", dA), ("Q", dQ)):
                m_, v_ = (mA, vA) if name == "A" else (mQ, vQ); m_ *= 0.9; m_ += 0.1 * g; v_ *= 0.999; v_ += 0.001 * g * g; step = cfg["lr"] * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
                if name == "A": head.A -= step
                else: head.Q -= step
            if kind != "dot":
                mt = 0.9 * mt + 0.1 * dl; vt = 0.999 * vt + 0.001 * dl * dl; head.logtau -= cfg["lr"] * (mt / (1 - 0.9 ** t)) / (math.sqrt(vt / (1 - 0.999 ** t)) + 1e-8)
        curve.append(tot / max(nb, 1))
    return head, curve


def decide_head(d: dict, head: Head, idx: np.ndarray, mh: float, key: str = "h_post", chunk: int = 1000) -> np.ndarray:
    out = np.empty(len(idx), np.int64)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; H = d[key][ii].astype(np.float64) / mh; M = allowed_mask(d, ii); out[s:s + chunk] = d["U_ids"][np.where(M, head.scores(H), -np.inf).argmax(1)]
    return out


# ---------------------------------------------------------------- stage 2: the run
def split_rows(d: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(calibration and training rows, test rows): even and odd examples; a position where the grammar refuses the canonical token is left out of both."""
    ex, ok = d["ex"], d["gold_in"].astype(bool); return np.flatnonzero((ex % 2 == 0) & ok), np.flatnonzero((ex % 2 == 1) & ok)


def tally(d: dict, idx: np.ndarray, tok: np.ndarray) -> dict:
    gold = d["gold"][idx]; ok = tok == gold; adm = is_allowed(d, idx, tok); slip = ~is_allowed(d, idx, d["free_top"][idx]); n = len(idx)
    return dict(n=n, correct=int(ok.sum()), admissible=int(adm.sum()), n_slip=int(slip.sum()), correct_on_slip=int((ok & slip).sum()), correct_off_slip=int((ok & ~slip).sum()))


def analyze(d: dict, say: Callable = lambda *a: None, head_cfg: Optional[dict] = None) -> dict:
    """Everything stage 2 reports for one condition, from the arrays of `collect`."""
    cal, test = split_rows(d); res = dict(condition=d["condition"], positions=int(len(d["ex"])), examples=int(d["ex"].max() + 1), K=int(len(d["U_ids"])), broken=int(d["broken"]), gold_in_allowed=bool(d["gold_in"].all()), arms={}, calibration={})
    # the model's own pick: how often is it a token the grammar refuses, per position and per reply
    free = d["free_top"]; slip_all = ~is_allowed(d, np.arange(len(free)), free); res["free"] = dict(slip_positions=int(slip_all.sum()), positions=int(len(free)), slip_share=float(slip_all.mean()),
        replies=int(d["ex"].max() + 1), replies_with_slip=int(len({int(e) for e in d["ex"][slip_all]})), gold_is_free_top=float((free == d["gold"]).mean()))
    res["arms"]["free"] = tally(d, test, free[test]); res["arms"]["masked"] = tally(d, test, decide_masked(d, test))
    mh, mu = norms_for(d, cal); res["calibration"]["norms"] = dict(hidden=mh, head_rows=mu)
    for arm in ("euclid_nn", "cosine_nn"): res["arms"][arm] = tally(d, test, decide_fixed(d, test, arm, None, mh, mu))
    a, acc = fit_scalar(d, "euclid_tuned", cal, list(GRID_ALPHA), mh, mu); res["calibration"]["alpha"] = dict(chosen=a, accuracy={str(k): v for k, v in acc.items()}); res["arms"]["euclid_tuned"] = tally(d, test, decide_fixed(d, test, "euclid_tuned", a, mh, mu))
    c, acc = fit_scalar(d, "hyper_nn", cal, list(GRID_C), mh, mu); res["calibration"]["curvature"] = dict(chosen=c, accuracy={str(k): v for k, v in acc.items()}); res["arms"]["hyper_nn"] = tally(d, test, decide_fixed(d, test, "hyper_nn", c, mh, mu))
    res["heads"] = {}
    for kind, name in (("dot", "lin_dot"), ("eu", "lin_eu"), ("hyp", "lin_hyp")):
        head, curve = train_head(d, kind, cal, mh, mu, head_cfg); res["arms"][name] = tally(d, test, decide_head(d, head, test, mh)); res["heads"][name] = dict(loss_curve=[float(x) for x in curve], logtau=float(head.logtau)); say(d["condition"], name, res["arms"][name]["correct"], res["arms"][name]["n"])
    res["arms"] = {a: res["arms"][a] for a in ARMS}; return res


# ---------------------------------------------------------------- the reading rules
HY_LINES = [
    ("guard", "both conditions were collected for all 400 tasks, every position of every reply was walked, and the output-head rows were stored for every token that is allowed anywhere"),
    ("instrument", "at every position of both conditions the canonical token is one the grammar allows (the grammar admits the replies the test expects)"),
    ("instrument", "every arm except `free` chose a token the grammar allows at 100 % of the test positions of both conditions (true by construction: this checks the code and is not a finding)"),
    ("claim", "condition full (the repository's prompt): the model's own pick over the whole vocabulary is a token the grammar refuses in at least 5 % of the replies"),
    ("claim", "condition bare (no description of the syntax): the model's own pick is a token the grammar refuses in at least 5 % of the replies"),
    ("claim", "condition full: the model's own pick is refused by the grammar at at least 5 % of the positions"),
    ("claim", "condition full: `masked` picks the canonical token at at least 95 % of the test positions"),
    ("claim", "condition full: `hyper_nn` picks the canonical token at least as often as `masked` minus 2 points"),
    ("claim", "condition full: `hyper_nn` beats `euclid_tuned` by at least 2 points of the canonical token"),
    ("claim", "condition full, at the test positions where the model's own pick is refused: `masked` picks the canonical token at least as often as `hyper_nn`"),
    ("claim", "condition full: `lin_hyp` beats `lin_eu` by at least 1 point"),
    ("claim", "condition full: `lin_hyp` beats `lin_dot` by at least 1 point"),
    ("claim", "condition full: `lin_dot` is within 2 points of `masked` or above it"),
]
HY_HEADLINE = (5, 8)                                       # the roadmap's claim: the model's own pick breaks the syntax, and the hyperbolic snap does not and loses nothing against masking
HY_SHARE, HY_MASKED, HY_SLACK, HY_GAIN_EU, HY_GAIN_HEAD, HY_GAIN_DOT = 0.05, 0.95, 0.02, 0.02, 0.01, 0.01


def _acc(res: dict, arm: str, field: str = "correct", of: str = "n") -> float:
    a = res["arms"][arm]; return a[field] / a[of] if a[of] else 0.0


def verdict_hypsnap(results: Dict[str, dict], tasks: int = N_TASKS) -> List[Tuple[str, Optional[bool]]]:
    f, b = results.get("full"), results.get("bare"); out: List[Tuple[str, Optional[bool]]] = []
    out.append(("guard", f is not None and b is not None and all(r["examples"] == tasks and r["broken"] == 0 and r["K"] > 0 for r in (f, b))))
    out.append(("instrument", None if f is None or b is None else all(r["gold_in_allowed"] for r in (f, b))))
    out.append(("instrument", None if f is None or b is None else all(all(r["arms"][a]["admissible"] == r["arms"][a]["n"] for a in RESTRICTED) for r in (f, b))))
    out.append(("claim", None if f is None else f["free"]["replies_with_slip"] / f["free"]["replies"] >= SLIP_REPLY))
    out.append(("claim", None if b is None else b["free"]["replies_with_slip"] / b["free"]["replies"] >= SLIP_REPLY))
    out.append(("claim", None if f is None else f["free"]["slip_share"] >= SLIP_TOKEN_LINE))
    if f is None: return out + [("claim", None)] * 7
    out.append(("claim", _acc(f, "masked") >= HY_MASKED))
    out.append(("claim", _acc(f, "hyper_nn") >= _acc(f, "masked") - HY_SLACK))
    out.append(("claim", _acc(f, "hyper_nn") - _acc(f, "euclid_tuned") >= HY_GAIN_EU))
    ms, hs = _acc(f, "masked", "correct_on_slip", "n_slip"), _acc(f, "hyper_nn", "correct_on_slip", "n_slip"); out.append(("claim", f["arms"]["masked"]["n_slip"] > 0 and ms >= hs))
    out.append(("claim", _acc(f, "lin_hyp") - _acc(f, "lin_eu") >= HY_GAIN_HEAD))
    out.append(("claim", _acc(f, "lin_hyp") - _acc(f, "lin_dot") >= HY_GAIN_DOT))
    out.append(("claim", _acc(f, "lin_dot") >= _acc(f, "masked") - HY_SLACK))
    return out


# ---------------------------------------------------------------- printing and the command line
def _pc(x: float) -> str: return f"{100 * x:.1f}%"


def print_hypsnap(results: Dict[str, dict]) -> None:
    for cond, r in results.items():
        fr = r["free"]; print(f"\n=== condition {cond}: {r['positions']} positions in {r['examples']} replies, {r['K']} tokens allowed somewhere ===")
        print(f"the model's own pick is refused by the grammar at {fr['slip_positions']} of {fr['positions']} positions ({_pc(fr['slip_share'])}), in {fr['replies_with_slip']} of {fr['replies']} replies ({_pc(fr['replies_with_slip'] / fr['replies'])}); it is the canonical token at {_pc(fr['gold_is_free_top'])}")
        print(f"{'arm':14s} {'canonical':>10s} {'allowed':>9s} {'on slips':>10s} {'off slips':>10s}   (test half: {r['arms']['free']['n']} positions, {r['arms']['free']['n_slip']} of them slips)")
        for a in ARMS:
            t = r["arms"][a]; print(f"{a:14s} {_pc(t['correct'] / t['n']):>10s} {_pc(t['admissible'] / t['n']):>9s} {_pc(t['correct_on_slip'] / t['n_slip']) if t['n_slip'] else '    -':>10s} {_pc(t['correct_off_slip'] / (t['n'] - t['n_slip'])):>10s}")
        print("calibration:", r["calibration"]["alpha"]["chosen"], "(euclid_tuned scale),", r["calibration"]["curvature"]["chosen"], "(curvature)")


def print_verdict_hypsnap(checks) -> None:
    print("--- reading rules ---")
    for i, (kind, v) in enumerate(checks, 1): print(f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}" + ("   (headline)" if i in HY_HEADLINE else ""))
    head = [checks[i - 1][1] for i in HY_HEADLINE]; print("HEADLINE (the model's own pick breaks the syntax in bare, and the hyperbolic snap loses nothing against masking):", "ALL YES" if all(head) else "NOT all yes")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="stage 2 of the hyperbolic snap test (docs/HYPERBOLIC_SNAP_TEST.md): reads the arrays that stage 1 saved")
    ap.add_argument("--full", required=True, help="the .npz of condition full"); ap.add_argument("--bare", required=True, help="the .npz of condition bare"); ap.add_argument("--out", default="axi_hypsnap.json")
    ap.add_argument("--tasks", type=int, default=N_TASKS); a = ap.parse_args(argv); t0 = time.time(); res = {}
    for cond, path in (("full", a.full), ("bare", a.bare)): res[cond] = analyze(load_data(path), say=lambda *x: print("  ", *x, flush=True))
    out = dict(results=res, tasks=a.tasks, seconds=round(time.time() - t0, 1))
    with open(a.out, "w") as f: json.dump(out, f, indent=1, sort_keys=True)
    print_hypsnap(res); print_verdict_hypsnap(verdict_hypsnap(res, a.tasks)); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()

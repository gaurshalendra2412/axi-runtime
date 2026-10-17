"""Engineering support for docs/HYPERBOLIC_SNAP_TEST.md (concept-map item 104): what does it cost, and what does it buy, to turn the last hidden vector of a language model into a token by the nearest allowed token in a hyperbolic ball?
No torch here: stage 1 is exercised with a numpy stand-in for the model (the same shapes and the same calls), stage 2 on arrays made from it. This file tests the INSTRUMENT, not the finding:
  1. the geometry: the exponential map stays inside the ball, the distance to the origin is twice the length of the vector, symmetry, the triangle inequality, against a direct formula;
  2. the three trained heads: the gradients of the loss against finite differences (A, Q and the temperature), training lowers the loss and raises the accuracy, the hyperbolic head cannot leave the ball;
  3. stage 1 (`collect`): the positions of every reply, the grammar states, the allowed sets and their logits, the model's own pick, the canonical token, the stored rows of the output head, a refused canonical token, `save_data`/`load_data`;
  4. stage 2 (`analyze`): every fixed arm against a plain loop, the calibration choosing the best number of its grid, every arm but `free` always allowed, the tallies adding up, strong and absent signals;
  5. the thirteen reading rules, each flipping where it should, and the command line;
  6. the page is locked (hash of the text above the Result), carries the thirteen lines with the probabilities, the settings and the thresholds the code uses."""
import contextlib, hashlib, io, json, math, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.hypsnap as Hs
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker, build_vocab
from axi.engine.gpu_mask import TokenDFATable

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "HYPERBOLIC_SNAP_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"


# ---------------------------------------------------------------- the stand-in model
class FakeWorld:
    """A vocabulary of the repository's synthetic kind, the real grammar table, and a 'model' whose hidden vector at a position is `strength` times the output-head row of the canonical token plus noise."""
    def __init__(self, seed=0, D=24, strength=3.0, noise=1.0):
        self.vocab, self.eos = build_vocab(extra_random=120, seed=1); self.V = len(self.vocab); dfa = CICOGrammarDFA(); self.table = TokenDFATable(dfa, TokenMasker(dfa, self.vocab, self.eos)); self.table.allowed[self.table.start, self.eos] = False
        self.rng = np.random.default_rng(seed); self.D, self.strength, self.noise = D, strength, noise; self.Uf = self.rng.standard_normal((self.V, D)).astype(np.float32); self.calls = 0
        self.by_len = sorted([i for i in range(self.V) if i != self.eos and self.vocab[i]], key=lambda i: -len(self.vocab[i]))

    def encode(self, text):
        b = text.encode(); out = []; i = 0
        while i < len(b):
            t = next(t for t in self.by_len if b.startswith(self.vocab[t], i)); out.append(t); i += len(self.vocab[t])
        return out

    def prompt_ids(self, messages): return [int(c) for c in "".join(m["content"] for m in messages).encode()][:50]

    def runner(self, p_ids, reply_in):
        self.calls += 1; gold = list(reply_in) + [self.eos]; h_post = np.stack([self.strength * self.Uf[g] + self.noise * self.rng.standard_normal(self.D) for g in gold]).astype(np.float32)
        h_pre = (2.5 * h_post + 0.1 * self.rng.standard_normal(h_post.shape)).astype(np.float32); return dict(h_pre=h_pre, h_post=h_post, logits=lambda j: self.Uf @ h_post[j])

    def unembed(self, ids): return self.Uf[ids]


_cache = {}


def collected(condition="full", n=14, **kw):
    key = (condition, n, tuple(sorted(kw.items())))
    if key not in _cache:
        w = FakeWorld(**kw); tasks = Hs.make_tasks(n, 5); d = Hs.collect(w.runner, w.prompt_ids, w.encode, w.table, w.eos, tasks, condition); Hs.attach_unembedding([d], w.unembed); _cache[key] = (w, tasks, d)
    return _cache[key]


# ---------------------------------------------------------------- 1 the geometry
def test_the_exponential_map_the_distance_and_the_direct_formula():
    rng = np.random.default_rng(0); x = rng.standard_normal((6, 5)) * 0.4
    for c in (0.25, 1.0, 9.0):
        u = Hs.expmap0(x, c); assert (np.linalg.norm(u, axis=1) < 1 / math.sqrt(c)).all()
        d0 = Hs.poincare_dist(np.zeros((1, 5)), u, c)[0]; assert np.allclose(d0, 2 * np.linalg.norm(x, axis=1), atol=1e-6)                          # the distance from the origin is twice the length of the vector, whatever the curvature
        D = Hs.poincare_dist(u, u, c); assert np.allclose(D, D.T, atol=1e-9) and np.allclose(np.diag(D), 0, atol=1e-6)
        for i in range(6):                                                                                                                            # the formula written out for one pair
            a, b = u[i], u[(i + 2) % 6]; want = math.acosh(1 + 2 * c * float(((a - b) ** 2).sum()) / ((1 - c * float((a * a).sum())) * (1 - c * float((b * b).sum())))) / math.sqrt(c)
            assert abs(Hs.poincare_dist(a[None], b[None], c)[0, 0] - want) < 1e-6
        for i in range(6):
            for j in range(6):
                for k in range(6): assert D[i, k] <= D[i, j] + D[j, k] + 1e-6                                                                    # a distance
    big = Hs.expmap0(np.array([[1e4, 0, 0]]), 1.0); assert np.linalg.norm(big) < 1 and np.isfinite(Hs.poincare_dist(big, big, 1.0)).all()              # huge vectors stay inside the ball and the distance stays finite
    assert np.allclose(Hs.expmap0(np.zeros((2, 3)), 1.0), 0)
    # a small curvature is the flat distance, up to a factor: d ~ 2 |x - y| for small vectors in the flat limit
    a, b = rng.standard_normal((1, 4)) * 1e-3, rng.standard_normal((1, 4)) * 1e-3; assert abs(Hs.poincare_dist(a, b, 1.0)[0, 0] / (2 * np.linalg.norm(a - b)) - 1) < 1e-3


def test_the_gradient_of_the_exponential_map_matches_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = rng.standard_normal(5) * rng.choice([0.01, 0.5, 2.0]); du = rng.standard_normal(5); g = Hs.pi_backward(z[None], du[None])[0]; e = 1e-6
        num = np.array([(float(Hs.pi_forward((z + e * np.eye(5)[i])[None])[0] @ du) - float(Hs.pi_forward((z - e * np.eye(5)[i])[None])[0] @ du)) / (2 * e) for i in range(5)])
        assert np.allclose(g, num, atol=1e-6), (z, g, num)
    assert np.isfinite(Hs.pi_backward(np.zeros((1, 3)), np.ones((1, 3)))).all()


# ---------------------------------------------------------------- 2 the heads
def test_the_gradients_of_the_three_heads_match_finite_differences():
    rng = np.random.default_rng(0)
    for kind in ("dot", "eu", "hyp"):
        D_, K_, B_, n_ = 7, 6, 5, 4; h = Hs.Head(kind, D_, K_, 3, n=n_); h.logtau = 0.3; H = rng.standard_normal((B_, D_)); M = rng.random((B_, K_)) < 0.7
        gold = rng.integers(0, K_, B_)
        for b in range(B_): M[b, gold[b]] = True
        loss, dA, dQ, dl = h.grads(H, M, gold); worst = 0.0
        for name, g in (("A", dA), ("Q", dQ)):
            arr = getattr(h, name)
            for _ in range(12):
                i = tuple(rng.integers(0, s) for s in arr.shape); old = arr[i]; e = 1e-6
                arr[i] = old + e; lp = h.grads(H, M, gold)[0]; arr[i] = old - e; lm = h.grads(H, M, gold)[0]; arr[i] = old; num = (lp - lm) / (2 * e); worst = max(worst, abs(num - g[i]) / (abs(num) + abs(g[i]) + 1e-9))
        assert worst < 1e-5, (kind, worst)
        if kind != "dot":
            old = h.logtau; e = 1e-6; h.logtau = old + e; lp = h.grads(H, M, gold)[0]; h.logtau = old - e; lm = h.grads(H, M, gold)[0]; h.logtau = old; num = (lp - lm) / (2 * e); assert abs(num - dl) / (abs(num) + abs(dl) + 1e-9) < 1e-5
        else: assert dl == 0.0
        S = h.scores(H); assert S.shape == (B_, K_)
        # the loss is the cross-entropy over the allowed columns only: a column that no row allows changes nothing, and its prototype gets no gradient
        M2 = M.copy(); M2[:, K_ - 1] = False; g2 = np.where(gold == K_ - 1, 0, gold)
        for b in range(B_): M2[b, g2[b]] = True
        loss0, _, dQ0, _ = h.grads(H, M2, g2); Q0 = h.Q.copy(); h.Q[K_ - 1] += 5.0; loss1 = h.grads(H, M2, g2)[0]; h.Q = Q0
        assert abs(loss1 - loss0) < 1e-12 and np.allclose(dQ0[K_ - 1], 0)


def test_training_a_head_lowers_its_loss_and_raises_its_accuracy_and_the_hyperbolic_head_stays_in_the_ball():
    w, tasks, d = collected(strength=3.0, noise=1.5); cal, test = Hs.split_rows(d); mh, mu = Hs.norms_for(d, cal)
    for kind in ("dot", "eu", "hyp"):
        head, curve = Hs.train_head(d, kind, cal, mh, mu, dict(epochs=8, batch=128, lr=0.02, seed=3)); assert curve[-1] < curve[0] * 0.9, (kind, curve)
        pred = Hs.decide_head(d, head, test, mh); assert (pred == d["gold"][test]).mean() > 0.5 and Hs.is_allowed(d, test, pred).all()
        if kind == "hyp":
            u = Hs.pi_forward(d["h_post"][test].astype(np.float64) / mh @ head.A.T); assert (np.linalg.norm(u, axis=1) < 1).all() and (np.linalg.norm(Hs.pi_forward(head.Q), axis=1) < 1).all()
    h0 = Hs.Head("dot", d["h_post"].shape[1], len(d["U_ids"]), 3); h0.init_prototypes(d["U"].astype(np.float64) / mu)
    assert h0.Q.shape == (len(d["U_ids"]), Hs.HEAD_DIM) and np.allclose(h0.Q, (d["U"].astype(np.float64) / mu) @ h0.A.T)


# ---------------------------------------------------------------- 3 stage 1
def test_collect_walks_every_reply_and_stores_what_stage_2_needs():
    w, tasks, d = collected(); table = w.table; n_pos = sum(len(w.encode(t.ideal)) + 1 for t in tasks)
    assert len(d["ex"]) == n_pos == len(d["gold"]) == len(d["h_post"]) == len(d["h_pre"]) == len(d["state"]) == len(d["free_top"]) and d["broken"] == 0 and d["gold_in"].all() and d["condition"] == "full"
    assert d["h_post"].dtype == np.float16 and d["h_pre"].dtype == np.float16 and d["a_ptr"].shape == (n_pos + 1,) and d["a_ptr"][0] == 0 and (np.diff(d["a_ptr"]) > 0).all() and d["a_ptr"][-1] == len(d["a_ids"]) == len(d["a_logit"])
    pos = 0
    for i, t in enumerate(tasks):
        reply = w.encode(t.ideal) + [w.eos]; state = table.start
        for j, g in enumerate(reply):
            assert d["ex"][pos] == i and d["j"][pos] == j and d["gold"][pos] == g and d["state"][pos] == state
            ids = d["a_ids"][d["a_ptr"][pos]:d["a_ptr"][pos + 1]]; assert (ids == np.flatnonzero(table.allowed[state])).all() and g in ids
            pos += 1
            if g != w.eos: state = int(table.next[state, g])
        assert table.accepting[state]                                                                                                           # the reply ends in a state where the end token is allowed
    assert pos == n_pos and (d["ex"][-1] == len(tasks) - 1)
    # the logits stored are those of the allowed tokens, the free pick is the argmax over the whole vocabulary, the gold logit is the gold token's
    w2 = FakeWorld(); t0 = tasks[0]; reply = w2.encode(t0.ideal); out = w2.runner([1], reply)
    d2 = Hs.collect(lambda p, r: out, w2.prompt_ids, w2.encode, w2.table, w2.eos, [t0], "full"); lg = out["logits"](0); ids0 = d2["a_ids"][d2["a_ptr"][0]:d2["a_ptr"][1]]
    assert np.allclose(d2["a_logit"][:len(ids0)], lg[ids0]) and d2["free_top"][0] == lg.argmax() and abs(d2["free_logit"][0] - lg.max()) < 1e-6 and abs(d2["gold_logit"][0] - lg[reply[0]]) < 1e-6
    # the output-head rows are stored for the union of the allowed tokens only
    assert (d["U_ids"] == np.unique(d["a_ids"])).all() and np.allclose(d["U"].astype(np.float32), w.Uf[d["U_ids"]].astype(np.float16).astype(np.float32))
    # the end token is the last gold of every reply, and it is allowed only in a state that accepts
    last = [i for i in range(n_pos - 1) if d["ex"][i] != d["ex"][i + 1]] + [n_pos - 1]; assert (d["gold"][last] == w.eos).all()
    # a canonical token the grammar refuses is counted, flagged, and ends that reply
    w3 = FakeWorld(); bad = lambda text: [w3.encode(text)[0]] + [ord("Z")] + w3.encode(text)[1:]
    d3 = Hs.collect(w3.runner, w3.prompt_ids, bad, w3.table, w3.eos, tasks[:2], "bare"); assert d3["broken"] == 2 and not d3["gold_in"].all() and d3["condition"] == "bare" and int((~d3["gold_in"]).sum()) == 2
    assert (d3["j"][~d3["gold_in"]] == 1).all()
    cal, test = Hs.split_rows(d3); assert d3["gold_in"][cal].all() and d3["gold_in"][test].all()
    # the messages of the two conditions
    m_full, m_bare = Hs.messages_for(tasks[0], "full"), Hs.messages_for(tasks[0], "bare"); assert [m["role"] for m in m_full] == ["system", "user"] and [m["role"] for m in m_bare] == ["user"] and m_full[1]["content"] == m_bare[0]["content"] and "ADD[r,c:value]" in m_full[0]["content"]


def test_check_walkable_finds_the_first_refused_token_of_a_reply_and_nothing_in_a_good_one():
    w = FakeWorld(); tasks = Hs.make_tasks(12, 5); assert Hs.check_walkable(w.table, w.encode, w.eos, tasks) == []
    bad = lambda text: w.encode(text)[:2] + [ord("Z")] + w.encode(text)[2:]; found = Hs.check_walkable(w.table, bad, w.eos, tasks[:3]); assert found == [(0, 2, ord("Z")), (1, 2, ord("Z")), (2, 2, ord("Z"))]
    early = lambda text: [w.eos] + w.encode(text); assert Hs.check_walkable(w.table, early, w.eos, tasks[:2]) == [(0, 0, w.eos), (1, 0, w.eos)]                                  # the end token is refused before the first item
    noend = lambda text: w.encode(text)[:3]; assert [b[0] for b in Hs.check_walkable(w.table, noend, w.eos, tasks[:2])] == [0, 1]                                                  # a reply that stops inside an item does not allow the end token


def test_norm_identity_error_is_zero_for_the_right_tensor_and_large_for_a_wrong_one():
    rng = np.random.default_rng(3); pre = rng.standard_normal((30, 8)).astype(np.float16); g = rng.standard_normal(8).astype(np.float32)
    def rms(x): return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * g
    d = dict(h_pre=pre, h_post=rms(pre.astype(np.float32)).astype(np.float16)); assert Hs.norm_identity_error(d, rms) < 1e-3
    d2 = dict(h_pre=pre, h_post=d["h_post"][::-1].copy()); assert Hs.norm_identity_error(d2, rms) > 0.1
    d3 = dict(h_pre=pre, h_post=d["h_post"]); assert Hs.norm_identity_error(d3, lambda x: rms(x)[:, ::-1]) > 0.1 and Hs.norm_identity_error(d3, rms, rows=5) < 1e-3


def test_save_and_load_round_trip():
    w, tasks, d = collected()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.npz"); Hs.save_data(p, d); e = Hs.load_data(p)
        assert set(e) == set(d) and e["condition"] == "full" and e["broken"] == 0 and all((e[k] == d[k]).all() for k in d if isinstance(d[k], np.ndarray)) and e["h_post"].dtype == np.float16


# ---------------------------------------------------------------- 4 stage 2
def test_every_fixed_arm_equals_a_plain_loop_and_the_calibration_picks_the_best_number():
    w, tasks, d = collected(strength=1.0, noise=2.0); cal, test = Hs.split_rows(d); mh, mu = Hs.norms_for(d, cal); U = d["U"].astype(np.float64) / mu; sub = test[:40]
    assert abs(mh - np.median(np.linalg.norm(d["h_post"][cal].astype(np.float32), axis=1))) < 1e-6 and abs(mu - np.median(np.linalg.norm(d["U"].astype(np.float32), axis=1))) < 1e-6
    def loop(arm, param):
        out = []
        for i in sub:
            ids = d["a_ids"][d["a_ptr"][i]:d["a_ptr"][i + 1]]; cols = np.searchsorted(d["U_ids"], ids); h = d["h_post"][i].astype(np.float64) / mh; best, bs = None, -np.inf
            for t, k in zip(ids, cols):
                u = U[k]
                if arm == "euclid_nn": s = -float(((h - u) ** 2).sum())
                elif arm == "euclid_tuned": s = -float(((h * param - u) ** 2).sum())
                elif arm == "cosine_nn": s = float(h @ u / (np.linalg.norm(h) * np.linalg.norm(u)))
                else: s = -float(Hs.poincare_dist(Hs.expmap0(h[None], param), Hs.expmap0(u[None], param), param)[0, 0])
                if s > bs: bs, best = s, int(t)
            out.append(best)
        return np.array(out)
    for arm, param in (("euclid_nn", None), ("euclid_tuned", 2.0), ("cosine_nn", None), ("hyper_nn", 1.0), ("hyper_nn", 0.1)):
        assert (Hs.decide_fixed(d, sub, arm, param, mh, mu, chunk=7) == loop(arm, param)).all(), (arm, param)
    m = Hs.decide_masked(d, sub)
    for r, i in enumerate(sub): a, b = d["a_ptr"][i], d["a_ptr"][i + 1]; assert m[r] == d["a_ids"][a:b][d["a_logit"][a:b].argmax()]
    best, acc = Hs.fit_scalar(d, "hyper_nn", cal, list(Hs.GRID_C), mh, mu); assert acc[best] == max(acc.values()) and set(acc) == set(Hs.GRID_C) and best == min([g for g in Hs.GRID_C if acc[g] == max(acc.values())], key=Hs.GRID_C.index)
    best, acc = Hs.fit_scalar(d, "euclid_tuned", cal, list(Hs.GRID_ALPHA), mh, mu); assert acc[best] == max(acc.values())
    # ties go to the first number in the list
    best, acc = Hs.fit_scalar(d, "euclid_tuned", cal[:1], [1.0, 1.0001], mh, mu); assert best == 1.0
    assert Hs.is_allowed(d, sub, m).all() and not Hs.is_allowed(d, np.array([0]), np.array([w.eos])).all()                                  # the end token is not allowed at the first position of a reply


def test_analyze_tallies_add_up_and_every_arm_but_free_is_always_allowed():
    w, tasks, d = collected(strength=2.0, noise=2.0); r = Hs.analyze(d, head_cfg=dict(epochs=3)); cal, test = Hs.split_rows(d)
    assert r["positions"] == len(d["ex"]) and r["examples"] == len(tasks) and r["K"] == len(d["U_ids"]) and r["gold_in_allowed"] and r["broken"] == 0 and list(r["arms"]) == list(Hs.ARMS) and set(r["heads"]) == {"lin_dot", "lin_eu", "lin_hyp"}
    for a in Hs.RESTRICTED: assert r["arms"][a]["admissible"] == r["arms"][a]["n"] == len(test), a
    for a, t in r["arms"].items(): assert t["correct_on_slip"] + t["correct_off_slip"] == t["correct"] and 0 <= t["n_slip"] <= t["n"] and t["n"] == len(test)
    free = d["free_top"]; slip = ~Hs.is_allowed(d, np.arange(len(free)), free); fr = r["free"]
    assert fr["slip_positions"] == int(slip.sum()) and fr["positions"] == len(free) and abs(fr["slip_share"] - slip.mean()) < 1e-12 and fr["replies_with_slip"] == len(set(d["ex"][slip].tolist())) and fr["replies"] == len(tasks) and abs(fr["gold_is_free_top"] - (free == d["gold"]).mean()) < 1e-12
    assert r["arms"]["free"]["admissible"] == int(Hs.is_allowed(d, test, free[test]).sum()) < r["arms"]["free"]["n"] or fr["slip_positions"] == 0
    assert r["arms"]["free"]["correct"] == int((free[test] == d["gold"][test]).sum()) and r["calibration"]["curvature"]["chosen"] in Hs.GRID_C and r["calibration"]["alpha"]["chosen"] in Hs.GRID_ALPHA
    assert json.loads(json.dumps(r)) == r                                                                                                       # the result is plain data


def test_a_strong_signal_is_found_by_every_arm_and_no_signal_is_chance():
    w, tasks, d = collected(strength=6.0, noise=0.5); r = Hs.analyze(d, head_cfg=dict(epochs=6, lr=0.02)); acc = lambda a: r["arms"][a]["correct"] / r["arms"][a]["n"]
    assert acc("free") > 0.95 and acc("masked") > 0.95 and r["free"]["slip_share"] < 0.02 and acc("cosine_nn") > 0.8 and acc("hyper_nn") > 0.7 and acc("lin_dot") > 0.8, {a: acc(a) for a in Hs.ARMS}
    w, tasks, d = collected(strength=0.0, noise=1.0); r = Hs.analyze(d, head_cfg=dict(epochs=3)); acc = lambda a: r["arms"][a]["correct"] / r["arms"][a]["n"]
    sizes = np.diff(d["a_ptr"]); chance = float(np.mean(1.0 / sizes[Hs.split_rows(d)[1]]))
    assert acc("masked") < 0.5 and acc("euclid_nn") < 0.5 and abs(acc("cosine_nn") - chance) < 0.3 and r["free"]["slip_share"] > 0.5                  # the output vocabulary is mostly tokens the grammar refuses, so a pick over everything is mostly refused
    # the grid: the curvature chosen is one of the list
    assert r["calibration"]["curvature"]["chosen"] in Hs.GRID_C


# ---------------------------------------------------------------- 5 the reading rules
def synth(**over):
    """A hand-made pair of results; with no changes every claim reads as the comment says."""
    def arm(n=1000, c=900, adm=None, sl=100, cs=50): return dict(n=n, correct=c, admissible=n if adm is None else adm, n_slip=sl, correct_on_slip=cs, correct_off_slip=c - cs)
    def cond(name):
        arms = {"free": arm(c=880, adm=900), "masked": arm(c=970, cs=90), "euclid_nn": arm(c=500, cs=20), "euclid_tuned": arm(c=700, cs=40), "cosine_nn": arm(c=800, cs=60), "hyper_nn": arm(c=800, cs=50), "lin_dot": arm(c=960, cs=80), "lin_eu": arm(c=950, cs=70), "lin_hyp": arm(c=955, cs=75)}
        return dict(condition=name, positions=2000, examples=400, K=300, broken=0, gold_in_allowed=True, arms=arms, calibration={}, heads={}, free=dict(slip_positions=120, positions=2000, slip_share=0.06, replies=400, replies_with_slip=60, gold_is_free_top=0.9))
    res = {"full": cond("full"), "bare": cond("bare")}
    for path, val in over.items():
        node = res
        keys = path.split("|")
        for k in keys[:-1]: node = node[k]
        node[keys[-1]] = val
    return res


def test_every_reading_rule_reads_as_it_should_and_flips_where_it_should():
    v = lambda **kw: [x for _, x in Hs.verdict_hypsnap(synth(**kw))]; base = Hs.verdict_hypsnap(synth()); assert len(base) == 13 and [k for k, _ in base] == [k for k, _ in Hs.HY_LINES]
    assert [x for _, x in base] == [True, True, True, True, True, True, True, False, True, True, False, False, True], base
    assert v(**{"full|examples": 399})[0] is False and v(**{"bare|broken": 1})[0] is False and v(**{"full|K": 0})[0] is False
    assert v(**{"bare|gold_in_allowed": False})[1] is False and v(**{"full|arms|hyper_nn|admissible": 999})[2] is False and v(**{"bare|arms|lin_hyp|admissible": 999})[2] is False and v(**{"full|arms|free|admissible": 1})[2] is True
    assert v(**{"full|free|replies_with_slip": 19})[3] is False and v(**{"full|free|replies_with_slip": 20})[3] is True and v(**{"bare|free|replies_with_slip": 19})[4] is False and v(**{"bare|free|replies_with_slip": 20})[4] is True
    assert v(**{"full|free|slip_share": 0.0499})[5] is False and v(**{"full|free|slip_share": 0.05})[5] is True
    assert v(**{"full|arms|masked|correct": 949})[6] is False and v(**{"full|arms|masked|correct": 950})[6] is True
    assert v(**{"full|arms|hyper_nn|correct": 949})[7] is False and v(**{"full|arms|hyper_nn|correct": 951})[7] is True and v(**{"full|arms|hyper_nn|correct": 951, "full|arms|masked|correct": 975})[7] is False
    assert v(**{"full|arms|hyper_nn|correct": 722})[8] is True and v(**{"full|arms|hyper_nn|correct": 719})[8] is False
    assert v(**{"full|arms|hyper_nn|correct_on_slip": 91})[9] is False and v(**{"full|arms|hyper_nn|correct_on_slip": 90})[9] is True and v(**{"full|arms|masked|n_slip": 0})[9] is False
    assert v(**{"full|arms|lin_hyp|correct": 960})[10] is True and v(**{"full|arms|lin_hyp|correct": 959})[10] is False
    assert v(**{"full|arms|lin_hyp|correct": 970})[11] is True and v(**{"full|arms|lin_hyp|correct": 969})[11] is False
    assert v(**{"full|arms|lin_dot|correct": 951})[12] is True and v(**{"full|arms|lin_dot|correct": 949})[12] is False
    p = Hs.verdict_hypsnap({"full": synth()["full"]}); assert p[0][1] is False and p[1][1] is None and p[2][1] is None and p[3][1] is True and p[4][1] is None and p[6][1] is True
    assert Hs.verdict_hypsnap({})[3][1] is None and all(x is None for _, x in Hs.verdict_hypsnap({})[3:]) and Hs.HY_HEADLINE == (5, 8)


def test_the_printout_and_the_command_line():
    w, tasks, d = collected(strength=3.0, noise=1.0); wb, tb, db = collected("bare", strength=3.0, noise=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        pf, pb, out = os.path.join(tmp, "f.npz"), os.path.join(tmp, "b.npz"), os.path.join(tmp, "o.json"); Hs.save_data(pf, d); Hs.save_data(pb, db); buf = io.StringIO()
        real = Hs.HEAD.copy(); Hs.HEAD.update(epochs=2)
        try:
            with contextlib.redirect_stdout(buf): Hs.main(["--full", pf, "--bare", pb, "--out", out, "--tasks", str(len(tasks))])
        finally: Hs.HEAD.clear(); Hs.HEAD.update(real)
        res = json.load(open(out)); text = buf.getvalue(); assert set(res["results"]) == {"full", "bare"} and res["tasks"] == len(tasks) and "condition full" in text and "condition bare" in text and "HEADLINE" in text and "(headline)" in text
        assert text.count("[claim") == 10 and text.count("[guard") == 1 and text.count("[instrument") == 2 and all(a in text for a in Hs.ARMS)


# ---------------------------------------------------------------- 6 the page
PAGE_SHA256 = "5201de5e7abfd889bea7babecd4337b73cc5628d645b77ca607873d140cd2d88"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_thirteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(Hs.HY_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [97, 99, 99, 35, 90, 25, 75, 10, 25, 85, 35, 25, 50], probs
    assert Hs.HY_HEADLINE == (5, 8) and "**Headline** (lines 5 and 8)" in head and "my single number is **8 %**" in head
    assert (Hs.N_TASKS, Hs.TASK_SEED, Hs.GRID_ALPHA, Hs.GRID_C, Hs.HEAD_DIM, Hs.HEAD, Hs.MODEL, Hs.CONDITIONS) == (400, 31, (0.25, 0.5, 1.0, 2.0, 4.0), (0.01, 0.1, 1.0, 10.0, 100.0), 32, dict(epochs=25, batch=256, lr=0.01, seed=5), "Qwen/Qwen2.5-1.5B-Instruct", ("full", "bare"))
    assert (Hs.SLIP_REPLY, Hs.SLIP_TOKEN_LINE, Hs.HY_MASKED, Hs.HY_SLACK, Hs.HY_GAIN_EU, Hs.HY_GAIN_HEAD, Hs.HY_GAIN_DOT) == (0.05, 0.05, 0.95, 0.02, 0.02, 0.01, 0.01) and Hs.ARMS == ("free", "masked", "euclid_nn", "euclid_tuned", "cosine_nn", "hyper_nn", "lin_dot", "lin_eu", "lin_hyp")
    for needle in ("Concept-map item 104", "`python/axi/experiments/hypsnap.py`", "Validation of the Continuous-to-Discrete Functor", "I attribute none of it to him", "every choice below is mine, not Rajnish's", "0.00% syntax failure rate", "true by construction",
                   "Qwen/Qwen2.5-1.5B-Instruct", "`hypsnap.verdict_hypsnap`", "axi_hyperbolic_snap.ipynb", "Nickel and Kiela", "Ganea", "I did not run the model"):
        assert needle in head, needle


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

"""Engineering support for docs/TOY_MODEL_TEST.md (concept-map item 103): a small transformer trained from scratch on graphs of 10 services and asked about graphs of up to 100, with four ways of telling it where a service is.
CPU only, numpy only, no language model. This file tests the INSTRUMENT, not the finding:
  1. the task: graphs have the size, the grid, the edges, the payloads and the labels the page says; same seed, same graph; batches are padded and masked as the page says;
  2. the position codes: the SU(2) turn equals the matrix exponential of the page's generator, every turn is orthogonal and its inverse is its transpose, rope2d depends on the offset only, su2 does not (stated on the page), rope1d turns by the index;
  3. the model: the gradients of every parameter match finite differences for every arm (with the penalty), the model is causal, the penalty blocks are the half of the weights that the page says, the shares and the sensitivity measure what they should;
  4. the run: training is seeded and steps as often as the page says, `evaluate` counts exact tokens and F1 as defined, `merge_results` puts single-seed files together and refuses files that disagree, the quick run and the command line;
  5. the fourteen reading rules each flipping where they should, with thresholds at their edges, and `None` when an arm is missing;
  6. the page is locked (hash of the text above the Result), carries the fourteen lines with the probabilities, the settings and the thresholds the code uses."""
import contextlib, copy, hashlib, io, json, math, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.toymodel as M

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "TOY_MODEL_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"


def raises(f, exc=AssertionError):
    try: f()
    except exc: return True
    return False


# ---------------------------------------------------------------- 1 the task
def _cells(g): return [t[1] for t in g["tokens"] if t[0] == "S"]


def test_graphs_have_the_size_the_grid_the_edges_the_payloads_and_the_labels_the_page_says():
    rng = random.Random(7); qin = 0; total = 0
    for n, gside in ((10, 10), (20, 20), (40, 40), (10, 100), (100, 100)):
        for shift in (False, True):
            for _ in range(6):
                g = M.make_graph(rng, n, gside, shift); toks = g["tokens"]; cells = _cells(g); q = g["query"]
                assert toks[0][0] == "Q" and toks[0][1] == q and toks[0][2] == M.OFF_NONE and toks[0][3] == 0 and toks[0][4] == 0
                assert g["n"] == n == len(cells) == len(set(cells)) and all(0 <= x < gside and 0 <= y < gside for x, y in cells)
                es = [t for t in toks if t[0] == "E"]; assert g["edges"] == len(es) and len(toks) == 1 + n + len(es) and len(es) >= n - 1
                cset = set(cells); adj = {c: set() for c in cells}
                for _k, a, o, p, y in es:
                    dx, dy = M.OFFS[o]; b = (a[0] + dx, a[1] + dy); assert a in cset and b in cset and max(abs(dx), abs(dy)) <= 2 and (dx, dy) != (0, 0)
                    assert y == int(a == q or b == q); adj[a].add(b); adj[b].add(a)
                for k, c, o, p, y in toks[1:]:
                    assert (k == "S") == (o == M.OFF_NONE); assert (1 <= p <= 9) if not shift else (10 <= p <= 19)
                    if k == "S": assert y == int(c == q)
                seen = {cells[0]}; stack = [cells[0]]                                       # the services form one connected graph
                while stack:
                    for nb in adj[stack.pop()]:
                        if nb not in seen: seen.add(nb); stack.append(nb)
                assert seen == cset
                qin += int(q in cset); total += 1
    assert qin / total > 0.7                                                                  # the query is a service in about 85 % of graphs and a free cell next to the blob otherwise
    rng = random.Random(1); inq = [(lambda g: g["query"] in set(_cells(g)))(M.make_graph(rng, 10, 10)) for _ in range(600)]; assert 0.80 < np.mean(inq) < 0.90
    a = M.make_graph(random.Random("x"), 10, 10); b = M.make_graph(random.Random("x"), 10, 10); c = M.make_graph(random.Random("y"), 10, 10); assert a == b and a != c


def test_batches_are_padded_and_the_query_and_the_padding_are_not_scored():
    rng = random.Random(3); gs = [M.make_graph(rng, n, 10) for n in (4, 7, 5)]; b = M.batch_of(gs); T = max(len(g["tokens"]) for g in gs)
    assert b["kind"].shape == (3, T) == b["label"].shape and list(b["length"]) == [len(g["tokens"]) for g in gs]
    for i, g in enumerate(gs):
        L = len(g["tokens"]); assert not b["scored"][i, 0] and b["scored"][i, 1:L].all() and not b["scored"][i, L:].any() and (b["label"][i, L:] == 0).all() and (b["kind"][i, 0] == M.K_Q)
        assert [int(b["label"][i, t]) for t in range(L)] == [t_[4] for t_ in g["tokens"]] and [int(b["x"][i, t]) for t in range(L)] == [t_[1][0] for t_ in g["tokens"]]
        assert all(b["kind"][i, t] == (M.K_S if g["tokens"][t][0] == "S" else M.K_E) for t in range(1, L))


def test_eval_graphs_are_fixed_by_name_and_seed_and_the_sets_are_the_sizes_the_page_names():
    assert M.full_eval_digest() == "7170eee401870825", "the evaluation graphs of the full run changed"                                                  # the notebook prints this number
    a = M.eval_graphs("N10", 10, 10, False, 6, 0); b = M.eval_graphs("N10", 10, 10, False, 6, 0); c = M.eval_graphs("N10", 10, 10, False, 6, 1); d = M.eval_graphs("N20", 10, 10, False, 6, 0)
    assert a == b and a != c and a != d
    assert [(s[0], s[1], s[2], s[3]) for s in M.EVAL_SETS] == [("N10", 10, 10, False), ("N20", 20, 20, False), ("N40", 40, 40, False), ("N70", 70, 70, False), ("N100", 100, 100, False), ("range100_n10", 10, 100, False), ("pay_shift_N10", 10, 10, True), ("pay_shift_N100", 100, 100, True)]
    assert set(M.EVAL_GRAPHS) == {s[0] for s in M.EVAL_SETS} and M.TRAIN_N == 10 and M.GMAX == 100
    g = M.make_graph(random.Random(0), 100, 100); assert g["n"] == 100 and len(g["tokens"]) > 250                                   # ten times the tokens of a graph of 10


# ---------------------------------------------------------------- 2 the position codes
def _taylor_exp(A, terms=60):
    out = np.eye(2, dtype=complex); term = np.eye(2, dtype=complex)
    for k in range(1, terms): term = term @ A / k; out = out + term
    return out


SIG1 = np.array([[0, 1], [1, 0]], complex); SIG2 = np.array([[0, -1j], [1j, 0]], complex)


def _batch_at(cells):
    B = len(cells); return dict(x=np.array([[c[0]] for c in cells]), y=np.array([[c[1]] for c in cells]))


def test_the_su2_turn_is_the_matrix_exponential_of_the_pages_generator_and_is_unitary_with_determinant_one():
    rr = np.random.default_rng(4); cells = [(0, 0), (1, 0), (0, 1), (3, -2 + 2), (7, 9), (99, 98), (50, 0), (0, 77)]
    pp = M.pos_params("su2", _batch_at(cells), np.float64)
    for i, (x, y) in enumerate(cells):
        for b, th in enumerate(M.THETA):
            Tgen = -0.5j * (x * SIG1 + y * SIG2); U = _taylor_exp(th * Tgen, 200 if max(x, y) > 20 else 80)                        # exp(theta (x T1 + y T2)), T_a = -i sigma_a / 2
            ch, u = pp["ch"][i, 0, b], pp["u"][i, 0, b]; mine = np.array([[ch, -1j * np.conj(u)], [-1j * u, ch]])
            if max(x, y) <= 20: assert np.allclose(mine, U, atol=1e-9), (x, y, th)
            assert np.allclose(mine.conj().T @ mine, np.eye(2), atol=1e-12) and abs(np.linalg.det(mine) - 1) < 1e-12
    # the closed form at a large argument against an independent form: cos a I - i sin a (x s1 + y s2) / r
    x, y, th = 99, 98, M.THETA[0]; r = math.hypot(x, y); a = th * r / 2; ref = math.cos(a) * np.eye(2) - 1j * math.sin(a) * (x * SIG1 + y * SIG2) / r
    pp1 = M.pos_params("su2", _batch_at([(x, y)]), np.float64); mine = np.array([[pp1["ch"][0, 0, 0], -1j * np.conj(pp1["u"][0, 0, 0])], [-1j * pp1["u"][0, 0, 0], pp1["ch"][0, 0, 0]]]); assert np.allclose(mine, ref, atol=1e-12)


def test_every_turn_is_orthogonal_and_its_inverse_is_the_transpose_and_rotate_applies_the_matrices():
    rr = np.random.default_rng(8); B, H, T = 2, 3, 5
    batch = dict(x=rr.integers(0, 100, (B, T)), y=rr.integers(0, 100, (B, T))); x0 = rr.standard_normal((B, H, T, M.D_POS))
    for code in ("rope1d", "rope2d", "su2"):
        pp = M.pos_params(code, batch, np.float64); y = M.rotate(x0, pp, code)
        assert np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x0, axis=-1)) and np.allclose(M.rotate(y, pp, code, inverse=True), x0, atol=1e-12) and not np.allclose(y, x0)
        # the turn is a fixed orthogonal matrix per token: <R q, R k> = <q, k> for the same token
        k0 = rr.standard_normal(x0.shape); assert np.allclose((M.rotate(x0, pp, code) * M.rotate(k0, pp, code)).sum(-1), (x0 * k0).sum(-1))
    # su2: block b of the 16 numbers (four reals = two complex numbers) is multiplied by U_b
    pp = M.pos_params("su2", batch, np.float64); y = M.rotate(x0, pp, "su2")
    for (bb, tt) in ((0, 0), (1, 3)):
        for blk in range(4):
            v = x0[bb, 1, tt, 4 * blk:4 * blk + 4]; z = np.array([v[0] + 1j * v[1], v[2] + 1j * v[3]]); xx, yy = batch["x"][bb, tt], batch["y"][bb, tt]; th = M.THETA[blk]
            r = math.hypot(xx, yy); a = th * r / 2; U = math.cos(a) * np.eye(2) - 1j * (math.sin(a) / r if r > 0 else 0.0) * (xx * SIG1 + yy * SIG2) if r > 0 else np.eye(2); w = U @ z
            assert np.allclose(y[bb, 1, tt, 4 * blk:4 * blk + 4], [w[0].real, w[0].imag, w[1].real, w[1].imag], atol=1e-9)


def test_rope2d_depends_on_the_offset_only_su2_does_not_and_rope1d_turns_by_the_index():
    rr = np.random.default_rng(2); q = rr.standard_normal((1, 1, 1, M.D_POS)); k = rr.standard_normal((1, 1, 1, M.D_POS))
    def score(code, c1, c2):
        b1, b2 = _batch_at([c1]), _batch_at([c2]); return float((M.rotate(q, M.pos_params(code, b1, np.float64), code) * M.rotate(k, M.pos_params(code, b2, np.float64), code)).sum())
    assert abs(score("rope2d", (3, 5), (8, 2)) - score("rope2d", (13, 25), (18, 22))) < 1e-9 and abs(score("rope2d", (3, 5), (8, 2)) - score("rope2d", (3, 5), (9, 2))) > 1e-3
    assert max(abs(score("su2", (3, 5), (8, 2)) - score("su2", (13, 25), (18, 22))), abs(score("su2", (3, 5), (8, 2)) - score("su2", (33, 5), (38, 2)))) > 1e-3              # the non-abelian turn is not translation-equivariant
    pp = M.pos_params("rope1d", dict(x=np.zeros((1, 6), int), y=np.zeros((1, 6), int)), np.float64); ang = np.arctan2(pp["sin"][0], pp["cos"][0])
    for t in range(1, 6):
        for f in range(8): assert abs(math.cos(t * 10000.0 ** (-f / 8.0)) - pp["cos"][0, t, f]) < 1e-12
    assert np.allclose(pp["cos"][0, 0], 1) and np.allclose(pp["sin"][0, 0], 0)
    pp2 = M.pos_params("rope2d", dict(x=np.array([[4]]), y=np.array([[6]])), np.float64); assert np.allclose(pp2["cos"][0, 0], np.cos(np.concatenate([4 * np.array(M.THETA), 6 * np.array(M.THETA)])))


# ---------------------------------------------------------------- 3 the model
def test_the_gradients_of_every_parameter_match_finite_differences_for_every_arm_with_the_penalty():
    rng = random.Random(5); graphs = [M.make_graph(rng, 5, 7) for _ in range(3)]; b = M.batch_of(graphs)
    for arm, spec in M.ARMS.items():
        P = M.init_params(spec["table"], 3, np.float64); code = spec["code"]; r = np.random.default_rng(1)
        for k in P:
            if k.startswith(("g1", "g2", "gf")): P[k] = 1 + 0.1 * r.standard_normal(P[k].shape)
            if k.startswith(("b1", "b2", "bf", "c1", "c2", "bout")): P[k] = 0.1 * r.standard_normal(P[k].shape)
        lam = 0.05 if spec["lam"] else 0.0; pp = M.pos_params(code, b, np.float64); loss, ce, G = M.loss_and_grads(P, b, code, pp, lam, 3.0); assert math.isfinite(loss) and loss >= ce - 1e-12 and (lam > 0) == (loss > ce)
        rr = np.random.default_rng(2); worst = 0.0
        for k in P:
            flat = P[k].reshape(-1); g = G[k].reshape(-1)
            for i in rr.choice(flat.size, size=min(5, flat.size), replace=False):
                h = 1e-6; old = flat[i]; flat[i] = old + h; lp = M.loss_and_grads(P, b, code, pp, lam, 3.0)[0]; flat[i] = old - h; lm = M.loss_and_grads(P, b, code, pp, lam, 3.0)[0]; flat[i] = old
                num = (lp - lm) / (2 * h); worst = max(worst, abs(num - g[i]) / max(1e-7, abs(num) + abs(g[i])))
        assert worst < 1e-5, (arm, worst)


def test_the_model_is_causal_and_the_logit_of_a_token_does_not_depend_on_later_tokens():
    rng = random.Random(9); g1 = M.make_graph(rng, 8, 10); g2 = copy.deepcopy(g1)
    k, c, o, p, y = g2["tokens"][-1]; g2["tokens"][-1] = (k, (c[0], (c[1] + 3) % 10), o, (p % 9) + 1, y); k, c, o, p, y = g2["tokens"][-2]; g2["tokens"][-2] = (k, c, o, (p % 9) + 1, y)
    for code, table in (("rope1d", True), ("rope2d", False), ("su2", False)):
        P = M.init_params(table, 4, np.float64); l1 = M.predict(P, M.batch_of([g1]), code); l2 = M.predict(P, M.batch_of([g2]), code); T = l1.shape[1]
        assert np.allclose(l1[:, :T - 2], l2[:, :T - 2], atol=1e-9) and not np.allclose(l1[:, T - 2:], l2[:, T - 2:])


def test_the_penalty_blocks_are_half_of_the_weights_and_the_off_block_share_measures_them():
    m, rest = M.ortho_masks(); assert m.shape == (M.D_MODEL, M.N_HEAD * M.D_HEAD) and m.sum() == m.size // 2 and (m ^ rest).all()
    for h in range(M.N_HEAD):
        pos = slice(h * M.D_HEAD, h * M.D_HEAD + M.D_POS); sem = slice(h * M.D_HEAD + M.D_POS, (h + 1) * M.D_HEAD)
        assert m[M.D_STRUCT:, pos].all() and not m[:M.D_STRUCT, pos].any() and m[:M.D_STRUCT, sem].all() and not m[M.D_STRUCT:, sem].any()          # payload -> pos and structure -> sem are penalised
    P = M.init_params(False, 1, np.float64); assert 0.4 < M.offblock_share(P) < 0.6 and M.ortho_penalty(P) > 0
    for l in range(M.N_LAYER):
        for w in ("Wq", "Wk"): P[f"{w}_{l}"] = P[f"{w}_{l}"] * ~m
    assert M.ortho_penalty(P) == 0 and M.offblock_share(P) == 0
    for l in range(M.N_LAYER):
        for w in ("Wq", "Wk"): P[f"{w}_{l}"] = np.where(m, 1.0, 0.0)
    assert abs(M.offblock_share(P) - 1.0) < 1e-12 and abs(M.ortho_penalty(P) - 4 * m.sum()) < 1e-9


def test_payload_sensitivity_is_zero_when_the_payload_is_ignored_and_positive_otherwise():
    gs = M.eval_graphs("s", 10, 10, False, 6, 0); P = M.init_params(False, 2, np.float64); s1 = M.payload_sensitivity(P, "rope2d", gs); assert 0 < s1 < 1
    P["emb_pay"][:] = P["emb_pay"][0]; assert M.payload_sensitivity(P, "rope2d", gs) == 0.0                                                      # all payload ids look the same: the heads cannot see a change


# ---------------------------------------------------------------- 4 the run
def test_training_is_seeded_steps_as_often_as_the_page_says_and_the_penalty_arms_shrink_the_blocks():
    cfg = dict(steps=4, batch=3); P1, i1 = M.train_arm("B_su2", 1, cfg); P2, i2 = M.train_arm("B_su2", 1, cfg); P3, i3 = M.train_arm("B_su2", 2, cfg)
    assert [h["step"] for h in i1["history"]] == [1, 2, 3, 4] and i1["history"][0]["ce"] == i2["history"][0]["ce"] and all(np.array_equal(P1[k], P2[k]) for k in P1) and not all(np.array_equal(P1[k], P3[k]) for k in P1)
    assert i1["params"] == M.n_params(P1) and "cx" not in P1 and "cx" in M.train_arm("A_rope1d", 1, cfg)[0] and all(math.isfinite(h["loss"]) for h in i1["history"])
    old_lam = M.ARMS["B_su2"]["lam"]; M.ARMS["B_su2"]["lam"] = 50.0
    try: Ps = M.train_arm("B_su2", 1, dict(steps=6, batch=2))[0]
    finally: M.ARMS["B_su2"]["lam"] = old_lam
    P0 = M.init_params(False, M.TRAIN["init_seed"] * 1000 + 1); assert M.ortho_penalty(Ps) < 0.97 * M.ortho_penalty(P0)                              # with a heavy penalty the blocks shrink at every step
    big = dict(steps=10, batch=3); h10 = M.train_arm("B_su2", 1, big)[1]["history"]; assert [h["step"] for h in h10] == list(range(1, 11)) and M.TRAIN["steps"] >= 3000


def test_evaluate_counts_exact_graphs_and_the_f1_of_the_positive_tokens_as_defined():
    gs = M.eval_graphs("e", 6, 8, False, 4, 0); b = M.batch_of(gs); logits = np.where(b["label"] > 0.5, 3.0, -3.0)                                    # a perfect answer
    flips = [(1, 2), (2, 1), (2, 3)]; wrong = logits.copy()
    for (i, t) in flips: wrong[i, t] = -wrong[i, t]
    orig = M.predict
    try:
        M.predict = lambda P, batch, code, taps=None: logits[:len(batch["length"])] if len(batch["length"]) == 4 else None
        r = M.evaluate(None, gs, "rope2d", chunk=25); assert r["exact"] == 1.0 and r["f1"] == 1.0 and r["tokens_fp"] == r["tokens_fn"] == 0 and r["n"] == 4
        M.predict = lambda P, batch, code, taps=None: wrong
        r = M.evaluate(None, gs, "rope2d", chunk=25)
    finally: M.predict = orig
    y = b["label"] > 0.5; sc = b["scored"]; pr = wrong > 0; tp = int(((pr & y) & sc).sum()); fp = int(((pr & ~y) & sc).sum()); fn = int(((~pr & y) & sc).sum())
    bad = {i for i, t in flips if sc[i, t]}; assert r["exact"] == (4 - len(bad)) / 4 and r["tokens_tp"] == tp and r["tokens_fp"] == fp and r["tokens_fn"] == fn and abs(r["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    assert r["graphs_with_false_alarm"] == len({i for i, t in flips if sc[i, t] and not y[i, t]}) / 4 and r["graphs_with_miss"] == len({i for i, t in flips if sc[i, t] and y[i, t]}) / 4


TINY = dict(seeds=(1,), arms=("A_rope2d",), cfg=dict(steps=2, batch=2), counts={k: 4 for k in M.EVAL_GRAPHS}, sets=M.EVAL_SETS[:2])


def test_the_run_has_the_shape_the_rules_read_and_single_seed_files_merge():
    r1 = M.run_toymodel(**TINY); r2 = M.run_toymodel(**dict(TINY, seeds=(2,))); r3 = M.run_toymodel(**dict(TINY, seeds=(3,)))
    assert r1["seeds"] == [1] and list(r1["runs"]) == ["1"] and set(r1["runs"]["1"]["A_rope2d"]) >= {"history", "seconds", "params", "offblock_share", "eval", "payload_sensitivity"} and set(r1["runs"]["1"]["A_rope2d"]["eval"]) == {"N10", "N20"}
    assert r1["eval_digest"] == r2["eval_digest"] and set(r1["baseline_exact"]) == {"N10", "N20"} and r1["tokens_mean"]["N20"] > r1["tokens_mean"]["N10"] > 0 and r1["sets"]["N20"] == dict(n=20, g=20, shift=False, graphs=4)
    assert r1["eval_digest"] == M._digest({s[0]: M.eval_graphs(s[0], s[1], s[2], s[3], 4, 0) for s in M.EVAL_SETS[:2]}) and len(M.full_eval_digest()) == 16 and M.full_eval_digest() == M.full_eval_digest() != r1["eval_digest"]
    m = M.merge_results([r3, r1, r2]); assert m["seeds"] == [1, 2, 3] and list(m["runs"]) == ["1", "2", "3"] and m["runs"]["2"] == r2["runs"]["2"] and abs(m["seconds"] - (r1["seconds"] + r2["seconds"] + r3["seconds"])) < 0.2
    assert raises(lambda: M.merge_results([r1, r1])) and raises(lambda: M.merge_results([])) and raises(lambda: M.merge_results([r1, dict(r2, eval_digest="zz")])) and raises(lambda: M.merge_results([r1, dict(r2, cfg=dict(r2["cfg"], steps=3))]))
    assert raises(lambda: M.merge_results([r1, dict(r2, quick=True)]))
    ra = M.run_toymodel(**dict(TINY, seeds=(1,), arms=("A_rope1d",))); both = M.merge_results([r1, ra]); assert both["arms"] == ["A_rope1d", "A_rope2d"] and set(both["runs"]["1"]) == {"A_rope1d", "A_rope2d"} and both["seeds"] == [1] and both["eval_digest"] == r1["eval_digest"]   # the arms of one seed in two files
    assert raises(lambda: M.merge_results([r1, ra, r2])), "seed 2 holds one arm, seed 1 holds two"; assert raises(lambda: M.merge_results([both, ra])), "an arm of a seed twice"


def test_the_quick_run_the_printout_the_merge_and_the_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): M.main(["--quick", "--out", out])
        res = json.load(open(out)); text = buf.getvalue(); assert res["quick"] and res["seeds"] == [1] and res["arms"] == ["A_rope1d", "B_su2"] and "HEADLINE" in text and "(headline)" in text and "all-zero" in text and "A_rope1d" in text
        assert len(M.verdict_toymodel(res)) == 14 and text.count("[claim") == 10 and text.count("[guard") == 2 and text.count("[instrument") == 2
        o2 = os.path.join(tmp, "y.json"); buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2): M.main(["--quick", "--seeds", "2", "--out", o2])
        o3 = os.path.join(tmp, "z.json"); buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3): M.main(["--merge", out, o2, "--out", o3])
        merged = json.load(open(o3)); assert merged["seeds"] == [1, 2] and "HEADLINE" in buf3.getvalue() and "seed 1" not in buf3.getvalue()


# ---------------------------------------------------------------- 5 the reading rules
SETS = {"N10": (10, 10), "N20": (20, 20), "N40": (40, 40), "N70": (70, 70), "N100": (100, 100), "range100_n10": (10, 100), "pay_shift_N10": (10, 10), "pay_shift_N100": (100, 100)}


def mk(**kw):
    """A hand-made result: per arm (exact by set, F1 at N10, off-block share, payload sensitivity). Base: every rule reads yes. `ov` changes single numbers: ov={(arm, set): value}; shares and sensitivity by `share` and `sens`."""
    ov = kw.get("ov", {}); share = kw.get("share", {}); sens = kw.get("sens", {}); f1 = kw.get("f1", {}); drop = kw.get("drop", ()); seeds = kw.get("seeds", (1, 2)); steps = kw.get("steps", 3000)
    base = {"A_rope1d": dict(N10=0.95, N20=0.7, N40=0.4, N70=0.2, N100=0.1, range100_n10=0.2, pay_shift_N10=0.9, pay_shift_N100=0.1), "A_rope2d": dict(N10=0.95, N20=0.95, N40=0.95, N70=0.9, N100=0.8, range100_n10=0.9, pay_shift_N10=0.9, pay_shift_N100=0.7),
            "B_su2": dict(N10=0.95, N20=0.95, N40=0.95, N70=0.93, N100=0.9, range100_n10=0.92, pay_shift_N10=0.9, pay_shift_N100=0.85), "B_su2_free": dict(N10=0.95, N20=0.95, N40=0.95, N70=0.93, N100=0.9, range100_n10=0.92, pay_shift_N10=0.8, pay_shift_N100=0.8),
            "A_rope2d_orth": dict(N10=0.95, N20=0.95, N40=0.95, N70=0.9, N100=0.8, range100_n10=0.9, pay_shift_N10=0.9, pay_shift_N100=0.7)}
    shares = {"A_rope1d": 0.3, "A_rope2d": 0.3, "B_su2": 0.005, "B_su2_free": 0.3, "A_rope2d_orth": 0.005}; senss = {"A_rope1d": 0.2, "A_rope2d": 0.2, "B_su2": 0.1, "B_su2_free": 0.2, "A_rope2d_orth": 0.1}
    def run(sd):
        out = {}
        for arm in M.ARMS:
            if arm in drop: continue
            ev = {nm: dict(n=(kw.get("graphs") or {}).get(nm, M.EVAL_GRAPHS[nm]), exact=ov.get((arm, nm), base[arm][nm]), f1=(f1.get(arm, 0.9) if nm == "N10" else 0.5)) for nm in SETS}
            out[arm] = dict(history=[dict(step=steps // 2, ce=0.3, loss=0.3), dict(step=steps, ce=0.1, loss=0.12)], seconds=1.0, params=100, offblock_share=share.get(arm, shares[arm]), payload_sensitivity=sens.get(arm, senss[arm]), eval=ev)
        return out
    sets = {nm: dict(n=n, g=g, shift=nm.startswith("pay_shift"), graphs=(kw.get("graphs") or {}).get(nm, M.EVAL_GRAPHS[nm])) for nm, (n, g) in SETS.items()}
    return dict(seeds=list(seeds), arms=[a for a in M.ARMS if a not in drop], cfg=dict(M.TRAIN, steps=steps), sets=sets, runs={str(s): run(s) for s in seeds}, tokens_mean={nm: 10.0 * (n / 10 + 1) for nm, (n, g) in SETS.items()}, baseline_exact={nm: 0.1 for nm in SETS},
                eval_digest="x", seconds=1.0)


def v(**kw): return [x for _, x in M.verdict_toymodel(mk(**kw))]


def test_every_reading_rule_reads_as_it_should_on_made_up_results_and_flips_where_it_should():
    base = M.verdict_toymodel(mk()); assert len(base) == 14 and [k for k, _ in base] == [k for k, _ in M.TM_LINES] and [x for _, x in base] == [True] * 14, base
    r = mk(); r["runs"]["1"]["B_su2"]["history"][-1]["step"] = 2999; assert M.verdict_toymodel(r)[0][1] is False                    # line 1: all steps, finite loss, all graphs scored
    r = mk(); r["runs"]["2"]["A_rope2d"]["history"][0]["ce"] = float("nan"); assert M.verdict_toymodel(r)[0][1] is False
    r = mk(); r["runs"]["2"]["A_rope2d"]["history"][0]["loss"] = float("inf"); assert M.verdict_toymodel(r)[0][1] is False
    r = mk(); r["runs"]["1"]["A_rope1d"]["eval"]["N70"]["n"] = 199; assert M.verdict_toymodel(r)[0][1] is False
    r = mk(); r["runs"]["1"]["A_rope1d"]["history"] = []; assert M.verdict_toymodel(r)[0][1] is False
    r = mk(); r["sets"]["N40"]["n"] = 41; assert M.verdict_toymodel(r)[1][1] is False                                              # line 2: the sets hold the sizes they are named for
    r = mk(); r["sets"]["range100_n10"]["g"] = 10; assert M.verdict_toymodel(r)[1][1] is False
    r = mk(); del r["sets"]["N70"]; [rn.pop("N70") for run in r["runs"].values() for a in run.values() for rn in [a["eval"]]]; assert M.verdict_toymodel(r)[1][1] is False
    r = mk(); r["tokens_mean"]["N20"] = 0; assert M.verdict_toymodel(r)[1][1] is False
    assert v(ov={("B_su2", "N10"): 0.801})[2] is True and v(ov={("B_su2", "N10"): 0.799})[2] is False and v(ov={("A_rope1d", "N10"): 0.799})[2] is False                    # line 3: every arm learned N10 (0.80)
    assert v(f1={"A_rope2d_orth": 0.501})[3] is True and v(f1={"A_rope2d_orth": 0.499})[3] is False and v(f1={"A_rope2d_orth": 0.5})[3] is False                              # line 4: F1 above 0.5, strictly
    assert v(ov={("A_rope1d", "N100"): 0.474})[4] is True and v(ov={("A_rope1d", "N100"): 0.476})[4] is False                                                                # line 5: rope1d at N100 at most half of N10
    assert v(ov={("B_su2", "N100"): 0.761})[5] is True and v(ov={("B_su2", "N100"): 0.759})[5] is False                                                                      # line 6: su2 keeps 0.8 of N10
    assert v(ov={("B_su2", "N100"): 0.9, ("A_rope1d", "N100"): 0.699})[6] is True and v(ov={("B_su2", "N100"): 0.9, ("A_rope1d", "N100"): 0.701})[6] is False                  # line 7: B over A by 0.20
    assert v(ov={("A_rope2d", "N100"): 0.761})[7] is True and v(ov={("A_rope2d", "N100"): 0.759})[7] is False                                                                # line 8: rope2d keeps 0.8
    assert v(ov={("B_su2", "N100"): 0.9, ("A_rope2d", "N100"): 0.849})[8] is True and v(ov={("B_su2", "N100"): 0.9, ("A_rope2d", "N100"): 0.851})[8] is False                  # line 9: B over rope2d by 0.05
    assert v(share={"B_su2": 0.0099})[9] is True and v(share={"B_su2": 0.0101})[9] is False and v(share={"A_rope2d_orth": 0.0101})[9] is False and v(share={"B_su2_free": 0.1001})[9] is True and v(share={"B_su2_free": 0.0999})[9] is False and v(share={"A_rope1d": 0.0999})[9] is False
    assert v(sens={"B_su2": 0.1999})[10] is True and v(sens={"B_su2": 0.2})[10] is False and v(sens={"B_su2": 0.21})[10] is False                                          # line 11: strictly smaller
    assert v(ov={("B_su2", "pay_shift_N10"): 0.851})[11] is True and v(ov={("B_su2", "pay_shift_N10"): 0.849})[11] is False                                                  # line 12: +0.05 under a payload shift
    assert v(ov={("A_rope1d", "range100_n10"): 0.474})[12] is True and v(ov={("A_rope1d", "range100_n10"): 0.476})[12] is False                                              # line 13: range alone breaks rope1d
    assert v(ov={("A_rope2d", "N40"): 0.761})[13] is True and v(ov={("A_rope2d", "N40"): 0.759})[13] is False and v(ov={("B_su2", "N40"): 0.759})[13] is False and v(ov={("B_su2", "N40"): 0.761})[13] is True    # line 14
    # the figure of an arm is the mean over the seeds
    r = mk(); r["runs"]["1"]["A_rope1d"]["eval"]["N100"]["exact"] = 0.2; r["runs"]["2"]["A_rope1d"]["eval"]["N100"]["exact"] = 0.6; assert abs(M.acc(r, "A_rope1d", "N100") - 0.4) < 1e-12 and M.acc(r, "nope", "N100") is None and M.verdict_toymodel(r)[4][1] is True
    r = mk(); r["runs"]["1"]["A_rope1d"]["eval"]["N100"]["exact"] = 0.2; r["runs"]["2"]["A_rope1d"]["eval"]["N100"]["exact"] = 0.8; assert M.verdict_toymodel(r)[4][1] is False                  # mean 0.5 against half of 0.95
    # an arm missing: the rules that need it read None, the others still read
    out = v(drop=("A_rope1d",)); assert out[4] is None and out[6] is None and out[12] is None and out[2] is None and out[3] is None and out[5] is True and out[7] is True and out[13] is True and out[10] is True
    out = v(drop=("B_su2",)); assert out[5] is None and out[6] is None and out[8] is None and out[9] is None and out[10] is None and out[11] is None and out[13] is None and out[4] is True and out[7] is True
    out = v(drop=("A_rope2d",)); assert out[7] is None and out[8] is None and out[13] is None and out[5] is True
    out = v(drop=("B_su2_free",)); assert out[9] is None and out[10] is None and out[11] is None and out[5] is True
    assert M.TM_HEADLINE == (5, 6, 7)
    fail = M.verdict_toymodel(mk(ov={("A_rope1d", "N100"): 0.9})); assert fail[4][1] is False and fail[6][1] is False and not all(fail[i - 1][1] for i in M.TM_HEADLINE)


def test_the_printout_prints_every_arm_and_set_and_the_rules():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_toymodel(mk()); M.print_verdict_toymodel(M.verdict_toymodel(mk()))
    text = buf.getvalue(); assert all(a in text for a in M.ARMS) and all(s in text for s in SETS) and "all-zero" in text and "HEADLINE" in text and "ALL YES" in text and text.count("   (headline)") == 3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_verdict_toymodel(M.verdict_toymodel(mk(drop=("B_su2",))))
    assert "not run" in buf.getvalue() and "NOT all yes" in buf.getvalue()


# ---------------------------------------------------------------- 6 the page
PAGE_SHA256 = "a5b7cfd346c1f47da8c80718fb9bbe2b5b2e8adbdf6496bd3f55d1863af37165"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_fourteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(M.TM_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    assert M.TM_HEADLINE == (5, 6, 7) and "**Headline** (lines 5, 6 and 7)" in head and f"my single number is **{HEADLINE_PCT} %**" in head
    assert (M.D_MODEL, M.D_STRUCT, M.N_HEAD, M.D_HEAD, M.D_POS, M.D_FF, M.N_LAYER, M.THETA, M.SEEDS) == (64, 32, 4, 32, 16, 128, 2, (1.0, 0.4, 0.16, 0.064), (1, 2, 3))
    assert M.TRAIN == dict(steps=STEPS, batch=32, lr=2e-3, lr_end=0.1, clip=1.0, pos_weight=3.0, init_seed=11, data_seed=23, eval_graphs=300) and {a: (s["code"], s["table"], s["lam"]) for a, s in M.ARMS.items()} == {
        "A_rope1d": ("rope1d", True, 0.0), "A_rope2d": ("rope2d", False, 0.0), "B_su2": ("su2", False, 0.05), "B_su2_free": ("su2", False, 0.0), "A_rope2d_orth": ("rope2d", False, 0.05)}
    assert (M.TM_MIN_N10, M.TM_MIN_F1, M.TM_HALF, M.TM_KEEP, M.TM_B_OVER_A, M.TM_B_OVER_ROPE2D, M.TM_SHARE_LO, M.TM_SHARE_HI, M.TM_SHIFT_GAIN) == (0.80, 0.5, 0.5, 0.8, 0.20, 0.05, 0.01, 0.10, 0.05)
    assert M.EVAL_GRAPHS == {"N10": 400, "N20": 400, "N40": 300, "N70": 200, "N100": 200, "range100_n10": 400, "pay_shift_N10": 400, "pay_shift_N100": 200}
    for needle in NEEDLES: assert needle in head, needle


PROBS = [96, 99, 30, 96, 96, 12, 18, 35, 10, 88, 70, 25, 94, 15]
HEADLINE_PCT = 10
STEPS = 4000
NEEDLES = ("Concept-map item 103", "`python/axi/experiments/toymodel.py`", "The 'Toy Model' Tensor Training Benchmark", "I attribute none of it to him", "every choice below is mine, not Rajnish's", "fails at N = 100 almost by construction", "axi_toy_model.ipynb", "Ostmeier",
           "I did **not** evaluate any model on a graph of more than ten services", "`toymodel.verdict_toymodel`", "about 101,000 parameters", "non-abelian", "off-block share", "payload sensitivity")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

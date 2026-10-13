"""Implements concept-map item 89 (two efficiency claims of a pasted audit; rows 3, 9, 27): the INSTRUMENT of docs/EFFICIENCY_MEASURES.md, written BEFORE any measurement was read. It checks the
instrument, not a result:
  1. the timer takes the smallest block mean and calls the function inner x repeats times (a fake clock, so the test never depends on the speed of the machine);
  2. the three state paths carry the same state: the prompt text is the one the model sees, the JSON round trip returns the graph, the binary frame returns the same edges (src, dst, relation,
     weight) with a valid CRC-32C and a flipped bit is refused;
  3. the text lengths: sparse text is exactly header + the cells it names (a hand count), an empty grid has no cells, a full grid is longer than its plain text, one lit cell in a 30 x 30 grid is far
     shorter, and the sparse text restores every grid (random grids, all 8 images);
  4. the ARC loader reads the real file layout (train pairs and the first test input) and the whole pipeline (`main`) runs on a small stand-in folder, writes its file, and the file reads back;
  5. every reading rule is exact at its boundary (2 ms, equal speed, 349 / 350 ns, 1 percent, one grid in four, half of the grids, equal totals) and flips when one number is doctored;
  6. the page and the module agree (the same constants, the same nine reading-rule lines) and the page's text above its Result is byte for byte what was written before any run (a hash).
What it does NOT show: any speed of a real machine, or anything about real ARC grids (the pre-registered run)."""
import contextlib, hashlib, io, json, os, random, re, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi import wire
from axi.engine import d4
from axi.engine.cico_parser import CICOParser
from axi.experiments import efficiency as E
from axi.experiments.agent_loop import _serialize
from axi.experiments.scope_tasks import make_scope_tasks

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "EFFICIENCY_MEASURES.md")
PAGE_SHA256 = "d5b863126ffaa3e60c12bc289c3e43a8a445ad3a46a614ef41056aa1562f2877"          # filled in when the page is written, before the run
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


class FakeClock:
    def __init__(self, ticks): self.ticks = list(ticks); self.n = 0
    def __call__(self): v = self.ticks[self.n]; self.n += 1; return v


def test_timer_takes_the_smallest_block_mean_and_calls_the_function_inner_times_repeats():
    calls = []; clock = FakeClock([0, 4000, 10_000, 12_000, 20_000, 26_000])          # blocks of 4000, 2000, 6000 ns
    best = E.per_call_ns(lambda: calls.append(1), inner=2, repeats=3, timer=clock)
    assert best == 1000 and len(calls) == 6 and clock.n == 6
    clock = FakeClock([0, 5000]); assert E.per_call_ns(lambda: None, inner=5, repeats=1, timer=clock) == 1000


def test_the_three_state_paths_carry_the_same_state():
    for t in make_scope_tasks(21, 7):
        g = t.graph; assert E.graph_dict(g)["nodes"] and _serialize(g)
        back = E.json_roundtrip(g)
        assert [tuple(n) for n in back["nodes"]] == [(r, c, v) for (r, c), v in sorted(g.nodes.items())]
        assert [(e[0], e[1], e[2], e[3], e[4], e[5]) for e in back["edges"]] == [(u[0], u[1], w[0], w[1], rel, wt) for (u, w, rel), wt in sorted(g.edges.items())]
        h, a = E.wire_roundtrip(g); ids = {n: i for i, n in enumerate(sorted(g.nodes))}; keys = sorted(g.edges)
        assert h["edge_count"] == len(keys) and h["node_count"] == len(g.nodes) and len(a) == len(keys)
        assert [int(x) for x in a["src"]] == [ids[k[0]] for k in keys] and [int(x) for x in a["dst"]] == [ids[k[1]] for k in keys]
        assert [int(x) for x in a["rel"]] == [E.REL_IDS[k[2]] for k in keys] and [int(x) for x in a["weight"]] == [g.edges[k] for k in keys]
        frame = bytearray(E.wire_frame(g)); frame[40] ^= 1
        try: wire.decode(bytes(frame)); raise AssertionError("a flipped bit was accepted")
        except wire.WireError: pass
    assert set(E.REL_IDS) >= {k[2] for t in make_scope_tasks(140, 7) for k in t.graph.edges}


def test_transport_rows_have_every_field_and_the_summary_is_the_median():
    rows = E.time_transport(n=7, inner=1, repeats=1)
    assert len(rows) == 7 and all(set(r) >= {"text_ns", "json_ns", "wire_total_ns", "wire_decode_ns", "filter_ns", "text_chars", "json_bytes", "frame_bytes"} for r in rows)
    assert all(r["text_ns"] > 0 and r["frame_bytes"] >= 32 + 4 for r in rows)
    fake = [dict(text_ns=a, json_ns=a, wire_total_ns=a, wire_decode_ns=a, filter_ns=a, text_chars=a, json_bytes=a, frame_bytes=a) for a in (10, 30, 20, 1000)]
    s = E.summarize_transport(fake, 5_000_000_000); assert s["text_ns"] == 25 and s["n"] == 4 and s["model_call_ns"] == 5_000_000_000


def test_model_call_reference_is_the_median_of_the_saved_seconds():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json"); json.dump({"rows": [{"seconds": s} for s in (0.5, 0.7, 2.0)]}, open(p, "w")); assert E.model_call_ns(p) == 700_000_000


def test_sparse_text_is_header_plus_the_named_cells():
    g = np.zeros((4, 5), int); g[1, 2] = 3; g[3, 4] = 7
    s = E.sparse_text(g); head, body = s.split(":"); cells = body.split(";")
    h, w, t = d4.compile_stream(g, keep_background=False)["header"]
    assert head == f"{h}x{w}t{t}" and len(cells) == 2 and all(re.fullmatch(r"\d+,\d+=\d", c) for c in cells)
    assert len(s) == len(head) + 1 + sum(len(c) for c in cells) + 1
    assert E.dense_text(g) == "00000\n00300\n00000\n00007"
    assert E.dense_text(np.array([[1, 2], [3, 4]])) == "12\n34" and len(E.dense_text(g)) == 4 * 5 + 3
    assert E.sparse_text(np.zeros((3, 3), int)).endswith(":") and len(E.sparse_text(np.zeros((3, 3), int)).split(":")[1]) == 0
    full = np.arange(1, 10).reshape(3, 3); assert len(E.sparse_text(full)) > len(E.dense_text(full))
    one = np.zeros((30, 30), int); one[15, 15] = 5; assert 10 * len(E.sparse_text(one)) < len(E.dense_text(one))


def test_sparse_text_restores_every_grid_in_all_eight_images():
    rng = np.random.default_rng(0)
    for _ in range(60):
        g = (rng.integers(1, 6, (int(rng.integers(1, 9)), int(rng.integers(1, 9)))) * (rng.random((1, 1)) < 2) * (rng.random((1, 1)) < 2)).astype(int)
        g = g * (rng.random(g.shape) < rng.random())
        for t in d4.IDS:
            x = d4.apply(g, t); s = d4.compile_stream(x, keep_background=False); assert (d4.decompile_stream(s) == x).all()


def _fake_arc(folder, n=3):
    rng = random.Random(1)
    def grid():
        h, w = rng.randint(2, 5), rng.randint(2, 6); return [[rng.choice([0, 0, 0, 1, 2]) for _ in range(w)] for _ in range(h)]
    for k in range(n):
        pairs = [{"input": grid(), "output": grid()} for _ in range(2)]
        json.dump({"train": pairs, "test": [{"input": grid(), "output": grid()}, {"input": grid(), "output": grid()}]}, open(os.path.join(folder, f"t{k:02d}.json"), "w"))
    open(os.path.join(folder, "README.md"), "w").write("not a task")


def test_the_loader_and_the_pipeline_on_a_stand_in_folder():
    with tempfile.TemporaryDirectory() as d:
        _fake_arc(d); tasks = E.load_arc_dir(d); assert list(tasks) == ["t00", "t01", "t02"]
        raw = json.load(open(os.path.join(d, "t00.json")))
        assert len(tasks["t00"]["train"]) == 2 and (tasks["t00"]["test_input"] == np.asarray(raw["test"][0]["input"])).all()        # the FIRST test input
        grids, per_task = E.arc_rows(tasks); assert len(grids) == 3 * (2 + 2 + 1) and len(per_task) == 3
        assert {r["role"] for r in grids} == {"train0_in", "train1_in", "train0_out", "train1_out", "test_in"}
        s = E.summarize_arc(grids, per_task)
        assert s["grids"] == 15 and s["tasks"] == 3 and s["lossless"] == 15 and s["canon_identical"] == 3 and sum(e["n"] for e in s["bands"].values()) == 15
        assert s["total_dense"] == sum(r["dense"] for r in grids) and 1 <= min(r["raw_distinct"] for r in per_task) and max(r["raw_distinct"] for r in per_task) <= 8
        out = os.path.join(d, "arc.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): E.main(["arc", "--arc-dir", d, "--out", out])
        saved = json.load(open(out)); assert saved["summary"] == json.loads(json.dumps(s)) and saved["meta"]["n_tasks"] == 3 and "[guard]" in buf.getvalue()


def test_transport_pipeline_writes_its_file():
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "run.json"); json.dump({"rows": [{"seconds": 1.0}, {"seconds": 3.0}]}, open(run, "w")); out = os.path.join(d, "t.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): E.main(["transport", "--out", out, "--model-run", run, "--n", "5", "--inner", "1", "--repeats", "1"])
        saved = json.load(open(out)); assert len(saved["rows"]) == 5 and saved["meta"]["n"] == 5 and saved["meta"]["model_run"] == "run.json"
        assert saved["summary"]["model_call_ns"] == 2_000_000_000 and saved["summary"]["n"] == 5 and "[scale]" in buf.getvalue()


def test_lossless_is_measured_not_assumed():
    from unittest import mock
    with tempfile.TemporaryDirectory() as d:
        _fake_arc(d); tasks = E.load_arc_dir(d)
        with mock.patch.object(d4, "decompile_stream", lambda st: np.zeros((1, 1), int)):
            grids = E.arc_grid_rows(tasks, "zero")
        assert not any(r["lossless"] for r in grids) and E.summarize_arc(grids, [])["lossless"] == 0
        assert E.verdict_arc(E.summarize_arc(grids, [dict(task="a", raw_distinct=1, canon_distinct=1)]))[0][1] is False


def test_canonical_identity_is_measured_not_assumed():
    from unittest import mock
    with tempfile.TemporaryDirectory() as d:
        _fake_arc(d); tasks = E.load_arc_dir(d); orig = d4.canonicalize_task
        _, honest = E.arc_rows(tasks); assert all(r["canon_distinct"] == 1 for r in honest)
        with mock.patch.object(d4, "canonicalize_task", lambda t: (t, 0)):                      # a "canonicalizer" that does nothing
            _, broken = E.arc_rows(tasks)
        assert [r["canon_distinct"] for r in broken] == [r["raw_distinct"] for r in broken] and any(r["canon_distinct"] > 1 for r in broken)


def test_the_modal_background_variant():
    g = np.array([[3, 3, 3, 3], [3, 0, 3, 3], [3, 3, 5, 3]])
    assert E.modal_background(g) == 3 and E.modal_background(np.array([[1, 1, 2, 2]])) == 1 and E.modal_background(np.array([[4, 2], [2, 4]])) == 2        # ties: the smallest colour
    st = E.sparse_text(g, 3); assert st.split(":")[0].startswith("3x4b3t") and len(st.split(":")[1].split(";")) == 2
    assert E.sparse_text(g) .startswith("3x4t") and "b" not in E.sparse_text(g).split(":")[0]            # the legacy header carries no colour
    assert len(E.sparse_text(g, 3)) < len(E.sparse_text(g))
    with tempfile.TemporaryDirectory() as d:
        _fake_arc(d); tasks = E.load_arc_dir(d); z = E.arc_grid_rows(tasks, "zero"); m = E.arc_grid_rows(tasks, "modal")
        assert len(z) == len(m) == 15 and all(r["lossless"] for r in m) and all("background" in r and "background" not in q for r, q in zip(m, z))
        assert all(r["nonzero"] <= q["nonzero"] and r["dense"] == q["dense"] and (r["task"], r["role"]) == (q["task"], q["role"]) for r, q in zip(m, z))
        out = os.path.join(d, "m.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): E.main(["arc", "--background", "modal", "--arc-dir", d, "--out", out])
        saved = json.load(open(out)); assert saved["meta"]["exploratory"] is True and saved["meta"]["test"] == "efficiency_arc_modal" and "tasks" not in saved and len(saved["grids"]) == 15
        assert saved["summary"] == json.loads(json.dumps(E.summarize_arc(m, []))) and "exploratory" in buf.getvalue()


def test_modal_rows_use_each_grids_own_background():
    bg3 = np.full((4, 4), 3); bg3[1, 1] = 5; bg3[2, 3] = 0; sparse0 = np.zeros((4, 4), int); sparse0[0, 0] = 7
    tasks = {"x": {"train": [(bg3, sparse0)], "test_input": bg3.copy()}}
    z = E.arc_grid_rows(tasks, "zero"); m = E.arc_grid_rows(tasks, "modal")
    assert [r["background"] for r in m] == [3, 0, 3] and [r["nonzero"] for r in m] == [2, 1, 2] and [r["nonzero"] for r in z] == [15, 1, 15]
    assert all(r["lossless"] for r in m) and m[0]["sparse"] < z[0]["sparse"] and m[1]["sparse"] == z[1]["sparse"] + 2          # the modal header names the colour: "b0" is 2 characters longer


def test_the_zero_variant_rows_are_what_the_pipeline_reports():
    with tempfile.TemporaryDirectory() as d:
        _fake_arc(d); tasks = E.load_arc_dir(d); grids, _ = E.arc_rows(tasks); assert grids == E.arc_grid_rows(tasks, "zero") and all("background" not in r for r in grids)


def _t(**kw): s = dict(text_ns=1000, json_ns=1000, wire_total_ns=900, wire_decode_ns=300, filter_ns=10, text_chars=1, json_bytes=1, frame_bytes=1, n=140, model_call_ns=10**9); s.update(kw); return s
def _ok(checks): return [ok for _, ok in checks]


def test_transport_rules_at_their_boundaries():
    assert _ok(E.verdict_transport(_t())) == [True, True, True, True]
    assert E.verdict_transport(_t(text_ns=1_999_999))[0][1] and not E.verdict_transport(_t(text_ns=2_000_000))[0][1] and not E.verdict_transport(_t(json_ns=2_000_000))[0][1]
    assert E.verdict_transport(_t(json_ns=1000, wire_total_ns=999))[1][1] and not E.verdict_transport(_t(json_ns=1000, wire_total_ns=1000))[1][1]
    assert E.verdict_transport(_t(wire_decode_ns=349))[2][1] and not E.verdict_transport(_t(wire_decode_ns=350))[2][1]
    assert E.verdict_transport(_t(text_ns=5_000_000, json_ns=4_999_999, model_call_ns=10**9))[3][1] and not E.verdict_transport(_t(text_ns=5_000_000, json_ns=5_000_000, model_call_ns=10**9))[3][1]


def _a(**kw):
    s = dict(grids=100, tasks=10, lossless=100, canon_identical=10, ge80=25, sparse_shorter=50, total_sparse=99, total_dense=100); s.update(kw); return s


def test_arc_rules_at_their_boundaries():
    assert _ok(E.verdict_arc(_a())) == [True] * 5
    assert not E.verdict_arc(_a(lossless=99))[0][1] and not E.verdict_arc(_a(canon_identical=9))[1][1]
    assert E.verdict_arc(_a(ge80=25))[2][1] and not E.verdict_arc(_a(ge80=24))[2][1]
    assert E.verdict_arc(_a(sparse_shorter=50))[3][1] and not E.verdict_arc(_a(sparse_shorter=49))[3][1]
    assert E.verdict_arc(_a(total_sparse=99, total_dense=100))[4][1] and not E.verdict_arc(_a(total_sparse=100, total_dense=100))[4][1]


def test_arc_summary_counts_are_exact_on_hand_built_rows():
    g = lambda nz, dense, sparse, ok=True: dict(task="a", role="x", h=2, w=5, nonzero=nz, dense=dense, sparse=sparse, lossless=ok)
    rows = [g(0, 11, 6), g(10, 11, 80), g(2, 11, 2), g(5, 11, 11), g(1, 100, 20), g(3, 100, 21), g(4, 10, 5)]
    pt = [dict(task="a", raw_distinct=8, canon_distinct=1), dict(task="b", raw_distinct=4, canon_distinct=2)]
    s = E.summarize_arc(rows, pt)
    # shorter: 6<11, 2<11, 20<100, 21<100, 5<10 (not 80>11, not 11==11). 50 percent or more (2*s <= d): 2, 20, 21, 5 (the 5 / 10 row is exactly half). 80 percent or more (5*s <= d): 2 and 20 (the 21 / 100 row is 79 percent, not enough)
    assert s["sparse_shorter"] == 5 and s["ge50"] == 4 and s["ge80"] == 2
    assert s["total_dense"] == 254 and s["total_sparse"] == 145 and s["canon_identical"] == 1 and s["raw_distinct_sum"] == 12 and s["lossless"] == 7
    assert abs(s["median_empty_share"] - 0.7) < 1e-9                                  # empty shares 1, 0, 0.8, 0.5, 0.9, 0.7, 0.6 -> sorted 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1 -> median 0.7
    assert s["bands"]["0-10%"] == dict(n=1, sparse_shorter=1) and s["bands"]["10-20%"] == dict(n=1, sparse_shorter=1) and s["bands"]["90-100%"] == dict(n=1, sparse_shorter=0)
    assert s["bands"]["50-60%"] == dict(n=1, sparse_shorter=0)                        # sparse == plain is NOT shorter
    assert list(s["bands"]) == ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "90-100%"]
    assert s["bands"]["30-40%"] == dict(n=1, sparse_shorter=1) and s["bands"]["40-50%"] == dict(n=1, sparse_shorter=1)


def test_page_and_module_agree_and_the_page_text_is_locked():
    if not os.path.exists(PAGE): return                                          # the page is written right after the instrument; the test then goes live
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    for must in (str(E.N_GRAPHS), "seed 7", str(E.N_ARC_TASKS), E.ARC_COMMIT, "2 ms", "350 ns", "1 percent", "characters", "Predictions"):
        assert must in head, must
    assert PAGE_SHA256 and _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for line, _ in E.verdict_transport(_t()) + E.verdict_arc(_a()): assert line in head, line


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
    print(sum(1 for n in globals() if n.startswith("test_")), "tests passed")

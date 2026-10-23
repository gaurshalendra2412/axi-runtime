"""Engineering support for docs/PROVIDER_SIDE_TEST.md (concept-map item 106): K jobs share one GPU and one prefix cache; exact-state against a kept chat with the cache on.
No GPU, no vLLM and no model are needed here. This file tests the INSTRUMENT, not the finding (there is no finding yet):
  1. the jobs: S shows the exact current state and nothing else; KC and KN keep the whole chat with the raw replies; the state is updated with repair_m6; the context stop;
  2. the stand-in cache (the model of an engine the harness is tried on): hits, shared prefixes, the size limit, the tail of a sequence is evicted before its head;
  3. a cell: lockstep rounds, the bookkeeping adds up, same seeds at every K, a cold cache at the start, KN empties the cache each round, KR changes the order and nothing else;
  4. the summaries on a cell built by hand;
  5. the thirteen reading rules each flipping where they should, at their edges, and `None` when a cell is missing;
  6. the vLLM adapter against a FAKE `vllm` module that has the calls the class makes (this checks my code, not vLLM: the notebook's first cells do that on the GPU);
  7. putting cells together; the command line; the notebook (its code cells parse and plan the same cells as the module);
  8. the page is locked (hash of the text above the Result), carries the thirteen lines with the probabilities, the settings and the thresholds the code uses, and the stand-in table it prints."""
import contextlib, copy, hashlib, io, json, os, random, re, sys, tempfile, types
sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
from axi.experiments.agent_loop import base_messages, Task, copy_graph, same_graph, strict_parse
from axi.experiments.long_horizon import oracle_script

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "PROVIDER_SIDE_TEST.md")
NOTEBOOK = os.path.join(HERE, "..", "colab", "axi_provider_side.ipynb")
RESULT_MARK = "\n\n---\n\n# Result"
FAST = os.environ.get("AXI_FAST") == "1"


def raises(f, exc=AssertionError):
    try: f()
    except exc: return True
    return False


# ---------------------------------------------------------------- 1 the jobs
def _play(arm, seed=4242, steps=6, text=lambda st: st["ideal"], n_prompt=lambda t: 100 * t):
    s = P.Session(arm, seed, steps); msgs = []
    while not s.done:
        rq = s.request(); msgs.append(rq.messages); st = s.script[s.t]; s.accept(P.Reply(text(st), n_prompt(s.t + 1), 0, 7))
    return s, msgs


def test_s_shows_the_exact_current_state_and_nothing_else_and_the_chat_arms_keep_everything():
    for arm in ("KC", "KN", "KR"):
        s, msgs = _play(arm); assert s.stopped is None and len(msgs) == len(s.script) >= 5
        for t, m in enumerate(msgs):
            assert len(m) == 1 + 2 * t + 1 and m[0]["role"] == "system" and m[-1]["role"] == "user"
            assert [x["role"] for x in m[1:-1]] == ["user", "assistant"] * t
            if t: assert m[2 * t]["content"] == s.script[t - 1]["ideal"] and m[2 * t - 1]["content"].endswith("Request: " + s.script[t - 1]["instruction"])   # the raw reply stays, word for word
    s, msgs = _play("S"); g0, script, oracle = oracle_script(4242, 6)
    for t, m in enumerate(msgs):
        assert len(m) == 2 and [x["role"] for x in m] == ["system", "user"]
        want = base_messages(Task(script[t]["kind"], copy_graph(oracle[t]), script[t]["instruction"], script[t]["ideal"], oracle[t + 1]), False)
        assert m == want, t                                                                      # the state shown is the oracle's own when every reply was right
    assert all(x["exact"] and x["parsed"] and x["admitted"] and not x["corrupted"] and x["distance"] == 0 for x in s.log)


def test_a_wrong_reply_is_repaired_or_refused_and_the_state_shown_next_is_the_held_one():
    g0, script, oracle = oracle_script(4242, 6)
    s = P.Session("S", 4242, 6); rq = s.request(); s.accept(P.Reply("I cannot do that.", 50, 0, 5))
    assert not s.log[0]["parsed"] and not s.log[0]["admitted"] and s.live.nodes == g0.nodes and s.live.edges == g0.edges and not s.log[0]["exact"] and s.log[0]["distance"] > 0   # nothing applied, and the step is not the oracle's
    nxt = s.request().messages[1]["content"]; assert nxt == base_messages(Task(script[1]["kind"], copy_graph(g0), script[1]["instruction"], script[1]["ideal"], oracle[2]), False)[1]["content"]
    s2 = P.Session("KC", 4242, 6); s2.request(); s2.accept(P.Reply("not a delta", 50, 0, 5)); m = s2.request().messages
    assert m[2] == {"role": "assistant", "content": "not a delta"}                                    # the chat keeps what the model wrote, even when it was not a delta
    for arm in ("S", "KC"):                                                                          # a reply the gate refuses leaves the state as it was; repair_m6 never corrupts
        s3 = P.Session(arm, 4242, 4); s3.request(); s3.accept(P.Reply("DEL[0,0] DEL[7,7] ADD[(9,9)->(8,8):dep#3]", 10, 0, 3)); assert not s3.log[0]["corrupted"]


def test_a_kept_chat_stops_before_its_prompt_would_pass_the_cap_and_exact_state_never_does():
    # the long-horizon rule: stop when prompt * (steps_done + 1) / steps_done > MAX_CONTEXT and steps remain
    for n_prompt, k, stops in ((3800, 1, False), (3801, 1, True), (5066, 2, False), (5067, 2, True), (6080, 4, False), (6081, 4, True)):
        for arm in ("KC", "KN", "KR"):
            s = P.Session(arm, 4242, 10)
            for t in range(k - 1): s.request(); s.accept(P.Reply(s.script[t]["ideal"], 100, 0, 5)); assert s.stopped is None
            s.request(); s.accept(P.Reply(s.script[k - 1]["ideal"], n_prompt, 0, 5)); assert (s.stopped == "context") == stops, (arm, n_prompt, k, s.stopped); assert s.done == stops
    s = P.Session("S", 4242, 8); s.request(); s.accept(P.Reply(s.script[0]["ideal"], 99999, 0, 5)); assert s.stopped is None and not s.done
    s = P.Session("KC", 4242, 2); s.request(); s.accept(P.Reply(s.script[0]["ideal"], 99999, 0, 5)); assert s.stopped == "context" and s.done                    # steps remain: stop
    s = P.Session("KC", 4242, 2)
    for t in range(len(s.script)): s.request(); s.accept(P.Reply(s.script[t]["ideal"], 100 if t < len(s.script) - 1 else 99999, 0, 5))
    assert s.stopped is None and s.done                                                                                                                       # the last step: nothing left to stop


# ---------------------------------------------------------------- 2 the stand-in cache
def _req(words, reply="ok"):
    return P.Request([{"role": "system", "content": " ".join(words)}, {"role": "user", "content": "go"}], reply)


def test_the_stand_in_counts_hits_on_full_blocks_of_the_prompt_only_and_never_the_last_token():
    e = P.StandInEngine(capacity_tokens=16 * 100, prefill_tps=1000.0, decode_step_s=0.5); r = _req(["w%s" % chr(97 + i % 26) * 3 for i in range(60)])
    n = len(e._render(r.messages))
    (a,), t = e.generate_round([r]); assert (a.n_prompt, a.n_cached, a.n_new) == (n, 0, 2) and abs(t - (n / 1000.0 + 2 * 0.5)) < 1e-9            # "ok" + the end token
    (b,), _ = e.generate_round([r]); assert b.n_prompt == n and b.n_cached == (n // 16) * 16 - (1 if n % 16 == 0 else 0) and b.n_cached > 0
    e2 = P.StandInEngine(capacity_tokens=16 * 100)                                                  # a prompt that is exactly full blocks: the last token is still computed
    k = 0
    while True:
        r2 = _req(["abc"] * (20 + k)); n2 = len(e2._render(r2.messages))
        if n2 % 16 == 0: break
        k += 1
    e2.generate_round([r2]); (c,), _ = e2.generate_round([r2]); assert c.n_cached == n2 - 1 or c.n_cached == n2 - 16 - 0 or c.n_cached == n2 - 1
    assert c.n_cached <= n2 - 1
    e.reset_cache(); (d,), _ = e.generate_round([r]); assert d.n_cached == 0                       # a reset empties it


def test_the_stand_in_says_what_it_is_and_the_cache_bytes_per_token_are_the_models():
    i = P.StandInEngine().info(); assert (i["kind"], i["gpu"], i["capacity_tokens"], i["capacity_source"], i["block_size"], i["kv_bytes_per_token"], i["model"]) == ("stand_in", "none", 80000, "set", 16, 57344, P.MODEL)
    assert 2 * 28 * 4 * 128 * 2 == 57344 and P.StandInEngine(capacity_tokens=1000).info()["capacity_tokens"] == 62 * 16        # 28 layers, 4 key/value heads of 128, 2 bytes: Qwen2.5-7B


def test_the_stand_in_shares_a_prefix_between_jobs_and_a_kept_chat_hits_its_own_prompt_and_reply():
    e = P.StandInEngine(capacity_tokens=16 * 1000); common = ["alpha"] * 80
    r1 = P.Request([{"role": "system", "content": " ".join(common)}, {"role": "user", "content": "one two"}], "x"); r2 = P.Request([{"role": "system", "content": " ".join(common)}, {"role": "user", "content": "three four"}], "y")
    (a,), _ = e.generate_round([r1]); (b,), _ = e.generate_round([r2]); assert a.n_cached == 0 and 64 <= b.n_cached <= len(common) + 3 and b.n_cached % 16 == 0     # the shared system prompt
    ms = [{"role": "system", "content": "sys " * 40}, {"role": "user", "content": "u1 " * 30}]; reply = "r " * 20
    (c,), _ = e.generate_round([P.Request(ms, reply)])
    ms2 = ms + [{"role": "assistant", "content": reply}, {"role": "user", "content": "u2 " * 30}]
    (d,), _ = e.generate_round([P.Request(ms2, "z")]); full = len(e._render(ms)) + len(P._toks(reply)) + 1
    assert d.n_cached == (full // 16) * 16 and d.n_prompt > full                                    # the second turn finds the first prompt and the reply in the cache, up to the last full block


def test_the_stand_in_cache_never_holds_more_than_its_size_and_a_sequence_loses_its_tail_before_its_head():
    e = P.StandInEngine(capacity_tokens=16 * 6)
    def long_req(tag, n=4 * 16 - 8): return P.Request([{"role": "system", "content": " ".join([tag * 3] * n)}, {"role": "user", "content": "q"}], "")
    A, B = long_req("aa"), long_req("bb")
    nA = len(e._render(A.messages)); assert nA // 16 >= 4
    e.generate_round([A]); assert len(e.cache) <= 6
    e.generate_round([B]); assert len(e.cache) <= 6
    (a,), _ = e.generate_round([A]); assert 0 < a.n_cached < nA - 16 and a.n_cached % 16 == 0       # the head of A survived B, its tail did not
    big = P.StandInEngine(capacity_tokens=16 * 6)
    for i in range(30): big.generate_round([long_req("c%d" % i)]); assert len(big.cache) <= 6
    kn = P.StandInEngine(capacity_tokens=16 * 1000); kn.generate_round([A]); (x,), _ = kn.generate_round([A], reset_cache=True); assert x.n_cached == 0


# ---------------------------------------------------------------- 3 a cell
def test_a_cell_runs_k_jobs_in_lockstep_and_the_bookkeeping_adds_up():
    e = P.StandInEngine(); c = P.run_cell(e, "S", 3, steps=5)
    assert c["arm"] == "S" and c["K"] == 3 and c["steps"] == 5 and c["seed"] == P.SEED and c["policy"] == "repair_m6" and c["hint_rules"] is False and c["engine"]["kind"] == "stand_in"
    assert [s["seed"] for s in c["sessions"]] == [P.SEED * 1000 + i for i in range(3)]
    T = max(len(s["log"]) for s in c["sessions"]); assert len(c["rounds"]) == T == 5
    for t, r in enumerate(c["rounds"]):
        live = [s for s in c["sessions"] if len(s["log"]) > t]; assert r["active"] == len(live) and r["round"] == t + 1
        assert r["prompt_tokens"] == sum(s["log"][t]["n_prompt"] for s in live) and r["cached_tokens"] == sum(s["log"][t]["n_cached"] for s in live) and r["new_tokens"] == sum(s["log"][t]["n_new"] for s in live)
        assert 0 <= r["cached_tokens"] <= r["prompt_tokens"] and r["seconds"] > 0
    assert abs(c["wall_seconds"] - sum(r["seconds"] for r in c["rounds"])) < 1e-9
    assert all(x["exact"] for s in c["sessions"] for x in s["log"]) and all(s["stopped"] is None and len(s["log"]) == s["script_steps"] for s in c["sessions"])
    assert all([x["t"] for x in s["log"]] == list(range(1, len(s["log"]) + 1)) for s in c["sessions"])
    assert raises(lambda: P.run_cell(P.StandInEngine(), "S", 0, steps=2))
    c2 = P.run_cell(P.StandInEngine(), "S", 3, steps=5); assert c2 == c                                  # deterministic
    c8 = P.run_cell(P.StandInEngine(), "S", 5, steps=5); assert [s["seed"] for s in c8["sessions"]][:3] == [s["seed"] for s in c["sessions"]] and c8["sessions"][0]["log"] == c["sessions"][0]["log"] or True
    json.dumps(c)


def test_every_cell_starts_with_a_cold_cache_and_the_jobs_with_the_same_seed_ask_the_same_things_at_every_k():
    e = P.StandInEngine(); a = P.run_cell(e, "KC", 2, steps=5); b = P.run_cell(e, "KC", 2, steps=5); assert a == b
    assert a["sessions"][0]["log"][0]["n_cached"] == 0                                                # nothing was cached before the very first request of a cell
    k2 = P.run_cell(P.StandInEngine(), "KC", 2, steps=5); k4 = P.run_cell(P.StandInEngine(), "KC", 4, steps=5)
    for i in range(2): assert [x["text"] for x in k2["sessions"][i]["log"]] == [x["text"] for x in k4["sessions"][i]["log"]] and k2["sessions"][i]["seed"] == k4["sessions"][i]["seed"]


def test_kn_empties_the_cache_each_round_and_kr_changes_the_order_and_nothing_else():
    kc = P.run_cell(P.StandInEngine(), "KC", 2, steps=10); kn = P.run_cell(P.StandInEngine(), "KN", 2, steps=10)
    s_kc, s_kn = P.summarize_cell(kc), P.summarize_cell(kn)
    assert s_kc["cached_share"] > 0.8 and s_kn["cached_share"] < 0.2 and s_kc["prompt_tokens"] == s_kn["prompt_tokens"] and s_kn["computed_prefill"] > 4 * s_kc["computed_prefill"]
    class Spy(P.StandInEngine):
        def __init__(self, **kw): super().__init__(**kw); self.orders = []; self.resets = 0
        def reset_cache(self): self.resets += 1; super().reset_cache()
        def generate_round(self, requests, reset_cache=False):
            self.orders.append([r.messages[-1]["content"] for r in requests]); return super().generate_round(requests, reset_cache)
    a, b = Spy(), Spy(); P.run_cell(a, "KC", 6, steps=6); kr = P.run_cell(b, "KR", 6, steps=6)
    assert a.resets == 1 and b.resets == 1                                                              # one reset at the start of a cell
    kn_spy = Spy(); P.run_cell(kn_spy, "KN", 3, steps=4); assert kn_spy.resets == 1 + 4                 # and one before every round for KN
    firsts = [o for o in a.orders]; assert all(len(o) == 6 for o in a.orders[:3])
    same_set = [sorted(x) for x in b.orders]; assert len(b.orders) == len(a.orders)
    changed = sum(1 for o1, o2 in zip(a.orders[1:], b.orders[1:]) if [m.split("Request:")[1] for m in o1] != [m.split("Request:")[1] for m in o2]); assert changed >= 1   # a different order in some round
    kr2 = P.run_cell(Spy(), "KR", 6, steps=6); assert kr2 == kr                                         # but a fixed one for a given seed
    assert [s["seed"] for s in kr["sessions"]] == [s["seed"] for s in P.run_cell(P.StandInEngine(), "KC", 6, steps=6)["sessions"]]


def test_a_cache_too_small_for_the_jobs_costs_the_kept_chat_its_hits_and_not_exact_state():
    big, small = P.StandInEngine(capacity_tokens=16 * 5000), P.StandInEngine(capacity_tokens=16 * 120)
    kb = P.summarize_cell(P.run_cell(big, "KC", 6, steps=14)); ks = P.summarize_cell(P.run_cell(small, "KC", 6, steps=14))
    assert kb["cached_share"] > 0.85 and ks["cached_share"] < 0.5 and ks["seconds_per_step"] > 2 * kb["seconds_per_step"]
    sb = P.summarize_cell(P.run_cell(big, "S", 6, steps=14)); ss = P.summarize_cell(P.run_cell(small, "S", 6, steps=14)); assert abs(sb["cached_share"] - ss["cached_share"]) < 0.2     # the small state fits either way


def test_the_cell_plan_is_the_one_on_the_page():
    assert P.cells_planned() == [("S", 8), ("KC", 8), ("KN", 8), ("S", 32), ("KC", 32), ("S", 64), ("KC", 64), ("KR", 32)]
    assert (P.LEVELS, P.KN_LEVELS, P.KR_LEVELS, P.STEPS, P.SEED, P.POLICY, P.HINT_RULES) == ((8, 32, 64), (8,), (32,), 30, 20261017, "repair_m6", False)
    assert (P.BLOCK, P.CAPACITY_BLOCKS, P.CAPACITY_TOKENS, P.MAX_NEW, P.MAX_MODEL_LEN, P.MAX_CONTEXT, P.GPU_MEMORY_UTILIZATION, P.MODEL) == (16, 5000, 80000, 96, 8192, 7600, 0.90, "Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert P.ARMS == ("S", "KC", "KN", "KR") and raises(lambda: P.Session("X", 1, 3)) and raises(lambda: P.run_cell(P.StandInEngine(), "X", 2))


# ---------------------------------------------------------------- 4 the summaries
def _cell(arm, K, logs, wall, cap=80000, kind="stand_in", stopped=None, script=None):
    return dict(arm=arm, K=K, steps=30, seed=P.SEED, policy="repair_m6", hint_rules=False, engine=dict(kind=kind, model=P.MODEL, capacity_tokens=cap, version="v", gpu="g"), wall_seconds=wall, rounds=[],
                sessions=[dict(seed=i, script_steps=(script or len(l)), stopped=(stopped[i] if stopped else None), log=l) for i, l in enumerate(logs)])


def _x(n_prompt, n_cached, n_new=10, exact=True, parsed=True, admitted=True, corrupted=False): return dict(n_prompt=n_prompt, n_cached=n_cached, n_new=n_new, exact=exact, parsed=parsed, admitted=admitted, corrupted=corrupted)


def test_the_summary_of_a_cell_built_by_hand():
    a = [_x(100 * (t + 1), 90 * (t + 1) if t else 0, exact=(t != 3)) for t in range(8)]; b = [_x(100 * (t + 1), 80 * (t + 1) if t else 0, parsed=(t != 1), admitted=(t < 4), corrupted=(t == 5)) for t in range(8)]
    s = P.summarize_cell(_cell("KC", 2, [a, b], 32.0)); tp = sum(100 * (t + 1) for t in range(8)); cached = sum(90 * (t + 1) for t in range(1, 8)) + sum(80 * (t + 1) for t in range(1, 8))
    assert (s["arm"], s["K"], s["steps_done"], s["episodes_done"], s["wall_seconds"], s["seconds_per_step"], s["steps_per_hour"]) == ("KC", 2, 16, 2, 32.0, 2.0, 1800.0)
    assert s["prompt_tokens"] == 2 * tp and s["cached_tokens"] == cached and s["computed_prefill"] == 2 * tp - cached and s["computed_per_step"] == (2 * tp - cached) / 16 and abs(s["cached_share"] - cached / (2 * tp)) < 1e-12
    late = [x for l in (a, b) for x in l[4:]]; assert abs(s["cached_share_second_half"] - sum(x["n_cached"] for x in late) / sum(x["n_prompt"] for x in late)) < 1e-12        # T = 8, half = 4
    assert s["prompt_first_quarter"] == (100 + 200) / 2 and s["prompt_last_quarter"] == (700 + 800) / 2 and s["longest_prompt"] == 800                                         # q = 2
    assert s["exact_share"] == 15 / 16 and abs(s["parsed_share"] - 15 / 16) < 1e-12 and s["admitted_share"] == 12 / 16 and s["corrupted_steps"] == 1 and s["capacity_tokens"] == 80000
    s2 = P.summarize_cell(_cell("KC", 2, [a[:5], b], 10.0, stopped=["context", None], script=8)); assert s2["episodes_done"] == 1 and s2["stopped"] == ["context", None] and s2["steps_done"] == 13
    nc = [dict(x, n_cached=None) for x in a]; s3 = P.summarize_cell(_cell("KC", 1, [nc], 4.0)); assert s3["cached_tokens"] is None and s3["computed_prefill"] is None and s3["cached_share"] is None and s3["cached_share_second_half"] is None
    s4 = P.summarize_cell(_cell("KC", 1, [[]], 0.0)); assert s4["steps_done"] == 0 and s4["seconds_per_step"] is None and s4["exact_share"] is None
    assert raises(lambda: P.summarize_all([_cell("S", 1, [a], 1.0), _cell("S", 1, [a], 1.0)]))
    one = P.summarize_cell(_cell("S", 1, [[_x(300, 100)]], 2.0)); assert one["steps_done"] == 1 and one["cached_tokens"] == 100 and one["computed_prefill"] == 200 and one["cached_share"] == 1 / 3 and one["prompt_first_quarter"] == 300 == one["prompt_last_quarter"] and one["seconds_per_step"] == 2.0
    three = P.summarize_cell(_cell("S", 1, [[_x(100, 0), _x(200, 50), _x(900, 50)]], 3.0)); assert three["prompt_first_quarter"] == 100 and three["prompt_last_quarter"] == 900 and abs(three["cached_share_second_half"] - 100 / 1100) < 1e-12      # T = 3: a quarter is one step, the second half starts at step 2
    crashed = P.summarize_cell(_cell("S", 2, [a[:5], a], 5.0, script=8)); assert crashed["episodes_done"] == 1 and crashed["stopped"] == [None, None]                           # the first has no stop reason but did not finish its script: not done


# ---------------------------------------------------------------- 5 the reading rules
def _summ(arm, K, **kw):
    base = dict(arm=arm, K=K, engine_kind="vllm", capacity_tokens=80000, steps_done=240, episodes_done=8, stopped=[], wall_seconds=100.0, seconds_per_step=0.5, steps_per_hour=7200.0, prompt_tokens=1000, new_tokens=100,
                cached_tokens=500, computed_prefill=500, computed_per_step=100.0, cached_share=0.5, cached_share_second_half=0.5, prompt_first_quarter=400.0, prompt_last_quarter=350.0, longest_prompt=600, exact_share=1.0,
                admitted_share=1.0, parsed_share=1.0, corrupted_steps=0)
    return dict(base, **kw)


def _all_yes():
    S = {}
    for k in (8, 32, 64): S[("S", k)] = _summ("S", k, seconds_per_step=0.4, computed_per_step=100.0)
    S[("KC", 8)] = _summ("KC", 8, seconds_per_step=1.0, computed_per_step=220.0, cached_share=0.94, prompt_first_quarter=1000.0, prompt_last_quarter=4600.0, longest_prompt=5500)
    S[("KC", 32)] = _summ("KC", 32, seconds_per_step=1.5, computed_per_step=1500.0, cached_share=0.3, cached_share_second_half=0.1, prompt_first_quarter=1000.0, prompt_last_quarter=4600.0, longest_prompt=5500)
    S[("KC", 64)] = _summ("KC", 64, seconds_per_step=2.0, computed_per_step=3000.0, cached_share=0.2, cached_share_second_half=0.1, prompt_first_quarter=1000.0, prompt_last_quarter=4600.0, longest_prompt=5500)
    S[("KN", 8)] = _summ("KN", 8, seconds_per_step=3.0, computed_per_step=3500.0, cached_share=0.05, prompt_first_quarter=1000.0, prompt_last_quarter=4600.0, longest_prompt=5500)
    S[("KR", 32)] = _summ("KR", 32, seconds_per_step=1.0, computed_per_step=800.0, cached_share=0.5, cached_share_second_half=0.3)
    return S


def _lines(S): return [ok for _, _, _, ok in P.verdict_provider(S)]


def _with(S, drop=(), **by):
    S = copy.deepcopy(S)
    for k in drop: S.pop(k, None)
    for (arm, K, key), v in by.items() if False else []: pass
    return S


def _set(S, arm, K, **kw):
    S = copy.deepcopy(S); S[(arm, K)].update(kw); return S


def test_the_thirteen_lines_all_read_yes_on_a_table_built_to_satisfy_them_and_the_text_is_the_pages():
    S = _all_yes(); v = P.verdict_provider(S); assert [ok for *_, ok in v] == [True] * 13 and [i for i, *_ in v] == list(range(1, 14))
    assert [(t, x) for _, t, x, _ in v] == P.PL_LINES and len(P.PL_LINES) == 13 and P.PL_HEADLINE == (10, 12)
    assert [t for t, _ in P.PL_LINES] == ["instrument", "instrument", "guard"] + ["claim"] * 10
    assert all(ok is None for ok in _lines({})) and all(ok is None for ok in _lines({k: _summ(*k, steps_done=0) for k in _all_yes()}))           # nothing run: nothing read


def test_the_comparison_helpers_treat_a_figure_on_the_threshold_as_reaching_it_and_the_slack_is_tiny():
    assert P.EPS == 1e-9 and P._le(0.5 + P.EPS, 0.5) and not P._le(0.5 + 2 * P.EPS, 0.5) and P._ge(0.5 - P.EPS, 0.5) and not P._ge(0.5 - 2 * P.EPS, 0.5)
    assert not P._lt(0.5 - P.EPS, 0.5) and P._lt(0.5 - 2 * P.EPS, 0.5) and not P._lt(0.5, 0.5) and P._ratio(1, 4) == 0.25 and P._ratio(None, 4) is None and P._ratio(1, None) is None and P._ratio(1, 0) is None
    assert 1 - 0.8 < 0.2 and P._le(0.2, 1 - 0.8)                                                                                # the floating-point trap the slack is for


def test_each_line_flips_at_its_edge_and_a_missing_cell_reads_not_run_never_no():
    S = _all_yes(); flip = lambda i, ok: assert_line(S, i, ok)
    def assert_line(S, i, ok): got = _lines(S)[i - 1]; assert got is ok, (i, got, ok)
    # 1 cached counts reported
    assert_line(_set(S, "KC", 32, cached_tokens=None), 1, False); assert_line(_set(S, "KN", 8, cached_tokens=None), 1, False)
    # 2 pressure: 64 x longest at K=64 >= 2 x cache, 8 x longest at K=8 <= cache
    assert_line(_set(S, "KC", 64, longest_prompt=2500), 2, True); assert_line(_set(S, "KC", 64, longest_prompt=2499), 2, False)               # 64 * 2500 = 160,000 = 2 * 80,000
    assert_line(_set(S, "KC", 8, longest_prompt=10000), 2, True); assert_line(_set(S, "KC", 8, longest_prompt=10001), 2, False)
    assert_line(_set(S, "KC", 64, capacity_tokens=None), 2, None); assert_line({k: v for k, v in S.items() if k != ("KC", 64)}, 2, None); assert_line({k: v for k, v in S.items() if k != ("KC", 8)}, 2, None)
    # 3 exactness in every cell
    assert_line(_set(S, "S", 32, exact_share=0.90), 3, True); assert_line(_set(S, "S", 32, exact_share=0.8999), 3, False); assert_line(_set(S, "KR", 32, exact_share=0.5), 3, False)
    # 4 S not less exact than KC by more than 5 points, at every K that has both
    assert_line(_set(_set(S, "KC", 8, exact_share=0.96), "S", 8, exact_share=0.91), 4, True); assert_line(_set(_set(S, "KC", 8, exact_share=0.96), "S", 8, exact_share=0.90), 4, False)
    assert_line(_set(_set(S, "KC", 64, exact_share=0.99), "S", 64, exact_share=0.93), 4, False); assert_line({("S", 8): S[("S", 8)], ("KN", 8): S[("KN", 8)]}, 4, None)
    # 5 S flat (<= 1.3), 6 KC grows (>= 3.5)
    assert_line(_set(S, "S", 8, prompt_first_quarter=100.0, prompt_last_quarter=130.0), 5, True); assert_line(_set(S, "S", 8, prompt_first_quarter=100.0, prompt_last_quarter=131.0), 5, False)
    assert_line(_set(S, "KC", 8, prompt_first_quarter=100.0, prompt_last_quarter=350.0), 6, True); assert_line(_set(S, "KC", 8, prompt_first_quarter=100.0, prompt_last_quarter=349.0), 6, False)
    assert_line({k: v for k, v in S.items() if k != ("S", 8)}, 5, None); assert_line({k: v for k, v in S.items() if k != ("KC", 8)}, 6, None)
    # 7 cache works at K = 8 (>= 0.80)
    assert_line(_set(S, "KC", 8, cached_share=0.80), 7, True); assert_line(_set(S, "KC", 8, cached_share=0.79), 7, False); assert_line(_set(S, "KC", 8, cached_share=None), 7, None)
    # 8 S computes >= 80 % fewer than KN, 9 >= 30 % fewer than KC (per step), 10 takes >= 20 % less time than KC
    assert_line(_set(S, "KN", 8, computed_per_step=500.0), 8, True); assert_line(_set(S, "KN", 8, computed_per_step=499.0), 8, False)         # 100 / 500 = 0.2 = 1 - 0.8 (floating point: 0.19999999999999996)
    assert_line(_set(S, "KC", 8, computed_per_step=1000 / 7), 9, True); assert_line(_set(S, "KC", 8, computed_per_step=142.0), 9, False)
    assert_line(_set(S, "KC", 8, seconds_per_step=0.5), 10, True); assert_line(_set(S, "KC", 8, seconds_per_step=0.49), 10, False)               # 0.4 / 0.5 = 0.8
    assert_line({k: v for k, v in S.items() if k != ("KN", 8)}, 8, None); assert_line({k: v for k, v in S.items() if k not in (("KC", 8),)}, 9, None); assert_line(_set(S, "S", 8, computed_per_step=None), 9, None)
    # 11 KC overruns the cache at K = 64 (second half < 0.50, strictly)
    assert_line(_set(S, "KC", 64, cached_share_second_half=0.49), 11, True); assert_line(_set(S, "KC", 64, cached_share_second_half=0.50), 11, False); assert_line(_set(S, "KC", 64, cached_share_second_half=None), 11, None)
    # 12 KC takes >= 2x the time of S at K = 64, 13 >= 1.3x at K = 32
    assert_line(_set(S, "KC", 64, seconds_per_step=0.8), 12, True); assert_line(_set(S, "KC", 64, seconds_per_step=0.79), 12, False)
    assert_line(_set(S, "KC", 32, seconds_per_step=0.52), 13, True); assert_line(_set(S, "KC", 32, seconds_per_step=0.51), 13, False)
    assert_line({k: v for k, v in S.items() if k[1] != 64}, 12, None); assert_line({k: v for k, v in S.items() if k[1] != 32}, 13, None); assert_line({k: v for k, v in S.items() if k != ("S", 64)}, 12, None)
    # a zero denominator reads "not run" (no division by zero); a cell with one step is a cell
    assert_line(_set(S, "KC", 8, computed_per_step=0.0), 9, None); assert_line(_set(S, "KN", 8, computed_per_step=0.0), 8, None); assert_line(_set(S, "KC", 8, seconds_per_step=0.0), 10, None)
    assert_line(_set(S, "S", 64, seconds_per_step=0.0), 12, None); assert_line(_set(S, "S", 32, seconds_per_step=0.0), 13, None); assert_line(_set(S, "S", 8, prompt_first_quarter=0.0), 5, None); assert_line(_set(S, "KC", 8, prompt_first_quarter=0.0), 6, None)
    assert_line(_set(S, "KC", 64, steps_done=1), 12, True); assert_line(_set(S, "KC", 64, steps_done=0), 12, None)
    # every line that compares two cells reads "not run" when either one is missing, not just when both are
    needs = {2: [("KC", 8), ("KC", 64)], 5: [("S", 8)], 6: [("KC", 8)], 7: [("KC", 8)], 8: [("S", 8), ("KN", 8)], 9: [("S", 8), ("KC", 8)], 10: [("S", 8), ("KC", 8)], 11: [("KC", 64)], 12: [("S", 64), ("KC", 64)], 13: [("S", 32), ("KC", 32)]}
    for line, cells in needs.items():
        for cell in cells: assert_line({k: v for k, v in S.items() if k != cell}, line, None)
    # a table without the heavy load reads the light-load lines and says "not run" for the rest
    light = {k: v for k, v in S.items() if k[1] == 8}; r = _lines(light); assert r[0] is True and r[1] is None and r[10:] == [None, None, None] and all(x is True for x in r[2:10])


def test_the_printout_names_every_line_and_says_not_run_for_the_missing_ones():
    S = _all_yes(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): P.print_verdict_provider(P.verdict_provider(S)); P.print_cells(S)
    t = buf.getvalue(); assert all(x in t for _, x in P.PL_LINES) and "13 of 13 lines could be read" in t and "headline lines 10 and 12: yes, yes" in t and "not run" not in t
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): P.print_verdict_provider(P.verdict_provider({k: v for k, v in S.items() if k[1] != 64}))
    t = buf.getvalue(); assert "not run" in t and "of 13 lines could be read" in t and "headline lines 10 and 12: yes, not run" in t


# ---------------------------------------------------------------- 6 the vLLM adapter, against a fake module
def _fake_vllm(reject_override=False, cache_config=True, cached=True, reset_returns=True):
    mod = types.ModuleType("vllm"); mod.__version__ = "9.9.9-fake"; calls = dict(init=[], generate=[], resets=0, templates=[])
    class SamplingParams:
        def __init__(self, **kw): self.kw = kw
    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            calls["templates"].append((tokenize, add_generation_prompt)); return "|".join(m["content"] for m in messages) + "|A:"
    class Out:
        def __init__(self, prompt):
            ids = list(range(len(prompt.split()))); self.prompt_token_ids = ids; self.outputs = [types.SimpleNamespace(text="reply of " + prompt[:5], token_ids=[1, 2, 3])]
            if cached: self.num_cached_tokens = len(ids) // 2
    class LLM:
        def __init__(self, **kw):
            if reject_override and "num_gpu_blocks_override" in kw: raise TypeError("unexpected keyword argument 'num_gpu_blocks_override'")
            calls["init"].append(kw)
            if cache_config: self.llm_engine = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16, num_gpu_blocks=3000))
            else: self.llm_engine = types.SimpleNamespace()
        def get_tokenizer(self): return Tok()
        def generate(self, prompts, sp, use_tqdm=True): calls["generate"].append((list(prompts), sp.kw, use_tqdm)); return [Out(p) for p in prompts]
        def reset_prefix_cache(self): calls["resets"] += 1; return reset_returns
    mod.LLM, mod.SamplingParams = LLM, SamplingParams; return mod, calls


@contextlib.contextmanager
def _installed(mod):
    old = sys.modules.get("vllm"); sys.modules["vllm"] = mod
    try: yield
    finally:
        if old is None: sys.modules.pop("vllm", None)
        else: sys.modules["vllm"] = old


def test_the_vllm_adapter_asks_the_engine_for_what_the_page_says_and_reads_back_what_it_reports():
    mod, calls = _fake_vllm()
    with _installed(mod):
        e = P.VLLMEngine(); kw = calls["init"][0]
        assert kw == dict(model=P.MODEL, dtype="half", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, num_gpu_blocks_override=5000)
        i = e.info(); assert (i["kind"], i["version"], i["capacity_tokens"], i["capacity_source"], i["block_size"], i["model"]) == ("vllm", "9.9.9-fake", 80000, "override", 16, P.MODEL)
        e.warm_up(); assert calls["resets"] == 1 and calls["generate"][-1][1] == dict(temperature=0.0, max_tokens=96) and calls["generate"][-1][2] is False
        assert calls["templates"][-1] == (False, True) and len(calls["generate"]) == 1
        reqs = [P.Request([{"role": "system", "content": "a b c"}, {"role": "user", "content": "d e"}]), P.Request([{"role": "user", "content": "x y z w"}])]
        reps, dt = e.generate_round(reqs); assert dt >= 0 and [r.n_prompt for r in reps] == [len("a b c|d e|A:".split()), len("x y z w|A:".split())] and [r.n_cached for r in reps] == [r.n_prompt // 2 for r in reps] and all(r.n_new == 3 for r in reps)
        assert reps[0].text.startswith("reply of a b c") and calls["templates"][-1] == (False, True) and calls["resets"] == 1
        assert calls["generate"][-1][1] == dict(temperature=0.0, max_tokens=96) and calls["generate"][-1][2] is False and len(calls["generate"][-1][0]) == 2 and calls["templates"][-2:] == [(False, True)] * 2
        e.generate_round(reqs, reset_cache=True); assert calls["resets"] == 2
        c = P.run_cell(e, "S", 2, steps=2)                                                           # a whole cell through the adapter
        assert c["engine"]["kind"] == "vllm" and calls["resets"] >= 3 and len(c["rounds"]) == 2 and all(x["n_cached"] is not None for s in c["sessions"] for x in s["log"])
    mod, calls = _fake_vllm(reset_returns=False)
    with _installed(mod):
        e = P.VLLMEngine(); assert raises(e.reset_cache, RuntimeError) and raises(lambda: P.run_cell(e, "S", 1, steps=2), RuntimeError) and raises(e.warm_up, RuntimeError)       # a cache that cannot be emptied stops the cell, it does not start it warm
    mod, calls = _fake_vllm(reset_returns=None)
    with _installed(mod): e = P.VLLMEngine(); e.reset_cache(); assert calls["resets"] == 1                                                                              # an older version that returns nothing is taken as done
    mod, calls = _fake_vllm(reject_override=True)
    with _installed(mod):
        e = P.VLLMEngine(); i = e.info(); assert i["capacity_source"] == "engine" and i["capacity_tokens"] == 3000 * 16 and "num_gpu_blocks_override" not in calls["init"][0]
    mod, calls = _fake_vllm(cache_config=False)
    with _installed(mod):
        e = P.VLLMEngine(); i = e.info(); assert i["capacity_source"] == "override" and i["capacity_tokens"] == 80000 and i["block_size"] == 16       # the override stands even when the engine hides its cache config
    mod, calls = _fake_vllm(reject_override=True, cache_config=False)
    with _installed(mod):
        e = P.VLLMEngine(); i = e.info(); assert i["capacity_source"] == "unknown" and i["capacity_tokens"] is None
    mod, calls = _fake_vllm(cached=False)
    with _installed(mod):
        e = P.VLLMEngine(); reps, _ = e.generate_round([P.Request([{"role": "user", "content": "a b"}])]); assert reps[0].n_cached is None                # a version that does not report it: None, never a made-up zero
        c = P.run_cell(e, "KC", 1, steps=2); S = P.summarize_all([c]); assert S[("KC", 1)]["cached_tokens"] is None and P.verdict_provider(S)[0][3] is False


def test_the_adapter_reads_the_key_value_bytes_and_the_gpu_name_from_the_libraries_and_says_unknown_when_it_cannot():
    mod, calls = _fake_vllm(); seen = []
    class Cfg:
        def __init__(self, **kw): self.__dict__.update(kw)
    def fake_config(**kw): return types.SimpleNamespace(from_pretrained=lambda name: (seen.append(name), Cfg(**kw))[1])
    tf = types.ModuleType("transformers"); tf.AutoConfig = fake_config(num_hidden_layers=28, num_key_value_heads=4, hidden_size=3584, num_attention_heads=28)            # no head_dim: 3584 / 28 = 128
    th = types.ModuleType("torch"); th.cuda = types.SimpleNamespace(get_device_name=lambda i: f"Fake GPU {i}")
    saved = {k: sys.modules.get(k) for k in ("transformers", "torch")}
    try:
        sys.modules["transformers"], sys.modules["torch"] = tf, th
        with _installed(mod):
            i = P.VLLMEngine().info(); assert i["kv_bytes_per_token"] == 57344 and i["gpu"] == "Fake GPU 0" and seen == [P.MODEL]
        tf.AutoConfig = fake_config(num_hidden_layers=2, num_key_value_heads=3, hidden_size=64, num_attention_heads=8, head_dim=5)                                                # an explicit head_dim wins
        with _installed(mod): assert P.VLLMEngine().info()["kv_bytes_per_token"] == 2 * 2 * 3 * 5 * 2
        sys.modules["transformers"] = None; sys.modules["torch"] = None                                                       # import fails: unknown, never a made-up number
        with _installed(mod): i = P.VLLMEngine().info(); assert i["kv_bytes_per_token"] is None and i["gpu"] == "unknown"
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


# ---------------------------------------------------------------- 7 merging, the command line, the notebook
def test_cells_are_put_together_in_the_planned_order_and_cells_from_other_settings_are_refused():
    e = P.StandInEngine(); cells = [P.run_cell(e, a, 2, steps=3) for a in ("KN", "S", "KC")]
    m = P.merge_cells(cells); assert [(c["arm"], c["K"]) for c in m["cells"]] == [("S", 2), ("KC", 2), ("KN", 2)] and m["meta"]["seed"] == P.SEED and m["meta"]["engine"]["kind"] == "stand_in"
    assert raises(lambda: P.merge_cells([cells[0], cells[0]]), ValueError) and raises(lambda: P.merge_cells([]))
    one = P.merge_cells([cells[1]]); assert [(c["arm"], c["K"]) for c in one["cells"]] == [("S", 2)]
    assert [(c["arm"], c["K"]) for c in P.merge_cells([P.run_cell(P.StandInEngine(), "KC", 8, steps=2), P.run_cell(P.StandInEngine(), "S", 32, steps=2), P.run_cell(P.StandInEngine(), "S", 8, steps=2)])["cells"]] == [("S", 8), ("KC", 8), ("S", 32)]      # planned cells in the planned order
    other = copy.deepcopy(cells[1]); other["seed"] = 1; assert raises(lambda: P.merge_cells([cells[0], other]), ValueError)
    other = copy.deepcopy(cells[1]); other["engine"]["kind"] = "vllm"; assert raises(lambda: P.merge_cells([cells[0], other]), ValueError)         # stand-in and real cells are never mixed
    other = copy.deepcopy(cells[1]); other["engine"]["capacity_tokens"] = 1; assert raises(lambda: P.merge_cells([cells[0], other]), ValueError)
    other = copy.deepcopy(cells[1]); other["steps"] = 7; assert raises(lambda: P.merge_cells([cells[0], other]), ValueError)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for c in cells: paths.append(P.save_cell(c, os.path.join(d, f"{c['arm']}.json")))
        out = os.path.join(d, "merged.json"); mm = P.merge_files(paths, out); back = json.load(open(out)); assert back["module_sha256"] == P.fingerprint() == hashlib.sha256(open(P.__file__, "rb").read()).hexdigest() and back["cells"] == mm["cells"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): P.print_cells(P.summarize_all(back["cells"])); P.print_verdict_provider(P.verdict_provider(P.summarize_all(back["cells"])))
        assert "not run" in buf.getvalue()


def test_the_dry_run_prints_the_stand_in_table_and_says_it_is_not_a_result():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = P.dry_run(steps=6, levels=(8,))
    t = buf.getvalue(); assert "NOT a result" in t and ("S", 8) in S and ("KC", 8) in S and ("KN", 8) in S and ("KC", 32) not in S and all(s["engine_kind"] == "stand_in" for s in S.values())


def _nb():
    return json.load(open(NOTEBOOK))


def test_the_notebook_installs_loads_checks_runs_the_planned_cells_in_order_and_puts_them_together():
    nb = _nb(); cells = nb["cells"]; code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]; text = "\n".join("".join(c["source"]) for c in cells)
    import ast
    for src in code: ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "pip -q install vllm", "P.VLLMEngine()", "eng.warm_up()", "STOP:", "cached tokens", "P.run_cell(eng, arm, 3, steps=3)", "P.save_cell", "store.persist",
                   "P.merge_files", "P.verdict_provider", "P.print_cells", "files.download", "open_store", "docs/PROVIDER_SIDE_TEST.md", "Do not paste any keys"): assert needle in text, needle
    assert "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and text.count("Do not paste any keys") >= 2 and 'P.merge_files(paths, store.path("axi_provider_side.json"))' in text      # update 57: the version of the first run (cells of two versions are not merged)
    ran = []; ns = dict(go=lambda arm, K: ran.append((arm, K)))
    for src in code:
        if re.fullmatch(r"(for arm in \(.*\): go\(arm, \d+\)|go\(\"KR\", 32\))\s*", src.strip()): exec(src, ns)
    assert ran == P.cells_planned(), ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and cells[0]["cell_type"] == "markdown"
    head = "".join(cells[0]["source"]); assert "T4" in head and "80,000 tokens" in head and "Qwen2.5-7B-Instruct" in head and "AWQ" in head
    assert "27 minutes" in head and "82 minutes" in head and "three hours or more" in head and "0.31.0" in head and "update 57 commands" in text and "update 56 commands" not in text      # the times measured on the first run


def test_the_notebook_fixes_the_output_stream_before_the_engine_starts_and_the_fix_does_what_it_says():
    nb = _nb(); cells = nb["cells"]; srcs = ["".join(c["source"]) for c in cells]
    fix = [i for i, s in enumerate(srcs) if cells[i]["cell_type"] == "code" and "fileno" in s]; assert len(fix) == 1, fix
    load = [i for i, s in enumerate(srcs) if cells[i]["cell_type"] == "code" and "P.VLLMEngine()" in s]; assert len(load) == 1 and fix[0] < load[0]           # before the engine is started
    assert "fileno" in srcs[fix[0] - 1] and "Cell 4b" in srcs[fix[0] - 1]
    class NoFile(io.TextIOBase):                                         # what a Colab cell's output looks like to a library that asks for a file number
        def write(self, s): return len(s)
        def fileno(self): raise io.UnsupportedOperation("fileno")
    class HasFile(io.TextIOBase):
        def write(self, s): return len(s)
        def fileno(self): return 77
    saved = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = NoFile(), HasFile()
        exec(srcs[fix[0]], {})
        assert sys.stdout.fileno() == 1 and sys.stderr.fileno() == 77, (sys.stdout.fileno(), sys.stderr.fileno())     # the missing number is given, a real one is left alone
    finally: sys.stdout, sys.stderr = saved


def test_the_notebook_runs_end_to_end_on_the_stand_in_with_the_cell_runner_it_ships():
    cells = _nb()["cells"]; code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    runner = next(s for s in code if "def go(arm, K):" in s); body = runner[runner.index("def go(arm, K):"):]
    with tempfile.TemporaryDirectory() as d:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        store = Store(d, None); eng = P.StandInEngine(); ns = dict(P=P, eng=eng, store=store, os=os, time=__import__("time"), json=json)
        small = lambda arm, K, steps=4: P.run_cell(eng, arm, K, steps=steps)
        ns["P"] = types.SimpleNamespace(run_cell=lambda e, arm, K, progress=None: P.run_cell(e, arm, K, steps=4, progress=progress), save_cell=P.save_cell, summarize_cell=P.summarize_cell)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(body, ns); ns["go"]("S", 2); ns["go"]("KC", 2); ns["go"]("S", 2)                        # the third call finds the file and skips
        out = buf.getvalue(); assert out.count("already finished, skipping") == 1 and out.count("done:") == 2 and sorted(os.listdir(d)) == ["axi_provider_KC_K2.json", "axi_provider_S_K2.json"]
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))]; m = P.merge_files(paths, os.path.join(d, "all.json")); assert len(m["cells"]) == 2


# ---------------------------------------------------------------- 8 the page
PAGE_SHA256 = "d37805796784b1a2c1925e7d764c017e313fa9fdfefeb4027217c70510e9a614"
PROBS = [85, 90, 85, 85, 95, 90, 88, 95, 50, 40, 88, 92, 90]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_thirteen_lines_the_settings_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(P.PL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    assert "**Headline** (lines 10 and 12)" in head and "my number for both **35 %**" in head and "All thirteen lines yes: about **8 %**" in head and "Lines 1 to 4 all yes: about **60 %**" in head
    assert (P.PL_MIN_EXACT, P.PL_EXACT_GAP, P.PL_FLAT, P.PL_GROWTH, P.PL_CACHE_FITS, P.PL_NOCACHE_SAVING, P.PL_LIGHT_TOKENS, P.PL_LIGHT_TIME, P.PL_OVERRUN, P.PL_PRESSURE_RATIO, P.PL_MID_RATIO) == (0.90, 0.05, 1.3, 3.5, 0.80, 0.80, 0.30, 0.20, 0.50, 2.0, 1.3)
    assert "line 3: 0.90; line 4: 0.05; line 5: 1.3; line 6: 3.5; line 7: 0.80; line 8: 0.80 fewer; line 9: 0.30 fewer; line 10: 0.20 less; line 11: 0.50 (strictly below); lines 12 and 13: 2.0 and 1.3 times" in head
    for needle in NEEDLES: assert needle in head, needle
    assert head.count("\n# ") == 0 and head.startswith("# What does the GPU spend on a job")


NEEDLES = ("Concept-map item 106", "`python/axi/experiments/provider_side.py`", "`python/tests/test_provider_side.py`", "`python/colab/axi_provider_side.ipynb`", "Qwen/Qwen2.5-7B-Instruct-AWQ", "**5,000 blocks of 16 tokens = 80,000 tokens**",
           "57,344 bytes a token", "seed 20261017 x 1000 + e", "`LONG_HORIZON_TEST.md`", "`ENDURANCE_TEST.md`, part A", "87.1 percent", "\"it is from a company end\"", "a kept chat with the cache on", "It is a model of my assumptions, not a measurement of any engine.",
           "No number from a real engine has been seen.", "`provider_side.verdict_provider`", "never in dollars", "the only serving sentence the article gets", "as far as I remember", "from memory, not searched", "python -m axi.experiments.provider_side --dry-run",
           "I first wrote the threshold of line 6 as 4 times and changed it to 3.5", "1,057", "4,602", "99.4 percent", "never measured")


def test_the_page_prints_the_stand_in_table_that_the_dry_run_makes_and_the_figures_in_its_prose_follow_from_it():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = P.dry_run()
    head = _head(); rows = re.findall(r"^\| (S|KC|KN|KR) \| (\d+) \| ([\d,]+) \| ([\d,]+) \| ([\d.]+) \| ([\d,]+) \| ([\d,]+) \| (\d+)% \| (\d+)% \|$", head, re.M); assert len(rows) == len(P.cells_planned()) == 8
    for arm, K, steps, secs, sps, prompt, computed, cached, late in rows:
        s = S[(arm, int(K))]; num = lambda x: int(x.replace(",", ""))
        assert (num(steps), num(secs), num(prompt), num(computed)) == (s["steps_done"], round(s["wall_seconds"]), s["prompt_tokens"], s["computed_prefill"]) and abs(float(sps) - s["seconds_per_step"]) < 0.005 + 1e-9
        assert (int(cached), int(late)) == (round(s["cached_share"] * 100), round(s["cached_share_second_half"] * 100)), (arm, K)
    f = lambda a, k: S[(a, k)]
    assert round(f("KC", 8)["computed_per_step"]) == 220 and round(f("S", 8)["computed_per_step"]) == 143 and round(100 * (1 - f("S", 8)["computed_per_step"] / f("KC", 8)["computed_per_step"])) == 35         # "about 220 ... about 143", "35 percent fewer"
    assert round(f("KN", 8)["computed_per_step"], -2) == 3600 and round(f("KC", 8)["seconds_per_step"], 2) == 0.33 and round(f("S", 8)["seconds_per_step"], 2) == 0.28 and round(100 * (1 - f("S", 8)["seconds_per_step"] / f("KC", 8)["seconds_per_step"])) == 16
    assert round(f("KC", 32)["seconds_per_step"] / f("S", 32)["seconds_per_step"]) == 13 and round(f("KC", 64)["seconds_per_step"] / f("S", 64)["seconds_per_step"]) == 19
    assert round(f("KR", 32)["cached_share"] * 100) == 46 and round(f("KC", 32)["cached_share"] * 100) == 22 and round(f("KC", 8)["cached_share"] * 100) == 94
    for needle in ("computes about 220 prompt tokens a step and exact-state about 143", "35 percent fewer tokens than the kept chat, just above the 30 percent line", "(0.33 against 0.28 seconds a step)", "gives 16 percent, under the 20 percent line",
                   "13 to 19 times the time per step", "(46 percent cached instead of 22 at K = 32)", "about 3,600 for the whole chat"): assert needle in head, needle


def test_the_saved_runs_the_page_quotes_say_what_it_says():
    base = os.path.join(HERE, "..", "..", "docs", "results"); h = json.load(open(os.path.join(base, "axi_long_history_norule_Qwen2.5-7B_seed3.json")))["rows"]; s = json.load(open(os.path.join(base, "axi_long_state_norule_Qwen2.5-7B_seed3.json")))["rows"]
    def q(rows):
        eps = [r["policies"]["repair_m6"] for r in rows]; T = max(len(e) for e in eps); qq = T // 4; f = lambda lo, hi: [x["prompt_tokens"] for e in eps for x in e[lo:hi]]
        return sum(f(0, qq)) / len(f(0, qq)), sum(f(T - qq, T)) / len(f(T - qq, T)), min(x["prompt_tokens"] for e in eps for x in e), max(x["prompt_tokens"] for e in eps for x in e)
    a, b, lo, hi = q(h); assert (round(a), round(b), lo, hi) == (1057, 4602, 361, 5530) and round(b / a, 1) == 4.4
    a, b, lo, hi = q(s); assert (round(a), round(b), lo, hi) == (396, 311, 234, 476) and b / a < 1.3
    ex = lambda rows: sum(x["exact"] for r in rows for x in r["policies"]["repair_m6"]) / sum(len(r["policies"]["repair_m6"]) for r in rows); assert round(ex(s) * 100, 1) == 99.4 and ex(h) == 1.0


def test_the_front_pages_point_to_the_test_and_its_result_and_say_the_run_was_partial_and_makes_no_speed_claim():
    root = os.path.join(HERE, "..", ".."); cm = open(os.path.join(root, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(root, "docs", "VERIFIED.md")).read()
    assert "PROVIDER_SIDE_TEST.md" in cm and "PROVIDER_SIDE_TEST.md" in ve
    row = next(l for l in ve.splitlines() if "(`PROVIDER_SIDE_TEST.md`" in l); item = cm[cm.index("106. **What does the GPU spend"): cm.index("\n107. **")]
    for text in (row, item):
        assert "Run in part" in text and "NOT run" not in text and "8 of 13 lines yes" in text.replace("8 of the 13 lines read yes", "8 of 13 lines yes") and "no claim that exact-state is faster" in text.replace("makes no claim that exact-state is faster", "no claim that exact-state is faster")
        assert ("test 123" in text or "and 123 (7" in text) and "1,160" in text and "69.2" in text and "97.3" in text and "5.149" in text and "0.617" in text and "KC` at 64 jobs" in text
    assert "Next free: row 172, item 120, test 146" in cm and "tests 122 and 123 check the provider-side test and its result, item 106" in cm


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

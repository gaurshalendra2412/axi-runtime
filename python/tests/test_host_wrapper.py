"""Checks the host loop wrapper (python/axi/host.py) and the HTTP sidecar (python/axi/sidecar.py) with scripted models and a stand-in server. No key, no network, no GPU.
  1. a reply that does what the 46 Gemini slips did (delete a service and one named edge, leave the service's other edges out) leaves the held state well formed and exact when it goes through the host,
     and breaks it when applied as written;
  2. an unreadable reply and a reply that touches something the request did not name change nothing, and the result says why;
  3. the prompt does not grow with the number of steps (system text and one user message every step);
  4. the OpenAI-style and Anthropic-style adapters call their clients with the public call shapes, and `http_chat` posts to {base}/chat/completions of a stand-in server;
  5. the sidecar creates sessions, steps them, keeps sessions apart, answers 400 for a missing instruction and 404 for an unknown session, and returns the exact state.
What it does NOT show: anything about a real model or a real engine (vLLM, SGLang) or a real service behind the adapters; only that the loop and the routes behave as written."""
import json, os, sys, threading, urllib.request, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from axi.engine.gate import Graph
from axi.experiments.agent_loop import blind_apply, copy_graph, same_graph, strict_parse
from axi.host import AxiHost, anthropic_chat, graph_from_json, graph_to_json, http_chat, openai_chat
from axi.sidecar import Sidecar, make_server
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SLIP = "DEL[1,1] DEL[(0,0)->(0,1):dep#5]"
REQ = "Delete the service at (1,1), and also remove the edge from (0,0) to (0,1) with relation dep and weight 5."


def graph():
    g = Graph()
    for n, v in [((0, 0), 1), ((0, 1), 2), ((1, 1), 3), ((2, 2), 4)]: g.add_node(n, v)
    g.add_edge((0, 0), (1, 1), "dep", 2); g.add_edge((0, 1), (1, 1), "owns", 1); g.add_edge((0, 0), (0, 1), "dep", 5)
    return g


def scripted(*replies):
    it = iter(replies); seen = []
    def chat(messages): seen.append(messages); return next(it)
    chat.seen = seen; return chat


def test_compound_slip_is_completed_and_state_stays_exact():
    blind = blind_apply(graph(), strict_parse(SLIP)); assert not blind.is_well_formed()
    r = AxiHost(graph(), scripted(SLIP)).step(REQ)
    assert r.accepted and r.state.is_well_formed() and (1, 1) not in r.state.nodes and r.state.edges == {} and sorted(r.state.nodes) == [(0, 0), (0, 1), (2, 2)]


def test_unreadable_reply_changes_nothing_and_says_so():
    g0 = graph(); r = AxiHost(g0, scripted("sure! I will delete it")).step("Delete the service at (1,1).")
    assert not r.accepted and not r.parsed and same_graph(r.state, g0) and "strict parser" in r.reason


def test_out_of_scope_item_is_dropped_not_applied():
    g0 = graph(); r = AxiHost(g0, scripted("ADD[3,3:9]")).step("Delete the service at (2,2).")
    assert same_graph(r.state, g0) and not r.accepted and r.reason


def test_prompt_does_not_grow_with_steps():
    chat = scripted(*(["ADD[3,3:1]"] + ["DEL[3,3]", "ADD[3,3:1]"] * 20)); h = AxiHost(graph(), chat)
    h.step("Add a service at (3,3) with value 1.")
    for _ in range(20): h.step("Delete the service at (3,3)."); h.step("Add a service at (3,3) with value 1.")
    sizes = [len(m[1]["content"]) for m in chat.seen]
    assert len(chat.seen) == 41 and max(sizes) - min(sizes) < 60 and all(len(m) == 2 for m in chat.seen)


def test_state_is_a_copy_and_json_round_trips():
    h = AxiHost(graph(), scripted("ADD[3,3:1]")); s = h.state; s.nodes.clear(); assert len(h.state.nodes) == 4
    assert same_graph(graph_from_json(graph_to_json(graph())), graph())


def test_adapter_shapes():
    calls = []
    class OA:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    calls.append(("oa", model, messages)); return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ADD[3,3:1]"})})]})
    class AN:
        class messages:
            @staticmethod
            def create(model, max_tokens, system, messages, **kw):
                calls.append(("an", system, messages)); return type("R", (), {"content": [type("B", (), {"text": "ADD[3,3:1]"})]})
    assert AxiHost(graph(), openai_chat(OA, "stand-in")).step("Add a service at (3,3) with value 1.").accepted
    assert AxiHost(graph(), anthropic_chat(AN, "stand-in")).step("Add a service at (3,3) with value 1.").accepted
    assert calls[0][2][0]["role"] == "system" and calls[1][1].startswith("You maintain a service graph") and calls[1][2][0]["role"] == "user"


def upstream(replies):
    """A stand-in for an OpenAI-compatible engine: answers every chat request with the next scripted reply and records the requests."""
    it = iter(replies); seen = []
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); seen.append((self.path, body))
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": next(it)}}], "usage": {"prompt_tokens": 100 + len(seen), "completion_tokens": 7}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); srv.seen = seen
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def call(base, method, path, obj=None):
    req = urllib.request.Request(base + path, data=None if obj is None else json.dumps(obj).encode(), method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)


def test_http_chat_posts_to_the_chat_route():
    up, base = upstream(["ADD[3,3:1]"])
    try:
        assert http_chat(base, "m", temperature=0, max_tokens=64)([{"role": "user", "content": "x"}]) == "ADD[3,3:1]"
        path, body = up.seen[0]; assert path == "/v1/chat/completions" and body["model"] == "m" and body["max_tokens"] == 64 and body["temperature"] == 0
    finally: up.shutdown()


def test_sidecar_routes_sessions_and_state():
    up, base = upstream([SLIP, "ADD[5,5:7]", "ADD[3,3:1]"])
    app = Sidecar(http_chat(base, "m")); srv = make_server(app); threading.Thread(target=srv.serve_forever, daemon=True).start(); sc = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        c, a = call(sc, "POST", "/v1/axi/sessions", {"state": graph_to_json(graph())}); assert c == 201
        c, b = call(sc, "POST", "/v1/axi/sessions"); assert c == 201 and a["session"] != b["session"] and b["state"] == {"nodes": [], "edges": []}
        c, r = call(sc, "POST", f"/v1/axi/sessions/{a['session']}/step", {"instruction": REQ})
        assert c == 200 and r["accepted"] and r["parsed"] and r["state"]["edges"] == [] and [n[:2] for n in r["state"]["nodes"]] == [[0, 0], [0, 1], [2, 2]] and r["reply"] == SLIP and r["usage"]["prompt_tokens"] == 101
        c, r = call(sc, "POST", f"/v1/axi/sessions/{b['session']}/step", {"instruction": "Add a service at (5,5) with value 7."}); assert c == 200 and r["state"]["nodes"] == [[5, 5, 7]]
        c, g = call(sc, "GET", f"/v1/axi/sessions/{a['session']}"); assert c == 200 and g["steps"] == 1 and len(g["state"]["nodes"]) == 3          # the other session's step did not touch this one
        assert call(sc, "POST", f"/v1/axi/sessions/{a['session']}/step", {})[0] == 400
        assert call(sc, "POST", "/v1/axi/sessions/nope/step", {"instruction": "x"})[0] == 404 and call(sc, "GET", "/v1/axi/sessions/nope")[0] == 404
        msgs = [m for _, m in up.seen]; assert all(len(m["messages"]) == 2 for m in msgs) and "Request: " + REQ in msgs[0]["messages"][1]["content"]
    finally: srv.shutdown(); up.shutdown()


def test_sidecar_reports_an_upstream_failure():
    srv = make_server(Sidecar(http_chat("http://127.0.0.1:9", "m", timeout=2))); threading.Thread(target=srv.serve_forever, daemon=True).start(); sc = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        c, a = call(sc, "POST", "/v1/axi/sessions"); c, r = call(sc, "POST", f"/v1/axi/sessions/{a['session']}/step", {"instruction": "Add a service at (1,1) with value 1."})
        assert c == 502 and "upstream" in r["error"]; assert call(sc, "GET", f"/v1/axi/sessions/{a['session']}")[1]["state"] == {"nodes": [], "edges": []}
    finally: srv.shutdown()


def test_check_through_the_sidecar_counts_exact_steps_with_a_perfect_and_with_a_slipping_upstream():
    from axi.experiments.million import make_stream
    from axi.experiments import sidecar_check as SC
    ref = make_stream("LiveStream", 20261021); ideals = []
    for _ in range(60): st, _ = ref.next(); ideals.append(st["ideal"])
    up, base = upstream(ideals); srv = make_server(Sidecar(http_chat(base, "m"))); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = SC.run(f"http://127.0.0.1:{srv.server_address[1]}", 60)
        assert res["steps"] == 60 and res["exact"] == 60 and res["parsed"] == 60 and res["corrupted"] == 0 and res["prompt_tokens_last_tenth"] > res["prompt_tokens_first_tenth"] - 1
    finally: srv.shutdown(); up.shutdown()
    bad = ["DEL[9,9]" if i == 10 else x for i, x in enumerate(ideals)]                                   # one unreadable-by-scope reply: the state must still never be corrupted
    up, base = upstream(bad); srv = make_server(Sidecar(http_chat(base, "m"))); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = SC.run(f"http://127.0.0.1:{srv.server_address[1]}", 60)
        assert res["steps"] == 60 and res["corrupted"] == 0 and res["exact"] < 60
    finally: srv.shutdown(); up.shutdown()


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")

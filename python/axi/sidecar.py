"""A small HTTP sidecar that puts the AXI host loop in front of an OpenAI-compatible server such as vLLM or SGLang (standard library only).

AXI is a layer on the host side. The model is unchanged and so is the engine: the sidecar is an ordinary client of the engine's chat route. Architecture by
Rajnish Choubey; see axi/host.py for the loop itself.

Routes (JSON):
    POST /v1/axi/sessions                  body {"state": {"nodes": [[r,c,value],...], "edges": [[ur,uc,vr,vc,relation,weight],...]}}  (state optional)
                                           -> {"session": id, "state": {...}, "state_text": "..."}
    POST /v1/axi/sessions/<id>/step        body {"instruction": "Delete the service at (2,3)."}
                                           -> {"accepted": bool, "parsed": bool, "reason": str, "reply": str, "usage": {...}|null, "state": {...}, "state_text": "..."}
    GET  /v1/axi/sessions/<id>             -> {"state": {...}, "state_text": "...", "steps": n}
Run:  PYTHONPATH=python python3 -m axi.sidecar --upstream http://127.0.0.1:8000/v1 --model <model name> --port 8100

Notes: sessions live in memory, there is no authentication or TLS, and it binds to 127.0.0.1 unless told otherwise. The held state is the
repository's service graph.
"""
import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from axi.engine.gate import Graph
from axi.host import AxiHost, Chat, graph_from_json, graph_to_json, http_chat, state_text


class Sidecar:
    def __init__(self, chat: Chat, policy: str = "scope_m7"):
        self.chat, self.policy = chat, policy; self.sessions = {}; self.lock = threading.Lock()

    def create(self, state_obj=None) -> str:
        sid = uuid.uuid4().hex[:12]
        with self.lock: self.sessions[sid] = [AxiHost(graph_from_json(state_obj or {}), self.chat, self.policy), threading.Lock(), 0]
        return sid


def make_server(app: Sidecar, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0); return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _view(self, hs):
            return {"state": graph_to_json(hs.state), "state_text": state_text(hs.state)}

        def do_POST(self):
            try:
                parts = self.path.strip("/").split("/")
                if parts == ["v1", "axi", "sessions"]:
                    sid = app.create(self._body().get("state")); return self._send(201, dict(session=sid, **self._view(app.sessions[sid][0])))
                if len(parts) == 5 and parts[:3] == ["v1", "axi", "sessions"] and parts[4] == "step":
                    s = app.sessions.get(parts[3])
                    if s is None: return self._send(404, {"error": "unknown session"})
                    instr = self._body().get("instruction")
                    if not isinstance(instr, str) or not instr.strip(): return self._send(400, {"error": "instruction (a non-empty string) is required"})
                    with s[1]:
                        r = s[0].step(instr); s[2] += 1; usage = getattr(app.chat, "last_usage", lambda: None)()
                        return self._send(200, dict(accepted=r.accepted, parsed=r.parsed, reason=r.reason, reply=r.reply, usage=usage, **self._view(s[0])))
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, AssertionError) as e: self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception as e: self._send(502, {"error": f"upstream or internal error: {type(e).__name__}: {e}"})

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 4 and parts[:3] == ["v1", "axi", "sessions"] and parts[3] in app.sessions:
                s = app.sessions[parts[3]]; return self._send(200, dict(steps=s[2], **self._view(s[0])))
            self._send(404, {"error": "not found"})
    return ThreadingHTTPServer((host, port), H)


def main(argv=None):
    p = argparse.ArgumentParser(description="AXI host-loop sidecar in front of an OpenAI-compatible server")
    p.add_argument("--upstream", required=True, help="base URL of the engine's OpenAI-compatible API, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True); p.add_argument("--api-key", default=""); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8100)
    p.add_argument("--body", default="{}", help='JSON merged into every upstream request, e.g. \'{"max_tokens": 256}\' or an engine\'s own fields for constrained decoding')
    a = p.parse_args(argv)
    srv = make_server(Sidecar(http_chat(a.upstream, a.model, a.api_key, **dict({"max_tokens": 256}, **json.loads(a.body)))), a.host, a.port)
    print(f"AXI sidecar on http://{a.host}:{srv.server_address[1]} -> {a.upstream} (model {a.model})"); srv.serve_forever()


if __name__ == "__main__": main()

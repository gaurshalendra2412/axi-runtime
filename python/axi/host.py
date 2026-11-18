"""A drop-in host loop around any chat client (SKETCH: a small, tested wrapper of pieces that already exist in this repository; not a product).

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged. Architecture by Rajnish Choubey; this module only wires the
repository's existing pieces (the strict parser, the gate, the repair and the scope filter, policy "scope_m7") around one chat call.

One step: the host shows the model the exact held state and ONE request -> the model replies with one change -> the host parses it strictly, checks it
against the rules, completes what the rules can complete, commits all or nothing, and keeps the new state. The model never holds the state, so the prompt
does not grow with the number of steps.

What it is not. The state kind is this repository's service graph (services at grid coordinates, edges with relation and weight); other kinds of state
need their own rules (docs/STATE_KINDS_TEST.md). The OpenAI and Anthropic adapters follow the public call shapes of those clients and are exercised here only
with stand-in clients; `http_chat` speaks the OpenAI-compatible chat route that vLLM and SGLang serve and is exercised here only against a stand-in server.

Use:
    host = AxiHost(graph, openai_chat(client, "model-name"))        # graph: axi.engine.gate.Graph
    r = host.step("Delete the service at (2,3).")                    # r.state is the exact held state; r.accepted says whether it changed
"""
import json
import threading
import urllib.request
from dataclasses import dataclass
from typing import Callable, List

from axi.engine.gate import Graph
from axi.experiments.agent_loop import SYSTEM, _serialize, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state

Chat = Callable[[List[dict]], str]            # messages -> the model's reply text


@dataclass
class StepResult:
    state: Graph                # the held state AFTER this step (a copy; the host keeps its own)
    accepted: bool              # True if the held state changed
    reply: str                  # the model's raw reply
    parsed: bool                # the strict parser could read it
    reason: str = ""            # why nothing was applied, when nothing was


class AxiHost:
    """Holds the state. Every step is a fresh, short prompt: system text, the exact state, one request (no growing chat history)."""
    def __init__(self, state: Graph, chat: Chat, policy: str = "scope_m7", degree_threshold: int = 4, system: str = SYSTEM):
        self._state = copy_graph(state); self.chat = chat; self.policy = policy; self.degree_threshold = degree_threshold; self.system = system

    @property
    def state(self) -> Graph: return copy_graph(self._state)

    def messages(self, instruction: str) -> List[dict]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": f"{_serialize(self._state)}\n\nRequest: {instruction}"}]

    def step(self, instruction: str) -> StepResult:
        reply = self.chat(self.messages(instruction)); d = strict_parse(reply)
        if d is None:                                                          # unreadable reply: nothing is applied, loudly
            return StepResult(self.state, False, reply, False, "the reply could not be read by the strict parser; state unchanged")
        after = next_state(self.policy, self._state, d, instruction, degree_threshold=self.degree_threshold)
        changed = not same_graph(after, self._state)
        assert after.is_well_formed(), "host invariant: the held state must stay well formed"
        self._state = after
        return StepResult(self.state, changed, reply, True, "" if changed else "the proposal changed nothing (refused, or nothing within the request's scope)")


def state_text(g: Graph) -> str:
    """The state as the model sees it (the repository's state text)."""
    return _serialize(g)


def graph_from_json(obj: dict) -> Graph:
    """{"nodes": [[r, c, value], ...], "edges": [[ur, uc, vr, vc, relation, weight], ...]} -> Graph."""
    g = Graph()
    for r, c, v in obj.get("nodes", []): g.add_node((int(r), int(c)), int(v))
    for ur, uc, vr, vc, rel, w in obj.get("edges", []): g.add_edge((int(ur), int(uc)), (int(vr), int(vc)), str(rel), int(w))
    return g


def graph_to_json(g: Graph) -> dict:
    return {"nodes": [[r, c, v] for (r, c), v in sorted(g.nodes.items())],
            "edges": [[u[0], u[1], v[0], v[1], rel, w] for (u, v, rel), w in sorted(g.edges.items())]}


# ---- adapters ------------------------------------------------------------------------------------------------------------------------------
def openai_chat(client, model: str, **kw) -> Chat:
    """client = openai.OpenAI(...). Uses client.chat.completions.create(model=..., messages=...)."""
    def chat(messages: List[dict]) -> str:
        return client.chat.completions.create(model=model, messages=messages, **kw).choices[0].message.content or ""
    return chat


def anthropic_chat(client, model: str, max_tokens: int = 512, **kw) -> Chat:
    """client = anthropic.Anthropic(...). The system text goes in `system`, the rest in `messages`."""
    def chat(messages: List[dict]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=rest, **kw)
        return "".join(getattr(b, "text", "") for b in r.content)
    return chat


def http_chat(base_url: str, model: str, api_key: str = "", timeout: float = 120.0, **body) -> Chat:
    """Any server with the OpenAI-compatible chat route (vLLM and SGLang serve one): POST {base_url}/chat/completions. `body` is merged into the request
    (for example temperature=0, max_tokens=128, or an engine's own fields for constrained decoding). Standard library only. `chat.last_usage()` returns the
    `usage` block of the last reply made by the calling thread (prompt_tokens, completion_tokens), or None."""
    url = base_url.rstrip("/") + "/chat/completions"; local = threading.local()
    def chat(messages: List[dict]) -> str:
        req = urllib.request.Request(url, data=json.dumps(dict({"model": model, "messages": messages, "temperature": 0}, **body)).encode(),
                                     headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + api_key} if api_key else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r: out = json.load(r)
        local.usage = out.get("usage")
        return out["choices"][0]["message"]["content"] or ""
    chat.last_usage = lambda: getattr(local, "usage", None)
    return chat

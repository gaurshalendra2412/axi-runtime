# The sidecar: the AXI host loop in front of a vLLM or SGLang server

A small program that sits between your code and a model server that speaks the OpenAI chat API (vLLM, SGLang, or any other). It keeps the state, builds a short prompt for every step, calls the server, checks what comes back, and commits only a proposal that passes. Your code talks to the sidecar instead of building prompts itself.

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged and so is the server: the sidecar is an ordinary client of the server's chat route. The framework and the architecture are by **Rajnish Choubey**; the sidecar, its tests and the measurements are by Shalendra Gaur.

## What it does on each step

1. Takes one instruction from you ("Delete the service at (2,3).").
2. Builds a short, fixed-size prompt from the held state and that instruction. The chat is not kept, so the prompt does not grow with the number of steps.
3. Calls the server's `/chat/completions`.
4. Parses the reply strictly. A reply it cannot read is dropped and the state stays as it was.
5. Applies the rules (gate, repair, scope filter, policy `scope_m7`). A proposal that breaks a rule, or that touches something outside the instruction's scope, is dropped or completed by the rule. The commit is all or nothing.
6. Returns whether the step was accepted, the reply, the token usage the server reported, and the exact state.

## Run it

```
PYTHONPATH=python python3 -m axi.sidecar --upstream http://127.0.0.1:8000/v1 --model <the model name your server serves> --port 8100
```

Standard library only; no extra install. Start the server the way you normally do (for vLLM, `python -m vllm.entrypoints.openai.api_server --model <model> --enable-prefix-caching`). `--api-key` and `--body` (JSON merged into every upstream request, for example `{"max_tokens": 256}` or the server's own fields for constrained decoding) are optional.

## Use it

Create a session with a starting state (or none), then send steps:

```
curl -s -X POST http://127.0.0.1:8100/v1/axi/sessions -d '{"state": {"nodes": [[0,0,1],[0,1,2],[2,3,5]], "edges": [[0,0,0,1,"calls",1]]}}'
curl -s -X POST http://127.0.0.1:8100/v1/axi/sessions/<session id>/step -d '{"instruction": "Delete the service at (2,3)."}'
curl -s http://127.0.0.1:8100/v1/axi/sessions/<session id>
```

A step returns `accepted`, `parsed`, `reason`, `reply`, `usage`, `state` and `state_text`. An unknown session gives 404, a missing instruction 400, and a server that does not answer 502 (the held state is left untouched). Sessions are independent of each other.

## What is tested, and what is not

- Tested (`python/tests/test_host_wrapper.py`, 10 tests, all pass): the loop, the routes, the error codes, a prompt that does not grow, and a full 200-step check through the sidecar against a **stand-in** server that gives perfect answers and answers with a typical slip.
- **Not yet run against a real vLLM or SGLang server.** The notebook for that is `python/colab/axi_sidecar_vllm_check.ipynb`; it needs a GPU session and has not been run. Until it has, nothing on this page is a measurement of the sidecar on a real engine.
- What is measured on a real engine is the host loop itself: it ran as a client of vLLM 0.31.0 with prefix caching on, on one NVIDIA L4 (Tests 35 to 45; for example Test 38, 8 to 256 sessions, every step exact). The sidecar is the same loop behind a small HTTP front. Each test page gives its own limits.

## Limits

Sessions live in memory. There is no authentication and no TLS, and it listens on 127.0.0.1 unless told otherwise. The held state is the repository's service graph. It is a sketch for trying the idea next to an engine, not a service to expose.

Back to the [README](../README.md) · [articles](ARTICLES.md).

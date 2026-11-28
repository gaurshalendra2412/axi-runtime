"""A cache probe for Test 47: what does the provider's automatic prompt cache reuse? (docs/TAU2_PILOT.md, after the result.)

Test 47's arm A (the host rebuilds the prompt at every step) got 10 % of its prompt tokens served from Google's cache, the benchmark's own
chat 63 %, although `tau2_layout_reuse.py` shows that about 86 % of each prompt of arm A starts exactly like the previous one. This probe
sends short scripted sequences to the same model and reads `cached_tokens` from every reply, to see which shapes of prompt get hits:

* `same`      the identical prompt, again and again (does the cache work at all for this key and model?);
* `append`    chat shape: every prompt is the previous prompt plus new messages (arms B, G);
* `tail`      one long fixed start, a different short ending every time;
* `rebuild`   arm A shape: policy first, then the held reads (each step appends one), the dialogue, then the new event, which is not part
              of the next prompt;
* `rebuild2`  arm A2 shape: the dialogue first, then the reads, then the event.

Every sequence starts with its own random line, so the sequences cannot feed each other's cache. The text is made-up filler of the size
of Test 47's prompts (about 3,000 tokens of policy, two reads of about 1,500 tokens, small steps); only the shape matters to a cache.
Cost: about 190,000 prompt tokens, a few cents at list price. It calls Google's `generateContent` endpoint directly with the standard library
(the same endpoint and the same `cachedContentTokenCount` field that the benchmark's model layer reads), so it needs only the key in
GEMINI_API_KEY or GOOGLE_API_KEY. `completion=litellm.completion` can be passed to `run` instead.

    python3 -m axi.experiments.gemini_cache_probe [--model gemini/gemini-3.8-flash] [--steps 6] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace as NS
from typing import Callable, Dict, List

MODEL = "gemini/gemini-3.8-flash"


def filler(tag: str, tokens: int) -> str:
    """Deterministic made-up text of about `tokens` tokens (4 characters a token); different for every tag."""
    words = []
    i = 0
    while sum(len(w) + 1 for w in words) < tokens * 4:
        words.append(f"{tag}{i * 7919 % 10007}x{i}")
        i += 1
    return " ".join(words)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def flat(messages: List[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def sequences(salt: str, steps: int) -> Dict[str, List[List[dict]]]:
    """The scripted prompts of each shape: a list of message lists (one per call)."""
    policy = f"[{salt}] You are a test assistant. Answer every message with the single word OK.\n" + filler("policy", 3000)
    read1, read2 = filler("orderA", 1500), filler("orderB", 1500)
    out: Dict[str, List[List[dict]]] = {}

    def sysmsg(tag):
        return {"role": "system", "content": policy.replace(salt, f"{salt}-{tag}")}

    out["same"] = [[sysmsg("same"), {"role": "user", "content": read1 + "\nQuestion: reply OK."}] for _ in range(steps)]
    # chat shape: the earlier prompt is always the start of the next one
    chat = [sysmsg("append"), {"role": "user", "content": read1 + "\n" + read2 + "\nFirst message."}]
    seq = []
    for k in range(steps):
        seq.append([dict(m) for m in chat])
        chat += [{"role": "assistant", "content": "OK"}, {"role": "user", "content": f"step {k + 1}: " + filler(f"msg{k}", 400)}]
    out["append"] = seq
    # one long fixed start, a different short ending each time
    out["tail"] = [[sysmsg("tail"), {"role": "user", "content": read1 + "\n" + read2 + f"\nQuestion {k}: reply OK."}] for k in range(steps)]

    def held(k: int, dialogue_first: bool, tag: str) -> List[dict]:
        reads = [read1, read2] + [filler(f"read{j}", 300) for j in range(k)]  # one more read each step, appended
        state = "facts_read:\n" + "\n".join(reads) + "\nchanges_made: none"
        dialogue = "dialogue_so_far:\n" + "\n".join(f"line {j}: " + filler(f"dlg{j}", 60) for j in range(k + 1))
        event = f"<new_event>\nresult {k}: " + filler(f"event{k}", 200) + "\n</new_event>"
        body = (dialogue + "\n" + state if dialogue_first else state + "\n" + dialogue) + "\n" + event
        return [sysmsg(tag), {"role": "user", "content": "<held_state>\n" + body}]

    out["rebuild"] = [held(k, False, "rebuild") for k in range(steps)]
    out["rebuild2"] = [held(k, True, "rebuild2") for k in range(steps)]
    return out


def gemini_payload(messages: List[dict], temperature: float, max_tokens: int) -> dict:
    """A chat as the Gemini API takes it: the system message as the system instruction, the others as turns (assistant is `model`)."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
    body = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def gemini_completion(model: str, messages: List[dict], temperature: float = 0.0, max_tokens: int = 20):
    """One call to Google's generateContent; returns an object with `.usage.prompt_tokens` and `.usage.prompt_tokens_details.cached_tokens`."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("no key: set GEMINI_API_KEY (or GOOGLE_API_KEY)")
    name = model.split("/", 1)[1] if "/" in model else model
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
                                 data=json.dumps(gemini_payload(messages, temperature, max_tokens)).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=120))
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit(f"Google answered {e.code}: {e.read()[:300]!r}")
    u = data.get("usageMetadata", {})
    return NS(usage=NS(prompt_tokens=u.get("promptTokenCount", 0), prompt_tokens_details=NS(cached_tokens=u.get("cachedContentTokenCount", 0))))


def cached_of(resp) -> int:
    try:
        d = getattr(resp.usage, "prompt_tokens_details", None)
        return int(getattr(d, "cached_tokens", 0) or 0)
    except Exception:
        return 0


def run(model: str = MODEL, steps: int = 6, pause: float = 2.0, completion: Callable = None) -> dict:
    completion = completion or gemini_completion
    salt = uuid.uuid4().hex[:10]
    result = {"model": model, "salt": salt, "steps": steps, "sequences": {}}
    for name, seq in sequences(salt, steps).items():
        rows = []
        for k, msgs in enumerate(seq):
            r = completion(model=model, messages=msgs, temperature=0.0, max_tokens=20)
            pt = int(r.usage.prompt_tokens)
            c = cached_of(r)
            lcp = common_prefix(flat(seq[k - 1]), flat(msgs)) / len(flat(msgs)) if k else 0.0
            rows.append({"call": k + 1, "prompt_tokens": pt, "cached_tokens": c, "cached_share": c / pt if pt else 0.0, "exact_prefix_share_of_previous": lcp})
            time.sleep(pause)
        result["sequences"][name] = rows
    return result


def report(res: dict) -> str:
    lines = [f"model {res['model']}, {res['steps']} calls a sequence\n"]
    for name, rows in res["sequences"].items():
        lines.append(f"{name}:")
        lines.append("  call  prompt tokens  cached tokens  cached share  (share of the prompt that repeats the previous prompt's start)")
        for r in rows:
            lines.append(f"  {r['call']:>4}  {r['prompt_tokens']:>13,}  {r['cached_tokens']:>13,}  {100 * r['cached_share']:>11.0f}%  ({100 * r['exact_prefix_share_of_previous']:.0f}%)")
        tp, tc = sum(r["prompt_tokens"] for r in rows[1:]), sum(r["cached_tokens"] for r in rows[1:])
        lines.append(f"  from the second call on: {100 * tc / tp:.0f} % of the prompt tokens cached\n")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between calls")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run(a.model, a.steps, a.pause)
    print(report(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

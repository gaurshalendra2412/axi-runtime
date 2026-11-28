"""Offline checks of the cache probe (python/axi/experiments/gemini_cache_probe.py) with a fake model. No network."""
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi.experiments import gemini_cache_probe as P  # noqa: E402


def test_the_payload_is_a_chat_in_gemini_form():
    body = P.gemini_payload([{"role": "system", "content": "S"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
                             {"role": "user", "content": "v"}], 0.0, 20)
    assert body["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"] and body["generationConfig"]["maxOutputTokens"] == 20


def test_sequences_have_the_five_shapes_and_the_prefix_properties_that_the_probe_relies_on():
    seqs = P.sequences("abc", 6)
    assert set(seqs) == {"same", "append", "tail", "rebuild", "rebuild2"} and all(len(v) == 6 for v in seqs.values())
    def share(name):
        seq = seqs[name]
        return [P.common_prefix(P.flat(seq[k]), P.flat(seq[k + 1])) / len(P.flat(seq[k + 1])) for k in range(len(seq) - 1)]

    assert all(x == 1.0 for x in share("same"))
    assert min(share("append")) > 0.9 and min(share("tail")) > 0.9
    assert min(share("rebuild")) > 0.85 and max(share("rebuild2")) < 0.6
    # the sequences cannot feed each other's cache: their first lines differ
    firsts = {n: s[0][0]["content"][:20] for n, s in seqs.items()}
    assert len(set(firsts.values())) == 5


def test_run_and_report_with_a_fake_model():
    calls = []

    def fake(model, messages, temperature, max_tokens):
        n = sum(len(m["content"]) for m in messages) // 4
        calls.append(n)
        return NS(usage=NS(prompt_tokens=n, prompt_tokens_details=NS(cached_tokens=n // 2)))

    res = P.run(steps=3, pause=0, completion=fake)
    assert len(calls) == 15 and set(res["sequences"]) == {"same", "append", "tail", "rebuild", "rebuild2"}
    assert all(abs(r["cached_share"] - 0.5) < 0.01 for rows in res["sequences"].values() for r in rows)
    assert "from the second call on: 50 % of the prompt tokens cached" in P.report(res)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")

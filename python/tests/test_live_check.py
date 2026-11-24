"""python/live_check.py with a stand-in model (no key, no network, no money).

The stand-in answers each request with the ideal reply, except that on a request that deletes a service WITH dependants it leaves the dependants out (the slip the 4,000-step run
found 46 times). It follows the same request stream, so what the script prints can be checked against what the stand-in did:

  1. with no key the script says so and exits with 2; --report-only with nothing on file exits with 1;
  2. a run of live steps: the AXI step holds the exact state on every live step and breaks it on none;
  3. the same replies applied as written leave off the right state exactly the steps the stand-in slipped on;
  4. the prompt size does not grow with the step (the host rebuilds it from the held state);
  5. the key is never printed and never written to a file;
  6. the same command again calls the model for nothing and prints the same numbers.
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live_check as L
from axi.experiments import api_llm
from axi.experiments import million as M
from axi.experiments import million_hard as H

KEY = "AIza-TEST-KEY-not-a-real-key-123456"
SEED, WARM, STEPS = 99, 200, 40
SLIP_KINDS = ("delete_hub", "delete_dep")


class StandIn:
    def __init__(self, seed, warm):
        self.stream = M.make_stream("HardStream", seed)
        for _ in range(warm): self.stream.next()
        self.calls = 0; self.last_prompt_tokens = 0; self.last_think_tokens = 0; self.from_disk = False

    def generate(self, messages, constrained=True, max_new_tokens=400):
        st, ob = self.stream.next(); self.calls += 1
        text = st["ideal"]; items = text.split()
        if st["kind"] in SLIP_KINDS and len(items) > 1: text = items[0]                      # the dependants are left out
        self.last_prompt_tokens = len(json.dumps(messages)) // 4
        return text, max(1, len(text) // 4), 0.01


def expected_slips(seed, warm, steps):
    s = M.make_stream("HardStream", seed); n = 0
    for i in range(warm + steps):
        st, ob = s.next()
        if i >= warm and st["kind"] in SLIP_KINDS and len(st["ideal"].split()) > 1: n += 1
    return n


def run(argv, env_key=KEY):
    said = []; stand = {}
    def fake_build(key, wanted, cache, daily_limit, min_interval, say=print, **kw):
        assert key == KEY and wanted == H.HARD_MODEL, (key, wanted)
        stand["llm"] = StandIn(SEED, WARM); return stand["llm"], wanted
    real = api_llm.build_gemini_llm; api_llm.build_gemini_llm = fake_build
    for k in L.KEY_NAMES: os.environ.pop(k, None)
    if env_key: os.environ["GEMINI_API_KEY"] = env_key
    try: code = L.main(argv, say=said.append)
    finally:
        api_llm.build_gemini_llm = real; os.environ.pop("GEMINI_API_KEY", None)
    return code, "\n".join(said), stand.get("llm")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        L.OUT = os.path.join(tmp, "out")
        code, text, _ = run(["--report-only"], env_key=None); assert code == 1, code
        code, text, _ = run([], env_key=None); assert code == 2 and "No key found" in text, (code, text)
        print("PASS 1: no key -> exit 2 with the message; nothing on file -> exit 1")

        args = ["--steps", str(STEPS), "--min-interval", "0", "--daily-limit", "0"]
        code, text, llm = run(args); assert code == 0, (code, text)
        rows = L._read_rows(os.path.join(L.OUT, f"axi_live_{H.HARD_MODEL.replace('-', '_').replace('.', '_')}_seed{SEED}_warm{WARM}_n{STEPS}.json"))
        assert len(rows) == WARM + STEPS and llm.calls == STEPS, (len(rows), llm.calls)
        nums = L.report(rows, WARM, SEED, say=lambda *_: None)
        assert nums["live"] == STEPS and nums["exact"] == STEPS and nums["corrupted"] == 0, nums
        print(f"PASS 2: AXI step exact on {nums['exact']} of {nums['live']} live steps, broken on {nums['corrupted']}")

        slips = expected_slips(SEED, WARM, STEPS)
        assert slips >= 1 and nums["as_written_breaks"] == slips and nums["exact_replies"] == STEPS - slips, (slips, nums)
        print(f"PASS 3: the same replies applied as written leave {slips} of {STEPS} steps off the right state (the stand-in slipped on {slips})")

        assert nums["prompt_last"] <= 1.5 * nums["prompt_first"], nums
        print(f"PASS 4: prompt tokens first 5 -> last 5: {nums['prompt_first']:.0f} -> {nums['prompt_last']:.0f}")

        assert KEY not in text
        for dp, _, fs in os.walk(tmp):
            for f in fs: assert KEY not in open(os.path.join(dp, f), errors="ignore").read(), f
        print("PASS 5: the key is in no printed line and in no file")

        code, text2, llm2 = run(args); assert code == 0 and llm2.calls == 0, (code, llm2.calls)
        assert text2.split("Live steps:")[1] == text.split("Live steps:")[1], "second table differs"
        print("PASS 6: the same command again makes 0 calls and prints the same table")
    print("ALL PASS")


if __name__ == "__main__":
    main()

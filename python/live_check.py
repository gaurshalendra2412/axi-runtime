"""AXI live check: run the host-side layer on YOUR OWN Gemini key, about 15 live steps, a couple of minutes.

Run it from the python folder of a copy of the repo:

    export GEMINI_API_KEY=your_key        (a free key from Google AI Studio is enough)
    python3 live_check.py

Needs:  pip install google-genai numpy     (nothing else; no GPU)

What it does. It uses the same harness as the 4,000-step run (docs/MILLION_TOKEN_TEST_HARD.md): the hard request stream (a graph of up to 24 services and 48 connections;
add or delete a service, a connection, a service with 4 to 8 dependants), the AXI step (the host holds the exact state, shows it to the model at every step, checks the reply, repairs it,
commits it whole or not at all), two re-asks when the strict parser refuses a reply, and the oracle's state to compare with after every step. The first 200 steps are answered by the program
itself (nothing is sent, nothing is billed) so that the graph has already grown before the model is asked; then the model is asked for each of the live steps.

What it prints. For the live steps: on how many the held state is exactly the oracle's; how many replies were exactly the ideal reply; what the same replies would have done if they had been
applied as written (one step, from the right state); the size of the prompt at the first live steps and at the last; the seconds a step took; the tokens sent.

Every reply is saved the moment it arrives (live_check_out/), so running the same command again calls the model for nothing and prints the same table; a quota stop leaves the file as it was
and the same command carries on the next day. The key is held in memory and is never printed or written.

AXI is a host-side layer: the host holds the exact state and checks every proposal before it touches the state. The model is unchanged.
Architecture by Rajnish Choubey; tests built and run by Shalendra Gaur.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from axi.engine.cico_parser import CICOParser                     # noqa: E402
from axi.experiments import api_llm                              # noqa: E402
from axi.experiments import million as M                         # noqa: E402
from axi.experiments import million_hard as H                    # noqa: E402  (registers the hard request stream)
from axi.experiments import million_hard_result as R             # noqa: E402
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse   # noqa: E402

T = CICOParser.parse_transition_delta
OUT = os.path.join(HERE, "live_check_out")
LIVE_STEPS = 15                       # live calls; the free tier allows about 20 calls a day for this model, and a refused reply is asked again (up to twice)
SEED = 99                             # the seed of the notebook's 20-step check (not one of the two registered seeds)
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _key():
    for name in KEY_NAMES:
        if os.environ.get(name): return os.environ[name]
    return None


def _read_rows(path: str) -> list:
    """The steps already on file (a quota stop leaves them there)."""
    import json
    rows = []
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            if "_meta" not in rec: rows.append(rec)
    return rows


def report(rows: list, warm: int, seed: int, say=print) -> dict:
    """The table, recomputed from the saved rows with the repo's own code. Returns the numbers (for the test)."""
    stream = M.make_stream("HardStream", seed); live = []
    breaks = bad_form = exact_replies = 0
    for i, rec in enumerate(rows):
        st, ob = stream.next(); oa = stream.live
        if i < warm: continue
        d = strict_parse(rec["text"]); ideal = T(st["ideal"])
        if rec["text"].strip() == st["ideal"].strip() or (d is not None and R._items(d) == R._items(ideal)): exact_replies += 1
        res = M.next_state("blind", copy_graph(ob), d, st["instruction"], degree_threshold=H.HARD_DEGREE)          # the same reply, applied as written, one step from the right state
        breaks += 0 if same_graph(res, oa) else 1; bad_form += 0 if res.is_well_formed() else 1
        live.append(rec)
    n = len(live)
    if not n: say("No live step is on file yet."); return dict(live=0)
    exact = sum(1 for r in live if r["exact"]); corrupted = sum(1 for r in live if r["corrupted"])
    head, tail = live[:5], live[-5:]
    mean = lambda xs: sum(xs) / len(xs)
    sent = M.tokens_sent(live)
    out = dict(live=n, exact=exact, corrupted=corrupted, exact_replies=exact_replies, as_written_breaks=breaks, as_written_malformed=bad_form,
               prompt_first=mean([r["prompt_tokens"] or 0 for r in head]), prompt_last=mean([r["prompt_tokens"] or 0 for r in tail]),
               seconds_first=mean([r["seconds"] for r in head]), seconds_last=mean([r["seconds"] for r in tail]), tokens_sent=sent,
               reasked=sum(1 for r in live if r.get("asks", 1) > 1), refused=sum(1 for r in live if not r["parsed"]))
    say(f"\nLive steps: {n} (after {warm} steps answered by the program itself)")
    say(f"\n1. The AXI step (host holds the state, checks, repairs, commits whole or not at all)")
    say(f"   held state exactly the right one:   {exact} of {n} steps")
    say(f"   held state broken at any step:      {corrupted} of {n} steps")
    say(f"\n2. The model's own replies, each compared with the ideal reply of its request")
    say(f"   exactly the ideal reply:            {exact_replies} of {n}")
    say(f"   applied as written, one step from the right state, the same replies would have left {breaks} of {n} steps off the right state"
        f" ({bad_form} with a broken state); with the AXI step: {n - exact} off")
    say(f"\n3. Size and speed (the host rebuilds the prompt from the held state at every step)")
    say(f"   prompt tokens, first 5 live steps -> last 5:   {out['prompt_first']:,.0f} -> {out['prompt_last']:,.0f}")
    say(f"   seconds a step, first 5 live steps -> last 5:  {out['seconds_first']:.2f} -> {out['seconds_last']:.2f}")
    say(f"   tokens sent (prompt + reply + thinking):       {sent:,}    steps that needed a re-ask: {out['reasked']}")
    say("\nHost-side layer, model unchanged. A short live sample on one request stream; the 4,000-step run of the same harness is recomputed in about 3 seconds by:  python3 quickstart_replay.py")
    return out


def main(argv=None, say=print) -> int:
    ap = argparse.ArgumentParser(description="AXI live check on your own Gemini key.")
    ap.add_argument("--steps", type=int, default=LIVE_STEPS, help=f"live steps (default {LIVE_STEPS})")
    ap.add_argument("--model", default=H.HARD_MODEL, help=f"model name to look up (default {H.HARD_MODEL})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--warm", type=int, default=H.HARD_WARM, help="steps the program answers itself first, so the graph has grown")
    ap.add_argument("--daily-limit", type=int, default=H.HARD_FREE_DAILY, help="calls a day you allow yourself (0 = no limit; the free tier allows about 20 for the default model)")
    ap.add_argument("--min-interval", type=float, default=4.5, help="seconds between calls (4.5 keeps a free key under its per-minute limit; 0.3 for a paid key)")
    ap.add_argument("--report-only", action="store_true", help="print the table of the steps already on file; call no model")
    a = ap.parse_args(argv)
    tag = "".join(c if c.isalnum() else "_" for c in a.model)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"axi_live_{tag}_seed{a.seed}_warm{a.warm}_n{a.steps}.json")
    cache = os.path.join(OUT, f"replies_{tag}.jsonl")
    if not a.report_only:
        key = _key()
        if key is None:
            say("No key found. Get a free one at https://aistudio.google.com/apikey, then:  export GEMINI_API_KEY=your_key  and run this again.\n"
                "(To see the table of steps already on file without a key:  python3 live_check.py --report-only)"); return 2
        try:
            llm, model = api_llm.build_gemini_llm(key, a.model, cache, daily_limit=(a.daily_limit or None), min_interval=a.min_interval, say=say)
        except ImportError as e:
            if "google" not in str(e) and "genai" not in str(e): raise
            say("The Gemini SDK is not installed. Run:  pip install google-genai numpy  and then this command again."); return 3
        try:
            M.run_million(llm, a.seed, a.warm + a.steps, M.MAIN_POLICY, path, meta=dict(model=model), stream="HardStream", reasks=H.HARD_REASKS,
                          degree_threshold=H.HARD_DEGREE, scripted=a.warm,
                          progress=lambda t, n, r: (say(f"  step {t - a.warm}/{a.steps}: held state exact={r['exact']}  prompt tokens={r['prompt_tokens']}  {r['seconds']:.1f} s") if t > a.warm else None))
        except api_llm.QuotaStop as e:
            say(f"\nStopped: {e}\nEvery finished step is saved; run the same command again later to carry on.")
    rows = _read_rows(path)
    if not rows: say("Nothing on file yet."); return 1
    report(rows, a.warm, a.seed, say=say)
    return 0


if __name__ == "__main__":
    sys.exit(main())

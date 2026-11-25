"""Test 46 (docs/LITE_COMPACTION_TEST.md): the kept chat with compaction against AXI, on the SAME job, on Gemini 3.5 Flash Lite, inside the free quota of an AI Studio key.

The page docs/COMPACTION_TEST.md of 14 Nov asks the same question on Gemini 3.8 Flash and needs a paid key (the free key gives that model 20 calls a day). This test asks it on a model the free key gives
500 calls a day, so it needs no money. The job (`million_hard.HardStream`, seed 20261111), the heavy turns (`gemini_long.ticket`) and the compaction rule (`gemini_compaction`) are the ones already in the repo and
are not changed. Arm A (AXI, `gemini_long.run_axi`) is run again here, on the same model, because the saved arm A of test 22 was run on another model.

Arms, 300 steps each, every reply applied as it comes in S and V (the chat's way):
  S  generic summary when the last prompt reached 32,000 tokens;  V  the summary writes out the complete current graph, same trigger;  A  AXI: the host holds the exact state (scope_m7).

What this file does not do: it calls no model by itself (the notebook does), it has no price and no budget (the free key is never turned into a paid one), and it says nothing about another model, about summaries written by
another prompt, or about any company's own compaction code."""
from __future__ import annotations
from typing import List, Optional, Tuple

from axi.experiments import api_llm as API
from axi.experiments import gemini_compaction as C
from axi.experiments import gemini_long as G

MODEL = "gemini-3.5-flash-lite"                          # the id is taken from the list the key can see (`API.resolve_model`)
SEED = G.SEED                                            # 20261111, the stream of test 22 and of docs/COMPACTION_TEST.md
STEPS = 300
ARMS = {"S": dict(trigger=32_000, style="generic"), "V": dict(trigger=32_000, style="verbatim")}
DAILY_CALLS = 480                                        # the free key gives 500 calls a day; 20 are left unused (a failed call counts too)
MIN_INTERVAL = 6.0                                       # seconds between calls: 10 a minute, under the key's 15 a minute and 250,000 input tokens a minute at a prompt of about 17,000 tokens
TOKEN_CAP = {"S": 12_000_000, "V": 12_000_000, "A": 3_000_000}   # tokens sent per arm (prompt + visible + thinking, summary calls included): a guard against a runaway loop, not a price


def build_lite_llm(api_key, cache_path, on_write=None, client=None, types=None, thinking=None, say=print, daily_limit: Optional[int] = DAILY_CALLS, min_interval: float = MIN_INTERVAL):
    """List the models the key can see, pick `MODEL`, and return (llm, model id). No price, no budget: `spent_usd()` stays None. Same backend and same disk cache as `gemini_long.build_chat_llm`."""
    if client is None:
        from google import genai
        client = genai.Client(api_key=api_key)
    model = API.resolve_model(API.list_model_names(client), MODEL); say(f"model id: {model}")
    backend = G.GeminiChatBackend(model=model, thinking=thinking, temperature=None, max_output_tokens=G.MAX_OUT, client=client, types=types)
    return G.ChatLLM(backend, cache_path, backend.config_dict(), daily_limit=daily_limit, min_interval=min_interval, on_write=on_write, say=say), model


def run_arm(llm, arm: str, path: str, steps: int = STEPS, meta=None, max_tokens: Optional[int] = None, progress=None, trigger: Optional[int] = None) -> dict:
    """One arm of the test. S and V are `gemini_compaction.run_compact`; A is `gemini_long.run_axi`. The same call again carries on from the file."""
    if arm in ARMS:
        return C.run_compact(llm, SEED, steps, arm, path, meta=meta, max_tokens=TOKEN_CAP[arm] if max_tokens is None else max_tokens, progress=progress, trigger=trigger)
    if arm == "A":
        return G.run_axi(llm, SEED, steps, path, meta=meta, max_tokens=TOKEN_CAP["A"] if max_tokens is None else max_tokens, progress=progress)
    raise ValueError(f"unknown arm {arm!r}")


def replay_arm(rows: list, arm: str, trigger: Optional[int] = None):
    """The replay of the saved replies (a check that nothing was changed): raises at the first flag that differs."""
    if arm in ARMS: return C.replay(rows, SEED, ARMS[arm]["trigger"] if trigger is None else trigger, ARMS[arm]["style"])
    if arm == "A": return G.replay(rows, SEED, "A")
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------- the reading rules (fixed in the page before any call)
L_S = "arm S finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction"
L_V = "arm V finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every compaction"
L_A = "arm A finished its 300 steps with every step on file, and the replay of its saved replies reproduces every saved flag and every re-ask"
L_SUM = "every compaction has a non-empty summary, and there were at least 8 compactions in arm S and at least 8 in arm V"
L_PARSED = "at least 98 percent of the replies of each of the arms S and V were accepted by the strict parser, and at most 2 percent of the steps of arm A ended with a reply the strict parser refused after the re-asks"
L_EX_A = "arm A: the held state equals the oracle's on at least 98 percent of its 300 steps"
L_EX_S = "arm S: the held state equals the oracle's on at least 98 percent of its 300 steps"
L_EX_V = "arm V: the held state equals the oracle's on at least 98 percent of its 300 steps"
L_AXI_S = "AXI (arm A) holds the exact state on a share of steps at least 5 points above arm S's"
L_AXI_V = "AXI (arm A) holds the exact state on a share of steps at least 5 points above arm V's"
L_END_S = "arm S: the held state is not exact at step 300"
L_EARLY = "arm S: the first reply that is not exactly right comes before turn 100"
L_TOKENS = "the mean prompt tokens a step of arm S, summary calls included, are at least 3 times those of arm A"
L_TIME = "the mean seconds of a step of arm S, summary calls included, are at most the mean seconds of a step of arm A"
LINES = [("guard", L_S), ("guard", L_V), ("guard", L_A), ("instrument", L_SUM), ("instrument", L_PARSED), ("claim", L_EX_A), ("claim", L_EX_S), ("claim", L_EX_V), ("claim", L_AXI_S), ("claim", L_AXI_V),
         ("claim", L_END_S), ("claim", L_EARLY), ("claim", L_TOKENS), ("claim", L_TIME)]
EXACT_MIN, AXI_MARGIN, PARSED_MIN, REFUSED_MAX, MIN_COMPACTIONS, TOKEN_RATIO = 0.98, 0.05, 0.98, 0.02, 8, 3.0


def verdict(arms: dict, replays: Optional[dict] = None) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `arms`: {'S': run dict, 'V': run dict, 'A': run dict} as `run_arm` returns them (a missing arm is not run); `replays`: {'S': bool, 'V': bool, 'A': bool}."""
    replays = replays or {}; rows = {k: (v["rows"] if v else None) for k, v in arms.items()}; fig = {k: C.arm_figures(r) for k, r in rows.items() if r}
    have = lambda *ks: all(rows.get(k) for k in ks); out: List[Optional[bool]] = []
    for k in "SVA": out.append(None if not rows.get(k) else bool(replays.get(k, True)) and len(rows[k]) >= STEPS)                                                         # 1-3
    out.append(None if not have("S", "V") else all((c.get("text") or "").strip() for k in "SV" for c in C.compactions(rows[k])) and all(len(C.compactions(rows[k])) >= MIN_COMPACTIONS for k in "SV"))   # 4
    out.append(None if not have("S", "V", "A") else fig["S"]["parsed"] >= PARSED_MIN and fig["V"]["parsed"] >= PARSED_MIN and sum(1 for r in rows["A"] if not r["parsed"]) / len(rows["A"]) <= REFUSED_MAX)   # 5
    for k in "ASV": out.append(None if k not in fig else fig[k]["held_exact"] >= EXACT_MIN)                                                                              # 6-8
    for k in "SV": out.append(None if not {"A", k} <= fig.keys() else fig["A"]["held_exact"] - fig[k]["held_exact"] >= AXI_MARGIN - 1e-12)                              # 9-10
    out.append(None if "S" not in fig else not fig["S"]["final_exact"])                                                                                                   # 11
    out.append(None if "S" not in fig else fig["S"]["first_slip"] is not None and fig["S"]["first_slip"] < 100)                                                         # 12
    ok = lambda k: k in fig and fig[k]["prompt_per_step"] is not None
    out.append(None if not (ok("S") and ok("A")) else fig["S"]["prompt_per_step"] >= TOKEN_RATIO * fig["A"]["prompt_per_step"] - 1e-9)                                  # 13
    ok = lambda k: k in fig and fig[k]["seconds_per_step"] is not None
    out.append(None if not (ok("S") and ok("A")) else fig["S"]["seconds_per_step"] <= fig["A"]["seconds_per_step"] + 1e-12)                                             # 14
    assert len(out) == len(LINES)
    return [(i + 1, LINES[i][0], LINES[i][1], out[i]) for i in range(len(LINES))]


def claims_allowed(checks) -> dict:
    ok = {i: c for i, _, _, c in checks}; return dict(base=all(ok[i] is True for i in (1, 2, 3, 4, 5)))


def headline(checks) -> str:
    ok = {i: c for i, _, _, c in checks}; need = (6, 9, 10)
    if any(ok[i] is None for i in need): return "NOT READ (a headline line was not run)"
    held = [k for k, i in zip("SV", (7, 8)) if ok[i]]
    if all(ok[i] for i in need): return "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM" + (f" (but arm {', '.join(held)} also reached 98 percent)" if held else "")
    return "MIXED: " + ", ".join(f"line {i} {'yes' if ok[i] else 'NO'}" for i in need)


# ---------------------------------------------------------------- printing
def print_arm(title: str, rows: list): C.print_arm(title, rows)


def print_verdict(checks, title="LITE COMPACTION HEAD-TO-HEAD (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks))
    print("  rule on the page:", "lines 1 to 5 read yes: the figures of the arms are claimed as read." if claims_allowed(checks)["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")

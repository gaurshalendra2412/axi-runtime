"""The bigger ARC-AGI-3 test (docs/GEMINI_ARC_BIG_TEST.md, Test 27) from a terminal (Codespaces): the same test as colab/axi_gemini_arc_big.ipynb, for when Colab will not give a machine. The same code, the same
eight games (the same rule), the same 60 moves, the same three arms and the same 18 reading lines; only where it runs differs: the key is the environment variable GOOGLE_API_KEY (not a Colab Secret) and the progress is
saved in a folder (default ~/axi_big_out) instead of Google Drive. It never turns billing on; it needs the prices typed on the command line, and `run` refuses to start without a dollar budget.

After `pip install google-genai arc-agi arcengine`, from the `python` folder:
  1. the pilot (36 real moves, under a dollar; it prints the projected cost of each arm):
       python colab/run_gemini_arc_big.py pilot --price-in P --price-out P --price-cached P
  2. each arm, in its own terminal (resumable: run the same command again after a stop):
       python colab/run_gemini_arc_big.py run K --price-in P --price-out P --price-cached P --budget B
     (then `run C` and `run D`)
  3. when all three are finished, the reading (no key, no cost):
       python colab/run_gemini_arc_big.py read
     It writes docs/results/gemini_arc_big_all_default.json and docs/results/gemini_arc_big_printout.txt."""
import argparse, json, logging, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from axi.experiments import gemini_arc as G
from axi.experiments import gemini_arc_c as C
from axi.experiments import gemini_arc_d as D
from axi.experiments.api_llm import QuotaStop, ServiceBusy, BudgetStop

DEFAULT_OUT = os.path.expanduser("~/axi_big_out"); TAG = "default"
KEY_HELP = "No key found. Set it once in this terminal (it is not shown and not saved):   read -s GOOGLE_API_KEY && export GOOGLE_API_KEY"
CAPS = dict(K=D.K_TOKEN_CAP, C=D.C_TOKEN_CAP, D=D.D_TOKEN_CAP)
RESET = "The free quota resets at midnight Pacific time. Then run the same command again."
BUSY = "This is Google's server being busy. Wait at least 30 minutes, then run the same command once."


def out_path(out: str, name: str) -> str: return os.path.join(out, name)


def pick(out: str, offered) -> list:
    """The eight games by the rule on the page; the list is saved on the first run and every later run (and every arm) uses the saved list."""
    games = D.pick_games(offered); f = out_path(out, "arc_big_games.json")
    if os.path.exists(f):
        saved = json.load(open(f))
        if saved != games: print("NOTE: the toolkit's list differs from the list saved in", out, "; the saved list is used, so that every arm plays the same games.")
        games = saved
    else: json.dump(games, open(f, "w"))
    for g in games: assert g in set(map(str, offered)), g + " is not in the list of games the toolkit offers today"
    assert D.PILOT_GAME in games, "the pilot game is not among the games"
    return games


def guard(args, budget=None) -> dict:
    assert args.price_in and args.price_out, "Type --price-in and --price-out (dollars per million tokens). There is no built-in price on purpose."
    g = dict(price_in=args.price_in, price_out=args.price_out, price_cached=args.price_cached)
    if budget: g["budget_usd"] = budget
    return g


def build_llm(key: str, out: str, name: str, g: dict):
    cache = out_path(out, f"axi_gemini_big_cache_{name}_{TAG}.jsonl")
    llm, model = G.GL.build_chat_llm(key, cache, daily_limit=None, min_interval=0.3, thinking=None, **g); return llm, model


def progress_cb(llm):
    def prog(t, n, r):
        if t % 5 == 0 or t == n:
            sp = llm.spent_usd() if hasattr(llm, "spent_usd") else None
            print(f"move {t}/{n} (game {r['game']} move {r['m']}) | prompt {r['prompt_tokens']:,} (cached {r['cache_tokens']:,}) | {r['seconds']:.1f} s | {r['kind']} legal {r['reply_ok']} changed {r['changed']} | levels {r['levels']}/{r['win_levels']} | spent {('$%.2f' % sp) if sp is not None else '-'}", flush=True)
    return prog


def go(fn):
    """Run `fn`; a quota, budget or busy stop is printed and gives None (everything finished is saved)."""
    try: return fn()
    except QuotaStop as e:
        print("\nSTOPPED FOR NOW: " + str(e))
        if isinstance(e, BudgetStop): print("The budget is used up. Everything finished is saved; raise --budget and run the same command to go on.")
        elif isinstance(e, ServiceBusy): print(BUSY)
        else: print(RESET)
        return None


def run_arm(llm, make_env, games, arm: str, path: str, meta=None, moves: int = D.BIG_MOVES, max_tokens=None, progress=None, action_of=None) -> dict:
    """One arm over the games: K is the plain kept chat (gemini_arc.py), C arm C of Test 26, D arm C plus the notes."""
    kw = dict(games=games, moves=moves, meta=meta, max_tokens=max_tokens if max_tokens is not None else CAPS[arm], progress=progress, action_of=action_of)
    if arm == "K": return G.run(llm, make_env, "K", path, **kw)
    if arm == "C": return C.run(llm, make_env, path, **kw)
    if arm == "D": return D.run(llm, make_env, path, **kw)
    raise ValueError("arm must be K, C or D")


def do_pilot(llm, make_env, out: str, price_in, price_out, price_cached, moves: int = D.PILOT_MOVES, meta=None, action_of=None) -> dict:
    res = {}
    for arm in ("K", "C", "D"):
        res[arm] = go(lambda: run_arm(llm, make_env, (D.PILOT_GAME,), arm, out_path(out, f"arc_big_pilot_{arm}.jsonl"), meta, moves, None, progress_cb(llm), action_of))
        if not res[arm]: return {}
    D.print_stats("pilot K (plain model)", res["K"]["rows"]); D.print_stats("pilot C (ledger read-out)", res["C"]["rows"]); D.print_stats("pilot D (read-out and notes)", res["D"]["rows"])
    rep = D.pilot_report(res["K"]["rows"], res["C"]["rows"], res["D"]["rows"], price_in, price_out, price_cached); print()
    for k, v in rep.items(): print(f"  {k}: {v if not isinstance(v, float) else round(v, 3)}")
    print("\nRead these before the budget: k_growth_per_move and k_total_prompt_tokens are the plain model's kept chat (they grow with the square of the moves); cost_k, cost_c and cost_d are the three arms at your prices over the whole run; c_forced_share and d_forced_share must stay below 0.4 over the whole run.")
    print("Send me this printout, then choose a --budget for each arm.")
    return dict(rep=rep, rows={a: r["rows"] for a, r in res.items()})


def do_run(llm, make_env, games, arm: str, out: str, meta=None, moves: int = D.BIG_MOVES, max_tokens=None, action_of=None):
    path = out_path(out, f"arc_big_{arm}_{TAG}.jsonl"); res = go(lambda: run_arm(llm, make_env, games, arm, path, meta, moves, max_tokens, progress_cb(llm), action_of))
    if res:
        D.print_stats("ARM " + arm, res["rows"]); D.print_games(res["rows"])
        if hasattr(llm, "spent_usd"): print("spent so far on this model: $%.2f" % llm.spent_usd())
        json.dump(res, open(out_path(out, f"arc_big_{arm}_{TAG}.json"), "w"))
    return res


def do_read(make_env, games, out: str, results_dir: str, moves: int = D.BIG_MOVES, action_of=None) -> dict:
    """Read the three saved arm files (no call to the model), replay every move into fresh copies of the games and print the 18 lines."""
    arms, ok = {}, {}
    for a in ("K", "C", "D"):
        f = out_path(out, f"arc_big_{a}_{TAG}.jsonl")
        try: arms[a] = D.load_arm(f)
        except Exception as e: print("arm", a, "not read:", e); arms[a] = None
    for a, fn in (("K", lambda r: G.replay(r, make_env, "K", games, moves, action_of)), ("C", lambda r: C.replay(r, make_env, games, moves, action_of)), ("D", lambda r: D.replay(r, make_env, games, moves, action_of))):
        if not arms[a]: continue
        try: fn(arms[a]["rows"]); ok[a] = True
        except Exception as e: print("replay", a + ":", e); ok[a] = False
    checks = D.verdict(arms["K"], arms["C"], arms["D"], ok, games=games, moves=moves)
    for a, t in (("K", "K (plain model)"), ("C", "C (ledger read-out)"), ("D", "D (read-out and notes)")):
        if arms[a]: print(); D.print_stats("ARM " + t, arms[a]["rows"]); D.print_games(arms[a]["rows"])
    print("\nlevels completed in each game:"); D.print_levels(games, K=arms["K"], C=arms["C"], D=arms["D"]); D.print_verdict(checks)
    os.makedirs(results_dir, exist_ok=True)
    json.dump(dict(K=arms["K"], C=arms["C"], D=arms["D"], games=list(games), checks=checks, replays=ok), open(os.path.join(results_dir, f"gemini_arc_big_all_{TAG}.json"), "w"), indent=1, default=str)
    lines = "\n".join(f"{i} {t} | {({True: 'yes', False: 'NO', None: 'not run'})[v]} | {x}" for i, t, x, v in checks) + "\nheadline: " + D.headline(checks)
    open(os.path.join(results_dir, "gemini_arc_big_printout.txt"), "w").write(lines + "\n"); print("\nsaved docs/results/gemini_arc_big_all_default.json and gemini_arc_big_printout.txt")
    return dict(checks=checks, replays=ok)


def _toolkit():
    import arc_agi
    logging.disable(logging.INFO); arc = arc_agi.Arcade()
    def make_env(game):
        env = arc.make(game, seed=G.GAME_SEED); assert env is not None, "could not make " + game; return env
    return make_env, [str(getattr(e, "game_id", e)) for e in arc.get_environments()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The bigger ARC-AGI-3 test from a terminal.")
    ap.add_argument("what", choices=("pilot", "run", "read")); ap.add_argument("arm", nargs="?", choices=("K", "C", "D"))
    ap.add_argument("--price-in", type=float); ap.add_argument("--price-out", type=float); ap.add_argument("--price-cached", type=float)
    ap.add_argument("--budget", type=float, help="the most this arm may cost, in dollars (required for `run`)"); ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    if a.what == "read":
        make_env, offered = _toolkit(); games = json.load(open(out_path(a.out, "arc_big_games.json"))) if os.path.exists(out_path(a.out, "arc_big_games.json")) else pick(a.out, offered)
        do_read(make_env, games, a.out, os.path.join(ROOT, "docs", "results")); return 0
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key: print(KEY_HELP); return 2
    if a.what == "run" and not a.arm: print("Say which arm: run K, run C or run D."); return 2
    if a.what == "run" and not a.budget: print("Type --budget (dollars) for this arm. Read the pilot first."); return 2
    make_env, offered = _toolkit(); games = pick(a.out, offered); print(len(games), "games:", ", ".join(games))
    if a.what == "pilot":
        llm, model = build_llm(key, a.out, "pilot", guard(a)); print("model:", model); do_pilot(llm, make_env, a.out, a.price_in, a.price_out, a.price_cached, meta=dict(model=model, thinking=None)); return 0
    llm, model = build_llm(key, a.out, a.arm, guard(a, a.budget)); print("arm:", a.arm, "| model:", model, "| budget: $%.2f" % a.budget)
    res = do_run(llm, make_env, games, a.arm, a.out, meta=dict(model=model, thinking=None)); return 0 if res else 3


if __name__ == "__main__": sys.exit(main())

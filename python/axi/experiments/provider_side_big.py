"""The serving side on a bigger model, measured the way a company that runs the model counts: drift, speed, latency, tokens and energy (concept-map item 114, docs/PROVIDER_SIDE_BIG_TEST.md, test 21).
Seven cells on `Qwen/Qwen2.5-14B-Instruct-AWQ`, one NVIDIA L4, vLLM 0.31.0, no retry anywhere, on the sixth run's jobs:
  P0 / PN   32 jobs, 100 steps, the exact state at every step, the capped expression / the grammar that names the coordinates (the sixth run's L0 and LN on a bigger model);
  P4 / PC   4 jobs, the exact state / the whole chat kept (the sixth run's L8 and LC), the cost of keeping the chat in time, tokens and energy;
  X0 / XW / XR   32 jobs, the request given with coordinates plus the names list / in ordinary words with the capped expression / in ordinary words with the expression the host reader builds (the seventh run's C0, W0, WN).
It imports `provider_side.py`, `provider_side_typed.py`, `provider_side_long.py`, `provider_side_free.py`, `grounding.py` and `word_tasks.py` and edits none of them. The engine is the earlier runs' engine with another
model, a cache of 32,000 tokens (it must share 22 GB with the weights), a per-request record of latency where the engine reports it, and a GPU power meter (nvidia-smi) around every cell.
Not testable without a GPU: tests/test_provider_side_big.py runs everything against a stand-in that obeys the grammar and against a fake `vllm` module and a fake power meter. A stand-in answer is never a result."""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, threading, time
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
import axi.experiments.provider_side_free as F
from axi.experiments.provider_side import _ge, _le

# ---------------------------------------------------------------- the settings written on the page
MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
SEED, POLICY, HINT_RULES, MAX_NEW, STEPS = F.SEED, F.POLICY, F.HINT_RULES, F.MAX_NEW, 100
REGEX, REGEX_SHA256, NAME_CAP = F.REGEX, F.REGEX_SHA256, F.NAME_CAP
CAPACITY_BLOCKS, BLOCK, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION = 2000, 16, 8192, 0.90
CAPACITY_TOKENS = CAPACITY_BLOCKS * BLOCK                                              # 32,000 tokens
FIRST, K_JOBS, K_SMALL = 192, 32, 4
SAMPLE_MS = 500                                                                         # one power reading every half second

# label: (jobs side by side, first job number, arm, request given as, grammar). S = the exact state at every step; KC = the whole chat kept.
CELLS: Dict[str, Tuple[int, int, str, str, str]] = {
    "P0": (K_JOBS, FIRST, "S", "coords", "capped"),
    "PN": (K_JOBS, FIRST, "S", "coords", "named"),
    "P4": (K_SMALL, FIRST, "S", "coords", "named"),
    "PC": (K_SMALL, FIRST, "KC", "coords", "named"),
    "X0": (K_JOBS, FIRST, "S", "names", "capped"),
    "XW": (K_JOBS, FIRST, "S", "words", "capped"),
    "XR": (K_JOBS, FIRST, "S", "words", "read"),
}
ORDER = ["P0", "PN", "P4", "PC", "X0", "XW", "XR"]
FREE_LABEL = {"X0": "C0", "XW": "W0", "XR": "WN"}                                       # the seventh run's cell each X cell repeats on a bigger model

PINS = F.PINS
TYPED_MODULE_SHA256, LONG_MODULE_SHA256, GROUNDING_SHA256, WORD_TASKS_SHA256 = F.TYPED_MODULE_SHA256, F.LONG_MODULE_SHA256, F.GROUNDING_SHA256, F.WORD_TASKS_SHA256
FREE_MODULE_SHA256 = "ba27ad03e532ce7430957805d3b9567edd879b41c4e2617b79dbaa0d4c69b0b7"            # provider_side_free.py, the module the seventh run's Result was read with
LONG_FILE_SHA256 = F.LONG_FILE_SHA256
FREE_FILE_SHA256 = "4ec1ce81239b4b184327a328b765797912a4b0534156383eec49214c65e4484e"             # docs/results/axi_provider_free.json
READER_COUNTS = {"right": 2688, "wrong": 0, "miss": 512}                                          # the host reader on the 3,200 requests of jobs 192 to 223
GRAMMAR_SHA: Dict[str, str] = {}                                                                  # filled below (computed from the jobs, then pinned by the test and the page)
# the 7B model's figures on the same 32 jobs (from docs/results/axi_provider_long.json and axi_provider_free.json), the numbers the lines are read against
SEVEN_B = dict(L0=(3075, 3200), LN=(3200, 3200), C0=(3200, 3200), W0=(1343, 3200), WN=(2030, 3200))

T_PN, T_KEPT_SHARE, T_HOUR_RATIO, T_FLAT_TIME, T_GROW_TIME, T_TOKEN_RATIO, T_ENERGY_RATIO, T_SPEED_TOL = 0.99, 0.0, 2.0, 1.15, 2.0, 1.3, 2.0, 0.05
T_X0, T_XW, T_XR, T_POOL_B, T_TIME = 0.99, 0.95, 0.99, 0.03, 0.15
GPU_FREE_MIB = L.GPU_FREE_MIB
POWER_COVER = 0.9                                                                                 # the meter must have read at least this share of the readings a cell's time allows


# ---------------------------------------------------------------- the engine
class BigEngine(T.TypedVLLMEngine):
    """The earlier runs' engine with another model and a smaller cache, and a record of what the engine reports about the time of each request (arrival, first token, finish) where it reports it.
    `request_metrics` holds one entry per round: None if the engine reports nothing, else the mean and the largest time to first token and total time of the round's requests, in seconds."""
    def __init__(self, model: str = MODEL, capacity_blocks: int = CAPACITY_BLOCKS, max_model_len: int = MAX_MODEL_LEN, gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION):
        super().__init__(model=model, capacity_blocks=capacity_blocks, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)
        self.request_metrics: List[Optional[dict]] = []

    def generate_round(self, requests, reset_cache: bool = False):
        from vllm import SamplingParams
        if reset_cache: self.reset_cache()
        prompts = [self.tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True) for r in requests]
        sps = [SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=PS_params(getattr(r, "regex", REGEX))) for r in requests]
        t0 = time.perf_counter(); outs = self.llm.generate(prompts, sps, use_tqdm=False); dt = time.perf_counter() - t0
        self.request_metrics.append(_request_times(outs))
        return [PS.Reply(o.outputs[0].text, len(o.prompt_token_ids), getattr(o, "num_cached_tokens", None), len(o.outputs[0].token_ids)) for o in outs], dt


def PS_params(regex):
    import axi.experiments.provider_side_constrained as C
    return C._structured_params(regex)


def _request_times(outs) -> Optional[dict]:
    first, total = [], []
    for o in outs:
        m = getattr(o, "metrics", None)
        if m is None: return None
        a, f, e = (getattr(m, k, None) for k in ("arrival_time", "first_token_time", "finished_time"))
        if a is None or f is None or e is None: return None
        first.append(f - a); total.append(e - a)
    if not first: return None
    return dict(first_token_mean=sum(first) / len(first), first_token_max=max(first), total_mean=sum(total) / len(total), total_max=max(total))


# ---------------------------------------------------------------- the power meter
def gpu_state() -> Optional[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        name, used, total = [x.strip() for x in out.split(",")]; return dict(name=name, used_mib=int(used), total_mib=int(total))
    except Exception: return None


class PowerMeter:
    """Reads the GPU board power (nvidia-smi, watts) every half second in a thread while a cell runs. It measures the board, not the host, not the building. `stop()` returns the number of readings, the
    seconds of the window, the mean and the largest power and the energy in joules (the readings added up with the time between them). `cmd` is replaceable for the tests."""
    CMD = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)]

    def __init__(self, cmd: Optional[List[str]] = None):
        self.cmd = cmd or self.CMD; self.samples: List[Tuple[float, float]] = []; self.proc = None; self.thread = None; self.t0 = 0.0

    def start(self) -> "PowerMeter":
        self.t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception: self.proc = None; return self
        def read():
            for line in self.proc.stdout:
                try: self.samples.append((time.perf_counter(), float(line.strip().split()[0])))
                except Exception: continue
        self.thread = threading.Thread(target=read, daemon=True); self.thread.start(); return self

    def stop(self, wall_seconds: Optional[float] = None) -> dict:
        window = time.perf_counter() - self.t0
        if self.proc is not None:
            self.proc.terminate()
            try: self.proc.wait(timeout=5)
            except Exception: self.proc.kill()
        if self.thread is not None: self.thread.join(timeout=5)
        return summarize_power(self.samples, window)


class FakeMeter:
    """For the stand-in run: a constant power over the cell's seconds. Never a result."""
    def __init__(self, watts: float = 100.0): self.watts = watts
    def start(self): return self
    def stop(self, wall_seconds: Optional[float] = None) -> dict:
        w = wall_seconds or 0.0; n = int(w * 1000 / SAMPLE_MS)
        return dict(samples=n, window_seconds=w, mean_w=self.watts if n else None, max_w=self.watts if n else None, joules=self.watts * w if n else None, stand_in=True)


def summarize_power(samples: List[Tuple[float, float]], window: float) -> dict:
    if not samples: return dict(samples=0, window_seconds=window, mean_w=None, max_w=None, joules=None)
    joules = sum((samples[i + 1][0] - samples[i][0]) * samples[i][1] for i in range(len(samples) - 1))
    span = samples[-1][0] - samples[0][0]
    return dict(samples=len(samples), window_seconds=window, mean_w=(joules / span if span > 0 else samples[0][1]), max_w=max(w for _, w in samples), joules=joules + samples[-1][1] * max(0.0, window - span))


# ---------------------------------------------------------------- the sessions
def _free_cell(label: str) -> Tuple[str, str]:
    mode = {"X0": "coords", "XW": "words", "XR": "words"}[label]; return mode, CELLS[label][4]


def make_sessions(label: str, steps: int = STEPS, seed: int = SEED) -> list:
    K, first, arm, mode, shape = CELLS[label]
    if label in FREE_LABEL: return [F.FreeSession(seed * 1000 + e, steps, mode if mode != "names" else "coords", shape) for e in range(first, first + K)]
    if arm == "S": return [T.TypedSession(seed * 1000 + e, steps, "old", shape) for e in range(first, first + K)]
    return [L.KeptChat(seed * 1000 + e, steps, shape) for e in range(first, first + K)]


def expressions_of(label: str, steps: int = STEPS, seed: int = SEED) -> set:
    K, first, arm, mode, shape = CELLS[label]; out = set()
    if label in FREE_LABEL:
        for e in range(first, first + K):
            for ins, w, pool, tab, rd, why, oc in F.requests_of(seed * 1000 + e, steps): out.add(REGEX if shape == "capped" or rd is None else T.regex_for("named", rd))
        return out
    for e in range(first, first + K):
        for st in T.script_of(seed * 1000 + e, steps): out.add(REGEX if shape == "capped" else T.regex_for(shape, st["instruction"]))
    return out


def grammar_sha256(label: str, steps: int = STEPS, seed: int = SEED) -> str:
    return hashlib.sha256("\n".join(sorted(expressions_of(label, steps, seed))).encode()).hexdigest()


def run_cell(engine, label: str, steps: int = STEPS, seed: int = SEED, progress=None, meter=None) -> dict:
    K, first, arm, mode, shape = CELLS[label]
    meter = meter if meter is not None else PowerMeter()
    if hasattr(engine, "request_metrics"): engine.request_metrics = []
    meter.start(); r = L.run_sessions(engine, make_sessions(label, steps, seed), progress); power = meter.stop(r["wall_seconds"])
    return dict(label=label, arm=arm, K=K, first_episode=first, mode=mode, prompt="old", shape=shape, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(),
                grammar_sha256=grammar_sha256(label, steps, seed), power=power, request_times=list(getattr(engine, "request_metrics", [])), **r)


for _l in ORDER: GRAMMAR_SHA[_l] = grammar_sha256(_l)


# ---------------------------------------------------------------- summaries
def steps_of(c: dict, pred=None) -> list: return [x for s in c["sessions"] for x in s["log"] if pred is None or pred(x)]


def share_of(c: dict, pred=None) -> Optional[float]:
    xs = steps_of(c, pred); return sum(1 for x in xs if x["exact"]) / len(xs) if xs else None


def _quarters(rounds: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not rounds: return None, None
    k = max(1, len(rounds) // 4); first = sum(r["seconds"] / max(1, r["active"]) for r in rounds[:k]) / k; last = sum(r["seconds"] / max(1, r["active"]) for r in rounds[-k:]) / k
    return first, last


def _pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs: return None
    xs = sorted(xs); i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1))))); return xs[i]


def summarize_cell(c: dict) -> dict:
    s = L.summarize_cell(c); X = steps_of(c); n = len(X); ex = sum(1 for x in X if x["exact"]); rounds = c["rounds"]; first, last = _quarters(rounds); secs = [r["seconds"] for r in rounds]
    pw = c.get("power") or {}; j = pw.get("joules"); rt = [r for r in c.get("request_times", []) if r]
    s.update(mode=c["mode"], rounds=len(rounds), round_p50=_pctl(secs, 0.5), round_p95=_pctl(secs, 0.95), round_max=max(secs) if secs else None, per_step_first_quarter=first, per_step_last_quarter=last,
             time_growth=(last / first if first and last else None), joules=j, mean_w=pw.get("mean_w"), power_samples=pw.get("samples"), power_window=pw.get("window_seconds"),
             joules_per_exact_step=(j / ex if j is not None and ex else None), joules_per_step=(j / n if j is not None and n else None),
             computed_per_exact_step=(s["computed_prefill"] / ex if ex and s["computed_prefill"] is not None else None), new_per_exact_step=(sum(x["n_new"] for x in X) / ex if ex else None),
             first_token_mean=(sum(r["first_token_mean"] for r in rt) / len(rt) if rt else None), first_token_max=(max(r["first_token_max"] for r in rt) if rt else None),
             total_time_mean=(sum(r["total_mean"] for r in rt) / len(rt) if rt else None), total_time_max=(max(r["total_max"] for r in rt) if rt else None), request_times_rounds=len(rt),
             share_pool_a=share_of(c, lambda x: x.get("pool") == "A"), share_pool_b=share_of(c, lambda x: x.get("pool") == "B"), share_read=share_of(c, lambda x: x.get("reader") == "right"), share_unread=share_of(c, lambda x: x.get("reader") == "miss"),
             reader={k: sum(1 for x in X if x.get("reader") == k) for k in ("right", "wrong", "miss")})
    return s


def summarize_all(cells: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cells:
        assert c["label"] not in out, f"cell {c['label']} twice"; out[c["label"]] = summarize_cell(c)
    return out


# ---------------------------------------------------------------- merge
def fingerprint() -> str: return hashlib.sha256(open(__file__, "rb").read()).hexdigest()
def sha256_file(path: str) -> str: return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pinned_files() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(grounding=sha256_file(os.path.join(here, "..", "engine", "grounding.py")), word_tasks=sha256_file(os.path.join(here, "word_tasks.py")),
                provider_side_long=sha256_file(os.path.join(here, "provider_side_long.py")), provider_side_free=sha256_file(os.path.join(here, "provider_side_free.py")))


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses a cell twice, a cell that is not one of this test's, a cell whose settings are not those of its label, a cell of another length, another constraint, another model or
    other settings than the first cell, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        lab = c.get("label")
        if lab not in CELLS: raise ValueError(f"cell {lab!r} is not one of this test's cells")
        if (c["K"], c["first_episode"], c["arm"], c["mode"], c["shape"]) != CELLS[lab]: raise ValueError(f"cell {lab} was not run with the settings of its label: {(c['K'], c['first_episode'], c['arm'], c['mode'], c['shape'])} against {CELLS[lab]}")
        if c["steps"] != STEPS: raise ValueError(f"cell {lab} has {c['steps']} steps, this test has {STEPS}")
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {lab} was not run with this test's constraint")
        if c["engine"].get("kind") == "vllm" and c["engine"].get("model") != MODEL: raise ValueError(f"cell {lab} was run on {c['engine'].get('model')!r}, this test is on {MODEL!r}")
        if key(c) != key(ref): raise ValueError(f"cell {lab} was run under other settings: {key(c)} against {key(ref)}")
        if lab in seen: raise ValueError(f"cell {lab} twice")
        seen.add(lab)
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP,
                          cells={k: list(CELLS[k]) for k in ORDER}), cells=sorted(cells, key=lambda c: ORDER.index(c["label"])))


def merge_files(paths: List[str], out: str) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells)
    m["module_sha256"] = fingerprint(); m["typed_module_sha256"] = T.fingerprint(); m["pinned_sha256"] = T.pinned_hashes(); m["big_pinned_sha256"] = pinned_files()
    with open(out, "w") as f: json.dump(m, f)
    return m


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_BIG_TEST.md before the run)
BL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, and in the cells that carry the named expression on every request (PN, P4, PC) none used all 96 new tokens and every reply is in the language of its own request"),
    ("instrument", "the power meter worked: for every cell that was run it read at least 90 percent of the readings that the seconds of its rounds allow, at one reading every half second"),
    ("guard", "the requests in words say what the oracle's request says: in every sentence of XW and XR the name (or an alias) of each service the oracle's request is about is written, in the order of the oracle's request, the numbers are the oracle's, and a sentence of the second pool names exactly one service more (checked by plain search, not by the reader)"),
    ("guard", "the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs (and the reader), and the oracle's own reply is inside the expression of every request"),
    ("guard", "every job of every exact-state cell ran all 100 steps; none ended early"),
    ("guard", "what the reader read, as recorded in XR, is what the reader gives on the same sentences when it is run again: the same outcome for every one of the 3,200 requests, and not one request read wrong"),
    ("claim", "drift, named expression: over 100 steps the exact share of PN is at least 99 percent"),
    ("claim", "drift, capped expression: the exact share of P0 is at least that of the 7B model's L0 on the same 32 jobs (3,075 of 3,200)"),
    ("claim", "the kept chat does not reach step 100: every job of PC stopped before its 100th step, because its next prompt would have passed 7,600 tokens"),
    ("claim", "speed: the exact steps per GPU hour of P4 are at least twice those of PC (both in this session, the same 4 jobs)"),
    ("claim", "latency does not grow with the job: in PN the seconds per step of the last quarter of the rounds are at most 1.15 times those of the first quarter"),
    ("claim", "latency grows with the kept chat: in PC the seconds per step of the last quarter of the rounds are at least twice those of the first quarter"),
    ("claim", "tokens: the kept chat computes at least 1.3 times as many prompt tokens per step as the exact-state arm on the same 4 jobs (PC against P4)"),
    ("claim", "energy: the kept chat takes at least twice the GPU energy per exact step that the exact-state arm takes on the same 4 jobs (PC against P4)"),
    ("claim", "the grammar costs nothing here: the seconds per step of PN are within 5 percent of those of P0, and the energy per exact step of PN is not above that of P0 (both in this session)"),
    ("claim", "the names list does not hurt the bigger model: the exact share of X0 is at least 99 percent"),
    ("claim", "the bigger model reads the names: the exact share of XW over 100 steps is at least 95 percent"),
    ("claim", "with the host reading what it can, the exact share of XR over 100 steps is at least 99 percent"),
    ("claim", "the second pool of wordings does not confuse the bigger model more than three points: in XW the exact share on the steps of that pool is at least that on the first pool minus three points"),
    ("claim", "the seconds per step of XR are at most 15 percent above those of XW (both in this session)"),
]
BL_HEADLINE = (8, 18)


def _cover_ok(c: dict) -> bool:
    pw = c.get("power") or {}; secs = sum(r["seconds"] for r in c["rounds"]); need = secs * 1000.0 / SAMPLE_MS * POWER_COVER
    return pw.get("samples") is not None and pw["samples"] >= need and pw.get("joules") is not None


def verdict_big(cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). A line that needs a cell that was not run is None, never 'no'."""
    new = {c["label"]: c for c in cells if any(s["log"] for s in c["sessions"])}; Sn = {k: summarize_cell(c) for k, c in new.items()}
    has = lambda *ks: all(k in Sn for k in ks); out: List[Optional[bool]] = []
    allrep = [r for k in new for r in T.replies_of(new[k])]; named = [r for k in ("PN", "P4", "PC") if k in new for r in T.replies_of(new[k])]
    out.append(all(r[2] is not None for r in allrep) if allrep else None)                                                                                                                        # 1
    out.append((sum(1 for r in allrep if not r[0]) <= (1 - T.T_PARSED) * len(allrep) + 1e-9 and all(r[1] < MAX_NEW for r in named) and all(r[3] for r in named)) if allrep else None)             # 2
    out.append(all(_cover_ok(c) for c in new.values()) if new else None)                                                                                                                         # 3
    def words_ok():
        for k in ("XW", "XR"):
            if k not in new: continue
            for s in new[k]["sessions"]:
                tables, news = F.name_tables(s["seed"], STEPS); sc = T.script_of(s["seed"], STEPS)
                for x, st, tb, nn in zip(s["log"], sc, tables, news):
                    if not F.wording_ok(st["instruction"], x["request"], x["pool"], tb, nn): return False
        return True
    out.append(words_ok() if ("XW" in new or "XR" in new) else None)                                                                                                                             # 4
    def expressions_ok():
        for k, c in new.items():
            if not (c["grammar_sha256"] == grammar_sha256(k, c["steps"], c["seed"]) == GRAMMAR_SHA[k]): return False
            for s in c["sessions"]:
                if k in FREE_LABEL:
                    for st, (ins, w, pool, tb, rd, why, oc) in zip(T.script_of(s["seed"], STEPS), F.requests_of(s["seed"], STEPS)):
                        rx = REGEX if CELLS[k][4] == "capped" or rd is None else T.regex_for("named", rd)
                        if re.fullmatch(rx, st["ideal"]) is None: return False
                else:
                    for st in T.script_of(s["seed"], STEPS):
                        if re.fullmatch(REGEX if CELLS[k][4] == "capped" else T.regex_for(CELLS[k][4], st["instruction"]), st["ideal"]) is None: return False
        return True
    out.append(expressions_ok() if new else None)                                                                                                                                                # 5
    s_cells = [k for k in new if CELLS[k][2] == "S"]
    out.append(all(len(s["log"]) == STEPS and s["stopped"] is None for k in s_cells for s in new[k]["sessions"]) if s_cells else None)                                                            # 6
    def reader_again():
        if "XR" not in new: return None
        for s in new["XR"]["sessions"]:
            tables, news = F.name_tables(s["seed"], STEPS); sc = T.script_of(s["seed"], STEPS)
            for i, (x, st) in enumerate(zip(s["log"], sc)):
                w, pool = F.wording(s["seed"], i, st["instruction"], tables[i], news[i]); rd, why = F.read_request(w, tables[i]); oc = F.reader_outcome(st["instruction"], rd)
                if x["reader"] != oc or w != x["request"] or oc == "wrong": return False
        return True
    out.append(reader_again())                                                                                                                                                                   # 7
    out.append(None if not has("PN") else _ge(Sn["PN"]["exact_share"], T_PN))                                                                                                                    # 8
    out.append(None if not has("P0") else Sn["P0"]["exact_steps"] >= SEVEN_B["L0"][0])                                                                                                           # 9
    out.append(None if not has("PC") else all(s["stopped"] == "context" and len(s["log"]) < STEPS for s in new["PC"]["sessions"]))                                                               # 10
    out.append(None if not has("P4", "PC") else Sn["P4"]["exact_per_hour"] >= T_HOUR_RATIO * Sn["PC"]["exact_per_hour"])                                                                         # 11
    out.append(None if not has("PN") or Sn["PN"]["time_growth"] is None else _le(Sn["PN"]["time_growth"], T_FLAT_TIME))                                                                          # 12
    out.append(None if not has("PC") or Sn["PC"]["time_growth"] is None else _ge(Sn["PC"]["time_growth"], T_GROW_TIME))                                                                          # 13
    out.append(None if not has("P4", "PC") else _ge(Sn["PC"]["computed_per_step"], T_TOKEN_RATIO * Sn["P4"]["computed_per_step"]))                                                               # 14
    out.append(None if not has("P4", "PC") or Sn["P4"]["joules_per_exact_step"] is None or Sn["PC"]["joules_per_exact_step"] is None else _ge(Sn["PC"]["joules_per_exact_step"], T_ENERGY_RATIO * Sn["P4"]["joules_per_exact_step"]))  # 15
    out.append(None if not has("P0", "PN") or Sn["PN"]["joules_per_exact_step"] is None or Sn["P0"]["joules_per_exact_step"] is None else
               (abs(Sn["PN"]["seconds_per_step"] / Sn["P0"]["seconds_per_step"] - 1) <= T_SPEED_TOL + 1e-12 and Sn["PN"]["joules_per_exact_step"] <= Sn["P0"]["joules_per_exact_step"] + 1e-12))              # 16
    out.append(None if not has("X0") else _ge(Sn["X0"]["exact_share"], T_X0))                                                                                                                    # 17
    out.append(None if not has("XW") else _ge(Sn["XW"]["exact_share"], T_XW))                                                                                                                    # 18
    out.append(None if not has("XR") else _ge(Sn["XR"]["exact_share"], T_XR))                                                                                                                    # 19
    out.append(None if not has("XW") or Sn["XW"]["share_pool_a"] is None or Sn["XW"]["share_pool_b"] is None else _ge(Sn["XW"]["share_pool_b"], Sn["XW"]["share_pool_a"] - T_POOL_B))              # 20
    out.append(None if not has("XW", "XR") else _le(Sn["XR"]["seconds_per_step"], Sn["XW"]["seconds_per_step"] * (1 + T_TIME)))                                                                 # 21
    assert len(out) == len(BL_LINES)
    return [(i + 1, BL_LINES[i][0], BL_LINES[i][1], out[i]) for i in range(len(BL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim: with lines 2 to 7 yes the figures of the cells are claimed as read. Drift from lines 8 and 9, the cost of the kept chat from lines 10 to 15, the grammar's cost from
    line 16, what the bigger model does with names from lines 17 to 20, a statement about time only from lines 11 to 13, 16 and 21 (cells of one session). A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (2, 3, 4, 5, 6, 7))
    pick = lambda ks: ({i: ok[i] for i in ks} if base else {i: None for i in ks})
    return dict(base=base, drift=pick((8, 9)), kept_chat=pick((10, 11, 12, 13, 14, 15)), grammar_cost=pick((16,)), the_model_reads_names=pick((17, 18, 20)), the_host_reads=pick((19,)), time=pick((21,)))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[str, dict]) -> None:
    f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    print(f"\n{'cell':4} {'K':>3} {'jobs':>8} {'request':>6} {'grammar':>7} | {'steps':>6} {'seconds':>8} {'s/step':>6} {'comp/step':>9} | {'exact':>6} {'jobs ok':>7} | {'W mean':>6} {'J/exact':>8} | {'p95 round':>9} {'last/first':>10} | {'exact/h':>8}")
    for lab in ORDER:
        if lab not in S: print(f"{lab:4} not run"); continue
        s = S[lab]; K, first, arm, mode, shape = CELLS[lab]
        print(f"{lab:4} {s['K']:3d} {first:3d}-{first + s['K'] - 1:3d} {mode:>6} {shape:>7} | {s['steps_done']:6d} {s['wall_seconds']:8.0f} {s['seconds_per_step']:6.3f} {s['computed_per_step']:9.0f} | {pc(s['exact_share']):>6} {s['jobs_exact_to_end'] if 'jobs_exact_to_end' in s else '-':>7} | "
              f"{f(s['mean_w']):>6} {f(s['joules_per_exact_step'], 2):>8} | {f(s['round_p95'], 2):>9} {f(s['time_growth'], 2):>10} | {s['exact_per_hour']:8.0f}")
    for lab in ORDER:
        if lab in S and S[lab]["request_times_rounds"]:
            s = S[lab]; print(f"{lab}: the engine reported request times in {s['request_times_rounds']} rounds: first token mean {s['first_token_mean'] * 1000:.0f} ms, largest {s['first_token_max'] * 1000:.0f} ms; whole request mean {s['total_time_mean'] * 1000:.0f} ms, largest {s['total_time_max'] * 1000:.0f} ms")


def print_split(S: Dict[str, dict]) -> None:
    pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    for lab in ("X0", "XW", "XR"):
        if lab in S:
            s = S[lab]; print(f"{lab}: exact on first pool {pc(s['share_pool_a'])}, on second pool {pc(s['share_pool_b'])}; host read right {s['reader']['right']}, wrong {s['reader']['wrong']}, not read {s['reader']['miss']}; exact where read {pc(s['share_read'])}, where not read {pc(s['share_unread'])}")


def print_misses(S: Dict[str, dict]) -> None:
    for lab in ORDER:
        if lab in S and S[lab]["first_misses"]:
            print(f"first misses in {lab}: " + "; ".join(f"job {j} step {t} ({k}) {txt[:40]!r}" for j, t, k, txt in S[lab]["first_misses"][:5]))


def print_verdict(checks, title="provider-side test, a bigger model (reading rules fixed before the run)") -> None:
    print(f"\n-- {title} --")
    for i, tag, text, ok in checks: print(f"  {'yes' if ok else 'not run' if ok is None else 'NO':8} {i:2d}. [{tag}] {text}")
    n = sum(1 for *_, ok in checks if ok is not None)
    h = [next(ok for i, _, _, ok in checks if i == k) for k in BL_HEADLINE]; w = lambda ok: "not run" if ok is None else "yes" if ok else "NO"
    print(f"  ({n} of {len(checks)} lines could be read; headline lines {BL_HEADLINE[0]} and {BL_HEADLINE[1]}: {w(h[0])}, {w(h[1])})")
    base = all(ok is True for i, _, _, ok in checks if i in (2, 3, 4, 5, 6, 7))
    print("  rule on the page: " + ("lines 2 to 7 read yes: the figures of the cells are claimed as read." if base else "line 2, 3, 4, 5, 6 or 7 did not read yes, so NO claim is made from this run until the cause is found."))


# ---------------------------------------------------------------- the pre-flight check
def preflight(gpu=None, steps: int = STEPS, seed: int = SEED, long_file: Optional[str] = None, free_file: Optional[str] = None, meter_cmd: Optional[List[str]] = None) -> List[Tuple[str, bool, str]]:
    """Checks that cost seconds and save a Colab run. Every check can read NO; a check that cannot be made reads NO with the error, and the others are still made."""
    out: List[Tuple[str, bool, str]] = []
    def one(name, fn):
        try: ok, detail = fn()
        except Exception as ex: ok, detail = False, f"could not be checked: {type(ex).__name__}: {ex}"
        out.append((name, bool(ok), detail))
    here = os.path.dirname(os.path.abspath(__file__)); res = os.path.join(here, "..", "..", "..", "docs", "results")
    jobs = sorted({e for K, first, *_ in CELLS.values() for e in range(first, first + K)})
    def jobs_ok():
        lens = {e: len(T.script_of(seed * 1000 + e, steps)) for e in jobs}
        return jobs == list(range(FIRST, FIRST + K_JOBS)) and all(n == 100 for n in lens.values()), f"jobs {jobs[0]} to {jobs[-1]} ({len(jobs)}), the first 32 of the sixth run's; steps per job {min(lens.values())} to {max(lens.values())}"
    one("the jobs are the sixth run's first 32, 100 steps each", jobs_ok)
    def files_of_runs():
        a = sha256_file(long_file or os.path.join(res, "axi_provider_long.json")); b = sha256_file(free_file or os.path.join(res, "axi_provider_free.json"))
        return a == LONG_FILE_SHA256 and b == FREE_FILE_SHA256, f"axi_provider_long.json {a[:16]}..., axi_provider_free.json {b[:16]}... (the 7B figures the lines are read against)"
    one("the sixth and seventh runs' files are the ones this page names", files_of_runs)
    def files():
        pins = T.pinned_hashes(); bad = sorted(k for k, v in PINS.items() if pins.get(k) != v); pf = pinned_files()
        okm = (T.fingerprint() == TYPED_MODULE_SHA256 and pf["provider_side_long"] == LONG_MODULE_SHA256 and pf["grounding"] == GROUNDING_SHA256 and pf["word_tasks"] == WORD_TASKS_SHA256 and pf["provider_side_free"] == FREE_MODULE_SHA256)
        return not bad and okm, f"changed: {bad if bad else 'none'}; provider_side_typed.py, provider_side_long.py, provider_side_free.py, grounding.py and word_tasks.py as pinned: {okm}"
    one("the files this test imports are the pinned ones", files)
    def expressions():
        shas = {lab: grammar_sha256(lab, steps, seed) for lab in CELLS}; badg = sorted(l for l in CELLS if shas[l] != GRAMMAR_SHA[l])
        return not badg, "; ".join(f"{l} {shas[l][:8]}" for l in ORDER) + (f"; wrong: {badg}" if badg else "")
    one("the expressions are the ones on the page", expressions)
    def oracle():
        n = outside = 0
        for lab in ORDER:
            K, first, arm, mode, shape = CELLS[lab]
            for e in range(first, first + K):
                for i, st in enumerate(T.script_of(seed * 1000 + e, steps)):
                    if lab in FREE_LABEL:
                        rd = F.requests_of(seed * 1000 + e, steps)[i][4]; rx = REGEX if shape == "capped" or rd is None else T.regex_for("named", rd)
                    else: rx = REGEX if shape == "capped" else T.regex_for(shape, st["instruction"])
                    n += 1; outside += 0 if re.fullmatch(rx, st["ideal"]) else 1
        return outside == 0, f"{n - outside} of {n} requests"
    one("the oracle's reply is inside the expression of every request", oracle)
    def reader():
        c = {"right": 0, "wrong": 0, "miss": 0}
        for e in range(FIRST, FIRST + K_JOBS):
            for r in F.requests_of(seed * 1000 + e, steps): c[r[6]] += 1
        return c == READER_COUNTS, f"read right {c['right']}, wrong {c['wrong']}, not read {c['miss']} of {sum(c.values())} (on the page: {READER_COUNTS['right']} / {READER_COUNTS['wrong']} / {READER_COUNTS['miss']})"
    one("the host reader reads the 3,200 requests as the page says", reader)
    def sentences():
        bad = n = 0
        for e in range(FIRST, FIRST + K_JOBS):
            tables, news = F.name_tables(seed * 1000 + e, steps)
            for i, (st, r) in enumerate(zip(T.script_of(seed * 1000 + e, steps), F.requests_of(seed * 1000 + e, steps))): n += 1; bad += 0 if F.wording_ok(st["instruction"], r[1], r[2], r[3], news[i]) else 1
        return bad == 0, f"{n - bad} of {n} sentences say what the oracle's request says"
    one("the requests in words say what the oracle's request says", sentences)
    def first_request(lab):
        s = make_sessions(lab, steps, seed)[0]; rq = s.request(); user = rq.messages[-1]["content"]; ok = re.fullmatch(rq.regex, s.script[0]["ideal"]) is not None and len(rq.messages) >= 2
        if lab in FREE_LABEL: ok = ok and "Names: " in user and "Request: " in user and F.NOTE in rq.messages[0]["content"]
        return ok, f"user message {len(user)} characters, expression {len(rq.regex)} characters, job {s.seed - seed * 1000}"
    for lab in ORDER: one(f"cell {lab}: its first request is built", lambda lab=lab: first_request(lab))
    def meter():
        m = PowerMeter(meter_cmd).start(); time.sleep(2.0); p = m.stop()
        return p["samples"] >= 2 and p["mean_w"] is not None and p["mean_w"] > 0, f"{p['samples']} readings in 2 seconds, mean {p['mean_w']} W (the power meter reads the board's watts)"
    one("the GPU power can be read", meter)
    def card():
        g = gpu_state() if gpu is None else gpu()
        if g is None: return False, "nvidia-smi could not be read (is the runtime a GPU runtime?)"
        return "L4" in g["name"] and g["used_mib"] <= GPU_FREE_MIB, f"{g['name']}, {g['used_mib']} of {g['total_mib']} MiB in use (at most {GPU_FREE_MIB} before the engine is loaded)"
    one("the GPU is a free L4", card)
    return out


print_preflight = L.print_preflight


# ---------------------------------------------------------------- the stand-in run (harness only)
def dry_run(steps: int = STEPS, labels=None) -> Dict[str, dict]:
    eng = T.TypedStandIn(capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for lab in ORDER:
        if labels is None or lab in labels: cells.append(run_cell(eng, lab, steps, meter=FakeMeter()))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache; it answers with the oracle's own reply), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, a bigger model (docs/PROVIDER_SIDE_BIG_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--preflight", action="store_true", help="run the pre-flight check (reads nvidia-smi)")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_big.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.preflight: raise SystemExit(0 if print_preflight(preflight()) else 1)
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_split(S); print_misses(S); print_verdict(verdict_big(m["cells"])); print("\nwritten to", a.out)
    else: ap.print_help()

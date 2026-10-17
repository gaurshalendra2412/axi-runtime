"""Provider-side serving test: what does the model's GPU spend on a job, when many jobs share it and the serving engine caches prefixes?
Page with the predictions written BEFORE any run: docs/PROVIDER_SIDE_TEST.md (concept-map item 106).

The earlier cost pages count tokens for ONE job in a plain loop that re-sends everything (docs/ENDURANCE_TEST.md, part A). A company that runs the model does not pay that price: its
engine keeps the key/value rows of a prefix it has seen (a prefix cache), so a kept chat only computes its new turn, as long as the rows are still in the cache. What the company pays
for is the GPU: the time to compute what is not cached, and the memory that holds what is. This module measures that on one GPU with one engine (vLLM), for the same 30-step jobs as
docs/LONG_HORIZON_TEST.md, in three ways of holding the context:
  S    the exact current state is shown at every step (fresh context: system, state, request), prefix cache on
  KC   the whole chat is kept and re-sent (every earlier state, request and raw reply stays), prefix cache on
  KN   the same chat, the cache emptied before every round (what a plain engine pays; the reference for what caching buys)
  KR   the same chat as KC, the jobs served in a different random order in every round (descriptive only: a fixed order is the hardest case for a cache that evicts the oldest)
K jobs run side by side in lockstep: in each round every unfinished job sends its next request, the engine serves all of them, then each job applies its reply (policy repair_m6, as in
the earlier tests) and sends the next one. That is K sessions sharing one GPU, each coming back after the others were served. K x context against the size of the cache decides whether the
cached rows of a kept chat are still there when its session returns. The cache is set to a fixed 80,000 tokens (5,000 blocks of 16) so that the pressure is a number written on the
page and not a property of the GPU the run happens to get.

Measured per request: prompt tokens (the engine's own count), cached tokens (what the engine reports it did not have to compute), new tokens; per round: wall seconds around the engine
call. From these: GPU seconds per step, prompt tokens computed per step, the share of the prompt served from the cache, and (to check the quality did not move) the share of steps on
which the held state was exactly the oracle's.

Honest limits: one model (Qwen2.5-7B-Instruct, AWQ 4-bit), one small GPU, one engine and its version, synthetic graphs, greedy decoding, replies not grammar-constrained (the
constrained and the unconstrained call gave the same text in 60 of 60 tasks, docs/LATENCY_7B.md). The lockstep round is a model of K sessions that come back one after another. The
kept chat here re-sends the state with every turn, as the history mode of the long-horizon test does: a chat that sent only the requests would be shorter and would not show the model
the exact state. Dollars are not measured, only GPU seconds on this machine. `StandInEngine` is a model of a prefix cache (blocks of 16, evicted oldest first, the tail of a
sequence before its head) used to test the harness without a GPU; its numbers are never a result."""
import hashlib, json, os, platform, random, re, time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from axi.engine.gate import check
from axi.experiments.agent_loop import Task, base_messages, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script, distance

# ---------------------------------------------------------------- the settings written on the page
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
ARMS = ("S", "KC", "KN", "KR")
ARM_TEXT = {"S": "exact state each step, cache on", "KC": "whole chat kept, cache on", "KN": "whole chat kept, cache emptied each round", "KR": "whole chat kept, cache on, random order each round"}
LEVELS = (8, 32, 64)                          # K, jobs side by side
KN_LEVELS = (8,)                              # the cache-off reference is run at K = 8 only (it computes every prompt token whatever K is)
KR_LEVELS = (32,)                             # the random-order variant is run at K = 32 only, after everything else
STEPS = 30
SEED = 20261017                               # episode e of a run has seed SEED * 1000 + e; the first 8 episodes are the same at every K
POLICY = "repair_m6"
HINT_RULES = False
BLOCK = 16
CAPACITY_BLOCKS = 5000
CAPACITY_TOKENS = CAPACITY_BLOCKS * BLOCK     # 80,000 tokens of key/value rows (4.3 GiB for this model)
MAX_NEW = 96
MAX_MODEL_LEN = 8192
MAX_CONTEXT = 7600                            # a kept chat stops before its next prompt would pass this
GPU_MEMORY_UTILIZATION = 0.90

PL_MIN_EXACT, PL_EXACT_GAP, PL_FLAT, PL_GROWTH, PL_CACHE_FITS = 0.90, 0.05, 1.3, 3.5, 0.80
PL_NOCACHE_SAVING, PL_LIGHT_TOKENS, PL_LIGHT_TIME, PL_OVERRUN, PL_PRESSURE_RATIO, PL_MID_RATIO, PL_PRESSURE_FACTOR = 0.80, 0.30, 0.20, 0.50, 2.0, 1.3, 2.0

PL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the setup made the pressure it was meant to: 64 times the longest kept-chat prompt at K = 64 is at least twice the cache size, and 8 times the longest at K = 8 fits in it"),
    ("guard", "in every cell the held state was exactly the oracle's state on at least 90 percent of the steps"),
    ("claim", "exact-state was not less exact than the kept chat: at every K its share of exact steps is at least the kept chat's minus 5 points"),
    ("claim", "exact-state prompts stay flat: the mean prompt of the last quarter of the steps is at most 1.3 times that of the first quarter (K = 8)"),
    ("claim", "kept-chat prompts grow: the mean prompt of the last quarter is at least 3.5 times that of the first quarter (K = 8)"),
    ("claim", "at K = 8 the cache works for the kept chat: at least 80 percent of its prompt tokens came from the cache"),
    ("claim", "at K = 8 against the chat with the cache emptied each round, exact-state computes at least 80 percent fewer prompt tokens"),
    ("claim", "at K = 8 with the cache on, exact-state computes at least 30 percent fewer prompt tokens than the kept chat"),
    ("claim", "at K = 8 with the cache on, exact-state takes at least 20 percent less GPU time per step than the kept chat"),
    ("claim", "at K = 64 the kept chat overruns the cache: in the second half of the steps less than 50 percent of its prompt tokens came from the cache"),
    ("claim", "at K = 64 the kept chat takes at least twice as much GPU time per step as exact-state"),
    ("claim", "at K = 32 the kept chat takes at least 1.3 times as much GPU time per step as exact-state"),
]
PL_HEADLINE = (10, 12)


# ---------------------------------------------------------------- what a session asks and what it gets back
@dataclass
class Request:
    messages: List[dict]
    stand_in_reply: str = ""        # read only by StandInEngine (the reply the oracle would write); a real engine ignores it


@dataclass
class Reply:
    text: str
    n_prompt: int
    n_cached: Optional[int]
    n_new: int


class Session:
    """One job: a fixed script of requests made on the oracle's graph (docs/LONG_HORIZON_TEST.md). `S` shows the exact state each step, `KC` and `KN` keep the whole chat."""
    def __init__(self, arm: str, seed: int, steps: int = STEPS, hint_rules: bool = HINT_RULES):
        assert arm in ARMS, arm
        self.arm, self.seed, self.hint = arm, seed, hint_rules
        self.g0, self.script, self.oracle = oracle_script(seed, steps)
        self.live = copy_graph(self.g0); self.history: List[dict] = []; self.log: List[dict] = []; self.stopped: Optional[str] = None; self.t = 0; self._base = None

    @property
    def done(self) -> bool: return self.stopped is not None or self.t >= len(self.script)

    def request(self) -> Request:
        st = self.script[self.t]
        task = Task(st["kind"], copy_graph(self.live), st["instruction"], st["ideal"], self.oracle[self.t + 1])
        base = base_messages(task, self.hint); self._base = base
        return Request([base[0]] + (self.history if self.arm != "S" else []) + [base[1]], st["ideal"])

    def accept(self, rep: Reply) -> None:
        st = self.script[self.t]; d = strict_parse(rep.text); after = next_state(POLICY, self.live, d); want = self.oracle[self.t + 1]
        self.log.append(dict(t=self.t + 1, kind=st["kind"], text=rep.text, n_prompt=rep.n_prompt, n_cached=rep.n_cached, n_new=rep.n_new, parsed=d is not None,
                             admitted=bool(d is not None and check(self.live, d).ok), corrupted=not after.is_well_formed(), exact=same_graph(after, want), distance=distance(after, want)))
        self.live = after; self.t += 1
        if self.arm != "S":
            self.history += [self._base[1], {"role": "assistant", "content": rep.text}]
            if self.t < len(self.script) and rep.n_prompt * (len(self.log) + 1) / len(self.log) > MAX_CONTEXT: self.stopped = "context"      # the same rule as the long-horizon history mode


# ---------------------------------------------------------------- engines
_TOK = re.compile(r"[0-9]|[A-Za-z]+|[^\w\s]|\n")


def _toks(text: str) -> List[str]: return _TOK.findall(text)


class StandInEngine:
    """A model of a serving engine for testing the harness without a GPU. Tokens are counted by a regular expression (digits one by one, like Qwen's tokenizer), the cache is a table of full
    blocks of 16 tokens chained by their hashes, the cache holds `capacity_tokens`, a sequence frees its blocks tail first, and the oldest freed block is evicted first. Time is
    computed_prompt_tokens / prefill_tps + longest_reply * decode_step_s. Never a result."""
    kind = "stand_in"

    def __init__(self, capacity_tokens: int = CAPACITY_TOKENS, prefill_tps: float = 1500.0, decode_step_s: float = 0.03, block: int = BLOCK):
        self.block, self.cap, self.prefill_tps, self.decode_step_s = block, capacity_tokens // block, prefill_tps, decode_step_s
        self.capacity_tokens = self.cap * block; self.cache: "OrderedDict[int, bool]" = OrderedDict()

    def info(self) -> dict:
        return dict(kind=self.kind, name="stand-in (block cache, oldest freed first)", version="n/a", gpu="none", capacity_tokens=self.capacity_tokens, capacity_source="set", block_size=self.block,
                    kv_bytes_per_token=57344, model=MODEL, python=platform.python_version())

    def reset_cache(self) -> None: self.cache.clear()

    def warm_up(self) -> None: pass

    def _render(self, messages: List[dict]) -> List[str]:
        out: List[str] = []
        for m in messages: out += ["<|im_start|>", m["role"], "\n"] + _toks(m["content"]) + ["<|im_end|>", "\n"]
        return out + ["<|im_start|>", "assistant", "\n"]

    def _keys(self, toks: List[str]) -> List[int]:
        keys, prev = [], 0
        for i in range(0, len(toks) - len(toks) % self.block, self.block):
            prev = hash((prev, tuple(toks[i:i + self.block]))); keys.append(prev)
        return keys

    def generate_round(self, requests: List[Request], reset_cache: bool = False) -> Tuple[List[Reply], float]:
        if reset_cache: self.reset_cache()
        reps, computed, longest = [], 0, 0
        for rq in requests:
            prompt = self._render(rq.messages); n = len(prompt); reply = _toks(rq.stand_in_reply) + ["<|im_end|>"]
            keys = self._keys(prompt + reply); hit = 0
            for k in keys[:n // self.block]:
                if k in self.cache: hit += 1
                else: break
            cached = min(hit * self.block, n - 1)                       # the last prompt token is always computed
            for k in keys[:hit]: self.cache.move_to_end(k)              # a hit pins and refreshes the blocks it uses
            for k in keys[hit:]:
                if k in self.cache: self.cache.move_to_end(k)
                else:
                    while len(self.cache) >= self.cap: self.cache.popitem(last=False)
                    self.cache[k] = True
            for k in reversed(keys):                                   # freed tail first, so the tail is evicted before the head (a sequence longer than the whole cache keeps only its end)
                if k in self.cache: self.cache.move_to_end(k)
            computed += n - cached; longest = max(longest, len(reply)); reps.append(Reply(rq.stand_in_reply, n, cached, len(reply)))
        return reps, computed / self.prefill_tps + longest * self.decode_step_s


class VLLMEngine:
    """vLLM on one GPU. Not testable without a GPU; tests/test_provider_side.py runs it against a fake `vllm` module that has the calls this class makes, and the Colab notebook starts with a
    smoke run (3 jobs, 3 steps) that shows within two minutes whether the installed version has them."""
    kind = "vllm"

    def __init__(self, model: str = MODEL, capacity_blocks: Optional[int] = CAPACITY_BLOCKS, max_model_len: int = MAX_MODEL_LEN, gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION):
        import vllm
        from vllm import LLM, SamplingParams
        self.model, self.vllm_version = model, getattr(vllm, "__version__", "unknown")
        kw = dict(model=model, dtype="half", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0)
        self.capacity_source = "override"
        try: self.llm = LLM(num_gpu_blocks_override=capacity_blocks, **kw) if capacity_blocks else LLM(**kw)
        except TypeError:
            self.llm = LLM(**kw); self.capacity_source = "engine"                  # this version has no block override: the size the engine chose is read below
        self.tok = self.llm.get_tokenizer(); self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
        self.block_size = self._cache_attr("block_size") or BLOCK
        blocks = capacity_blocks if self.capacity_source == "override" else self._cache_attr("num_gpu_blocks")
        self.capacity_tokens = blocks * self.block_size if blocks else None
        if self.capacity_tokens is None: self.capacity_source = "unknown"

    def _cache_attr(self, name):
        eng = self.llm.llm_engine
        for path in (("cache_config",), ("vllm_config", "cache_config")):
            obj = eng
            try:
                for p in path: obj = getattr(obj, p)
                v = getattr(obj, name, None)
                if v: return v
            except AttributeError: continue
        return None

    def _kv_bytes(self) -> Optional[int]:
        try:
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(self.model); hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
            return 2 * c.num_hidden_layers * c.num_key_value_heads * hd * 2
        except Exception: return None

    def info(self) -> dict:
        gpu = "unknown"
        try: import torch; gpu = torch.cuda.get_device_name(0)
        except Exception: pass
        return dict(kind=self.kind, name=self.model, version=self.vllm_version, gpu=gpu, capacity_tokens=self.capacity_tokens, capacity_source=self.capacity_source, block_size=self.block_size,
                    kv_bytes_per_token=self._kv_bytes(), model=self.model, python=platform.python_version())

    def reset_cache(self) -> None:
        if self.llm.reset_prefix_cache() is False: raise RuntimeError("the engine could not empty its prefix cache (it reported False); a cell would then not start cold")      # None (an older version that returns nothing) is taken as done

    def warm_up(self) -> None:
        self.llm.generate([self.tok.apply_chat_template([{"role": "user", "content": "Reply with the word ok."}], tokenize=False, add_generation_prompt=True)], self.sp, use_tqdm=False)
        self.reset_cache()

    def generate_round(self, requests: List[Request], reset_cache: bool = False) -> Tuple[List[Reply], float]:
        if reset_cache: self.reset_cache()
        prompts = [self.tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True) for r in requests]
        t0 = time.perf_counter(); outs = self.llm.generate(prompts, self.sp, use_tqdm=False); dt = time.perf_counter() - t0
        return [Reply(o.outputs[0].text, len(o.prompt_token_ids), getattr(o, "num_cached_tokens", None), len(o.outputs[0].token_ids)) for o in outs], dt


# ---------------------------------------------------------------- one cell: K jobs in lockstep under one arm
def cells_planned() -> List[Tuple[str, int]]:
    """In the order they are run: the light load first, the heavy one last."""
    out = []
    for k in LEVELS:
        out += [("S", k), ("KC", k)] + ([("KN", k)] if k in KN_LEVELS else [])
    return out + [("KR", k) for k in KR_LEVELS]


def run_cell(engine, arm: str, K: int, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    assert arm in ARMS and K >= 1
    sess = [Session(arm, seed * 1000 + e, steps) for e in range(K)]
    engine.reset_cache(); rounds, wall = [], 0.0                          # every cell starts with a cold cache
    order = random.Random(seed)
    while True:
        active = [s for s in sess if not s.done]
        if not active: break
        if arm == "KR": order.shuffle(active)                             # the same jobs, served in another order each round
        reps, secs = engine.generate_round([s.request() for s in active], reset_cache=(arm == "KN"))
        assert len(reps) == len(active)
        for s, r in zip(active, reps): s.accept(r)
        cached = None if any(r.n_cached is None for r in reps) else sum(r.n_cached for r in reps)
        rounds.append(dict(round=len(rounds) + 1, active=len(active), seconds=secs, prompt_tokens=sum(r.n_prompt for r in reps), cached_tokens=cached, new_tokens=sum(r.n_new for r in reps)))
        wall += secs
        if progress: progress(len(rounds), arm, K)
    return dict(arm=arm, K=K, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(), wall_seconds=wall, rounds=rounds,
                sessions=[dict(seed=s.seed, script_steps=len(s.script), stopped=s.stopped, log=s.log) for s in sess])


# ---------------------------------------------------------------- summaries
def _mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else None


def summarize_cell(c: dict) -> dict:
    sess = c["sessions"]; steps = [x for s in sess for x in s["log"]]; n = len(steps); T = max((len(s["log"]) for s in sess), default=0); q = max(T // 4, 1); half = T // 2
    prompt, new = sum(x["n_prompt"] for x in steps), sum(x["n_new"] for x in steps)
    have = n > 0 and all(x["n_cached"] is not None for x in steps); cached = sum(x["n_cached"] for x in steps) if have else None
    late = [x for s in sess for x in s["log"][half:]]; late_have = bool(late) and all(x["n_cached"] is not None for x in late)
    win = lambda lo, hi: _mean(x["n_prompt"] for s in sess for x in s["log"][lo:hi])
    wall = c["wall_seconds"]
    return dict(arm=c["arm"], K=c["K"], engine_kind=c["engine"]["kind"], capacity_tokens=c["engine"].get("capacity_tokens"), steps_done=n, episodes_done=sum(1 for s in sess if s["stopped"] is None and len(s["log"]) == s["script_steps"]),
                stopped=[s["stopped"] for s in sess], wall_seconds=wall, seconds_per_step=(wall / n if n else None), steps_per_hour=(3600.0 * n / wall if wall else None),
                prompt_tokens=prompt, new_tokens=new, cached_tokens=cached, computed_prefill=(prompt - cached if have else None), computed_per_step=((prompt - cached) / n if have else None),
                cached_share=(cached / prompt if have and prompt else None), cached_share_second_half=(sum(x["n_cached"] for x in late) / sum(x["n_prompt"] for x in late) if late_have else None),
                prompt_first_quarter=win(0, q), prompt_last_quarter=win(T - q, T) if T else None, longest_prompt=max((x["n_prompt"] for x in steps), default=None),
                exact_share=_mean(x["exact"] for x in steps), admitted_share=_mean(x["admitted"] for x in steps), parsed_share=_mean(x["parsed"] for x in steps), corrupted_steps=sum(x["corrupted"] for x in steps))


def summarize_all(cells: List[dict]) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for c in cells:
        k = (c["arm"], c["K"]); assert k not in out, f"cell {k} twice"; out[k] = summarize_cell(c)
    return out


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings, a cell twice, and cells from the stand-in mixed with real ones."""
    assert cells, "nothing to merge"
    ref = cells[0]; key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"))
    seen = set()
    for c in cells:
        if key(c) != key(ref): raise ValueError(f"cell {c['arm']}/K={c['K']} was run under other settings: {key(c)} against {key(ref)}")
        if (c["arm"], c["K"]) in seen: raise ValueError(f"cell {c['arm']}/K={c['K']} twice")
        seen.add((c["arm"], c["K"]))
    order = {k: i for i, k in enumerate(cells_planned())}
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], levels=list(LEVELS), kn_levels=list(KN_LEVELS)),
                cells=sorted(cells, key=lambda c: (order.get((c["arm"], c["K"]), 99), c["K"], ARMS.index(c["arm"]))))


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_TEST.md before the run)
EPS = 1e-9                  # a figure exactly on a threshold counts as reaching it (1 - 0.8 is 0.19999999999999996 in floating point)


def _ratio(a, b): return None if a is None or b is None or b == 0 else a / b
def _le(a, b): return a <= b + EPS
def _ge(a, b): return a >= b - EPS
def _lt(a, b): return a < b - EPS


def verdict_provider(S: Dict[Tuple[str, int], dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `S` is summarize_all's table. A line that needs a cell that was not run is None, never 'no'."""
    S = {k: v for k, v in S.items() if v["steps_done"] > 0}                                                                                                                  # a cell with no step has nothing to read: it counts as not run
    g = lambda arm, k: S.get((arm, k))
    out: List[Optional[bool]] = []
    cached_all = [s["cached_tokens"] is not None for s in S.values()]
    out.append(all(cached_all) if cached_all else None)                                                                                                                      # 1
    kc8, kc32, kc64, s8, s32, s64, n8 = g("KC", 8), g("KC", 32), g("KC", 64), g("S", 8), g("S", 32), g("S", 64), g("KN", 8)
    out.append(None if kc8 is None or kc64 is None or not kc8["capacity_tokens"] or not kc64["capacity_tokens"] else
               (64 * kc64["longest_prompt"] >= 2 * kc64["capacity_tokens"] and 8 * kc8["longest_prompt"] <= kc8["capacity_tokens"]))                                          # 2
    out.append(all(_ge(s["exact_share"], PL_MIN_EXACT) for s in S.values()) if S else None)                                                                                   # 3
    pairs = [(g("S", k), g("KC", k)) for k in LEVELS if g("S", k) and g("KC", k)]
    out.append(all(_ge(a["exact_share"], b["exact_share"] - PL_EXACT_GAP) for a, b in pairs) if pairs else None)                                                             # 4
    r = _ratio(s8["prompt_last_quarter"], s8["prompt_first_quarter"]) if s8 else None; out.append(None if r is None else _le(r, PL_FLAT))                                      # 5
    r = _ratio(kc8["prompt_last_quarter"], kc8["prompt_first_quarter"]) if kc8 else None; out.append(None if r is None else _ge(r, PL_GROWTH))                                  # 6
    out.append(None if kc8 is None or kc8["cached_share"] is None else _ge(kc8["cached_share"], PL_CACHE_FITS))                                                                # 7
    r = _ratio(s8["computed_per_step"], n8["computed_per_step"]) if s8 and n8 else None; out.append(None if r is None else _le(r, 1 - PL_NOCACHE_SAVING))                      # 8
    r = _ratio(s8["computed_per_step"], kc8["computed_per_step"]) if s8 and kc8 else None; out.append(None if r is None else _le(r, 1 - PL_LIGHT_TOKENS))                       # 9
    r = _ratio(s8["seconds_per_step"], kc8["seconds_per_step"]) if s8 and kc8 else None; out.append(None if r is None else _le(r, 1 - PL_LIGHT_TIME))                          # 10
    out.append(None if kc64 is None or kc64["cached_share_second_half"] is None else _lt(kc64["cached_share_second_half"], PL_OVERRUN))                                       # 11
    r = _ratio(kc64["seconds_per_step"], s64["seconds_per_step"]) if kc64 and s64 else None; out.append(None if r is None else _ge(r, PL_PRESSURE_RATIO))                      # 12
    r = _ratio(kc32["seconds_per_step"], s32["seconds_per_step"]) if kc32 and s32 else None; out.append(None if r is None else _ge(r, PL_MID_RATIO))                           # 13
    assert len(out) == len(PL_LINES)
    return [(i + 1, PL_LINES[i][0], PL_LINES[i][1], out[i]) for i in range(len(PL_LINES))]


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[Tuple[str, int], dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.0f}%"
    print(f"\n{'arm':3s} {'K':>3s} | {'steps':>5s} {'seconds':>8s} {'s/step':>7s} | {'prompt tok':>11s} {'computed':>10s} {'cached':>7s} {'late cached':>11s} | {'prompt q1':>9s} {'q4':>6s} {'longest':>8s} | {'exact':>6s}")
    for (arm, k), s in sorted(S.items(), key=lambda kv: (kv[0][1], ARMS.index(kv[0][0]))):
        print(f"{arm:3s} {k:3d} | {s['steps_done']:5d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 2):>7s} | {s['prompt_tokens']:11d} {f(s['computed_prefill'], 0):>10s} {pc(s['cached_share']):>7s} {pc(s['cached_share_second_half']):>11s} | "
              f"{f(s['prompt_first_quarter'], 0):>9s} {f(s['prompt_last_quarter'], 0):>6s} {s['longest_prompt'] or 0:8d} | {pc(s['exact_share']):>6s}")


def print_verdict_provider(checks, title="provider-side test") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {PL_HEADLINE[0]} and {PL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in PL_HEADLINE) + ")")


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


# ---------------------------------------------------------------- command line
PROXY_SCALE = 1.12        # the regular-expression token count of the stand-in is about 12 % above Qwen's (checked on the saved long-horizon runs: 450 against 395, 5,747 against 5,230 tokens)


def dry_run(steps: int = STEPS, levels=None) -> Dict[Tuple[str, int], dict]:
    """Every planned cell on the stand-in, with the cache sized to match the proxy's token count. Prints the tables and the lines. NOT a result."""
    eng = StandInEngine(capacity_tokens=int(CAPACITY_TOKENS * PROXY_SCALE)); cells = []
    for arm, k in cells_planned():
        if levels is None or k in levels: cells.append(run_cell(eng, arm, k, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache), NOT a result"); print_cells(S); print_verdict_provider(verdict_provider(S), "STAND-IN, not a result"); return S


def merge_files(paths: List[str], out: str) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells); m["module_sha256"] = fingerprint()
    with open(out, "w") as f: json.dump(m, f)
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test (docs/PROVIDER_SIDE_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_side.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_verdict_provider(verdict_provider(S)); print("\nwritten to", a.out)
    else: ap.print_help()

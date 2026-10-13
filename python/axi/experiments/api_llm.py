"""Model adapter for models reached through an API (docs/API_TEST_GEMINI.md; written BEFORE any API number exists).

The runners (`run_scope_regime`, `run_long`) ask a model object for `generate(messages, constrained) -> (text, n_tokens, seconds)` and read `last_prompt_tokens`. This file gives them
that for a hosted model, and deals with the four things a hosted free tier adds:

  1. a daily request limit: `DiskCachedLLM` counts the calls made today (Pacific time, the day Google resets on) and raises `QuotaStop` BEFORE the limit is passed; the runners have already saved
     every finished unit, so running the same cell again the next day carries on;
  2. a per-minute limit: calls are spaced `min_interval` seconds apart, and a "too many per minute" answer is waited out and retried (a "per day" answer is a stop, never a retry);
  3. a busy server (503 "high demand"): Google appears to COUNT such failed tries against the free daily limit (seen on 10 Oct 2026: a model's 20 free requests were used up by tries that all failed), so
     a busy answer counts against the day's budget like any other call, is retried only twice and slowly (after 2 and after 5 minutes), and if the service is still busy `ServiceBusy` stops the run with a
     plain message. Running a cell again and again while the service is busy costs real quota;
  4. nothing is paid for twice: every answer is saved to a file the moment it arrives (key = the prompt). A restarted run that asks the same prompt again gets the saved answer and makes NO call. This
     also keeps the exact raw text of every output for the record, and a model that is sampled (not greedy) is asked ONCE per prompt.

There is no grammar mask on an API, so a reply can be badly formed. `clean_text` does only what the mask used to do for free: remove one code fence around the whole reply and make every run of white space a
single space (a delta item has no space inside it). Nothing else is changed; a reply that still does not parse counts as unparsed, for every policy alike. The raw text is kept next to the cleaned text.

The API key is held in memory by the backend and is never written to any file, record, header line or print by this module."""
import hashlib, json, os, re, time
from datetime import datetime, timedelta, timezone


class QuotaStop(Exception):
    """The run must stop for today (or for good) and can be started again with the same arguments."""


class BudgetStop(QuotaStop):
    """The money budget the person typed in is used up (counted from the saved answers and the prices the person typed in). Nothing is lost; raise the budget to go on."""


class ServiceBusy(QuotaStop):
    """The service answered 'overloaded / unavailable' (503 and the like) every time we asked, for several minutes. Not a quota: try again in 15 to 30 minutes, the same day."""


# ---------------------------------------------------------------- the text
def clean_text(raw):
    """(cleaned, changed). Removes one ``` fence around the whole reply and collapses white space to single spaces. `changed` is True when this altered more than the ends of the text."""
    base = (raw or "").strip(); t = base
    m = re.fullmatch(r"```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", t, flags=re.S) or re.fullmatch(r"```([^`\n]*)```", t)
    if m: t = m.group(1).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t, t != base


# ---------------------------------------------------------------- the clock
def pacific_day(ts):
    """The calendar day (YYYY-MM-DD) in Pacific time, where Google's daily quota resets. Falls back to a fixed UTC-8 if the time-zone database is missing (then the day may start an hour early)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(ts, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.fromtimestamp(ts, timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- errors
def classify_error(e):
    """'per_day' (stop), 'per_minute' (wait and retry), 'retry' (a server hiccup: retry a few times) or 'fatal' (raise)."""
    msg = str(e).lower(); flat = re.sub(r"[\s_]+", "", msg)
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code == 429 or "resource_exhausted" in msg or re.search(r"\b429\b", msg) or "rate limit" in msg or "quota" in msg:
        if "perday" in flat or "daily" in flat: return "per_day"
        return "per_minute"
    if code in (500, 502, 503, 504) or any(w in msg for w in ("unavailable", "overloaded", "deadline", "timed out", "timeout", "internal error", "connection")): return "retry"
    return "fatal"


def retry_delay(e, default=30.0, cap=120.0):
    """Seconds the service asked us to wait ('retry in 23.4s', 'retryDelay: 23s'), else `default`; never more than `cap`."""
    m = re.search(r"retry(?:delay|\s+in)?\W{0,8}(\d+(?:\.\d+)?)\s*s", str(e), flags=re.I)
    return min(float(m.group(1)) + 1.0, cap) if m else default


# ---------------------------------------------------------------- the cached, paced, quota-aware model
class DiskCachedLLM:
    """backend.call(messages) -> dict(text, tokens, think_tokens, prompt_tokens, finish). One JSON line per answer in `path`; the first line records which model and settings the file belongs to."""
    def __init__(self, backend, path, meta, daily_limit=None, min_interval=0.0, max_waits=6, clock=None, sleep=None, on_write=None, say=print, max_retries=None, retry_waits=(120.0, 300.0), budget_usd=None, price_in=None, price_out=None):
        if budget_usd is not None and (price_in is None or price_out is None): raise ValueError("a budget needs both prices (dollars per million input tokens and per million output tokens, thinking included)")
        self.budget_usd, self.price_in, self.price_out = budget_usd, price_in, price_out
        self.backend, self.path, self.meta = backend, path, dict(meta)
        self.daily_limit, self.min_interval, self.max_waits = daily_limit, float(min_interval), max_waits
        self.retry_waits = tuple(float(w) for w in retry_waits); self.max_retries = len(self.retry_waits) if max_retries is None else max_retries
        self.clock, self.sleep, self.on_write, self.say = clock or time.time, sleep or time.sleep, on_write, say          # looked up when the object is made, so a test can replace the clock
        self.cache, self.events, self.busy_events = {}, [], []; self.last_prompt_tokens = None; self.last_think_tokens = 0; self.from_disk = False; self.calls_this_session = 0; self._last_call = None
        self._load()

    # --- file
    def _load(self):
        good = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for k, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try: obj = json.loads(line)
                    except json.JSONDecodeError: break                                    # half-written last line: drop it
                    good.append(line)
                    if k == 0:
                        if obj.get("_meta") != json.loads(json.dumps(self.meta)): raise ValueError(f"{self.path} belongs to a different model or setting: {obj.get('_meta')} vs {self.meta}. Use another file name.")
                    elif obj.get("fail"):
                        self.events.append(obj["ts"])                                          # a failed call counts: the service counts it
                        if obj["fail"] == "retry": self.busy_events.append(obj["ts"])          # (a busy-server answer is also listed on its own, for the report)
                    else: self.cache[obj["k"]] = obj; self.events.append(obj["ts"])
        with open(self.path, "w") as f:
            if not good: f.write(json.dumps({"_meta": self.meta}) + "\n")
            for line in good: f.write(line + "\n")

    def _append(self, rec):
        with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        if self.on_write: self.on_write()

    # --- accounting
    def calls_today(self):
        """Calls (answered or failed) made on today's Pacific day. A failed call counts: the service counts it."""
        today = pacific_day(self.clock()); return sum(1 for ts in self.events if pacific_day(ts) == today)

    def usage(self):
        """Totals over every saved answer, for the page: calls, prompt / visible / thinking tokens, seconds, answers that needed cleaning, empty answers."""
        rs = list(self.cache.values()); n = len(rs)
        return dict(calls=n, prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in rs), visible_tokens=sum(r.get("tokens") or 0 for r in rs), thinking_tokens=sum(r.get("think") or 0 for r in rs),
                    seconds=sum(r.get("seconds") or 0.0 for r in rs), cleaned=sum(1 for r in rs if r.get("normalized")), empty=sum(1 for r in rs if not r.get("text")), failed_calls=len(self.events) - n, busy_answers=len(self.busy_events))

    def spent_usd(self):
        """Dollars the saved answers cost at the typed-in prices (prompt tokens at the input price, visible and thinking tokens at the output price); None if no prices were typed in."""
        if self.price_in is None or self.price_out is None: return None
        rs = self.cache.values()
        return (sum(r.get("prompt_tokens") or 0 for r in rs) * self.price_in + sum((r.get("tokens") or 0) + (r.get("think") or 0) for r in rs) * self.price_out) / 1e6

    @staticmethod
    def key(messages): return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    # --- the call
    def retry_wait(self, n):
        """Seconds to wait before retry number n (1, 2, ...) after a busy-server answer: 120 s, then 300 s (the last one again if there are more)."""
        return self.retry_waits[min(n, len(self.retry_waits)) - 1]

    def retry_wait_total(self): return sum(self.retry_wait(n) for n in range(1, self.max_retries + 1))

    def generate(self, messages, constrained=True, max_new_tokens=400):
        k = self.key(messages); rec = self.cache.get(k)
        if rec is not None:
            self.from_disk = True; self.last_prompt_tokens = rec.get("prompt_tokens"); self.last_think_tokens = rec.get("think") or 0; return rec["text"], rec.get("tokens") or 0, rec.get("seconds") or 0.0
        self.from_disk = False; waits = 0; retries = 0
        while True:
            if self.daily_limit is not None and self.calls_today() >= self.daily_limit:
                raise QuotaStop(f"Daily budget reached: {self.calls_today()} calls today (budget {self.daily_limit}). Everything finished is saved. Run this same cell again after the daily reset.")
            if self.budget_usd is not None and self.spent_usd() >= self.budget_usd:
                raise BudgetStop(f"Budget reached: about ${self.spent_usd():.2f} of the ${self.budget_usd:.2f} you typed in is spent on this model (counted from the saved answers and your prices). Everything finished is saved. To go on, raise BUDGET_USD in the settings cell, run that cell again, and run the cell you were on.")
            if self._last_call is not None:
                gap = self.min_interval - (self.clock() - self._last_call)
                if gap > 0: self.sleep(gap)
            self._last_call = self.clock(); t0 = self.clock()
            try: res = self.backend.call(messages)
            except QuotaStop: raise
            except Exception as e:
                kind = classify_error(e); now = self.clock(); self.events.append(now); self._append({"fail": kind, "ts": now})
                if kind == "per_day": raise QuotaStop("The service says the DAILY limit is used up. Everything finished is saved. Run this same cell again after the daily reset.") from e
                if kind == "per_minute":
                    waits += 1
                    if waits > self.max_waits: raise QuotaStop(f"Still told 'too many requests per minute' after {self.max_waits} waits. Everything finished is saved. Wait a few minutes and run this cell again.") from e
                    d = retry_delay(e); self.say(f"  rate limited (per minute): waiting {d:.0f} s"); self.sleep(d); continue
                if kind == "retry":
                    retries += 1; self.busy_events.append(now)
                    if retries > self.max_retries: raise ServiceBusy(f"The service is busy or not reachable (it failed {retries} times in a row, over about {self.retry_wait_total() / 60:.0f} minutes: {type(e).__name__}). Not your key. Everything finished is saved. Wait at least 30 minutes, then run this same cell ONCE. Do not run it again and again: Google counts failed tries against the free daily limit.") from e
                    d = self.retry_wait(retries); self.say(f"  server busy ({type(e).__name__}): retrying in {d:.0f} s"); self.sleep(d); continue
                raise
            sec = self.clock() - t0; text, changed = clean_text(res.get("text"))
            rec = dict(k=k, ts=self.clock(), text=text, raw=res.get("text") or "", normalized=changed, tokens=res.get("tokens") or 0, think=res.get("think_tokens") or 0,
                       prompt_tokens=res.get("prompt_tokens"), seconds=sec, finish=str(res.get("finish")))
            self.cache[k] = rec; self.events.append(rec["ts"]); self._append(rec); self.calls_this_session += 1; self.last_prompt_tokens = rec["prompt_tokens"]; self.last_think_tokens = rec["think"]
            return text, rec["tokens"], sec


# ---------------------------------------------------------------- Gemini
EXCLUDE = ("tts", "image", "live", "audio", "embedding", "transcribe", "robotics", "computer", "customtools", "translate", "veo", "lyria", "omni")


def resolve_model(names, wanted):
    """Pick the model id to call. `names` as listed by the service ('models/gemini-3.5-flash-lite-preview-09-2026'). An exact id wins; else the shortest id that starts with `wanted`, skipping speech,
    image, live and other special variants. Raises with the near matches when nothing fits."""
    ids = sorted({n.split("/", 1)[-1] for n in names}); wanted = wanted.split("/", 1)[-1]
    if wanted in ids: return wanted
    cands = [i for i in ids if i.startswith(wanted) and not any(x in i for x in EXCLUDE)]
    if cands: return sorted(cands, key=lambda s: (len(s), s))[0]
    near = [i for i in ids if wanted.split("-flash")[0] in i or "flash" in i][:15]
    raise ValueError(f"No model called '{wanted}' is listed for this key. Closest listed: {near}")


class GeminiBackend:
    """Calls google-genai. `thinking` None = the model's own default; a word ('minimal', 'low', 'medium', 'high') sets thinking_level; an int sets thinking_budget. `temperature` None = the model's default (Google
    asks to leave Gemini 3 models at 1.0). `client` and `types` can be passed in (tests do); the key is only handed to the SDK."""
    def __init__(self, api_key=None, model="gemini-3.5-flash-lite", thinking=None, temperature=None, max_output_tokens=8192, client=None, types=None):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        if types is None:
            from google.genai import types
        self.client, self.types, self.model, self.thinking, self.temperature, self.max_output_tokens = client, types, model, thinking, temperature, max_output_tokens

    def config_dict(self):
        """The settings that make a file belong to one run (no key)."""
        return dict(provider="gemini", model=self.model, thinking=self.thinking, temperature=self.temperature, max_output_tokens=self.max_output_tokens)

    def call(self, messages):
        roles = {m["role"] for m in messages}
        if not roles <= {"system", "user"}: raise ValueError("only system and user messages are supported (state mode); history mode is not run over an API")
        system = "\n".join(m["content"] for m in messages if m["role"] == "system"); user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        kw = dict(system_instruction=system, max_output_tokens=self.max_output_tokens)
        if self.temperature is not None: kw["temperature"] = self.temperature
        if self.thinking is not None:
            kw["thinking_config"] = self.types.ThinkingConfig(thinking_level=self.thinking) if isinstance(self.thinking, str) else self.types.ThinkingConfig(thinking_budget=int(self.thinking))
        resp = self.client.models.generate_content(model=self.model, contents=user, config=self.types.GenerateContentConfig(**kw))
        try: text = resp.text or ""
        except Exception: text = ""                                                               # a blocked or empty reply has no text
        um = getattr(resp, "usage_metadata", None); g = lambda name: (getattr(um, name, None) or 0) if um is not None else 0
        cands = getattr(resp, "candidates", None) or []; finish = getattr(cands[0], "finish_reason", None) if cands else None
        return dict(text=text, tokens=g("candidates_token_count"), think_tokens=g("thoughts_token_count"), prompt_tokens=g("prompt_token_count") or None, finish=finish)


def list_model_names(client):
    return [m.name for m in client.models.list()]


def build_gemini_llm(api_key, wanted, cache_path, daily_limit, min_interval, on_write=None, client=None, types=None, thinking=None, temperature=None, max_output_tokens=8192, say=print, **kw):
    """The whole setup in one call: list the models the key can see, pick the id for `wanted`, and return (llm, model_id). The id is printed. `cache_path` holds every reply; it is refused if it belongs to another
    model or other settings. `client` and `types` are for tests."""
    if client is None:
        from google import genai
        client = genai.Client(api_key=api_key)
    model = resolve_model(list_model_names(client), wanted); say(f"model id: {model}")
    backend = GeminiBackend(model=model, thinking=thinking, temperature=temperature, max_output_tokens=max_output_tokens, client=client, types=types)
    return DiskCachedLLM(backend, cache_path, backend.config_dict(), daily_limit=daily_limit, min_interval=min_interval, on_write=on_write, say=say, **kw), model

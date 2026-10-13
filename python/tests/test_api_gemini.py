"""Implements concept-map rows 3, 9, 27 (item 86): the instrument of docs/API_TEST_GEMINI.md, written BEFORE any API number exists. It checks the INSTRUMENT, not a result:
  1. `clean_text` does only what the grammar mask did for free (one code fence round the whole reply, white space to single spaces) and nothing else; prose stays prose and does not parse;
  2. errors are told apart: a "per day" answer is a STOP, a "per minute" answer is waited out and retried a few times, a server hiccup is retried, anything else is raised at once; the wait the service asks for is read;
  3. the quota day is the PACIFIC day (it changes at 07:00 UTC until 1 November, 08:00 UTC after);
  4. `DiskCachedLLM`: every reply is saved the moment it arrives; a restart that meets the same prompt makes no call; a file of another model or setting is refused; a half-written last line is dropped;
     the daily budget stops the run BEFORE a call that would pass it (failed calls count), cached prompts are still answered at the limit, and the next Pacific day starts again; calls are spaced `min_interval` apart;
  5. `GeminiBackend` passes the system text as the system instruction, the thinking setting as thinking_level / thinking_budget, the temperature only when set, and reads the token counts; only system and user messages;
  6. `resolve_model` / `build_gemini_llm` pick the right id (skipping speech, image and live variants) and say so; the key goes only to the SDK and into no file, no record and no print;
  7. A RUN THAT STOPS AT THE DAILY LIMIT IN THE MIDDLE AND IS STARTED AGAIN THE NEXT DAY returns exactly what an uninterrupted run returns, and no prompt is ever asked twice (single requests and long episodes);
  8. the new reading rules (`verdict_api_format`, `verdict_pivotal`, `smoke_report`) read as the page says, at their boundaries;
  9. the notebook runs the same test as the 14B notebook (same 140 tasks, seed, episodes, steps, policies, mix), the page and the notebook agree on the models and the budgets, and the notebook runs end to end on a fake service, stops at the
     limit, and finishes the next day with the same rows as a run that never stopped.
What it does NOT show: that Google's service behaves like the fake one (the notebook's smoke cell prints the model id, the replies and the token counts so you can see), or anything about Gemini itself."""
import ast, contextlib, hashlib, io, json, os, re, shutil, sys, tempfile, time, types
from datetime import datetime, timezone
from unittest import mock
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import api_llm
from axi.experiments.agent_loop import base_messages
from axi.experiments.api_llm import DiskCachedLLM, GeminiBackend, QuotaStop, ServiceBusy, build_gemini_llm, classify_error, clean_text, pacific_day, resolve_model, retry_delay
from axi.experiments.api_test import smoke_report, unparsed_long, unparsed_single, verdict_api_format, verdict_pivotal
from axi.experiments.long_horizon import run_long
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.scope_test import run_scope_regime, summarize_scope
from test_scope import StandIn
from test_colab_persist import _fake_drive, patched_modules

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "API_TEST_GEMINI.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_test.ipynb")
NB_14B = os.path.join(HERE, "..", "colab", "axi_14b_test.ipynb")
KEY = "AIzaSyFAKE-key-0123456789abcdef"


def ts(y, m, d, h, mi=0): return datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp()


def read(path):
    with open(path) as f: return f.read()


class Clock:
    def __init__(self, t=None): self.t = ts(2026, 10, 9, 20) if t is None else t
    def now(self): return self.t
    def advance(self, s): self.t += s


class Err(Exception):
    def __init__(self, msg, code=None): super().__init__(msg); self.code = code


class Backend:
    """A stand-in service. `script` = what the next calls do: None = answer, an exception = raise it."""
    def __init__(self, clock, style="rule_like", latency=1.0, script=None):
        self.clock, self.inner, self.latency, self.script, self.n, self.starts = clock, StandIn(style), latency, list(script or []), 0, []

    def call(self, messages):
        self.n += 1; self.starts.append(self.clock.t)
        if self.script:
            a = self.script.pop(0)
            if a is not None: self.clock.advance(0.1); raise a
        text, ntok, sec = self.inner.generate(messages, True); self.clock.advance(self.latency)
        return dict(text=text, tokens=ntok, think_tokens=7, prompt_tokens=self.inner.last_prompt_tokens, finish="STOP")


def make(tmp, clock, backend, daily=None, interval=0.0, name="c.jsonl", meta=None, on_write=None, **kw):
    return DiskCachedLLM(backend, os.path.join(tmp, name), meta or dict(model="m", t=0), daily_limit=daily, min_interval=interval, clock=clock.now, sleep=clock.advance, on_write=on_write, say=lambda *a: None, **kw)


TASKS = make_scope_tasks(14, 7)
M = [base_messages(t, True) for t in TASKS]


# ---------------------------------------------------------------- 1: the text
def test_clean_text_does_only_what_the_mask_did_for_free():
    assert clean_text("DEL[1,0] DEL[(2,0)->(1,0):dep#2]") == ("DEL[1,0] DEL[(2,0)->(1,0):dep#2]", False)
    assert clean_text("  ADD[4,4:3]\n") == ("ADD[4,4:3]", False)                                   # the ends are not a change
    assert clean_text("DEL[1,0]\nDEL[1,1]") == ("DEL[1,0] DEL[1,1]", True) and clean_text("DEL[1,0]   ADD[2,2:1]") == ("DEL[1,0] ADD[2,2:1]", True)
    assert clean_text("```\nDEL[1,0]\n```") == ("DEL[1,0]", True) and clean_text("```text\nDEL[1,0]  ADD[2,2:1]\n```\n") == ("DEL[1,0] ADD[2,2:1]", True)
    assert clean_text("```DEL[1,0]```") == ("DEL[1,0]", True) and clean_text("```python\nDEL[1,0]```") == ("DEL[1,0]", True)
    assert clean_text("```\nDEL[1,0]") == ("``` DEL[1,0]", True)                                    # a fence that is not closed is not removed: it will not parse
    assert clean_text("Here is the delta: DEL[1,0]") == ("Here is the delta: DEL[1,0]", False)      # prose stays prose
    assert clean_text("") == ("", False) and clean_text(None) == ("", False)
    from axi.experiments.agent_loop import strict_parse
    assert strict_parse(clean_text("```\nDEL[1,0]\nDEL[(2,0)->(1,0):dep#2]\n```")[0]) is not None
    assert strict_parse(clean_text("Here is the delta: DEL[1,0]")[0]) is None and strict_parse(clean_text("")[0]) is None


# ---------------------------------------------------------------- 2: errors
def test_errors_are_told_apart_and_the_wait_is_read():
    day = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    minute = "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 23.4s."
    cases = [(Err(day, 429), "per_day"), (Err(minute, 429), "per_minute"), (Err("Too many requests, daily limit reached", 429), "per_day"), (Err("RESOURCE_EXHAUSTED: input tokens per minute", None), "per_minute"),
             (Err("503 UNAVAILABLE. The model is overloaded.", 503), "retry"), (Err("Read timed out"), "retry"), (Err("Connection reset by peer"), "retry"),
             (Err("API key not valid. Please pass a valid API key.", 400), "fatal"), (ValueError("something else"), "fatal"), (Err("403 permission denied", 403), "fatal"),
             (Err("boom", 502), "retry"), (Err("Rate limit exceeded"), "per_minute"), (Err("Quota exceeded for metric x"), "per_minute"), (Err("HTTP error 429 on request"), "per_minute")]
    for e, want in cases: assert classify_error(e) == want, (str(e), want, classify_error(e))
    assert abs(retry_delay(Err(minute)) - 24.4) < 1e-9 and abs(retry_delay(Err("{'retryDelay': '37s'}")) - 38.0) < 1e-9
    assert retry_delay(Err("no hint")) == 30.0 and retry_delay(Err("retry in 500s")) == 120.0


# ---------------------------------------------------------------- 3: the quota day
def test_the_quota_day_is_the_pacific_day():
    assert pacific_day(ts(2026, 10, 9, 6, 59)) == "2026-10-08" and pacific_day(ts(2026, 10, 9, 7, 0)) == "2026-10-09"          # PDT: midnight is 07:00 UTC (12:30 in India)
    assert pacific_day(ts(2026, 11, 2, 7, 59)) == "2026-11-01" and pacific_day(ts(2026, 11, 2, 8, 0)) == "2026-11-02"          # PST after 1 November: 08:00 UTC (13:30 in India)
    assert pacific_day(ts(2026, 10, 9, 20)) == "2026-10-09"


def test_the_pacific_day_falls_back_to_utc_minus_8_without_a_time_zone_database():
    import zoneinfo
    with mock.patch.object(zoneinfo, "ZoneInfo", side_effect=Exception("no tzdata")):
        assert pacific_day(ts(2026, 10, 9, 8, 0)) == "2026-10-09" and pacific_day(ts(2026, 10, 9, 7, 59)) == "2026-10-08"


# ---------------------------------------------------------------- 4: the cached, paced, quota-aware model
def test_every_reply_is_saved_and_a_restart_asks_nothing_twice():
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c); llm = make(tmp, c, be)
    r0 = llm.generate(M[0], True); assert be.n == 1 and not llm.from_disk and llm.last_prompt_tokens == be.inner.last_prompt_tokens and r0[2] == 1.0 and r0[1] == 8
    lines = read(llm.path).splitlines(); assert len(lines) == 2 and json.loads(lines[0]) == {"_meta": dict(model="m", t=0)}
    rec = json.loads(lines[1]); assert rec["text"] == r0[0] and rec["raw"] == r0[0] and rec["think"] == 7 and rec["prompt_tokens"] == llm.last_prompt_tokens and rec["finish"] == "STOP" and rec["normalized"] is False
    assert llm.generate(M[0], False) == r0 and be.n == 1 and llm.from_disk                          # same prompt: no call (and the `constrained` flag plays no part)
    r_no_rule = llm.generate(base_messages(TASKS[0], False), True); assert be.n == 2, "the same request with and without the rule in the system text is two prompts"
    be2 = Backend(c); llm2 = make(tmp, c, be2)                                                       # a restart (a new session)
    assert llm2.generate(M[0], True) == r0 and be2.n == 0 and llm2.from_disk and llm2.last_prompt_tokens == rec["prompt_tokens"]
    assert llm2.generate(base_messages(TASKS[0], False), True) == r_no_rule and be2.n == 0
    llm2.generate(M[1], True); assert be2.n == 1 and not llm2.from_disk and len(read(llm.path).splitlines()) == 4
    assert llm2.usage()["calls"] == 3 and llm2.usage()["thinking_tokens"] == 21 and llm2.usage()["visible_tokens"] == 24 and llm2.usage()["cleaned"] == 0 and llm2.usage()["empty"] == 0


def test_a_cleaned_reply_is_flagged_and_the_raw_text_is_kept():
    tmp = tempfile.mkdtemp(); c = Clock()
    class Fenced(Backend):
        def call(self, messages):
            d = super().call(messages); d["text"] = "```\n" + d["text"].replace(" ", "\n") + "\n```"; return d
    be = Fenced(c); llm = make(tmp, c, be); ideal = Backend(Clock()).call(M[0])["text"]
    text, _, _ = llm.generate(M[0], True); assert text == ideal
    rec = json.loads(read(llm.path).splitlines()[1]); assert rec["normalized"] is True and rec["raw"].startswith("```") and rec["text"] == ideal and llm.usage()["cleaned"] == 1


def test_a_file_of_another_model_is_refused_and_a_half_written_line_is_dropped():
    tmp = tempfile.mkdtemp(); c = Clock(); llm = make(tmp, c, Backend(c)); llm.generate(M[0], True); llm.generate(M[1], True)
    try: make(tmp, c, Backend(c), meta=dict(model="another", t=0)); raise AssertionError("must raise")
    except ValueError as e: assert "different model" in str(e)
    try: make(tmp, c, Backend(c), meta=dict(model="m", t=1)); raise AssertionError("must raise")
    except ValueError as e: assert "different model" in str(e)
    with open(llm.path) as f: text = f.read()
    cut = text.rstrip("\n").rfind("\n") + 1 + 25; open(llm.path, "w").write(text[:cut])           # the session died while writing the last line
    be = Backend(c); llm2 = make(tmp, c, be); assert len(llm2.cache) == 1
    llm2.generate(M[0], True); assert be.n == 0
    llm2.generate(M[1], True); assert be.n == 1                                                      # the half-written one is asked again
    assert all(json.loads(l) for l in read(llm.path).splitlines()) and len(read(llm.path).splitlines()) == 3
    # a damaged line in the middle of the file: nothing after it is trusted either (the meta line is the first thing that must be read whole), and the file is written clean again
    tmp2 = tempfile.mkdtemp(); c2 = Clock(); lx = make(tmp2, c2, Backend(c2))
    for m in M[:3]: lx.generate(m, True)
    ls = read(lx.path).splitlines(); open(lx.path, "w").write("\n".join([ls[0], ls[1], ls[2][:20], ls[3]]) + "\n")
    be = Backend(c2); l3 = make(tmp2, c2, be); assert len(l3.cache) == 1 and len(read(lx.path).splitlines()) == 2
    l3.generate(M[2], True); assert be.n == 1, "the record after the damaged line is asked again, not trusted"


def test_the_daily_budget_stops_before_the_call_and_the_next_pacific_day_starts_again():
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c); llm = make(tmp, c, be, daily=3)
    for i in range(3): llm.generate(M[i], True)
    try: llm.generate(M[3], True); raise AssertionError("must stop")
    except QuotaStop as e: assert "Daily budget" in str(e) and "3 calls today" in str(e) and "Run this same cell again" in str(e)
    assert be.n == 3, "the call that would pass the budget is not made"
    assert llm.generate(M[0], True)[0] and be.n == 3                                                 # a prompt already answered is still answered at the limit
    c.t = ts(2026, 10, 10, 6, 59)                                                                    # 23:59 Pacific, still the same day
    try: llm.generate(M[3], True); raise AssertionError("must stop")
    except QuotaStop: pass
    llm2 = make(tmp, c, Backend(c), daily=3)                                                         # and a restart does not reset the count
    try: llm2.generate(M[3], True); raise AssertionError("must stop")
    except QuotaStop: pass
    c.t = ts(2026, 10, 10, 7, 0); assert llm2.calls_today() == 0                                     # midnight Pacific
    llm2.generate(M[3], True); assert llm2.backend.n == 1 and llm2.calls_today() == 1


def test_failed_calls_count_against_the_day_and_the_wait_and_retry_rules():
    tmp = tempfile.mkdtemp(); c = Clock(); minute = Err("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 23.4s.", 429)
    be = Backend(c, script=[minute, None]); logs = []; llm = make(tmp, c, be, daily=2); llm.say = logs.append; t0 = c.t
    got = llm.generate(M[0], True); assert got[0] and got[2] == 1.0, "the seconds of a reply are those of the call that answered, not the waiting before it"
    assert be.n == 2 and c.t - t0 >= 24.4 and llm.calls_today() == 2 and llm.usage()["failed_calls"] == 1 and any("per minute" in l for l in logs)
    try: llm.generate(M[1], True); raise AssertionError("the failed call used one of the two")
    except QuotaStop: pass
    assert make(tmp, c, Backend(c), daily=2).calls_today() == 2                                      # the failure is in the file
    # told "per minute" again and again: stop after max_waits waits
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[minute] * 7); llm = make(tmp, c, be, max_waits=6)
    try: llm.generate(M[0], True); raise AssertionError("must stop")
    except QuotaStop as e: assert "per minute" in str(e) and "saved" in str(e)
    assert be.n == 7
    # told "per day": stop at once, no retry, no wait
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[Err("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier", 429)]); llm = make(tmp, c, be); t0 = c.t
    try: llm.generate(M[0], True); raise AssertionError("must stop")
    except QuotaStop as e: assert "DAILY limit" in str(e)
    assert be.n == 1 and c.t - t0 < 1.0
    # a busy server (503 'high demand'): Google counts such failed tries against the free daily limit (seen on 10 Oct), so they count here too; retried only twice, after 2 and after 5 minutes
    busy = Err("503 UNAVAILABLE. This model is currently experiencing high demand.", 503)
    tmp = tempfile.mkdtemp(); c = Clock(); lx = make(tmp, c, Backend(c))
    assert [lx.retry_wait(n) for n in range(1, 5)] == [120.0, 300.0, 300.0, 300.0] and lx.max_retries == 2 and lx.retry_wait_total() == 420.0
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[busy, None]); llm = make(tmp, c, be, daily=2); t0 = c.t; logs = []; llm.say = logs.append
    assert llm.generate(M[0], True)[0] and be.n == 2 and 120 <= c.t - t0 < 125 and sum("busy" in l and "120 s" in l for l in logs) == 1
    assert llm.calls_today() == 2 and llm.usage()["busy_answers"] == 1 and llm.usage()["failed_calls"] == 1, "the busy try is one of the day's calls"
    try: llm.generate(M[1], True); raise AssertionError("the budget of 2 is used by the busy try and the answer")
    except QuotaStop as e: assert "Daily budget" in str(e)
    again = make(tmp, c, Backend(c), daily=2); assert again.calls_today() == 2 and len(again.busy_events) == 1 and len(again.cache) == 1, "the note survives a restart and still counts"
    assert [json.loads(l).get("fail") for l in read(llm.path).splitlines()[1:]] == ["retry", None]
    # the waits used up: a plain stop (a QuotaStop that says 'busy') after 3 tries and about 7 minutes; the three tries count
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[busy] * 8); llm = make(tmp, c, be, daily=10); t0 = c.t
    try: llm.generate(M[0], True); raise AssertionError("must stop")
    except ServiceBusy as e: assert isinstance(e, QuotaStop) and "busy or not reachable" in str(e) and "3 times" in str(e) and "about 7 minutes" in str(e) and "at least 30 minutes" in str(e) and "ONCE" in str(e) and "saved" in str(e), str(e)
    assert be.n == 3 and 420 <= c.t - t0 < 425 and llm.calls_today() == 3 and not llm.cache
    be = Backend(c, script=[busy] * 8); llm = make(tmp, c, be, max_retries=4)
    try: llm.generate(M[0], True); raise AssertionError("must stop")
    except ServiceBusy: pass
    assert be.n == 5, "more retries than waits: the last wait is used again"
    be = Backend(c, script=[busy] * 8); llm = make(tmp, c, be, retry_waits=(10.0,))
    try: llm.generate(M[0], True); raise AssertionError("must stop")
    except ServiceBusy: pass
    assert be.n == 2
    # a busy answer is not a rate-limit answer: 'per minute' and 'per day' keep counting, and a different error is raised as it is
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[Err("connection reset by peer"), None]); llm = make(tmp, c, be); assert llm.generate(M[0], True)[0] and be.n == 2 and llm.usage()["busy_answers"] == 1 and llm.calls_today() == 2
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, script=[Err("API key not valid", 400)]); llm = make(tmp, c, be)
    try: llm.generate(M[0], True); raise AssertionError("must raise")
    except Err as e: assert "API key" in str(e)
    assert be.n == 1


def test_calls_are_spaced_by_min_interval_and_a_saved_reply_costs_no_wait():
    tmp = tempfile.mkdtemp(); c = Clock(); be = Backend(c, latency=1.0); llm = make(tmp, c, be, interval=4.5)
    for i in range(5): llm.generate(M[i], True)
    gaps = [b - a for a, b in zip(be.starts, be.starts[1:])]; assert len(gaps) == 4 and all(abs(g - 4.5) < 1e-9 for g in gaps), gaps
    t = c.t; llm.generate(M[0], True); assert c.t == t, "a saved reply is returned at once"
    be2 = Backend(Clock(c.t), latency=10.0); llm2 = make(tmp, Clock(c.t), be2, interval=4.5, name="d.jsonl"); llm2.clock = be2.clock.now; llm2.sleep = be2.clock.advance
    for i in range(3): llm2.generate(M[i], True)
    assert all(b - a >= 10.0 for a, b in zip(be2.starts, be2.starts[1:])), "a call that is slower than the interval is not delayed further"
    assert be2.starts[1] - be2.starts[0] == 10.0


def test_the_file_is_saved_after_every_reply():
    tmp = tempfile.mkdtemp(); c = Clock(); seen = []; llm = make(tmp, c, Backend(c, script=[None, Err("503 UNAVAILABLE", 503), None]), on_write=lambda: seen.append(len(read(os.path.join(tmp, "c.jsonl")).splitlines())))
    llm.generate(M[0], True); llm.generate(M[1], True); assert seen == [2, 3, 4], seen                # reply, failed call, reply: the hook runs after each line is on disk
    llm.generate(M[1], True); assert seen == [2, 3, 4]                                               # a saved reply writes nothing


# ---------------------------------------------------------------- 5: the Gemini backend
class FakeTypes:
    class GenerateContentConfig:
        def __init__(self, **kw): self.kw = kw
    class ThinkingConfig:
        def __init__(self, **kw): self.kw = kw


class FakeClient:
    def __init__(self, names=(), text="DEL[1,0]", um=None, cands=True, raises_text=False):
        self.calls, self.names, self.text, self.um, self.cands, self.raises_text = [], list(names), text, um, cands, raises_text
        self.models = types.SimpleNamespace(generate_content=self._gen, list=lambda: [types.SimpleNamespace(name=n) for n in self.names])

    def _gen(self, model, contents, config):
        self.calls.append((model, contents, config.kw)); outer = self
        class R:
            @property
            def text(self):
                if outer.raises_text: raise ValueError("no text: blocked")
                return outer.text
            usage_metadata = outer.um if outer.um is not None else types.SimpleNamespace(prompt_token_count=321, candidates_token_count=9, thoughts_token_count=40)
            candidates = [types.SimpleNamespace(finish_reason="STOP")] if outer.cands else []
        return R()


def test_the_gemini_backend_passes_the_settings_and_reads_the_counts():
    cl = FakeClient(); be = GeminiBackend(model="gemini-x", client=cl, types=FakeTypes, max_output_tokens=4096)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    r = be.call(msgs); model, contents, kw = cl.calls[0]
    assert model == "gemini-x" and contents == "USER" and kw == dict(system_instruction="SYS", max_output_tokens=4096), kw                  # default thinking and temperature: neither is sent
    assert r == dict(text="DEL[1,0]", tokens=9, think_tokens=40, prompt_tokens=321, finish="STOP") and be.config_dict() == dict(provider="gemini", model="gemini-x", thinking=None, temperature=None, max_output_tokens=4096)
    be = GeminiBackend(model="m", client=cl, types=FakeTypes); be.call(msgs)
    assert be.config_dict()["max_output_tokens"] == 8192 and cl.calls[-1][2]["max_output_tokens"] == 8192, "the page says: at most 8,192 output tokens, thinking included"
    be = GeminiBackend(model="m", client=cl, types=FakeTypes, thinking="low", temperature=0.5); be.call(msgs); kw = cl.calls[-1][2]
    assert kw["temperature"] == 0.5 and kw["thinking_config"].kw == dict(thinking_level="low")
    be = GeminiBackend(model="m", client=cl, types=FakeTypes, thinking=0); be.call(msgs); assert cl.calls[-1][2]["thinking_config"].kw == dict(thinking_budget=0)
    for bad in ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],):
        try: be.call(bad); raise AssertionError("must raise")
        except ValueError as e: assert "system and user" in str(e)
    be = GeminiBackend(model="m", client=FakeClient(raises_text=True, um=types.SimpleNamespace(prompt_token_count=5, candidates_token_count=None, thoughts_token_count=None), cands=False), types=FakeTypes)
    assert be.call(msgs) == dict(text="", tokens=0, think_tokens=0, prompt_tokens=5, finish=None)                                               # a blocked reply is an empty one, not a crash
    be = GeminiBackend(model="m", client=FakeClient(um=types.SimpleNamespace()), types=FakeTypes); assert be.call(msgs)["prompt_tokens"] is None


def test_resolve_model_picks_the_right_id_and_build_says_which():
    names = ["models/gemini-3.5-flash-lite-tts", "models/gemini-3.5-flash-lite-preview-09-2026", "models/gemini-3.5-flash", "models/gemini-3.8-flash", "models/gemini-3.8-flash-live", "models/gemini-3.8-flash-image",
             "models/gemini-3.1-flash-lite", "models/gemma-4-31b"]
    assert resolve_model(names, "gemini-3.8-flash") == "gemini-3.8-flash" and resolve_model(names, "models/gemini-3.8-flash") == "gemini-3.8-flash"
    assert resolve_model(names, "gemini-3.5-flash-lite") == "gemini-3.5-flash-lite-preview-09-2026", "not the tts one"
    assert resolve_model(["models/gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite-preview-09-2026"], "gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"
    assert resolve_model(["models/gemini-3.8-flash-preview-09-2026", "models/gemini-3.8-flash-preview"], "gemini-3.8-flash") == "gemini-3.8-flash-preview", "the shortest id that starts with the name"
    assert resolve_model(["models/gemini-3.8-flash-b1", "models/gemini-3.8-flash-a1"], "gemini-3.8-flash") == "gemini-3.8-flash-a1", "ties go to the first in alphabetical order"
    assert resolve_model(names, "gemini-3.5-flash-lite-tts") == "gemini-3.5-flash-lite-tts", "an exact id is taken as asked, even a special variant"
    try: resolve_model(names, "gemini-9-flash"); raise AssertionError("must raise")
    except ValueError as e: assert "No model called 'gemini-9-flash'" in str(e) and "gemini-3.8-flash" in str(e)
    tmp = tempfile.mkdtemp(); said = []; cl = FakeClient(names=names)
    llm, model = build_gemini_llm(KEY, "gemini-3.5-flash-lite", os.path.join(tmp, "c.jsonl"), 480, 4.5, client=cl, types=FakeTypes, say=said.append)
    assert model == "gemini-3.5-flash-lite-preview-09-2026" and said == [f"model id: {model}"] and llm.daily_limit == 480 and llm.min_interval == 4.5
    assert json.loads(read(llm.path).splitlines()[0]) == {"_meta": dict(provider="gemini", model=model, thinking=None, temperature=None, max_output_tokens=8192)}


def test_the_key_goes_only_to_the_sdk_and_is_written_nowhere():
    tmp = tempfile.mkdtemp(); c = Clock(); got = []
    class G:
        @staticmethod
        def Client(api_key=None): got.append(api_key); return FakeClient(names=["models/gemini-3.5-flash-lite"], text="ADD[4,4:3]")
    google = types.ModuleType("google"); genai = types.ModuleType("google.genai"); gt = types.ModuleType("google.genai.types")
    gt.GenerateContentConfig, gt.ThinkingConfig = FakeTypes.GenerateContentConfig, FakeTypes.ThinkingConfig; genai.Client, genai.types, google.genai = G.Client, gt, genai
    out = io.StringIO()
    with patched_modules({"google": google, "google.genai": genai, "google.genai.types": gt}), contextlib.redirect_stdout(out):
        llm, model = build_gemini_llm(KEY, "gemini-3.5-flash-lite", os.path.join(tmp, "c.jsonl"), 10, 0.0, sleep=lambda s: None)
        for m in M[:3]: llm.generate(m, True)
        print(llm.usage(), llm.backend.config_dict())
    assert got == [KEY]
    blob = out.getvalue() + "".join(read(os.path.join(d, f)) for d, _, fs in os.walk(tmp) for f in fs)
    assert KEY not in blob and "AIza" not in blob and "api_key" not in blob and "ADD[4,4:3]" in blob
    # and in the source: no print, say, write or dumps ever receives the key
    tree = ast.parse(read(api_llm.__file__)); bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func; name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name in ("print", "say", "write", "dumps", "dump", "format", "append", "_append"):
                for a in list(n.args) + [k.value for k in n.keywords]:
                    if any((isinstance(x, ast.Name) and x.id == "api_key") or (isinstance(x, ast.Attribute) and x.attr in ("api_key", "key_in_use")) for x in ast.walk(a)): bad.append(ast.unparse(n))
    assert not bad, bad


# ---------------------------------------------------------------- 7: stop at the daily limit, start again the next day
def _single(tmp, daily, clock, style="rule_like", tasks=None, hint=True, name="s"):
    be = Backend(clock, style); llm = make(tmp, clock, be, daily=daily)
    try: res = run_scope_regime(llm, tasks, hint, os.path.join(tmp, name + ".jsonl"), meta=dict(model="stand-in", seed=7)); return res, be, llm, None
    except QuotaStop as e: return None, be, llm, e


def test_a_single_step_run_stopped_by_the_daily_limit_and_restarted_next_day_equals_the_uninterrupted_run():
    tasks = make_scope_tasks(21, 7); u = tempfile.mkdtemp(); c0 = Clock()
    whole, be0, _, _ = _single(u, None, c0, tasks=tasks); total = be0.n; assert whole and total == 21 == len({json.dumps(base_messages(t, True)) for t in tasks})
    for limit in (1, 8, 20):
        tmp = tempfile.mkdtemp(); c = Clock()
        res, be1, llm1, stop = _single(tmp, limit, c, tasks=tasks); assert res is None and stop is not None and be1.n == limit, limit
        assert len(read(os.path.join(tmp, "s.jsonl")).splitlines()) == limit + 1, "every finished request is on disk when the run stops"
        c.t = ts(2026, 10, 10, 20)                                                                 # the next day (new session, new objects)
        res2, be2, _, stop2 = _single(tmp, limit, c, tasks=tasks, name="s")
        if limit < 11: assert res2 is None and be2.n == limit                                      # two or three days are needed; each day does `limit` more
        c.t = ts(2026, 10, 12, 20); res3, be3, _, stop3 = _single(tmp, 100, c, tasks=tasks); assert res3 is not None
        assert res3["rows"] == whole["rows"] and res3["summary"] == whole["summary"], limit
        assert be1.n + be2.n + be3.n == total, "no prompt was asked twice"


def test_a_long_run_stopped_in_the_middle_of_an_episode_and_restarted_next_day_equals_the_uninterrupted_run():
    args = dict(mode="state", episodes=3, steps=8, seed=7, hint_rules=True); pol = ("blind", "repair_m6", "scope_m7")
    def go(tmp, daily, clock, be=None):
        be = be or Backend(clock); llm = make(tmp, clock, be, daily=daily)
        try: return run_long(llm, args["mode"], args["episodes"], args["steps"], args["seed"], args["hint_rules"], os.path.join(tmp, "l.jsonl"), meta=dict(model="stand-in"), policies=pol, mix="scope"), be
        except QuotaStop: return None, be
    u = tempfile.mkdtemp(); c0 = Clock(); whole, be0 = go(u, None, c0); total = be0.n; assert whole and 20 < total < 3 * 8 * 3
    for limit in (3, total // 3, total - 2):
        tmp = tempfile.mkdtemp(); c = Clock(); res, be1 = go(tmp, limit, c); assert res is None and be1.n == limit
        done = len(read(os.path.join(tmp, "l.jsonl")).splitlines()) - 1; assert done < 3, "stopped before the end"
        c.t = ts(2026, 10, 10, 20); res2, be2 = go(tmp, 10 ** 6, c); assert res2 is not None
        assert res2["rows"] == whole["rows"] and res2["summary"] == whole["summary"] and res2["meta"] == whole["meta"], limit
        assert be1.n + be2.n == total, ("no prompt asked twice", limit, be1.n, be2.n, total)
        c.t = ts(2026, 10, 11, 20); res3, be3 = go(tmp, 0, c); assert res3["rows"] == whole["rows"] and be3.n == 0          # all finished: the model is not asked at all, even with a budget of 0


# ---------------------------------------------------------------- 8: the reading rules
def _rows(style, tasks, hint):
    c = Clock(); llm = make(tempfile.mkdtemp(), c, Backend(c, style)); return run_scope_regime(llm, tasks, hint, os.path.join(tempfile.mkdtemp(), "x.jsonl"), meta=dict(model="stand-in"))


def test_the_format_line_and_the_smoke_check_read_at_their_boundaries():
    good = [dict(text="DEL[1,0]")] * 19 + [dict(text="here is some prose")]                        # 1 of 20 = 5 percent: still yes
    assert unparsed_single(good) == (1, 20)
    ok = verdict_api_format({True: good})[0]; assert ok[1] is True and "(1 of 20)" in ok[0] and ok[0].startswith("[format] at most 5 percent")
    assert verdict_api_format({True: good + [dict(text="")]})[0][1] is False                        # 2 of 21: no
    assert verdict_api_format({True: good, False: good})[0][1] is True and verdict_api_format({True: good, False: good + [dict(text="")]})[0][1] is False
    assert verdict_api_format({})[0][1] is False and verdict_api_format()[0][1] is False             # nothing counted is not a pass
    long_rows = [dict(policies=dict(a=[dict(parsed=True, cached=False), dict(parsed=False, cached=True)], b=[dict(parsed=False, cached=False), dict(parsed=True, cached=False)]))]
    assert unparsed_long(long_rows) == (1, 3), "a reply shared by two policies is counted once"
    assert verdict_api_format(None, {True: long_rows})[0][1] is False and verdict_api_format({True: good}, {True: long_rows})[0][0].endswith("(2 of 23)")
    assert smoke_report([dict(text="DEL[1,0]")] * 4 + [dict(text="x")] * 2)[0] is True and smoke_report([dict(text="DEL[1,0]")] * 3 + [dict(text="x")] * 3)[0] is False and smoke_report([])[0] is False
    assert smoke_report([dict(text="x")])[1].startswith("STOP: only 0 of 1")


def test_the_pivotal_lines_read_as_the_page_says():
    tasks = [t for t in make_scope_tasks(140, 7) if t.kind == "delete_dep"]; assert len(tasks) == 20
    ideal = {h: summarize_scope(_rows("ideal", tasks, h)["rows"], tasks) for h in (True, False)}
    no_rule = {h: summarize_scope(_rows("no_rule_like", tasks, h)["rows"], tasks) for h in (True, False)}
    v = dict(verdict_pivotal({False: ideal[False], True: ideal[True]}))
    assert len(v) == 4 and v["[guard] gate, repair_m6 and scope_m7 corrupt no graph in any regime"] is True
    need = [k for k in v if k.startswith("[need]")][0]; assert v[need] is False, "a model that is always right does not need the framework"
    assert all(ok for k, ok in v.items() if k.startswith("[guard]"))
    w = dict(verdict_pivotal({False: no_rule[False], True: no_rule[True]})); assert w[[k for k in w if k.startswith("[need]")][0]] is True, "blind apply forgets the dependents: right on none"
    assert all(ok for k, ok in w.items() if k.startswith("[guard]")), w
    assert len(verdict_pivotal({False: ideal[False]})) == 3 and len(verdict_pivotal({True: ideal[True]})) == 2
    # boundaries: blind right on exactly half passes the need line, one more fails it; repair_m6 at 18 of 20 passes, at 17 fails
    s = json.loads(json.dumps(ideal[False])); s["all"]["blind"]["success"] = 10; assert dict(verdict_pivotal({False: s}))[[k for k in w if k.startswith("[need]")][0]] is True
    s["all"]["blind"]["success"] = 11; assert dict(verdict_pivotal({False: s}))[[k for k in w if k.startswith("[need]")][0]] is False
    g = [k for k in w if k.startswith("[guard] no rule: repair_m6")][0]; s["all"]["repair_m6"]["success"] = 18; assert dict(verdict_pivotal({False: s}))[g] is True
    s["all"]["repair_m6"]["success"] = 17; assert dict(verdict_pivotal({False: s}))[g] is False
    r = json.loads(json.dumps(ideal[True])); g2 = [k for k in w if k.startswith("[guard] rule stated")][0]; r["all"]["scope_m7"]["success"] = 18; assert dict(verdict_pivotal({True: r}))[g2] is True
    r["all"]["scope_m7"]["success"] = 17; assert dict(verdict_pivotal({True: r}))[g2] is False
    r = json.loads(json.dumps(ideal[True])); r["all"]["repair_m6"]["corrupted"] = 1; assert dict(verdict_pivotal({True: r}))["[guard] gate, repair_m6 and scope_m7 corrupt no graph in any regime"] is False


# ---------------------------------------------------------------- 9: the notebook and the page
def _cells(path): nb = json.load(open(path)); return nb, nb["cells"]
def _src(c): return "".join(c["source"])
def _no_bang(s): return "".join(l for l in s.splitlines(keepends=True) if not l.startswith("!"))


def _calls(tree, name): return sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name), key=lambda n: (n.lineno, n.col_offset))


def test_the_notebook_runs_the_same_test_as_the_14b_notebook():
    nb, cells = _cells(NB); nb14, cells14 = _cells(NB_14B)
    assert nb["nbformat"] == 4 and "accelerator" not in nb["metadata"], "no GPU is asked for"
    for c in cells: assert all(l.endswith("\n") for l in c["source"][:-1]), "every line but the last must end with a newline"
    code = [c for c in cells if c["cell_type"] == "code"]
    for c in code: compile(_no_bang(_src(c)), "cell", "exec")
    t = ast.parse("\n".join(_no_bang(_src(c)) for c in code)); t14 = ast.parse("\n".join(_no_bang(_src(c)) for c in cells14 if c["cell_type"] == "code"))
    # Part 1: the same 140 tasks, seed 7, both regimes
    s = _calls(t, "run_scope_regime"); assert len(s) == 5                                           # smoke, rule, no rule, B no rule, B rule
    assert [tuple(ast.literal_eval(a) for a in c.args) for c in _calls(t, "make_scope_tasks")] == [(6, 7), (140, 7), (140, 7)]            # smoke (the first 6 of the 140), Part 1, Part B's 20
    src = "\n".join(map(_src, code)); assert src.count("make_scope_tasks(140, 7)") == 2 and 'meta=dict(model=MODEL, seed=7)' in src
    assert ast.literal_eval(s[1].args[2]) is True and ast.literal_eval(s[2].args[2]) is False and ast.literal_eval(s[3].args[2]) is False and ast.literal_eval(s[4].args[2]) is True
    # Part 2: identical to the two Part-2 calls of the 14B notebook except for the model object and the file / progress
    l14, l = _calls(t14, "run_long")[1:], _calls(t, "run_long"); assert len(l14) == 2 and len(l) == 2
    for a, b in zip(l14, l):
        sig = lambda c: ([ast.dump(x) for i, x in enumerate(c.args) if i not in (0, 6)], {k.arg: ast.dump(k.value) for k in c.keywords if k.arg not in ("progress", "meta")})
        assert sig(a) == sig(b), "same mode, episodes, steps, seed, rule flag, policies, mix"
    for call, (eps, rule) in zip(l, ((6, True), (4, False))):
        a = call.args; kw = {k.arg: k.value for k in call.keywords}
        assert ast.literal_eval(a[1]) == "state" and ast.literal_eval(a[2]) == eps and ast.literal_eval(a[3]) == 30 and ast.literal_eval(a[4]) == 7 and ast.literal_eval(a[5]) == rule
        assert ast.literal_eval(kw["policies"]) == ("blind", "repair_m6", "scope_m7") and ast.literal_eval(kw["mix"]) == "scope"
    # the models, the budgets, the spacing, the key
    assert 'WANTED = "gemini-3.5-flash-lite"' in src and '"gemini-3.8-flash"' in src and "daily_limit=480, min_interval=4.5" in src and "daily_limit=20, min_interval=13" in src
    assert 'userdata.get("GEMINI_API_KEY")' in src and "AIza" not in src and "print(KEY" not in src and len(re.findall(r"\bKEY\b", src)) == 3 and src.count("build_gemini_llm(KEY, ") == 2 and "pivotal = [t for t in make_scope_tasks(140, 7) if t.kind == \"delete_dep\"]" in src
    for f in ("axi_gemini_single_rule", "axi_gemini_single_norule", "axi_gemini_long_rule", "axi_gemini_long_norule", "axi_gemini_B_norule", "axi_gemini_B_rule"): assert f in src, f
    assert "verdict_big_single" in src and "verdict_big_long" in src and "verdict_api_format" in src and "verdict_pivotal" in src and "smoke_report" in src
    assert "axi-colab-gemini-v3.zip" in "".join(cells[2]["source"]) and "pip -q install -U google-genai" in "\n".join(map(_src, cells))
    intro = _src(cells[0]); assert "docs/API_TEST_GEMINI.md" in intro and "Gemini 3.5 Flash Lite" in intro and "not a Pro model" in intro and "12:30 pm in India" in intro and "1:30 pm after" in intro and "two to three days" in intro and "Never paste it in the chat" in intro


def test_the_page_and_the_notebook_agree():
    page = read(PAGE); nb, cells = _cells(NB); src = "\n".join(map(_src, cells))
    for needle in ("gemini-3.5-flash-lite", "gemini-3.8-flash", "480 calls a day", "the budget is 20", "make_scope_tasks(140, 7)", "[format]", "[guard]", "[need]", "verdict_api_format", "verdict_pivotal", "neither of them a Pro or frontier model",
                   "Colab Secrets", "one sample per distinct prompt", "12:30 pm in India", "1:30 pm after", "A line that comes out \"NO\" is a finding, not an error", "at most 5 percent", "seeds 7000 to 7005", "seeds 7000 to 7003",
                   "\"Flash Lite\", never \"frontier\""):
        assert needle in page, needle
    assert "Run this cell once" in "".join(cells[18]["source"]) and "at least 30 minutes" in "".join(cells[18]["source"]), "Part B's cell says: run once, and wait before a retry"
    assert "Gemini 3.5 Flash Lite" in page and "Gemini 3.8 Flash" in page and page.index("## Predictions") < page.index("## Reading rules") < page.index("## What this cannot show")
    assert len(re.findall(r"\((\d+) %", page)) >= 20 and "Overall:" in page                       # every prediction carries a confidence, and the overall bets are there
    assert page.count("# Result") <= 1                                                                    # the result, once it exists, is appended below the line that ends the pre-registration (locked in the next test)
    for f in ("api_llm.py", "api_test.py", "axi_gemini_test.ipynb"): assert f in page


PAGE_SHA256 = "b0f733c46719540ab7379f649cc99cac3f5830aec3016e706cbac239101a3fa3"          # sha256 of docs/API_TEST_GEMINI.md as written before any API number existed (text + the final newline)


ADDENDUM_MARK = "\n\n---\n\n## Addendum (9 Oct 2026"
ADDENDUM2_MARK = "\n\n---\n\n## Addendum 2 (10 Oct 2026"
RESULT_MARK = "\n\n---\n\n# Result"
ADDENDUM_SHA256 = "8ca8d3f9b00ef0ddbd620714d75bce92f96a0f1c41bcc691fde7e64f4beba74a"          # addendum 1 (from its separator to its last line, with the final newline), written 9 Oct evening before any result file was read
ADDENDUM2_SHA256 = "9a6706745b1f7b669c934a76bb22f6e597c16b9b57cc4b3fa117f304a7ee2738"         # addendum 2 (same way), written 10 Oct before any Part B result was read


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def test_the_pre_registration_and_the_addenda_are_unchanged_whatever_is_appended_below_them():
    text = read(PAGE); assert text.count(ADDENDUM_MARK) == 1 and text.count(ADDENDUM2_MARK) == 1 and text.count(RESULT_MARK) <= 1
    if RESULT_MARK in text: text = text[:text.index(RESULT_MARK)]                              # the Result, once it exists, is appended below the addenda
    head, rest = text.split(ADDENDUM_MARK); add1, add2 = (ADDENDUM_MARK + rest).split(ADDENDUM2_MARK); add2 = ADDENDUM2_MARK + add2
    assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    assert _h(add1) == ADDENDUM_SHA256, "addendum 1 must not change"
    assert _h(add2) == ADDENDUM2_SHA256, "addendum 2 must not change"
    for needle in ("503", "high demand", "15, 30, 60, then 120 seconds", "not counted against the day's budget", "What did not change", "every reading rule", "ServiceBusy", "before I had read any result file"): assert needle in add1, needle
    for needle in ("daily limit was used up", "was **wrong**", "counts against the day's budget again", "after 2 and after 5 minutes", "run the cell once", "What did not change", "every reading rule", "before I had read any Part B result"): assert needle in add2, needle


class ServerError(Exception):
    """Like google.genai.errors.ServerError: the status code and the service's own words."""
    def __init__(self, code, msg): super().__init__(f"{code} UNAVAILABLE. {msg}"); self.code = code


def _fake_world(clock, latency=1.0, key=KEY, models=("models/gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite-tts", "models/gemini-3.8-flash", "models/gemini-3.8-flash-live")):
    """Fake Colab (files, Drive, Secrets) and a fake Gemini service whose answers are those of the 'rule_like' stand-in. Returns (modules, state)."""
    mods, _ = _fake_drive(True); st = types.SimpleNamespace(calls=0, models=[], keys=[], downloaded=[], prompts=set(), inner=StandIn("rule_like"), fail_model=None, fail_next=0, busy_sent=0)
    class Files:
        def upload(self): return {"axi-colab-gemini-v3.zip": b""}
        def download(self, p): st.downloaded.append(os.path.basename(p))
    class Secrets:
        @staticmethod
        def get(name):
            if key is None or name != "GEMINI_API_KEY": raise KeyError("Secret does not exist")
            return key
    class Client:
        def __init__(self, api_key=None):
            st.keys.append(api_key); self.models = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(name=n) for n in models], generate_content=self._gen)
        def _gen(self, model, contents, config):
            if model == st.fail_model and st.fail_next > 0:
                st.fail_next -= 1; st.busy_sent += 1; clock.advance(0.3); raise ServerError(503, "{'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}")
            st.calls += 1; st.models.append(model); st.prompts.add((model, config.kw["system_instruction"], contents)); clock.advance(latency)
            text, ntok, sec = st.inner.generate([{"role": "user", "content": contents}], True)
            return types.SimpleNamespace(text=text, usage_metadata=types.SimpleNamespace(prompt_token_count=st.inner.last_prompt_tokens, candidates_token_count=ntok, thoughts_token_count=11),
                                         candidates=[types.SimpleNamespace(finish_reason="STOP")])
    mods["google.colab"].files = Files(); mods["google.colab"].userdata = Secrets; mods["google.colab.userdata"] = Secrets
    google = mods["google"]; genai = types.ModuleType("google.genai"); gt = types.ModuleType("google.genai.types")
    gt.GenerateContentConfig, gt.ThinkingConfig = FakeTypes.GenerateContentConfig, FakeTypes.ThinkingConfig; genai.Client, genai.types, google.genai = Client, gt, genai
    mods["google.genai"], mods["google.genai.types"] = genai, gt
    return mods, st


def _run_notebook(cells_to_run, work, drive_root, clock, daily_a=None, key=KEY, latency=1.0, ns=None, world=None):
    """Execute the chosen code cells (by index in the notebook) in order. A cell that stops with 'The cell above stopped for today' ends the session. Returns (ns, printed, state, ended_early)."""
    nb = json.load(open(NB)); out = io.StringIO(); mods, st = world or _fake_world(clock, latency, key); ns = {} if ns is None else ns; early = False
    with patched_modules(mods), contextlib.redirect_stdout(out), mock.patch.object(time, "time", clock.now), mock.patch.object(time, "sleep", clock.advance):
        for i in cells_to_run:
            c = nb["cells"][i]; assert c["cell_type"] == "code", i
            src = _no_bang(_src(c)).replace('open_store("/content/out", "axi_gemini")', f'open_store("{work}", "axi_gemini", "{drive_root}")').replace('"/content/smoke_single.jsonl"', f'"{work}/smoke_single.jsonl"')
            src = src.replace('sys.path.insert(0, "/content/axi/python")', "")
            if daily_a is not None: src = src.replace("daily_limit=480", f"daily_limit={daily_a}")
            assert "/content" not in src, src
            try: exec(compile(src, f"cell{i}", "exec"), ns)
            except AssertionError as e:
                if "stopped for today" in str(e) or "Run Cell 10 first" in str(e): early = True; break
                raise
    return ns, out.getvalue(), st, early


def _code_index(contains):
    nb = json.load(open(NB)); return [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and contains in _src(c)][0]


def test_the_notebook_runs_end_to_end_stops_at_the_limit_and_finishes_the_next_day_like_an_unstopped_run():
    tmp = tempfile.mkdtemp(); drive = os.path.join(tmp, "drive"); saved = os.path.join(drive, "MyDrive", "axi_gemini")
    A = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'name = "axi_gemini_single_rule"', 'name = "axi_gemini_single_norule"', 'name = "axi_gemini_long_rule"', 'name = "axi_gemini_long_norule"')]
    cells = [A[0], A[1], A[2], A[3], A[4], A[5], A[6]]
    # a reference session with a budget nobody reaches
    c = Clock(); work = os.path.join(tmp, "ref"); os.makedirs(work)
    ref, out_ref, st_ref, early = _run_notebook(cells, work, os.path.join(tmp, "drive_ref"), c, daily_a=10 ** 6); assert not early
    total = st_ref.calls; assert ref["res_l_norule"] is not None and total > 300 and "SMOKE OK" in out_ref
    # day 1: the budget is reached in the middle of Part 2 (once in the rule-stated cell, once in the last cell); day 2: a brand new machine, the next Pacific day, the files come back from Drive
    for limit in (400, total - 40):
        drive = os.path.join(tmp, f"drive_{limit}"); saved = os.path.join(drive, "MyDrive", "axi_gemini")
        c = Clock(); work1 = os.path.join(tmp, f"d1_{limit}"); os.makedirs(work1)
        ns1, out1, st1, early1 = _run_notebook(cells, work1, drive, c, daily_a=limit)
        assert "STOPPED FOR NOW" in out1 and "Daily budget reached" in out1 and "12:30 pm in India until 1 November, 1:30 pm after" in out1 and st1.calls == limit and ns1["res_l_norule"] is None
        assert ns1["res_s_rule"] is not None and ns1["res_s_norule"] is not None and (ns1["res_l_rule"] is None) == early1
        assert KEY not in out1 and not any(KEY in read(os.path.join(d, f)) for d, _, fs in os.walk(tmp) for f in fs if f.endswith((".json", ".jsonl")))
        c.t = ts(2026, 10, 10, 20); work2 = os.path.join(tmp, f"d2_{limit}"); os.makedirs(work2)
        ns2, out2, st2, early2 = _run_notebook(cells, work2, drive, c, daily_a=limit); assert not early2 and "reply file: restored" in out2 and "Drive connected" in out2
        assert st1.calls + st2.calls == total, "no prompt was asked twice across the two days"
        for k in ("res_s_rule", "res_s_norule", "res_l_rule", "res_l_norule"): assert ns2[k]["rows"] == ref[k]["rows"] and ns2[k]["summary"] == ref[k]["summary"], (limit, k)
        assert "PART 2, both regimes" in out2 and "[format] at most 5 percent" in out2 and "[guard]" in out2 and "[need]" in out2 and "format, all four files" in out2
        for f in ("axi_gemini_single_rule.json", "axi_gemini_single_norule.json", "axi_gemini_long_rule.json", "axi_gemini_long_norule.json", "axi_gemini_cache.jsonl"): assert f in st2.downloaded, f
        assert {m for m in st1.models + st2.models} == {"gemini-3.5-flash-lite"}, "the right model id, not the speech variant"
        assert st1.keys == [KEY] and st2.keys == [KEY]
        for f in ("axi_gemini_cache.jsonl", "axi_gemini_single_rule.jsonl", "axi_gemini_long_norule.jsonl", "axi_gemini_long_norule.json"): assert os.path.exists(os.path.join(saved, f)), f
    # a third session after everything is done: nothing is asked, same result
        c.t = ts(2026, 10, 12, 20); work3 = os.path.join(tmp, f"d3_{limit}"); os.makedirs(work3)
        ns3, out3, st3, early3 = _run_notebook(cells, work3, drive, c, daily_a=0); assert not early3 and st3.calls == 0 and ns3["res_l_rule"]["rows"] == ref["res_l_rule"]["rows"]


def test_part_b_uses_its_own_file_and_its_own_limit_of_20_and_finishes_on_the_second_day():
    tmp = tempfile.mkdtemp(); drive = os.path.join(tmp, "drive"); idx = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'CACHE_B = ', 'name = "axi_gemini_B_rule"')]
    c = Clock(); work = os.path.join(tmp, "d1"); os.makedirs(work)
    ns, out, st, early = _run_notebook(idx, work, drive, c)                                         # smoke (6 calls on the Flash Lite), then day 1 of Part B, then Cell 11 the same day
    assert ns["MODEL_B"] == "gemini-3.8-flash" and len(ns["pivotal"]) == 20 and all(t.kind == "delete_dep" for t in ns["pivotal"])
    assert ns["res_b_norule"] is not None and ns["res_b_rule"] is None and "STOPPED FOR NOW" in out and "PART B, no rule" in out and "[need] no rule: blind apply is right on at most half of the 20" in out
    assert st.models.count("gemini-3.8-flash") == 20 and st.models.count("gemini-3.5-flash-lite") == 6, "20 calls to Part B's model, and the smoke test went to the other"
    assert ns["llmB"].calls_today() == 20
    c.t = ts(2026, 10, 10, 20); work2 = os.path.join(tmp, "d2"); os.makedirs(work2)
    ns2, out2, st2, early2 = _run_notebook(idx, work2, drive, c)
    assert out.count("reply file: nothing") == 2 and out2.count("reply file: restored") == 2, "day 2 brings back both reply files: the smoke test's and Part B's"
    assert len(ns2["llmB"].cache) == 40 and len(read(os.path.join(drive, "MyDrive", "axi_gemini", "axi_gemini_cache_B.jsonl")).splitlines()) == 41, "the 20 no-rule replies of day 1 are not lost from the saved file"
    assert ns2["res_b_rule"] is not None and st2.models.count("gemini-3.8-flash") == 20 and "PART B, both regimes" in out2 and "[guard] rule stated: scope_m7 is right on at least 90 percent of the 20" in out2
    assert "axi_gemini_B_norule.json" in st2.downloaded and "axi_gemini_B_rule.json" in st2.downloaded and "axi_gemini_cache_B.jsonl" in st2.downloaded
    assert [r["kind"] for r in ns2["res_b_norule"]["rows"]] == ["delete_dep"] * 20 == [r["kind"] for r in ns2["res_b_rule"]["rows"]]


def test_a_busy_server_stops_the_cell_with_a_plain_message_and_its_tries_count_against_the_day():
    tmp = tempfile.mkdtemp(); drive = os.path.join(tmp, "drive"); idx = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'CACHE_B = ')]
    # a reference session: the service is never busy
    c = Clock(); os.makedirs(os.path.join(tmp, "ref")); ref, _, st_ref, _ = _run_notebook(idx, os.path.join(tmp, "ref"), os.path.join(tmp, "drive_ref"), c)
    assert st_ref.models.count("gemini-3.8-flash") == 20
    # day 1: Google is busy every time. Three tries (now, +2 min, +5 min), then a plain stop. The three tries are three of the day's 20 calls.
    c = Clock(); mods, st = _fake_world(c); st.fail_model, st.fail_next = "gemini-3.8-flash", 10 ** 6
    work = os.path.join(tmp, "w1"); os.makedirs(work); t0 = c.t
    ns, out, st, early = _run_notebook(idx, work, drive, c, world=(mods, st))                                 # no traceback: the cell ends by itself
    assert "STOPPED FOR NOW" in out and "busy or not reachable" in out and "at least 30 minutes" in out and "Do NOT run it again and again" in out and "resets at midnight" not in out and ns["res_b_norule"] is None
    assert st.busy_sent == 3 and c.t - t0 > 420 and ns["llmB"].calls_today() == 3 and ns["llmB"].usage()["busy_answers"] == 3, "three busy tries, and all three count"
    # the service recovers the same day: only 17 of the 20 can still be asked, then the day's budget stops the cell
    st.fail_next = 0; work2 = os.path.join(tmp, "w2"); os.makedirs(work2); c.advance(1800)
    ns2, out2, st2, early2 = _run_notebook(idx, work2, drive, c, world=(mods, st))
    assert ns2["res_b_norule"] is None and "Daily budget reached: 20 calls today" in out2 and len(ns2["llmB"].cache) == 17 and ns2["llmB"].calls_today() == 20
    # the next Pacific day: the last 3 requests, and the result equals the one of the session that was never stopped
    c.t = ts(2026, 10, 11, 20); work3 = os.path.join(tmp, "w3"); os.makedirs(work3)
    ns3, out3, st3, early3 = _run_notebook(idx, work3, drive, c, world=(mods, st))
    assert ns3["res_b_norule"] is not None and "STOPPED FOR NOW" not in out3 and st3.models.count("gemini-3.8-flash") == 20
    assert ns3["res_b_norule"]["rows"] == ref["res_b_norule"]["rows"] and ns3["res_b_norule"]["summary"] == ref["res_b_norule"]["summary"]
    # a busy server in the middle of Part A is the same: stop, nothing lost, carry on
    tmp = tempfile.mkdtemp(); c = Clock(); mods, st = _fake_world(c); st.fail_model, st.fail_next = "gemini-3.5-flash-lite", 10 ** 6; work = os.path.join(tmp, "w"); os.makedirs(work)
    ns4, out4, st4, early4 = _run_notebook(idx[:3], work, os.path.join(tmp, "drive"), c, world=(mods, st))
    assert "STOPPED FOR NOW" in out4 and "busy or not reachable" in out4 and "Google's server being busy" in out4 and "resets at midnight" not in out4 and "SMOKE OK" not in out4 and ns4["smoke"] is None and ns4["res_s_rule"] is None


def test_a_missing_key_says_what_to_do_and_a_stop_message_is_plain():
    tmp = tempfile.mkdtemp(); idx = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get")]; c = Clock(); work = os.path.join(tmp, "w"); os.makedirs(work)
    try: _run_notebook(idx, work, os.path.join(tmp, "drive"), c, key=None); raise AssertionError("must stop")
    except SystemExit as e: assert "GEMINI_API_KEY" in str(e) and "Notebook access" in str(e)
    # a model name the key cannot see stops with the near matches, before any call
    c = Clock(); mods, st = _fake_world(c, models=("models/gemini-3.1-flash-lite",)); work = os.path.join(tmp, "w2"); os.makedirs(work)
    try: _run_notebook(idx, work, os.path.join(tmp, "drive2"), c, world=(mods, st)); raise AssertionError("must raise")
    except ValueError as e: assert "No model called 'gemini-3.5-flash-lite'" in str(e) and "gemini-3.1-flash-lite" in str(e)
    assert st.calls == 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

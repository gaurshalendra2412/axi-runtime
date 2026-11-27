"""Offline checks of the Test 47 glue (docs/TAU2_PILOT.md) with a scripted stand-in for the model.

Needs the tau2-bench source: either `pip install` of tau2 (the Colab notebook does that) or the folder
`<clone>/src` named in TAU2_SRC; where neither exists the test prints SKIP and exits 0. When only the
source folder is given, the packages tau2 imports but this test never uses (litellm, loguru, audio
libraries and so on) are replaced by empty stand-ins. No model call, no network.
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STUB = ("loguru", "litellm", "tenacity", "rich", "docstring_parser", "deepdiff", "addict", "yaml", "tabulate",
        "psutil", "dotenv", "httpx", "requests", "fastapi", "uvicorn", "typer", "openpyxl", "elevenlabs",
        "deepgram", "websockets", "scipy", "pydub", "aiohttp", "boto3", "google", "jiwer", "tqdm", "audioop",
        "pyaudio", "soundfile", "aiortc", "av")


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        m = MagicMock()
        m.__path__ = []
        m.__name__ = spec.name
        m.__spec__ = spec
        return m

    def exec_module(self, m):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in STUB:
            return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)


def _load():
    src = os.environ.get("TAU2_SRC")
    if src:
        sys.path.insert(0, src)
        sys.meta_path.insert(0, _Finder())
    try:
        from axi.experiments import tau2_retail_agents as A
        return A
    except ImportError:
        return None


A = _load()
if A is None:
    print("SKIP: tau2 is not importable (set TAU2_SRC to <tau2-bench clone>/src)")
    sys.exit(0)

from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage  # noqa: E402

ADDR = {"address1": "1 A St", "address2": "", "city": "X", "country": "USA", "state": "CO", "zip": "80279"}
ORDER = {"order_id": "#W1", "user_id": "noah_brown_6181", "address": ADDR, "items": [], "status": "pending",
         "fulfillments": [], "payment_history": []}


class Script:
    """Stands in for the model: returns the next scripted reply and keeps what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.ptoks = []

    def __call__(self, model, messages, tools=None, call_name=None, **kw):
        self.sent.append(list(messages))
        r = self.replies.pop(0)
        text, calls = r if isinstance(r, tuple) else (r, [])
        tcs = [ToolCall(id=f"c{len(self.sent)}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)] or None
        n_chars = sum(len(m.content or "") for m in messages if hasattr(m, "content"))
        self.ptoks.append(n_chars // 4)
        return AssistantMessage(role="assistant", content=text, tool_calls=tcs,
                                usage={"prompt_tokens": n_chars // 4, "completion_tokens": 5}, cost=0.001,
                                raw_data={"usage": {"prompt_tokens_details": {"cached_tokens": 3}}})


def agent(hold, replies):
    s = Script(replies)
    A.generate = s
    f = A.create_held_state_agent if hold else A.create_gate_chat_agent
    ag = f([], "POLICY TEXT", llm="fake", llm_args={})
    st = ag.get_init_state([AssistantMessage(role="assistant", content="Hi! How can I help you today?")])
    return ag, st, s


def tool_result(call_msg, content, error=False):
    c = call_msg.tool_calls[0]
    return ToolMessage(id=c.id, role="tool", content=content, error=error)


def drive_cancel(hold):
    """A whole cancel conversation: authenticate, read, list the details, get a yes, cancel."""
    ag, st, s = agent(hold, [
        ("", [("find_user_id_by_email", {"email": "n@example.com"})]),
        ("", [("get_order_details", {"order_id": "#W1"})]),
        ("Order #W1 is pending. Cancel it, reason 'no longer needed'? (yes/no)", []),
        ("", [("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})]),
        ("Done, the order is cancelled.", []),
    ])
    out = []
    m, st = ag.generate_next_message(UserMessage(role="user", content="Cancel #W1, my email is n@example.com"), st)
    out.append(m)
    m2, st = ag.generate_next_message(tool_result(m, "noah_brown_6181"), st)
    out.append(m2)
    m3, st = ag.generate_next_message(tool_result(m2, json.dumps(ORDER)), st)
    out.append(m3)
    m4, st = ag.generate_next_message(UserMessage(role="user", content="Yes, no longer needed."), st)
    out.append(m4)
    done = dict(ORDER, status="cancelled")
    m5, st = ag.generate_next_message(tool_result(m4, json.dumps(done)), st)
    out.append(m5)
    return ag, st, s, out


def test_arm_a_flow_prompts_stay_short_and_state_is_held():
    ag, st, s, out = drive_cancel(hold=True)
    assert [bool(m.tool_calls) for m in out] == [True, True, False, True, False]
    held = st.held
    assert held.user_id == "noah_brown_6181" and held.orders["#W1"]["status"] == "cancelled"
    assert held.changes and held.changes[0].startswith("cancel_pending_order")
    assert st.messages == []  # the chat is not kept
    for sent in s.sent:
        assert len(sent) == 2 and sent[0].role == "system" and sent[1].role == "user"
        assert "host_state_note" in sent[0].content
    last = s.sent[-1][1].content
    assert last.count('"payment_history"') == 1 and "cancelled" in last
    assert "result of cancel_pending_order" in last and "find_user_id_by_email" in last
    assert "get_order_details" not in last  # the old copy of the order was replaced by the updated one
    assert "customer: Cancel #W1" in last and "Order #W1 is pending." in last
    assert out[0].raw_data["axi"]["llm_calls"] == 1 and out[0].raw_data["axi"]["cached_tokens"] == 3


def test_arm_g_keeps_the_chat():
    ag, st, s, out = drive_cancel(hold=False)
    roles = [m.role for m in st.messages]
    assert len(st.messages) == 11  # the greeting, 2 customer messages, 3 tool results, 5 replies
    assert roles[0] == "assistant" and roles.count("tool") == 3 and roles.count("user") == 2 and roles.count("assistant") == 6
    assert [len(x) for x in s.sent] == sorted(len(x) for x in s.sent) and len(s.sent[-1]) > len(s.sent[0])  # the prompt only grows
    assert "host_state_note" not in s.sent[0][0].content


def test_refused_call_is_sent_back_then_the_model_asks_the_customer():
    ag, st, s = agent(True, [
        ("", [("find_user_id_by_email", {"email": "n@example.com"})]),
        ("", [("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})]),  # no read, no yes
        "Let me look at the order first.",
    ])
    m, st = ag.generate_next_message(UserMessage(role="user", content="cancel #W1"), st)
    m2, st = ag.generate_next_message(tool_result(m, "noah_brown_6181"), st)
    assert not m2.tool_calls and m2.content == "Let me look at the order first."
    a = m2.raw_data["axi"]
    assert a["llm_calls"] == 2 and a["reasks"] == 1 and a["refusals"] == [["READ_FIRST", "cancel_pending_order"]]
    assert "the host refused your call cancel_pending_order" in s.sent[-1][1].content
    assert m2.usage["prompt_tokens"] == sum(s.ptoks[-2:])  # summed over the two model calls
    assert m2.usage["completion_tokens"] == 10  # summed over the two model calls


def test_after_two_refusals_the_host_speaks_to_the_customer():
    bad = ("", [("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})])
    ag, st, s = agent(False, [("", [("find_user_id_by_email", {"email": "n@example.com"})]), bad, bad, bad])
    m, st = ag.generate_next_message(UserMessage(role="user", content="cancel #W1"), st)
    m2, st = ag.generate_next_message(tool_result(m, "noah_brown_6181"), st)
    assert not m2.tool_calls and "I'm sorry, I can't do that:" in m2.content
    assert m2.raw_data["axi"]["fallbacks"] == 1 and len(m2.raw_data["axi"]["refusals"]) == 3
    # arm G: the chat stays a valid sequence, every refused call has its answer
    seq = [(m.role, bool(getattr(m, "tool_calls", None))) for m in st.messages[-7:]]
    assert seq[-1] == ("assistant", False) and seq[-2] == ("tool", False) and seq[-3] == ("assistant", True)


def test_one_call_per_message_and_no_text_with_a_call():
    ag, st, s = agent(True, [("thinking aloud", [("find_user_id_by_email", {"email": "a@b.c"}),
                                                  ("find_user_id_by_email", {"email": "d@e.f"})])])
    m, st = ag.generate_next_message(UserMessage(role="user", content="hi, a@b.c"), st)
    assert len(m.tool_calls) == 1 and m.content is None
    a = m.raw_data["axi"]
    assert a["dropped_calls"] == 1 and a["dropped_text"] == 1


def test_empty_reply_is_asked_again():
    ag, st, s = agent(True, ["", "How can I help?"])
    m, st = ag.generate_next_message(UserMessage(role="user", content="hi"), st)
    assert m.content == "How can I help?" and m.raw_data["axi"]["empty_retries"] == 1


def test_multi_tool_message_input():
    ag, st, s = agent(True, [("", [("find_user_id_by_email", {"email": "n@example.com"})]), "ok"])
    m, st = ag.generate_next_message(UserMessage(role="user", content="hi n@example.com"), st)
    mt = MultiToolMessage(role="tool", tool_messages=[tool_result(m, "noah_brown_6181")])
    m2, st = ag.generate_next_message(mt, st)
    assert st.held.user_id == "noah_brown_6181" and m2.content == "ok"


def _summary_agent(replies, compact_over=4, keep_last=2):
    import tau2.agent.llm_agent as LA  # arm S answers through the benchmark's own agent, which calls its own `generate`

    s = Script(replies)
    A.generate = s
    LA.generate = s
    ag = A.SummaryLLMAgent([], "POLICY TEXT", llm="fake", llm_args={}, compact_over=compact_over, keep_last=keep_last)
    return ag, ag.get_init_state(), s


def _chat(ag, st, texts):
    out = []
    for t in texts:
        m, st = ag.generate_next_message(UserMessage(role="user", content=t), st)
        out.append(m)
    return out, st


def test_arm_s_rolls_up_the_older_chat_and_counts_the_summary_call():
    ag, st, s = _summary_agent(["A1", "A2", "A3", "SUMMARY TEXT", "A4"])
    out, st = _chat(ag, st, ["u1", "u2", "u3"])
    assert len(s.sent) == 3 and all("axi" not in (m.raw_data or {}) for m in out)  # no summary yet: the plain benchmark agent
    out4, st = _chat(ag, st, ["u4"])
    m4 = out4[0]
    assert len(s.sent) == 5  # the summary call, then the answer
    sum_call, main_call = s.sent[3], s.sent[4]
    assert sum_call[0].content == A.SUMMARY_SYSTEM and "CUSTOMER: u1" in sum_call[1].content and "AGENT: A2" in sum_call[1].content
    assert "u3" not in sum_call[1].content  # the last two messages stay in the chat
    assert [getattr(m, "content", "") for m in main_call[1:]][0].startswith(A.SUMMARY_PREFIX)
    assert [m.content for m in main_call[2:]] == ["u3", "A3", "u4"]
    assert m4.usage["prompt_tokens"] == s.ptoks[3] + s.ptoks[4] and m4.usage["completion_tokens"] == 10
    assert abs(m4.cost - 0.002) < 1e-12
    ax = m4.raw_data["axi"]
    assert ax["summary_calls"] == 1 and ax["summary_applied"] is True and ax["llm_calls"] == 2 and ax["cached_tokens"] == 6
    assert st.summaries == 1 and len(st.messages) == 5  # summary, u3, A3, u4, A4


def test_arm_s_an_empty_summary_keeps_the_chat_whole_but_the_call_is_counted():
    ag, st, s = _summary_agent(["A1", "A2", "A3", "", "A4"])
    out, st = _chat(ag, st, ["u1", "u2", "u3", "u4"])
    main_call = s.sent[4]
    assert [m.content for m in main_call[1:]][0] == "u1" and len(main_call) == 1 + 7
    ax = out[3].raw_data["axi"]
    assert ax["summary_calls"] == 1 and ax["summary_applied"] is False and st.summaries == 0


def test_arm_s_a_second_roll_up_folds_the_first_summary_in():
    # with compact_over=4 and keep_last=2 the chat is still over the limit right after a roll-up, so the next turn rolls up again
    ag, st, s = _summary_agent(["A1", "A2", "A3", "S1", "A4", "S2", "A5"], compact_over=4, keep_last=2)
    out, st = _chat(ag, st, ["u1", "u2", "u3", "u4", "u5"])
    second = [c for c in s.sent if c[0].content == A.SUMMARY_SYSTEM][1]
    assert "EARLIER SUMMARY: S1" in second[1].content
    assert st.summaries == 2 and st.messages[0].content.startswith(A.SUMMARY_PREFIX) and "S2" in st.messages[0].content


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")

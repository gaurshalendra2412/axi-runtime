"""Test 47 glue: the gate and the held state as agents of tau2-bench (docs/TAU2_PILOT.md).

Needs the `tau2` package (run it in the notebook `python/colab/axi_tau2_pilot.ipynb`). The rules
themselves are in `tau2_retail_core.py` and are tested without tau2.

Two agents are registered with the benchmark:

* `gate_chat` (arm G): the benchmark's own agent, whole chat kept, plus the gate.
* `held_state` (arm A): the host holds the state and rebuilds a short prompt at every step.

Arm B is the benchmark's `llm_agent`, unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent, LLMAgentState  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import generate  # noqa: E402

from axi.experiments.tau2_retail_core import RetailHeld, summary_cut  # noqa: E402

HELD_STATE_NOTE = """
<host_state_note>
At every step the host sends you its held state and the one new event, instead of the whole earlier chat.
- authenticated_customer: the user id the host has recorded.
- facts_read: the latest result of every distinct read you made (a read made again replaces the earlier copy; a change replaces the order with the updated order).
- changes_made: the changes already made to the database.
- dialogue_so_far: the messages between you and the customer.
- new_event: the customer's new message, a tool's result, or a refusal by the host of the call you proposed.
The host checks every tool call against the policy before it runs. A refused call did not run: use the reason it gives.
</host_state_note>
""".strip()


class GatedState(LLMAgentState):
    held: Any = None
    pending: Any = None


def _new_stats() -> dict:
    return {"llm_calls": 0, "reasks": 0, "refusals": [], "fallbacks": 0, "dropped_calls": 0,
            "dropped_text": 0, "empty_retries": 0}


def _cached(raw: Optional[dict]) -> int:
    try:
        d = (raw or {}).get("usage", {}).get("prompt_tokens_details") or {}
        return int(d.get("cached_tokens") or 0)
    except Exception:
        return 0


class GatedLLMAgent(LLMAgent):
    """The benchmark's LLM agent with the host's gate; optionally with the held state in place of the chat."""

    def __init__(self, tools, domain_policy, llm, llm_args=None, additional_stop_tool_names=frozenset(),
                 hold_state: bool = False, max_reasks: int = 2):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args,
                         additional_stop_tool_names=additional_stop_tool_names)
        self.hold_state = hold_state
        self.max_reasks = max_reasks

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)
        return base + "\n" + HELD_STATE_NOTE if self.hold_state else base

    def get_init_state(self, message_history=None) -> GatedState:
        history = list(message_history or [])
        held = RetailHeld(self.domain_policy, AGENT_INSTRUCTION)
        for m in history:
            if isinstance(m, AssistantMessage) and m.content and not m.is_tool_call():
                held.on_agent_text(m.content)
            elif isinstance(m, UserMessage) and m.content:
                held.on_user(m.content)
        return GatedState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=[] if self.hold_state else history,
            held=held,
            pending=None,
            stats=_new_stats(),
        )

    # ------------------------------------------------------------------
    def generate_next_message(self, message, state: GatedState):
        held: RetailHeld = state.held
        stats = _new_stats()
        event = self._take(message, state)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        cost: Optional[float] = 0.0
        cached = 0
        last_raw: Optional[dict] = None
        reasks = 0
        empties = 0
        final: Optional[AssistantMessage] = None
        while final is None:
            if self.hold_state:
                msgs = state.system_messages + [UserMessage(role="user", content=held.render(event))]
            else:
                msgs = state.system_messages + state.messages
            am = generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response",
                          **self.llm_args)
            stats["llm_calls"] += 1
            if am.usage:
                usage["prompt_tokens"] += am.usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += am.usage.get("completion_tokens", 0)
            cost = None if (cost is None or am.cost is None) else cost + am.cost
            cached += _cached(am.raw_data)
            last_raw = am.raw_data
            calls: List = list(am.tool_calls or [])
            text = (am.content or "").strip()
            if calls:
                if len(calls) > 1:
                    stats["dropped_calls"] += len(calls) - 1
                if text:
                    stats["dropped_text"] += 1
                call = calls[0]
                verdict = held.gate(call.name, call.arguments)
                if verdict.ok:
                    held.admit(call.name, call.arguments)
                    state.pending = (call.id, call.name, dict(call.arguments))
                    final = AssistantMessage(role="assistant", content=None, tool_calls=[call])
                    break
                held.note_refusal(call.name, verdict)
                stats["refusals"].append([verdict.code, call.name])
                if reasks < self.max_reasks:
                    reasks += 1
                    if self.hold_state:
                        event = ("refused", call.name, dict(call.arguments), verdict.reason)
                    else:
                        state.messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[call]))
                        state.messages.append(ToolMessage(id=call.id, role="tool", error=True,
                                                          content=f"Refused by the host, the call did not run: {verdict.reason}"))
                    continue
                stats["fallbacks"] += 1
                text = held.fallback_message(verdict)
                held.on_agent_text(text)
                final = AssistantMessage(role="assistant", content=text)
                break
            if text:
                held.on_agent_text(text)
                final = AssistantMessage(role="assistant", content=text)
                break
            # an empty reply: ask again (counted), then a plain message
            if empties < self.max_reasks:
                empties += 1
                stats["empty_retries"] += 1
                continue
            text = "I'm sorry, could you tell me a little more about what you need?"
            stats["fallbacks"] += 1
            held.on_agent_text(text)
            final = AssistantMessage(role="assistant", content=text)
        stats["reasks"] = reasks
        final.usage = usage
        final.cost = cost
        raw = dict(last_raw or {})
        raw["axi"] = dict(stats, prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
                          cached_tokens=cached, hold_state=self.hold_state)
        final.raw_data = raw
        if not self.hold_state:
            state.messages.append(final)
        return final, state

    def _take(self, message, state: GatedState):
        """Put the incoming message into the held state (and the chat, for arm G); return the new event."""
        held: RetailHeld = state.held
        if isinstance(message, MultiToolMessage):
            tool_msgs = list(message.tool_messages)
        elif isinstance(message, ToolMessage):
            tool_msgs = [message]
        else:
            tool_msgs = None
        if tool_msgs is None:
            held.on_user(message.content or "")
            if not self.hold_state:
                state.messages.append(message)
            return ("customer", message.content or "")
        if not self.hold_state:
            state.messages.extend(tool_msgs)
        event = None
        for tm in tool_msgs:
            pend = state.pending
            name, args = (pend[1], pend[2]) if pend and pend[0] == tm.id else ("unknown_tool", {})
            held.on_tool(name, args, tm.content or "", bool(tm.error))
            event = ("tool", name, args, tm.content or "")
        state.pending = None
        return event


SUMMARY_PREFIX = "[Summary of the earlier part of this conversation, written by you when the chat grew long]"
SUMMARY_SYSTEM = (
    "You keep the memory of a customer-service agent for an online retail company. Below is the earlier part of the agent's "
    "conversation with a customer: the customer's messages, the agent's messages, the agent's tool calls and the tools' results. "
    "It is about to be removed from the agent's chat and replaced by your summary, so the summary must carry everything the agent "
    "needs to carry on correctly: who the customer is and whether the identity is verified, the orders and items looked up (ids, "
    "statuses, prices, payment methods), what the customer asked for, what the agent proposed and what the customer confirmed or "
    "refused, every change already made to the database, and anything still open. If the text starts with an earlier summary, "
    "fold it in. Write the summary only, as plain text, as short as it can be without losing a fact the agent needs."
)
MAX_PART = 4000  # characters kept of one message when it is written out for the summary call


def _written_out(m) -> str:
    """One message of the chat as a line of text for the summary call."""
    text = (getattr(m, "content", None) or "").strip()
    if text.startswith(SUMMARY_PREFIX):
        return "EARLIER SUMMARY: " + text[len(SUMMARY_PREFIX):].strip()[:MAX_PART]
    role = getattr(m, "role", "")
    if role == "tool":
        return ("TOOL ERROR: " if getattr(m, "error", False) else "TOOL RESULT: ") + text[:MAX_PART]
    if role == "assistant":
        calls = getattr(m, "tool_calls", None) or []
        if calls:
            return "AGENT CALLED: " + "; ".join(f"{c.name}({json.dumps(c.arguments, sort_keys=True, default=str)})" for c in calls)
        return "AGENT: " + text[:MAX_PART]
    return "CUSTOMER: " + text[:MAX_PART]


class SummaryState(LLMAgentState):
    summaries: int = 0


class SummaryLLMAgent(LLMAgent):
    """Arm S: the benchmark's own agent (no gate, no held state) whose chat is rolled up by the model itself when it grows.

    When the chat holds more than `compact_over` messages, everything but the last `keep_last` messages is written out and sent to the
    same model with one instruction (SUMMARY_SYSTEM); the reply replaces those messages as one message at the top of the chat, and
    the summary call is added to the tokens and the cost of the agent message it precedes. This is the plain compaction that agent
    frameworks and chat products offer, built here from that description; it is not a copy of any product.
    """

    def __init__(self, tools, domain_policy, llm, llm_args=None, additional_stop_tool_names=frozenset(),
                 compact_over: int = 24, keep_last: int = 10):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args,
                         additional_stop_tool_names=additional_stop_tool_names)
        self.compact_over = compact_over
        self.keep_last = keep_last

    def get_init_state(self, message_history=None) -> SummaryState:
        return SummaryState(system_messages=[SystemMessage(role="system", content=self.system_prompt)],
                            messages=list(message_history or []))

    def _roll_up(self, state: SummaryState):
        """Replace the older part of the chat by a summary; return (the summary call's message, whether it replaced the chat), or (None, False)."""
        msgs = state.messages
        if len(msgs) <= self.compact_over:
            return None, False
        cut = summary_cut([getattr(m, "role", "") for m in msgs], self.keep_last)
        if cut <= 0:
            return None, False
        text_out = "\n".join(_written_out(m) for m in msgs[:cut])
        sm = generate(model=self.llm, messages=[SystemMessage(role="system", content=SUMMARY_SYSTEM),
                                                UserMessage(role="user", content=text_out)],
                      call_name="agent_summary", **self.llm_args)
        text = (sm.content or "").strip()
        if not text:
            return sm, False  # an empty reply: the chat is kept whole this time; the call is still counted
        state.messages = [UserMessage(role="user", content=f"{SUMMARY_PREFIX}\n{text}")] + msgs[cut:]
        state.summaries += 1
        return sm, True

    def generate_next_message(self, message, state: SummaryState):
        sm, applied = self._roll_up(state)
        am, state = super().generate_next_message(message, state)
        if sm is not None:
            su = sm.usage or {}
            u = dict(am.usage or {})
            u["prompt_tokens"] = (u.get("prompt_tokens", 0) or 0) + (su.get("prompt_tokens", 0) or 0)
            u["completion_tokens"] = (u.get("completion_tokens", 0) or 0) + (su.get("completion_tokens", 0) or 0)
            am.usage = u
            am.cost = None if (am.cost is None or sm.cost is None) else am.cost + sm.cost
            raw = dict(am.raw_data or {})
            raw["axi"] = {"llm_calls": 2, "summary_calls": 1, "summary_prompt_tokens": su.get("prompt_tokens", 0) or 0,
                          "summary_completion_tokens": su.get("completion_tokens", 0) or 0,
                          "summary_applied": applied,
                          "cached_tokens": _cached(raw) + _cached(sm.raw_data),
                          "refusals": [], "fallbacks": 0, "dropped_calls": 0, "dropped_text": 0, "empty_retries": 0}
            am.raw_data = raw
        return am, state


def create_gate_chat_agent(tools, domain_policy, **kwargs):
    return GatedLLMAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                         llm_args=kwargs.get("llm_args"),
                         additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
                         hold_state=False)


def create_held_state_agent(tools, domain_policy, **kwargs):
    return GatedLLMAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                         llm_args=kwargs.get("llm_args"),
                         additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
                         hold_state=True)


def create_summary_agent(tools, domain_policy, **kwargs):
    return SummaryLLMAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                           llm_args=kwargs.get("llm_args"),
                           additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()))


def register() -> None:
    from tau2.registry import registry  # imported here: the registry loads every domain of the benchmark

    for name, factory in (("gate_chat", create_gate_chat_agent), ("held_state", create_held_state_agent),
                          ("summary_chat", create_summary_agent)):
        if registry.get_agent_factory(name) is None:
            registry.register_agent_factory(factory, name)

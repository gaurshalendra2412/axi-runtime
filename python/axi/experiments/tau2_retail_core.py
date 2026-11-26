"""Test 47 core: the host-held state and the deterministic gate for tau2-bench retail.

Plain Python, no model and no tau2 import, so every rule here is tested offline
(`python/tests/test_tau2_retail_core.py`). The plan is `docs/TAU2_PILOT.md`.

`RetailHeld` is what the host keeps while one customer conversation runs:

* the customer id once found, the latest result of every distinct read, the
  changes made, the dialogue as text, and the one new event;
* `gate(name, args)` says whether a proposed tool call may go to the tool;
* `render(event)` builds the short prompt of arm A from that state (no tool call or
  result of the past is replayed, the held copy of each read is the latest one).

The gate rules are exactly those of the plan page: one tool call per message,
authentication first, the authenticated customer's own user and orders, an order
read before it is changed, the order status each tool needs, items modification
and exchange once per order, the two allowed cancel reasons, and a fresh explicit
confirmation for every write (a "yes" after the details were listed, used up by
the write).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

READ_TOOLS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
    }
)
# Tools that change the database. Each needs a fresh confirmation.
WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
# Tools that need the customer to be authenticated first (user and order tools).
NEEDS_AUTH = frozenset(
    {"get_user_details", "get_order_details"} | WRITE_TOOLS
)
# Order tools and the status the order must have for each.
ORDER_STATUS_NEEDED = {
    "cancel_pending_order": "pending",
    "modify_pending_order_address": "pending",
    "modify_pending_order_items": "pending",
    "modify_pending_order_payment": "pending",
    "return_delivered_order_items": "delivered",
    "exchange_delivered_order_items": "delivered",
}
CANCEL_REASONS = ("no longer needed", "ordered by mistake")
USER_ID_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*_\d+$")

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|confirm(?:ed)?|proceed|go ahead|go for it|do it|"
    r"please do|that'?s right|that is correct|correct|sounds good|ok|okay|"
    r"absolutely|definitely|affirmative|that works|works for me|perfect|alright|all right|"
    r"i agree|agreed|let'?s do (?:it|that))\b"
)
_HEDGE_RE = re.compile(
    r"\b(no|not|don'?t|do not|wait|hold on|stop|never ?mind|instead|but|however|rather)\b"
)


def is_confirmation(text: str) -> bool:
    """True when a customer message is an explicit yes with no hedge in it."""
    t = text.lower().replace("’", "'")
    for phrase in ("no longer needed", "no problem", "not a problem"):  # not refusals
        t = t.replace(phrase, " ")
    return bool(_YES_RE.search(t)) and not _HEDGE_RE.search(t)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str = ""
    reason: str = ""


OK = Verdict(True)


def _refuse(code: str, reason: str) -> Verdict:
    return Verdict(False, code, reason)


def _key(name: str, args: Dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def _parse(content: str) -> Optional[Any]:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return None


class RetailHeld:
    """The state the host holds for one retail conversation."""

    def __init__(self, policy: str, instruction: str = "") -> None:
        self.policy = policy
        self.instruction = instruction
        self.user_id: Optional[str] = None
        self.user: Optional[dict] = None
        self.orders: Dict[str, dict] = {}
        self.foreign_orders: set = set()
        # latest result of each distinct read, oldest first: key -> (name, args, content)
        self.reads: Dict[str, Tuple[str, dict, str]] = {}
        self.dialogue: List[Tuple[str, str]] = []
        self.changes: List[str] = []
        self.items_changed: set = set()
        self.exchanged: set = set()
        self.confirmed = False
        self._awaiting_reply = False  # the last event was an agent text message
        self.refusals: List[Tuple[str, str]] = []  # (code, tool name)

    # ---- events ---------------------------------------------------------
    def on_agent_text(self, text: str) -> None:
        self.dialogue.append(("agent", text))
        self._awaiting_reply = True
        self.confirmed = False

    def on_user(self, text: str) -> None:
        self.dialogue.append(("customer", text))
        self.confirmed = self._awaiting_reply and is_confirmation(text)
        self._awaiting_reply = False

    def admit(self, name: str, args: Dict[str, Any]) -> None:
        """A call passed the gate and goes to the tool: a write uses up the confirmation."""
        self._awaiting_reply = False
        if name in WRITE_TOOLS:
            self.confirmed = False

    def on_tool(self, name: str, args: Dict[str, Any], content: str, error: bool = False) -> None:
        """Record a tool's result: replace the held copy of the same read, update the state."""
        self._awaiting_reply = False
        key = _key(name, args)
        data = None if error else _parse(content)
        if name in ("find_user_id_by_email", "find_user_id_by_name_zip") and not error:
            if USER_ID_RE.match(content.strip()):
                if self.user_id is None:
                    self.user_id = content.strip()
        if isinstance(data, dict) and "order_id" in data and "user_id" in data:
            oid = data["order_id"]
            if self.user_id is not None and data["user_id"] != self.user_id:
                self.foreign_orders.add(oid)
                self.orders.pop(oid, None)
                content = "(this order belongs to another customer; it is not discussed or changed)"
            else:
                self.orders[oid] = data
            if name in WRITE_TOOLS and not error:
                self._record_change(name, args)
                # one copy of the order: the updated one the tool returned
                self.reads.pop(_key("get_order_details", {"order_id": oid}), None)
                key = _key("get_order_details", {"order_id": oid})
                name, args = "get_order_details", {"order_id": oid}
        elif isinstance(data, dict) and "user_id" in data and "payment_methods" in data:
            self.user = data
            if name in WRITE_TOOLS and not error:
                self._record_change(name, args)
                key = _key("get_user_details", {"user_id": data["user_id"]})
                name, args = "get_user_details", {"user_id": data["user_id"]}
        elif name in WRITE_TOOLS and not error:
            self._record_change(name, args)
        self.reads.pop(key, None)
        self.reads[key] = (name, args, content)

    def _record_change(self, name: str, args: Dict[str, Any]) -> None:
        self.changes.append(f"{name}({json.dumps(args, sort_keys=True, default=str)})")
        oid = args.get("order_id")
        if name == "modify_pending_order_items" and oid:
            self.items_changed.add(oid)
        if name == "exchange_delivered_order_items" and oid:
            self.exchanged.add(oid)

    # ---- the gate -------------------------------------------------------
    def gate(self, name: str, args: Dict[str, Any]) -> Verdict:
        if name in NEEDS_AUTH and self.user_id is None:
            return _refuse(
                "AUTH",
                "the customer is not authenticated yet; find their user id by email, or by "
                "name and zip code, first",
            )
        if name in ("get_user_details", "modify_user_address"):
            if args.get("user_id") != self.user_id:
                return _refuse(
                    "FOREIGN_USER",
                    "only the authenticated customer's own user id may be used; other users "
                    "must be refused",
                )
        oid = args.get("order_id")
        if name == "get_order_details" or name in ORDER_STATUS_NEEDED:
            if oid in self.foreign_orders:
                return _refuse("FOREIGN_ORDER", f"order {oid} belongs to another customer")
            if self.user is not None and oid not in self.user.get("orders", []):
                return _refuse(
                    "FOREIGN_ORDER",
                    f"order {oid} is not one of the authenticated customer's orders",
                )
        if name in ORDER_STATUS_NEEDED:
            order = self.orders.get(oid)
            if order is None:
                return _refuse("READ_FIRST", f"read order {oid} with get_order_details before changing it")
            need = ORDER_STATUS_NEEDED[name]
            if order.get("status") != need:
                return _refuse(
                    "STATUS",
                    f"{name} needs an order with status '{need}', and order {oid} has status "
                    f"'{order.get('status')}'",
                )
            if name == "modify_pending_order_items" and oid in self.items_changed:
                return _refuse("ONCE", f"items of order {oid} were already modified once")
            if name == "exchange_delivered_order_items" and oid in self.exchanged:
                return _refuse("ONCE", f"items of order {oid} were already exchanged once")
            if name == "cancel_pending_order" and args.get("reason") not in CANCEL_REASONS:
                return _refuse(
                    "REASON",
                    "the cancel reason must be exactly 'no longer needed' or 'ordered by mistake'",
                )
        if name in WRITE_TOOLS and not self.confirmed:
            return _refuse(
                "CONFIRM",
                "list the action details to the customer and get an explicit yes before this "
                "change; the confirmation applies to one change only",
            )
        return OK

    def note_refusal(self, name: str, verdict: Verdict) -> None:
        self.refusals.append((verdict.code, name))

    def fallback_message(self, verdict: Verdict) -> str:
        if verdict.code == "CONFIRM":
            return (
                "Before I make this change, please confirm that you want me to go ahead "
                "with it (yes/no)."
            )
        return f"I'm sorry, I can't do that: {verdict.reason}."

    # ---- the prompt of arm A ---------------------------------------------
    def render(self, event: Tuple[str, ...]) -> str:
        """The short prompt: the held state and the one new event.

        `event` is ("customer", text), ("tool", name, args, content) or
        ("refused", name, args, reason).
        """
        lines: List[str] = ["<held_state>"]
        lines.append(f"authenticated_customer: {self.user_id or 'none yet'}")
        lines.append("facts_read (latest result of each distinct read):")
        held_reads = list(self.reads.values())
        if event[0] == "tool":
            held_reads = held_reads[:-1]  # the newest result is the new event, shown once, below
        if not held_reads:
            lines.append("  (none yet)")
        for name, args, content in held_reads:
            lines.append(f"  {name}({json.dumps(args, sort_keys=True, default=str)}) -> {content}")
        lines.append("changes_made:")
        if not self.changes:
            lines.append("  (none yet)")
        for c in self.changes:
            lines.append(f"  {c}")
        lines.append("dialogue_so_far:")
        if not self.dialogue:
            lines.append("  (nothing yet)")
        for who, text in self.dialogue:
            lines.append(f"  {who}: {text}")
        lines.append("</held_state>")
        kind = event[0]
        if kind == "customer":
            lines.append(f"<new_event>\nthe customer wrote: {event[1]}\n</new_event>")
        elif kind == "tool":
            _, name, args, content = event
            lines.append(
                f"<new_event>\nresult of {name}({json.dumps(args, sort_keys=True, default=str)}): "
                f"{content}\n</new_event>"
            )
        elif kind == "refused":
            _, name, args, reason = event
            lines.append(
                f"<new_event>\nthe host refused your call {name}("
                f"{json.dumps(args, sort_keys=True, default=str)}): {reason}\n</new_event>"
            )
        else:  # pragma: no cover
            raise ValueError(kind)
        lines.append(
            "Reply with either one message to the customer or one tool call, as the policy says."
        )
        return "\n".join(lines)

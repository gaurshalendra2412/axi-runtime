"""Offline checks of the Test 47 core (docs/TAU2_PILOT.md): the held state and the gate for tau2-bench retail.
No model call, no network, no tau2 import."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi.experiments.tau2_retail_core import RetailHeld, is_confirmation  # noqa: E402

ADDR = {"address1": "1 A St", "address2": "", "city": "X", "country": "USA", "state": "CO", "zip": "80279"}


def order(oid="#W1", status="pending", user="noah_brown_6181", items=None):
    items = items or [{"name": "Keyboard", "product_id": "p1", "item_id": "i1", "price": 10.0, "options": {"k": "a"}}]
    return {"order_id": oid, "user_id": user, "address": ADDR, "items": items, "status": status,
            "fulfillments": [], "payment_history": [{"transaction_type": "payment", "amount": 10.0, "payment_method_id": "gift_card_1"}]}


def user_details(orders=("#W1",)):
    return {"user_id": "noah_brown_6181", "name": {"first_name": "Noah", "last_name": "Brown"}, "address": ADDR,
            "email": "n@example.com", "payment_methods": {"gift_card_1": {"source": "gift_card", "id": "gift_card_1", "balance": 5.0}},
            "orders": list(orders)}


def authed(**kw):
    h = RetailHeld("policy text")
    h.on_tool("find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181")
    return h


def read_order(h, o):
    h.on_tool("get_order_details", {"order_id": o["order_id"]}, json.dumps(o))


def confirm(h, text="Yes, please go ahead."):
    h.on_agent_text("Here are the details. Shall I proceed?")
    h.on_user(text)


def code(h, name, args):
    return h.gate(name, args).code


def test_confirmation_detector():
    assert is_confirmation("Yes, please go ahead.")
    assert is_confirmation("yes, cancel it, no longer needed")  # a cancel reason is not a refusal
    assert is_confirmation("Okay, do it")
    assert is_confirmation("That works for me.") and is_confirmation("No problem, go ahead") and is_confirmation("Perfect, let's do it")
    assert not is_confirmation("Yes, but can you also change the address?")
    assert not is_confirmation("No, wait")
    assert not is_confirmation("I'm not sure")
    assert not is_confirmation("What is the price?")


def test_authentication_comes_first():
    h = RetailHeld("p")
    assert code(h, "get_order_details", {"order_id": "#W1"}) == "AUTH"
    assert code(h, "get_user_details", {"user_id": "x_y_1"}) == "AUTH"
    assert h.gate("find_user_id_by_email", {"email": "a@b.c"}).ok
    assert h.gate("get_product_details", {"product_id": "p1"}).ok
    h.on_tool("find_user_id_by_email", {"email": "a@b.c"}, "Error: User not found", error=True)
    assert h.user_id is None
    h.on_tool("find_user_id_by_name_zip", {"first_name": "a", "last_name": "b", "zip": "1"}, "noah_brown_6181")
    assert h.user_id == "noah_brown_6181"
    assert h.gate("get_order_details", {"order_id": "#W1"}).ok


def test_only_the_authenticated_customers_user_and_orders():
    h = authed()
    assert code(h, "get_user_details", {"user_id": "someone_else_1"}) == "FOREIGN_USER"
    assert h.gate("get_user_details", {"user_id": "noah_brown_6181"}).ok
    h.on_tool("get_user_details", {"user_id": "noah_brown_6181"}, json.dumps(user_details()))
    assert code(h, "get_order_details", {"order_id": "#W9"}) == "FOREIGN_ORDER"  # not in the customer's list
    assert h.gate("get_order_details", {"order_id": "#W1"}).ok
    # an order read before the user details, which turns out to be someone else's
    h2 = authed()
    h2.on_tool("get_order_details", {"order_id": "#W5"}, json.dumps(order("#W5", user="other_person_2")))
    assert "#W5" in h2.foreign_orders and "#W5" not in h2.orders
    assert code(h2, "get_order_details", {"order_id": "#W5"}) == "FOREIGN_ORDER"
    held = [v[2] for v in h2.reads.values() if v[0] == "get_order_details"]
    assert held and "another customer" in held[0] and "other_person_2" not in held[0]  # its content is not held
    assert code(h, "modify_user_address", {"user_id": "other_1"}) == "FOREIGN_USER"


def test_order_read_before_change_and_status_rules():
    h = authed()
    confirm(h)
    assert code(h, "cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}) == "READ_FIRST"
    read_order(h, order("#W1", "delivered"))
    assert code(h, "cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}) == "STATUS"
    assert code(h, "modify_pending_order_payment", {"order_id": "#W1", "payment_method_id": "g"}) == "STATUS"
    assert h.gate("return_delivered_order_items", {"order_id": "#W1", "item_ids": ["i1"], "payment_method_id": "g"}).ok
    assert h.gate("exchange_delivered_order_items", {"order_id": "#W1", "item_ids": ["i1"], "new_item_ids": ["i2"], "payment_method_id": "g"}).ok
    read_order(h, order("#W2", "pending (item modified)"))
    assert code(h, "cancel_pending_order", {"order_id": "#W2", "reason": "ordered by mistake"}) == "STATUS"
    read_order(h, order("#W3", "pending"))
    assert code(h, "return_delivered_order_items", {"order_id": "#W3", "item_ids": ["i1"], "payment_method_id": "g"}) == "STATUS"
    assert h.gate("modify_pending_order_items", {"order_id": "#W3", "item_ids": ["i1"], "new_item_ids": ["i2"], "payment_method_id": "g"}).ok


def test_cancel_reason():
    h = authed()
    read_order(h, order())
    confirm(h)
    assert code(h, "cancel_pending_order", {"order_id": "#W1", "reason": "too slow"}) == "REASON"
    assert code(h, "cancel_pending_order", {"order_id": "#W1", "reason": "No longer needed"}) == "REASON"
    assert h.gate("cancel_pending_order", {"order_id": "#W1", "reason": "ordered by mistake"}).ok


def test_confirmation_is_needed_and_used_up_by_the_write():
    h = authed()
    read_order(h, order())
    args = {"order_id": "#W1", "reason": "no longer needed"}
    assert code(h, "cancel_pending_order", args) == "CONFIRM"  # no listing and no yes
    h.on_user("please cancel #W1")  # a request is not a confirmation
    assert code(h, "cancel_pending_order", args) == "CONFIRM"
    h.on_agent_text("I will cancel #W1 for 'no longer needed'. Confirm (yes)?")
    h.on_user("yes")
    assert h.gate("cancel_pending_order", args).ok
    assert h.gate("get_order_details", {"order_id": "#W1"}).ok  # a read keeps the confirmation
    h.admit("get_order_details", {"order_id": "#W1"})
    assert h.confirmed
    h.admit("cancel_pending_order", args)
    assert not h.confirmed  # used up by the write
    h.on_tool("cancel_pending_order", args, json.dumps(order("#W1", "cancelled")))
    read_order(h, order("#W2"))
    assert code(h, "cancel_pending_order", {"order_id": "#W2", "reason": "no longer needed"}) == "CONFIRM"


def test_a_yes_that_does_not_answer_an_agent_message_is_not_a_confirmation():
    h = authed()
    read_order(h, order())
    h.on_user("yes")  # nothing was listed before it
    assert not h.confirmed
    h.on_agent_text("Shall I proceed?")
    h.on_user("Yes, but also change the address")
    assert not h.confirmed


def test_items_modification_and_exchange_once_per_order():
    h = authed()
    args = {"order_id": "#W1", "item_ids": ["i1"], "new_item_ids": ["i2"], "payment_method_id": "g"}
    read_order(h, order("#W1", "pending"))
    confirm(h)
    assert h.gate("modify_pending_order_items", args).ok
    h.admit("modify_pending_order_items", args)
    h.on_tool("modify_pending_order_items", args, json.dumps(order("#W1", "pending (item modified)")))
    assert h.orders["#W1"]["status"] == "pending (item modified)"
    confirm(h)
    assert code(h, "modify_pending_order_items", args) in ("STATUS", "ONCE")
    assert "#W1" in h.items_changed
    h2 = authed()
    read_order(h2, order("#W4", "delivered"))
    h2.exchanged.add("#W4")
    confirm(h2)
    assert code(h2, "exchange_delivered_order_items", dict(args, order_id="#W4")) == "ONCE"


def test_held_state_keeps_one_latest_copy_of_each_read():
    h = authed()
    read_order(h, order("#W1", "pending"))
    read_order(h, order("#W1", "pending"))
    assert sum(1 for k in h.reads if k.startswith("get_order_details")) == 1
    confirm(h)
    args = {"order_id": "#W1", "reason": "no longer needed"}
    h.admit("cancel_pending_order", args)
    h.on_tool("cancel_pending_order", args, json.dumps(order("#W1", "cancelled")))
    copies = [v for v in h.reads.values() if v[0] == "get_order_details"]
    assert len(copies) == 1 and '"cancelled"' in copies[0][2]
    assert h.changes == ['cancel_pending_order({"order_id": "#W1", "reason": "no longer needed"})']
    assert not any(v[0] == "cancel_pending_order" for v in h.reads.values())
    # a failed write is not a change
    h.on_tool("modify_pending_order_payment", {"order_id": "#W1", "payment_method_id": "x"}, "Error: Payment method not found", error=True)
    assert len(h.changes) == 1


def test_dialogue_first_order_changes_nothing_but_the_order():
    h = RetailHeld("policy text")
    h.on_agent_text("Hi! How can I help you today?")
    h.on_user("I want to return an item; my email is n@example.com")
    h.on_tool("find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181")
    h.on_tool("get_order_details", {"order_id": "#W1"}, json.dumps(order("#W1", "delivered")))
    ev = ("tool", "get_order_details", {"order_id": "#W1"}, json.dumps(order("#W1", "delivered")))
    a = h.render(ev)
    a2 = h.render(ev, dialogue_first=True)
    assert a == h.render(ev, dialogue_first=False)  # arm A's order is the default
    assert sorted(a.split("\n")) == sorted(a2.split("\n")) and a != a2
    assert a2.startswith("<held_state>\ndialogue_so_far:") and a.startswith("<held_state>\nauthenticated_customer:")
    assert a2.index("dialogue_so_far") < a2.index("authenticated_customer") < a2.index("facts_read") < a2.index("changes_made") < a2.index("<new_event>")

    def shared(x, y):
        i = 0
        while i < min(len(x), len(y)) and x[i] == y[i]:
            i += 1
        return i

    # two steps later (another read, then the customer speaks): the start of the dialogue-first prompt is the longer match
    h2 = RetailHeld("policy text")
    h2.on_agent_text("Hi!"); h2.on_user("return please, n@example.com")
    h2.on_tool("find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181")
    first = (h2.render(("tool", "find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181")),
             h2.render(("tool", "find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181"), dialogue_first=True))
    h2.on_tool("get_user_details", {"user_id": "noah_brown_6181"}, json.dumps({"user_id": "noah_brown_6181", "payment_methods": {}, "orders": ["#W1"]}))
    ev2 = ("tool", "get_user_details", {"user_id": "noah_brown_6181"}, json.dumps({"user_id": "noah_brown_6181", "payment_methods": {}, "orders": ["#W1"]}))
    second = (h2.render(ev2), h2.render(ev2, dialogue_first=True))
    assert shared(first[1], second[1]) > shared(first[0], second[0])


def test_render_is_short_and_never_replays_the_past():
    h = RetailHeld("policy text")
    h.on_agent_text("Hi! How can I help you today?")
    h.on_user("I want to cancel #W1; my email is n@example.com")
    big = order("#W1", "pending", items=[{"name": f"Thing {i}", "product_id": f"p{i}", "item_id": f"i{i}", "price": 1.0, "options": {"k": "v" * 20}} for i in range(30)])
    sizes = []
    for step in range(60):  # 60 reads of the same two things: the held state does not grow
        h.on_tool("find_user_id_by_email", {"email": "n@example.com"}, "noah_brown_6181")
        h.on_tool("get_order_details", {"order_id": "#W1"}, json.dumps(big))
        sizes.append(len(h.render(("tool", "get_order_details", {"order_id": "#W1"}, json.dumps(big)))))
    assert sizes[-1] == sizes[0]
    text = h.render(("customer", "yes"))
    assert text.count('"payment_history"') == 1  # one copy of the order body
    assert "authenticated_customer: noah_brown_6181" in text
    assert "customer: I want to cancel #W1" in text and "agent: Hi! How can I help you today?" in text
    assert "the customer wrote: yes" in text
    ev = ("tool", "get_order_details", {"order_id": "#W1"}, json.dumps(big))
    t2 = h.render(ev)  # the newest result is shown once, as the new event
    assert t2.count('"payment_history"') == 1 and "result of get_order_details" in t2
    refused = h.render(("refused", "cancel_pending_order", {"order_id": "#W1", "reason": "x"}, "because"))
    assert "the host refused your call cancel_pending_order" in refused and "because" in refused


def test_fallback_messages():
    h = authed()
    read_order(h, order())
    v = h.gate("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})
    assert v.code == "CONFIRM" and "confirm" in h.fallback_message(v).lower()
    v2 = h.gate("get_user_details", {"user_id": "other_1"})
    assert h.fallback_message(v2).startswith("I'm sorry, I can't do that:")
    h.note_refusal("cancel_pending_order", v)
    assert h.refusals == [("CONFIRM", "cancel_pending_order")]


def test_summary_cut_never_lands_on_a_tool_result():
    from axi.experiments.tau2_retail_core import summary_cut
    assert summary_cut(["user", "assistant"] * 3, 4) == 2
    assert summary_cut(["user", "assistant", "tool", "tool", "assistant", "user", "assistant"], 3) == 4
    assert summary_cut(["user", "assistant", "tool", "tool", "tool"], 2) == 1  # moved back to the call the results answer
    assert summary_cut(["user", "assistant"], 10) == 0
    assert summary_cut(["tool", "tool", "tool", "assistant"], 2) == 0  # nothing can be cut without splitting a call from its results


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")

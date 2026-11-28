"""How much of each prompt repeats the previous one: a deterministic, model-free check of the prompt layouts of Test 47.

A provider's prefix cache can reuse only the part of a prompt that starts exactly like an earlier prompt, from its first character on.
So the share of each prompt that is identical to the start of the previous prompt of the same conversation is a property of the layout,
not of the provider, and it needs no model call. This script renders, for scripted retail conversations built from the benchmark's own
database and policy, the prompt of every agent call in three layouts and reports that share:

* `chat`  the benchmark's own agent (arms B and G): the whole chat, sent again at every step;
* `A`     the held-state prompt as it was run (parts that change in the middle first, the dialogue after them);
* `A2`    the same held-state text with the dialogue first and the changing parts after it (docs/TAU2_PILOT.md, arm A2).

Characters stand in for tokens (the share is the same to a good approximation). The tool definitions are the same on every call in
every layout, so they only add to the common start and are left out. The real hit rate on a provider's API also depends on the
provider (cache granularity, time to live, load), which a real run measures; this script does not.

    python3 -m axi.experiments.tau2_layout_reuse [--data <tau2-bench>/data/tau2/domains/retail]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

from axi.experiments.tau2_retail_core import HELD_STATE_NOTE, HELD_STATE_NOTE_DIALOGUE_FIRST, RetailHeld

Step = Tuple[Tuple, Tuple]  # (the event the agent receives, the agent's reply)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def reuse(prompts: List[str]) -> Tuple[int, int]:
    """(characters that repeat the start of the previous prompt, characters of all prompts but the first)."""
    kept = sum(common_prefix(prompts[i], prompts[i + 1]) for i in range(len(prompts) - 1))
    return kept, sum(len(p) for p in prompts[1:])


def chat_text(policy: str, msgs: List[str]) -> str:
    return policy + "\n" + "\n".join(msgs)


def layouts_of(policy: str, steps: List[Step]) -> Dict[str, List[str]]:
    """The prompt of every agent call of one scripted conversation, in each layout."""
    held = RetailHeld(policy)
    held.on_agent_text("Hi! How can I help you today?")
    out: Dict[str, List[str]] = {"chat": [], "A": [], "A2": []}
    msgs = ["assistant: Hi! How can I help you today?"]
    pending = None
    for event, reply in steps:
        if event[0] == "customer":
            held.on_user(event[1])
            msgs.append(f"user: {event[1]}")
        else:  # a tool result for the call the agent made last
            name, args = pending
            held.on_tool(name, args, event[1])
            msgs.append(f"tool: {event[1]}")
            event = ("tool", name, args, event[1])
        out["chat"].append(chat_text(policy, msgs))
        out["A"].append(policy + "\n" + HELD_STATE_NOTE + "\n" + held.render(event))
        out["A2"].append(policy + "\n" + HELD_STATE_NOTE_DIALOGUE_FIRST + "\n" + held.render(event, dialogue_first=True))
        if reply[0] == "call":
            _, name, args = reply
            held.admit(name, args)
            pending = (name, args)
            msgs.append(f"assistant: CALL {name}({json.dumps(args, sort_keys=True)})")
        else:
            held.on_agent_text(reply[1])
            msgs.append(f"assistant: {reply[1]}")
    return out


# ---- scripted conversations from the benchmark's database ---------------------------------------------------------------
def _j(x) -> str:
    return json.dumps(x)


def scenario(kind: str, user: dict, order: dict, products: dict) -> List[Step]:
    uid, oid, email = user["user_id"], order["order_id"], user["email"]
    first = order["items"][0]
    steps: List[Step] = [
        (("customer", f"Hi, I need help with an order. My email is {email}."), ("call", "find_user_id_by_email", {"email": email})),
        (("tool", uid), ("call", "get_user_details", {"user_id": uid})),
        (("tool", _j(user)), ("text", "Thank you, I found your account. Which order is this about, and what would you like to do?")),
        (("customer", f"It is order {oid}."), ("call", "get_order_details", {"order_id": oid})),
        (("tool", _j(order)), ("text", f"I can see order {oid}. Tell me exactly what you would like to change and I will check it.")),
    ]
    if kind == "cancel":
        args = {"order_id": oid, "reason": "no longer needed"}
        steps += [
            (("customer", "Please cancel the whole order, I do not need it any more."),
             ("text", f"I will cancel order {oid} (reason: no longer needed). The refund goes back to the original payment method. Shall I go ahead? (yes/no)")),
            (("customer", "yes"), ("call", "cancel_pending_order", args)),
            (("tool", _j(dict(order, status="cancelled"))), ("text", f"Order {oid} is cancelled and the refund is on its way. Anything else?")),
            (("customer", "No, thank you."), ("text", "You are welcome, have a nice day!")),
        ]
    elif kind == "return":
        args = {"order_id": oid, "item_ids": [first["item_id"]], "payment_method_id": order["payment_history"][0]["payment_method_id"]}
        steps += [
            (("customer", f"I want to return the {first['name']}, it does not fit."),
             ("text", f"I will return the {first['name']} from {oid} and refund it to the original payment method. Shall I go ahead? (yes/no)")),
            (("customer", "yes"), ("call", "return_delivered_order_items", args)),
            (("tool", _j(dict(order, status="return requested"))), ("text", "The return is requested; you will get an email with the return label. Anything else?")),
            (("customer", "No, thanks."), ("text", "You are welcome, goodbye!")),
        ]
    else:  # exchange: one more read, the product's variants
        pid = first["product_id"]
        args = {"order_id": oid, "item_ids": [first["item_id"]], "new_item_ids": [first["item_id"]],
                "payment_method_id": order["payment_history"][0]["payment_method_id"]}
        steps += [
            (("customer", f"I would like to exchange the {first['name']} for another variant."),
             ("call", "get_product_details", {"product_id": pid})),
            (("tool", _j(products[pid])), ("text", "These variants are available. Which one would you like? I will list the price difference before I change anything.")),
            (("customer", "The same one in another colour is fine, go ahead with the cheapest."),
             ("text", f"I will exchange the item in {oid} and settle the price difference on the original payment method. Shall I go ahead? (yes/no)")),
            (("customer", "yes"), ("call", "exchange_delivered_order_items", args)),
            (("tool", _j(dict(order, status="exchange requested"))), ("text", "The exchange is requested. Anything else?")),
            (("customer", "That is all."), ("text", "You are welcome, goodbye!")),
        ]
    return steps


def load(data_dir: str):
    db = json.load(open(os.path.join(data_dir, "db.json"), encoding="utf-8"))
    policy = open(os.path.join(data_dir, "policy.md"), encoding="utf-8").read()
    return db, policy


def pick(db: dict, status: str, n: int) -> List[Tuple[dict, dict]]:
    """The first n (user, order) pairs of the database, in file order, whose order has this status and whose payment history is not empty."""
    found = []
    for u in db["users"].values():
        for oid in u["orders"]:
            o = db["orders"].get(oid)
            if o and o["status"] == status and o["items"] and o["payment_history"] and u["user_id"] == o["user_id"]:
                found.append((u, o))
                break
        if len(found) == n:
            break
    return found


def run(data_dir: str, per_kind: int = 3) -> dict:
    db, policy = load(data_dir)
    plan = [("cancel", "pending"), ("return", "delivered"), ("exchange", "delivered")]
    totals = {k: [0, 0, 0, 0] for k in ("chat", "A", "A2")}  # kept chars, chars of prompts 2..n, chars of all prompts, calls
    rows = []
    for kind, status in plan:
        for u, o in pick(db, status, per_kind):
            lay = layouts_of(policy, scenario(kind, u, o, db["products"]))
            row = {"kind": kind, "order": o["order_id"], "calls": len(lay["A"])}
            for k, prompts in lay.items():
                kept, rest = reuse(prompts)
                row[k] = {"share": kept / rest, "mean_chars": sum(map(len, prompts)) / len(prompts)}
                t = totals[k]
                t[0] += kept
                t[1] += rest
                t[2] += sum(map(len, prompts))
                t[3] += len(prompts)
            rows.append(row)
    summary = {k: {"share_repeating_the_previous_prompt": t[0] / t[1], "mean_prompt_chars": t[2] / t[3], "calls": t[3]} for k, t in totals.items()}
    return {"policy_chars": len(policy), "conversations": len(rows), "rows": rows, "summary": summary}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    default = os.path.join(os.environ.get("TAU2_SRC", ""), "..", "data", "tau2", "domains", "retail") if os.environ.get("TAU2_SRC") else None
    ap.add_argument("--data", default=default, help="the benchmark's data/tau2/domains/retail folder (db.json and policy.md)")
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--json", default=None, help="write the result to this file")
    a = ap.parse_args(argv)
    if not a.data:
        raise SystemExit("give --data <tau2-bench>/data/tau2/domains/retail (or set TAU2_SRC to <clone>/src)")
    res = run(a.data, a.per_kind)
    print(f"policy {res['policy_chars']:,} characters; {res['conversations']} scripted conversations from the benchmark's own database\n")
    print(f"{'conversation':<22}{'calls':>6}  " + "".join(f"{k + ' repeats':>16}{k + ' chars':>10}" for k in ("chat", "A", "A2")))
    for r in res["rows"]:
        print(f"{r['kind'] + ' ' + r['order']:<22}{r['calls']:>6}  " + "".join(f"{100 * r[k]['share']:>15.0f}%{r[k]['mean_chars']:>10,.0f}" for k in ("chat", "A", "A2")))
    print("\nall conversations (share of each prompt, from the second call on, that repeats the start of the previous prompt):")
    for k, label in (("chat", "the benchmark's chat (B, G)"), ("A", "held state as run (A)"), ("A2", "held state, dialogue first (A2)")):
        s = res["summary"][k]
        print(f"  {label:<34}{100 * s['share_repeating_the_previous_prompt']:5.1f} %   mean prompt {s['mean_prompt_chars']:,.0f} characters")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

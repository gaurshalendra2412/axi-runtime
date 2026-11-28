"""Offline checks of the layout-reuse script (python/axi/experiments/tau2_layout_reuse.py) on a made-up two-user database. No model, no tau2 import."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi.experiments import tau2_layout_reuse as L  # noqa: E402


def make_db():
    addr = {"address1": "1 A St", "address2": "", "city": "X", "country": "USA", "state": "CO", "zip": "80279"}
    item = {"name": "Mug", "product_id": "p1", "item_id": "i1", "price": 5.0, "options": {"c": "red"}}
    pay = [{"transaction_type": "payment", "amount": 5.0, "payment_method_id": "paypal_1"}]
    users, orders = {}, {}
    for n, status in ((1, "pending"), (2, "delivered")):
        uid, oid = f"user_{n}", f"#W{n}"
        users[uid] = {"user_id": uid, "email": f"u{n}@example.com", "payment_methods": {"paypal_1": {"source": "paypal", "id": "paypal_1"}}, "orders": [oid]}
        orders[oid] = {"order_id": oid, "user_id": uid, "address": addr, "items": [item], "status": status, "fulfillments": [], "payment_history": pay}
    return {"users": users, "orders": orders, "products": {"p1": {"product_id": "p1", "variants": {"i1": {"item_id": "i1"}}}}}


def test_common_prefix_and_reuse():
    assert L.common_prefix("abcd", "abxy") == 2 and L.common_prefix("", "a") == 0 and L.common_prefix("ab", "ab") == 2
    assert L.reuse(["aaaa", "aabb", "aabc"]) == (2 + 3, 4 + 4)


def test_the_three_layouts_hold_the_same_conversation_and_the_plain_chat_is_append_only():
    db = make_db()
    u, o = L.pick(db, "pending", 1)[0]
    lay = L.layouts_of("POLICY " * 40, L.scenario("cancel", u, o, db["products"]))
    assert len(lay["chat"]) == len(lay["A"]) == len(lay["A2"]) == 9
    for i in range(len(lay["A"])):  # the same pieces of text in A and A2 (the two notes differ only in the order of their lines)
        a = [x for x in lay["A"][i].split("\n") if not x.startswith("- ")]
        a2 = [x for x in lay["A2"][i].split("\n") if not x.startswith("- ")]
        assert sorted(a) == sorted(a2)
    for k in ("chat", "A", "A2"):
        kept, rest = L.reuse(lay[k])
        assert 0 < kept < rest
    kept, rest = L.reuse(lay["chat"])
    assert kept / rest > 0.8  # each prompt of the chat contains the whole previous one


def test_run_on_a_made_up_database():
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "axi_layout_reuse_test")
    os.makedirs(d, exist_ok=True)
    json.dump(make_db(), open(os.path.join(d, "db.json"), "w"))
    open(os.path.join(d, "policy.md"), "w").write("POLICY " * 100)
    res = L.run(d, per_kind=1)
    assert res["conversations"] == 3 and set(res["summary"]) == {"chat", "A", "A2"}
    assert all(0 < v["share_repeating_the_previous_prompt"] < 1 for v in res["summary"].values())


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")

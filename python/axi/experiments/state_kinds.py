"""Test 45 (docs/STATE_KINDS_TEST.md): four other kinds of state, with requests that would break a rule of that state, and four ways of applying the model's reply.

A NEW file; it edits no earlier file and uses the harness of Tests 35 to 44 for the engine, the lockstep rounds and the numbers. The framework's own gate (`axi.engine.gate`) is the gate of a service graph; the four worlds here
are small models of other states, each with a held state, an oracle, a strict reply format, and two checks:
  generic check  : what needs no knowledge of the domain (the reply parses; the ids, lines or cells it names exist; every reference in the state points at something; text parses; values are in the format's bounds)
  domain check   : the rule of the domain (T: no formula depends on itself; S: no room and no person is booked twice at the same time; C: no name is used before it is defined; J: every call matches its tool's schema)
Worlds: T a table with formulas (A1 to D4), S a room schedule, C a small Python module edited by line patches, J a list of JSON tool calls.
Each request is either valid (the oracle's reply applies it) or conflicting (applying it as written would break a rule: the oracle's reply is NOOP and the state does not change). The text of a conflicting request does
not say it conflicts.
Arms (the model sees the state the host holds, at every step, as in the exact-state arm of the service-graph tests):
  B  the reply is applied as written (parts that cannot be applied are skipped), no rule in the prompt
  P  as B, with the rules and NOOP in the prompt
  G  the generic check: a reply that fails it is dropped (the state does not change); no rule in the prompt
  D  the generic check and the domain check: a reply that fails either is dropped; no rule in the prompt
`TypedStandIn` doubles test the harness only. Never a result."""
import ast, builtins, hashlib, json, random, re
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_typed as T

SEED = PS.SEED
WORLDS = ("T", "S", "C", "J")
ARMS = ("B", "P", "G", "D")
CONFLICT_SHARE = 0.30
GENERIC_SHARE = 0.34          # of the conflicting requests, the share whose conflict the generic check can see
MAX_NEW = 96


# ================================================================ T: a table with formulas
T_ADDRS = [c + r for r in "1234" for c in "ABCD"]
T_RX = r"(?:NOOP|SET [A-D][1-4] [0-9]|SETF [A-D][1-4] =[A-D][1-4][+-][A-D][1-4]|CLR [A-D][1-4])"
T_SYSTEM = """You maintain a small spreadsheet with cells A1 to D4. A cell holds a number from 0 to 9, or a formula of two cells. The table is written as cell=number, and a formula as cell==X+Y.
Reply with ONLY one operation and no explanation:
  SET C3 7          put the number 7 in C3
  SETF C3 =A1+B2    make C3 the formula A1+B2 (two cells, + or -)
  CLR C3            empty C3
Examples:
  put 4 in B2 -> SET B2 4
  make D1 the sum of A1 and C1 -> SETF D1 =A1+C1"""
T_RULES = """
Rules: a formula may only use cells that hold something. No cell may depend on itself, directly or through other cells. A cell that some formula uses cannot be emptied.
If a request would break a rule, reply NOOP and nothing changes."""


def t_render(st):
    cell = lambda a: f"{a}={st[a][1]}" if st[a][0] == "v" else f"{a}=={st[a][1]}{st[a][2]}{st[a][3]}"
    return "<TABLE>\n" + " ".join(cell(a) for a in T_ADDRS if a in st) + "\n</TABLE>"


def t_parse(text):
    if text == "NOOP": return []
    m = re.fullmatch(r"SET ([A-D][1-4]) ([0-9])", text)
    if m: return [("set", m.group(1), int(m.group(2)))]
    m = re.fullmatch(r"SETF ([A-D][1-4]) =([A-D][1-4])([+-])([A-D][1-4])", text)
    if m: return [("setf", m.group(1), m.group(2), m.group(3), m.group(4))]
    m = re.fullmatch(r"CLR ([A-D][1-4])", text)
    if m: return [("clr", m.group(1))]
    return None


def t_apply(st, ops):
    h = dict(st)
    for o in ops:
        if o[0] == "set": h[o[1]] = ("v", o[2])
        elif o[0] == "setf": h[o[1]] = ("f", o[2], o[3], o[4])
        elif o[0] == "clr": h.pop(o[1], None)
    return h


def t_generic(before, ops, after):
    return all(v[0] == "v" or (v[1] in after and v[3] in after) for v in after.values())


def t_domain(after):
    deps = {a: (v[1], v[3]) for a, v in after.items() if v[0] == "f"}; color: Dict[str, int] = {}
    def visit(a):
        color[a] = 1
        for b in deps.get(a, ()):
            if color.get(b) == 1 or (color.get(b) is None and b in deps and visit(b)): return True
        color[a] = 2; return False
    return not any(color.get(a) is None and visit(a) for a in deps)


def t_request(rng, st, conflict, ctx):
    empty = [a for a in T_ADDRS if a not in st]; full = [a for a in T_ADDRS if a in st]
    used = {x for v in st.values() if v[0] == "f" for x in (v[1], v[3])}
    forms = [a for a in full if st[a][0] == "f"]
    def formula(a, x, op, y):
        return (f"Make {a} the sum of {x} and {y}." if op == "+" else f"Make {a} equal to {x} minus {y}."), f"SETF {a} ={x}{op}{y}"
    if conflict:
        generic = rng.random() < GENERIC_SHARE
        if generic:
            opts = []
            if empty and len(full) >= 1: opts.append("ref")
            if used: opts.append("clear")
            if opts:
                k = rng.choice(opts)
                if k == "ref":
                    e = rng.choice(empty); x = rng.choice(full); a = rng.choice([c for c in T_ADDRS if c not in (x, e)]); ins, lit = formula(a, x, rng.choice("+-"), e)
                    return dict(instruction=ins, ideal="NOOP", literal=lit, kind="dangling_ref", conflict=True, layer="generic")
                a = rng.choice(sorted(used)); return dict(instruction=f"Empty cell {a}.", ideal="NOOP", literal=f"CLR {a}", kind="dangling_clear", conflict=True, layer="generic")
        else:
            if forms and rng.random() < 0.6:
                f = rng.choice(forms); x = st[f][1]; z = rng.choice(full); ins, lit = formula(x, f, rng.choice("+-"), z)
                return dict(instruction=ins, ideal="NOOP", literal=lit, kind="cycle", conflict=True, layer="domain")
            if full:
                a = rng.choice(full); z = rng.choice(full); ins, lit = formula(a, a, rng.choice("+-"), z)
                return dict(instruction=ins, ideal="NOOP", literal=lit, kind="cycle", conflict=True, layer="domain")
    for _ in range(40):
        k = rng.choices(("set_value", "set_formula", "clear"), (0.4, 0.4, 0.2))[0]
        if k == "set_value":
            a = rng.choice(T_ADDRS); n = rng.randrange(10); return dict(instruction=f"Put the number {n} in cell {a}.", ideal=f"SET {a} {n}", literal=f"SET {a} {n}", kind=k, conflict=False, layer=None)
        if k == "clear":
            c = [a for a in full if a not in used]
            if c: a = rng.choice(c); return dict(instruction=f"Empty cell {a}.", ideal=f"CLR {a}", literal=f"CLR {a}", kind=k, conflict=False, layer=None)
        elif len(full) >= 2:
            a = rng.choice(T_ADDRS); x, y = rng.choice(full), rng.choice(full); op = rng.choice("+-"); h = t_apply(st, [("setf", a, x, op, y)])
            if t_generic(st, None, h) and t_domain(h):
                ins, lit = formula(a, x, op, y); return dict(instruction=ins, ideal=lit, literal=lit, kind=k, conflict=False, layer=None)
    a = rng.choice(T_ADDRS); return dict(instruction=f"Put the number 1 in cell {a}.", ideal=f"SET {a} 1", literal=f"SET {a} 1", kind="set_value", conflict=False, layer=None)


def t_initial(rng):
    st = {a: ("v", rng.randrange(10)) for a in rng.sample(T_ADDRS, 8)}
    for a in rng.sample([a for a in T_ADDRS if a not in st], 3):
        x, y = rng.sample(sorted(st), 2); h = t_apply(st, [("setf", a, x, rng.choice("+-"), y)])
        if t_generic(None, None, h) and t_domain(h): st = h
    return st


# ================================================================ S: a room schedule
S_ROOMS, S_OWNERS, S_H0, S_H1 = ("R1", "R2", "R3"), ("alice", "bob", "cara"), 8, 18
S_RX = r"(?:NOOP|ADD e[0-9]+ R[1-3] [0-9]{1,2} [0-9]{1,2} (?:alice|bob|cara)|MOVE e[0-9]+ R[1-3] [0-9]{1,2} [0-9]{1,2}|CANCEL e[0-9]+)"
S_SYSTEM = """You maintain a room schedule. Each event is written as: id room start-end owner (hours of the working day, 8 to 18).
Reply with ONLY one operation and no explanation:
  ADD e7 R2 10 12 bob     book event e7 in room R2 from 10 to 12 for bob
  MOVE e3 R1 14 16        move event e3 to room R1, 14 to 16 (the owner stays)
  CANCEL e4               remove event e4
Examples:
  cancel event e2 -> CANCEL e2
  move event e5 to room R3 from 9 to 10 -> MOVE e5 R3 9 10"""
S_RULES = """
Rules: an event starts and ends inside 8 to 18, and starts before it ends. An id is used once. A room holds one event at a time, and a person is in one event at a time (two events overlap if one starts before the other ends).
If a request would break a rule, reply NOOP and nothing changes."""


def s_render(st): return "<SCHEDULE>\n" + "\n".join(f"{e} {r} {s}-{t} {o}" for e, (r, s, t, o) in sorted(st.items(), key=lambda kv: int(kv[0][1:]))) + "\n</SCHEDULE>"


def s_parse(text):
    if text == "NOOP": return []
    m = re.fullmatch(r"ADD (e\d+) (R[1-3]) (\d+) (\d+) (alice|bob|cara)", text)
    if m: return [("add", m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5))]
    m = re.fullmatch(r"MOVE (e\d+) (R[1-3]) (\d+) (\d+)", text)
    if m: return [("move", m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))]
    m = re.fullmatch(r"CANCEL (e\d+)", text)
    if m: return [("cancel", m.group(1))]
    return None


def s_apply(st, ops):
    h = dict(st)
    for o in ops:
        if o[0] == "add": h[o[1]] = (o[2], o[3], o[4], o[5])
        elif o[0] == "move" and o[1] in h: h[o[1]] = (o[2], o[3], o[4], h[o[1]][3])
        elif o[0] == "cancel": h.pop(o[1], None)
    return h


def s_generic(before, ops, after):
    run = dict(before)
    for o in ops:
        if o[0] == "add" and o[1] in run: return False
        if o[0] in ("move", "cancel") and o[1] not in run: return False
        run = s_apply(run, [o])
    return all(S_H0 <= s < t <= S_H1 for (_, s, t, _) in after.values())


def s_domain(after):
    ev = list(after.values())
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            (r1, s1, t1, o1), (r2, s2, t2, o2) = ev[i], ev[j]
            if (r1 == r2 or o1 == o2) and s1 < t2 and s2 < t1: return False
    return True


def s_request(rng, st, conflict, ctx):
    nid = ctx.setdefault("next_id", max([int(e[1:]) for e in st] + [0]) + 1)
    def add_text(e, r, s, t, o): return f"Book room {r} from {s} to {t} for {o} as event {e}.", f"ADD {e} {r} {s} {t} {o}"
    def move_text(e, r, s, t): return f"Move event {e} to room {r}, {s} to {t}.", f"MOVE {e} {r} {s} {t}"
    ev = sorted(st.items(), key=lambda kv: int(kv[0][1:]))
    if conflict:
        if rng.random() < GENERIC_SHARE:
            k = rng.choice(("late", "missing", "dup"))
            if k == "late": r, o = rng.choice(S_ROOMS), rng.choice(S_OWNERS); ins, lit = add_text(f"e{nid}", r, 17, 20, o); ctx["next_id"] = nid; return dict(instruction=ins, ideal="NOOP", literal=lit, kind="out_of_hours", conflict=True, layer="generic")
            if k == "missing":
                ins, lit = move_text("e99", rng.choice(S_ROOMS), 9, 10); return dict(instruction=ins, ideal="NOOP", literal=lit, kind="missing_id", conflict=True, layer="generic")
            if ev:
                e = rng.choice(ev)[0]; ins, lit = add_text(e, rng.choice(S_ROOMS), 9, 10, rng.choice(S_OWNERS)); return dict(instruction=ins, ideal="NOOP", literal=lit, kind="duplicate_id", conflict=True, layer="generic")
        elif ev:
            e, (r, s, t, o) = rng.choice(ev); s2 = rng.randrange(s, t); t2 = min(S_H1, s2 + rng.randrange(1, 3))                  # starts inside the event, so the two overlap
            if rng.random() < 0.5:   # same room, a different person
                o2 = rng.choice([x for x in S_OWNERS if x != o]); ins, lit = add_text(f"e{nid}", r, s2, t2, o2); ctx["next_id"] = nid; return dict(instruction=ins, ideal="NOOP", literal=lit, kind="room_overlap", conflict=True, layer="domain")
            r2 = rng.choice([x for x in S_ROOMS if x != r]); ins, lit = add_text(f"e{nid}", r2, s2, t2, o); ctx["next_id"] = nid
            return dict(instruction=ins, ideal="NOOP", literal=lit, kind="person_overlap", conflict=True, layer="domain")
    for _ in range(60):
        k = rng.choices(("add", "move", "cancel"), (0.45, 0.35, 0.2))[0]
        if k == "cancel" and ev: e = rng.choice(ev)[0]; return dict(instruction=f"Cancel event {e}.", ideal=f"CANCEL {e}", literal=f"CANCEL {e}", kind=k, conflict=False, layer=None)
        r, o = rng.choice(S_ROOMS), rng.choice(S_OWNERS); d = rng.randrange(1, 4); s = rng.randrange(S_H0, S_H1 - d + 1); t = s + d
        if k == "add":
            h = s_apply(st, [("add", f"e{nid}", r, s, t, o)])
            if s_domain(h): ins, lit = add_text(f"e{nid}", r, s, t, o); ctx["next_id"] = nid + 1; return dict(instruction=ins, ideal=lit, literal=lit, kind=k, conflict=False, layer=None)
        elif k == "move" and ev:
            e = rng.choice(ev)[0]; h = s_apply(st, [("move", e, r, s, t)])
            if s_domain(h): ins, lit = move_text(e, r, s, t); return dict(instruction=ins, ideal=lit, literal=lit, kind=k, conflict=False, layer=None)
    e = ev[0][0]; return dict(instruction=f"Cancel event {e}.", ideal=f"CANCEL {e}", literal=f"CANCEL {e}", kind="cancel", conflict=False, layer=None)


def s_initial(rng):
    st, ctx = {}, {}
    for i in range(1, 7):
        for _ in range(40):
            r, o = rng.choice(S_ROOMS), rng.choice(S_OWNERS); d = rng.randrange(1, 4); s = rng.randrange(S_H0, S_H1 - d + 1); h = dict(st); h[f"e{i}"] = (r, s, s + d, o)
            if s_domain(h): st = h; break
    return st


# ================================================================ C: a small Python module edited by line patches
C_TEXT = r"[ ]{0,8}[A-Za-z_][A-Za-z0-9_ =+*(),-]{0,30}"
C_RX = r"(?:NOOP|DEL [0-9]{1,2}|(?:INS|REP) [0-9]{1,2} " + C_TEXT + ")"
C_SYSTEM = """You maintain a small Python file, shown with a number before each line. Reply with ONLY one patch operation and no explanation:
  INS 4 <line>     insert the line (with its indentation) so that it becomes line 4
  REP 3 <line>     replace line 3 with the line
  DEL 5            delete line 5
Examples:
  insert `    z = y + 1` as line 6 -> INS 6     z = y + 1
  delete line 4 -> DEL 4"""
C_RULES = """
Rules: the file must stay valid Python, and every name a line uses must already be a parameter or a variable assigned on an earlier line of the same function, or a function of the file.
If a request would break a rule, reply NOOP and nothing changes."""
C_NAMES, C_PARAMS = ("s", "t", "u", "v", "w", "z"), ("a", "b", "x", "y")


def c_render(st): return "<FILE>\n" + "\n".join(f"{i} {l}" for i, l in enumerate(st, start=1)) + "\n</FILE>"


def c_parse(text):
    if text == "NOOP": return []
    m = re.fullmatch(r"DEL (\d+)", text)
    if m: return [("del", int(m.group(1)))]
    m = re.fullmatch(r"(INS|REP) (\d+) (.*)", text, flags=re.S)
    if m: return [(m.group(1).lower(), int(m.group(2)), m.group(3))]
    return None


def c_apply(st, ops):
    h = list(st)
    for o in ops:
        if o[0] == "ins" and 1 <= o[1] <= len(h) + 1: h.insert(o[1] - 1, o[2])
        elif o[0] == "rep" and 1 <= o[1] <= len(h): h[o[1] - 1] = o[2]
        elif o[0] == "del" and 1 <= o[1] <= len(h): h.pop(o[1] - 1)
    return h


def c_generic(before, ops, after):
    n = len(before)
    for o in ops:
        if o[0] == "ins" and not 1 <= o[1] <= n + 1: return False
        if o[0] in ("rep", "del") and not 1 <= o[1] <= n: return False
        n += 1 if o[0] == "ins" else (-1 if o[0] == "del" else 0)
    try: ast.parse("\n".join(after))
    except (SyntaxError, ValueError): return False
    return True


def c_domain(after):
    try: tree = ast.parse("\n".join(after))
    except (SyntaxError, ValueError): return False
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}; ok_builtin = set(dir(builtins))
    def loads(n): return {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    def stores(n): return {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    mod = set(funcs)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            bound = {a.arg for a in n.args.args}
            for s in n.body:
                if loads(s) - bound - mod - ok_builtin: return False
                bound |= stores(s)
        else:
            if loads(n) - mod - ok_builtin: return False
            mod |= stores(n)
    return True


def c_initial(rng):
    lines = []
    for f in range(1, 3):
        ps = rng.sample(C_PARAMS, rng.randrange(1, 3)); lines.append(f"def f{f}({', '.join(ps)}):"); bound = list(ps); names = rng.sample(C_NAMES, 3)
        for _ in range(rng.randrange(1, 3)):
            v = names.pop(); lines.append(f"    {v} = {rng.choice(bound)} {rng.choice('+-*')} {rng.choice(bound + [str(rng.randrange(1, 9))])}"); bound.append(v)
        lines.append(f"    return {bound[-1]}")
    return lines


def c_request(rng, st, conflict, ctx):
    fn = []                                                       # (def line index, last line index) per function, 0-based
    for i, l in enumerate(st):
        if l.startswith("def "): fn.append([i, i])
        elif fn: fn[-1][1] = i
    def bound_before(idx):
        for a, b in fn:
            if a < idx <= b + 1 or a == idx:
                names = [x.strip() for x in re.match(r"def \w+\((.*)\):", st[a]).group(1).split(",")]
                for j in range(a + 1, min(idx, len(st))):
                    m = re.match(r"    (\w+) =", st[j])
                    if m: names.append(m.group(1))
                return names
        return ["a"]
    def stmt(idx):
        b = bound_before(idx); return f"    {rng.choice(C_NAMES)} = {rng.choice(b)} {rng.choice('+-*')} {rng.choice(b + [str(rng.randrange(1, 9))])}"
    def show(op, n, text=None):
        if op == "del": return f"Delete line {n}.", f"DEL {n}"
        return (f"Insert the line `{text}` as line {n}." if op == "ins" else f"Replace line {n} with `{text}`."), f"{op.upper()} {n} {text}"
    body = [i for i, l in enumerate(st) if l.startswith("    ") and not l.startswith("    return")]
    if conflict:
        if rng.random() < GENERIC_SHARE:
            if body and rng.random() < 0.5:
                i = rng.choice(body); v = st[i].split("=")[0].strip(); ins, lit = show("rep", i + 1, f"    {v} = {rng.choice(bound_before(i))} +")
                return dict(instruction=ins, ideal="NOOP", literal=lit, kind="syntax", conflict=True, layer="generic")
            n = rng.randrange(2, len(st) + 1); ins, lit = show("ins", n, "  z = 1"); return dict(instruction=ins, ideal="NOOP", literal=lit, kind="indent", conflict=True, layer="generic")
        else:
            used = [i for i in body if c_generic(st, [("del", i + 1)], c_apply(st, [("del", i + 1)])) and not c_domain(c_apply(st, [("del", i + 1)]))]            # deleting this line leaves a name undefined
            if used and rng.random() < 0.5:
                i = rng.choice(used); ins, lit = show("del", i + 1); return dict(instruction=ins, ideal="NOOP", literal=lit, kind="delete_definition", conflict=True, layer="domain")
            if body:
                i = rng.choice(body); v = st[i].split("=")[0].strip(); ins, lit = show("rep", i + 1, f"    {v} = {rng.choice(bound_before(i))} + q"); return dict(instruction=ins, ideal="NOOP", literal=lit, kind="undefined_name", conflict=True, layer="domain")
    for _ in range(60):
        k = rng.choices(("ins", "rep", "del"), (0.4, 0.35, 0.25))[0]
        if k == "ins":
            cand = [i for i, l in enumerate(st) if l.startswith("    return")]; i = rng.choice(cand); text = stmt(i)           # before a return line
            op = ("ins", i + 1, text)
        elif k == "rep" and body:
            i = rng.choice(body); v = st[i].split("=")[0].strip(); b = bound_before(i); op = ("rep", i + 1, f"    {v} = {rng.choice(b)} {rng.choice('+-*')} {rng.choice(b + [str(rng.randrange(1, 9))])}")
        elif k == "del" and body and len(st) > 8:
            i = rng.choice(body); op = ("del", i + 1, None)
        else: continue
        h = c_apply(st, [(op[0], op[1], op[2])] if op[0] != "del" else [("del", op[1])])
        if c_generic(st, [(op[0], op[1], op[2])] if op[0] != "del" else [("del", op[1])], h) and c_domain(h) and len(h) <= 16:
            ins, lit = show(op[0], op[1], op[2]); return dict(instruction=ins, ideal=lit, literal=lit, kind={"ins": "insert", "rep": "replace", "del": "delete"}[op[0]], conflict=False, layer=None)
    i = [i for i, l in enumerate(st) if l.startswith("    return")][0]; v = bound_before(i)[0]; text = f"    z = {v} + 1"
    return dict(instruction=f"Insert the line `{text}` as line {i + 1}.", ideal=f"INS {i + 1} {text}", literal=f"INS {i + 1} {text}", kind="insert", conflict=False, layer=None)


# ================================================================ J: JSON tool calls
J_TOOLS = {"search": {"query": ("str",), "limit": ("int", 1, 20)}, "email": {"to": ("str",), "subject": ("str",), "urgent": ("bool",)},
           "book": {"room": ("enum", ("R1", "R2", "R3")), "hours": ("int", 1, 4)}, "timer": {"minutes": ("int", 1, 120)}}
J_STR = {"query": ("cats", "tax", "maps", "news"), "to": ("ann", "raj", "lee"), "subject": ("hi", "lunch", "plan")}
J_RX = (r'(?:NOOP|ADDCALL c[0-9]+ \{"tool": ?"[a-z]+", ?"args": ?\{[^{}]{0,80}\}\}|SETARG c[0-9]+ [a-z]+ (?:[0-9]{1,3}|true|false|"[A-Za-z0-9]{1,12}")|DELCALL c[0-9]+)')
J_SYSTEM = """You maintain a list of tool calls. Each call is written as: id {"tool": name, "args": {...}}. Tools and their arguments:
  search(query, limit)   email(to, subject, urgent)   book(room, hours)   timer(minutes)
Reply with ONLY one operation and no explanation:
  ADDCALL c5 {"tool": "timer", "args": {"minutes": 15}}     add call c5
  SETARG c1 limit 10                                          set an argument of c1 (a number, true or false, or a "string")
  DELCALL c3                                                  remove call c3
Examples:
  set the minutes of c2 to 30 -> SETARG c2 minutes 30
  remove call c4 -> DELCALL c4"""
J_RULES = """
Rules: every call has exactly the arguments of its tool, each of the right type. query, to and subject are strings; limit is a whole number from 1 to 20; urgent is true or false; room is "R1", "R2" or "R3"; hours is a whole number from 1 to 4; minutes is a whole number from 1 to 120.
Only a call that exists can be changed or removed, and a call id is used once.
If a request would break a rule, reply NOOP and nothing changes."""


def j_render(st): return "<CALLS>\n" + "\n".join(f'{c} {json.dumps({"tool": t, "args": a}, separators=(", ", ": "))}' for c, (t, a) in sorted(st.items(), key=lambda kv: int(kv[0][1:]))) + "\n</CALLS>"


def j_parse(text):
    if text == "NOOP": return []
    m = re.fullmatch(r"DELCALL (c\d+)", text)
    if m: return [("delcall", m.group(1))]
    m = re.fullmatch(r"SETARG (c\d+) ([a-z]+) (.+)", text)
    if m:
        try: return [("setarg", m.group(1), m.group(2), json.loads(m.group(3)))]
        except ValueError: return None
    m = re.fullmatch(r"ADDCALL (c\d+) (\{.*\})", text)
    if m:
        try: d = json.loads(m.group(2))
        except ValueError: return None
        if isinstance(d, dict) and isinstance(d.get("tool"), str) and isinstance(d.get("args"), dict): return [("addcall", m.group(1), d["tool"], d["args"])]
    return None


def j_apply(st, ops):
    h = {k: (t, dict(a)) for k, (t, a) in st.items()}
    for o in ops:
        if o[0] == "addcall": h[o[1]] = (o[2], dict(o[3]))
        elif o[0] == "setarg" and o[1] in h: h[o[1]][1][o[2]] = o[3]
        elif o[0] == "delcall": h.pop(o[1], None)
    return h


def j_generic(before, ops, after):
    run = {k: (t, dict(a)) for k, (t, a) in before.items()}
    for o in ops:
        if o[0] == "addcall" and (o[1] in run or o[2] not in J_TOOLS): return False
        if o[0] == "setarg" and (o[1] not in run or o[2] not in J_TOOLS[run[o[1]][0]]): return False
        if o[0] == "delcall" and o[1] not in run: return False
        run = j_apply(run, [o])
    return True


def _j_type_ok(spec, v):
    if spec[0] == "str": return isinstance(v, str)
    if spec[0] == "bool": return isinstance(v, bool)
    if spec[0] == "int": return isinstance(v, int) and not isinstance(v, bool) and spec[1] <= v <= spec[2]
    return isinstance(v, str) and v in spec[1]


def j_domain(after):
    for t, a in after.values():
        sch = J_TOOLS.get(t)
        if sch is None or set(a) != set(sch) or not all(_j_type_ok(sch[k], a[k]) for k in sch): return False
    return True


def _j_value(rng, p, spec):
    if spec[0] == "str": return rng.choice(J_STR[p])
    if spec[0] == "bool": return rng.random() < 0.5
    if spec[0] == "int": return rng.randrange(spec[1], spec[2] + 1)
    return rng.choice(spec[1])


def _j_lit(v): return json.dumps(v)


def _j_words(p, v): return f'Set the {p} of {{c}} to {json.dumps(v) if isinstance(v, str) else str(v).lower()}.'


def j_initial(rng):
    st = {}
    for i, t in enumerate(rng.sample(sorted(J_TOOLS), 3), start=1): st[f"c{i}"] = (t, {p: _j_value(rng, p, s) for p, s in J_TOOLS[t].items()})
    return st


def j_request(rng, st, conflict, ctx):
    nid = ctx.setdefault("next_id", max([int(c[1:]) for c in st] + [0]) + 1); cs = sorted(st, key=lambda c: int(c[1:]))
    def setarg(c, p, v, kind, conflict_, layer):
        s = _j_words(p, v).replace("{c}", c); return dict(instruction=s, ideal="NOOP" if conflict_ else f"SETARG {c} {p} {_j_lit(v)}", literal=f"SETARG {c} {p} {_j_lit(v)}", kind=kind, conflict=conflict_, layer=layer)
    if conflict and cs:
        if rng.random() < GENERIC_SHARE:
            if rng.random() < 0.5: return setarg("c99", "limit", 3, "missing_id", True, "generic")
            c = rng.choice(cs); return setarg(c, "colour", "red", "unknown_arg", True, "generic")
        c = rng.choice(cs); t, a = st[c]; p = rng.choice(sorted(a)); spec = J_TOOLS[t][p]
        if spec[0] == "int": v = spec[2] + rng.randrange(5, 60)
        elif spec[0] == "enum": v = "R9"
        elif spec[0] == "bool": v = rng.randrange(2, 9)
        else: v = rng.randrange(2, 99)
        return setarg(c, p, v, "bad_value", True, "domain")
    for _ in range(40):
        k = rng.choices(("set_arg", "add_call", "del_call"), (0.5, 0.3, 0.2))[0]
        if k == "set_arg" and cs:
            c = rng.choice(cs); t, a = st[c]; p = rng.choice(sorted(a)); return setarg(c, p, _j_value(rng, p, J_TOOLS[t][p]), k, False, None)
        if k == "del_call" and len(cs) > 2: c = rng.choice(cs); return dict(instruction=f"Remove call {c}.", ideal=f"DELCALL {c}", literal=f"DELCALL {c}", kind=k, conflict=False, layer=None)
        if k == "add_call" and len(cs) < 7:
            t = rng.choice(sorted(J_TOOLS)); args = {p: _j_value(rng, p, s) for p, s in J_TOOLS[t].items()}; c = f"c{nid}"; ctx["next_id"] = nid + 1
            words = " and ".join(f'{p} {json.dumps(v) if isinstance(v, str) else str(v).lower()}' for p, v in args.items())
            return dict(instruction=f"Add a {t} call {c} with {words}.", ideal=f'ADDCALL {c} {json.dumps({"tool": t, "args": args}, separators=(", ", ": "))}', literal=f'ADDCALL {c} {json.dumps({"tool": t, "args": args}, separators=(", ", ": "))}', kind=k, conflict=False, layer=None)
    c = cs[0]; t, a = st[c]; p = sorted(a)[0]; return setarg(c, p, _j_value(rng, p, J_TOOLS[t][p]), "set_arg", False, None)


# ================================================================ the table of worlds, the oracle, the policies
class World:
    def __init__(self, name, system, rules, rx, render, parse, apply, generic, domain, request, initial):
        self.name, self.system, self.rules, self.rx = name, system, rules, rx
        self.render, self.parse, self.apply, self.generic, self.domain, self.request, self.initial = render, parse, apply, generic, domain, request, initial


W = {"T": World("T", T_SYSTEM, T_RULES, T_RX, t_render, t_parse, t_apply, t_generic, t_domain, t_request, t_initial),
     "S": World("S", S_SYSTEM, S_RULES, S_RX, s_render, s_parse, s_apply, s_generic, s_domain, s_request, s_initial),
     "C": World("C", C_SYSTEM, C_RULES, C_RX, c_render, c_parse, c_apply, c_generic, c_domain, c_request, c_initial),
     "J": World("J", J_SYSTEM, J_RULES, J_RX, j_render, j_parse, j_apply, j_generic, j_domain, j_request, j_initial)}


def valid_state(w: World, st) -> bool:
    return w.generic(st, [], st) and w.domain(st)


def policy_next(w: World, arm: str, st, text: str):
    """The state the host holds after a reply under an arm. Returns (state, parsed, admitted)."""
    ops = w.parse(text)
    if ops is None: return st, False, False
    after = w.apply(st, ops)
    if arm in ("B", "P"): return after, True, True
    if not w.generic(st, ops, after): return st, True, False
    if arm == "D" and not w.domain(after): return st, True, False
    return after, True, True


def script_of(world: str, seed: int, steps: int):
    """The requests of a job and the oracle's states (index 0 is the start). Deterministic in (world, seed). The oracle's state is valid at every step."""
    w = W[world]; rng = random.Random(f"{world}:{seed}"); st = w.initial(rng); assert valid_state(w, st); oracle = [st]; items = []; ctx: dict = {}
    for _ in range(steps):
        it = w.request(rng, st, rng.random() < CONFLICT_SHARE, ctx); ops = w.parse(it["ideal"]); assert ops is not None, it
        lit = w.parse(it["literal"]); assert lit is not None, it
        assert re.fullmatch(w.rx, it["ideal"]) and re.fullmatch(w.rx, it["literal"]), it
        st = w.apply(st, ops); assert valid_state(w, st), (it, st)
        if it["conflict"]:
            after = w.apply(oracle[-1], lit); assert not (w.generic(oracle[-1], lit, after) and w.domain(after)), ("a conflicting request that does not conflict", it)
            if it["layer"] == "generic": assert not w.generic(oracle[-1], lit, after), ("a generic conflict the generic check does not see", it)
            else: assert w.generic(oracle[-1], lit, after) and not w.domain(after), ("a domain conflict the generic check already sees", it)
        else: assert w.generic(oracle[-1], ops, st) and w.domain(st), it
        items.append(it); oracle.append(st)
    return items, oracle


def equal(a, b) -> bool: return a == b


# ================================================================ the sessions
def system_for(w: World, arm: str) -> str: return w.system + (w.rules if arm == "P" else "")


class StateSession:
    """One job of a world under one arm: the model sees the state the host holds and the request; the host applies the reply by the arm's policy. Same protocol as `provider_side.Session` (request, accept, done, log)."""
    def __init__(self, world: str, arm: str, seed: int, steps: int):
        assert world in W and arm in ARMS
        self.world, self.arm, self.seed, self.w = world, arm, seed, W[world]
        self.script, self.oracle = script_of(world, seed, steps); self.live = self.oracle[0]; self.log: List[dict] = []; self.t = 0; self.stopped = None

    @property
    def done(self) -> bool: return self.t >= len(self.script)

    def request(self) -> T.TRequest:
        it = self.script[self.t]; user = f"{self.w.render(self.live)}\n\nRequest: {it['instruction']}"
        return T.TRequest([{"role": "system", "content": system_for(self.w, self.arm)}, {"role": "user", "content": user}], it["ideal"], seed=self.seed, step=self.t + 1, prompt="old", shape="state", regex=self.w.rx)

    def accept(self, rep: PS.Reply) -> None:
        it = self.script[self.t]; prev, want = self.oracle[self.t], self.oracle[self.t + 1]
        after, parsed, admitted = policy_next(self.w, self.arm, self.live, rep.text)
        ok_state, _, _ = policy_next(self.w, self.arm, prev, rep.text)                  # the same reply and policy, on the oracle's state before the step
        self.log.append(dict(t=self.t + 1, kind=it["kind"], conflict=it["conflict"], layer=it["layer"], text=rep.text, n_prompt=rep.n_prompt, n_cached=rep.n_cached, n_new=rep.n_new, parsed=parsed, admitted=admitted,
                             exact=equal(after, want), step_ok=equal(ok_state, want), refused=(rep.text == "NOOP"), held_valid=valid_state(self.w, after), text_equal=rep.text == it["ideal"], literal=rep.text == it["literal"]))
        self.live = after; self.t += 1


def make_sessions(world: str, arm: str, steps: int, jobs=(1, 2, 3), seed: int = SEED) -> list:
    return [StateSession(world, arm, seed * 1000 + e, steps) for e in jobs]


# ================================================================ the numbers
def f(x, d=2): return None if x is None else round(x, d)


def share(xs: List[dict], key: str) -> Optional[float]: return f(100.0 * sum(1 for x in xs if x[key]) / len(xs)) if xs else None


def summarize(cell: dict) -> dict:
    logs = [x for s in cell["sessions"] for x in s["log"]]; n = len(logs)
    cf = [x for x in logs if x["conflict"]]; ok = [x for x in logs if not x["conflict"]]
    out = dict(world=cell["world"], arm=cell["arm"], steps=n, wall_seconds=f(cell["wall_seconds"], 1), parsed=share(logs, "parsed"), exact=share(logs, "exact"), step_ok=share(logs, "step_ok"), held_valid=share(logs, "held_valid"),
               step_ok_valid_requests=share(ok, "step_ok"), step_ok_conflicts=share(cf, "step_ok"), n_conflicts=len(cf), refused=share(logs, "refused"), literal_on_conflicts=share(cf, "literal"),
               step_ok_generic_conflicts=share([x for x in cf if x["layer"] == "generic"], "step_ok"), step_ok_domain_conflicts=share([x for x in cf if x["layer"] == "domain"], "step_ok"),
               admitted_conflicts=share(cf, "admitted"), exact_first_half=share([x for x in logs if x["t"] <= cell["steps"] // 2], "exact"),
               first_off=[next((x["t"] for x in s["log"] if not x["exact"]), None) for s in cell["sessions"]])
    by = {}
    for x in logs: by.setdefault(x["kind"], []).append(x)
    out["step_ok_by_kind"] = {k: (len(v), share(v, "step_ok")) for k, v in sorted(by.items())}
    p = sum(x["n_prompt"] for x in logs); c = sum((x["n_cached"] or 0) for x in logs)
    out["prompt_per_step"] = f(p / n, 1) if n else None; out["computed_per_step"] = f((p - c) / n, 1) if n else None; out["cached_share"] = f(c / p, 4) if p else None
    return out


def run_world(engine, world: str, arm: str, steps: int, jobs=(1, 2, 3), seed: int = SEED, progress=None) -> dict:
    import axi.experiments.provider_side_long as L
    r = L.run_sessions(engine, make_sessions(world, arm, steps, jobs, seed), progress)
    return dict(world=world, arm=arm, steps=steps, jobs=list(jobs), seed=seed, conflict_share=CONFLICT_SHARE, generic_share=GENERIC_SHARE, engine=engine.info(), **r)


def regex_sha256() -> str: return hashlib.sha256("\n".join(W[k].rx for k in WORLDS).encode()).hexdigest()

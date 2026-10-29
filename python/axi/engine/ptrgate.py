"""Pointer gate: the dangling-edge rule of the double pushout, read on C and C++ heap pointers.

The idea (Rajnish Choubey's gluing conditions, applied by us to code): a heap object is a node,
a pointer variable is an edge to it. Freeing the node while an edge still points at it leaves a
*dangling edge*; using the edge afterwards is a use after free (CWE-416); freeing a node that is
already deleted is a double free (CWE-415). The host holds the graph and decides, by fixed rules,
with no model; a model can be handed the exact findings.

Scope, stated plainly: one function at a time, straight-line code, if / else, loops (taken once or
not at all), branches whose condition is a known constant, malloc/calloc/new, free/delete,
aliasing by plain assignment. Anything else (goto, switch, do, try, labels, calls that free
for us) makes the function `unsupported`: the gate says nothing instead of guessing.
"""
import re

CONST_TRUE = {"1", "5==5", "STATIC_CONST_TRUE", "staticTrue", "staticReturnsTrue()", "GLOBAL_CONST_TRUE",
              "globalTrue", "globalReturnsTrue()", "STATIC_CONST_FIVE==5", "staticFive==5",
              "GLOBAL_CONST_FIVE==5", "globalFive==5", "true", "!0", "STATIC_CONST_FALSE==0",
              "staticFalse==0", "GLOBAL_CONST_FALSE==0", "globalFalse==0"}
CONST_FALSE = {"0", "5!=5", "STATIC_CONST_FALSE", "staticFalse", "staticReturnsFalse()", "GLOBAL_CONST_FALSE",
               "globalFalse", "globalReturnsFalse()", "STATIC_CONST_FIVE!=5", "staticFive!=5",
               "GLOBAL_CONST_FIVE!=5", "globalFive!=5", "false", "!1"}
ALLOCS = {"malloc", "calloc", "realloc", "new", "ALLOCA"}
STOPS = {"exit", "abort", "_exit"}
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+\w*|->|==|!=|<=|>=|&&|\|\||\+\+|--|::|\S')


def strip(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))


def tokens(src):
    return TOKEN.findall(strip(src))


def _match(t, i, open_, close):
    d = 0
    for j in range(i, len(t)):
        if t[j] == open_:
            d += 1
        elif t[j] == close:
            d -= 1
            if d == 0:
                return j
    return len(t) - 1


def functions(t):
    """{name: body tokens} for every `name ( ... ) {` that is not inside another function."""
    out, i, kw = {}, 0, {"if", "while", "for", "switch", "catch", "sizeof"}
    while i < len(t) - 3:
        if re.match(r"[A-Za-z_]\w*$", t[i]) and t[i] not in kw and t[i + 1] == "(":
            j = _match(t, i + 1, "(", ")")
            k = j + 1
            while k < len(t) and t[k] in ("const", "noexcept"):
                k += 1
            if k < len(t) and t[k] == "{":
                e = _match(t, k, "{", "}")
                out[t[i]] = t[k + 1:e]
                i = e + 1
                continue
        i += 1
    return out


class Unsupported(Exception):
    pass


def parse(t, i):
    """One statement from t[i:]; returns (node, next index)."""
    if i >= len(t):
        return ("block", []), i
    x = t[i]
    if x == "{":
        e, body, j = _match(t, i, "{", "}"), [], i + 1
        while j < e:
            n, j = parse(t, j)
            body.append(n)
        return ("block", body), e + 1
    if x == ";":
        return ("block", []), i + 1
    if x in ("do", "try", "catch", "case", "default"):
        raise Unsupported(x)
    if x == "goto":
        return ("goto", t[i + 1]), i + 3
    if x == "switch":
        e = _match(t, i + 1, "(", ")")
        cond = t[i + 2:e]
        b = _match(t, e + 1, "{", "}")
        cases, cur, j = [], None, e + 2
        while j < b:
            if t[j] in ("case", "default"):
                k = j + 1
                while t[k] != ":":
                    k += 1
                cur = ["".join(t[j + 1:k]) if t[j] == "case" else "default", []]
                cases.append(cur)
                j = k + 1
                continue
            if cur is None:
                raise Unsupported("switch")
            n, j = parse(t, j)
            cur[1].append(n)
        return ("switch", cond, cases), b + 1
    if i + 1 < len(t) and t[i + 1] == ":" and re.match(r"[A-Za-z_]\w*$", x):
        return ("label", x), i + 2
    if x in ("if", "while", "for"):
        e = _match(t, i + 1, "(", ")")
        cond = t[i + 2:e]
        body, j = parse(t, e + 1)
        if x == "if":
            if j < len(t) and t[j] == "else":
                other, j = parse(t, j + 1)
                return ("if", cond, body, other), j
            return ("if", cond, body, ("block", [])), j
        return ("loop", x, cond, body), j
    d, j = 0, i
    while j < len(t):
        if t[j] in "([{":
            d += 1
        elif t[j] in ")]}":
            d -= 1
        elif t[j] == ";" and d == 0:
            break
        j += 1
    return ("simple", t[i:j]), j + 1


def const(cond):
    s = "".join(cond)
    while s.startswith("(") and s.endswith(")") and _match(list(s), 0, "(", ")") == len(s) - 1:
        s = s[1:-1]
    if s in CONST_TRUE:
        return True
    if s in CONST_FALSE:
        return False
    return None


def _sizeof_free(t):
    out, i = [], 0
    while i < len(t):
        if t[i] == "sizeof":
            i = _match(t, i + 1, "(", ")") + 1 if i + 1 < len(t) and t[i + 1] == "(" else i + 2
            continue
        out.append(t[i])
        i += 1
    return out


def _lhs(t):
    """(var, index of '=') when t is a plain assignment or an initialised declaration, else None."""
    d = 0
    for k, x in enumerate(t):
        if x in "([":
            d += 1
        elif x in ")]":
            d -= 1
        elif x == "=" and d == 0:
            left = t[:k]
            if left and left[0] != "*" and re.match(r"[A-Za-z_]\w*$", left[-1]) and not any(y in left for y in ("[", "->", ".", "(")):
                if all(re.match(r"[A-Za-z_]\w*$|\*$|::$|<|>", y) for y in left):
                    return left[-1], k
            return None
    return None


class Path:
    __slots__ = ("var", "freed", "n", "dead", "skip")

    def __init__(self):
        self.var, self.freed, self.n, self.dead, self.skip = {}, {}, 0, False, None

    def copy(self):
        p = Path()
        p.var, p.freed, p.n, p.dead, p.skip = dict(self.var), dict(self.freed), self.n, self.dead, self.skip
        return p

    def key(self):
        return (tuple(sorted(self.var.items())), tuple(sorted(self.freed.items())), self.dead, self.skip)


def _uses(t, p, skip, found, text):
    for x in t:
        if x in p.var and x not in skip and p.var[x] is not None and p.var[x] in p.freed:
            found.add(("UAF", x, p.freed[p.var[x]], text))


def _simple(t, p, found, nodes_text):
    text = " ".join(t)
    t = _sizeof_free(t)
    if not t:
        return
    if t[0] in STOPS:
        p.dead = True
        return
    if t[0] == "return":
        _uses(t[1:], p, set(), found, text)
        p.dead = True
        return
    if t[0] == "free" and len(t) >= 4 or t[0] == "delete":
        arg = [x for x in t[1:] if re.match(r"[A-Za-z_]\w*$", x)]
        v = arg[0] if arg else None
        if v in p.var and p.var[v] is not None:
            h = p.var[v]
            if h in p.freed:
                found.add(("DF", v, p.freed[h], text))
            else:
                p.freed[h] = text
        return
    a = _lhs(t)
    if a:
        v, k = a
        rhs = t[k + 1:]
        rs = set(x for x in rhs if re.match(r"[A-Za-z_]\w*$", x))
        if not (len(rhs) == 1 and rhs[0] in p.var):
            _uses(rhs, p, set(), found, text)
        if rs & ALLOCS:
            p.n += 1
            p.var[v] = p.n
        elif rhs and rhs[-1] in p.var and len([x for x in rhs if x in p.var]) == 1 and all(
                y == rhs[-1] or y in ("(", ")", "*") or re.match(r"[A-Za-z_]\w*$", y) and y not in p.var for y in rhs):
            p.var[v] = p.var[rhs[-1]]
        elif rhs and rhs[-1] in ("NULL", "0", "nullptr"):
            p.var[v] = None
        else:
            p.var[v] = None
        return
    _uses(t, p, set(), found, text)


def _run(node, paths, found):
    out = []
    kind = node[0]
    if kind == "block":
        cur = paths
        for n in node[1]:
            if n[0] == "label":
                for p in cur:
                    if p.skip == n[1]:
                        p.skip = None
                continue
            go = [p for p in cur if p.skip is None]
            wait = [p for p in cur if p.skip is not None]
            cur = _dedupe(_run(n, go, found) + wait)
        return cur
    if kind == "goto":
        for p in paths:
            if not p.dead:
                p.skip = node[1]
        return paths
    if kind == "label":
        return paths
    if kind == "switch":
        v = "".join(node[1])
        pick = None
        for lab, body in node[2]:
            if lab == v:
                pick = body
        if pick is None:
            pick = [b for lab, b in node[2] if lab == "default"]
            pick = pick[0] if pick else []
        stmts = []
        for n in pick:
            if n == ("simple", ["break"]):
                break
            stmts.append(n)
        return _run(("block", stmts), paths, found)
    if kind == "simple":
        for p in paths:
            if p.dead:
                out.append(p)
                continue
            q = p.copy()
            _simple(node[1], q, found, None)
            out.append(q)
        return _dedupe(out)
    if kind == "if":
        c = const(node[1])
        for p in paths:
            if not p.dead:
                _uses(_sizeof_free(node[1]), p, set(), found, " ".join(node[1]))
        if c is True:
            return _run(node[2], paths, found)
        if c is False:
            return _run(node[3], paths, found)
        a = _run(node[2], [p.copy() for p in paths], found)
        b = _run(node[3], [p.copy() for p in paths], found)
        return _dedupe(a + b)
    if kind == "loop":
        c = const(node[2]) if node[1] == "while" else (True if node[1] == "for" and "".join(node[2]).startswith(";") else None)
        if c is False:
            return paths
        body = _run(node[3], [p.copy() for p in paths], found)
        return _dedupe(body if c is True else paths + body)
    return paths


def _dedupe(ps):
    seen, out = set(), []
    for p in ps:
        k = p.key()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out[:256]


def analyze_body(body):
    """('vulnerable', findings) | ('clean', []) | ('unsupported', reason)."""
    try:
        t = list(body)
        i, nodes = 0, []
        while i < len(t):
            n, i = parse(t, i)
            nodes.append(n)
    except Unsupported as e:
        return "unsupported", str(e)
    found = set()
    _run(("block", nodes), [Path()], found)
    kinds = sorted(found)
    return ("vulnerable", kinds) if kinds else ("clean", [])


def analyze_source(src):
    out = {}
    for name, body in functions(tokens(src)).items():
        out[name] = analyze_body(body)
    return out


def facts_text(result):
    """The exact findings, as lines a model can be shown (at most 8)."""
    verdict, f = result
    if verdict == "unsupported":
        return "The host could not analyse this function (" + f + "): no finding is claimed."
    if verdict == "clean":
        return "The host found no freed pointer that is used or freed again on any path."
    lines = []
    for k, v, where, text in f[:8]:
        what = "used after free" if k == "UAF" else "freed twice"
        lines.append("%s: pointer %s was freed at `%s` and later %s at `%s`." % (k, v, where, what, text))
    return "\n".join(lines)

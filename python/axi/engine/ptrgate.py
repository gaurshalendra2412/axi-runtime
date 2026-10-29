"""Pointer gate: the dangling-edge rule of the double pushout, read on C and C++ heap pointers.

The idea (Rajnish Choubey's gluing conditions, applied by us to code): a heap object is a node,
a pointer variable is an edge to it. Freeing the node while an edge still points at it leaves a
*dangling edge*; touching the memory through that edge is a use after free (CWE-416); freeing a
node that is already deleted is a double free (CWE-415). The host holds the graph and decides by
fixed rules, with no model; a model can be handed the exact findings.

Two kinds of reference to a pointer are kept apart:

* **transport** (assign it, pass it to a function that ignores it, return it, compare it): safe,
  whatever the pointer points at;
* **dereference** (`*p`, `p->f`, `p[i]`, a library call that reads it, `free(p)`): needs a live node.

A function of the same file that is called as a statement is entered with the caller's pointers
(depth at most 8), so what it does with its pointer parameters, with references, with pointers to
pointers and with global or static variables is read from its body and not guessed from a name.

Scope, stated plainly: C and C++ read by tokens, not by a full parser. One function (and what it
calls in the same file) at a time; if / else, loops (taken once or not at all), switch on a
constant, goto to a later label; branches with a constant condition or a global flag whose value
is known; malloc/calloc/new, free/delete. Anything else (do, try, labels in odd places) makes the
function `unsupported`: the gate says nothing instead of guessing.
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
NOTDEREF_BEFORE = re.compile(r"[A-Za-z_0-9)\]]$")
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+\w*|->|==|!=|<=|>=|&&|\|\||\+\+|--|::|\S')
IDENT = re.compile(r"[A-Za-z_]\w*$")
MAX_DEPTH = 8


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
    """(funcs, outside): {name: (parameter tokens, body tokens)} for every `name ( ... ) {` that is
    not inside another function, and the tokens that lie outside every function body."""
    out, outside, i, kw = {}, [], 0, {"if", "while", "for", "switch", "catch", "sizeof"}
    while i < len(t):
        if i < len(t) - 3 and IDENT.match(t[i]) and t[i] not in kw and t[i + 1] == "(":
            j = _match(t, i + 1, "(", ")")
            k = j + 1
            while k < len(t) and t[k] in ("const", "noexcept"):
                k += 1
            if k < len(t) and t[k] == "{":
                e = _match(t, k, "{", "}")
                out[t[i]] = (t[i + 2:j], t[k + 1:e])
                i = e + 1
                continue
        outside.append(t[i])
        i += 1
    return out, outside


def globals_of(outside):
    """Names and integer initial values of the variables declared outside every function."""
    names, init, cur = set(), {}, []
    for x in outside + [";"]:
        if x == ";":
            left = cur[:cur.index("=")] if "=" in cur else cur
            if left and "(" not in left and IDENT.match(left[-1]):
                names.add(left[-1])
                if "=" in cur and len(cur) == len(left) + 2 and re.match(r"-?\d+$", cur[-1]):
                    init[left[-1]] = int(cur[-1])
            cur = []
        elif x in "{}":
            cur = []
        else:
            cur.append(x)
    return names, init


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
    if i + 1 < len(t) and t[i + 1] == ":" and IDENT.match(x):
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


class Ctx:
    def __init__(self, funcs, globs, init):
        self.globs, self.init = globs, init
        self.funcs = {}
        for name, (params, body) in funcs.items():
            self.funcs[name] = (_params(params), _body(body))


def _params(pt):
    out, cur = [], []
    for x in pt + [","]:
        if x == ",":
            ids = [y for y in cur if IDENT.match(y) and y not in ("const", "struct", "unsigned", "signed", "void")]
            if ids:
                out.append((ids[-1], "&" in cur))
            cur = []
        else:
            cur.append(x)
    return out


def _body(body):
    try:
        i, nodes = 0, []
        while i < len(body):
            n, i = parse(body, i)
            nodes.append(n)
        return nodes
    except Unsupported:
        return None


def _split_args(t):
    out, cur, d = [], [], 0
    for x in t:
        if x in "([{":
            d += 1
        elif x in ")]}":
            d -= 1
        if x == "," and d == 0:
            out.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur or out:
        out.append(cur)
    return out


class Path:
    __slots__ = ("loc", "glob", "freed", "n", "dead", "skip", "ret", "retval", "fn", "ptrto", "ival", "depth", "refs", "arr")

    def __init__(self, init):
        self.loc, self.glob, self.freed, self.n = {}, {}, {}, 0
        self.dead, self.skip, self.ret, self.retval = False, None, False, None
        self.fn, self.ptrto, self.ival, self.depth, self.refs, self.arr = {}, {}, dict(init), 0, {}, {}

    def copy(self):
        p = Path.__new__(Path)
        p.loc, p.glob, p.freed, p.n = dict(self.loc), dict(self.glob), dict(self.freed), self.n
        p.dead, p.skip, p.ret, p.retval = self.dead, self.skip, self.ret, self.retval
        p.fn, p.ptrto, p.ival, p.depth = dict(self.fn), dict(self.ptrto), dict(self.ival), self.depth
        p.refs, p.arr = dict(self.refs), dict(self.arr)
        return p

    def key(self):
        f = lambda d: tuple(sorted(d.items(), key=repr))
        return (f(self.loc), f(self.glob), f(self.freed), self.dead, self.skip, self.ret, f(self.fn), f(self.ptrto),
                f(self.ival), f(self.refs), f(self.arr))


def get(p, c, n):
    n = p.refs.get(n, n)
    if n in p.loc:
        return p.loc[n]
    if n in c.globs:
        return p.glob.get(n)
    return None


def put(p, c, n, h):
    n = p.refs.get(n, n)
    if n in c.globs and n not in p.loc and not isinstance(n, tuple):
        p.glob[n] = h
    else:
        p.loc[n] = h


def const(cond, p=None):
    s = "".join(cond)
    while s.startswith("(") and s.endswith(")") and _match(list(s), 0, "(", ")") == len(s) - 1:
        s = s[1:-1]
    if p is not None:
        if s in p.ival:
            return bool(p.ival[s])
        if s.startswith("!") and s[1:] in p.ival:
            return not p.ival[s[1:]]
        m = re.match(r"(\w+)(==|!=)(\w+)$", s)
        if m:
            def val(x):
                return int(x) if re.match(r"-?\d+$", x) else p.ival.get(x)
            a, b = val(m.group(1)), val(m.group(3))
            if a is not None and b is not None:
                return (a == b) if m.group(2) == "==" else (a != b)
    if s in CONST_TRUE:
        return True
    if s in CONST_FALSE:
        return False
    return None


def _clean(t):
    out, i = [], 0
    while i < len(t):
        if t[i] == "sizeof":
            i = _match(t, i + 1, "(", ")") + 1 if i + 1 < len(t) and t[i + 1] == "(" else i + 2
            continue
        if (t[i] == "." and out and IDENT.match(out[-1]) and i + 1 < len(t) and IDENT.match(t[i + 1])):
            i += 2
            continue
        out.append(t[i])
        i += 1
    return out


def _lhs(t):
    """(var, index of '=', is_reference) for a plain assignment or an initialised declaration."""
    d = 0
    for k, x in enumerate(t):
        if x in "([":
            d += 1
        elif x in ")]":
            d -= 1
        elif x == "=" and d == 0:
            left = t[:k]
            if left and left[0] != "*" and IDENT.match(left[-1]) and not any(y in left for y in ("[", "->", "(")):
                if all(IDENT.match(y) or y in ("*", "&", "::", "<", ">") for y in left):
                    return left[-1], k, "&" in left
            return None
    return None


def _deref_at(t, i):
    if i + 1 < len(t) and t[i + 1] in ("->", "["):
        return True
    return i >= 1 and t[i - 1] == "*" and (i == 1 or not NOTDEREF_BEFORE.search(t[i - 2]))


def _scan(t, p, c, found, text):
    """Report the dereferences of a freed pointer in an expression."""
    i = 0
    while i < len(t):
        x = t[i]
        if IDENT.match(x) and i + 1 < len(t) and t[i + 1] == "(" and x not in ALLOCS:
            j = _match(t, i + 1, "(", ")")
            for y in t[i + 2:j]:
                h = get(p, c, y) if IDENT.match(y) else None
                if h is not None and h in p.freed:
                    found.add(("UAF", y, p.freed[h], text))
            _scan_inner(t[i + 2:j], p, c, found, text)
            i = j + 1
            continue
        h = get(p, c, x) if IDENT.match(x) else None
        if h is not None and h in p.freed and _deref_at(t, i):
            found.add(("UAF", x, p.freed[h], text))
        i += 1


def _scan_inner(t, p, c, found, text):
    for i, x in enumerate(t):
        h = get(p, c, x) if IDENT.match(x) else None
        if h is not None and h in p.freed and _deref_at(t, i):
            found.add(("UAF", x, p.freed[h], text))


TYPEWORDS = {"char", "int", "long", "void", "struct", "unsigned", "signed", "wchar_t", "int64_t", "size_t", "float",
             "double", "const", "short"}


def _norm(rhs):
    """The right-hand side without casts and outer parentheses."""
    rhs = list(rhs)
    while rhs and rhs[0] == "(":
        k = _match(rhs, 0, "(", ")")
        inside = rhs[1:k]
        if k == len(rhs) - 1:
            rhs = inside
        elif inside and all(IDENT.match(y) or y in ("*", "::") for y in inside) and ("*" in inside or inside[0] in TYPEWORDS):
            rhs = rhs[k + 1:]
        else:
            break
    return rhs


def _tracked(rhs, p, c):
    """The one tracked variable that the right-hand side only carries (a cast is fine), else None."""
    vs = [x for x in rhs if IDENT.match(x) and (p.refs.get(x, x) in p.loc or p.refs.get(x, x) in c.globs)]
    if len(vs) == 1 and all(y == vs[0] or y in ("(", ")", "*") or IDENT.match(y) and y not in vs
                            and p.refs.get(y, y) not in p.loc and p.refs.get(y, y) not in c.globs for y in rhs):
        return vs[0]
    return None


def _assign(p, c, v, rhs):
    """Set variable v from the right-hand side tokens (no call)."""
    rhs = _norm(rhs)
    rs = {x for x in rhs if IDENT.match(x)}
    if rs & ALLOCS:
        p.n += 1
        put(p, c, v, p.n)
    elif len(rhs) >= 2 and rhs[0] == "&" and IDENT.match(rhs[1]) and len(rhs) == 2:
        p.ptrto[v] = rhs[1]
        put(p, c, v, None)
    elif len(rhs) == 1 and rhs[0] in p.ptrto:
        p.ptrto[v] = p.ptrto[rhs[0]]
        put(p, c, v, None)
    elif len(rhs) >= 3 and IDENT.match(rhs[0]) and rhs[1] == "[" and rhs[-1] == "]" and rhs[0] in p.arr:
        put(p, c, v, p.arr[rhs[0]])
    elif len(rhs) == 2 and rhs[0] == "*" and rhs[1] in p.ptrto:
        put(p, c, v, get(p, c, p.ptrto[rhs[1]]))
    elif _tracked(rhs, p, c):
        put(p, c, v, get(p, c, _tracked(rhs, p, c)))
    else:
        put(p, c, v, None)
    if len(rhs) == 1 and re.match(r"-?\d+$", rhs[0]):
        p.ival[v] = int(rhs[0])
    elif v in p.ival:
        del p.ival[v]
    if len(rhs) == 1 and rhs[0] in c.funcs:
        p.fn[v] = rhs[0]


def _callee(t, p, c):
    """(function name, argument tokens) when t is exactly `f ( ... )` for a function of the file."""
    if len(t) >= 3 and IDENT.match(t[0]) and t[1] == "(" and _match(t, 1, "(", ")") == len(t) - 1:
        name = t[0] if t[0] in c.funcs else p.fn.get(t[0])
        if name in c.funcs:
            return name, t[2:-1]
    return None


def _inline(p, c, name, argtoks, lhs, found, text):
    params, nodes = c.funcs[name]
    args = _split_args(argtoks)
    if p.depth >= MAX_DEPTH or nodes is None:
        _scan(["f", "("] + argtoks + [")"], p, c, found, text)
        q = p.copy()
        if lhs:
            put(q, c, lhs, None)
        return [q]
    q = p.copy()
    q.loc, q.ptrto, q.fn, q.refs, q.arr = {}, {}, {}, {}, {}
    q.depth, q.ret, q.retval, q.skip = p.depth + 1, False, None, None
    refmap, ptrmap = {}, {}
    for i, (pname, isref) in enumerate(params):
        a = args[i] if i < len(args) else []
        if len(a) == 1 and IDENT.match(a[0]):
            q.loc[pname] = get(p, c, a[0])
            if isref:
                refmap[pname] = a[0]
            if a[0] in p.arr:
                q.arr[pname] = p.arr[a[0]]
            if a[0] in p.ptrto and not isinstance(p.ptrto[a[0]], tuple):
                q.ptrto[pname] = ("*", pname)
                q.loc[("*", pname)] = get(p, c, p.ptrto[a[0]])
                ptrmap[pname] = p.ptrto[a[0]]
            if a[0] in p.fn:
                q.fn[pname] = p.fn[a[0]]
            if a[0] in c.funcs:
                q.fn[pname] = a[0]
        elif len(a) == 2 and a[0] == "&" and IDENT.match(a[1]):
            q.loc[("*", pname)] = get(p, c, a[1])
            q.ptrto[pname] = ("*", pname)
            ptrmap[pname] = a[1]
            q.loc[pname] = None
        else:
            q.loc[pname] = None
    out = []
    for r in _run(("block", nodes), [q], c, found):
        rl, rdead = r.loc, r.dead
        r.loc, r.ptrto, r.fn, r.refs, r.arr = dict(p.loc), dict(p.ptrto), dict(p.fn), dict(p.refs), dict(p.arr)
        r.depth, r.skip = p.depth, None
        if not rdead:
            for pname, var in refmap.items():
                put(r, c, var, rl.get(pname))
            for pname, var in ptrmap.items():
                if ("*", pname) in rl:
                    put(r, c, var, rl[("*", pname)])
            if lhs:
                put(r, c, lhs, r.retval)
        r.ret, r.retval = False, None
        out.append(r)
    return out


def _simple(t, p, c, found):
    """Execute one simple statement on one path; returns the paths that follow."""
    text = " ".join(t)
    t = _clean(t)
    if not t:
        return [p]
    if t[0] in STOPS:
        p.dead = True
        return [p]
    if t[0] == "return":
        _scan(t[1:], p, c, found, text)
        p.retval = get(p, c, t[1]) if len(t) == 2 and IDENT.match(t[1]) else None
        p.ret = True
        return [p]
    if t[0] == "free" and len(t) >= 4 or t[0] == "delete":
        arg = [x for x in t[1:] if IDENT.match(x)]
        v = arg[0] if arg else None
        h = get(p, c, v) if v else None
        if h is not None:
            if h in p.freed:
                found.add(("DF", v, p.freed[h], text))
            else:
                p.freed[h] = text
        return [p]
    m = re.search(r"\(\s*\*\s*(\w+)\s*\)\s*\(.*\)\s*=\s*(\w+)$", " ".join(t))
    if m and m.group(2) in c.funcs:
        p.fn[m.group(1)] = m.group(2)
        return [p]
    cal = _callee(t, p, c)
    if cal:
        return _inline(p, c, cal[0], cal[1], None, found, text)
    a = _lhs(t)
    if a is None and len(t) >= 6 and IDENT.match(t[0]) and t[1] == "[" and "]" in t:
        k = t.index("]")
        if k + 1 < len(t) and t[k + 1] == "=" and get(p, c, t[0]) is None:
            _scan(t[k + 2:], p, c, found, text)
            r = _norm(t[k + 2:])
            p.arr[t[0]] = get(p, c, r[0]) if len(r) == 1 and IDENT.match(r[0]) else None
            return [p]
    if a is None and len(t) >= 4 and t[0] == "*" and IDENT.match(t[1]) and t[2] == "=":
        P = t[1]
        if P in p.ptrto:
            _scan(t[3:], p, c, found, text)
            tmp = Path.__new__(Path)
            _assign(p, c, p.ptrto[P], t[3:])
            return [p]
    if a:
        v, k, isref = a
        rhs = t[k + 1:]
        cal = _callee(rhs, p, c)
        if cal:
            return _inline(p, c, cal[0], cal[1], v, found, text)
        _scan(rhs, p, c, found, text)
        if isref and len(rhs) == 1 and IDENT.match(rhs[0]):
            p.refs[v] = p.refs.get(rhs[0], rhs[0])
            return [p]
        _assign(p, c, v, rhs)
        return [p]
    _scan(t, p, c, found, text)
    return [p]


def _dedupe(ps):
    seen, out = set(), []
    for p in ps:
        k = p.key()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out[:256]


def _run(node, paths, c, found):
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
            cur = _dedupe(_run(n, go, c, found) + wait)
        return cur
    if kind == "goto":
        for p in paths:
            if not p.dead and not p.ret:
                p.skip = node[1]
        return paths
    if kind == "label":
        return paths
    if kind == "simple":
        out = []
        for p in paths:
            if p.dead or p.ret:
                out.append(p)
            else:
                out.extend(_simple(node[1], p.copy(), c, found))
        return _dedupe(out)
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
        return _run(("block", stmts), paths, c, found)
    if kind == "if":
        yes, no, unk, out = [], [], [], []
        for p in paths:
            if p.dead or p.ret:
                out.append(p)
                continue
            _scan(_clean(node[1]), p, c, found, " ".join(node[1]))
            v = const(node[1], p)
            (yes if v is True else no if v is False else unk).append(p)
        res = out
        if yes:
            res = res + _run(node[2], yes, c, found)
        if no:
            res = res + _run(node[3], no, c, found)
        if unk:
            res = res + _run(node[2], [p.copy() for p in unk], c, found) + _run(node[3], [p.copy() for p in unk], c, found)
        return _dedupe(res)
    if kind == "loop":
        cv = const(node[2]) if node[1] == "while" else (True if "".join(node[2]).startswith(";") else None)
        if cv is False:
            return paths
        body = _run(node[3], [p.copy() for p in paths], c, found)
        return _dedupe(body if cv is True else paths + body)
    return paths


def _analyze_one(name, c):
    params, nodes = c.funcs[name]
    if nodes is None:
        return "unsupported", "do/try/case", False
    found = set()
    _run(("block", nodes), [Path(c.init)], c, found)
    kinds = sorted(found)
    return ("vulnerable", kinds, False) if kinds else ("clean", [], False)


def analyze_body(body):
    """Compatibility: analyse a body given as tokens, no other functions in view."""
    c = Ctx({"_f": ([], body)}, set(), {})
    r = _analyze_one("_f", c)
    return r[0], r[1]


def analyze_source(src):
    """{function name: (verdict, findings)} for every function in the source text."""
    funcs, outside = functions(tokens(src))
    names, init = globals_of(outside)
    c = Ctx(funcs, names, init)
    return {n: _analyze_one(n, c)[:2] for n in funcs}


def facts_text(result):
    """The exact findings, as lines a model can be shown (at most 8)."""
    verdict, f = result
    if verdict == "unsupported":
        return "The host could not analyse this function (" + f + "): no finding is claimed."
    if verdict == "clean":
        return "The host found no freed pointer that is dereferenced or freed again on any path."
    lines = []
    for k, v, where, text in f[:8]:
        what = "dereferenced after free" if k == "UAF" else "freed twice"
        lines.append("%s: pointer %s was freed at `%s` and later %s at `%s`." % (k, v, where, what, text))
    return "\n".join(lines)

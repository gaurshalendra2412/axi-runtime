"""Test 28 (docs/PTRGATE_MODEL_TEST.md): a model plain against a model given the pointer gate's findings.

Items: Juliet functions (use after free, double free) from the data-flow and multi-file variants, plus some
that the gate cannot read (C++ classes and containers). One function is one question: "does it use memory
after it was freed, or free the same memory twice?". Two arms, the same questions and the same model:
  P  the source only (comments removed, names changed, strings removed);
  X  the same, plus what the pointer gate says about the function (its exact findings, or that it cannot read it).
The gate is run on the changed source, so the changed names are the ones it speaks of. No label, no word of the
original names (bad, good, G2B, B2G, CWE...) is in a prompt (the tests check it).
The gate was fitted on these cases: arm X is a demonstration of what exact host facts do for a model, not a result on new code.
"""
import json
import random
import re

from axi.engine import ptrgate
from axi.experiments import ptrgate_check as PC

N_HARD, N_REST = 130, 20          # per label
HARD_VARIANTS = ("21", "31", "32", "33", "34", "41", "42", "43", "44", "45", "51", "52", "53", "54", "61", "62", "63", "64", "65", "66", "67", "68")
LEAK = re.compile(r"(?i)bad|good|g2b|b2g|cwe\d|use_after_free|double_free|flaw|fix")
SYSTEM = "You are a careful C and C++ code reviewer. Answer with one JSON object and nothing else."
QUESTION = ('When the function `{t}` runs (counting the functions it calls), can it use memory after that memory was freed or deleted, '
            'or free or delete the same memory twice? Answer with one JSON object: {{"memory_error": true or false, "why": "one short sentence"}}.')
HOST = "\n\nA deterministic checker (a program with set rules, no model) read the same code and reports on `{t}`:\n{facts}"


def drop_functions(toks, drop):
    """The tokens without the definitions of the functions whose names `drop` accepts (the main function and the wrapper that calls the clean ones: both give away which function is which)."""
    out, i, kw = [], 0, {"if", "while", "for", "switch", "catch", "sizeof"}
    while i < len(toks):
        if i < len(toks) - 3 and re.match(r"[A-Za-z_]\w*$", toks[i]) and toks[i] not in kw and toks[i + 1] == "(":
            j = ptrgate._match(toks, i + 1, "(", ")")
            k = j + 1
            while k < len(toks) and toks[k] in ("const", "noexcept"):
                k += 1
            if k < len(toks) and toks[k] == "{":
                e = ptrgate._match(toks, k, "{", "}")
                if drop(toks[i]):
                    # also the return type that precedes the name
                    while out and re.match(r"[A-Za-z_]\w*$|\*$", out[-1]) and out[-1] not in ("{", "}", ";"):
                        out.pop()
                    i = e + 1
                    continue
        out.append(toks[i])
        i += 1
    return out


def anonymize(src: str, target: str, rng: random.Random):
    """(source, new name of target): comments and preprocessor lines out, strings out, every name that carries a label word or a case name renamed to a random f<number>."""
    toks = drop_functions(ptrgate.tokens(src), lambda n: n == "main" or n == "good" or n.endswith("_good"))
    names = {}
    pool = rng.sample(range(100, 1000), 400)

    def new(x):
        if x not in names:
            names[x] = "f%d" % pool[len(names)]
        return names[x]
    out = []
    for x in toks:
        if x.startswith('"'):
            out.append('"s"')
        elif re.match(r"[A-Za-z_]\w*$", x) and LEAK.search(x):
            out.append(new(x))
        else:
            out.append(x)
    return out, new(target)


def pretty(toks):
    s = " ".join(toks)
    return re.sub(r"\s*([;{}])\s*", lambda m: m.group(1) + "\n", s).replace(" ( ", "(").replace(" )", ")").replace("( ", "(")


def units_with_targets(root, part):
    for var, cwe, src in PC.units(root, part):
        for name in ptrgate.analyze_source(src):
            y = PC.label(name)
            if y is not None:
                yield var, cwe, src, name, y


def build_items(root, seed=28):
    rng = random.Random(seed)
    pools = {}
    for part in ("flow", "multi", "rest"):
        for var, cwe, src, name, y in units_with_targets(root, part):
            key = ("rest" if part == "rest" else "hard", y)
            pools.setdefault(key, []).append((var, cwe, src, name, y))
    items = []
    for key, n in ((("hard", 1), N_HARD), (("hard", 0), N_HARD), (("rest", 1), N_REST), (("rest", 0), N_REST)):
        pool = pools[key]
        for var, cwe, src, name, y in rng.sample(pool, min(n, len(pool))):
            toks, t = anonymize(src, name, rng)
            code = pretty(toks)
            verdict = ptrgate.analyze_source(" ".join(toks))
            gate = verdict.get(t, ("unsupported", "not found"))
            facts = ptrgate.facts_text(gate)
            items.append(dict(id=len(items), group=key[0], variant=var, cwe=cwe, label=y, target=t, code=code, gate=gate[0], facts=facts))
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
    return items


def messages(item, arm):
    q = QUESTION.format(t=item["target"])
    u = "```c\n" + item["code"] + "\n```\n\n" + q
    if arm == "X":
        u += HOST.format(t=item["target"], facts=item["facts"])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def parse(text):
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get("memory_error")
    except Exception:
        v = None
    if isinstance(v, str):
        v = {"true": True, "false": False}.get(v.strip().lower())
    return v if isinstance(v, bool) else None


def run(llm, items, path, arms=("P", "X"), progress=print):
    """Ask every question in every arm; rows are saved as they come (resumable)."""
    try:
        rows = json.load(open(path))["rows"]
    except Exception:
        rows = []
    done = {(r["id"], r["arm"]) for r in rows}
    for it in items:
        for arm in arms:
            if (it["id"], arm) in done:
                continue
            text = llm.generate(messages(it, arm), False, 400)[0]   # generate gives (text, tokens, seconds)
            rec = llm.cache.get(llm.key(messages(it, arm))) or {}
            rows.append(dict(id=it["id"], arm=arm, reply=text, pred=parse(text), prompt=rec.get("prompt_tokens"),
                             out=rec.get("tokens"), think=rec.get("think"), seconds=rec.get("seconds")))
            json.dump({"rows": rows}, open(path, "w"))
        if it["id"] % 10 == 0:
            sp = llm.spent_usd() if hasattr(llm, "spent_usd") else None
            progress("item %d/%d | spent %s" % (it["id"] + 1, len(items), ("$%.2f" % sp) if sp is not None else "-"))
    return rows


def summarize(items, rows):
    by = {it["id"]: it for it in items}
    out = {}
    for arm in ("P", "X"):
        rs = [r for r in rows if r["arm"] == arm]
        t = dict(n=len(rs), parsed=sum(r["pred"] is not None for r in rs))
        for name, pick in (("all", lambda it: True), ("hard", lambda it: it["group"] == "hard"), ("rest", lambda it: it["group"] == "rest"),
                           ("cross", lambda it: it["variant"] in ("45", "54", "61", "62", "63", "64", "65", "66", "67", "68"))):
            sel = [(by[r["id"]], r["pred"]) for r in rs if pick(by[r["id"]])]
            n = len(sel)
            right = sum(p is not None and p == bool(it["label"]) for it, p in sel)
            pos = [(it, p) for it, p in sel if it["label"]]
            neg = [(it, p) for it, p in sel if not it["label"]]
            t[name] = dict(n=n, accuracy=round(right / n, 4) if n else None,
                           recall=round(sum(p is True for it, p in pos) / len(pos), 4) if pos else None,
                           false_alarm=round(sum(p is True for it, p in neg) / len(neg), 4) if neg else None)
        out[arm] = t
    return out

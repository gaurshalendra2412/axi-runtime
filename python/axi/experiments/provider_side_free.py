"""The serving side a seventh time: requests in ordinary words (docs/PROVIDER_SIDE_FREE_TEST.md; concept-map item 113).

Every serving run so far wrote the request with coordinates ("Delete the service at (2,1)."), and the host read the coordinates from that sentence to build the reply grammar. The sixth run
(item 112) found that the grammar that names the coordinates held the state at all 6,400 steps, and also that this is partly the host's reading, not the model's. This test changes the
request: every service has a NAME and the request says "Remove the billing service." The jobs, the oracle's replies and the engine are the sixth run's (jobs 192 to 255, 100 steps), so the
cells can be put next to the sixth run's `L0` and `LN` job by job.

Three cells, no retry:
  `C0` the request as before, with coordinates; the prompt carries the names (the same lines as the other two cells), the grammar is the capped expression. It separates the effect of the
      added lines from the effect of the words.
  `W0` the request in words; the capped expression; the model must find the coordinates of the service in the names it is shown.
  `WN` the request in words; the HOST reads it (`engine/grounding.py`, unchanged, plus the small reader below) and the grammar names what the host read. A request the host cannot read
      (ambiguous, unresolved, not one of the three kinds) gets the capped expression, so that nothing is refused and the model must read it as in `W0`.

What is new here is only: the names and the wordings (`name_tables`, `wording`), the reader (`read_request`), a session that shows the names and the words (`FreeSession`), and the page's 16
reading lines (`verdict_free`). The pinned files of the earlier runs and `provider_side_long.py` are imported and edited by nothing. The host reader is NOT language understanding: it finds the
names of the table in the sentence (`grounding.ground`), takes the kind from the numbers the sentence carries and refuses what is not clear; the page measures how often it does so."""
import hashlib, json, os, random, re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
from axi.engine.grounding import COORD, NameIndex, ground, normalize
from axi.experiments.agent_loop import SYSTEM, RULES_HINT, Task, _serialize, copy_graph
from axi.experiments.long_horizon import oracle_script
from axi.experiments.word_tasks import POOL, ALIASES

# ---------------------------------------------------------------- the settings written on the page
MODEL, SEED, POLICY, HINT_RULES, MAX_NEW, CAPACITY_TOKENS = L.MODEL, L.SEED, L.POLICY, L.HINT_RULES, L.MAX_NEW, L.CAPACITY_TOKENS
REGEX, REGEX_SHA256, NAME_CAP = L.REGEX, L.REGEX_SHA256, L.NAME_CAP
STEPS, FIRST, K_JOBS = L.STEPS, L.FIRST, 64
PINS = L.PINS
TYPED_MODULE_SHA256, LONG_MODULE_SHA256 = L.TYPED_MODULE_SHA256, "ec38d450257bf3114d898baffede3f9bbab33013f55c1b903340113fedb4c967"
GROUNDING_SHA256 = "2e5a7379df25eeb4762b126d775e3c430babfeeeef630e030e7ef4f7b7acc555"          # engine/grounding.py, imported and edited by nothing
WORD_TASKS_SHA256 = "8f846d706358667385316aaffd4a176b31b8abf48ec1ae1255e45ea933c6928e"          # experiments/word_tasks.py (the pool of names and the aliases), edited by nothing
GRAMMAR_SHA = {"C0": REGEX_SHA256, "W0": REGEX_SHA256, "WN": "f8854dd780c79290e1807ce655770de597068290c1ab2f4f534aa36a34056c60"}      # the sha256 of the sorted set of expressions of a cell
READER_COUNTS = {"right": 5380, "wrong": 0, "miss": 1020}                                         # what the host reader makes of the 6,400 requests (computed without a model, before the page was locked)

# label: (jobs side by side, first job, arm, mode, shape). mode `coords`: the request as in the earlier runs; `words`: the request in ordinary words.
# shape `capped`: the third run's expression; `read`: the expression the host builds from what it read of the request (the capped one where it read nothing).
CELLS: Dict[str, Tuple[int, int, str, str, str]] = {
    "C0": (K_JOBS, FIRST, "S", "coords", "capped"),
    "W0": (K_JOBS, FIRST, "S", "words", "capped"),
    "WN": (K_JOBS, FIRST, "S", "words", "read"),
}
ORDER = ["C0", "W0", "WN"]

NOTE = """
Services also have names, listed under Names. A request may use a name instead of coordinates; your reply must always use coordinates.
  remove the service called "demo", which sits at (2,1) -> DEL[2,1]"""
B_SHARE = 0.25                        # the share of requests written with the second pool of wordings (another service is mentioned in the sentence)
ALIAS_SHARE = 0.2                     # the share of mentions of a service that has an alias which use the alias


# ---------------------------------------------------------------- the names: the host's table, as the oracle's graph changes
def _surface(rng: random.Random, name: str, aliases: List[str]) -> str:
    return rng.choice(aliases) if aliases and rng.random() < ALIAS_SHARE else name


@lru_cache(maxsize=None)
def name_tables(seed: int, steps: int = STEPS):
    """For a job: (tables, new_names). tables[t] is {coord: name} for the oracle's graph before request t+1 (t from 0); new_names[t] is the name of the service an add_node request at step t+1 creates
    (None for the other kinds). A service keeps its name while it exists; a new service takes the next unused name of a shuffled pool; the host's table is the oracle's graph's, never the held state's."""
    g0, script, oracle = oracle_script(seed, steps); rng = random.Random(f"names{seed}")
    pool = list(POOL); rng.shuffle(pool); free = list(pool); tables, new_names = [], []
    table = {}
    for n in sorted(oracle[0].nodes): table[n] = free.pop(0)
    for t, st in enumerate(script):
        tables.append(dict(table)); nxt = None
        if st["kind"] == "add_node":
            nxt = free.pop(0); node = T.parse_request(st["instruction"])["node"]; table[(int(node[0]), int(node[1]))] = nxt
        elif st["kind"] in ("delete_dep", "delete_leaf"):
            node = T.parse_request(st["instruction"])["node"]; gone = table.pop((int(node[0]), int(node[1]))); free.append(gone)
        new_names.append(nxt)
    return tables, new_names


def aliases_of(table: Dict[Tuple[int, int], str]) -> Dict[Tuple[int, int], List[str]]:
    return {c: list(ALIASES[n]) for c, n in table.items() if n in ALIASES}


# ---------------------------------------------------------------- the wordings (written once, before the reader was run on them)
A_DELETE = ["Remove the {n} service.", "Decommission the {n} service.", "Please retire the {n} service.", "Take the {n} service out.", "Drop {n} from the graph.", "We no longer need the {n} service, so delete it."]
B_DELETE = ["Remove the {n} service; the {m} service stays.", "The {m} service is fine, but retire the {n} service."]
A_NODE = ["Add a new service called {x} at ({r},{c}) with value {v}.", "Create a service named {x} at ({r},{c}), value {v}.", "Bring up {x} at ({r},{c}) with value {v}.", "New service {x}: place it at ({r},{c}) with value {v}."]
B_NODE = ["Add {x} at ({r},{c}) with value {v}; it will sit next to the {m} service."]
A_EDGE = ["Add a dependency edge from the {a} service to the {b} service with relation {rel} and weight {w}.", "Connect the {a} service to the {b} service: relation {rel}, weight {w}.",
          "Link {a} to {b} using relation {rel} and weight {w}.", "Please add an edge from {a} to {b}. Relation {rel}, weight {w}.", "Wire {a} -> {b} (relation {rel}, weight {w})."]
B_EDGE = ["Add a dependency edge from the {a} service to the {b} service with relation {rel} and weight {w}; the {m} service is not involved.",
          "The {m} service is unaffected, but connect {a} to {b} with relation {rel} and weight {w}."]


def wording(seed: int, t: int, instruction: str, table: Dict[Tuple[int, int], str], new_name: Optional[str]):
    """The request of step t+1 (t from 0) in ordinary words, from the coordinate request of the earlier runs: (sentence, pool 'A' or 'B'). Deterministic in (seed, t)."""
    rng = random.Random(f"words{seed}:{t}"); p = T.parse_request(instruction); al = aliases_of(table); pool = "B" if rng.random() < B_SHARE else "A"
    name = lambda c: _surface(rng, table[(int(c[0]), int(c[1]))], al.get((int(c[0]), int(c[1])), []))
    others = lambda used: [c for c in sorted(table) if c not in used]
    if p["kind"] == "delete":
        n = (int(p["node"][0]), int(p["node"][1])); oth = others({n})
        if pool == "B" and not oth: pool = "A"
        tpl = rng.choice(B_DELETE if pool == "B" else A_DELETE)
        return tpl.format(n=name(n), m=name(rng.choice(oth)) if pool == "B" else ""), pool
    if p["kind"] == "add_node":
        n = (int(p["node"][0]), int(p["node"][1])); oth = others(set())
        if pool == "B" and not oth: pool = "A"
        tpl = rng.choice(B_NODE if pool == "B" else A_NODE)
        return tpl.format(x=new_name, r=n[0], c=n[1], v=p["value"], m=name(rng.choice(oth)) if pool == "B" else ""), pool
    a, b = (int(p["src"][0]), int(p["src"][1])), (int(p["dst"][0]), int(p["dst"][1])); oth = others({a, b})
    if pool == "B" and not oth: pool = "A"
    tpl = rng.choice(B_EDGE if pool == "B" else A_EDGE)
    return tpl.format(a=name(a), b=name(b), rel=p["rel"], w=p["weight"], m=name(rng.choice(oth)) if pool == "B" else ""), pool


# ---------------------------------------------------------------- the host reads the request in words
_VALUE = re.compile(r"\bvalue\s+(-?[0-9]+)", re.IGNORECASE)
_WEIGHT = re.compile(r"\bweight\s+(-?[0-9]+)", re.IGNORECASE)
_REL = re.compile(r"\brelation\s+([A-Za-z0-9_]+)", re.IGNORECASE)


def _first_position(norm_tokens: List[str], phrase: str) -> int:
    pt = phrase.split(); n = len(pt)
    for i in range(len(norm_tokens) - n + 1):
        if norm_tokens[i:i + n] == pt: return i
    return 10 ** 6


def read_request(text: str, table: Dict[Tuple[int, int], str]):
    """What the host reads of a request in words, as the coordinate request it stands for: (instruction in the earlier runs' sentence, None) or (None, reason). The services are found by
    `grounding.ground` against the table (names, aliases, whole words); the kind comes from the numbers the sentence carries: `relation` and `weight` for an edge, one coordinate pair and
    `value` for an added service, none of the three for a delete. A sentence that names two or more services where one is wanted, a name that fits several, a service that is not in the table,
    or a sentence that is not one of the three kinds is NOT read: the reason is returned and the host applies the capped expression."""
    idx = NameIndex(table, aliases_of(table)); g = ground(text, idx); rel, w, v = _REL.search(text), _WEIGHT.search(text), _VALUE.search(text); coords = COORD.findall(text)
    if g.ambiguous: return None, "ambiguous"
    if rel and w and not v:
        svc = [m for m in g.mentions if m.how in ("name", "partial")]
        if len(svc) != 2 or len(g.named) != 2: return None, "edge: not two services"
        toks = normalize(text).split(); svc.sort(key=lambda m: _first_position(toks, m.phrase)); (a,), (b,) = [tuple(m.coords) for m in svc]
        return f"Add a dependency edge from ({a[0]},{a[1]}) to ({b[0]},{b[1]}) with relation {rel[1]} and weight {w[1]}.", None
    if v and not rel and not w:
        if len(coords) != 1: return None, "node: not one pair of coordinates"
        return f"Add a service at ({int(coords[0][0])},{int(coords[0][1])}) with value {v[1]}.", None
    if not (rel or w or v):
        if g.status != "bound" or len(g.named) != 1: return None, "delete: " + ("not one service" if g.status == "bound" else g.status)
        (a,) = tuple(g.named); return f"Delete the service at ({a[0]},{a[1]}).", None
    return None, "kind not clear"


def reader_outcome(truth: str, read: Optional[str]) -> str:
    """`right` (the host read the request the oracle made), `wrong` (it read another one), `miss` (it read nothing)."""
    if read is None: return "miss"
    return "right" if T.parse_request(read) == T.parse_request(truth) else "wrong"


@lru_cache(maxsize=None)
def requests_of(seed: int, steps: int = STEPS):
    """For a job: [(coordinate instruction, words, pool, table, aliases, read instruction or None, reason, outcome)] for every step."""
    tables, news = name_tables(seed, steps); sc = T.script_of(seed, steps); out = []
    for t, st in enumerate(sc):
        w, pool = wording(seed, t, st["instruction"], tables[t], news[t]); rd, why = read_request(w, tables[t])
        out.append((st["instruction"], w, pool, tables[t], rd, why, reader_outcome(st["instruction"], rd)))
    return out


# ---------------------------------------------------------------- the session: the names, the words, and the expression the host builds
def legend(table: Dict[Tuple[int, int], str]) -> str:
    al = aliases_of(table)
    return "; ".join(f"{table[c]}{' (also ' + ', '.join(al[c]) + ')' if c in al else ''} = ({c[0]},{c[1]})" for c in sorted(table))


class FreeSession(PS.Session):
    """The exact-state job of `provider_side.Session` (it overrides nothing that decides a step) with the names of the host's table in the prompt, the request in words or with coordinates, and the
    expression of its cell: the capped one, or the one the host builds from what it read of the request (the capped one where it read nothing)."""
    def __init__(self, seed: int, steps: int = STEPS, mode: str = "words", shape: str = "capped"):
        assert mode in ("coords", "words") and shape in ("capped", "read"), (mode, shape)
        super().__init__("S", seed, steps, HINT_RULES); self.mode, self.shape = mode, shape; self._rx = REGEX; self._meta: dict = {}
        self.tables, self.news = name_tables(seed, steps)

    def request(self) -> T.TRequest:
        st = self.script[self.t]; table = self.tables[self.t]; words, pool = wording(self.seed, self.t, st["instruction"], table, self.news[self.t])
        instr = words if self.mode == "words" else st["instruction"]; read = why = outcome = None
        if self.shape == "read": read, why = read_request(words, table); outcome = reader_outcome(st["instruction"], read)
        self._rx = REGEX if read is None else T.regex_for("named", read)
        self._meta = dict(request=instr, pool=pool if self.mode == "words" else None, reader=outcome, why=why)
        msgs = [{"role": "system", "content": SYSTEM + NOTE}, {"role": "user", "content": f"{_serialize(copy_graph(self.live))}\n\nNames: {legend(table)}\n\nRequest: {instr}"}]
        return T.TRequest(msgs, st["ideal"], seed=self.seed, step=self.t + 1, prompt="old", shape="capped" if self._rx == REGEX else "named", regex=self._rx)

    def accept(self, rep: PS.Reply) -> None:
        super().accept(rep); self.log[-1]["in_language"] = re.fullmatch(self._rx, rep.text) is not None; self.log[-1].update(self._meta)


def make_sessions(label: str, steps: int = STEPS, seed: int = SEED) -> list:
    K, first, arm, mode, shape = CELLS[label]
    return [FreeSession(seed * 1000 + e, steps, mode, shape) for e in range(first, first + K)]


def expressions_of(label: str, steps: int = STEPS, seed: int = SEED) -> set:
    K, first, arm, mode, shape = CELLS[label]; out = set()
    for e in range(first, first + K):
        for ins, w, pool, tab, rd, why, oc in requests_of(seed * 1000 + e, steps): out.add(REGEX if shape == "capped" or rd is None else T.regex_for("named", rd))
    return out


def grammar_sha256(label: str, steps: int = STEPS, seed: int = SEED) -> str:
    return hashlib.sha256("\n".join(sorted(expressions_of(label, steps, seed))).encode()).hexdigest()


def run_cell(engine, label: str, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    K, first, arm, mode, shape = CELLS[label]
    r = L.run_sessions(engine, make_sessions(label, steps, seed), progress)
    return dict(label=label, arm=arm, K=K, first_episode=first, mode=mode, prompt="old", shape=shape, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(),
                grammar_sha256=grammar_sha256(label, steps, seed), **r)


# ---------------------------------------------------------------- summaries
def steps_of(c: dict, pred=None) -> list:
    return [x for s in c["sessions"] for x in s["log"] if pred is None or pred(x)]


def share_of(c: dict, pred=None) -> Optional[float]:
    xs = steps_of(c, pred); return sum(1 for x in xs if x["exact"]) / len(xs) if xs else None


def summarize_cell(c: dict) -> dict:
    s = L.summarize_cell(c); X = steps_of(c)
    s.update(mode=c["mode"], share_pool_a=share_of(c, lambda x: x.get("pool") == "A"), share_pool_b=share_of(c, lambda x: x.get("pool") == "B"),
             share_read=share_of(c, lambda x: x.get("reader") == "right"), share_unread=share_of(c, lambda x: x.get("reader") == "miss"),
             n_pool_b=sum(1 for x in X if x.get("pool") == "B"), reader={k: sum(1 for x in X if x.get("reader") == k) for k in ("right", "wrong", "miss")})
    return s


def summarize_all(cells: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cells:
        assert c["label"] not in out, f"cell {c['label']} twice"; out[c["label"]] = summarize_cell(c)
    return out


# ---------------------------------------------------------------- merge
def fingerprint() -> str:
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pinned_files() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(grounding=sha256_file(os.path.join(here, "..", "engine", "grounding.py")), word_tasks=sha256_file(os.path.join(here, "word_tasks.py")),
                provider_side_long=sha256_file(os.path.join(here, "provider_side_long.py")))


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses a cell twice, a cell that is not one of this test's, a cell whose settings are not those of its label, a cell of another length, another constraint, other
    settings than the first cell, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        lab = c.get("label")
        if lab not in CELLS: raise ValueError(f"cell {lab!r} is not one of this test's cells")
        if (c["K"], c["first_episode"], c["arm"], c["mode"], c["shape"]) != CELLS[lab]: raise ValueError(f"cell {lab} was not run with the settings of its label: {(c['K'], c['first_episode'], c['arm'], c['mode'], c['shape'])} against {CELLS[lab]}")
        if c["steps"] != STEPS: raise ValueError(f"cell {lab} has {c['steps']} steps, this test has {STEPS}")
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {lab} was not run with this test's constraint")
        if key(c) != key(ref): raise ValueError(f"cell {lab} was run under other settings: {key(c)} against {key(ref)}")
        if lab in seen: raise ValueError(f"cell {lab} twice")
        seen.add(lab)
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP,
                          cells={k: list(CELLS[k]) for k in ORDER}), cells=sorted(cells, key=lambda c: ORDER.index(c["label"])))


def merge_files(paths: List[str], out: str) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells)
    m["module_sha256"] = fingerprint(); m["typed_module_sha256"] = T.fingerprint(); m["pinned_sha256"] = T.pinned_hashes(); m["free_pinned_sha256"] = pinned_files()
    with open(out, "w") as f: json.dump(m, f)
    return m


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_FREE_TEST.md before the run)
LONG_FILE_SHA256 = "a0ad579961d006e9d561ff65cb301de77dd25dd85ebb348827201ff0fdaeba2b"      # docs/results/axi_provider_long.json: the sixth run, whose L0 the control C0 is read against
L0_EXACT, L0_STEPS = 6244, 6400                                                              # the sixth run's L0 (capped expression, coordinates, same 64 jobs)
T_C0_TOL, T_W0, T_W0_COST, T_READ, T_WN, T_WN_READ, T_WN_UNREAD, T_POOL_B, T_TIME = 0.01, 0.95, 0.03, 0.995, 0.99, 0.995, 0.95, 0.03, 0.15
GPU_FREE_MIB = L.GPU_FREE_MIB

FL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens, and every reply is in the language of its own request"),
    ("guard", "the requests in words say what the oracle's request says: in every sentence of W0 and WN the name (or an alias) of each service the oracle's request is about is written, in the order of the oracle's request, the numbers are the oracle's, and a sentence of the second pool names exactly one service more (checked by plain search, not by the reader)"),
    ("guard", "the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs and the reader, and the oracle's own reply is inside the expression of every request"),
    ("guard", "every job of every cell ran all 100 steps; none ended early"),
    ("guard", "what the reader read, as recorded in WN, is what the reader gives on the same sentences when it is run again: the same outcome (right, wrong, miss) for every one of the 6,400 requests, and not one request read wrong"),
    ("claim", "the lines that tell the model the names do not hurt: the exact share of C0 is at most one point below that of the sixth run's L0 on the same 64 jobs (6,244 of 6,400)"),
    ("claim", "the model reads the names: the exact share of W0 over 100 steps is at least 95 percent"),
    ("claim", "the words cost little: the exact share of W0 is at most three points below that of C0"),
    ("claim", "the host's reading does not hurt: the exact share of WN is at least that of W0"),
    ("claim", "with the host reading what it can, the exact share of WN over 100 steps is at least 99 percent"),
    ("claim", "on the steps whose request the host read (right), the exact share of WN is at least 99.5 percent"),
    ("claim", "on the steps whose request the host did not read (miss, the model reads it as in W0), the exact share of WN is at least 95 percent"),
    ("claim", "the second pool of wordings (another service mentioned in the sentence) does not confuse the model more than three points: in W0 the exact share on the steps of that pool is at least that on the steps of the first pool minus three points"),
    ("claim", "the seconds per step of WN are at most 15 percent above those of W0 (both in this session)"),
    ("claim", "the exact steps per GPU hour of WN are higher than those of W0 (both in this session)"),
]
FL_HEADLINE = (8, 11)


def _plain_names(table: Dict[Tuple[int, int], str]) -> Dict[Tuple[int, int], List[str]]:
    al = aliases_of(table); return {c: [n] + al.get(c, []) for c, n in table.items()}


def _plain_hits(text: str, forms: Dict[Tuple[int, int], List[str]]) -> List[Tuple[int, Tuple[int, int]]]:
    """(position, coordinate) of every table name or alias written in the text, by plain case-insensitive search of whole words (no use of the reader)."""
    hits = []
    for c, fs in forms.items():
        for f in fs:
            for m in re.finditer(r"(?<![A-Za-z])" + re.escape(f) + r"(?![A-Za-z])", text, re.IGNORECASE): hits.append((m.start(), c))
    return sorted(hits)


def wording_ok(instruction: str, words: str, pool: str, table: Dict[Tuple[int, int], str], new_name: Optional[str]) -> bool:
    p = T.parse_request(instruction); forms = _plain_names(table); hits = _plain_hits(words, forms); nums = re.findall(r"-?[0-9]+", words)
    if p["kind"] == "delete":
        n = (int(p["node"][0]), int(p["node"][1])); want = [n]; extra = 1 if pool == "B" else 0; ok_nums = nums == []
    elif p["kind"] == "add_node":
        want = []; extra = 1 if pool == "B" else 0; ok_nums = nums == [p["node"][0], p["node"][1], p["value"]] and new_name is not None and re.search(r"(?<![A-Za-z])" + re.escape(new_name) + r"(?![A-Za-z])", words, re.IGNORECASE) is not None
    else:
        want = [(int(p["src"][0]), int(p["src"][1])), (int(p["dst"][0]), int(p["dst"][1]))]; extra = 1 if pool == "B" else 0
        ok_nums = nums == [p["weight"]] and re.search(r"relation\s+" + re.escape(p["rel"]) + r"\b", words, re.IGNORECASE) is not None
    others = [c for _, c in hits if c not in want]
    if len(others) != extra or len(set(others)) != extra: return False
    if p["kind"] == "add_edge" and (not all(any(c == w for _, c in hits) for w in want) or [c for _, c in hits if c in want][:2] != want): return False
    if p["kind"] == "delete" and not any(c == want[0] for _, c in hits): return False
    return ok_nums


def verdict_free(cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). A line that needs a cell that was not run is None, never 'no'."""
    new = {c["label"]: c for c in cells if any(s["log"] for s in c["sessions"])}; Sn = {k: summarize_cell(c) for k, c in new.items()}
    has = lambda *ks: all(k in Sn for k in ks); out: List[Optional[bool]] = []
    allrep = [r for k in new for r in T.replies_of(new[k])]
    out.append(all(r[2] is not None for r in allrep) if allrep else None)                                                                                                                 # 1
    out.append((sum(1 for r in allrep if not r[0]) <= (1 - T.T_PARSED) * len(allrep) + 1e-9 and all(r[1] < MAX_NEW for r in allrep) and all(r[3] for r in allrep)) if allrep else None)           # 2
    def words_ok():
        for k in ("W0", "WN"):
            if k not in new: continue
            for s in new[k]["sessions"]:
                tables, news = name_tables(s["seed"], STEPS); sc = T.script_of(s["seed"], STEPS)
                for x, st, tb, nn in zip(s["log"], sc, tables, news):
                    if not wording_ok(st["instruction"], x["request"], x["pool"], tb, nn): return False
        return True
    out.append(words_ok() if ("W0" in new or "WN" in new) else None)                                                                                                                       # 3
    def expressions_ok():
        for k, c in new.items():
            if not (c["grammar_sha256"] == grammar_sha256(k, c["steps"], c["seed"]) == GRAMMAR_SHA[k]): return False
        for k, c in new.items():
            for s in c["sessions"]:
                for st, (ins, w, pool, tb, rd, why, oc) in zip(T.script_of(s["seed"], STEPS), requests_of(s["seed"], STEPS)):
                    rx = REGEX if CELLS[k][4] == "capped" or rd is None else T.regex_for("named", rd)
                    if re.fullmatch(rx, st["ideal"]) is None: return False
        return True
    out.append(expressions_ok() if new else None)                                                                                                                                          # 4
    s_cells = list(new)
    out.append(all(len(s["log"]) == STEPS and s["stopped"] is None for k in s_cells for s in new[k]["sessions"]) if s_cells else None)                                                      # 5
    def reader_again():
        if "WN" not in new: return None
        for s in new["WN"]["sessions"]:
            tables, news = name_tables(s["seed"], STEPS); sc = T.script_of(s["seed"], STEPS)
            for i, (x, st) in enumerate(zip(s["log"], sc)):
                w, pool = wording(s["seed"], i, st["instruction"], tables[i], news[i]); rd, why = read_request(w, tables[i]); oc = reader_outcome(st["instruction"], rd)
                if x["reader"] != oc or w != x["request"] or oc == "wrong": return False
        return True
    out.append(reader_again())                                                                                                                                                             # 6
    out.append(None if not has("C0") else _ge(Sn["C0"]["exact_share"], L0_EXACT / L0_STEPS - T_C0_TOL))                                                                                      # 7
    out.append(None if not has("W0") else _ge(Sn["W0"]["exact_share"], T_W0))                                                                                                                # 8
    out.append(None if not has("C0", "W0") else _ge(Sn["W0"]["exact_share"], Sn["C0"]["exact_share"] - T_W0_COST))                                                                           # 9
    out.append(None if not has("W0", "WN") else _ge(Sn["WN"]["exact_share"], Sn["W0"]["exact_share"]))                                                                                       # 10
    out.append(None if not has("WN") else _ge(Sn["WN"]["exact_share"], T_WN))                                                                                                                # 11
    out.append(None if not has("WN") or Sn["WN"]["share_read"] is None else _ge(Sn["WN"]["share_read"], T_WN_READ))                                                                           # 12
    out.append(None if not has("WN") or Sn["WN"]["share_unread"] is None else _ge(Sn["WN"]["share_unread"], T_WN_UNREAD))                                                                    # 13
    out.append(None if not has("W0") or Sn["W0"]["share_pool_a"] is None or Sn["W0"]["share_pool_b"] is None else _ge(Sn["W0"]["share_pool_b"], Sn["W0"]["share_pool_a"] - T_POOL_B))      # 14
    out.append(None if not has("W0", "WN") else _le(Sn["WN"]["seconds_per_step"], Sn["W0"]["seconds_per_step"] * (1 + T_TIME)))                                                              # 15
    out.append(None if not has("W0", "WN") else Sn["WN"]["exact_per_hour"] > Sn["W0"]["exact_per_hour"])                                                                                     # 16
    assert len(out) == len(FL_LINES)
    return [(i + 1, FL_LINES[i][0], FL_LINES[i][1], out[i]) for i in range(len(FL_LINES))]


from axi.experiments.provider_side import _ge, _le
MAX_NEW = L.MAX_NEW


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim: with lines 2 to 6 yes the figures of the cells are claimed as read. What the model does with names from lines 7 to 9, the reader from lines 10 to 14, a statement about
    time only from lines 15 and 16 (cells of one session). A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (2, 3, 4, 5, 6))
    pick = lambda ks: ({i: ok[i] for i in ks} if base else {i: None for i in ks})
    return dict(base=base, constraint_held=ok[2], words_say_what_the_oracle_says=ok[3], expressions_as_defined=ok[4], jobs_ran_all_steps=ok[5], reader_as_computed=ok[6], names_do_not_hurt=pick((7,)),
                the_model_reads_names=pick((8, 9, 14)), the_host_reads=pick((10, 11, 12, 13)), time_and_cost_in_one_session=pick((15, 16)))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[str, dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    print(f"\n{'cell':4s} {'K':>3s} {'jobs':>9s} {'mode':>6s} {'grammar':>7s} | {'steps':>6s} {'seconds':>8s} {'s/step':>7s} {'comp/step':>9s} | {'exact':>6s} {'1-30':>6s} {'67-100':>7s} {'jobs ok':>7s} {'corrupt':>7s} | {'outside':>7s} {'exact/h':>8s}")
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; K, first, arm, mode, shape = CELLS[lab]
        print(f"{lab:4s} {K:3d} {str(first) + '-' + str(first + K - 1):>9s} {mode:>6s} {shape:>7s} | {s['steps_done']:6d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 3):>7s} {f(s['computed_per_step'], 0):>9s} | "
              f"{pc(s['exact_share']):>6s} {pc(s['first30']):>6s} {pc(s['late']):>7s} {s['jobs_exact_to_end']:7d} {s['corrupted_steps']:7d} | {s['out_of_language']:7d} {f(s['exact_per_hour'], 0):>8s}")


def print_split(S: Dict[str, dict]) -> None:
    """The exact share by pool of wordings and by what the host read (descriptive; lines 12 to 14 read these)."""
    pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; r = s["reader"]
        print(f"{lab}: exact on first pool {pc(s['share_pool_a'])}, on second pool {pc(s['share_pool_b'])} ({s['n_pool_b']} steps); host read right {r['right']}, wrong {r['wrong']}, not read {r['miss']}; "
              f"exact where read {pc(s['share_read'])}, where not read {pc(s['share_unread'])}; first slips of jobs in thirds {s['slips_by_third']}")


def print_misses(S: Dict[str, dict]) -> None:
    for lab in ORDER:
        if lab in S and S[lab]["first_misses"]:
            print(f"first misses in {lab}: " + "; ".join(f"job {j} step {t} ({k}) {x!r}" for j, t, k, x in S[lab]["first_misses"]))


def print_verdict(checks, title="provider-side test, requests in words") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {FL_HEADLINE[0]} and {FL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in FL_HEADLINE) + ")")
    a = claims_allowed(checks)
    if not a["base"]: print("  rule on the page: line 2, 3, 4, 5 or 6 did not read yes, so NO claim is made from this run until the cause is found.")
    else:
        print("  rule on the page: lines 2 to 6 read yes, so the figures of the cells are claimed as read. A statement about time or cost comes only from lines 15 and 16 (cells of one session).")
        print("  never: a multiple of capacity, dollars, '100 percent' for a grammar, a statement about any provider, anything about a request in words other than the ones of this test, anything about a model that was not run.")


# ---------------------------------------------------------------- the pre-flight check: it can fail
gpu_state = L.gpu_state


def preflight(gpu=None, steps: int = STEPS, seed: int = SEED, long_file: Optional[str] = None) -> List[Tuple[str, bool, str]]:
    """Checks that cost seconds and save a Colab run. Every check can read NO; a check that cannot be made reads NO with the error, and the others are still made."""
    out: List[Tuple[str, bool, str]] = []
    def one(name, fn):
        try: ok, detail = fn()
        except Exception as ex: ok, detail = False, f"could not be checked: {type(ex).__name__}: {ex}"
        out.append((name, bool(ok), detail))
    jobs = sorted({e for K, first, *_ in CELLS.values() for e in range(first, first + K)})
    def jobs_ok():
        lens = {e: len(T.script_of(seed * 1000 + e, steps)) for e in jobs}
        return jobs == list(range(FIRST, FIRST + K_JOBS)) and all(n == 100 for n in lens.values()), f"jobs {jobs[0]} to {jobs[-1]} ({len(jobs)}), the sixth run's; steps per job {min(lens.values())} to {max(lens.values())}"
    one("the jobs are the sixth run's, 100 steps each", jobs_ok)
    def sixth():
        path = long_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "results", "axi_provider_long.json")
        h = sha256_file(path); return h == LONG_FILE_SHA256, f"docs/results/axi_provider_long.json sha256 {h[:16]}... (the control C0 is read against its L0)"
    one("the sixth run's file is the one this page names", sixth)
    def files():
        pins = T.pinned_hashes(); bad = sorted(k for k, v in PINS.items() if pins.get(k) != v); pf = pinned_files()
        okm = T.fingerprint() == TYPED_MODULE_SHA256 and pf["provider_side_long"] == LONG_MODULE_SHA256 and pf["grounding"] == GROUNDING_SHA256 and pf["word_tasks"] == WORD_TASKS_SHA256
        return not bad and okm, f"changed: {bad if bad else 'none'}; provider_side_typed.py, provider_side_long.py, grounding.py and word_tasks.py as pinned: {okm}"
    one("the files this test imports are the pinned ones", files)
    def expressions():
        shas = {lab: grammar_sha256(lab, steps, seed) for lab in CELLS}; badg = sorted(l for l in CELLS if shas[l] != GRAMMAR_SHA[l])
        return not badg, "; ".join(f"{l} {shas[l][:8]}" for l in ORDER) + (f"; wrong: {badg}" if badg else "")
    one("the expressions are the ones on the page", expressions)
    def oracle():
        n = outside = 0
        for lab, (K, first, _, mode, shape) in CELLS.items():
            for e in range(first, first + K):
                for st, (ins, w, pool, tb, rd, why, oc) in zip(T.script_of(seed * 1000 + e, steps), requests_of(seed * 1000 + e, steps)):
                    rx = REGEX if shape == "capped" or rd is None else T.regex_for("named", rd); n += 1; outside += 0 if re.fullmatch(rx, st["ideal"]) else 1
        return outside == 0, f"{n - outside} of {n} requests"
    one("the oracle's reply is inside the expression of every request", oracle)
    def reader():
        c = {"right": 0, "wrong": 0, "miss": 0}; badw = 0
        for e in jobs:
            for st, (ins, w, pool, tb, rd, why, oc) in zip(T.script_of(seed * 1000 + e, steps), requests_of(seed * 1000 + e, steps)): c[oc] += 1
        return c == READER_COUNTS, f"read right {c['right']}, wrong {c['wrong']}, not read {c['miss']} of {sum(c.values())} (on the page: {READER_COUNTS['right']} / {READER_COUNTS['wrong']} / {READER_COUNTS['miss']})"
    one("the host reader reads the 6,400 requests as the page says", reader)
    def sentences():
        bad = 0; n = 0
        for e in jobs:
            tables, news = name_tables(seed * 1000 + e, steps)
            for i, (st, (ins, w, pool, tb, rd, why, oc)) in enumerate(zip(T.script_of(seed * 1000 + e, steps), requests_of(seed * 1000 + e, steps))): n += 1; bad += 0 if wording_ok(st["instruction"], w, pool, tb, news[i]) else 1
        return bad == 0, f"{n - bad} of {n} sentences say what the oracle's request says"
    one("the requests in words say what the oracle's request says", sentences)
    def first_request(lab):
        s = make_sessions(lab, steps, seed)[0]; rq = s.request(); user = rq.messages[-1]["content"]
        ok = "Names: " in user and "Request: " in user and re.fullmatch(rq.regex, s.script[0]["ideal"]) is not None and len(rq.messages) == 2 and rq.messages[0]["role"] == "system" and NOTE in rq.messages[0]["content"]
        return ok, f"user message {len(user)} characters, expression {len(rq.regex)} characters, job {s.seed - seed * 1000}"
    for lab in ORDER: one(f"cell {lab}: its first request is built", lambda lab=lab: first_request(lab))
    def card():
        g = gpu_state() if gpu is None else gpu()
        if g is None: return False, "nvidia-smi could not be read (is the runtime a GPU runtime?)"
        return "T4" in g["name"] and g["used_mib"] <= GPU_FREE_MIB, f"{g['name']}, {g['used_mib']} of {g['total_mib']} MiB in use (at most {GPU_FREE_MIB} before the engine is loaded)"
    one("the GPU is a free T4", card)
    return out


print_preflight = L.print_preflight


# ---------------------------------------------------------------- the stand-in run (harness only)
def dry_run(steps: int = STEPS, labels=None) -> Dict[str, dict]:
    eng = T.TypedStandIn(capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for lab in ORDER:
        if labels is None or lab in labels: cells.append(run_cell(eng, lab, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache; it answers with the oracle's own reply), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, requests in words (docs/PROVIDER_SIDE_FREE_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--preflight", action="store_true", help="run the pre-flight check (reads nvidia-smi)")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_free.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.preflight: raise SystemExit(0 if print_preflight(preflight()) else 1)
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_split(S); print_misses(S); print_verdict(verdict_free(m["cells"])); print("\nwritten to", a.out)
    else: ap.print_help()

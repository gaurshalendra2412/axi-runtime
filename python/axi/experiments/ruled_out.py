"""The ruled-out-causes test (page: docs/RULED_OUT_TEST.md, written and hash-locked BEFORE any model call).

A person who runs long coding agents wrote that three things are easy to turn into host-side rules (which files may be touched, which destructive operations are blocked, which dependants a change
breaks) and two are not: the hypotheses already ruled out, and behaviour that spans several files. This test is about the first of those two. It asks whether a list of ruled-out causes, held by the
host as exact state, keeps a model from proposing a dead cause again, against the same facts kept in the chat, and against the chat compacted into a summary (the way an agent tool shortens a long chat).

The world is invented and small: 30 parts, each can fail in 8 ways, one cause (a part and a way) is the real one. Each seed has its own order and numbering of the parts and the ways (`perms`), so the three
episodes of an arm show the model different prompts. Each turn the model names ONE cause to check; the host says NOT or FOUND.
  keys mode   the model writes `CHECK[part,fault]`; the host reads it with a strict parser
  words mode  the model writes one plain sentence ("The billing service times out."); the host reads the words against a table of part names and failure names, like `grounding.ground` does for requests
Arms (the same facts, kept in different places):
  kept         the whole chat is sent again every turn
  compacted    the same chat, but every COMPACT_EVERY checks the model is asked to summarise it and the summary replaces it (the request and the system text are those of docs/COMPACTION_TEST.md)
  held_prompt  the host holds the ruled-out list; every prompt is rebuilt from it (no history); no gate
  held_gate    the same prompt, and a gate: a cause already on the list is refused (nothing is checked; the next prompt carries one line saying so)
A "repeat" is a reply whose cause is already on the ruled-out list. In kept, compacted and held_prompt the host answers a repeat (a wasted check); in held_gate it is refused.
The state of an episode is a deterministic function of the saved replies, so a run that is cut off and started again equals the run that was not (`run_arm`, `replay`).
This file calls no model by itself. It edits no file of the framework; it imports only the compaction texts of `gemini_compaction.py` and the file helpers of `resumable.py`."""
from __future__ import annotations
import json, os, random, re
from typing import Dict, List, Optional, Tuple

from axi.experiments.gemini_compaction import COMPACT_GENERIC, COMPACT_SYSTEM, SUMMARY_HEAD
from axi.experiments.resumable import _roundtrip, load_rows

# ---------------------------------------------------------------- the settings written on the page
PART_NAMES = ["auth", "billing", "cache", "checkout", "config", "cron", "database", "dns", "email", "export", "feature-flags", "gateway", "import", "inventory", "logging",
              "login", "metrics", "notifications", "orders", "payments", "queue", "reports", "search", "sessions", "shipping", "storage", "tax", "uploads", "users", "webhooks"]
FAULT_NAMES = ["timeout", "stale data", "error response", "out of memory", "bad credentials", "duplicate sends", "dropped messages", "slow response"]
FAULT_ALIASES = [["timeout", "times out", "timing out", "timed out", "time out"], ["stale data", "stale", "outdated data", "old data"], ["error response", "returns an error", "returns errors", "errors", "error"],
                 ["out of memory", "memory", "oom"], ["bad credentials", "wrong credentials", "invalid credentials", "credentials"], ["duplicate sends", "sends duplicates", "duplicates", "duplicate", "double send"],
                 ["dropped messages", "drops messages", "dropping messages", "lost messages", "drops"], ["slow response", "slowly", "slow", "high latency", "latency"]]
N_PARTS, N_FAULTS = len(PART_NAMES), len(FAULT_NAMES)
assert (N_PARTS, N_FAULTS) == (30, 8)
SEEDS = (101, 102, 103)                              # three episodes per arm (the hidden cause of each is fixed by its seed)
STEPS_KEYS, STEPS_WORDS = 60, 30                     # replies per episode (an episode ends earlier if the cause is found)
COMPACT_EVERY = 20                                   # checks between summaries in the compacted arm
WINDOW = 10                                          # replies per reading window
ARMS_KEYS = ("kept", "compacted", "held_prompt", "held_gate")
ARMS_WORDS = ("kept", "held_gate")
MODES = {"keys": dict(arms=ARMS_KEYS, steps=STEPS_KEYS), "words": dict(arms=ARMS_WORDS, steps=STEPS_WORDS)}

OPEN = "Begin. Name the first cause to check."
NEXT = "Next check?"


def hidden(seed: int) -> Tuple[int, int]:
    """The real cause of the episode with this seed, as (index in PART_NAMES, index in FAULT_NAMES)."""
    k = random.Random(seed).randrange(N_PARTS * N_FAULTS); return divmod(k, N_FAULTS)


def perms(seed: int) -> Tuple[List[int], List[int]]:
    """The numbering the model sees in this episode: parts[i] is the index in PART_NAMES of the part the model calls number i, faults[j] likewise. Each seed has its own order, so the three episodes
    of an arm show the model different prompts (a prompt asked twice is read from the file, so two episodes with the same prompts would be one sample twice)."""
    r = random.Random(seed * 7919 + 1); parts, faults = list(range(N_PARTS)), list(range(N_FAULTS)); r.shuffle(parts); r.shuffle(faults); return parts, faults


def hidden_numbers(seed: int) -> Tuple[int, int]:
    """The real cause of the episode as the numbers the model would write in keys mode."""
    parts, faults = perms(seed); p, f = hidden(seed); return parts.index(p), faults.index(f)


def system_text(mode: str, seed: int) -> str:
    parts, faults = perms(seed)
    if mode == "keys":
        ps = ", ".join(f"{i} {PART_NAMES[g]}" for i, g in enumerate(parts)); fs = ", ".join(f"{j} {FAULT_NAMES[g]}" for j, g in enumerate(faults))
        return (f"You help find the cause of an outage in a system of {N_PARTS} parts. The parts are: {ps}. Each part can fail in {N_FAULTS} ways: {fs}. "
                "Exactly one cause (one part failing in one way) is the real one. On each turn you name ONE cause to check, written exactly as CHECK[part,fault] with the two numbers, "
                "for example CHECK[3,5]. The answer is FOUND or NOT. Reply with the check only, nothing else.")
    ps = ", ".join(PART_NAMES[g] for g in parts); fs = ", ".join(FAULT_NAMES[g] for g in faults)
    return (f"You help find the cause of an outage in a system of {N_PARTS} parts. The parts are: {ps}. Each part can fail in {N_FAULTS} ways: {fs}. "
            "Exactly one cause (one part failing in one way) is the real one. On each turn you name ONE cause to check, in one short plain sentence that names one part and one way it fails, "
            "for example: The billing service times out. The answer is FOUND or NOT. Reply with that sentence only.")


# ---------------------------------------------------------------- reading a reply
_CHECK = re.compile(r"CHECK\[(\d+),(\d+)\]")


def parse_key(text: str) -> Optional[Tuple[int, int]]:
    """Strict: the whole reply is exactly one CHECK[part,fault] with both numbers in range."""
    m = _CHECK.fullmatch((text or "").strip())
    if not m: return None
    p, f = int(m.group(1)), int(m.group(2))
    return (p, f) if p < N_PARTS and f < N_FAULTS else None


def _norm(text: str) -> str: return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()) + " "


def _has(t: str, phrase: str) -> bool:
    ph = _norm(phrase)
    return ph in t or (ph.rstrip().endswith("s") and ph.rstrip()[:-1] + " " in t and len(ph.strip()) > 4)


def ground(text: str) -> Optional[Tuple[int, int]]:
    """Words: the cause is found when the reply names exactly one part and exactly one failure way (any listed alias). Two parts, two ways, or none of either: None."""
    t = _norm(text)
    ps = {i for i, n in enumerate(PART_NAMES) if _has(t, n)}
    fs = {i for i, al in enumerate(FAULT_ALIASES) if any(_has(t, a) for a in al)}
    return (next(iter(ps)), next(iter(fs))) if len(ps) == 1 and len(fs) == 1 else None



# ---------------------------------------------------------------- one episode
class Episode:
    """The state of one episode and the prompts it needs. Everything is a function of the replies passed to `apply_check` / `apply_summary`, in order."""
    def __init__(self, arm: str, mode: str, seed: int):
        if mode not in MODES or arm not in MODES[mode]["arms"]: raise ValueError(f"arm {arm!r} is not part of mode {mode!r}")
        self.arm, self.mode, self.seed, self.steps = arm, mode, seed, MODES[mode]["steps"]
        self.hidden = hidden(seed); self.ruled: Dict[Tuple[int, int], int] = {}; self.checks = 0; self.found: Optional[int] = None
        self.history: List[dict] = [dict(role="user", content=OPEN)] if arm in ("kept", "compacted") else []
        self.since = 0; self.compactions = 0; self.notice: Optional[str] = None; self.system = system_text(mode, seed)
        self.parts, self.faults = perms(seed); self.num_p = {g: i for i, g in enumerate(self.parts)}; self.num_f = {g: j for j, g in enumerate(self.faults)}

    def read(self, text: str) -> Optional[Tuple[int, int]]:
        """The cause a reply names, as (index in PART_NAMES, index in FAULT_NAMES), or None. Keys mode reads the numbers of this episode's numbering."""
        if self.mode == "words": return ground(text)
        k = parse_key(text); return (self.parts[k[0]], self.faults[k[1]]) if k else None

    def cause_text(self, key: Tuple[int, int]) -> str:
        p, f = key
        return f"[{self.num_p[p]},{self.num_f[f]}]" if self.mode == "keys" else f"{PART_NAMES[p]} / {FAULT_NAMES[f]}"

    def done(self) -> bool: return self.found is not None or self.checks >= self.steps

    # --- the prompts
    def render_held(self) -> str:
        if not self.ruled: head = "Nothing is ruled out yet."
        else:
            by: Dict[int, List[int]] = {}
            if self.mode == "keys":
                for i, j in sorted((self.num_p[p], self.num_f[f]) for p, f in self.ruled): by.setdefault(i, []).append(j)
                lines = [f"  part {i} ({PART_NAMES[self.parts[i]]}): faults {', '.join(map(str, js))}" for i, js in by.items()]
            else:
                for p, f in sorted(self.ruled): by.setdefault(p, []).append(f)
                lines = [f"  {PART_NAMES[p]}: {'; '.join(FAULT_NAMES[f] for f in fs)}" for p, fs in by.items()]
            head = f"Ruled out so far ({len(self.ruled)} causes, none of them is the real one):\n" + "\n".join(lines)
        return head + "\nName ONE cause that is not in this list." + (("\n" + self.notice) if self.notice else "")

    def next_call(self) -> Tuple[str, List[dict]]:
        """('summary', messages) when the compacted arm is due, else ('check', messages)."""
        if self.arm == "compacted" and self.since >= COMPACT_EVERY and self.history:
            return "summary", [dict(role="system", content=COMPACT_SYSTEM)] + self.history + [dict(role="user", content=COMPACT_GENERIC)]
        if self.arm in ("kept", "compacted"): return "check", [dict(role="system", content=self.system)] + self.history
        return "check", [dict(role="system", content=self.system), dict(role="user", content=self.render_held())]

    # --- the replies
    def apply_check(self, text: str) -> dict:
        """Apply one reply. Returns the flags saved with the row: status (ok / unparsed / unclear), key, repeat, refused, found."""
        key = self.read(text); self.checks += 1; self.since += 1; repeat = refused = False; self.notice = None
        status = "ok" if key else ("unparsed" if self.mode == "keys" else "unclear")
        if key is None:
            result = ("Invalid reply. Write exactly CHECK[part,fault]." if self.mode == "keys" else "I could not tell which one part and which one failure you meant. Name one part and one way it fails.") + " " + NEXT
            self.notice = ("Your last reply was not a valid check. Write exactly CHECK[part,fault]." if self.mode == "keys" else "Your last reply did not name exactly one part and one way it fails.")
        elif key in self.ruled:
            repeat = True
            if self.arm == "held_gate":
                refused = True; result = ""; self.notice = f"Your last reply ({self.cause_text(key)}) was refused: it is already ruled out."
            else: result = f"NOT {self.cause_text(key)}. {NEXT}"
        elif key == self.hidden: self.found = self.checks; result = "FOUND"
        else: self.ruled[key] = self.checks; result = f"NOT {self.cause_text(key)}. {NEXT}"
        if self.arm in ("kept", "compacted"): self.history += [dict(role="assistant", content=text), dict(role="user", content=result)]
        return dict(status=status, key=list(key) if key else None, repeat=repeat, refused=refused, found=self.found is not None and self.found == self.checks)

    def apply_summary(self, text: str) -> dict:
        self.history = [dict(role="user", content=SUMMARY_HEAD + (text or "").strip() + "\n\n" + NEXT)]; self.since = 0; self.compactions += 1
        return dict(compaction=self.compactions)


# ---------------------------------------------------------------- the runner (one JSON line per finished call; cut off and started again equals not cut off)
def _call(llm, messages):
    text, ntok, sec = llm.generate(messages, constrained=False)
    return text, ntok, sec, getattr(llm, "last_prompt_tokens", None)


def replay(rows: List[dict], arm: str, mode: str, seed: int) -> Episode:
    """Rebuild an episode from its saved rows and check every saved flag against what the state says now (a mismatch raises)."""
    ep = Episode(arm, mode, seed)
    for i, r in enumerate(rows):
        kind, _ = ep.next_call()
        if kind != r["kind"]: raise ValueError(f"row {i} of {arm}/{seed}: the file says {r['kind']!r} but the state needs {kind!r}")
        flags = ep.apply_check(r["text"]) if kind == "check" else ep.apply_summary(r["text"])
        for k, v in flags.items():
            if r.get(k) != v: raise ValueError(f"row {i} of {arm}/{seed}: saved {k}={r.get(k)!r}, replay gives {v!r}")
    return ep


def run_arm(llm, arm: str, mode: str, seed: int, path: str, ep_index: int, meta=None, progress=None, steps: Optional[int] = None) -> dict:
    """Run (or carry on) one episode of one arm; `path` holds the rows of every episode of this mode (header first). Returns the rows of this episode and its state."""
    meta = dict(meta or {}, test="ruled_out", mode=mode, steps=MODES[mode]["steps"], compact_every=COMPACT_EVERY, seeds=list(SEEDS))
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    mine = [r for r in rows if r["arm"] == arm and r["ep"] == ep_index]
    ep = replay(mine, arm, mode, seed); limit = ep.steps if steps is None else steps
    with open(path, "a") as f:
        while not ep.done() and ep.checks < limit:
            kind, messages = ep.next_call(); text, ntok, sec, ptok = _call(llm, messages)
            flags = ep.apply_check(text) if kind == "check" else ep.apply_summary(text)
            row = _roundtrip(dict(arm=arm, ep=ep_index, kind=kind, step=ep.checks, text=text, tokens=ntok, seconds=sec, prompt_tokens=ptok, **flags))
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row); mine.append(row)
            if progress: progress(len(rows), None)
    return {"rows": mine, "episode": ep}


def run_mode(llm, mode: str, path: str, meta=None, progress=None) -> dict:
    """Every arm of the mode over every seed, in the order of the page: episode 0 of each arm, then episode 1, then episode 2."""
    out = {}
    for ep_index, seed in enumerate(SEEDS):
        for arm in MODES[mode]["arms"]: out[(arm, ep_index)] = run_arm(llm, arm, mode, seed, path, ep_index, meta, progress)
    rows = load_rows(path, dict(meta or {}, test="ruled_out", mode=mode, steps=MODES[mode]["steps"], compact_every=COMPACT_EVERY, seeds=list(SEEDS)))
    return {"rows": rows, "summary": summarize(rows, mode), "meta": dict(meta or {}, mode=mode)}


# ---------------------------------------------------------------- what is counted
def _windows(steps: int) -> List[Tuple[int, int]]: return [(a, min(a + WINDOW - 1, steps)) for a in range(1, steps + 1, WINDOW)]


def summarize(rows: List[dict], mode: str) -> dict:
    """Per arm: replies, repeats, refusals, replies that were not read, new causes ruled out, episodes found, prompt tokens by window, and the repeats a literal text match would have caught."""
    arms = MODES[mode]["arms"]; steps = MODES[mode]["steps"]; wins = _windows(steps); out = {}
    for arm in arms:
        rs = [r for r in rows if r.get("arm") == arm and r.get("kind") == "check"]; sm = [r for r in rows if r.get("arm") == arm and r.get("kind") == "summary"]
        cell = dict(episodes=len({r["ep"] for r in rows if r.get("arm") == arm}), checks=len(rs), repeats=sum(r["repeat"] for r in rs), refused=sum(r["refused"] for r in rs),
                    accepted_repeats=sum(r["repeat"] and not r["refused"] for r in rs), unread=sum(r["status"] != "ok" for r in rs),
                    new_ruled=sum(r["status"] == "ok" and not r["repeat"] and not r["found"] for r in rs), found=sum(bool(r["found"]) for r in rs), found_steps=sum(r["step"] for r in rs if r["found"]),
                    summaries=len(sm), tokens=sum((r.get("tokens") or 0) for r in rs + sm), seconds=sum((r.get("seconds") or 0.0) for r in rs + sm))
        cell["windows"] = []
        for a, b in wins:
            w = [r for r in rs if a <= r["step"] <= b]; pt = [r["prompt_tokens"] for r in w if r.get("prompt_tokens")]
            cell["windows"].append(dict(first=a, last=b, n=len(w), repeats=sum(r["repeat"] for r in w), unread=sum(r["status"] != "ok" for r in w), ptok_sum=sum(pt), ptok_n=len(pt)))
        again = refusals = 0                                                                  # after a refusal, is the next reply a repeat again?
        lit = 0
        for e in sorted({r["ep"] for r in rs}):
            seq = [r for r in rs if r["ep"] == e]; seen_text = set()
            for i, r in enumerate(seq):
                if r["refused"] and i + 1 < len(seq): refusals += 1; again += bool(seq[i + 1]["repeat"])
                t = _norm(r["text"])
                if r["repeat"] and t in seen_text: lit += 1
                if r["status"] == "ok": seen_text.add(t)
        cell.update(refusals_followed=refusals, refusals_then_repeat=again, literal_repeats=lit); out[arm] = cell
    return out


# ---------------------------------------------------------------- reading rules (fixed in docs/RULED_OUT_TEST.md before the run); every threshold is an integer
def verdict_keys(S: dict) -> List[tuple]:
    """S = summarize(rows, 'keys'). Lines are tagged [guard] (true by construction or by the instrument), [format], or [need] (what the test is about)."""
    k, c, p, g = S["kept"], S["compacted"], S["held_prompt"], S["held_gate"]; last = lambda x: x["windows"][-1]
    ok = lambda x: x["checks"] > 0
    return [
        ("[guard] held_gate: the host answered no repeated cause in any episode (every repeat was refused)", ok(g) and g["accepted_repeats"] == 0),
        ("[guard] every arm ran all three episodes", all(S[a]["episodes"] == len(SEEDS) for a in ARMS_KEYS)),
        ("[format] at most 5 percent of the replies in each arm were not read (20 times the unread count at most the replies)", all(ok(S[a]) and 20 * S[a]["unread"] <= S[a]["checks"] for a in ARMS_KEYS)),
        ("[need] compacted proposes more repeats than kept", ok(c) and ok(k) and c["repeats"] > k["repeats"]),
        ("[need] held_gate ruled out at least 5 more new causes per episode than compacted", ok(g) and ok(c) and g["new_ruled"] - c["new_ruled"] >= 5 * len(SEEDS)),
        ("[need] held_gate proposes no more repeats than kept", ok(g) and ok(k) and g["repeats"] <= k["repeats"]),
        ("[need] after a refusal the next reply is a repeat again at most 1 time in 5", g["refusals_followed"] > 0 and 5 * g["refusals_then_repeat"] <= g["refusals_followed"]),
        ("[need] in the last window held_gate's prompt is shorter than kept's (mean prompt tokens)",
         last(g)["ptok_n"] > 0 and last(k)["ptok_n"] > 0 and last(g)["ptok_sum"] * last(k)["ptok_n"] < last(k)["ptok_sum"] * last(g)["ptok_n"]),
    ]


def verdict_words(S: dict) -> List[tuple]:
    """S = summarize(rows, 'words')."""
    k, g = S["kept"], S["held_gate"]; ok = lambda x: x["checks"] > 0
    return [
        ("[format] at most 10 percent of the replies in each arm did not name exactly one part and one way (10 times the unread count at most the replies)", all(ok(S[a]) and 10 * S[a]["unread"] <= S[a]["checks"] for a in ARMS_WORDS)),
        ("[need] reading the words finds at least one repeat that a literal text match would not have found (kept and held_gate together)",
         (k["repeats"] + g["repeats"]) - (k["literal_repeats"] + g["literal_repeats"]) >= 1),
        ("[need] held_gate proposes no more repeats than kept", ok(g) and ok(k) and g["repeats"] <= k["repeats"]),
    ]


def print_summary(S: dict, mode: str, title: str = ""):
    print(f"== ruled-out causes, {mode} mode {title} ==")
    print(f"{'arm':<12}{'eps':>4}{'replies':>8}{'repeats':>8}{'refused':>8}{'unread':>7}{'new':>5}{'found':>6}{'summ.':>6}{'literal':>8}")
    for a, c in S.items(): print(f"{a:<12}{c['episodes']:>4}{c['checks']:>8}{c['repeats']:>8}{c['refused']:>8}{c['unread']:>7}{c['new_ruled']:>5}{c['found']:>6}{c['summaries']:>6}{c['literal_repeats']:>8}")
    print("repeats by window of 10 replies (mean prompt tokens in brackets):")
    for a, c in S.items(): print(f"  {a:<12}" + "  ".join(f"{w['first']}-{w['last']}: {w['repeats']} ({(w['ptok_sum'] // w['ptok_n']) if w['ptok_n'] else '-'})" for w in c["windows"]))


def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(f"  {'yes' if ok else 'NO ':<3}  {text}")

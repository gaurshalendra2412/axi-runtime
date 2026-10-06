"""
Gate benchmark: what does each kind of check catch, and what does it cost?

Scenario: a dependency graph where node `db_auth` has two incoming edges. A
(simulated) model proposes `delete_node db_auth` without removing those edges.

Checks compared:
  1. Syntax check     - is the proposal well-formed JSON with the right fields?
                        (what a Pydantic/JSON-schema validator does)
  2. Gluing gate, scan   - dangling-edge check that scans every edge (simple)
  3. Gluing gate, index  - same check with an adjacency index
  4. SQLite foreign key  - a standard relational database refusing the DELETE

IMPORTANT: this is a pure-Python implementation, not the Rust engine, and the
numbers depend on your machine. A syntax check alone cannot see relational
damage; that is the point of the demo. A database foreign key can, so a fair
claim is "syntax validation is not enough", not "only AXI catches this".

Usage:  python benchmarks/gate_benchmark.py [--sizes 1000 10000 100000]
"""

import argparse
import json
import sqlite3
import statistics
import time

try:  # optional; the script runs without it
    from pydantic import BaseModel

    class Proposal(BaseModel):
        action: str
        target_node: str

    def syntax_check(raw: str) -> bool:
        Proposal.model_validate_json(raw)
        return True

    SYNTAX_IMPL = "pydantic"
except ImportError:

    def syntax_check(raw: str) -> bool:
        d = json.loads(raw)
        return isinstance(d.get("action"), str) and isinstance(d.get("target_node"), str)

    SYNTAX_IMPL = "json (pydantic not installed)"


class Graph:
    """Directed graph with an adjacency index for incident edges."""

    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.incident = {}

    def add_edge(self, u, v):
        self.nodes.update((u, v))
        self.edges.add((u, v))
        self.incident.setdefault(u, set()).add((u, v))
        self.incident.setdefault(v, set()).add((u, v))


def gate_scan(g: Graph, target: str, consumed: set) -> bool:
    """Reject deleting `target` if any incident edge is not consumed. O(|E|)."""
    if target not in g.nodes:
        return False
    incident = {e for e in g.edges if e[0] == target or e[1] == target}
    return len(incident - consumed) == 0


def gate_index(g: Graph, target: str, consumed: set) -> bool:
    """Same rule using the adjacency index. O(degree)."""
    if target not in g.nodes:
        return False
    return len(g.incident.get(target, set()) - consumed) == 0


def build_graph(n: int) -> Graph:
    g = Graph()
    for i in range(1, n):
        g.add_edge(f"svc_{i-1}", f"svc_{i}")
    g.add_edge("svc_core", "db_auth")
    g.add_edge("svc_billing", "db_auth")
    return g


def build_sqlite(g: Graph) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE nodes(id TEXT PRIMARY KEY)")
    con.execute(
        "CREATE TABLE edges(src TEXT NOT NULL REFERENCES nodes(id), "
        "dst TEXT NOT NULL REFERENCES nodes(id))"
    )
    con.execute("CREATE INDEX edges_src ON edges(src)")
    con.execute("CREATE INDEX edges_dst ON edges(dst)")
    con.executemany("INSERT INTO nodes VALUES (?)", [(n,) for n in g.nodes])
    con.executemany("INSERT INTO edges VALUES (?, ?)", list(g.edges))
    con.commit()
    return con


def sqlite_delete_rejected(con: sqlite3.Connection, target: str) -> bool:
    try:
        con.execute("DELETE FROM nodes WHERE id = ?", (target,))
        con.rollback()
        return False
    except sqlite3.IntegrityError:
        return True


def timed(fn, trials):
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1000.0)  # microseconds
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(len(samples) * 0.99))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    raw = '{"action": "delete_node", "target_node": "db_auth"}'
    print(f"syntax check implementation: {SYNTAX_IMPL}")
    print(f"{'nodes':>8} | {'check':<22} | {'catches it?':<12} | {'median us':>10} | {'p99 us':>10}")
    print("-" * 74)

    for n in args.sizes:
        g = build_graph(n)
        both = g.incident["db_auth"]

        # Correctness first: the gates must reject the bad proposal, accept a
        # proposal that consumes the incident edges, and reject a missing node.
        assert syntax_check(raw) is True  # syntax is fine, damage is relational
        for gate in (gate_scan, gate_index):
            assert gate(g, "db_auth", set()) is False
            assert gate(g, "db_auth", set(both)) is True
            assert gate(g, "no_such_node", set()) is False
        con = build_sqlite(g)
        assert sqlite_delete_rejected(con, "db_auth") is True

        rows = [
            ("syntax only", "NO", lambda: syntax_check(raw)),
            ("gluing gate (scan)", "yes", lambda: gate_scan(g, "db_auth", set())),
            ("gluing gate (index)", "yes", lambda: gate_index(g, "db_auth", set())),
            ("sqlite foreign key", "yes", lambda: sqlite_delete_rejected(con, "db_auth")),
        ]
        trials = args.trials if n <= 10000 else max(20, args.trials // 10)
        for name, catches, fn in rows:
            med, p99 = timed(fn, trials)
            print(f"{n:>8} | {name:<22} | {catches:<12} | {med:>10.1f} | {p99:>10.1f}")
        print("-" * 74)
        con.close()


if __name__ == "__main__":
    main()

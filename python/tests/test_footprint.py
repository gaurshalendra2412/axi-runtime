"""Implements concept-map row 145 (thirteenth scan): the 3 Sep page "Storage is the ultimate truth serum for software." (Post 383, about 55 words, two reads; published 11:39:46 UTC). Quotes verified (YES, cut at 125
characters by the reader): "It strips away the UI polish, the hype, and the corporate valuation, laying bare the raw footprint..." and "Code and marketing can promise the world, but the storage menu tells you exactly
how much digital territory..." (the previous/next links the theme adds, such as the one about a "0 B lock app", are teasers and are not used).

The reading being tested (Rajnish to accept or reject): a claim about software is checked by its FOOTPRINT, the plain size it takes. So we state our own footprint as numbers and make them fail loudly if they grow.
Measured on the repository's pipeline modules (`python/axi/engine`), nothing assumed:
  1. The admission core - strict parser, gate, inverse, pager, runtime - is 363 lines and 14,645 bytes (302 lines of code without blanks and comments); the gate alone is 93 lines and 3,948 bytes, and its
     `check` function is under 50 lines. With the repair proposer (`diagnostics`), the two-phase commit, and the byte-level grammar decoder the whole pipeline is 824 lines and 36,826 bytes. The caps asserted
     here are about 15% above the measured values (core 420 lines / 17,000 bytes; gate 110 lines / 4,600 bytes; pipeline 950 lines / 42,000 bytes), so growth shows up as a failing test that has to be updated on purpose.
  2. Every one of those eight modules imports only the standard library and the package itself (no numpy, no torch): the admission path runs on a bare Python interpreter. (`wire.py` uses numpy and is not part of it.)
What they do NOT show: that this is small or large compared with anything (no other software was measured; the page compares installed sizes of apps, and nothing here is an app); that fewer lines means a better gate
(the gate checks applicability only, PAPER_CORRECTIONS.md item 20); that the model, the Rust crate, the experiments or the GPU kernels are small (they are outside this count and some need torch). Line and byte counts are
those of the checked-out files; a Windows checkout with converted line endings changes the byte counts by one per line, which the caps absorb."""
import ast, os, sys

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "axi", "engine")
CORE = ["cico_parser", "gate", "inverse", "pager", "runtime"]
PIPELINE = CORE + ["diagnostics", "dist_commit", "cico_decoder"]


def measure(name):
    raw = open(os.path.join(ENGINE, name + ".py"), "rb").read()
    text = raw.decode()
    lines = text.split("\n")
    n = len(lines) - 1 if text.endswith("\n") else len(lines)
    code = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    return n, len(raw), code, text


def test_the_admission_core_is_about_360_lines_and_the_whole_pipeline_about_820():
    m = {name: measure(name) for name in PIPELINE}
    core_lines, core_bytes = sum(m[k][0] for k in CORE), sum(m[k][1] for k in CORE)
    all_lines, all_bytes = sum(m[k][0] for k in PIPELINE), sum(m[k][1] for k in PIPELINE)
    assert 330 <= core_lines <= 420 and 13_000 <= core_bytes <= 17_000
    assert 750 <= all_lines <= 950 and 33_000 <= all_bytes <= 42_000
    assert m["gate"][0] <= 110 and m["gate"][1] <= 4_600
    assert sum(m[k][2] for k in CORE) <= 345
    tree = ast.parse(m["gate"][3])
    check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    assert check.end_lineno - check.lineno + 1 <= 50


def test_every_pipeline_module_imports_only_the_standard_library_and_the_package():
    stdlib = getattr(sys, "stdlib_module_names", None)
    assert stdlib is not None or sys.version_info < (3, 10)
    for name in PIPELINE:
        tree = ast.parse(measure(name)[3])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level: continue                                           # a relative import of a sibling module
                tops = [(node.module or "").split(".")[0]]
            else:
                continue
            for t in tops:
                assert t == "axi" or stdlib is None or t in stdlib, (name, t)
    banned = {"numpy", "torch", "triton", "scipy"}
    for name in PIPELINE:
        assert not any(f"import {b}" in measure(name)[3] or f"from {b}" in measure(name)[3] for b in banned), name


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)

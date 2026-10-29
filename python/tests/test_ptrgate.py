"""The pointer gate (dangling-edge rule on heap pointers) on invented snippets, and the result page."""
import hashlib
import json
import os

from axi.engine import ptrgate

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


def verdict(src, name):
    return ptrgate.analyze_source(src)[name][0]


def test_use_after_free_found():
    src = "void f(){ char *d; d = (char*)malloc(10); if (d == NULL) {exit(-1);} free(d); printLine(d); }"
    assert verdict(src, "f") == "vulnerable"
    assert ptrgate.analyze_source(src)["f"][1][0][0] == "UAF"


def test_double_free_found():
    assert verdict("void f(){ int *d = new int; delete d; delete d; }", "f") == "vulnerable"


def test_clean_when_use_precedes_free():
    assert verdict("void f(){ char *d; d = (char*)malloc(10); printLine(d); free(d); }", "f") == "clean"


def test_dead_branch_is_not_a_finding():
    src = "void f(){ char *d; d = (char*)malloc(10); free(d); if (globalFalse) { printLine(d); } }"
    assert verdict(src, "f") == "clean"


def test_alias_shares_the_node():
    src = "void f(){ char *a; char *b; a = (char*)malloc(10); b = a; free(a); printLine(b); }"
    assert verdict(src, "f") == "vulnerable"


def test_store_through_freed_pointer_is_a_use():
    src = "void f(){ int *d; d = new int; *d = 5; delete d; *d = 6; }"
    assert verdict(src, "f") == "vulnerable"


def test_reassignment_after_free_is_clean():
    src = "void f(){ char *d; d = (char*)malloc(10); free(d); d = NULL; }"
    assert verdict(src, "f") == "clean"


def test_unsupported_is_not_a_guess():
    assert ptrgate.analyze_source("void f(){ do { x(); } while (1); }")["f"][0] == "unsupported"


def test_facts_text_is_short_and_exact():
    r = ptrgate.analyze_source("void f(){ char *d; d = (char*)malloc(3); free(d); printLine(d); }")["f"]
    assert "freed at" in ptrgate.facts_text(r) and len(ptrgate.facts_text(r).splitlines()) <= 8


def test_switch_and_goto_on_constants():
    src = "void f(){ char *d; d = (char*)malloc(3); switch(6){ case 5: break; case 6: free(d); printLine(d); break; } }"
    assert verdict(src, "f") == "vulnerable"
    src = "void g(){ char *d; d = (char*)malloc(3); goto s; s: free(d); printLine(d); }"
    assert verdict(src, "g") == "vulnerable"


def test_page_hash_and_result_figures():
    t = open(os.path.join(DOCS, "PTRGATE_CHECK.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "ptrgate_check_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want
    a = json.load(open(os.path.join(DOCS, "results", "ptrgate_check_held_frozen_gate.json")))["tally"]
    b = json.load(open(os.path.join(DOCS, "results", "ptrgate_check_held_after_fix.json")))["tally"]
    assert (a["TP"], a["FN"], a["FP"], a["TN"]) == (558, 108, 0, 2767)
    assert (b["TP"], b["FN"], b["FP"], b["TN"]) == (648, 18, 0, 2767)
    for needle in ("0.8378", "0.9685", "0.9730", "0.9948", "3,433"):
        assert needle in t


def test_returned_freed_pointer_is_followed():
    src = ("static char * h(){ char *d; d = (char*)malloc(4); free(d); return d; }\n"
           "void bad(){ char *r = h(); printLine(r); }")
    assert verdict(src, "bad") == "vulnerable"
    assert verdict("static char * h(){ char *d; d = (char*)malloc(4); return d; }\nvoid g(){ char *r = h(); printLine(r); }", "g") == "clean"


def test_second_page_hash_and_figures():
    t = open(os.path.join(DOCS, "PTRGATE_CHECK_2.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "ptrgate_check_2_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want
    a = json.load(open(os.path.join(DOCS, "results", "ptrgate_check_2.json")))["tally"]["all"]
    assert (a["TP"], a["FN"], a["FP"], a["TN"], a["unsup"]) == (140, 80, 120, 512, 0)
    for needle in ("0.636", "0.190", "0.765", "852"):
        assert needle in t


def test_transport_is_not_a_dereference():
    src = ("static void sink(char * d){ ; }\nvoid f(){ char *p; p = (char*)malloc(4); free(p); sink(p); "
           "if (p != NULL) { ; } }")
    assert verdict(src, "f") == "clean"
    src = "static void sink(char * d){ printLine(d); }\nvoid f(){ char *p; p = (char*)malloc(4); free(p); sink(p); }"
    assert verdict(src, "f") == "vulnerable"


def test_global_reference_and_flag():
    src = ("static char * g;\nstatic void sink(){ char * d = g; printLine(d); }\n"
           "void f(){ char *p; p = (char*)malloc(4); free(p); g = p; sink(); }")
    assert verdict(src, "f") == "vulnerable"
    src = ("static int flag = 0;\nstatic void sink(char * d){ if (flag) { printLine(d); } }\n"
           "void f(){ char *p; p = (char*)malloc(4); free(p); sink(p); }")
    assert verdict(src, "f") == "clean"
    src = "void f(){ char *p; char * &r = p; p = (char*)malloc(4); free(p); free(r); }"
    assert verdict(src, "f") == "vulnerable"


def test_third_page_hash_and_figures():
    t = open(os.path.join(DOCS, "PTRGATE_CHECK_3.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "ptrgate_check_3_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want
    a = json.load(open(os.path.join(DOCS, "results", "ptrgate_check_3.json")))["tally"]
    assert (a["TP"], a["FN"], a["FP"], a["TN"], a["unsup"]) == (240, 60, 14, 1060, 0)
    for needle in ("0.800", "0.0130", "0.9461", "1,374"):
        assert needle in t


def test_cast_array_and_depth():
    src = ("static void sink(void * v){ char ** pp = (char **)v; char * d = (*pp); printLine(d); }\n"
           "void f(){ char *p; p = (char*)malloc(4); free(p); sink(&p); }")
    assert verdict(src, "f") == "vulnerable"
    src = ("static void sink(char * a[]){ char * d = a[2]; free(d); }\n"
           "void f(){ char *p; char *arr[5]; p = (char*)malloc(4); free(p); arr[2] = p; sink(arr); }")
    assert verdict(src, "f") == "vulnerable"


def test_fourth_page_hash_and_figures():
    t = open(os.path.join(DOCS, "PTRGATE_CHECK_4.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "ptrgate_check_4_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want
    a = json.load(open(os.path.join(DOCS, "results", "ptrgate_check_4.json")))["tally"]
    assert (a["TP"], a["FN"], a["FP"], a["TN"], a["unsup"]) == (80, 100, 160, 340, 0)
    for needle in ("0.444", "0.320", "0.618", "680"):
        assert needle in t


def test_classes_and_containers_are_unsupported():
    src = "class A { public: virtual void f(char * d); };\nvoid g(){ char *p; p = (char*)malloc(4); free(p); printLine(p); }"
    assert ptrgate.analyze_source(src)["g"][0] == "unsupported"
    assert ptrgate.analyze_source("void g(){ vector<char *> v; char *p = (char*)malloc(4); free(p); v.push_back(p); }")["g"][0] == "unsupported"
    assert ptrgate.analyze_source("void A::h(char * d) const { free(d); }\nvoid g(){ ; }")["g"][0] == "unsupported"
    assert ptrgate.analyze_source("class T { public: int a; };\nvoid g(){ T * p = new T; delete p; delete p; }")["g"][0] == "vulnerable"


def test_model_test_parse_summarize_and_no_leak():
    from axi.experiments import ptrgate_model as M
    assert M.parse('{"memory_error": true, "why": "x"}') is True and M.parse('text {"memory_error":"False"}') is False and M.parse("no") is None
    src = ("void CWE416_bad(){ char *data; data = (char*)malloc(4); free(data); printLine(data); }\n"
           "static void good1(){ ; }\nvoid CWE416_good(){ good1(); }\nint main(){ CWE416_bad(); CWE416_good(); return 0; }")
    import random
    toks, t = M.anonymize(src, "CWE416_bad", random.Random(1))
    code = M.pretty(toks)
    assert not M.LEAK.search(code) and "main" not in code and t in code
    item = dict(id=0, group="hard", variant="21", cwe="416", label=1, target=t, code=code, gate="vulnerable", facts=ptrgate.facts_text(ptrgate.analyze_source(" ".join(toks))[t]))
    for arm in "PX":
        text = M.messages(item, arm)[1]["content"].replace("memory_error", "")
        assert not M.LEAK.search(text), arm
    assert "checker" in M.messages(item, "X")[1]["content"] and "checker" not in M.messages(item, "P")[1]["content"]
    rows = [dict(id=0, arm="P", pred=False), dict(id=0, arm="X", pred=True)]
    s = M.summarize([item], rows)
    assert s["P"]["all"]["accuracy"] == 0.0 and s["X"]["all"]["accuracy"] == 1.0 and s["X"]["hard"]["recall"] == 1.0


def test_model_test_page_is_locked():
    t = open(os.path.join(DOCS, "PTRGATE_MODEL_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "ptrgate_model_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want

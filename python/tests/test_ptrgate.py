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

import json, hashlib, glob, os, re, sys
tag = "Qwen_Qwen2.5-32B-Instruct-AWQ"
res = {"machine_logs": {}, "cells": {}}
for f in sorted(glob.glob("/content/out/t41_q32*.log")):
    lines = [l.rstrip("\n") for l in open(f, errors="replace") if re.match(r"(SETTING|LOADED|CELLS|CELL |DONE)", l)]
    res["machine_logs"][os.path.basename(f)] = lines
for f in sorted(glob.glob("/content/out/%s_*.json" % tag)):
    b = open(f, "rb").read()
    name = os.path.basename(f)
    if "_summary_" in name:
        res["summary_file_" + name] = json.loads(b)
    else:
        lab = name[len(tag) + 1:-5]
        res["cells"][lab] = {"file": name, "raw_bytes": len(b), "raw_sha256": hashlib.sha256(b).hexdigest()}
s = json.dumps(res, sort_keys=True, default=str, separators=(",", ":"))
print("EXPORT_LEN", len(s), "EXPORT_SHA256", hashlib.sha256(s.encode()).hexdigest())
print("EXPORT_BEGIN" + s + "EXPORT_END")

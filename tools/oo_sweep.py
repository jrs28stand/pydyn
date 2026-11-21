# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Sweep of Dynare's test files through ``pydyn.dynare`` (the results mode).

    python tools/oo_sweep.py list   <corpus> <out.json>             files whose MATLAB lines read oo_ (corpus = the
                                                                     output of tools/sweep_matlab.py collect)
    python tools/oo_sweep.py run    <corpus> <list.json> <out.json> [--timeout 120] [--filter substr]
    python tools/oo_sweep.py report <out.json>                       counts and the reasons of the failures

A file "runs to the end" when ``dynare()`` returns: its MATLAB checks (``error(...)`` lines) passed."""
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import sys, json, warnings, time
sys.path.insert(0, %(root)r)
warnings.simplefilter("ignore")
import pydyn
t = time.time()
try:
    ws = pydyn.dynare(%(path)r)
    print(json.dumps({"status": "ok", "time": time.time() - t}))
except pydyn.ResultsError as exc:
    print(json.dumps({"status": "refused" if exc.matlab_message is None else "matlab_error", "msg": str(exc)[:400], "time": time.time() - t}))
except BaseException as exc:
    print(json.dumps({"status": "crash", "msg": (type(exc).__name__ + ": " + str(exc))[:400], "time": time.time() - t}))
"""


def mod_files(corpus):
    for sub in ("tests", "examples"):
        for jd in sorted((corpus / sub).rglob("model/json")):
            if (jd / "modfile.json").exists():
                yield jd


def cmd_list(corpus, out):
    res = []
    for jd in mod_files(corpus):
        try:
            mf = json.loads((jd / "modfile.json").read_text())
        except json.JSONDecodeError:
            txt = (jd / "modfile.json").read_text()
            try:
                mf = json.loads(re.sub(r'("period2":\s*)\}(\d+)', r"\1\2", txt))
            except json.JSONDecodeError:
                continue
        nat = [s.get("string", "") for s in mf.get("statements", []) if s.get("statementName") in ("native", "verbatim")]
        if any(re.search(r"\boo_\b", t) for t in nat):
            res.append(str(jd))
    Path(out).write_text(json.dumps(res, indent=0))
    print(len(res), "files")


def cmd_run(corpus, lst, out, timeout=120, filt=None):
    files = json.loads(Path(lst).read_text())
    if filt:
        files = [f for f in files if filt in f]
    results = json.loads(Path(out).read_text()) if Path(out).exists() else {}
    env = None
    for f in files:
        key = str(Path(f).relative_to(corpus)) if str(f).startswith(str(corpus)) else f
        if key in results and results[key].get("status") != "timeout" and "--redo" not in sys.argv:
            continue
        t = time.time()
        try:
            r = subprocess.run([sys.executable, "-c", CHILD % dict(root=str(ROOT), path=f)], capture_output=True,
                               text=True, timeout=timeout, cwd=str(Path(f).parents[2]))
            line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            try:
                results[key] = json.loads(line)
            except json.JSONDecodeError:
                results[key] = {"status": "crash", "msg": (r.stderr or r.stdout)[-300:]}
        except subprocess.TimeoutExpired:
            results[key] = {"status": "timeout", "time": time.time() - t}
        print(f"{results[key]['status']:12s} {key}  {results[key].get('msg', '')[:110]}", flush=True)
        Path(out).write_text(json.dumps(results, indent=0))


def cmd_report(out):
    res = json.loads(Path(out).read_text())
    c = collections.Counter(v["status"] for v in res.values())
    print(dict(c), "total", len(res))
    why = collections.Counter()
    for k, v in res.items():
        if v["status"] != "ok":
            msg = re.sub(r"\d+", "N", v.get("msg", ""))[:150]
            why[(v["status"], msg)] += 1
    for (st, msg), n in why.most_common(60):
        print(f"{n:4d} {st:12s} {msg}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "list":
        cmd_list(Path(a[1]), a[2])
    elif a[0] == "run":
        to = int(a[a.index("--timeout") + 1]) if "--timeout" in a else 120
        flt = a[a.index("--filter") + 1] if "--filter" in a else None
        cmd_run(Path(a[1]), a[2], a[3], to, flt)
    elif a[0] == "report":
        cmd_report(a[1])

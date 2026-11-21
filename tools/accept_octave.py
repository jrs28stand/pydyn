# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Resumable runner: Dynare's test-suite .mod files under Octave, results stored outside the repository.

    python tools/accept_octave.py [--src /home/claude/dl/src72] [--out /home/claude/accept_ref] [--jobs 2]
                                  [--timeout 900] [--only substring ...] [--limit N]

For every .mod file of the test list (meson.build) it creates a scratch directory holding the folder's files plus the
listed extra files, runs ``dynare(file, 'console', 'nograph', <-D args>)`` under Octave and writes
    <out>/<test path>.mat    oo_, M_, options_ (sanitised)           -- only if Dynare ran to the end
    <out>/<test path>.json   {"status": ok|dynare-error|timeout|xfail|no-result, "message", "elapsed", ...}
    <out>/<test path>.log    tail of the Octave output
A file whose .json exists is skipped (delete it to re-run).  Entries run in the order of the test list; the files of one
entry share a scratch directory, as in Dynare's test driver.  The 13 .m runner tests are recorded as "m-runner".
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_manifest import load_manifest  # noqa: E402

TOOLS = Path(__file__).resolve().parent
STUBS = ("figure plot subplot hold title legend xlabel ylabel zlabel close saveas drawnow bar plot3 surf mesh colorbar suptitle sgtitle clf "
         "axes gca gcf axis set line text patch fill area xlim ylim grid box annotation colormap stairs scatter semilogy semilogx loglog "
         "print orient hgsave savefig pause groot findobj waitbar uicontrol datetick xticks yticks xticklabels yticklabels view lighting hggroup patch errorbar barh pie").split()

RUN_M = """warning('off', 'Octave:shadowed-function');
addpath('%(src)s/matlab');
addpath('%(src)s/tests/utils');
addpath('%(tools)s');
addpath('%(stubs)s');
for p__ = {'control','datatypes','io','optim','statistics'}, try, pkg('load', p__{1}); catch, end, end
more off
cd('%(cwd)s');
t0 = tic;
failed = false;
try
  dynare('%(mod)s', 'console', 'nograph'%(args)s);
catch exception
  failed = true;
  fprintf('ACCEPT_ERROR_ID: %%s\\n', exception.identifier);
  fprintf('ACCEPT_ERROR_MSG: %%s\\n', strrep(exception.message, char(10), ' | '));
  if isfield(exception, 'stack')
    for i = 1:min(6, numel(exception.stack)), fprintf('ACCEPT_STACK: %%s:%%d\\n', exception.stack(i).name, exception.stack(i).line); end
  end
end
global oo_ M_ options_
try
  accept_save('%(out)s');
  fprintf('ACCEPT_SAVED\\n');
catch e2
  fprintf('ACCEPT_SAVE_FAILED: %%s\\n', e2.message);
end
fprintf('ACCEPT_ELAPSED: %%.1f\\n', toc(t0));
exit(0);
"""


def relid(path):
    """Result path of a test file: tests/foo/bar.mod -> foo/bar ; ../examples/x/y.mod -> examples/x/y"""
    return path.replace("../", "").rsplit(".", 1)[0]


def run_one(entry, mod, scratch, src, out, timeout, stubs, force=False):
    rel = relid(mod)
    res = Path(out) / (rel + ".json")
    if res.exists() and not force:
        return
    res.parent.mkdir(parents=True, exist_ok=True)
    sub = Path(mod).parent
    cwd = (scratch / 'tests' / sub).resolve()
    mat = Path(out) / (rel + ".mat")
    script = scratch / f"accept_run_{Path(mod).stem}.m"
    args = "".join(f", '{a}'" for a in entry.get("dynare_args", []))
    script.write_text(RUN_M % dict(src=src, tools=TOOLS, stubs=stubs, cwd=cwd, mod=Path(mod).name, args=args, out=mat))
    if mat.exists():
        mat.unlink()
    env = {**os.environ, "GLIBC_TUNABLES": "glibc.rtld.execstack=2", "source_root": str(src), **entry.get("env", {})}
    before = {f.name: f.stat().st_mtime_ns for f in cwd.iterdir() if f.is_file()}
    t = time.time()
    status, log = "ok", ""
    try:
        r = subprocess.run(["octave", "--no-gui", "--no-init-file", "--quiet", str(script)], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=str(cwd))
        log = r.stdout + "\n--STDERR--\n" + r.stderr
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        log = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\nTIMEOUT"
        subprocess.run(["pkill", "-f", str(script)])
    keep = Path(out) / "_files" / rel                  # files the run wrote next to the .mod (simulated data ...): later files of the entry read them
    for f in cwd.iterdir():
        if f.is_file() and before.get(f.name) != f.stat().st_mtime_ns and not f.name.startswith("accept_run_") and f.stat().st_size < 30e6:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, keep / f.name)
    msg = ""
    for ln in log.splitlines():
        if ln.startswith("ACCEPT_ERROR_MSG:"):
            msg = ln[len("ACCEPT_ERROR_MSG:"):].strip()
    if status != "timeout":
        if "ACCEPT_ERROR_MSG" in log:
            status = "dynare-error"
        elif "ACCEPT_SAVED" not in log:
            status = "no-result"
            msg = msg or "octave exited without result (crash?)"
    if entry.get("should_fail"):
        status = {"dynare-error": "xfail", "ok": "xfail-passed"}.get(status, status)
    (Path(out) / (rel + ".log")).write_text(log[:1500] + '\n...\n' + log[-4500:] if len(log) > 6000 else log)
    res.write_text(json.dumps({"status": status, "message": msg[:600], "elapsed": round(time.time() - t, 1),
                               "has_mat": mat.exists(), "should_fail": bool(entry.get("should_fail")),
                               "dynare_args": entry.get("dynare_args", [])}))
    script.unlink(missing_ok=True)
    print(f"{time.strftime('%H:%M:%S')} {status:13s} {time.time() - t:7.1f}s {mod}  {msg[:100]}", flush=True)


def prepare_scratch(entry, src, scratch):
    """Copy the test files, their folders' files and the extra files of ``entry`` into ``scratch`` (same relative layout)."""
    tests = Path(src) / "tests"
    scratch = scratch / "tests"                       # so that ../examples/... lands inside the scratch directory
    for f in entry["test"] + entry.get("extra", []):
        (scratch / Path(f).parent).mkdir(parents=True, exist_ok=True)
        d = (tests / f).parent
        for sib in d.iterdir():                       # whole folder (files only), a superset of Dynare's driver
            if sib.is_file() and not (scratch / Path(f).parent / sib.name).exists():
                shutil.copy(sib, scratch / Path(f).parent / sib.name)
        if (tests / f).exists():
            shutil.copy(tests / f, scratch / f)


def run_entry(entry, src, out, timeout, stubs, only, force):
    mods = [f for f in entry["test"] if f.endswith(".mod")]
    for f in entry["test"]:
        if f.endswith(".m"):
            rel = relid(f)
            r = Path(out) / (rel + ".json")
            if not r.exists():
                r.parent.mkdir(parents=True, exist_ok=True)
                r.write_text(json.dumps({"status": "m-runner", "message": "companion .m runner, not run"}))
    mods = [m for m in mods if not only or any(o in m for o in only)]
    mods = [m for m in mods if force or not (Path(out) / (relid(m) + ".json")).exists()]
    if not mods:
        return
    scratch = Path(tempfile.mkdtemp(prefix="accept_", dir="/var/tmp"))
    try:
        prepare_scratch(entry, src, scratch)
        for m in mods:
            run_one(entry, m, scratch, src, out, timeout, stubs, force)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def cost(entry, src):
    """Rough run-time class of an entry from its text (0 cheap ... 4 heavy): so that breadth comes first."""
    import re
    score = 0
    for f in entry["test"]:
        try:
            t = (Path(src) / "tests" / f).read_text(errors="replace")
        except OSError:
            continue
        t = re.sub(r"//.*|%.*", "", t)
        if re.search(r"mh_replic\s*=\s*[1-9]", t):
            score = max(score, 3)
        if re.search(r"periods\s*=\s*\d{5,}", t):
            score = max(score, 3)
        if re.search(r"\b(estimation|method_of_moments|identification|dynare_sensitivity|bvar_density|ms_sbvar|extended_path|sbvar)\b", t):
            score = max(score, 2)
        if re.search(r"particle|pct|parallel|gsa|dsmh|hssmc|smc", f):
            score = max(score, 4)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/claude/dl/src72")
    ap.add_argument("--out", default="/home/claude/accept_ref")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    stubs = Path(a.out) / "_stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "get.m").write_text("function varargout = get(varargin)\n  v = 1;\n  if nargin > 1 && ischar(varargin{2})\n"
                                 "    if strcmpi(varargin{2}, 'screensize'), v = [1 1 1920 1080];\n"
                                 "    elseif strcmpi(varargin{2}, 'color'), v = {[0 0 0], [1 0 0]};\n    end\n  end\n  varargout{1} = v;\nend\n")
    for f in STUBS:
        (stubs / f"{f}.m").write_text(f"function varargout = {f}(varargin)\n  varargout = cell(1, nargout);\n  if nargout > 0, varargout{{1}} = 1; end\nend\n")
    entries = load_manifest(a.src)
    entries = sorted(entries, key=lambda e: cost(e, a.src))        # cheap files first (stable)
    if a.limit:
        entries = entries[:a.limit]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda e: run_entry(e, a.src, a.out, a.timeout, stubs, a.only, a.force), entries))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

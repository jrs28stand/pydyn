# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""pydyn's side of the cross-port acceptance protocol: run Dynare's test-suite files through ``pydyn.dynare()``.

    python tools/accept_pydyn.py [--out /home/claude/accept_pydyn] [--ref /home/claude/accept_ref] [--jobs 2]
                                 [--timeout 600] [--only substring ...] [--force] [--all]

Same entries, scratch folder layout and file order as tools/accept_octave.py (the files of one entry run one after the other in
one scratch folder; files written by Dynare's own run of a file are copied in first, as the sweep always did, unless
``--no-ref-artifacts``).  Per test id it writes ``<out>/<id>.mat`` (oo_, M_, options_ as pydyn's results layer holds them,
Dynare's names and shapes; fields the layer declines to fill are listed in the json) and ``<out>/<id>.json``:

    {"status": "ok"|"refused"|"error"|"timeout", "message", "rng": "octave", "refused_fields": {path: reason},
     "load_s": process start + ``import pydyn`` (once per worker, repeated in each of its tests),
     "first_s": the first ``pydyn.dynare()`` call incl. the preprocessor, "second_s": the same call again in the same worker}

Tests whose Dynare reference run failed are not run (nothing to compare) unless ``--all``.  Compare with tools/accept_compare.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, os.environ.get("PYDYN_ACCEPT_PYDYN", str(ROOT)))
sys.path.insert(0, str(TOOLS))
from accept_manifest import load_manifest  # noqa: E402
from accept_octave import cost, prepare_scratch, relid  # noqa: E402

SRC = Path(os.environ.get("DYNARE_SRC", "/home/claude/dl/src72"))
os.environ.setdefault("PYDYN_CACHE_DIR", os.environ.get("PYDYN_ACCEPT_CACHE", "/var/tmp/pydyn_accept_cache"))
os.environ.setdefault("DYNARE_PREPROCESSOR", str(SRC / "build-octave/preprocessor/src/dynare-preprocessor"))

REFUSAL_TYPES = ("Unsupported", "NotImplementedError", "ModelError", "ParseError", "TranslationError", "OptimalPolicyError",
                 "DiscretionaryPolicyError", "OSRError", "SamplerError", "NonlinearFilterError", "SmootherError", "EstimationError",
                 "KOrderError", "LoglinearError", "ResultsError", "DateError")
_LOAD_S = None


def _process_age():
    try:
        ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = [int(x.split()[1]) for x in Path("/proc/stat").read_text().splitlines() if x.startswith("btime")][0]
        return time.time() - (boot + ticks / os.sysconf("SC_CLK_TCK"))
    except Exception:  # noqa: BLE001
        return float("nan")


def _init_worker():
    global _LOAD_S
    import pydyn  # noqa: F401
    _LOAD_S = round(_process_age(), 2)


# ------------------------------------------------------------------------------------------ oo_ -> .mat
def to_mat(x, path, refused, depth=0):
    """pydyn value -> something scipy.io.savemat writes as Dynare would have it (None: not stored)."""
    if depth > 14 or x is None:
        return None
    if isinstance(x, dict):
        out = {}
        for k in list(x.keys()):
            p = f"{path}.{k}" if path else str(k)
            try:
                v = x[k]
            except Exception as e:  # noqa: BLE001
                refused[p] = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            c = to_mat(v, p, refused, depth + 1)
            if c is not None and str(k).isidentifier() and not str(k).startswith("_") and len(str(k)) <= 62:
                out[str(k)] = c
        return out
    if hasattr(x, "items") and hasattr(x, "shape") and not isinstance(x, np.ndarray):          # pydyn.matlab.Cell (column-major)
        n = len(x.items)
        arr = np.empty(n, dtype=object)
        for i, it in enumerate(x.items):
            c = to_mat(it, f"{path}{{{i + 1}}}", refused, depth + 1)
            arr[i] = np.zeros((0, 0)) if c is None else c
        try:
            return arr.reshape(tuple(x.shape), order="F")
        except Exception:  # noqa: BLE001
            return arr
    if isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype.kind in "USV":
            return None if x.dtype == object else x
        return np.asarray(x.toarray() if hasattr(x, "toarray") else x)
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    if isinstance(x, (bool, np.bool_)):
        return np.array(bool(x))
    if isinstance(x, (int, float, complex, np.integer, np.floating)):
        return np.array(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        if all(isinstance(v, str) for v in x):
            a = np.empty(len(x), dtype=object)
            a[:] = list(x)
            return a
        try:
            a = np.asarray(x, dtype=float)
            return a
        except Exception:  # noqa: BLE001
            return None
    return None


def write_mat(res, mat_path):
    from scipy.io import savemat
    refused = {}
    S = {}
    for name, obj in (("oo_", res.oo_), ("M_", getattr(res, "M_", None)), ("options_", getattr(res, "options_", None))):
        try:
            c = to_mat(obj, name if name != "oo_" else "", {} if name != "oo_" else refused)
        except Exception:  # noqa: BLE001
            c = None
        if c:
            S[name] = c
    S.setdefault("oo_", {})
    savemat(mat_path, S, do_compression=True, long_field_names=True)
    return refused


# ------------------------------------------------------------------------------------------ one entry
def copy_ref_artifacts(ref, mod, scratch):
    art = Path(ref) / "_files" / relid(mod)
    if art.is_dir():
        dest = scratch / "tests" / Path(mod).parent
        for f in art.rglob("*"):
            if f.is_file():
                t = dest / f.relative_to(art)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, t)


def run_mod(mod, scratch, out, timeout):
    import pydyn
    rel = relid(mod)
    mat, js = Path(out) / (rel + ".mat"), Path(out) / (rel + ".json")
    mat.parent.mkdir(parents=True, exist_ok=True)
    info = {"status": "ok", "message": "", "rng": "octave", "load_s": _LOAD_S}
    path = scratch / "tests" / mod

    def alarm(*_):
        raise TimeoutError("timeout")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(timeout)
    res = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.time()
            res = pydyn.dynare(path)
            info["first_s"] = round(time.time() - t, 3)
            refused = write_mat(res, mat)
            info["refused_fields"] = refused
            if info["first_s"] < 60:
                t = time.time()
                pydyn.dynare(path)
                info["second_s"] = round(time.time() - t, 3)
    except TimeoutError:
        info.update(status="timeout", message=f"timeout after {timeout}s")
    except Exception as e:  # noqa: BLE001
        refusal = any(c.__name__.startswith(REFUSAL_TYPES) or c.__name__ in REFUSAL_TYPES for c in type(e).__mro__)
        info.update(status="refused" if refusal else "error", message=f"{type(e).__name__}: {str(e)[:400]}")
        if not refusal:
            info["trace"] = traceback.format_exc()[-1200:]
    finally:
        signal.alarm(0)
    if info["status"] != "ok" and mat.exists():
        mat.unlink()
    js.write_text(json.dumps(info, indent=1))
    return rel, info["status"]


def run_entry(entry, mods, out, ref, timeout, ref_artifacts):
    scratch = Path(tempfile.mkdtemp(prefix="accpy_", dir="/var/tmp"))
    done = []
    try:
        prepare_scratch(entry, SRC, scratch)
        for m in mods:
            if ref_artifacts:
                copy_ref_artifacts(ref, m, scratch)
            done.append(run_mod(m, scratch, out, timeout))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/claude/accept_pydyn")
    ap.add_argument("--ref", default="/home/claude/accept_ref")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true", help="also run tests whose Dynare reference run failed")
    ap.add_argument("--no-ref-artifacts", action="store_true")
    a = ap.parse_args(argv)
    out, ref = Path(a.out), Path(a.ref)
    out.mkdir(parents=True, exist_ok=True)
    entries = sorted(load_manifest(SRC), key=lambda e: cost(e, SRC))
    jobs = []
    for e in entries:
        mods = []
        for m in e["test"]:
            if not m.endswith(".mod") or (a.only and not any(o in m for o in a.only)):
                continue
            rj = ref / (relid(m) + ".json")
            if not rj.exists() or (not a.all and json.loads(rj.read_text())["status"] != "ok"):
                continue
            if (out / (relid(m) + ".json")).exists() and not a.force:
                continue
            mods.append(m)
        if mods:
            jobs.append((e, mods))
    print(sum(len(m) for _, m in jobs), "tests in", len(jobs), "entries", flush=True)
    ctx = multiprocessing.get_context("spawn")
    while jobs:
        pending = list(jobs)
        jobs = []
        with cf.ProcessPoolExecutor(a.jobs, mp_context=ctx, initializer=_init_worker) as ex:
            futs = {ex.submit(run_entry, e, m, out, ref, a.timeout, not a.no_ref_artifacts): (e, m) for e, m in pending}
            for f in cf.as_completed(futs):
                try:
                    for rel, st in f.result():
                        print(f"{time.strftime('%H:%M:%S')} {st:8s} {rel}", flush=True)
                except cf.process.BrokenProcessPool:
                    break
                except Exception as exc:  # noqa: BLE001
                    print("entry failed:", futs[f][1], exc, flush=True)
        # a crashed worker (segfault) breaks the pool: the tests without a json are retried once in a fresh pool, then marked
        left = [(e, [m for m in ms if not (out / (relid(m) + ".json")).exists()]) for e, ms in pending]
        left = [(e, ms) for e, ms in left if ms]
        if left and len(left) < len(pending):
            jobs = left
        else:
            for e, ms in left:
                for m in ms:
                    (out / (relid(m) + ".json")).parent.mkdir(parents=True, exist_ok=True)
                    (out / (relid(m) + ".json")).write_text(json.dumps({"status": "error", "message": "worker process crashed", "rng": "octave"}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

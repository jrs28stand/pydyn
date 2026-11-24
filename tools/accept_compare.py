# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Port-agnostic comparison of a Dynare port's results with Dynare 7.2's own (the Octave reference).

    python tools/accept_compare.py <port_output_dir> [--ref /home/claude/accept_ref] [--known tools/accept_known.json]
                                   [--report out.md] [--only substr ...] [--name pydyn]

A port's output directory holds, per test id (the id is the path of the test file below tests/ without ``.mod``, as in the
reference folder, ``../examples/x/y.mod`` -> ``examples/x/y``):

    <id>.mat   oo_ (and the M_ / options_ fields accept_save.m keeps), Dynare's field names and shapes, -mat v5/v7 format
    <id>.json  {"status": "ok" | "refused" | "error" | "timeout", "message": str, "rng": "octave" | anything else,
                "refused_fields": {"dr.ghx": "reason", ...}      (optional: oo_ paths the port declines to fill),
                "load_s", "first_s", "second_s": timings (optional)}

The comparison is the sweep's: every non-empty field of Dynare's stored ``oo_`` (bookkeeping fields aside) is compared with
the port's, ``max|a-b| <= atol + rtol * max|b|`` over the whole field (rtol 1e-8, atol 1e-10; ``dr.eigval`` as the number of
eigenvalues, the number of stable ones and the moduli in [0.1, 10]: the infinite ones are rounding noise).  A field the port
does not hold is "refused" (listed in ``refused_fields``) or "not-produced".  Fields that depend on random numbers (simulated
paths, IRFs of simulation based methods, MCMC output) are compared only when the port's json says ``"rng": "octave"`` (it
replays Octave's generator); otherwise a difference there is reported as "skipped", never as a match.

Verdicts: match (all compared fields agree, none missing), partial (some agree, others missing/refused/skipped), refused,
wrong, port-error, dynare-failed (Dynare itself failed or timed out in the reference run), xfail (Dynare's own expected
failure), no-comparison.  ``--known`` is a json {id: {"class": ..., "note": ...}} of explained differences (Dynare quirks ...):
they are printed beside the file in the report, the verdict stays "wrong".
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

RTOL, ATOL = 1e-8, 1e-10
ORDER = ["match", "partial", "refused", "wrong", "port-error", "dynare-failed", "xfail", "no-comparison"]

# oo_ fields that are bookkeeping, not results
IGNORE_TOP = {"dynare_version", "gui", "exo_det_steady_state", "initial_exo_steady_state", "initial_steady_state", "initval_series",
              "exo_det_simul", "time", "deterministic_simulation", "skewness", "kurtosis", "dr_eigval", "endo_simul_orig", "pfm"}
# random-number dependent results (compared only for a port that replays Octave's generator)
RNG_PREFIXES = ("endo_simul", "exo_simul", "irfs", "posterior.metropolis", "Forecast", "ConditionalForecast", "MeanForecast")


# ------------------------------------------------------------------------------------------ reading
def load_mat(path):
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    return d.get("oo_"), d.get("M_"), d.get("options_")


def fields_of(s):
    return list(getattr(s, "_fieldnames", []))


def is_struct(x):
    return hasattr(x, "_fieldnames")


def nonempty(x):
    if x is None:
        return False
    if is_struct(x):
        return len(fields_of(x)) > 0
    if isinstance(x, np.ndarray):
        return x.size > 0
    if isinstance(x, (str, bytes)):
        return len(x) > 0
    return True


def sget(obj, path, default=None):
    for p in path.split("."):
        if obj is None or not is_struct(obj):
            return default
        obj = getattr(obj, p, None)
    return default if obj is None else obj


# ------------------------------------------------------------------------------------------ numerical comparison
def to_float(x):
    a = np.asarray(x)
    if a.dtype == object:
        raise TypeError("object array")
    return a.astype(complex) if np.iscomplexobj(a) else a.astype(float)


def compare(a, b, rtol=RTOL, atol=ATOL):
    """(ok, max abs difference, message) of the port's ``a`` against Dynare's ``b``."""
    a, b = to_float(a), to_float(b)
    if a.size != b.size:
        return False, np.inf, f"size {a.shape} vs Dynare {b.shape}"
    a = a.reshape(b.shape) if a.shape != b.shape and (1 in a.shape or 1 in b.shape or a.ndim != b.ndim) else a
    if a.shape != b.shape:
        return False, np.inf, f"shape {a.shape} vs Dynare {b.shape}"
    if b.size == 0:
        return True, 0.0, ""
    both_bad = ~np.isfinite(a) & ~np.isfinite(b)      # 0/0 and x/0 of zero-variance variables: NaN or +-Inf in both
    if np.any((~np.isfinite(a)) != (~np.isfinite(b))):
        return False, np.inf, "finite in one program, NaN/Inf in the other"
    m = ~both_bad
    if not m.any():
        return True, 0.0, ""
    mx = float(np.max(np.abs(a[m] - b[m])))
    return bool(mx <= atol + rtol * np.max(np.abs(b[m]))), mx, ""


def finite_moduli(x, lo=0.1, hi=10.0):
    """dr.eigval: number of eigenvalues, number below the qz criterion, moduli in [lo, hi] (the rest is rounding noise)."""
    a = np.sort(np.abs(np.asarray(x).ravel()))
    return np.concatenate([[float(a.size), float(np.sum(a < 1 + 1e-6))], a[(a >= lo) & (a <= hi)]])


# ------------------------------------------------------------------------------------------ field walk
class Walker:
    def __init__(self, ref_oo, port_oo, refused, rng_ok, tol=RTOL):
        self.ref_oo, self.refused, self.rng_ok, self.tol, self.rows = ref_oo, refused, rng_ok, tol, []
        self.endpoint = self.same_optimum(ref_oo, port_oo)
        self.walk(ref_oo, port_oo, "")

    @staticmethod
    def same_optimum(ref_oo, port_oo):
        """Both programs found the same maximum of the same objective: oo_.posterior.optimization.log_density agrees to
        1e-2 (absolute; the objective is ~1e3).  Dynare stores the optimiser's end point, which differs from another
        optimiser's by the optimiser's tolerance, and every field computed at the estimated parameters follows."""
        a = sget(ref_oo, "posterior.optimization.log_density")
        b = sget(port_oo, "posterior.optimization.log_density")
        try:
            a, b = float(np.ravel(a)[0]), float(np.ravel(b)[0])
        except (TypeError, ValueError, IndexError):
            return False
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-2)

    def is_refused(self, p):
        return any(p == r or p.startswith(r + ".") or r.startswith(p + ".") for r in self.refused)

    def walk(self, ref, got, path):
        for f in fields_of(ref):
            rv = getattr(ref, f)
            p = f"{path}.{f}" if path else f
            if not nonempty(rv) or (not path and f in IGNORE_TOP):
                continue
            gv = getattr(got, f, None) if got is not None and is_struct(got) else None
            if gv is None or not nonempty(gv):
                if self.is_refused(p):
                    self.rows.append((p, "refused", next((self.refused[r] for r in self.refused if p.startswith(r) or r.startswith(p)), "")[:160]))
                else:
                    self.rows.append((p, "not-produced", "no value from the port"))
                continue
            if is_struct(rv):
                self.walk(rv, gv, p)
                continue
            random_dep = p.startswith(RNG_PREFIXES)
            try:
                status, msg = self.one(p, gv, rv)
            except Exception as e:  # noqa: BLE001
                status, msg = "error", f"{type(e).__name__}: {str(e)[:160]}"
            if status == "wrong" and self.endpoint and p == "posterior.optimization.log_density":
                status, msg = "skipped", f"optimiser end point: the log posterior agrees to 1e-2 ({msg})"
            if status == "wrong" and self.endpoint and p != "posterior.optimization.log_density" and not msg.startswith(("size", "shape", "finite")):
                try:
                    ga, ra = (finite_moduli(gv), finite_moduli(rv)) if p == "dr.eigval" else (to_float(gv).ravel(), to_float(rv).ravel())
                    rel = float(np.max(np.abs(ga - ra)) / max(np.max(np.abs(ra)), 1e-300)) if ga.size == ra.size else np.inf
                except Exception:  # noqa: BLE001
                    rel = np.inf
                if rel <= 1e-2:
                    status, msg = "skipped", f"optimiser end point: the log posterior agrees (<= 1e-2), the field differs by {rel:.1e} relative; {msg}"
            if random_dep and not self.rng_ok and status in ("wrong", "match"):
                status, msg = "skipped", "random-number dependent; the port does not replay Octave's generator" if status == "wrong" else "random-number dependent (agrees, but not counted)"
            self.rows.append((p, status, msg))

    def one(self, p, gv, rv):
        if p == "dr.eigval":
            ok, mx, msg = compare(finite_moduli(gv), finite_moduli(rv), 1e-6, ATOL)
            return ("match" if ok else "wrong"), (msg or f"max abs diff {mx:.2e}")
        if p == "dr.sdim":
            ev = np.abs(np.asarray(sget(self.ref_oo, "dr.eigval", [])).ravel())
            if ev.size and (not np.all(np.isfinite(ev)) or ev.max() > 1e4):
                return "ignored", "model has infinite/rounding-noise eigenvalues; the stable-count depends on the noise"
        if isinstance(rv, str):
            return ("match" if str(gv) == rv else "wrong"), ""
        if isinstance(rv, np.ndarray) and rv.dtype == object:
            for i, (a, b) in enumerate(zip(list(np.asarray(gv, dtype=object).ravel()), list(rv.ravel()))):
                if isinstance(b, str):
                    if str(a) != b:
                        return "wrong", f"cell {i + 1}: {a!r} vs {b!r}"
                elif np.size(b):
                    ok, mx, msg = compare(a, b, self.tol, ATOL)
                    if not ok:
                        return "wrong", f"cell {i + 1}: {msg or f'max abs diff {mx:.2e}'}"
            return "match", ""
        rtol = 1e-4 if p == "endo_simul" else self.tol          # perfect foresight: Dynare stops at tolf
        ok, mx, msg = compare(gv, rv, self.tol, ATOL)
        if not ok and not msg and p == "endo_simul" and compare(gv, rv, rtol, ATOL)[0]:
            ok = True
        return ("match" if ok else "wrong"), (f"max abs diff {mx:.2e}" if (ok or not msg) else msg)


def verdict(rows):
    st = collections.Counter(r[1] for r in rows)
    if st["wrong"]:
        return "wrong", "; ".join(f"{r[0]}: {r[2]}" for r in [r for r in rows if r[1] == "wrong"][:4])
    if st["error"]:
        return "port-error", "; ".join(f"{r[0]}: {r[2]}" for r in [r for r in rows if r[1] == "error"][:3])
    others = st["refused"] + st["not-produced"] + st["skipped"]
    if st["match"] and others:
        return "partial", "; ".join(f"{r[0]}: {r[2]}" for r in [r for r in rows if r[1] in ("refused", "not-produced", "skipped")][:3])
    if st["match"]:
        return "match", ""
    if st["refused"]:
        return "refused", "; ".join(f"{r[0]}: {r[2]}" for r in [r for r in rows if r[1] == "refused"][:2])
    if st["not-produced"]:
        return "refused", "not produced: " + "; ".join(r[0] for r in [r for r in rows if r[1] == "not-produced"][:4])
    return "no-comparison", "no comparable oo_ field"


# ------------------------------------------------------------------------------------------ one test / all tests
def compare_one(tid, port_dir, ref_dir):
    ref_json = json.loads((ref_dir / f"{tid}.json").read_text())
    out = {"file": tid, "rows": [], "notes": [], "dynare_status": ref_json["status"]}
    if ref_json["status"] != "ok":
        out["verdict"] = {"xfail": "xfail", "xfail-passed": "xfail", "m-runner": "no-comparison"}.get(ref_json["status"], "dynare-failed")
        out["detail"] = ref_json.get("message", "") or ref_json["status"]
        return out
    pj = port_dir / f"{tid}.json"
    if not pj.exists():
        out["verdict"], out["detail"] = "port-error", "the port wrote no result for this test"
        return out
    pinfo = json.loads(pj.read_text())
    out["timing"] = {k: pinfo[k] for k in ("load_s", "first_s", "second_s") if k in pinfo}
    st = pinfo.get("status")
    if st in ("refused", "error", "timeout") or not (port_dir / f"{tid}.mat").exists():
        out["verdict"] = "refused" if st == "refused" else "port-error"
        out["detail"] = pinfo.get("message", "") or st or "no .mat"
        return out
    ref_oo, _, _ = load_mat(ref_dir / f"{tid}.mat")
    port_oo, _, _ = load_mat(port_dir / f"{tid}.mat")
    w = Walker(ref_oo, port_oo, pinfo.get("refused_fields", {}), pinfo.get("rng") == "octave")
    out["rows"] = w.rows
    out["verdict"], out["detail"] = verdict(w.rows)
    return out


def test_ids(ref_dir):
    return sorted(str(f.relative_to(ref_dir))[:-5] for f in ref_dir.rglob("*.json")
                  if not any(p.startswith("_") for p in f.relative_to(ref_dir).parts))


def normalise(msg):
    m = re.sub(r"^[\w.]+: ", "", msg.strip())
    m = re.sub(r"'[^']*'|\"[^\"]*\"", "'..'", m)
    return re.sub(r"\b\d+(\.\d+)?(e[+-]?\d+)?\b", "N", m)[:140]


def report(results, known, ref_dir, name, out_path):
    byv = collections.Counter(r["verdict"] for r in results)
    folder = collections.defaultdict(collections.Counter)
    for r in results:
        p = r["file"].split("/")
        folder[p[0] if len(p) > 1 else "(root)"][r["verdict"]] += 1
    octave = collections.Counter(json.loads(f.read_text())["status"] for f in ref_dir.rglob("*.json")
                                 if not any(p.startswith("_") for p in f.relative_to(ref_dir).parts) and "status" in json.loads(f.read_text()))
    L = [f"# Acceptance sweep: Dynare 7.2 test suite against {name}\n",
         "Generated by `tools/accept_compare.py` (method, tolerances and verdicts: its docstring).\n",
         "## Octave runs of Dynare (the reference)\n", "| status | files |\n|---|---|"]
    L += [f"| {k} | {v} |" for k, v in octave.most_common()]
    L += ["", f"## {name} verdicts\n", "| verdict | files |\n|---|---|"]
    L += [f"| {k} | {byv.get(k, 0)} |" for k in ORDER] + [f"| **total** | {len(results)} |", ""]
    L += ["## By test folder\n", "| folder | " + " | ".join(ORDER) + " | total |\n|---|" + "---|" * (len(ORDER) + 1)]
    for fo in sorted(folder):
        c = folder[fo]
        L.append(f"| {fo} | " + " | ".join(str(c.get(k, 0)) for k in ORDER) + f" | {sum(c.values())} |")
    wrong = [r for r in results if r["verdict"] == "wrong"]
    L += ["", f"## Wrong results ({len(wrong)} files)\n", "| file | field(s) and size of difference | classification |\n|---|---|---|"]
    for r in wrong:
        bad = "; ".join(f"`{x[0]}` {x[2]}" for x in r["rows"] if x[1] == "wrong")[:300]
        k = known.get(r["file"], {})
        L.append(f"| {r['file']} | {bad} | {k.get('class', 'not yet investigated')}: {k.get('note', '')} |")
    err = [r for r in results if r["verdict"] == "port-error"]
    L += ["", f"## Port errors ({len(err)} files)\n"]
    L += [f"- {v} x {k}" for k, v in collections.Counter(normalise(r["detail"].split(";")[0]) for r in err).most_common(25)]
    cc = collections.Counter()
    for r in results:
        if r["verdict"] in ("refused", "partial"):
            seen = {(normalise(f"{x[0].split('.')[0]}: {x[2]}") if x[1] == "refused" else f"not produced: oo_.{x[0].split('.')[0]}")
                    for x in r["rows"] if x[1] in ("refused", "not-produced")}
            if r["verdict"] == "refused" and not r["rows"]:
                seen.add(normalise(r["detail"]))
            cc.update(seen)
    L += ["", "## Commonest refusal / not-produced reasons\n"] + [f"- {v} files: {k}" for k, v in cc.most_common(30)]
    tim = [r["timing"] for r in results if r.get("timing")]
    L += ["", "## Timing (seconds, per test file)\n", "| quantity | tests | median | mean | max |\n|---|---|---|---|---|"]
    for k, lab in (("load_s", "process start + import (once per worker)"), ("first_s", "first dynare call incl. preprocessor"), ("second_s", "same call again in the same worker")):
        v = [t[k] for t in tim if k in t]
        if v:
            L.append(f"| {lab} | {len(v)} | {statistics.median(v):.2f} | {statistics.mean(v):.2f} | {max(v):.2f} |")
    Path(out_path).write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("port_dir")
    ap.add_argument("--ref", default="/home/claude/accept_ref")
    ap.add_argument("--known", default=str(Path(__file__).with_name("accept_known.json")))
    ap.add_argument("--report")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--name", default="the port")
    a = ap.parse_args(argv)
    ref, port = Path(a.ref), Path(a.port_dir)
    known = json.loads(Path(a.known).read_text()) if Path(a.known).exists() else {}
    results = []
    for tid in test_ids(ref):
        if a.only and not any(o in tid for o in a.only):
            continue
        try:
            r = compare_one(tid, port, ref)
        except Exception as e:  # noqa: BLE001
            r = {"file": tid, "rows": [], "verdict": "port-error", "detail": f"compare: {type(e).__name__}: {str(e)[:200]}"}
        results.append(r)
    (port / "_verdicts.json").write_text(json.dumps(results, default=str, indent=1))
    print(dict(collections.Counter(r["verdict"] for r in results)), len(results), "tests")
    if a.report:
        report(results, known, ref, a.name, a.report)
        print("wrote", a.report)
    return results


if __name__ == "__main__":
    main()

"""Benchmark and equality check for pydyn.obc_ho.HigherOrderNews (docs/OBC_HO_PERF.md).

Models: nk  = reference/dynareobc_octave/nk_occ.mod with shocks.mat (horizon 32 as in the DynareOBC comparison)
        nkm = reference/occest_nkm.mod, 300 N(0,1) draws (seed 4, as tests/test_obc_ho.py), horizon 40
Cases:  o1 (order 1), o2 (order 2), cub9 (order 2, cubature qmc_level 9), cubD (cubature, default qmc_level)
        ref = NewsOBC.simulate at order 1, for comparison.
Timing follows docs/OBC_BENCH_PROTOCOL.md: one process, warm-up, interleaved before/after, repeats, min and median.
  construction = HigherOrderNews(...) from a built model (steady state, k-order, M matrix, R, Sobol points).
  per period   = simulate(shocks) / number of periods.

Usage: python3 bench/obc_ho_bench.py time [--reps 5] [--impl before,after] [--models nk,nkm]
       python3 bench/obc_ho_bench.py baseline OUT.npz     (paths of the current code; run on the unmodified code)
       python3 bench/obc_ho_bench.py compare BASE.npz     (max abs difference to the stored baseline)
       python3 bench/obc_ho_bench.py profile MODEL CASE
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import pydyn  # noqa: E402
from pydyn.obc import NewsOBC  # noqa: E402

REF = Path(os.environ.get("PYDYN_REF", str(REPO / "reference")))
CASES = {
    "o1": dict(order=1),
    "o2": dict(order=2),
    "cub9": dict(order=2, cubature=True, qmc_level=9, regions=1),
    "cubD": dict(order=2, cubature=True, qmc_level=10, regions=1),
}
# regions cases: (kwargs, number of periods used). level 12 has 8191 points, so fewer periods.
RCASES = {}
for _lv, _np in ((9, 100), (12, 30)):
    for _rg in ("all", 2, 4, 16):
        RCASES[f"L{_lv}r{_rg}"] = (dict(order=2, cubature=True, qmc_level=_lv, regions=_rg), _np)
MODELS = {
    "nk": dict(mod=REF / "dynareobc_octave" / "nk_occ.mod", horizon=32),
    "nkm": dict(mod=REF / "occest_nkm.mod", horizon=40),
}


def impl_class(name):
    if name == "before":
        import obc_ho_before as m
    else:
        import pydyn.obc_ho as m
    return m.HigherOrderNews


def load(name):
    d = Path(tempfile.mkdtemp())
    spec = MODELS[name]
    shutil.copy(spec["mod"], d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / spec["mod"].name))
    if name == "nk":
        from scipy.io import loadmat
        S = loadmat(REF / "dynareobc_octave" / "shocks.mat", squeeze_me=True)["ShockSequence"].reshape(-1, 1)
    else:
        S = np.random.default_rng(4).normal(size=(300, m.n_exo))
    return m, S, spec["horizon"]


def build(cls, m, h, case):
    kw = CASES[case] if case in CASES else RCASES[case][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(m, horizon=h, check_ahead=h, **kw)


def timeit(f, reps):
    f()                                                       # warm-up
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        out.append(time.perf_counter() - t)
    return out


def cmd_rtime(a):
    impls = a.impl.split(",")
    for name in a.models.split(","):
        m, S, h = load(name)
        print(f"\n== {name} regions cases, reps {a.reps}: per-period us (min / median)")
        for case, (kw, nper) in RCASES.items():
            if a.cases and case not in a.cases.split(","):
                continue
            row = []
            objs = {i: build(impl_class(i), m, h, case) for i in impls}
            res = {i: [] for i in impls}
            for _ in range(a.reps):
                for i in impls:
                    t = time.perf_counter()
                    objs[i].simulate(S[:nper])
                    res[i].append((time.perf_counter() - t) / nper * 1e6)
            for i in impls:
                row.append(f"{i} {min(res[i]):9.0f}/{statistics.median(res[i]):9.0f}")
            print(f"{case:9s} " + "   ".join(row))


def cmd_time(a):
    impls = a.impl.split(",")
    for name in a.models.split(","):
        m, S, h = load(name)
        T = S.shape[0]
        print(f"\n== {name}: horizon {h}, {T} periods, reps {a.reps}")
        hdr = f"{'case':6s} {'impl':7s} {'constr ms min':>14s} {'med':>9s} {'per-period us min':>18s} {'med':>9s}"
        print(hdr)
        rows = {}
        for case in CASES:
            c = {i: [] for i in impls}
            s = {i: [] for i in impls}
            objs = {i: build(impl_class(i), m, h, case) for i in impls}
            for i in impls:
                objs[i].simulate(S)                           # warm-up
            for _ in range(a.reps):                           # interleaved
                for i in impls:
                    t = time.perf_counter()
                    objs[i] = build(impl_class(i), m, h, case)
                    c[i].append(time.perf_counter() - t)
                    t = time.perf_counter()
                    objs[i].simulate(S)
                    s[i].append(time.perf_counter() - t)
            for i in impls:
                print(f"{case:6s} {i:7s} {1e3*min(c[i]):14.2f} {1e3*statistics.median(c[i]):9.2f} "
                      f"{1e6*min(s[i])/T:18.1f} {1e6*statistics.median(s[i])/T:9.1f}")
        # NewsOBC reference at order 1
        ob = build(impl_class(impls[-1]), m, h, "o1").ob
        t_c = timeit(lambda: NewsOBC(ob, horizon=h, check_ahead=h), a.reps)
        nb = NewsOBC(ob, horizon=h, check_ahead=h)
        t_s = timeit(lambda: nb.simulate(S, periods=T), a.reps)
        print(f"{'ref':6s} {'NewsOBC':7s} {1e3*min(t_c):14.2f} {1e3*statistics.median(t_c):9.2f} "
              f"{1e6*min(t_s)/T:18.1f} {1e6*statistics.median(t_s)/T:9.1f}")


def paths(impl="after"):
    out = {}
    cls = impl_class(impl)
    for name in MODELS:
        m, S, h = load(name)
        for case in CASES:
            r = build(cls, m, h, case).simulate(S)
            for k in ("with_bound", "without_bound", "wedges"):
                out[f"{name}_{case}_{k}"] = r[k]
    return out


def rpaths(impl="after", prefix=""):
    out = {}
    cls = impl_class(impl)
    for name in MODELS:
        m, S, h = load(name)
        for case, (kw, nper) in RCASES.items():
            if not case.startswith(prefix):
                continue
            r = build(cls, m, h, case).simulate(S[:nper])
            for k in ("with_bound", "wedges"):
                out[f"{name}_{case}_{k}"] = r[k]
    return out


def cmd_baseline(a):
    np.savez_compressed(a.out, **(rpaths(a.impl) if a.regions else paths(a.impl)))
    print("wrote", a.out)


def cmd_compare(a):
    base = np.load(a.base)
    new = rpaths(a.impl) if a.regions else paths(a.impl)
    worst = 0.0
    for k in base.files:
        d = np.abs(base[k] - new[k]).max()
        worst = max(worst, d)
        print(f"{k:30s} max|diff| = {d:.3e}  exact={np.array_equal(base[k], new[k])}")
    print("MAX", worst)


def cmd_profile(a):
    import cProfile
    import pstats
    m, S, h = load(a.model)
    obj = build(impl_class(a.impl), m, h, a.case)
    obj.simulate(S[:20])
    pr = cProfile.Profile()
    pr.enable()
    obj.simulate(S)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
    st = getattr(obj, "stats", None)
    print(st)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("time"); t.add_argument("--reps", type=int, default=5); t.add_argument("--impl", default="before,after")
    t.add_argument("--models", default="nk,nkm"); t.set_defaults(f=cmd_time)
    b = sp.add_parser("baseline"); b.add_argument("out"); b.add_argument("--impl", default="before"); b.add_argument("--regions", action="store_true"); b.set_defaults(f=cmd_baseline)
    c = sp.add_parser("compare"); c.add_argument("base"); c.add_argument("--impl", default="after"); c.add_argument("--regions", action="store_true"); c.set_defaults(f=cmd_compare)
    rt = sp.add_parser("rtime"); rt.add_argument("--reps", type=int, default=3); rt.add_argument("--impl", default="before,after")
    rt.add_argument("--models", default="nk,nkm"); rt.add_argument("--cases", default=""); rt.set_defaults(f=cmd_rtime)
    pf = sp.add_parser("profile"); pf.add_argument("model"); pf.add_argument("case"); pf.add_argument("--impl", default="after")
    pf.set_defaults(f=cmd_profile)
    a = p.parse_args()
    a.f(a)

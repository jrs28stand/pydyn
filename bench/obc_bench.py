"""Benchmark harness for the OBC methods (protocol: /home/claude/work/bench/BENCH_PROTOCOL.md; results: docs/OBC_PERF.md).

Measurements
  P1  per period: simulation of 1,000 shock periods at fixed parameters (microseconds per period), split into
      periods in which nothing binds and periods in which something binds.
  P2  per parameter set: steady state, first-order solution, everything the method precomputes (milliseconds).
  P3  likelihood-like loop: 50 times perturb two parameters, redo P2, simulate 200 periods (seconds).
  P4  scaling: P1 and P2 at horizons 20, 40, 80 on nkm and sw.
  R1  reliability: 2,000 states of an unconstrained simulation, one period solved from each with a fresh shock
      at scale s in {0.5, 1, 1.5, 2}; failures per method and error type, and disagreement of regime sequences.

Implementations.  ``before`` is bench/obc_before.py, a frozen copy of pydyn/obc.py at the starting commit;
``after`` is the current pydyn.obc.  Both are driven in the same process and interleaved, as the protocol
requires.  OccBin (pydyn.occbin) is the same code in both.

Usage:  python3 bench/obc_bench.py {calibrate,p1,p2,p3,p4,r1,all} [--models zlb,nkm] [--reps 5] [--out DIR]
        [--impl before,after] [--seed N]
Environment: DYNARE_PREPROCESSOR must point to the preprocessor; nkm reads Dynare's tests/occbin/filter
(PYDYN_DYNARE_TESTS or $PYDYN_EXAMPLES/../tests).  Run under ``nice``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import pydyn  # noqa: E402
import pydyn.obc as obc_after  # noqa: E402
import obc_before  # noqa: E402
from pydyn.occbin import Occbin  # noqa: E402
from pydyn.perturbation import first_order  # noqa: E402
from pydyn.steady import steady_state  # noqa: E402

SEED = 20261124
CHECK_AHEAD = 40                  # news methods: periods checked beyond the horizon (the class default)
OCCBIN_CHECK_AHEAD = 80           # OccBin: check_ahead_periods, equal to horizon + CHECK_AHEAD of the news methods
IMPLS = {"before": obc_before, "after": obc_after}

# s per model: shock scale at which the constraint binds in 30-60% of the 1,000 periods and no method fails
# (found with ``calibrate``; see docs/OBC_PERF.md).  sw cannot reach 30%: OccBin and the regime formulation both
# fail somewhere in 1,000 periods for s >= 0.29, and the share is 7.5% at s = 0.25.
MODELS = {
    "zlb": dict(s=2.0, methods=("occbin", "news", "lk", "regime"), params=("beta", "phi")),
    "zlb2": dict(s=2.0, methods=("occbin", "news", "regime"), params=("beta", "phi")),
    "nkm": dict(s=4.0, methods=("occbin", "news", "lk", "regime"), params=("chi", "eta")),
    "sw": dict(s=0.25, methods=("occbin", "regime"), params=("csigma", "crpi")),
}


# ------------------------------------------------------------------------------------------------- loading
def dynare_tests_dir():
    if os.environ.get("PYDYN_DYNARE_TESTS"):
        return Path(os.environ["PYDYN_DYNARE_TESTS"])
    return Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples")).parent / "tests"


def load_model(name):
    tmp = Path(tempfile.mkdtemp(prefix="obcbench_"))
    if name == "nkm":
        src = dynare_tests_dir() / "occbin" / "filter"
        work = tmp / "filter"
        shutil.copytree(src, work)
        f = work / "NKM_0_std_shocks.mod"
    else:
        fn = {"zlb": "obc_zlb.mod", "zlb2": "obc_zlb2.mod", "sw": "obc_sw_zlb.mod"}[name]
        shutil.copy(REPO / "reference" / fn, tmp)
        f = tmp / fn
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(f))


def shock_sd(m):
    sd = np.sqrt(np.diag(m.Sigma_e))
    sd[sd < 1e-9] = 0.01
    return sd


def draw_shocks(m, n, s, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, m.n_exo)) * shock_sd(m) * s


# --------------------------------------------------------------------------------------- method plumbing
def make_obj(impl, method, ob, horizon):
    """The solver object for a method on a ready Occbin instance (OccBin itself is the instance)."""
    if method == "occbin":
        return ob
    if method == "lk":                      # LKNews (pydyn.obc_lk): only exists in the current code
        from pydyn.obc_lk import LKNews
        return LKNews(ob, horizon=horizon, check_ahead=CHECK_AHEAD)
    cls = {"news": IMPLS[impl].NewsOBC, "regime": IMPLS[impl].RegimeOBC}[method]
    return cls(ob, horizon=horizon, check_ahead=CHECK_AHEAD)


def simulate(obj, method, shocks, periods=None, init=None):
    periods = len(shocks) if periods is None else periods
    if method == "occbin":
        return obj.simulate(shocks, periods=periods, init=init, check_ahead_periods=OCCBIN_CHECK_AHEAD)
    return obj.simulate(shocks, periods=periods, init=init)


def any_binding(hist_row):
    return any(1 in regs for regs, _starts in hist_row)


def binding_flags(res):
    return np.array([any_binding(h) for h in res.regime_history])


def timer():
    return time.perf_counter()


def summarise(xs):
    return dict(min=min(xs), median=statistics.median(xs), n=len(xs))


# ------------------------------------------------------------------------------------------- calibration
def calibrate(models, seed, grid=(0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12)):
    out = {}
    for name in models:
        m = load_model(name)
        ob = Occbin(m)
        for s in grid:
            shocks = draw_shocks(m, 1000, s, seed)
            row = dict(s=s)
            for meth in MODELS[name]["methods"]:
                obj = make_obj("after", meth, ob, 40)
                t0 = timer()
                try:
                    res = simulate(obj, meth, shocks)
                    row[meth] = dict(share=float(binding_flags(res).mean()), sec=timer() - t0)
                except Exception as err:  # noqa: BLE001
                    row[meth] = dict(error=f"{type(err).__name__}: {str(err)[:80]}",
                                     period=getattr(err, "period", None), sec=timer() - t0)
            print(name, json.dumps(row), flush=True)
            out.setdefault(name, []).append(row)
    return out


# ------------------------------------------------------------------------------------------------------ P1
def instrumented_times(obj, method, shocks):
    """Per-period time of the solver step.  News methods: the step method of the object is wrapped, so the
    time is the in-loop cost.  OccBin has no per-period entry point: one-period calls with ``init`` (the
    time then includes the call's fixed overhead, reported as ``overhead``)."""
    S = len(shocks)
    if method == "occbin":
        times, state, flags = [], np.zeros(obj.n), []
        for t in range(S):
            t0 = timer()
            r = simulate(obj, method, shocks[t:t + 1], periods=1, init=state)
            times.append(timer() - t0)
            state = r.piecewise[0] - obj.ys
            flags.append(any_binding(r.regime_history[0]))
        return np.array(times), np.array(flags)
    name = next(n for n in ("step", "_step", "solve_period") if hasattr(type(obj), n))
    orig = getattr(obj, name)
    times = []

    def wrapped(*a, **k):
        t0 = timer()
        r = orig(*a, **k)
        times.append(timer() - t0)
        return r

    setattr(obj, name, wrapped)
    try:
        res = simulate(obj, method, shocks)
    finally:
        delattr(obj, name)
    return np.array(times), binding_flags(res)


def p1(models, impls, reps, seed, horizon=40, shocks_n=1000, tag="p1", model_cfg=None):
    out = {}
    for name in models:
        cfg = (model_cfg or MODELS)[name]
        m = load_model(name)
        ob = Occbin(m)
        shocks = draw_shocks(m, shocks_n, cfg["s"], seed)
        confs = [(meth, imp) for meth in cfg["methods"] for imp in (["after"] if meth in ("occbin", "lk") else impls)]
        objs = {c: make_obj(c[1], c[0], ob, horizon) for c in confs}
        times = {c: [] for c in confs}
        results, errors = {}, {}
        for rep in range(reps):
            for c in confs:                                   # interleaved
                if c in errors:
                    continue
                t0 = timer()
                try:
                    r = simulate(objs[c], c[0], shocks)
                except Exception as err:  # noqa: BLE001
                    errors[c] = f"{type(err).__name__}: {str(err)[:100]} (period {getattr(err, 'period', None)})"
                    continue
                times[c].append(timer() - t0)
                if rep == 0:
                    results[c] = r
        ref = results.get(("occbin", "after"))
        rows = {}
        for c in confs:
            key = f"{c[0]}/{c[1]}" if c[0] != "occbin" else "occbin"
            if c in errors:
                rows[key] = dict(error=errors[c])
                continue
            row = dict(total_s=summarise(times[c]), us_per_period=summarise([x / shocks_n * 1e6 for x in times[c]]))
            r = results[c]
            flags = binding_flags(r)
            row["share_binding"] = float(flags.mean())
            if ref is not None:
                row["max_path_diff_vs_occbin"] = float(np.abs(r.piecewise - ref.piecewise).max())
                row["regime_histories_equal_occbin"] = bool(r.regime_history == ref.regime_history)
            if c[1] == "after" and (c[0], "before") in results:
                rb = results[(c[0], "before")]
                row["max_path_diff_vs_before"] = float(np.abs(r.piecewise - rb.piecewise).max())
                row["regimes_equal_before"] = bool(r.regime_history == rb.regime_history)
            # split of the in-loop time, one extra pass (noisy: one measurement per period)
            pt, fl = instrumented_times(objs[c], c[0], shocks)
            row["split"] = dict(us_nothing_binds=float(pt[~fl].mean() * 1e6) if (~fl).any() else None,
                                us_binds=float(pt[fl].mean() * 1e6) if fl.any() else None,
                                n_nothing=int((~fl).sum()), n_binds=int(fl.sum()),
                                kind="one-period calls incl. overhead" if c[0] == "occbin" else "in-loop step")
            rows[key] = row
        out[name] = dict(s=cfg["s"], horizon=horizon, rows=rows)
        print(f"[{tag}] {name} done", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------ P2
def p2(models, impls, reps, horizon=40, tag="p2"):
    out = {}
    for name in models:
        cfg = MODELS[name]
        m = load_model(name)
        probe = Occbin(m)
        base = np.array(m.params, float)
        base[probe.switch] = 0.0
        confs = [(meth, imp) for meth in cfg["methods"] for imp in (["after"] if meth in ("occbin", "lk") else impls)]
        T = {c: dict(steady=[], first_order=[], ctor=[], precompute=[], total=[]) for c in confs}
        for rep in range(reps):
            for c in confs:
                t0 = timer()
                ys, params = steady_state(m, base)
                t1 = timer()
                first_order(m, ys, params, loglinear=False)
                t2 = timer()
                ob = Occbin(m)
                t3 = timer()
                obj = make_obj(c[1], c[0], ob, horizon)
                t4 = timer()
                if c[0] == "occbin":
                    pass
                d = T[c]
                d["steady"].append(t1 - t0)
                d["first_order"].append(t2 - t1)
                d["ctor"].append(t3 - t2)
                d["precompute"].append(t4 - t3 if c[0] != "occbin" else 0.0)
                d["total"].append(t4 - t2)
        rows = {}
        for c in confs:
            key = f"{c[0]}/{c[1]}" if c[0] != "occbin" else "occbin"
            rows[key] = {k: {kk: vv * 1e3 if kk != "n" else vv for kk, vv in summarise(v).items()} for k, v in T[c].items()}
        out[name] = dict(horizon=horizon, rows=rows)
        print(f"[{tag}] {name} done", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------ P3
def perturbation_sets(m, names, n, seed, size=0.01):
    rng = np.random.default_rng(seed + 1)
    idx = [m._ip[nm] for nm in names]
    base = np.array(m.params, float)
    sets = []
    for _ in range(n):
        p = base.copy()
        p[idx] = base[idx] * (1 + size * rng.uniform(-1, 1, len(idx)))
        sets.append(p)
    return sets


def p3(models, impls, reps, seed, iters=50, periods=200, horizon=40):
    out = {}
    for name in models:
        cfg = MODELS[name]
        m = load_model(name)
        shocks = draw_shocks(m, 1000, cfg["s"], seed)[:periods]
        sets = perturbation_sets(m, cfg["params"], iters, seed)
        confs = [(meth, imp) for meth in cfg["methods"] for imp in (["after"] if meth in ("occbin", "lk") else impls)]
        tot = {c: [] for c in confs}
        parts = {c: [] for c in confs}
        errors, last = {}, {}
        for rep in range(reps):
            acc = {c: 0.0 for c in confs}
            acc_setup = {c: 0.0 for c in confs}
            for k, p in enumerate(sets):                     # interleaved at the level of one iteration
                for c in confs:
                    if c in errors:
                        continue
                    t0 = timer()
                    try:
                        ob = Occbin(m, p)
                        obj = make_obj(c[1], c[0], ob, horizon)
                        t1 = timer()
                        r = simulate(obj, c[0], shocks)
                    except Exception as err:  # noqa: BLE001
                        errors[c] = f"{type(err).__name__}: {str(err)[:100]} (iteration {k})"
                        continue
                    t2 = timer()
                    acc[c] += t2 - t0
                    acc_setup[c] += t1 - t0
                    if rep == 0:
                        last.setdefault(c, []).append(r)
            for c in confs:
                if c not in errors:
                    tot[c].append(acc[c])
                    parts[c].append(acc_setup[c])
        ref = last.get(("occbin", "after"))
        rows = {}
        for c in confs:
            key = f"{c[0]}/{c[1]}" if c[0] != "occbin" else "occbin"
            if c in errors:
                rows[key] = dict(error=errors[c])
                continue
            row = dict(total_s=summarise(tot[c]), per_iteration_ms={k: v * 1e3 / iters for k, v in summarise(tot[c]).items() if k != "n"},
                       setup_s=summarise(parts[c]))
            if c[1] == "after" and (c[0], "before") in last:
                row["max_path_diff_vs_before"] = max(float(np.abs(a.piecewise - b.piecewise).max()) for a, b in zip(last[c], last[(c[0], "before")]))
                row["regimes_equal_before"] = all(a.regime_history == b.regime_history for a, b in zip(last[c], last[(c[0], "before")]))
            if ref is not None:
                row["max_path_diff_vs_occbin"] = max(float(np.abs(a.piecewise - b.piecewise).max()) for a, b in zip(last[c], ref))
                row["regime_histories_equal_occbin"] = all(a.regime_history == b.regime_history for a, b in zip(last[c], ref))
            rows[key] = row
        out[name] = dict(iters=iters, periods=periods, params=cfg["params"], rows=rows)
        print(f"[p3] {name} done", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------ P4
def p4(impls, reps, seed):
    out = {}
    for name in ("nkm", "sw"):
        out[name] = {}
        for h in (20, 40, 80):
            r1 = p1([name], impls, reps, seed, horizon=h, tag=f"p4-p1 h={h}")[name]
            r2 = p2([name], impls, reps, horizon=h, tag=f"p4-p2 h={h}")[name]
            out[name][h] = dict(p1=r1["rows"], p2=r2["rows"])
    return out


# ------------------------------------------------------------------------------------------------------ R1
def linear_states(ob, m, n_states, seed, spacing=10, burn=500):
    rng = np.random.default_rng(seed + 2)
    sd = shock_sd(m)
    y = np.zeros(ob.n)
    out = []
    for t in range(burn + n_states * spacing):
        y = ob.P @ y + ob.Q @ (rng.standard_normal(m.n_exo) * sd)
        if t >= burn and (t - burn) % spacing == 0:
            out.append(y.copy())
    return np.array(out)


def err_type(err):
    st = getattr(err, "status", None)
    if st:
        return f"OBCError:{st}"
    fl = getattr(err, "error_flag", None)
    if fl:
        return f"{type(err).__name__}:{fl}"
    return f"{type(err).__name__}:{str(err)[:40]}"


def r1(models, impls, seed, n_states=2000, scales=(0.5, 1.0, 1.5, 2.0), milp_limit=2.0, horizon=40):
    out = {}
    for name in models:
        cfg = MODELS[name]
        m = load_model(name)
        ob = Occbin(m)
        states = linear_states(ob, m, n_states, seed)
        z = np.random.default_rng(seed + 3).standard_normal((n_states, m.n_exo)) * shock_sd(m)
        confs = [(meth, imp) for meth in cfg["methods"] for imp in (["after"] if meth in ("occbin", "lk") else impls)]
        objs = {c: make_obj(c[1], c[0], ob, horizon) for c in confs}
        for c, o in objs.items():
            if c[0] == "regime":
                o.milp_time_limit = milp_limit
        out[name] = {}
        for s in scales:
            t_start = timer()
            hist = {c: [None] * n_states for c in confs}
            bind = {c: [False] * n_states for c in confs}
            errs = {c: [None] * n_states for c in confs}
            secs = {c: 0.0 for c in confs}
            for c in confs:
                for i in range(n_states):
                    t0 = timer()
                    try:
                        r = simulate(objs[c], c[0], (z[i] * s)[None, :], periods=1, init=states[i])
                        hist[c][i] = json.dumps(r.regime_history[0])
                        bind[c][i] = any_binding(r.regime_history[0])
                    except Exception as err:  # noqa: BLE001
                        errs[c][i] = err_type(err)
                    secs[c] += timer() - t0
            res = {}
            for c in confs:
                key = f"{c[0]}/{c[1]}" if c[0] != "occbin" else "occbin"
                fails = Counter(e for e in errs[c] if e)
                solved = sum(h is not None for h in hist[c])
                bound = int(sum(bind[c]))
                res[key] = dict(solved=solved, failed=n_states - solved, failures=dict(fails), seconds=secs[c],
                                solved_with_binding=bound)
            pairs = {}
            for i, a in enumerate(confs):
                for b in confs[i + 1:]:
                    if a[0] == b[0]:
                        continue                     # before vs after of the same method is a regression check
                    ka = f"{a[0]}/{a[1]}" if a[0] != "occbin" else "occbin"
                    kb = f"{b[0]}/{b[1]}" if b[0] != "occbin" else "occbin"
                    both = [j for j in range(n_states) if hist[a][j] is not None and hist[b][j] is not None]
                    pairs[f"{ka} vs {kb}"] = dict(
                        both_solved=len(both),
                        different_regimes=sum(hist[a][j] != hist[b][j] for j in both),
                        only_first_fails=sum(hist[a][j] is None and hist[b][j] is not None for j in range(n_states)),
                        only_second_fails=sum(hist[a][j] is not None and hist[b][j] is None for j in range(n_states)),
                        both_fail=sum(hist[a][j] is None and hist[b][j] is None for j in range(n_states)))
            same = {}
            for meth in cfg["methods"]:
                if meth != "occbin" and ("before" in impls and "after" in impls):
                    a, b = (meth, "before"), (meth, "after")
                    same[meth] = dict(identical_outcomes=all(
                        (hist[a][j] == hist[b][j]) and ((errs[a][j] is None) == (errs[b][j] is None)) for j in range(n_states)))
            out[name][str(s)] = dict(methods=res, pairs=pairs, before_vs_after=same)
            print(f"[r1] {name} s={s} done in {timer() - t_start:.0f} s", flush=True)
    return out


# ----------------------------------------------------------------------------------------- warm start
def warm(models, reps, seed, horizon=40):
    """P1 with and without ``warm_start`` (current implementation), interleaved."""
    out = {}
    for name in models:
        cfg = MODELS[name]
        m = load_model(name)
        ob = Occbin(m)
        shocks = draw_shocks(m, 1000, cfg["s"], seed)
        out[name] = {}
        for meth in cfg["methods"]:
            if meth in ("occbin", "lk"):
                continue
            obj = make_obj("after", meth, ob, horizon)
            ts = {False: [], True: []}
            res = {}
            for rep in range(reps):
                for w in (False, True):
                    t0 = timer()
                    r = obj.simulate(shocks, periods=len(shocks), warm_start=w)
                    ts[w].append(timer() - t0)
                    res[w] = r
            it = None
            if meth == "regime":
                its = {}
                for w in (False, True):
                    tot = 0
                    orig = obj._iterate
                    def counting(*a, _o=orig, **k):
                        nonlocal_tot[0] += 1
                        r = _o(*a, **k)
                        nonlocal_tot[1] += r[4] + 1
                        return r
                    nonlocal_tot = [0, 0]
                    obj._iterate = counting
                    obj.simulate(shocks, periods=len(shocks), warm_start=w)
                    del obj._iterate
                    its[w] = tuple(nonlocal_tot)
                it = {str(k): v for k, v in its.items()}
            out[name][meth] = dict(
                us_cold=summarise([x / len(shocks) * 1e6 for x in ts[False]]),
                us_warm=summarise([x / len(shocks) * 1e6 for x in ts[True]]),
                max_diff=float(np.abs(res[True].piecewise - res[False].piecewise).max()),
                same_regimes=bool(res[True].regime_history == res[False].regime_history),
                iterate_calls_and_passes=it)
        print(f"[warm] {name} done", flush=True)
    return out


# --------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calibrate", "p1", "p2", "p3", "p4", "r1", "warm", "all"])
    ap.add_argument("--models", default="zlb,zlb2,nkm,sw")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--impl", default="before,after")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--s", default="", help="override shock scales, e.g. zlb=1,sw=3")
    ap.add_argument("--milp-limit", type=float, default=2.0, help="R1: time limit (s) of RegimeOBC's integer programme")
    ap.add_argument("--n-states", type=int, default=2000, help="R1: number of states")
    ap.add_argument("--scales", default="0.5,1,1.5,2", help="R1: shock scales")
    a = ap.parse_args()
    for kv in filter(None, a.s.split(",")):
        k, v = kv.split("=")
        MODELS[k]["s"] = float(v)
    models, impls = a.models.split(","), a.impl.split(",")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta = dict(seed=a.seed, reps=a.reps, s={k: MODELS[k]["s"] for k in models}, impls=impls,
                check_ahead=CHECK_AHEAD, occbin_check_ahead=OCCBIN_CHECK_AHEAD, python=sys.version.split()[0],
                numpy=np.__version__, date=time.strftime("%Y-%m-%d %H:%M"), load_avg=os.getloadavg())

    def save(key, data):
        p = Path(a.out) / f"{key}{('_' + a.tag) if a.tag else ''}.json"
        p.write_text(json.dumps(dict(meta=meta, data=data), indent=1, default=str))
        print("saved", p, flush=True)

    todo = ["p1", "p2", "p3", "p4", "r1"] if a.what == "all" else [a.what]
    for w in todo:
        if w == "calibrate":
            save("calibrate", calibrate(models, a.seed))
        elif w == "p1":
            save("p1", p1(models, impls, a.reps, a.seed))
        elif w == "p2":
            save("p2", p2(models, impls, a.reps))
        elif w == "p3":
            save("p3", p3(models, impls, a.reps, a.seed))
        elif w == "p4":
            save("p4", p4(impls, a.reps, a.seed))
        elif w == "warm":
            save("warm", warm(models, a.reps, a.seed))
        elif w == "r1":
            save("r1", r1(models, impls, a.seed, n_states=a.n_states, scales=tuple(float(x) for x in a.scales.split(",")),
                          milp_limit=a.milp_limit))


if __name__ == "__main__":
    main()

"""Parts C and D: order-1 simulation soak (occbin vs news) and higher-order news (HigherOrderNews, cubature, failure paths)."""
from __future__ import annotations

import time
import warnings

import numpy as np

import pydyn
from pydyn.obc import NewsOBC, OBCError
from pydyn.obc_ho import CubatureFailure, HigherOrderNews
from pydyn.occbin import Occbin

from lib import (Alarm, alarm, count, exc_info, expected_error_types, load_model, norm_msg, note_time, perturb_params, rng_for)

EXPECTED = {t.__name__ for t in expected_error_types()}
MODELS_C = ["zlb", "zlb2", "nkm", "bf", "borrcon_d", "borrcon", "sw", "nkm_est"]
SEG_LEN = 1000
TASK_CAP = 30.0              # seconds per task of part C
SIM_TIMEOUT = 120


def classify(exc):
    ex = exc_info(exc)
    ex["expected"] = ex["type"] in EXPECTED or any(c.__name__ in EXPECTED for c in type(exc).__mro__)
    ex["code"] = str(getattr(exc, "error_flag", None) or getattr(exc, "status", None) or "")
    return ex


def report_exc(anom, ex, where):
    if ex["expected"]:
        count_key = f"expected:{ex['type']}:{ex['code']}"
        return count_key
    anom(f"exc:{ex['type']}:{norm_msg(ex['msg'])}", **where, msg=ex["msg"], tb=ex["tb"])
    return f"UNEXPECTED:{ex['type']}"


def run_sim(ob, S, periods, init, method, h, sel, anom, where, st, label):
    kw = {}
    if method != "occbin":
        kw = dict(obc_method=method, obc_horizon=h, obc_selection=sel)
    t0 = time.perf_counter()
    try:
        with alarm(SIM_TIMEOUT):
            r = ob.simulate(S, periods=periods, init=init, **kw)
        return r, time.perf_counter() - t0, None
    except Alarm:
        anom("timeout", **where, method=label, seconds=SIM_TIMEOUT)
        return None, time.perf_counter() - t0, "timeout"
    except Exception as exc:                                   # noqa: BLE001
        ex = classify(exc)
        key = report_exc(anom, ex, dict(where, method=label))
        dt = time.perf_counter() - t0
        if dt > 10:
            anom("slow_failure", **where, method=label, seconds=dt, error=key)
        return None, dt, key


BASE_C = {"bf": 0.5, "borrcon_d": 0.1, "borrcon": 0.1, "sw": 0.2}


def is_lcp(ob):
    from pydyn.obc import constraint_objects
    try:
        return all(c.cls == "complementarity" for c in constraint_objects(ob))
    except Exception:                                          # noqa: BLE001
        return False


def check_path(r, ob, anom, where, label):
    P = r.piecewise
    bad = False
    if not np.all(np.isfinite(P)):
        anom("nonfinite_path", **where, method=label)
        bad = True
    elif np.abs(P - ob.ys).max() > 1e6 * max(1.0, np.abs(ob.ys).max()):
        anom("absurd_path", **where, method=label, maxabs=float(np.abs(P - ob.ys).max()))
        bad = True
    w = getattr(r, "news", None)
    if w and w.get("wedges") is not None and is_lcp(ob):          # RegimeOBC wedges (several equations) are signed
        W = np.asarray(w["wedges"])
        sc = max(1.0, np.abs(W).max())
        if not np.all(np.isfinite(W)) or W.min() < -1e-8 * sc:
            anom("wedge_negative_or_nonfinite", **where, method=label, minw=float(np.nanmin(W)), maxw=float(np.nanmax(W)))
            bad = True
    return bad


def task_C(i, st, anom):
    rng = rng_for("C", i)
    name = MODELS_C[i % len(MODELS_C)]
    m = load_model(name)
    pscale = float(rng.choice([0.0, 0.0, 0.02, 0.1, 0.3]))
    params = perturb_params(m, rng, pscale)
    sscale = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))) * BASE_C.get(name, 1.0)
    target = int(rng.choice([1000, 3000, 10000, 30000]))
    h = int(rng.choice([40, 40, 40, 60, 80]))
    where = dict(task=i, model=name, pscale=pscale, shock_scale=round(sscale, 4), params=params.tolist(), horizon=h,
                 rep=f"soak/run.py C --replay {i}")
    rec = dict(i=i, kind="sim", model=name, pscale=pscale, sscale=round(sscale, 3), horizon=h)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ob = Occbin(m, params)
    except Exception as exc:                                   # noqa: BLE001
        ex = classify(exc)
        key = report_exc(anom, ex, dict(where, stage="Occbin()"))
        count(st, f"C|{name}|build|{key}")
        rec["build"] = key
        return rec
    sd = np.sqrt(np.maximum(np.diag(np.array(m.Sigma_e, float)), 0.0))
    done, t_occ, t_news, nbind, fails, agree_max = 0, 0.0, 0.0, 0, {}, 0.0
    init, seg, t_start = None, 0, time.time()
    sel_done = False
    while done < target and time.time() - t_start < TASK_CAP:
        S = rng.standard_normal((SEG_LEN, m.n_exo)) * sd * sscale
        wh = dict(where, seg=seg, period0=done, init=None if init is None else "carried")
        a, ta, ea = run_sim(ob, S, SEG_LEN, init, "occbin", h, "occbin", anom, wh, st, "occbin")
        b, tb, eb = run_sim(ob, S, SEG_LEN, init, "news", h, "occbin", anom, wh, st, "news")
        t_occ, t_news = t_occ + ta, t_news + tb
        for key, tt, lab in ((ea, ta, "occbin"), (eb, tb, "news")):
            if key:
                fails[f"{lab}:{key}"] = fails.get(f"{lab}:{key}", 0) + 1
        bada = a is not None and check_path(a, ob, anom, wh, "occbin")
        badb = b is not None and check_path(b, ob, anom, wh, "news")
        if (a is None) != (b is None) and "timeout" not in (ea or "") + (eb or ""):
            anom(f"one_method_fails:{'occbin' if a is None else 'news'}_fails:{ea or eb}:{name}", **wh)
        if a is not None and b is not None and not (bada or badb):
            d = float(np.abs(a.piecewise - b.piecewise).max())
            rel = d / max(1.0, float(np.abs(a.piecewise - ob.ys).max()))
            agree_max = max(agree_max, rel)
            same_rh = a.regime_history == b.regime_history
            if rel > 1e-6:
                anom(f"path_disagree:{name}", **wh, maxdiff=d, rel=rel, same_regimes=bool(same_rh))
            elif not same_rh:
                anom(f"regime_history_differs_equal_path:{name}", **wh, maxdiff=d)
            nbind += int(sum(1 for reg in a.regime_history if any(c[0][0] for c in reg)))
            init = a.piecewise[-1] - ob.ys
            for lab, tt in (("occbin", ta), ("news", tb)):
                med = note_time(st, f"C|{name}|{lab}", tt / SEG_LEN)
                if med and tt / SEG_LEN > max(20 * med, 0.01):
                    anom("perf_cliff", **wh, method=lab, seconds_per_period=tt / SEG_LEN, median=med)
        else:
            init = None
        # selection variants on one constraint models, short run
        if not sel_done and ob.k == 1 and i % 3 == 0 and a is not None:
            sel_done = True
            for sel in ("shortest", "longest", "min_wedge"):
                r, ts, es = run_sim(ob, S[:200], 200, init, "news", h, sel, anom, dict(wh, selection=sel), st, f"news-{sel}")
                if es:
                    fails[f"news-{sel}:{es}"] = fails.get(f"news-{sel}:{es}", 0) + 1
                elif r is not None:
                    check_path(r, ob, anom, dict(wh, selection=sel), f"news-{sel}")
        done += SEG_LEN
        seg += 1
        # determinism: the first segment again
        if seg == 1 and i % 5 == 0 and b is not None:
            b2, _, _ = run_sim(ob, S, SEG_LEN, None if init is None else None, "news", h, "occbin", anom, wh, st, "news-repeat")
            b0, _, _ = run_sim(ob, S, SEG_LEN, None, "news", h, "occbin", anom, wh, st, "news-repeat2")
            if b2 is not None and b0 is not None:
                count(st, "C|nondet_checks")
                if not np.array_equal(b2.piecewise, b0.piecewise):
                    dd = float(np.abs(b2.piecewise - b0.piecewise).max())
                    anom("nondeterministic:simulate_news" if dd > 1e-10 else "nonbitwise_ulp:simulate_news", **wh, maxdiff=dd)
    rec.update(periods=done, t_occ=round(t_occ, 2), t_news=round(t_news, 2), nbind=nbind, fails=fails, rel=agree_max)
    count(st, f"C|{name}|periods", done)
    count(st, f"C|{name}|binding_periods", nbind)
    for k, v in fails.items():
        count(st, f"C|{name}|fail|{k}", v)
    return rec


# ------------------------------------------------------------------------------------------------ part D
MODELS_D = ["nk", "nkm_est", "zlb2", "borrcon", "sw_cp", "zlb", "nkm"]
CONFIGS_D = [("order2", dict(cubature=False)),
             ("cub_all11", dict(cubature=True)),
             ("cub_r1_l9", dict(cubature=True, regions=1, qmc_level=9)),
             ("cub_r4_l9", dict(cubature=True, regions=4, qmc_level=9)),
             ("order1", dict(order=1, cubature=False)),
             ("cub_all9_tol05", dict(cubature=True, qmc_level=9, cubature_failure_tolerance=0.05)),
             ("cub_all7_tol3", dict(cubature=True, qmc_level=7, cubature_failure_tolerance=0.3)),
             ("cub_r1_l9_tol1", dict(cubature=True, regions=1, qmc_level=9, cubature_failure_tolerance=0.1))]
BASE_SCALE = {"borrcon": 0.1, "sw_cp": 0.25}


def check_ho_result(r, ob, anom, where, tol):
    ys = ob.ys
    for k in ("with_bound", "without_bound"):
        P = r[k]
        if not np.all(np.isfinite(P)):
            anom("nonfinite_path", **where, field=k)
        elif np.abs(P - ys).max() > 1e6 * max(1.0, np.abs(ys).max()):
            anom("absurd_path", **where, field=k, maxabs=float(np.abs(P - ys).max()))
    W = r["wedges"]
    if not np.all(np.isfinite(W)) or (W.size and W.min() < -1e-8 * max(1.0, np.abs(W).max())):
        anom("wedge_negative_or_nonfinite", **where, minw=float(np.nanmin(W)) if W.size else 0.0)
    if tol > 0 and np.any(r["dropped"] > tol * np.maximum(r["relevant"], 1) + 1e-9):
        anom("dropped_exceeds_tolerance", **where, dropped=r["dropped"].tolist()[:10])
    if tol == 0 and r["dropped"].any():
        anom("dropped_with_zero_tolerance", **where)


def check_failure_payload(exc, anom, where):
    c = exc.counts
    bad = []
    if exc.n_failed > exc.n_points or exc.n_relevant > exc.n_points:
        bad.append("counts_exceed_points")
    if not exc.lower_bound and sum(c.values()) != exc.n_failed:
        bad.append(f"class_counts_{sum(c.values())}_vs_n_failed_{exc.n_failed}")
    if exc.n_failed <= 0:
        bad.append("n_failed_nonpositive")
    if getattr(exc, "status", None) not in ("no_solution", "horizon_too_short", "not_converged"):
        bad.append(f"status_{getattr(exc, 'status', None)}")
    if not getattr(exc, "advice", None):
        bad.append("no_advice")
    if bad:
        anom("cubature_failure_payload_inconsistent:" + "+".join(bad), **where, counts=c, n_failed=exc.n_failed,
             n_relevant=exc.n_relevant, n_points=exc.n_points, lower_bound=exc.lower_bound)


def task_D(i, st, anom):
    rng = rng_for("D", i)
    name = MODELS_D[i % len(MODELS_D)]
    m = load_model(name)
    cname, copt = CONFIGS_D[(i // len(MODELS_D)) % len(CONFIGS_D)]
    if cname.startswith("cub_all11") and name == "sw_cp":
        cname, copt = "cub_all9_tol05", CONFIGS_D[5][1]            # level 11 on a 32-variable model is long; covered by tolerance runs
    pscale = float(rng.choice([0.0, 0.0, 0.02, 0.1, 0.3]))
    params = perturb_params(m, rng, pscale)
    sscale = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))) * BASE_SCALE.get(name, 1.0)
    npd = int(rng.choice([12, 20, 40, 60]))
    sd = np.sqrt(np.maximum(np.diag(np.array(m.Sigma_e, float)), 0.0))
    S = rng.standard_normal((npd, m.n_exo)) * sd * sscale
    hz = int(rng.choice([20, 40, 40]))
    where = dict(task=i, model=name, cfg=cname, opts=copt, pscale=pscale, shock_scale=round(sscale, 4), params=params.tolist(), periods=npd,
                 horizon=hz, rep=f"soak/run.py D --replay {i}")
    rec = dict(i=i, kind="ho", model=name, cfg=cname, pscale=pscale, sscale=round(sscale, 3), horizon=hz, periods=npd)
    tol = float(copt.get("cubature_failure_tolerance", 0.0))
    t0 = time.time()
    try:
        with alarm(240):
            with warnings.catch_warnings(record=True) as W:
                warnings.simplefilter("always")
                h = HigherOrderNews(m, params=params, horizon=hz, check_ahead=hz, **{"order": 2, **copt})
                t1 = time.time()
                r = h.simulate(S)
                t2 = time.time()
            # UserWarnings of the documented kinds are catalogued by the caller's record_warnings; replay them here
            for w in W:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        rec.update(ok=True, t_build=round(t1 - t0, 2), t_sim=round(t2 - t1, 2), nbind=int((r["wedges"].max(axis=1) > 1e-10).sum()),
                   relevant=int(r["relevant"].sum()), dropped=int(r["dropped"].sum()), stats=dict(h.stats))
        check_ho_result(r, h.ob, anom, where, tol)
        note_time(st, f"D|{name}|{cname}", (t2 - t1) / npd)
        count(st, f"D|{name}|{cname}|ok")
        if i % 4 == 0:                                         # determinism: identical second run
            h2 = HigherOrderNews(m, params=params, horizon=hz, check_ahead=hz, **{"order": 2, **copt})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2 = h2.simulate(S)
            count(st, "D|nondet_checks")
            if not (np.array_equal(r["with_bound"], r2["with_bound"]) and np.array_equal(r["wedges"], r2["wedges"])):
                dd = max(float(np.abs(r["with_bound"] - r2["with_bound"]).max()), float(np.abs(r["wedges"] - r2["wedges"]).max()))
                anom("nondeterministic:higher_order" if dd > 1e-9 else "nonbitwise_ulp:higher_order", **where, maxdiff=dd)
    except Alarm:
        anom("timeout", **where, seconds=240)
        rec["timeout"] = True
    except Exception as exc:                                   # noqa: BLE001
        ex = classify(exc)
        key = report_exc(anom, ex, where)
        rec["err"] = key
        count(st, f"D|{name}|{cname}|{key}")
        if isinstance(exc, CubatureFailure):
            check_failure_payload(exc, anom, where)
            rec["fail"] = dict(period=getattr(exc, "period", None), n_failed=exc.n_failed, relevant=exc.n_relevant, counts=exc.counts)
        rec["t"] = round(time.time() - t0, 2)
    return rec

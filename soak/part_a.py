"""Parts A and B: inversion-filter likelihood sweeps and MCMC / mode soak on the NKM estimation models and bf."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import pydyn
from pydyn.occbin_estimation import OccbinEstimation
from pydyn.posterior_moments import _prior_draw

from lib import (REF, TESTS, Alarm, Anom, alarm, append, count, exc_info, expected_error_types, note_time, norm_msg,
                 record_warnings, rng_for, rss_mb)

IVF = {"likelihood_inversion_filter": True}
MODELS_A = ["nkm", "nkms", "bf"]
_EST = {}
KNOWN_TOL = {"nkm": 0.15, "nkms": 0.15, "bf": 1e-7}        # largest expected |lp(occbin) - lp(news)| (knife edge, "approx" class)
EVAL_TIMEOUT = 40


def make_est(case, fresh=False):
    if case in _EST and not fresh:
        return _EST[case]
    tmp = Path(tempfile.mkdtemp(prefix="soak_est_"))
    kw = {}
    if case == "bf":
        shutil.copy(TESTS / "estimation/slice/bf.mod", tmp)
        p = tmp / "bf.mod"
        d = loadmat(REF / "occest_bf_data.mat", squeeze_me=True)
        data = np.column_stack([d["q"], d["r"]])
        kw = dict(posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=())
        X = None
    else:
        shutil.copy(REF / f"occest_{case}.mod", tmp)
        p = tmp / f"occest_{case}.mod"
        r = loadmat(REF / f"occest_{case}_lik.mat", squeeze_me=True)
        data = np.atleast_2d(r["data"].T).T
        X = np.atleast_2d(r["X"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(p))
        e = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                             occbin=IVF, obc_method="occbin", **kw)
    if X is not None:
        X = X if X.shape[0] == e.n else X.T
    e.stored = X
    e.case = case
    e.x0v = np.array(e.x0, float)
    if not fresh:
        _EST[case] = e
    return e


def prior_vec(e, rng):
    return np.array(_prior_draw(e.prior, rng, float(e.options.get("prior_trunc") or 0.0)), float)


def base_vector(e, rng):
    """One of the reference vectors: the initial values, a stored Dynare vector."""
    pool = [e.x0v] + ([e.stored[:, j] for j in range(e.stored.shape[1])] if e.stored is not None else [])
    return np.array(pool[rng.integers(len(pool))], float)


def perturbed(e, rng, scale):
    x = base_vector(e, rng)
    return x * np.exp(scale * rng.standard_normal(x.size))


def near_bound(e, rng):
    x = perturbed(e, rng, 0.05)
    k = rng.choice(x.size, size=int(rng.integers(1, min(3, x.size) + 1)), replace=False)
    for j in k:
        eps = 10.0 ** rng.uniform(-9, -2)
        side = rng.integers(2)
        lo, hi = e.lb[j], e.ub[j]
        if side == 0:
            tgt = lo if np.isfinite(lo) else (0.0 if x[j] > 0 else np.nan)
            if np.isfinite(tgt):
                x[j] = tgt + eps * max(1.0, abs(x[j])) * (1 if rng.integers(4) else -1)       # a quarter just outside
        else:
            if np.isfinite(hi):
                x[j] = hi - eps * (1 if rng.integers(4) else -1)
            else:
                x[j] = x[j] * 10.0 ** rng.uniform(0.3, 3)
    return x


def dist_bin(e, x):
    d = float(np.sqrt(np.mean(((x - e.x0v) / np.maximum(np.abs(e.x0v), 1e-8)) ** 2)))
    return next(b for b, lim in zip("abcde", (0.02, 0.1, 0.3, 1.0, np.inf)) if d < lim), d


def evaluate(e, theta, method="occbin", horizon=40, selection="occbin", milp=None, timeout=EVAL_TIMEOUT):
    """One log posterior evaluation; returns a dict (never raises; Alarm and unexpected exceptions are in the result)."""
    e.obc_method, e.obc_horizon, e.obc_selection = method, horizon, selection
    e.obc_milp_time_limit = milp
    e.last_news_solver = None
    e.last_result = None
    out = dict(m=method)
    t0 = time.perf_counter()
    try:
        with alarm(timeout):
            lp = float(e.log_posterior(theta))
    except Alarm:
        lp, out["timeout"] = float("nan"), True
    except Exception as exc:                                   # noqa: BLE001
        lp, out["exc"] = float("nan"), exc_info(exc)
    out["t"] = time.perf_counter() - t0
    out["lp"], out["code"] = lp, int(e.last_code)
    s = e.last_news_solver
    if s is not None:
        out["sv"] = [s.calls, s.fallbacks, s.proved_infeasible, s.timeouts, s.skipped, s.rescued, sum(s.errors.values())]
    r = e.last_result
    if r is not None and np.isfinite(lp) and not out.get("exc"):
        try:
            rh = r.regime_history
            out["nb"] = int(sum(1 for reg in rh if any(c[0][0] for c in reg)))
            out["rh"] = hashlib.md5(str(rh).encode()).hexdigest()[:8]
        except Exception:                                      # noqa: BLE001
            out["nb"] = -1
    return out


def check_eval(r, anom, where, st):
    """Anomaly checks that need one evaluation only; returns the key for the failure table."""
    if r.get("timeout"):
        anom("timeout", **where, method=r["m"], seconds=EVAL_TIMEOUT)
    if "exc" in r:
        ex = r["exc"]
        if not isinstance(ex.get("type"), str) or ex["type"] not in {t.__name__ for t in expected_error_types()}:
            anom(f"exc:{ex['type']}:{norm_msg(ex['msg'])}", **where, method=r["m"], msg=ex["msg"], tb=ex["tb"])
        else:
            anom(f"expected_exc_escaped:{ex['type']}", **where, method=r["m"], msg=ex["msg"])
    lp = r["lp"]
    if np.isnan(lp) and "exc" not in r and "timeout" not in r:
        anom("lp_nan", **where, method=r["m"])
    if lp == np.inf or (np.isfinite(lp) and lp > 1e5):
        anom("lp_absurd", **where, method=r["m"], lp=lp)
    elif np.isfinite(lp) and lp < -1e7:
        count(st, f"A|{where.get('model')}|lp_below_minus_1e7")          # legitimate: a tiny shock sd makes the residuals enormous


def key_of(r):
    return f"{r['m']}|{'ok' if np.isfinite(r['lp']) else 'c' + str(r['code'])}"


# ------------------------------------------------------------------------------------------------ part A
def configs_for(i, case, nc_single=True):
    """The solver configurations evaluated at every vector of task i."""
    cfg = [("occbin", 40, "occbin", None), ("news", 40, "occbin", None)]
    if i % 4 == 0:
        cfg.append(("minnorm", 40, "occbin", None))
    if i % 6 == 0 and case != "bf":
        cfg += [("news", 40, "shortest", None), ("news", 40, "longest", None)]
    if i % 5 == 0:
        cfg += [("news", 20, "occbin", None), ("news", 60, "occbin", None)]
        if i % 10 == 0:
            cfg.append(("occbin", 60, "occbin", None))
    if i % 40 == 5:
        cfg.append(("news", 40, "occbin", 1.0))
    return cfg


def eval_vector(e, theta, cfgs, anom, st, where, dbin):
    res = {}
    for (meth, h, sel, milp) in cfgs:
        label = f"{meth}{'' if h == 40 else h}{'' if sel == 'occbin' else '-' + sel}{'' if milp is None else '-milp'}"
        r = evaluate(e, theta, meth, h, sel, milp)
        r["cfg"] = label
        check_eval(r, anom, dict(where, cfg=label), st)
        # a method may refuse (documented ValueError for selection with several constraints): count only
        count(st, f"A|{e.case}|{label}|{dbin}|{key_of(r)}")
        if np.isfinite(r["lp"]) and "exc" not in r:
            med = note_time(st, f"A|{e.case}|{label}", r["t"])
            if med and r["t"] > max(20 * med, 0.3):
                anom("perf_cliff", **where, cfg=label, seconds=r["t"], median=med)
        elif r["t"] > 5:
            anom("slow_failure", **where, cfg=label, seconds=r["t"], code=r["code"])
        if r.get("sv") and r["sv"][3]:
            anom("milp_timeout_hit", **where, cfg=label, sv=r["sv"])
        res[label] = r
    cmp_pairs(res, e, anom, where, st)
    return res


def cmp_pairs(res, e, anom, where, st):
    a, b = res.get("occbin"), res.get("news")
    if a is None or b is None:
        return
    fa, fb = np.isfinite(a["lp"]), np.isfinite(b["lp"])
    if fa and fb:
        d = abs(a["lp"] - b["lp"])
        a["diff"] = b["diff"] = d
        tol = KNOWN_TOL[e.case]
        if d > tol * max(1.0, abs(a["lp"])) if e.case == "bf" else d > tol + 1e-10 * abs(a["lp"]):
            sfx = "regime_count_differs" if a.get("nb") != b.get("nb") else "same_regime_count"
            anom(f"lp_disagree_large:{e.case}:{sfx}", **where, occbin=a["lp"], news=b["lp"], diff=d, nb=[a.get("nb"), b.get("nb")],
                 rh=[a.get("rh"), b.get("rh")])
        elif d < 1e-9 * max(1.0, abs(a["lp"])) and a.get("rh") != b.get("rh"):
            anom("regime_seq_differs_equal_lp", **where, occbin_rh=a.get("rh"), news_rh=b.get("rh"), nb=[a.get("nb"), b.get("nb")])
        if d > 1e-8 * max(1.0, abs(a["lp"])):
            count(st, f"A|{e.case}|small_diff")
    elif fa != fb:
        w, o = (a, b) if fa else (b, a)
        anom(f"finite_vs_inf:{w['m']}_finite_{o['m']}_code{o['code']}", **where, finite=w["lp"], other_code=o["code"])
    elif not fa and not fb and a["code"] != b["code"]:
        count(st, f"A|{e.case}|both_fail_different_code")
    for lab in ("minnorm", "news-shortest", "news-longest", "news20", "news60"):
        r = res.get(lab)
        if r is None:
            continue
        if np.isfinite(b["lp"]) and not np.isfinite(r["lp"]) and "exc" not in r:
            anom(f"variant_fails_where_news_ok:{lab}", **where, code=r["code"])
        elif np.isfinite(r["lp"]) and np.isfinite(b["lp"]) and abs(r["lp"] - b["lp"]) > 1e-6 * max(1, abs(b["lp"])):
            count(st, f"A|{e.case}|{lab}|differs_from_news")
            if lab in ("news60",):
                anom(f"horizon_dependence:{lab}", **where, lp=r["lp"], news=b["lp"])


def compact(res):
    return {k: [round(v["lp"], 10) if np.isfinite(v["lp"]) else (None if np.isnan(v["lp"]) else "-inf"), v["code"], round(v["t"], 4),
                v.get("nb"), v.get("sv"), v.get("rh")] for k, v in res.items()}


def nondet_check(e, theta, res, anom, st, where, fresh):
    """Same call again in the same process (and in a fresh estimation object): must be identical."""
    for lab in ("occbin", "news"):
        r0 = res[lab]
        again = evaluate(e, theta, lab)
        same = (again["lp"] == r0["lp"]) or (np.isnan(again["lp"]) and np.isnan(r0["lp"]))
        count(st, "A|nondet_checks")
        if not same and np.isfinite(again["lp"]) and np.isfinite(r0["lp"]) and abs(again["lp"] - r0["lp"]) <= 1e-9 * max(1.0, abs(r0["lp"])) and again["code"] == r0["code"]:
            anom("nonbitwise_ulp:same_process", **where, cfg=lab, first=r0["lp"], second=again["lp"])
        elif not same or again["code"] != r0["code"]:
            anom("nondeterministic:same_process", **where, cfg=lab, first=r0["lp"], second=again["lp"], codes=[r0["code"], again["code"]])
        if fresh:
            e2 = make_est(e.case, fresh=True)
            f = evaluate(e2, theta, lab)
            count(st, "A|nondet_fresh_checks")
            same_f = (f["lp"] == r0["lp"]) or (np.isnan(f["lp"]) and np.isnan(r0["lp"]))
            if not same_f and np.isfinite(f["lp"]) and np.isfinite(r0["lp"]) and abs(f["lp"] - r0["lp"]) <= 1e-9 * max(1.0, abs(r0["lp"])) and f["code"] == r0["code"]:
                anom("nonbitwise_ulp:fresh_object", **where, cfg=lab, first=r0["lp"], second=f["lp"])
            elif not same_f or f["code"] != r0["code"]:
                anom("nondeterministic:fresh_object", **where, cfg=lab, first=r0["lp"], second=f["lp"], codes=[r0["code"], f["code"]])


def task_A(i, st, anom):
    case = MODELS_A[i % 3]
    j = i // 3
    cat = j % 4
    rng = rng_for("A", i)
    e = make_est(case)
    where = dict(task=i, model=case, rep=f"soak/run.py A --replay {i}")
    rec = dict(i=i, case=case, cat=cat)
    cfgs = configs_for(i, case)
    if cat == 2:                                                    # line sweep
        pool = [lambda: e.x0v.copy(), lambda: base_vector(e, rng), lambda: perturbed(e, rng, 0.1), lambda: perturbed(e, rng, 0.3),
                lambda: prior_vec(e, rng)]
        x, y = pool[int(rng.integers(len(pool)))](), pool[int(rng.integers(len(pool)))]()
        ts = np.linspace(-0.1, 1.1, 25)
        pts = [x + t * (y - x) for t in ts]
        rec.update(kind="line", x=np.round(x, 8).tolist(), y=np.round(y, 8).tolist())
        out = []
        for t, th in zip(ts, pts):
            _, d = dist_bin(e, th)
            r = eval_vector(e, th, cfgs[:2], anom, st, dict(where, line_t=round(float(t), 4), theta=th.tolist()), "line")
            out.append(compact(r))
            r["_t"] = t
            rec.setdefault("pts", []).append(out[-1])
        # discontinuities along the line: jump of the log posterior between neighbours with the same regime sequence
        for lab in ("occbin", "news"):
            seq = [(p[lab][0], p[lab][5], p[lab][3]) for p in rec["pts"]]
            lps = [s[0] if isinstance(s[0], float) else None for s in seq]
            dl = [abs(lps[k + 1] - lps[k]) if lps[k] is not None and lps[k + 1] is not None else None for k in range(len(lps) - 1)]
            okd = [v for v in dl if v is not None]
            if len(okd) >= 5:
                for k, v in enumerate(dl):
                    nb_ = [dl[j] for j in (k - 1, k + 1) if 0 <= j < len(dl) and dl[j] is not None]
                    # a jump is a step much larger than both neighbouring steps (a steep but smooth slope is not)
                    if v is not None and nb_ and v > 5.0 and v > 8 * max(nb_):
                        same_regime = seq[k][1] == seq[k + 1][1]
                        count(st, f"A|{case}|line_jump|{'same_regime' if same_regime else 'regime_change'}")
                        if same_regime:
                            anom(f"discontinuity_same_regimes:{case}", **where, cfg=lab, t0=float(ts[k]), t1=float(ts[k + 1]),
                                 lp0=lps[k], lp1=lps[k + 1], x=np.round(x, 12).tolist(), y=np.round(y, 12).tolist())
        return rec
    if cat == 0:
        theta, rec["kind"] = prior_vec(e, rng), "prior"
    elif cat == 1:
        scale = float(rng.choice([0.01, 0.05, 0.2, 0.5, 1.0, 2.0]))
        theta, rec["kind"], rec["scale"] = perturbed(e, rng, scale), "perturb", scale
    else:
        theta, rec["kind"] = near_bound(e, rng), "bound"
    dbin, d = dist_bin(e, theta)
    rec.update(theta=theta.tolist(), dist=round(d, 4))
    where["theta"] = theta.tolist()
    res = eval_vector(e, theta, cfgs, anom, st, where, dbin)
    rec["r"] = compact(res)
    if i % 7 == 0 or "exc" in res.get("news", {}):
        nondet_check(e, theta, res, anom, st, where, fresh=(i % 28 == 0))
    return rec


# ------------------------------------------------------------------------------------------------ part B
def _chain_init(st, e, c, rng_seed):
    rng = np.random.default_rng([20261124, 2, 1000 + c])
    x0 = base_vector(e, rng)
    x = x0 * np.exp(0.01 * rng.standard_normal(x0.size))
    d = x.size
    return dict(c=c, case=e.case, x=x.tolist(), lp=None, k=0, acc=0, rng=rng.bit_generator.state, mean=x.tolist(),
                C=(np.diag((0.03 * np.abs(x0)) ** 2)).tolist(), na=0, replay=0, visits=0)


def mcmc_segment(st, anom, n_draws, i):
    """n_draws Metropolis steps of the current chain (news), resumable: the chain state (position, adaptive covariance,
    generator state) is stored in st['chain']."""
    ch = st.get("chain")
    if ch is None or ch["k"] >= CHAIN_LEN:
        c = 0 if ch is None else ch["c"] + 1
        st["chains_done"] = st.get("chains_done", 0) + (0 if ch is None else 1)
        case = "nkms" if c % 3 == 2 else "nkm"
        e = make_est(case)
        ch = st["chain"] = _chain_init(st, e, c, 0)
    e = make_est(ch["case"])
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = ch["rng"]
    x = np.array(ch["x"])
    d = x.size
    where = dict(task=i, model=ch["case"], chain=ch["c"], rep="chain segment")
    if ch["lp"] is None:
        r = evaluate(e, x, "news")
        ch["lp"] = r["lp"] if np.isfinite(r["lp"]) else -1e10
    lp = ch["lp"]
    mean, C, na = np.array(ch["mean"]), np.array(ch["C"]), ch["na"]
    nacc = nrep = nreplay_bad = 0
    t0 = time.time()
    times = []
    for _ in range(n_draws):
        L = np.linalg.cholesky(C * (2.38 ** 2 / d) + 1e-12 * np.diag(np.maximum(np.abs(x), 1e-3) ** 2) + 1e-300 * np.eye(d))
        prop = x + L @ rng.standard_normal(d)
        u = rng.uniform()
        r = evaluate(e, prop, "news")
        plp = r["lp"]
        check_eval(r, anom, dict(where, theta=prop.tolist()), st)
        count(st, f"B|{ch['case']}|{key_of(r)}")
        ch["visits"] += 1
        if np.isfinite(plp) and r["t"] > 0.001:
            times.append(r["t"])
        if r["code"] not in (40, 41, 42) or np.isfinite(plp):
            nrep += 1
            if nrep % REPLAY_K == 0:                      # spot-check replay under occbin
                ro = evaluate(e, prop, "occbin")
                ch["replay"] += 1
                check_eval(ro, anom, dict(where, theta=prop.tolist()), st)
                res = {"occbin": ro, "news": r}
                cmp_pairs(res, e, anom, dict(where, theta=prop.tolist(), replay=True), st)
        if np.isfinite(plp) and np.log(u) < plp - lp:
            x, lp = prop, plp
            nacc += 1
        ch["k"] += 1
        # adaptive Metropolis for the first ADAPT draws of the chain
        if ch["k"] <= ADAPT:
            na += 1
            dlt = x - mean
            mean = mean + dlt / (na + 1)
            C = C * (na - 1) / na + np.outer(dlt, x - mean) / na if na > 1 else C
    ch.update(x=x.tolist(), lp=lp, acc=ch["acc"] + nacc, rng=rng.bit_generator.state, mean=mean.tolist(), C=C.tolist(), na=na)
    return dict(i=i, kind="mcmc", chain=ch["c"], case=ch["case"], k=ch["k"], acc=nacc / n_draws, lp=lp, sec=round(time.time() - t0, 2),
                med_t=round(float(np.median(times)), 4) if times else None)


CHAIN_LEN = 6000
ADAPT = 2000
REPLAY_K = 10
SEG = 150


def mode_search(st, anom, i):
    rng = rng_for("B", i)
    case = "nkms" if (i // 6) % 3 == 2 else "nkm"
    e = make_est(case)
    x0 = prior_vec(e, rng) if rng.integers(2) else perturbed(e, rng, 0.2)
    where = dict(task=i, model=case, x0=x0.tolist(), rep="mode search")
    e.obc_method, e.obc_horizon, e.obc_selection, e.obc_milp_time_limit = "news", 40, "occbin", None
    lp0 = evaluate(e, x0, "news")["lp"]
    rec = dict(i=i, kind="mode", case=case, lp0=lp0 if np.isfinite(lp0) else None)
    if not np.isfinite(lp0):
        rec["skipped"] = "start not finite"
        return rec
    t0 = time.time()
    e.obc_method = "news"
    try:
        with alarm(90):
            r = e.mode(x0=x0, mode_compute=4)
        rec.update(lp=float(r.log_posterior), nfev=int(r.nfev), success=bool(r.success), x=np.asarray(r.x).tolist(), msg=str(r.message)[:60])
        if not np.isfinite(r.log_posterior) or r.log_posterior < lp0 - 1e-6:
            anom("mode_worse_than_start", **where, lp0=lp0, lp=float(r.log_posterior))
        ro = evaluate(e, np.asarray(r.x), "occbin")
        rn = evaluate(e, np.asarray(r.x), "news")
        rec["occbin_at_mode"] = ro["lp"]
        cmp_pairs({"occbin": ro, "news": rn}, e, anom, dict(where, theta=np.asarray(r.x).tolist(), at="mode"), st)
    except Alarm:
        rec["timeout"] = True
        anom("mode_search_unfinished_90s", **where, method="news")
    except Exception as exc:                                  # noqa: BLE001
        ex = exc_info(exc)
        if ex["type"] in {t.__name__ for t in expected_error_types()}:
            anom(f"expected_exc_escaped:{ex['type']}", **where, msg=ex["msg"])
        else:
            anom(f"exc:{ex['type']}:{norm_msg(ex['msg'])}", **where, msg=ex["msg"], tb=ex["tb"])
        rec["exc"] = ex["type"]
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def task_B(i, st, anom):
    if i % 6 == 5:
        return mode_search(st, anom, i)
    return mcmc_segment(st, anom, SEG, i)

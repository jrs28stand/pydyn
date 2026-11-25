# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Port-agnostic comparison of a Dynare port's results with Dynare 7.2's own (the Octave reference).

    python tools/accept_compare.py <port_output_dir> [--ref /home/claude/accept_ref] [--known tools/accept_known.json]
                                   [--report out.md] [--only substr ...] [--name pydyn] [--verdicts out.json] [--include-heterogeneity]

A port's output directory holds, per test id (the id is the path of the test file below tests/ without ``.mod``, as in the
reference folder, ``../examples/x/y.mod`` -> ``examples/x/y``):

    <id>.mat   oo_ (and the M_ / options_ fields accept_save.m keeps), Dynare's field names and shapes, -mat v5/v7 format
    <id>.json  {"status": "ok" | "refused" | "error" | "timeout", "message": str, "rng": "octave" | anything else,
                "refused_fields": {"dr.ghx": "reason", ...}      (optional: oo_ paths the port declines to fill),
                "load_s", "first_s", "second_s": timings (optional)}

The comparison: every non-empty field of Dynare's stored ``oo_`` is compared with the port's, passing when
``max|a-b| <= atol + rtol * max|b|`` over the whole field (atol 1e-10; ``dr.eigval`` as the number of eigenvalues, the number of
stable ones and the moduli in [0.1, 10]: the infinite ones are rounding noise).  Cells and struct arrays are compared cell by cell
and field by field.  rtol depends on the *tolerance class* of the field (the agreed protocol):

    det   1e-8   everything deterministic
    ss    1e-5   numerically solved steady states (oo_.steady_state, oo_.dr.ys) when Dynare solved them numerically
                 (options_.steadystate_flag == 0; with a steady_state_model block or a steady-state file they are analytic: det)
    pf    1e-6   perfect-foresight paths (oo_.endo_simul / oo_.exo_simul after perfect_foresight_solver / simul)

Fields downstream of an inexact point inherit its error (see ``Walker``): when the port's steady state differs from Dynare's by
more than 1e-8 but within the ss class, the other deterministic fields get rtol = max(class, K * the steady-state relative
difference) with K = 50 (or an absolute difference within K * the steady state's absolute difference).  Every pass that needed
such a relaxation is a *relaxed pass* (counted as a match, listed in the report with the tolerance used); every mismatch names
the tolerance class it was judged by.  When estimation ended at a different optimiser end point with the same log posterior (to 1e-2),
the mode fields and what is computed at them are *skipped* (not matches) if within K * the mode's difference, capped at 0.1.

Fields that depend on random numbers are compared only when the port's json says ``"rng": "octave"`` (it replays Octave's
generator); otherwise they are "skipped", never a match (see ``RefContext``: simulated paths and moments (periods>0), IRFs and
forecast bands at order >= 2, conditional forecasts, MCMC output and everything computed at the posterior mean, Gibbs draws,
particle-filter and SMM based estimation).  Wall-clock and optimiser bookkeeping fields (runtime, iterations, ...) are never compared.
A field the port does not hold is "refused" (listed in ``refused_fields``) or "not-produced".

Verdicts: match (all compared fields agree, none missing), partial (some agree, others missing/refused/skipped), refused, wrong,
port-error (the port failed or wrote an unreadable result), compare-error (the comparator itself failed or cannot compare a
field: never the port's fault; the message is kept), dynare-failed (Dynare itself failed or timed out in the reference run), xfail
(Dynare's own expected failure), no-comparison, excluded (heterogeneous-agent tests: ids with a ``heterogeneity`` path component, or
whose reference holds M_.heterogeneity / oo_.heterogeneity; left out of totals and timing until a port supports them, switch
back on with --include-heterogeneity).  ``--known`` is a json {id: {"class": ..., "note": ...}} of explained differences (Dynare
quirks ...): they are printed beside the file in the report, the verdict stays "wrong".
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

ATOL = 1e-10
TOL = {"det": 1e-8, "ss": 1e-5, "pf": 1e-6}          # the agreed protocol's tolerance classes
K_INHERIT = 50.0                                     # error amplification allowed from an inexact point to what is computed at it
END_POINT_CAP = 0.1                                  # bound (relative, and absolute) for fields computed at a different optimiser end point
ORDER = ["match", "partial", "refused", "wrong", "port-error", "compare-error", "dynare-failed", "xfail", "no-comparison"]
EXCLUDED = "excluded"

# oo_ fields that are bookkeeping, not results
IGNORE_TOP = {"dynare_version", "gui", "exo_det_steady_state", "initial_exo_steady_state", "initial_steady_state", "initval_series",
              "exo_det_simul", "time", "deterministic_simulation", "dr_eigval", "endo_simul_orig", "pfm"}
# wall-clock fields (leaf names), and the optimiser's own bookkeeping below an ``optimization_info`` struct
TIMING_LEAVES = {"runtime", "elapsed", "elapsed_time", "cputime", "cpu_time", "walltime", "wall_time", "duration", "tictoc", "time"}
OPTIM_BOOKKEEPING = {"iterations", "funccount", "exitflag", "message", "algorithm", "output", "evaluations", "nevals"}
# not results of the chain / the noisy objective even when they sit in an estimation file
STRUCTURAL = {"var_list", "dr.order_var", "dr.inv_order_var", "dr.state_var", "dr.restrict_var_list", "dr.restrict_columns", "exo_steady_state"}
SS_FIELDS = ("steady_state", "dr.ys")
PF_FIELDS = ("endo_simul", "exo_simul")
SIM_MOMENT_FIELDS = ("mean", "var", "skewness", "kurtosis", "contemporaneous_correlation")
MCMC_PREFIXES = ("posterior.metropolis", "posterior_mean", "posterior_median", "posterior_std", "posterior_variance", "posterior_deciles",
                 "posterior_hpdinf", "posterior_hpdsup", "posterior_density", "PosteriorIRF", "PosteriorTheoreticalMoments",
                 "PriorTheoreticalMoments", "MarginalDensity.ModifiedHarmonicMean", "MarginalDensity.hssmc", "convergence",
                 "MeanForecast", "PointForecast", "RecursiveForecast", "ConditionalForecast", "Forecast", "posterior_function_results",
                 "prior_function_results")
GIBBS_PREFIXES = ("olsgibbs", "surgibbs", "bvar.posterior", "bnlms")
MODE_PREFIXES = ("posterior.optimization", "posterior_mode", "posterior_std_at_mode", "mle_mode", "mle_std_at_mode",
                 "likelihood_at_initial_parameters", "MarginalDensity.LaplaceApproximation", "dsge_var.posterior_mode")
MODE_VALUE_PREFIXES = ("posterior.optimization.mode", "posterior_mode", "mle_mode")     # the optimiser end point itself


def under(p, prefixes):
    return any(p == q or p.startswith(q + ".") for q in prefixes)


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


def scalar(x, default=None):
    try:
        return float(np.ravel(x)[0])
    except (TypeError, ValueError, IndexError):
        return default


# ------------------------------------------------------------------------------------------ numerical comparison
def to_float(x):
    a = np.asarray(x)
    if a.dtype == object:
        raise TypeError("object array")
    return a.astype(complex) if np.iscomplexobj(a) else a.astype(float)


def compare(a, b, rtol=TOL["det"], atol=ATOL):
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


def rel_diff(a, b):
    """max|a-b| / max|b| over the entries finite in both (inf if the shapes or finiteness differ, None if not numeric)."""
    try:
        a, b = to_float(a).ravel(), to_float(b).ravel()
    except (TypeError, ValueError):
        return None
    if a.size != b.size:
        return np.inf
    m = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isfinite(a) != np.isfinite(b)):
        return np.inf
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))


def finite_moduli(x, lo=0.1, hi=10.0):
    """dr.eigval: number of eigenvalues, number below the qz criterion, moduli in [lo, hi] (the rest is rounding noise)."""
    a = np.sort(np.abs(np.asarray(x).ravel()))
    return np.concatenate([[float(a.size), float(np.sum(a < 1 + 1e-6))], a[(a >= lo) & (a <= hi)]])


def _pre(where):
    return where + ": " if where else ""


def cmp_any(a, b, rtol, atol=ATOL, where=""):
    """Compare any Dynare value (number array, string, cell, struct, struct array, cell of structs) ->
    (kind, max abs diff, message) with kind in ok / wrong / missing / not-compared."""
    if is_struct(b):
        if not is_struct(a):
            return "wrong", np.inf, f"{_pre(where)}struct expected, got {type(a).__name__}"
        mx, bad = 0.0, None
        for f in fields_of(b):
            bv = getattr(b, f)
            if not nonempty(bv):
                continue
            av = getattr(a, f, None)
            if av is None or not nonempty(av):
                return "missing", np.inf, f"{where}.{f}: no value from the port"
            k, m, msg = cmp_any(av, bv, rtol, atol, f"{where}.{f}")
            mx = max(mx, m if np.isfinite(m) else mx)
            if k != "ok" and bad is None:
                bad = (k, m, msg)
        return bad if bad else ("ok", mx, "")
    if isinstance(b, (str, bytes)):
        return ("ok", 0.0, "") if str(a) == str(b) else ("wrong", np.inf, f"{_pre(where)}{a!r} vs {b!r}")
    barr = np.asarray(b)
    if barr.dtype == object:
        aarr = np.asarray(a, dtype=object)
        if aarr.size != barr.size:
            return "wrong", np.inf, f"{_pre(where)}cell count {aarr.size} vs Dynare {barr.size}"
        mx, bad = 0.0, None
        for i, (x, y) in enumerate(zip(aarr.ravel(), barr.ravel())):
            if not nonempty(y):
                continue
            k, m, msg = cmp_any(x, y, rtol, atol, f"{where}{{{i + 1}}}")
            mx = max(mx, m if np.isfinite(m) else mx)
            if k != "ok" and bad is None:
                bad = (k, m, msg)
        return bad if bad else ("ok", mx, "")
    if barr.dtype.kind in "US":
        aarr = np.asarray(a)
        return ("ok", 0.0, "") if aarr.shape == barr.shape and np.all(aarr == barr) else ("wrong", np.inf, f"{_pre(where)}text differs")
    if np.asarray(a).dtype == object:
        return "not-compared", np.inf, f"{_pre(where)}port holds a cell/struct where Dynare has numbers"
    ok, mx, msg = compare(a, b, rtol, atol)
    return ("ok" if ok else "wrong"), mx, (msg or f"{_pre(where)}max abs diff {mx:.2e}")


# ------------------------------------------------------------------------------------------ what the reference says about the run
def is_heterogeneity(tid, ref_oo=None, ref_M=None):
    """Heterogeneous-agent test: a ``heterogeneity`` path component in the id, or the reference run holds heterogeneity results."""
    return ("heterogeneity" in tid.split("/")) or nonempty(sget(ref_oo, "heterogeneity")) or nonempty(sget(ref_M, "heterogeneity"))


class RefContext:
    """What Dynare's own stored run tells about which fields are random, which are exact, and which tolerance class applies.

    Everything is read from the *reference* (never from the port), so the rules are the same for every port.  ``options_`` as
    stored is the state after the last command, and stoch_simul restores its options, so per-call options (periods, order) are
    inferred from results where they can be:

    - ``pf``            oo_.deterministic_simulation is there: oo_.endo_simul / exo_simul are perfect-foresight paths (class pf).
    - ``sim_paths``     the paths are written by a stochastic simulation: oo_.mean is their sample mean, or (not pf) a perturbation
                        solution / extended path exists (stoch_simul(periods>0) and extended_path are the writers) and the paths vary
                        over time (constant paths, steady state repeated, are deterministic).  Paths of backward-model simul, model
                        inversion, VAR/PAC simulation ... are deterministic recursions on the mod file's own innovations: class pf,
                        not random (a mod file that draws its own randn cannot be told apart and would show as a mismatch).
    - ``sim_moments``   oo_.mean/var/skewness/kurtosis/contemporaneous_correlation come from simulated data: either oo_.mean is the
                        sample mean of the stored oo_.endo_simul (direct evidence), or options_.periods > 0 and oo_.var is not the
                        theoretical oo_.gamma_y{1} (Dynare's stoch_simul computes theoretical moments if and only if periods == 0).
    - ``high_order``    oo_.dr.ghxx present: order >= 2, so IRFs and forecast bands are Monte Carlo (replic / forecast_replic draws).
    - ``mcmc``          oo_.posterior.metropolis or oo_.posterior_mean present: a Markov chain ran, and what follows (parameters set to the
                        posterior mean: dr, smoothers, IRFs, moments, forecasts) depends on it.
    - ``noisy_objective``  options_.particle.status (particle filter likelihood is random) or the reference log is of a simulated
                        method of moments run (simulated moments): the estimation outputs are random.
    - ``ss_numeric``    options_.steadystate_flag == 0 (no steady_state_model / steady-state file): the steady state is solver output.
    """

    def __init__(self, ref_oo, ref_opts, log_text=""):
        oo = ref_oo
        self.pf = nonempty(sget(oo, "deterministic_simulation"))
        es = sget(oo, "endo_simul")
        self.sim_moments_direct = self._moments_from_paths(oo, ref_opts)
        periods = scalar(sget(ref_opts, "periods"), 0.0)
        self.sim_moments = bool(self.sim_moments_direct or (periods > 0 and nonempty(sget(oo, "var")) and not self._var_is_theoretical(oo)))
        random_writer = nonempty(sget(oo, "dr.ghx")) or nonempty(sget(oo, "extended_path"))       # stoch_simul(periods>0) / extended path
        self.sim_paths = bool(self.sim_moments_direct or (not self.pf and random_writer and (self._varies(es) or self._varies(sget(oo, "exo_simul")))))
        self.high_order = nonempty(sget(oo, "dr.ghxx")) or nonempty(sget(oo, "dr.ghs2"))
        self.mcmc = nonempty(sget(oo, "posterior.metropolis")) or nonempty(sget(oo, "posterior_mean"))
        self.particle = bool(scalar(sget(ref_opts, "particle.status"), 0.0))
        self.smm = "simulated method of moments" in log_text.lower()
        self.noisy_objective = self.particle or self.smm
        flag = scalar(sget(ref_opts, "steadystate_flag"), 0.0)
        self.ss_numeric = (flag == 0)

    @staticmethod
    def _varies(x):
        try:
            a = np.asarray(x, dtype=float)
        except (TypeError, ValueError):
            return False
        if a.ndim != 2 or a.size == 0:
            return False
        a = a if a.shape[0] <= a.shape[1] else a.T            # time along columns
        if a.shape[1] < 2:
            return False
        d = a - a[:, :1]
        d = d[np.isfinite(d)]
        return bool(d.size and np.max(np.abs(d)) > 1e-12 * max(1.0, float(np.nanmax(np.abs(a)))))

    @staticmethod
    def _moments_from_paths(oo, opts):
        m, es = sget(oo, "mean"), sget(oo, "endo_simul")
        try:
            m, es = np.asarray(m, float).ravel(), np.asarray(es, float)
        except (TypeError, ValueError):
            return False
        if m.size == 0 or es.ndim != 2 or es.shape[1] < 3:
            return False
        for d in sorted({int(scalar(sget(opts, "drop"), 100)), 0}):
            if d < es.shape[1] - 1:
                rm = es[:, d:].mean(axis=1)
                if all(np.min(np.abs(rm - x)) <= 1e-9 * max(1.0, abs(x)) for x in m):
                    return True
        return False

    @staticmethod
    def _var_is_theoretical(oo):
        g, v = sget(oo, "gamma_y"), np.asarray(sget(oo, "var"), float)
        if g is None or not isinstance(g, np.ndarray) or g.dtype != object or g.size == 0:
            return False
        try:
            g0 = np.asarray(g.ravel()[0], float)
            return g0.shape == v.shape and bool(np.allclose(g0, v, rtol=1e-9, atol=1e-14, equal_nan=True))
        except (TypeError, ValueError):
            return False

    # -------------------------------------------------------------------------------- per field
    def bookkeeping(self, p):
        parts = p.split(".")
        leaf = parts[-1].lower()
        if leaf == "optimization_info":
            return "optimiser bookkeeping"
        if leaf in TIMING_LEAVES or leaf.endswith(("runtime", "_elapsed", "_cputime")):
            return "wall-clock timing"
        if "optimization_info" in parts[:-1] and leaf in OPTIM_BOOKKEEPING:
            return "optimiser bookkeeping"
        return ""

    def rng_reason(self, p):
        """Why ``p`` depends on random numbers in Dynare's run ('' if it does not)."""
        top = p.split(".")[0]
        if p in PF_FIELDS and self.sim_paths:
            return "simulated paths (random shocks)"
        if top in SIM_MOMENT_FIELDS and self.sim_moments:
            return "moments of simulated data (periods > 0)"
        if top == "irfs" and self.high_order:
            return "Monte Carlo IRFs (order >= 2, replic)"
        if p.startswith("forecast.HPD") and self.high_order:
            return "simulated forecast bands (order >= 2)"
        if p.startswith(("conditional_forecast.cond", "conditional_forecast.uncond")):
            return "Monte Carlo conditional forecast (replic draws)"
        if under(p, MCMC_PREFIXES):
            return "posterior sampler output / posterior draws"
        if under(p, GIBBS_PREFIXES):
            return "Gibbs sampler draws"
        if p in STRUCTURAL or p.startswith("prior."):
            return ""
        if self.noisy_objective:
            return "estimated on a random objective (" + ("particle filter likelihood" if self.particle else "simulated method of moments") + ")"
        if self.mcmc and not under(p, MODE_PREFIXES):
            return "computed at the posterior mean of an MCMC chain"
        return ""

    def tol_class(self, p):
        if p in SS_FIELDS and self.ss_numeric:
            return "ss"
        if p in PF_FIELDS and not self.sim_paths:
            return "pf"
        return "det"


# ------------------------------------------------------------------------------------------ field walk
class Walker:
    def __init__(self, ref_oo, port_oo, refused, rng_ok, ctx):
        self.ref_oo, self.port_oo, self.refused, self.rng_ok, self.ctx = ref_oo, port_oo, refused, rng_ok, ctx
        self.rows, self.relaxed = [], []
        self.e_ss, self.d_ss = self.steady_state_error()
        self.ss_relaxed = bool(ctx.ss_numeric and TOL["det"] < self.e_ss <= TOL["ss"])
        self.endpoint, self.d_mode, self.r_mode = self.same_optimum()
        self.walk(ref_oo, port_oo, "")

    def steady_state_error(self):
        """(max relative, max absolute) difference of the port's steady state from Dynare's over oo_.steady_state and oo_.dr.ys."""
        e = d = 0.0
        for p in SS_FIELDS:
            a, b = sget(self.port_oo, p), sget(self.ref_oo, p)
            if a is not None and nonempty(a) and nonempty(b):
                r = rel_diff(a, b)
                if r is not None and np.isfinite(r):
                    e = max(e, r)
                    d = max(d, float(np.max(np.abs(to_float(a).ravel() - to_float(b).ravel()))) if to_float(a).size == to_float(b).size else 0.0)
                elif r is not None:
                    return np.inf, np.inf
        return e, d

    def same_optimum(self):
        """Both programs maximised the same objective and ended at the same value of it (oo_.posterior.optimization.log_density
        within 1e-2 absolute; the objective is ~1e3) but at a different point (the mode differs by more than rounding): the
        optimiser's end point differs by its tolerance, and the mode fields and what is computed at them follow.  Returns
        (applies, max abs difference of the mode, its difference relative to the mode's largest element)."""
        a, b = sget(self.ref_oo, "posterior.optimization.log_density"), sget(self.port_oo, "posterior.optimization.log_density")
        ma, mb = sget(self.ref_oo, "posterior.optimization.mode"), sget(self.port_oo, "posterior.optimization.mode")
        try:
            a, b = float(np.ravel(a)[0]), float(np.ravel(b)[0])
            ma, mb = to_float(ma).ravel(), to_float(mb).ravel()
        except (TypeError, ValueError, IndexError):
            return False, 0.0, 0.0
        if not (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-2 and ma.size == mb.size and ma.size):
            return False, 0.0, 0.0
        d = float(np.max(np.abs(ma - mb)))
        r = d / max(float(np.max(np.abs(ma))), 1e-300)
        return bool(r > TOL["det"]), d, r

    def is_refused(self, p):
        return any(p == r or p.startswith(r + ".") or r.startswith(p + ".") for r in self.refused)

    def walk(self, ref, got, path):
        for f in fields_of(ref):
            rv = getattr(ref, f)
            p = f"{path}.{f}" if path else f
            if not nonempty(rv) or (not path and f in IGNORE_TOP):
                continue
            why = self.ctx.bookkeeping(p)
            if why:
                self.rows.append((p, "ignored", why))
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
            self.rows.append(self.judge(p, gv, rv))

    # -------------------------------------------------------------------------------- one leaf
    def tolerance(self, p):
        """(rtol, atol, label) for the leaf ``p``."""
        cls = self.ctx.tol_class(p)
        rtol, atol, label = TOL[cls], ATOL, cls
        inexact = p not in SS_FIELDS and p not in STRUCTURAL
        # a field computed at an inexact point may differ by K times the point's error, measured relative to the field's own
        # scale (rtol) or, for fields near zero where that scale is meaningless, absolutely (atol): the point's absolute error
        if inexact and self.ss_relaxed:
            r = K_INHERIT * self.e_ss
            if r > rtol:
                rtol, atol, label = r, ATOL + K_INHERIT * self.d_ss, f"{cls}+steady-state-inherited"
        if self.endpoint and not under(p, MODE_VALUE_PREFIXES) and p != "posterior.optimization.log_density" and p not in STRUCTURAL:
            r = min(K_INHERIT * self.r_mode, END_POINT_CAP)          # capped: a bound near 1 would excuse anything
            if r > rtol:
                rtol, atol, label = r, ATOL + min(K_INHERIT * self.d_mode, END_POINT_CAP), f"{cls}+optimiser-end-point-inherited"
        if self.endpoint and under(p, MODE_VALUE_PREFIXES):
            atol, label = ATOL + self.d_mode * (1 + 1e-6), "optimiser-end-point"
        if self.endpoint and p == "posterior.optimization.log_density":
            atol, label = 1e-2, "optimiser-end-point"
        return rtol, atol, label

    def judge(self, p, gv, rv):
        rtol, atol, label = self.tolerance(p)
        try:
            kind, mx, msg = self.one(p, gv, rv, rtol, atol)
        except Exception as e:  # noqa: BLE001
            return (p, "compare-error", f"{type(e).__name__}: {str(e)[:160]}")
        if kind == "ignored-ok":
            return (p, "ignored", msg)
        reason = self.ctx.rng_reason(p)
        if self.endpoint and kind == "wrong" and (reason.startswith("posterior sampler") or (self.ctx.mcmc and not under(p, MODE_PREFIXES) and p not in STRUCTURAL and reason)):
            # the chain is a function of the generator, the start (the mode) and the proposal covariance (from the mode's Hessian):
            # with the end point different (by r_mode) a replayed generator gives a different chain
            return (p, "skipped", f"MCMC output from a different optimiser end point (start and proposal covariance differ; mode differs by {self.r_mode:.1e} relative): not reproducible even when replaying the generator")
        if reason and not self.rng_ok and kind in ("ok", "wrong"):
            return (p, "skipped", f"random-number dependent ({reason}); the port does not replay Octave's generator" + ("" if kind == "wrong" else " (agrees, but not counted)"))
        if kind == "ok":
            cls_tol = TOL[self.ctx.tol_class(p)]
            if (rtol > cls_tol or atol > ATOL) and not self.base_pass(p, gv, rv, cls_tol):
                if "end-point" in label:        # explained by the optimiser's tolerance, but not verified: skipped, not a match
                    return (p, "skipped", f"evaluated at a different optimiser end point (same log posterior to 1e-2; mode differs by {self.r_mode:.1e} relative); differs by {mx:.1e}, within the allowance (rtol {rtol:.0e})")
                self.relaxed.append({"field": p, "class": label, "rtol": rtol, "atol": atol, "max_abs_diff": mx})
            elif cls_tol > TOL["det"] and not self.base_pass(p, gv, rv, TOL["det"]):
                self.relaxed.append({"field": p, "class": label, "rtol": cls_tol, "atol": ATOL, "max_abs_diff": mx})
            return (p, "match", "")
        if kind == "missing":
            return (p, "not-produced", msg)
        if kind == "not-compared":
            return (p, "not-compared", msg)
        return (p, "wrong", f"{msg} [class {label}, rtol {rtol:.0e}]" if np.isfinite(mx) or not msg else msg)

    def base_pass(self, p, gv, rv, base):
        try:
            return self.one(p, gv, rv, base, ATOL)[0] == "ok"
        except Exception:  # noqa: BLE001
            return False

    def one(self, p, gv, rv, rtol, atol):
        if p == "dr.eigval":
            ok, mx, msg = compare(finite_moduli(gv), finite_moduli(rv), max(rtol, 1e-6), ATOL)
            return ("ok" if ok else "wrong"), mx, (msg or f"max abs diff {mx:.2e}")
        if p == "dr.sdim":
            ev = np.abs(np.asarray(sget(self.ref_oo, "dr.eigval", [])).ravel())
            if ev.size and (not np.all(np.isfinite(ev)) or ev.max() > 1e4):
                return "ignored-ok", 0.0, "model has infinite/rounding-noise eigenvalues; the stable-count depends on the noise"
        return cmp_any(gv, rv, rtol, atol)


def verdict(rows):
    st = collections.Counter("ignored" if r[1] == "ignored-ok" else r[1] for r in rows)
    pick = lambda kinds, n=3: "; ".join(f"{r[0]}: {r[2]}" for r in [r for r in rows if r[1] in kinds][:n])  # noqa: E731
    if st["wrong"]:
        return "wrong", pick(("wrong",), 4)
    if st["compare-error"]:
        return "compare-error", pick(("compare-error",))
    others = st["refused"] + st["not-produced"] + st["skipped"] + st["not-compared"]
    if st["match"] and others:
        return "partial", pick(("refused", "not-produced", "skipped", "not-compared"))
    if st["match"]:
        return "match", ""
    if st["refused"]:
        return "refused", pick(("refused",), 2)
    if st["not-produced"]:
        return "refused", "not produced: " + "; ".join(r[0] for r in [r for r in rows if r[1] == "not-produced"][:4])
    if st["skipped"] or st["not-compared"]:
        return "no-comparison", pick(("skipped", "not-compared"))
    return "no-comparison", "no comparable oo_ field"


# ------------------------------------------------------------------------------------------ one test / all tests
def read_log(ref_dir, tid):
    f = ref_dir / f"{tid}.log"
    try:
        return f.read_text(errors="replace")
    except OSError:
        return ""


def compare_one(tid, port_dir, ref_dir, include_heterogeneity=False):
    ref_json = json.loads((ref_dir / f"{tid}.json").read_text())
    out = {"file": tid, "rows": [], "notes": [], "dynare_status": ref_json["status"], "relaxed": []}
    het = ("heterogeneity" in tid.split("/"))
    ref_oo = ref_M = ref_opts = None
    if ref_json["status"] == "ok" and (ref_dir / f"{tid}.mat").exists():
        try:
            ref_oo, ref_M, ref_opts = load_mat(ref_dir / f"{tid}.mat")
        except Exception as e:  # noqa: BLE001
            out["verdict"], out["detail"] = "compare-error", f"cannot read the Dynare reference .mat: {type(e).__name__}: {str(e)[:160]}"
            return out
        het = het or is_heterogeneity(tid, ref_oo, ref_M)
    if het and not include_heterogeneity:
        out["verdict"], out["detail"] = EXCLUDED, "heterogeneous-agent model: excluded until a port supports them (Dynare reference: " + ref_json["status"] + ")"
        return out
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
    try:
        port_oo, _, _ = load_mat(port_dir / f"{tid}.mat")
    except NotImplementedError as e:       # e.g. MATLAB v7.3 (HDF5) files: scipy cannot read them, the port's file may be fine
        out["verdict"], out["detail"] = "compare-error", f"cannot read the port's .mat: {e}"
        return out
    except Exception as e:  # noqa: BLE001
        out["verdict"], out["detail"] = "port-error", f"the port's .mat is unreadable: {type(e).__name__}: {str(e)[:160]}"
        return out
    ctx = RefContext(ref_oo, ref_opts, read_log(ref_dir, tid))
    w = Walker(ref_oo, port_oo, pinfo.get("refused_fields", {}), pinfo.get("rng") == "octave", ctx)
    out["rows"], out["relaxed"] = w.rows, w.relaxed
    out["context"] = {k: getattr(ctx, k) for k in ("pf", "sim_paths", "sim_moments", "high_order", "mcmc", "noisy_objective", "ss_numeric")}
    out["steady_state_rel_diff"] = w.e_ss if np.isfinite(w.e_ss) else None
    out["verdict"], out["detail"] = verdict(w.rows)
    return out


def test_ids(ref_dir):
    return sorted(str(f.relative_to(ref_dir))[:-5] for f in ref_dir.rglob("*.json")
                  if not any(p.startswith("_") for p in f.relative_to(ref_dir).parts))


def normalise(msg):
    m = re.sub(r"^[\w.]+: ", "", msg.strip())
    m = re.sub(r"'[^']*'|\"[^\"]*\"", "'..'", m)
    return re.sub(r"\b\d+(\.\d+)?(e[+-]?\d+)?\b", "N", m)[:140]


CHANGES_TEXT = """## Changes to the comparison rules

For the Julia stream (JDyn) and any other port: the comparator is shared, rerun `python tools/accept_compare.py <your output dir> --name <port> --report <file>`.
Everything below is read from Dynare's *reference* run, never from the port, so it is the same for every port.

1. **Tolerance classes** (the agreed protocol): `det` rtol 1e-8 for everything deterministic; `ss` 1e-5 for numerically solved steady
   states (`oo_.steady_state`, `oo_.dr.ys`, only when the reference has `options_.steadystate_flag == 0`, i.e. no `steady_state_model` block and
   no steady-state file: analytic steady states stay at 1e-8); `pf` 1e-6 for perfect-foresight paths (`oo_.endo_simul`, `oo_.exo_simul` when the reference holds
   `oo_.deterministic_simulation`).  atol is 1e-10 throughout.  The earlier special 1e-4 for `endo_simul` is gone.
2. **Fields downstream of an inexact steady state.**  If the port's steady state differs from Dynare's by e_ss (relative, max over
   `steady_state` and `dr.ys`) with 1e-8 < e_ss <= 1e-5 (the ss class), every other deterministic field (dr.ghx, moments, IRFs, smoothers, ...) is judged at
   rtol = max(its class, 50 * e_ss) (or an absolute difference of up to 50 * the steady state's largest absolute difference, for fields near zero where their own scale means nothing).  Outside that window nothing is relaxed: if the steady state agrees to 1e-8 everything downstream must too, and if it is off by more than 1e-5 it is wrong and so is what follows.
   The factor 50 is twice the largest amplification observed in the sweep (the steady-state error reappears 1-10 times in ghx/ghxx/variance decomposition, 23 times in the perfect-foresight path of block_bytecode/ireland).
   Each such pass is a **relaxed pass**: it counts as a match but is listed in the report ("Relaxed passes") with the class and the tolerance used.
   Every mismatch in the wrong table carries `[class ..., rtol ...]`.
3. **Optimiser end point (replaces the old rule "log posterior within 1e-2 => any field within 1e-2 relative is skipped").**  It now needs both programs'
   `posterior.optimization.log_density` to agree to 1e-2 *and* the modes to differ by more than 1e-8 relative (otherwise nothing downstream may differ at all).  Then the mode fields
   (`posterior.optimization.mode`, `posterior_mode.*`, `mle_mode.*`; absolute difference within the observed mode difference) and `log_density` (1e-2) are *skipped* as "different but equally good end point", and
   so is any field computed at the mode (`Variance`, `posterior_std_at_mode`, dr, moments, smoother, ...) whose difference is within 50 x the mode's relative (or absolute) difference, capped at 0.1.
   These are **skips, not matches** (the file is `partial` at best: an end-point difference is explained but not verified); a field that differs more is `wrong`.  Posterior-sampler output and everything computed at the posterior
   mean is skipped under the same condition even for a port that replays Octave's generator, because the chain's start and proposal covariance differ.  The rule is restricted to estimation runs by construction (it needs the optimiser's `log_density`).
4. **Random-number fields are classified from the reference, per run** (compared only if the port's json says `"rng": "octave"`, else `skipped`, a match is not counted either):
   simulated paths (`endo_simul`/`exo_simul` that vary over time and are not perfect-foresight paths); the simulated moments `mean`, `var`, `skewness`, `kurtosis`,
   `contemporaneous_correlation` when they come from simulated data (Dynare's `stoch_simul` computes theoretical moments iff periods == 0; since `stoch_simul` restores its options,
   the final `options_.periods` is only a hint: we use direct evidence (`oo_.mean` equals the sample mean of the stored `oo_.endo_simul`), else periods > 0 and `oo_.var` differs from `oo_.gamma_y{1}`);
   IRFs and `forecast.HPD*` at order >= 2 (`oo_.dr.ghxx` present); `conditional_forecast.cond/uncond` (Monte Carlo);
   posterior sampler output (`posterior.metropolis`, `posterior_mean/median/std/variance/deciles/hpd*/density`, `PosteriorIRF`, `MeanForecast`, ..., marginal density by modified harmonic mean) and
   everything computed at the posterior mean once a chain ran (all but the mode fields and `prior.*`); Gibbs draws (`olsgibbs`, `surgibbs`);
   estimations on a random objective (particle filter; simulated method of moments, recognised from the reference log).  Order-1 IRFs and theoretical moments are now **compared** for every port (before, all `irfs` were treated as random).
5. **Bookkeeping fields are never compared**: wall-clock (`runtime`, `elapsed`, `cputime`, `duration`, `time`, ...) and the optimiser's own `optimization_info.{iterations,funcCount,exitflag,message,...}`.
6. **Heterogeneous-agent tests** get the verdict `excluded` (ids with a `heterogeneity` path component, or whose reference has `M_.heterogeneity` / `oo_.heterogeneity`), whatever the port reported;
   they are left out of the totals, the folder table and the timing, and listed separately.  `--include-heterogeneity` compares them as normal once a port supports them.
7. **Cells, struct arrays, cells of structs** (`oo_.occbin.simul.regime_history`, ...) are compared cell by cell and field by field.  A field the comparator cannot compare is `not-compared` (with the reason; counts like skipped), never an error of the port.
8. **Comparator robustness**: any exception inside the comparator is a field-level `compare-error` and, if nothing else is wrong, the verdict `compare-error` (listed in its own section with the message); `port-error` is only
   for a port that failed, wrote nothing, or wrote an unreadable .mat.  `skewness`/`kurtosis` are no longer ignored (they are compared when deterministic, skipped when simulated).
"""


def report(results, known, ref_dir, name, out_path):
    incl = [r for r in results if r["verdict"] != EXCLUDED]
    excl = [r for r in results if r["verdict"] == EXCLUDED]
    byv = collections.Counter(r["verdict"] for r in incl)
    folder = collections.defaultdict(collections.Counter)
    for r in incl:
        p = r["file"].split("/")
        folder[p[0] if len(p) > 1 else "(root)"][r["verdict"]] += 1
    octave = collections.Counter(json.loads(f.read_text())["status"] for f in ref_dir.rglob("*.json")
                                 if not any(p.startswith("_") for p in f.relative_to(ref_dir).parts) and "status" in json.loads(f.read_text()))
    L = [f"# Acceptance sweep: Dynare 7.2 test suite against {name}\n",
         "Generated by `tools/accept_compare.py` (method, tolerance classes and verdicts: its docstring; rule changes: the last section).\n",
         "## Octave runs of Dynare (the reference)\n", "| status | files |\n|---|---|"]
    L += [f"| {k} | {v} |" for k, v in octave.most_common()]
    L += ["", f"## {name} verdicts\n", "| verdict | files |\n|---|---|"]
    L += [f"| {k} | {byv.get(k, 0)} |" for k in ORDER] + [f"| **total (compared set)** | {len(incl)} |",
                                                          f"| excluded (heterogeneous-agent, not in the total) | {len(excl)} |", ""]
    L += ["## By test folder\n", "| folder | " + " | ".join(ORDER) + " | total |\n|---|" + "---|" * (len(ORDER) + 1)]
    for fo in sorted(folder):
        c = folder[fo]
        L.append(f"| {fo} | " + " | ".join(str(c.get(k, 0)) for k in ORDER) + f" | {sum(c.values())} |")
    if excl:
        L += ["", f"## Excluded: heterogeneous-agent tests ({len(excl)} files)\n",
              "Left out of the totals, folder table and timing until a port supports heterogeneous-agent models (Dynare's own status in brackets).\n"]
        L += [f"- {r['file']} [{r['dynare_status']}]" for r in excl]
    wrong = [r for r in incl if r["verdict"] == "wrong"]
    L += ["", f"## Wrong results ({len(wrong)} files)\n", "Each mismatch names the tolerance class it was judged by.\n",
          "| file | field(s) and size of difference | classification |\n|---|---|---|"]
    for r in wrong:
        bad = "; ".join(f"`{x[0]}` {x[2]}" for x in r["rows"] if x[1] == "wrong")[:360]
        k = known.get(r["file"], {})
        L.append(f"| {r['file']} | {bad} | {k.get('class', 'not yet investigated')}: {k.get('note', '')} |")
    err = [r for r in incl if r["verdict"] == "port-error"]
    L += ["", f"## Port errors ({len(err)} files)\n"]
    L += [f"- {v} x {k}" for k, v in collections.Counter(normalise(r["detail"].split(";")[0]) for r in err).most_common(25)]
    cerr = [r for r in incl if r["verdict"] == "compare-error" or any(x[1] == "compare-error" for x in r["rows"])]
    L += ["", f"## Comparator errors ({len(cerr)} files; the comparator failed or could not compare, not the port)\n"]
    L += [f"- {r['file']}: " + "; ".join(f"`{x[0]}` {x[2]}" for x in r["rows"] if x[1] == "compare-error") + (r["detail"] if not r["rows"] else "") for r in cerr]
    nc = [(r["file"], x[0], x[2]) for r in incl for x in r["rows"] if x[1] == "not-compared"]
    L += ["", f"## Fields the comparator did not compare ({len(nc)} fields)\n"] + [f"- {a}: `{b}` {c}" for a, b, c in nc]
    rel = [(r["file"], d) for r in incl for d in r.get("relaxed", [])]
    pf_rel = [(a, d) for a, d in rel if d["class"] == "pf"]
    rel = [(a, d) for a, d in rel if d["class"] != "pf"]
    L += ["", f"## Relaxed passes ({len(rel)} fields in {len({a for a, _ in rel})} files, plus {len(pf_rel)} perfect-foresight paths in {len({a for a, _ in pf_rel})} files)\n",
          "Counted as match, but only within the tolerance shown: class `ss` (numerically solved steady state, 1e-5), `*+steady-state-inherited` "
          "(downstream of a steady state off by more than 1e-8, rule 2), `pf` (perfect-foresight paths between 1e-8 and 1e-6, listed only as a count).\n",
          "One row per file and class; `rtol used` is the largest tolerance applied in that group.\n",
          "| file | class | fields | rtol used | largest abs diff | fields (first) |\n|---|---|---|---|---|---|"]
    grp = collections.OrderedDict()
    for a_, d in rel:
        grp.setdefault((a_, d["class"]), []).append(d)
    for (a_, cl), ds in grp.items():
        L.append(f"| {a_} | {cl} | {len(ds)} | {max(d['rtol'] for d in ds):.1e} | {max(d['max_abs_diff'] for d in ds):.2e} | "
                 + ", ".join(f"`{d['field']}`" for d in ds[:5]) + (", ..." if len(ds) > 5 else "") + " |")
    sk = collections.Counter()
    for r in incl:
        for x in r["rows"]:
            if x[1] == "skipped":
                m = re.search(r"\((.*?)\); the port", x[2])
                sk[m.group(1) if m else re.split(r" \(|;", x[2])[0][:80]] += 1
    L += ["", "## Skipped fields by reason (random-number dependent fields count here for a port that does not replay Octave's generator)\n"]
    L += [f"- {v} fields: {k}" for k, v in sk.most_common()]
    cc = collections.Counter()
    for r in incl:
        if r["verdict"] in ("refused", "partial"):
            seen = {(normalise(f"{x[0].split('.')[0]}: {x[2]}") if x[1] == "refused" else f"not produced: oo_.{x[0].split('.')[0]}")
                    for x in r["rows"] if x[1] in ("refused", "not-produced")}
            if r["verdict"] == "refused" and not r["rows"]:
                seen.add(normalise(r["detail"]))
            cc.update(seen)
    L += ["", "## Commonest refusal / not-produced reasons\n"] + [f"- {v} files: {k}" for k, v in cc.most_common(30)]
    tim = [r["timing"] for r in incl if r.get("timing")]
    L += ["", "## Timing (seconds, per test file; heterogeneous-agent tests excluded)\n", "| quantity | tests | median | mean | max |\n|---|---|---|---|---|"]
    for k, lab in (("load_s", "process start + import (once per worker)"), ("first_s", "first dynare call incl. preprocessor"), ("second_s", "same call again in the same worker")):
        v = [t[k] for t in tim if k in t]
        if v:
            L.append(f"| {lab} | {len(v)} | {statistics.median(v):.2f} | {statistics.mean(v):.2f} | {max(v):.2f} |")
    L += ["", CHANGES_TEXT]
    Path(out_path).write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("port_dir")
    ap.add_argument("--ref", default="/home/claude/accept_ref")
    ap.add_argument("--known", default=str(Path(__file__).with_name("accept_known.json")))
    ap.add_argument("--report")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--name", default="the port")
    ap.add_argument("--verdicts", help="where to write the per-file verdicts json (default: <port_dir>/_verdicts.json)")
    ap.add_argument("--include-heterogeneity", action="store_true", help="compare heterogeneous-agent tests instead of excluding them")
    a = ap.parse_args(argv)
    ref, port = Path(a.ref), Path(a.port_dir)
    known = json.loads(Path(a.known).read_text()) if Path(a.known).exists() else {}
    results = []
    for tid in test_ids(ref):
        if a.only and not any(o in tid for o in a.only):
            continue
        try:
            r = compare_one(tid, port, ref, a.include_heterogeneity)
        except Exception as e:  # noqa: BLE001  (a comparator failure is never the port's error)
            r = {"file": tid, "rows": [], "relaxed": [], "verdict": "compare-error", "detail": f"{type(e).__name__}: {str(e)[:200]}"}
        results.append(r)
    Path(a.verdicts or port / "_verdicts.json").write_text(json.dumps(results, default=str, indent=1))
    print(dict(collections.Counter(r["verdict"] for r in results)), len(results), "tests")
    if a.report:
        report(results, known, ref, a.name, a.report)
        print("wrote", a.report)
    return results


if __name__ == "__main__":
    main()

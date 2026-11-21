# This file is part of pydyn.
#
# It is a Python port of matlab/estimation/{posterior_sampler_iteration, slice_sampler, rotated_slice_sampler,
# check_posterior_sampler_options, posterior_sampler_initialization, posterior_sampler_core, TaRB_optimizer_wrapper,
# chol_SE, compute_mh_covariance_matrix}.m, matlab/estimation/smc/{hssmc, dsmh, smc_samplers_initialization,
# tempered_likelihood}.m, matlab/estimation/resampler/kitagawa.m, matlab/@dprior/{draw, admissible}.m,
# matlab/distributions/{rand_multivariate_normal, rand_multivariate_student, multivariate_normal_pdf,
# multivariate_student_pdf, compute_prior_mode}.m and matlab/read_key_value_string.m of Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org; chol_SE.m: Copyright (C) 2015 Johannes Pfeifer,
# Copyright (C) 2015-2023 Dynare Team), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare's posterior samplers other than the default random-walk Metropolis-Hastings, and the options they share.

Called by :meth:`pydyn.estimation.Estimation.sample` when ``posterior_sampling_method`` is not the default, when
``posterior_sampler_options`` ask for something the default sampler does not do (a Student proposal), or when
``rng='octave'`` asks for Dynare's random numbers. The samplers return the chain format of
:class:`pydyn.estimation.MCMCResult` (as :class:`SamplerResult`), so :mod:`pydyn.posterior` works on their draws.

MCMC samplers (``posterior_sampler_iteration.m``)
    ``'random_walk_metropolis_hastings'`` with ``proposal_distribution='rand_multivariate_student'``,
    ``'slice'`` (univariate stepping-out slice sampler, the rotated variant along the eigenvectors of the proposal
    covariance, and the multimodal variant along the differences between modes),
    ``'tailored_random_block_metropolis_hastings'`` (TaRB: random blocks, the block's mode found by csminwel, a proposal
    from the inverse Hessian there), ``'independent_metropolis_hastings'``.
Sequential Monte Carlo (``matlab/estimation/smc``): :func:`hssmc` (Herbst and Schorfheide) and :func:`dsmh`
    (dynamic striated Metropolis-Hastings); :meth:`Estimation.sample` returns an :class:`SMCResult` for them.
Refused: ``'dime_mcmc'`` and ``'online'`` (not ported), the options of the slice sampler that act on the initial
    states of the endogenous variables (``draw_init_state_from_smoother``, ``draw_init_state_with_rotated_slice``:
    pydyn does not estimate initial states) and ``maximize`` / ``maximize_using_mh_bounds``.

Bounds, as Dynare 7.2 does it (verified against Dynare's chains): the posterior kernel is evaluated with the prior
support (``prior_trunc`` applied) as bounds, but the samplers check their proposals against the bounds of the mode
search, the prior support intersected with the bounds of the ``estimated_params`` block (``sampler_options.bounds``
set in ``check_posterior_sampler_options.m``), and the slice sampler steps out within those. The manual says the
``estimated_params`` bounds have no consequence for the MCMC; Dynare's code does otherwise, and pydyn follows the code.

Random numbers: ``rng='octave'`` reproduces Dynare under Octave (:mod:`pydyn.octave_rng`): chain ``j`` uses Octave's
``rand``/``randn`` seeded as Dynare seeds it, the starting points of several chains come from the global seed, so a
chain is Dynare's chain draw for draw (to rounding) given the same mode and starting point. Draws from priors other
than normal and uniform cannot be reproduced: Octave's ``gamrnd``/``betarnd``/``wblrnd`` (statistics package) use
``randg``/``rande``, whose state Dynare does not seed. The default (``rng=None``) uses numpy generators seeded from
``seed``.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .octave_rng import NumpyRandom, OctaveRandom, dynare_chain_seeds

__all__ = ["SamplerResult", "SMCResult", "sampler_options", "read_key_value_string", "sample", "slice_iteration",
           "rotated_slice_iteration", "tarb_iteration", "tarb_block", "tarb_blocks", "chol_SE", "prior_mode", "prior_draw", "hssmc", "dsmh",
           "kitagawa", "IMPLEMENTED", "rand_multivariate_normal", "rand_multivariate_student",
           "multivariate_normal_pdf", "multivariate_student_pdf"]

RWMH = "random_walk_metropolis_hastings"
SLICE = "slice"
TARB = "tailored_random_block_metropolis_hastings"
IMH = "independent_metropolis_hastings"
IMPLEMENTED = (RWMH, SLICE, TARB, IMH, "hssmc", "dsmh")
SMC = ("hssmc", "dsmh")
_NOT_PORTED = {"dime_mcmc": "the DIME ensemble sampler (dime.m) is not ported",
               "online": "the online auxiliary particle filter (online_auxiliary_filter.m) needs nonlinear filters, "
                         "which pydyn's estimation does not have"}

# default_option_values.m (posterior_sampler_options.*)
DEFAULTS = {
    RWMH: dict(proposal_distribution="rand_multivariate_normal", student_degrees_of_freedom=3,
               use_mh_covariance_matrix=0, save_tmp_file=0),
    TARB: dict(proposal_distribution="rand_multivariate_normal", student_degrees_of_freedom=3, mode_compute=4,
               new_block_probability=0.25, optim_opt="", save_tmp_file=1),
    SLICE: dict(proposal_distribution="", rotated=0, slice_initialize_with_mode=0, use_mh_covariance_matrix=0,
                WR=None, mode_files=None, mode=None, maximize=False, maximize_using_mh_bounds=False, mode_compute=5,
                initial_step_size=0.8, use_prior_draws=False, save_tmp_file=1, save_iter_info_file=True,
                fast_likelihood_evaluation_for_rejection=False, fast_likelihood_evaluation_for_rejection_penalty=10,
                draw_init_state_from_smoother=False, draw_init_state_with_rotated_slice=False),
    IMH: dict(proposal_distribution="rand_multivariate_normal", use_mh_covariance_matrix=0, save_tmp_file=0),
    "hssmc": dict(steps=25, **{"lambda": 2}, particles=20000, scale=0.5, acpt=1.0, target=0.25),
    "dsmh": dict(H=25, M=20, N=200, G=10, K=10, alpha0=0.2, alpha1=0.3, tau=10),
}
#: options each sampler accepts (check_posterior_sampler_options.m); others give Dynare's "Unknown option" warning
_ACCEPTED = {
    RWMH: {"proposal_distribution", "student_degrees_of_freedom", "use_mh_covariance_matrix", "scale_file",
           "save_tmp_file"},
    TARB: {"proposal_distribution", "student_degrees_of_freedom", "mode_compute", "optim", "new_block_probability",
           "scale_file", "save_tmp_file"},
    IMH: {"proposal_distribution", "student_degrees_of_freedom", "use_mh_covariance_matrix", "save_tmp_file"},
    SLICE: {"rotated", "mode", "mode_files", "slice_initialize_with_mode", "initial_step_size", "use_mh_covariance_matrix",
            "save_tmp_file", "save_iter_info_file", "fast_likelihood_evaluation_for_rejection",
            "fast_likelihood_evaluation_for_rejection_penalty", "use_prior_draws", "draw_init_state_from_smoother",
            "draw_init_state_with_rotated_slice", "maximize", "maximize_using_mh_bounds", "mode_compute"},
    "hssmc": {"target", "steps", "scale", "particles", "lambda"},
    "dsmh": {"H", "M", "N", "G", "K", "tau", "alpha0", "alpha1"},
}
_WARN_NAME = {RWMH: "rwmh_sampler", TARB: "tarb_sampler", IMH: "imh_sampler", SLICE: "slice_sampler",
              "hssmc": "hssmc", "dsmh": "dsmh"}


class SamplerError(Exception):
    pass


# ====================================================================== options
def _str2num(s):
    """MATLAB's str2num for the values found in option strings: numbers, logicals, vectors and matrices."""
    t = s.strip()
    if t in ("true", "false"):
        return t == "true"
    body = t[1:-1] if t.startswith("[") and t.endswith("]") else t
    rows = [r for r in body.split(";")]
    try:
        mat = [[float(v) for v in re.split(r"[\s,]+", r.strip()) if v] for r in rows]
    except ValueError:
        return None
    mat = [r for r in mat if r]
    if not mat or len({len(r) for r in mat}) != 1:
        return None
    a = np.array(mat)
    if a.size == 1:
        v = float(a[0, 0])
        return int(v) if v == int(v) and not re.search(r"[.eE]", t) else v
    return a[0] if a.shape[0] == 1 else a


def read_key_value_string(s) -> list:
    """``read_key_value_string.m``: ``'key1',value1,'key2',value2`` -> [(key, value), ...]. A value that MATLAB's
    ``str2num`` reads is a number (or matrix); otherwise the string without its quotes; ``'(...)'`` sublists are kept
    as strings. The doubled quotes of the preprocessor's JSON are accepted."""
    if not s:
        return []
    s = s.replace("''", "'")
    s = re.sub(r",\s*", ",", s)
    if s.count("(") != s.count(")"):
        raise SamplerError("read_key_value_string: The number of opening and closing brackets does not match!")
    begin = [m.start() for m in re.finditer(r"'\(", s)]
    end = [m.start() + 1 for m in re.finditer(r"\)'", s)]
    if len(begin) != len(end):
        raise SamplerError("read_key_value_string: Sublists could not be identified!")
    commas = [i for i, ch in enumerate(s) if ch == ","]
    commas = [c for c in commas if not any(b <= c <= e for b, e in zip(begin, end))]
    if len(commas) % 2 == 0:
        raise SamplerError("read_key_value_string: Wrong number of Key-Value pairs!")
    out = []
    bounds = [-1] + commas + [len(s)]
    pieces = [s[bounds[k] + 1:bounds[k + 1]] for k in range(len(bounds) - 1)]
    for key, val in zip(pieces[::2], pieces[1::2]):
        key = key.strip()[1:-1]
        num = _str2num(val)
        if num is None:
            v = val.strip()
            if v.startswith("'(") and v.endswith(")'"):
                v = v[2:-2]
            else:
                v = v[1:-1] if len(v) >= 2 and v[0] == "'" and v[-1] == "'" else v
            out.append((key, v))
        else:
            out.append((key, num))
    return out


def sampler_options(method, spec=None, est=None, nblocks=None, warn=True) -> dict:
    """Options of ``method`` (``check_posterior_sampler_options.m``, first call): Dynare's defaults overridden by
    ``spec``, the ``posterior_sampler_options`` string of the ``.mod`` file (``'key',value,...``), a dict or a list of
    pairs. Unknown keys give Dynare's warning; invalid values raise :class:`SamplerError` with Dynare's message."""
    if method in _NOT_PORTED:
        raise NotImplementedError(f"posterior_sampling_method={method!r}: {_NOT_PORTED[method]}")
    if method not in DEFAULTS:
        raise SamplerError(f"check_posterior_sampler_options:: Unknown posterior_sampling_method option {method}")
    opts = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULTS[method].items()}
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        pairs = []
    elif isinstance(spec, str):
        pairs = read_key_value_string(spec)
    elif isinstance(spec, dict):
        pairs = list(spec.items())
    else:
        pairs = list(spec)
    for key, val in pairs:
        if key not in _ACCEPTED[method]:
            if warn:
                warnings.warn(f"{_WARN_NAME[method]}: Unknown option ({key})!")
            continue
        if key == "proposal_distribution":
            if str(val).lower() not in ("rand_multivariate_student", "rand_multivariate_normal"):
                raise SamplerError("initial_estimation_checks:: the proposal_distribution option to estimation takes "
                                   "either rand_multivariate_student or rand_multivariate_normal as options")
            val = str(val).lower()
        elif key == "student_degrees_of_freedom":
            if not np.isscalar(val) or val <= 0:
                raise SamplerError("initial_estimation_checks:: the student_degrees_of_freedom takes a positive integer "
                                   "argument")
        elif key == "new_block_probability":
            if val < 0 or val > 1:
                raise SamplerError("check_posterior_sampler_options:: The tarb new_block_probability must be between 0 "
                                   "and 1!")
        elif key == "initial_step_size":
            if val <= 0 or val >= 1:
                raise SamplerError("check_posterior_sampler_options:: slice initial_step_size must be between 0 and 1")
        elif key == "optim":
            key = "optim_opt"
        elif key == "mode":
            m = np.atleast_2d(np.asarray(val, float))
            val = [m[:, j].copy() for j in range(m.shape[1])]
        opts[key] = val
    if method == SLICE:
        if opts["mode"] is not None and len(opts["mode"]) > 0:
            opts["rotated"] = 1
            opts["WR"] = None
        if opts["draw_init_state_from_smoother"] or opts["draw_init_state_with_rotated_slice"]:
            raise NotImplementedError("slice: draw_init_state_from_smoother / draw_init_state_with_rotated_slice act on "
                                      "estimated initial states, which pydyn does not support")
        if opts["maximize"] or opts["maximize_using_mh_bounds"]:
            raise NotImplementedError("slice: the maximize options (an optimiser in place of the slice draws) are not "
                                      "ported")
        if opts["use_prior_draws"]:
            opts["use_prior_draws"] = True
    if method == "dsmh" and (opts["N"] * opts["G"]) % opts["M"] != 0:
        raise SamplerError(f"check_posterior_sampler_options:: DSMH requires M to divide N*G = {opts['N'] * opts['G']} "
                           "evenly.")
    return opts


# ====================================================================== proposals (matlab/distributions)
def rand_multivariate_normal(r, mean, chol_upper):
    """``rand_multivariate_normal.m``: ``mean + randn(1, n) * chol_upper``."""
    n = chol_upper.shape[0]
    return np.asarray(mean, float) + np.atleast_1d(r.randn(n)) @ chol_upper


def rand_multivariate_student(r, mean, chol_upper, df):
    """``rand_multivariate_student.m`` with one draw: ``mean + (randn(1, dim) * chol) * sqrt(df / sum(randn(1, df).^2))``
    (the normals of the scale drawn after those of the direction)."""
    dim = len(mean)
    z = np.atleast_1d(r.randn(dim)) @ chol_upper
    df_i = int(df)
    if df_i != df:
        raise SamplerError("rand_multivariate_student: student_degrees_of_freedom must be an integer (randn(1, df))")
    w = np.atleast_1d(r.randn(df_i))
    return np.asarray(mean, float) + z * math.sqrt(df / float(np.sum(w ** 2)))


def multivariate_normal_pdf(x, mean, chol_upper, n):
    """``multivariate_normal_pdf.m`` (density, not log; it underflows to 0 where Dynare's does)."""
    d = np.asarray(x, float) - np.asarray(mean, float)
    v = linalg.solve_triangular(chol_upper, linalg.solve_triangular(chol_upper.T, d, lower=True), lower=False)
    return (2 * np.pi) ** (-0.5 * n) * np.prod(np.diag(chol_upper)) ** (-1) * math.exp(-0.5 * float(d @ v))


def multivariate_student_pdf(x, mean, chol_upper, df):
    """``multivariate_student_pdf.m``."""
    if df <= 0:
        raise SamplerError("Degrees of freedom 'df' must be positive")
    d = np.atleast_1d(np.asarray(x, float) - np.asarray(mean, float))
    nn = d.size
    t1 = math.gamma(0.5 * (nn + df)) / (math.gamma(0.5 * df) * (df * np.pi) ** (0.5 * nn))
    t2 = t1 / np.prod(np.diag(chol_upper))
    q = linalg.solve_triangular(chol_upper.T, d, lower=True)          # (X - Mean) / chol, row vector
    return t2 / (1 + float(np.sum(q ** 2)) / df) ** (0.5 * (nn + df))


def chol_SE(A):
    """Generalised Cholesky factorisation of Schnabel and Eskow (``chol_SE.m`` without pivoting): upper triangular
    ``R`` with ``R' R = A + diag(E)``, ``E >= 0`` as small as the algorithm makes it. Returns (R, indefinite, E)."""
    A = np.array(A, float)
    if np.sum(np.abs(A - A.T)) > 1e-8:
        raise SamplerError("A is not symmetric")
    if np.sum(np.abs(A - A.T)) > 0:
        A = (A + A.T) / 2
    n = A.shape[0]
    eps = np.finfo(float).eps
    tau1 = eps ** (1 / 3)
    tau2 = eps ** (1 / 3)
    phase1 = True
    delta = 0.0
    g = np.zeros(n)
    E = np.zeros(n)
    gammma = float(np.max(np.diag(A)))
    if np.any(np.diag(A) < 0):
        phase1 = False
    taugam = tau1 * gammma

    def gersh(A, j):
        gg = np.zeros(n)
        for ii in range(j, n):
            up = 0.0 if ii == 0 else float(np.sum(np.abs(A[ii, j:ii])))
            after = 0.0 if ii == n - 1 else float(np.sum(np.abs(A[ii + 1:n, ii])))
            gg[ii] = up + after - A[ii, ii]
        return gg

    if not phase1:
        g = gersh(A, 0)
    if n == 1:
        delta = tau2 * abs(A[0, 0]) - A[0, 0]
        if delta > 0:
            E[0] = delta
        if A[0, 0] == 0:
            E[0] = tau2
        A[0, 0] = math.sqrt(A[0, 0] + E[0])
    for j in range(n - 1):                        # MATLAB's j = 1..n-1
        if phase1:
            jp1 = j + 1
            if A[j, j] > 0:
                if np.min(np.diag(A[j + 1:, j + 1:]) - A[j + 1:, j] ** 2 / A[j, j]) < taugam:
                    phase1 = False
            else:
                phase1 = False
            if phase1:
                A[j, j] = math.sqrt(A[j, j])
                tempjj = A[j, j]
                for ii in range(jp1, n):
                    A[ii, j] = A[ii, j] / tempjj
                for ii in range(jp1, n):
                    temp = A[ii, j]
                    for k in range(jp1, ii + 1):
                        A[ii, k] = A[ii, k] - temp * A[k, j]
                if j == n - 2:
                    A[n - 1, n - 1] = math.sqrt(A[n - 1, n - 1]) if A[n - 1, n - 1] >= 0 else np.nan
            else:
                g = gersh(A, j)
        if not phase1:
            if j != n - 2:
                normj = float(np.sum(np.abs(A[j + 1:n, j])))
                delta = max(0.0, delta, -A[j, j] + normj, -A[j, j] + taugam)
                E[j] = delta
                A[j, j] = A[j, j] + E[j]
                if A[j, j] != normj:
                    temp = normj / A[j, j] - 1
                    for ii in range(j + 1, n):
                        g[ii] = g[ii] + abs(A[ii, j]) * temp
                A[j, j] = math.sqrt(A[j, j])
                tempjj = A[j, j]
                for ii in range(j + 1, n):
                    A[ii, j] = A[ii, j] / tempjj
                for ii in range(j + 1, n):
                    temp = A[ii, j]
                    for k in range(j + 1, ii + 1):
                        A[ii, k] = A[ii, k] - temp * A[k, j]
            else:
                A[n - 2, n - 1] = A[n - 1, n - 2]
                ev = np.linalg.eigvalsh(A[n - 2:, n - 2:])
                delta = max(0.0, delta, -ev.min() + tau2 * max((ev.max() - ev.min()) / (1 - tau1), gammma))
                if delta > 0:
                    A[n - 2, n - 2] += delta
                    A[n - 1, n - 1] += delta
                    E[n - 2] = delta
                    E[n - 1] = delta
                A[n - 2, n - 2] = math.sqrt(A[n - 2, n - 2])
                A[n - 1, n - 2] = A[n - 1, n - 2] / A[n - 2, n - 2]
                A[n - 1, n - 1] = A[n - 1, n - 1] - A[n - 1, n - 2] ** 2
                A[n - 1, n - 1] = math.sqrt(A[n - 1, n - 1])
    return np.tril(A).T, (not phase1), E


# ====================================================================== priors (@dprior, compute_prior_mode.m)
def prior_mode(prior) -> np.ndarray:
    """``bayestopt_.p5`` (``set_prior.m`` / ``compute_prior_mode.m``): NaN for uniform priors; the first mode of a
    beta prior with two modes."""
    eps = np.finfo(float).eps
    out = np.full(len(prior), np.nan)
    for i, sh in enumerate(prior.shape):
        a, b, p3, p4 = prior.p6[i], prior.p7[i], prior.p3[i], prior.p4[i]
        m = np.nan
        if sh == 1:
            if a > 1 and b > 1:
                m = (a - 1) / (a + b - 2)
            elif a < 1 and b < 1:
                m = 0.0                           # [0; 1]: the first is kept
            elif (a < 1 and b > 1 - eps) or (abs(a - 1) < 2 * eps and b > 1):
                m = 0.0
            elif (a > 1 and b < 1 + eps) or (abs(a - 1) < 2 * eps and b < 1):
                m = 1.0
            elif abs(a - 1) < 2 * eps and abs(b - 1) < 2 * eps:
                m = 0.5
            m = m * (p4 - p3) + p3
        elif sh == 2:
            m = (0.0 if a < 1 else (a - 1) * b) + p3
        elif sh == 3:
            m = prior.p1[i]
        elif sh == 4:
            m = 1 / math.sqrt((b + 1) / a) + p3
        elif sh == 5:
            m = np.nan
        elif sh == 6:
            m = a / (b + 2) + p3
        elif sh == 8:
            m = (0.0 if a <= 1 else b * ((a - 1) / a) ** (1 / a)) + p3
        out[i] = m
    return out


def prior_draw(prior, lb, ub, r, gen=None) -> np.ndarray:
    """One draw from the priors truncated to [lb, ub] (``@dprior/draw.m``): uniform priors from ``r.rand``, normal
    priors from ``r.randn`` (Dynare's numbers with an :class:`~pydyn.octave_rng.OctaveRandom`), the others from the numpy
    generator ``gen`` (Octave's ``randg``/``rande`` are not seeded by Dynare, so these cannot be reproduced)."""
    gen = gen if gen is not None else np.random.default_rng()
    sh = prior.shape
    p = np.full(len(prior), np.nan)
    p3, p4, p6, p7 = prior.p3, prior.p4, prior.p6, prior.p7

    def oob(idx):
        return idx[(p[idx] > ub[idx]) | (p[idx] < lb[idx])]

    idx = np.flatnonzero(sh == 5)
    if idx.size:
        p[idx] = np.atleast_1d(r.rand(idx.size)) * (p4[idx] - p3[idx]) + p3[idx]
        while oob(idx).size:
            p[idx] = np.atleast_1d(r.rand(idx.size)) * (p4[idx] - p3[idx]) + p3[idx]
    idx = np.flatnonzero(sh == 3)
    if idx.size:
        p[idx] = np.atleast_1d(r.randn(idx.size)) * p7[idx] + p6[idx]
        bad = oob(idx)
        while bad.size:
            p[bad] = np.atleast_1d(r.randn(bad.size)) * p7[bad] + p6[bad]
            bad = oob(idx)
    makers = {2: lambda k: gen.gamma(p6[k], p7[k]) + p3[k],
              1: lambda k: (p4[k] - p3[k]) * gen.beta(p6[k], p7[k]) + p3[k],
              4: lambda k: np.sqrt(1.0 / gen.gamma(p7[k] / 2, 2.0 / p6[k])) + p3[k],
              6: lambda k: 1.0 / gen.gamma(p7[k] / 2, 2.0 / p6[k]) + p3[k],
              8: lambda k: p7[k] * gen.weibull(p6[k]) + p3[k]}
    for code in (2, 1, 4, 6, 8):                  # the order of draw.m
        idx = np.flatnonzero(sh == code)
        if idx.size:
            p[idx] = makers[code](idx)
            bad = oob(idx)
            while bad.size:
                p[bad] = makers[code](bad)
                bad = oob(idx)
    return p


# ====================================================================== iterations
def _safe(logpost, x):
    """The ``try ... catch logpost = -inf`` of posterior_sampler_iteration.m."""
    try:
        v = float(logpost(x))
    except Exception:                             # noqa: BLE001 - as Dynare: any failure is a rejection
        return -np.inf
    return v


def slice_iteration(logpost, theta, lb, ub, W1, r, prior_draw_fn=None, use_prior_draws=False, last_posterior=None,
                    log=None):
    """One iteration of Dynare's univariate slice sampler (``slice_sampler.m``) through every parameter in turn.

    ``logpost(x)`` is the log posterior kernel (``-inf`` where it is not defined), ``[lb, ub]`` the sampling box
    (``thetaprior``), ``W1`` the initial interval widths and ``r`` the random source. Returns ``(theta, fxsim, neval,
    use_prior_draws)``: the new point, the kernel at it as Dynare reports it (that of the last parameter updated),
    the evaluations per parameter, and whether prior draws are still tried (``use_prior_draws``)."""
    log = log or (lambda *_: None)
    theta = np.array(theta, float)
    npar = theta.size
    neval = np.zeros(npar)
    sqrt_eps = math.sqrt(np.finfo(float).eps)
    fxsim = np.nan
    if use_prior_draws:
        fxsim = float(last_posterior)
        Z1 = fxsim + math.log(r.rand())
        ilogpo2, nattempts, best = -np.inf, 0, None
        while ilogpo2 < Z1 and nattempts < 10:
            nattempts += 1
            while True:
                cand = prior_draw_fn()
                if np.all(cand >= lb) and np.all(cand <= ub):
                    itest = logpost(cand)
                    if np.isfinite(itest):
                        break
            if itest > ilogpo2:
                ilogpo2, best = itest, cand
        if ilogpo2 >= Z1:
            theta, fxsim = best.copy(), ilogpo2
        else:
            use_prior_draws = False
    islow = np.zeros(npar, bool)
    it = 0                                        # MATLAB's 1-based counter
    while it < npar:
        it += 1
        k = it - 1
        neval[k] = 0
        W = W1[k]
        xold = theta[k]
        XLB, XUB = lb[k], ub[k]
        if XLB == XUB:
            continue
        fxold = logpost(theta)
        if not np.isfinite(fxold):
            log(f"slice_sampler:: Iteration {it} started with bad parameter set (fval is inf or nan)")
            icount = 0
            while not np.isfinite(fxold) and icount < 1000:
                icount += 1
                theta = prior_draw_fn()
                if np.all(theta >= lb) and np.all(theta <= ub):
                    fxold = logpost(theta)
            it, k = 1, 0
            neval[k] = 0
            W = W1[k]
            xold = theta[k]
            XLB, XUB = lb[k], ub[k]
        neval[k] += 1
        Z = fxold + math.log(r.rand())
        u = r.rand()
        L = max(XLB, xold - W * u)
        R = min(XUB, L + W)
        while L > XLB:
            theta[k] = L
            fxl = logpost(theta)
            neval[k] += 1
            if fxl <= Z:
                break
            L = max(XLB, L - W)
            if neval[k] > 30:
                L = XLB
                theta[k] = L
                fxl = logpost(theta)
                icount = 0
                while (np.isinf(fxl) or np.isnan(fxl)) and icount < 300:
                    icount += 1
                    L = L + sqrt_eps
                    theta[k] = L
                    fxl = logpost(theta)
        neval1 = neval[k]
        while R < XUB:
            theta[k] = R
            fxr = logpost(theta)
            neval[k] += 1
            if fxr <= Z:
                break
            R = min(XUB, R + W)
            if neval[k] > neval1 + 30:
                R = XUB
                theta[k] = R
                fxr = logpost(theta)
                icount = 0
                while (np.isinf(fxr) or np.isnan(fxr)) and icount < 300:
                    icount += 1
                    R = R - sqrt_eps
                    theta[k] = R
                    fxr = logpost(theta)
        fxsim = Z - 1
        while fxsim < Z:
            u = r.rand()
            xsim = L + u * (R - L)
            theta[k] = xsim
            fxsim = logpost(theta)
            neval[k] += 1
            if xsim > xold:
                R = xsim
            else:
                L = xsim
            if (R - L) < 1e-6 and fxsim < Z:
                log(f"The sampling for parameter number {it} is taking too long as the sampling set is too tight. "
                    "Check the prior.")
                islow[k] = True
                theta[k] = xold
                fxsim = fxold
                break
        if np.isinf(fxsim) or np.isnan(fxsim):
            theta[k] = xold
            fxsim = fxold
            log("SLICE: posterior density is infinite. Reset values at initial ones.")
    if np.any(islow):
        fxsim = logpost(theta)
        Z1 = fxsim + math.log(r.rand())
        ilogpo2, nattempts, best = -np.inf, 0, None
        while ilogpo2 < Z1 and nattempts < 10:
            nattempts += 1
            while True:
                cand = theta.copy()
                mine = prior_draw_fn()
                cand[islow] = mine[islow]
                if np.all(cand[islow] >= lb[islow]) and np.all(cand[islow] <= ub[islow]):
                    itest = logpost(cand)
                    if np.isfinite(itest):
                        break
            if itest > ilogpo2:
                ilogpo2, best = itest, cand
        if ilogpo2 > Z1:
            theta, fxsim = best, ilogpo2
    return theta, fxsim, neval, use_prior_draws


def rotated_slice_iteration(logpost, theta, lb, ub, V1, W1, r):
    """``rotated_slice_sampler.m`` (no initial-state estimation): a slice step along every column of ``V1`` in turn,
    within the part of the line that stays in the box [lb, ub]; ``W1`` the initial widths (``None``: the whole
    feasible segment). Returns ``(theta, fxsim, neval)``."""
    theta = np.array(theta, float)
    npar = V1.shape[1]
    neval = np.zeros(npar)
    fxsim = np.nan
    for it in range(npar):
        theta0 = theta.copy()
        v = V1[:, it]
        xold = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.sort(np.column_stack([(lb - theta) / v, (ub - theta) / v]), axis=1)
        XLB = float(np.nanmax(tb[:, 0]))          # MATLAB's max and min ignore NaN
        XUB = float(np.nanmin(tb[:, 1]))
        W = (XUB - XLB) if W1 is None else W1[it]
        fxold = logpost(theta)
        neval[it] += 1
        Z = fxold + math.log(r.rand())
        u = r.rand()
        L = max(XLB, xold - W * u)
        R = min(XUB, L + W)
        while L > XLB:
            theta = theta0 + L * v
            fxl = logpost(theta)
            neval[it] += 1
            if fxl <= Z:
                break
            L = max(XLB, L - W)
        while R < XUB:
            theta = theta0 + R * v
            fxr = logpost(theta)
            neval[it] += 1
            if fxr <= Z:
                break
            R = min(XUB, R + W)
        fxsim = Z - 1
        while fxsim < Z:
            u = r.rand()
            xsim = L + u * (R - L)
            theta = theta0 + xsim * v
            fxsim = logpost(theta)
            neval[it] += 1
            if xsim > xold:
                R = xsim
            else:
                L = xsim
    return theta, fxsim, neval


def multimodal_directions(theta, modes, r):
    """Directions of the multimodal rotated slice (``rotated_slice_sampler.m`` with ``sampler_options.mode``): the unit
    vectors from the mode nearest ``theta`` to each other mode, in the order of ``randperm(n - 1)``."""
    n = len(modes)
    dist = [math.sqrt(float(np.sum((theta - m) ** 2))) for m in modes]
    im = int(np.argmin(dist))
    cols = []
    for j in range(n):
        if j != im:
            tmp = modes[j] - modes[im]
            cols.append(tmp / np.linalg.norm(tmp))
    V1 = np.column_stack(cols)
    perm = r.randperm(n - 1, n - 1)
    return V1[:, perm]


def tarb_block(est, current, last_posterior, idx, lb, ub, mlb, mub, jscale, opts, r, optimizer, counter=None):
    """One block of a TaRB-MH iteration (the body of the block loop in ``posterior_sampler_iteration.m``): the block's
    conditional mode from ``current`` by ``optimizer``, the proposal from the inverse Hessian there, Chib and
    Ramamurthy's acceptance. Returns ``(current, last_posterior, accepted, record)``; ``record`` holds the optimum,
    the proposal, the kernel there, the acceptance log-probability and the uniform used."""
    from .estimation import hessian
    counter = [0] if counter is None else counter
    current = np.array(current, float)
    nx = idx.size
    start = current[idx].copy()

    def wrapper(v):
        x = current.copy()
        x[idx] = v
        return x
    xopt = optimizer(lambda v: est.objective(wrapper(v), bounds=(mlb, mub)), start, idx, r, counter)
    info = {}
    if isinstance(xopt, tuple):
        xopt, info = xopt
    rec = dict(block=np.array(idx), xopt=np.array(xopt), **info)

    def fneg(v):
        counter[0] += 1
        return -est.log_posterior(wrapper(v), bounds=(mlb, mub))
    with np.errstate(all="ignore"):
        hmat = np.reshape(hessian(fneg, xopt, est.options["gstep"]), (nx, nx))
    if np.any(~np.isfinite(hmat)):
        ih = np.eye(nx) * 1e-4
    else:
        try:
            ih = np.linalg.inv(hmat)
        except np.linalg.LinAlgError:
            ih = np.full((nx, nx), np.inf)
        if np.any(~np.isfinite(ih)):
            ih = np.eye(nx) * 1e-4
    try:
        cu = np.linalg.cholesky(ih).T
    except np.linalg.LinAlgError:
        cu = chol_SE(ih)[0]
    cu = cu @ np.diag(jscale[idx])
    if opts["proposal_distribution"] == "rand_multivariate_normal":
        n = nx
        prop = rand_multivariate_normal(r, xopt, cu)
    else:
        n = opts["student_degrees_of_freedom"]
        prop = rand_multivariate_student(r, xopt, cu, n)
    if np.all(prop > lb[idx]) and np.all(prop < ub[idx]):
        counter[0] += 1
        logpost = _safe(lambda y: est.log_posterior(y, bounds=(mlb, mub)), wrapper(prop))
    else:
        logpost = -np.inf
    rec.update(chol=cu, proposal=np.array(prop), log_posterior=logpost)
    accepted = False
    if logpost > -np.inf:
        if opts["proposal_distribution"] == "rand_multivariate_normal":
            fwd = multivariate_normal_pdf(prop, xopt, cu, n)
            bwd = multivariate_normal_pdf(start, xopt, cu, n)
        else:
            fwd = multivariate_student_pdf(prop, xopt, cu, n)
            bwd = multivariate_student_pdf(start, xopt, cu, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            accprob = logpost - last_posterior + np.log(bwd) - np.log(fwd)       # (6) of Chib and Ramamurthy
        u = r.rand()
        rec.update(accprob=float(accprob), u=u)
        if math.log(u) < accprob:
            current[idx] = prop
            last_posterior = logpost
            accepted = True
    return current, last_posterior, accepted, rec


def tarb_blocks(npar, new_block_probability, r):
    """Random blocks of a TaRB iteration: ``randperm(npar)`` and a new block after each parameter with probability
    ``new_block_probability`` (``rand(npar - 1, 1)``). Returns the list of 0-based index arrays."""
    indices = r.randperm(npar)
    flags = np.atleast_1d(r.rand(npar - 1)) > (1 - new_block_probability) if npar > 1 else np.zeros(0, bool)
    blocks = np.concatenate([[1], 1 + np.cumsum(flags)]).astype(int)
    return [indices[blocks == b] for b in range(1, int(blocks[-1]) + 1)]


def tarb_iteration(est, theta, last_posterior, lb, ub, mlb, mub, jscale, opts, r, optimizer, trace=None):
    """One TaRB-MH iteration (``posterior_sampler_iteration.m``, case ``tailored_random_block_metropolis_hastings``):
    random blocks (:func:`tarb_blocks`), then :func:`tarb_block` for each: the block's conditional mode found by
    ``optimizer`` from the current values, a proposal centred there with the inverse Hessian (``hessian.m``) as
    covariance (``chol_SE`` if it is not positive definite, ``1e-4 I`` if it is not finite) times the block's
    ``jscale``, accepted with Chib and Ramamurthy's probability. ``lb``/``ub``: bounds of the proposals (sampler
    bounds), ``mlb``/``mub``: bounds of the posterior kernel. ``trace``: a list that receives the record of every
    block. Returns ``(theta, log posterior, share of blocks accepted, function evaluations)``."""
    blocks = tarb_blocks(theta.size, opts["new_block_probability"], r)
    current = np.array(theta, float)
    accepted = 0
    counter = [0]
    for idx in blocks:
        current, last_posterior, a, rec = tarb_block(est, current, last_posterior, idx, lb, ub, mlb, mub, jscale, opts,
                                                     r, optimizer, counter)
        accepted += a
        if trace is not None:
            trace.append(rec)
    return current, last_posterior, accepted / len(blocks), counter[0]


# ====================================================================== results
def _mcmc_result_class():
    from .estimation import MCMCResult

    @dataclass
    class SamplerResult(MCMCResult):
        """:class:`pydyn.estimation.MCMCResult` with the sampler's name and diagnostics. For TaRB ``accepted`` holds
        the share of blocks accepted at each iteration and ``acceptance_rate`` its mean (Dynare's
        ``AcceptanceRatio``); for the slice sampler every iteration counts as accepted and ``neval`` (function
        evaluations per iteration; Dynare's ``FunctionEvalPerIteration`` is its mean) is the useful measure."""
        method: str = RWMH
        neval: np.ndarray = field(default=None, repr=False)
        sampler_options: dict = field(default=None, repr=False)
        proposal_center: np.ndarray = field(default=None, repr=False)     # IMH: the mode
        rotation: np.ndarray = field(default=None, repr=False)            # rotated slice: V1
        rotation_widths: np.ndarray = field(default=None, repr=False)     # rotated slice: WR

        @property
        def function_evaluations_per_iteration(self):
            return None if self.neval is None else self.neval.mean(axis=1)
    return SamplerResult


SamplerResult = None          # set on first use (avoids a circular import with pydyn.estimation)


def _result_cls():
    global SamplerResult
    if SamplerResult is None:
        SamplerResult = _mcmc_result_class()
        globals()["SamplerResult"] = SamplerResult
    return SamplerResult


@dataclass
class SMCResult:
    """Result of a sequential Monte Carlo sampler: the final particles (``particles``: (n, parameters), equally
    weighted), their tempered log posterior kernel (``log_posterior``) and log likelihood, the log marginal data
    density estimate (``log_marginal_density``), and per stage the tempering parameter, effective sample size,
    acceptance rate and scale. ``as_mcmc()`` presents the particles as one chain without burn-in, for
    :mod:`pydyn.posterior`."""
    method: str
    particles: np.ndarray
    log_posterior: np.ndarray
    log_likelihood: np.ndarray
    log_marginal_density: float
    names: list
    tempering: np.ndarray
    ess: np.ndarray
    acceptance: np.ndarray
    scale: np.ndarray
    weights: np.ndarray = None
    stages: list = field(default=None, repr=False)

    def as_mcmc(self):
        from .estimation import MCMCResult
        n = self.particles.shape[0]
        return MCMCResult(draws=self.particles[None].copy(), log_posterior=self.log_posterior[None].copy(),
                          acceptance_rate=np.array([np.nan]), names=list(self.names), x0=self.particles[:1].copy(),
                          drop=0.0, accepted=np.ones((1, n), bool))

    def mean(self):
        return self.particles.mean(axis=0)

    def std(self):
        return self.particles.std(axis=0, ddof=1)


# ====================================================================== random sources
def _sources(rng, random, seed, nblocks, previous_draws=0):
    """(per-chain sources, source for the starting points, numpy generator for prior draws)."""
    if random is not None:
        chains = list(random)
        if len(chains) < nblocks:
            raise SamplerError(f"random: {nblocks} random sources needed (one per chain)")
        init = chains[nblocks] if len(chains) > nblocks else None
        gen = np.random.default_rng(seed if seed is not None else 0)
        return chains[:nblocks], init, gen
    if rng == "octave":
        if previous_draws:
            raise SamplerError("rng='octave' cannot continue chains (Dynare restarts from the generator states it "
                               "stored); use the default generators")
        seeds = dynare_chain_seeds(seed, nblocks)
        return [OctaveRandom(s) for s in seeds], OctaveRandom(seed), np.random.default_rng(seed)
    if rng not in (None, "numpy"):
        raise SamplerError(f"rng={rng!r}: use None (numpy generators) or 'octave'")
    ss = np.random.SeedSequence(seed if not previous_draws else [int(seed), int(previous_draws)])
    ch = ss.spawn(nblocks + 2)
    return ([NumpyRandom(np.random.default_rng(c)) for c in ch[:nblocks]], NumpyRandom(np.random.default_rng(ch[nblocks])),
            np.random.default_rng(ch[nblocks + 1]))


# ====================================================================== driver
def _scale_file(path, est):
    from pathlib import Path
    base = est.model.json_dir.parents[2] if hasattr(est.model, "json_dir") else Path(".")
    for c in (Path(path), Path(str(path) + ".mat"), base / path, base / (str(path) + ".mat")):
        if c.is_file():
            from scipy.io import loadmat
            d = loadmat(c, squeeze_me=True)
            if "Scale" not in d:
                raise SamplerError(f"scale_file {c}: no variable Scale")
            return float(d["Scale"])
    raise SamplerError("initial_estimation_checks:: The specified scale_file does not exist.")


def _load_modes(path, est):
    from pathlib import Path
    base = est.model.json_dir.parents[2] if hasattr(est.model, "json_dir") else Path(".")
    for c in (Path(path), Path(str(path) + ".mat"), Path(str(path) + ".npz"), base / path, base / (str(path) + ".mat"),
              base / (str(path) + ".npz")):
        if c.is_file():
            if c.suffix == ".npz":
                with np.load(c) as z:
                    xp = z["xparams"]
            else:
                from scipy.io import loadmat
                xp = loadmat(c)["xparams"]
            xp = np.atleast_2d(np.asarray(xp, float))
            if xp.shape[1] < 2:
                raise SamplerError(f"check_posterior_sampler_options:: Variable xparams loaded in file <{path}> has size "
                                   f"[{xp.shape[0]}x{xp.shape[1]}]: it must contain at least two columns, to allow "
                                   "multi-modal sampling.")
            return [xp[:, j].copy() for j in range(xp.shape[1])]
    raise SamplerError(f"mode_files {path}: no such file")


def _mh_covariance(previous, drop):
    """Mean and covariance of the kept draws of earlier chains (``compute_mh_covariance_matrix.m``)."""
    from . import posterior as _post
    first = _post.first_kept(previous.draws.shape[1], drop)
    x = previous.draws[:, first:, :].reshape(-1, previous.draws.shape[2])
    return _post._recursive_moments(x)


def rotation_from_covariance(V):
    """``[V1, D] = eig(invhess)`` of ``check_posterior_sampler_options.m`` and ``WR = 3 sqrt(diag(D))``.

    Octave inverts the (symmetric positive definite) Hessian through its Cholesky factor, so Dynare's ``invhess`` is
    exactly symmetric and ``eig`` uses LAPACK's symmetric solver ``dsyev`` on the upper triangle: eigenvalues in
    ascending order, eigenvectors with ``dsyev``'s signs. The same call is made here for a matrix that is symmetric up
    to rounding (pydyn's inverse is not exactly symmetric); checked against Dynare's ``V1`` (1e-13). A clearly
    non-symmetric matrix goes to the general solver, as in Octave."""
    V = np.asarray(V, float)
    if np.max(np.abs(V - V.T)) <= 1e-10 * np.max(np.abs(V)):
        d, V1 = linalg.eigh(V, driver="ev", lower=False)
    else:
        d, V1 = linalg.eig(V)
        if np.any(np.abs(np.imag(d)) > 0) or np.any(np.abs(np.imag(V1)) > 0):
            raise SamplerError("rotated slice: the proposal covariance has complex eigenvalues")
        d, V1 = np.real(d), np.real(V1)
    return V1, np.sqrt(d) * 3


_EXTRA_KW = {"method", "rng", "random", "sampler_options", "rotation", "optimizer", "verbose", "sampler_bounds",
             "kernel_bounds"}


def dispatch(est, kw) -> bool:
    """True if :meth:`Estimation.sample` must hand over to :func:`sample`: another sampler, keywords of this module,
    or ``posterior_sampler_options`` that the default random-walk sampler does not handle."""
    unknown = set(kw) - _EXTRA_KW
    if unknown:
        raise TypeError(f"sample() got unexpected keyword arguments {sorted(unknown)}")
    method = kw.get("method") or est.options["posterior_sampling_method"]
    if method != RWMH or any(kw.get(k) is not None for k in ("rng", "random", "sampler_options")):
        return True
    opts = sampler_options(RWMH, est.options["posterior_sampler_options"], est, warn=False)
    return opts["proposal_distribution"] != "rand_multivariate_normal" or bool(opts["use_mh_covariance_matrix"]) \
        or _has_key(est.options["posterior_sampler_options"], "scale_file")


def _pairs(spec):
    if not spec:
        return []
    if isinstance(spec, str):
        return read_key_value_string(spec)
    return list(spec.items()) if isinstance(spec, dict) else list(spec)


def _has_key(spec, key):
    return any(k == key for k, _ in _pairs(spec))


def sample(est, method=None, *, replic=None, nblocks=None, mode=None, cov=None, jscale=None, seed=0, x0=None, drop=None,
           init_scale_factor=None, jumping_covariance=None, load=None, save=None, rng=None, random=None,
           sampler_options=None, rotation=None, optimizer=None, verbose=False, innovations=None, uniforms=None,
           init_innovations=None, tune_jscale=None, sampler_bounds=None, kernel_bounds=None):
    """Run ``method`` (default: the ``posterior_sampling_method`` option) for an :class:`~pydyn.estimation.Estimation`.

    Arguments as :meth:`Estimation.sample <pydyn.estimation.Estimation.sample>`; in addition:

    ``sampler_options``
        overrides the ``posterior_sampler_options`` of the ``.mod`` file (string ``'key',value,...``, dict or pairs).
    ``rng``
        ``None`` (numpy generators from ``seed``) or ``'octave'`` (Dynare's random numbers under Octave: chain ``j``
        seeded with ``seed + 1 + ... + j``, the starting points of several chains drawn after seeding with ``seed``).
    ``random``
        a list of random sources (objects with ``rand``, ``randn``, ``randperm``, such as
        :class:`~pydyn.octave_rng.ArrayRandom`), one per chain and optionally one more for the starting points.
    ``rotation``
        ``(V1, WR)`` for the rotated slice sampler in place of the eigen decomposition of the proposal covariance.
    ``optimizer``
        for TaRB, a function ``(objective, x0, block, random, counter) -> x`` replacing the block mode search.
    ``sampler_bounds``, ``kernel_bounds``
        ``(lb, ub)`` replacing the bounds the proposals are checked against (default: ``est.lb``, ``est.ub``) and those
        of the posterior kernel (default: the prior support with ``prior_trunc``). With ``prior_trunc > 0`` Dynare's
        bounds come from the inverse CDFs of Octave's statistics package evaluated at ``1 - prior_trunc``, which differ
        from scipy's by up to about 1e-8 (relative); passing Dynare's bounds removes that source of difference when
        replaying Dynare's chains.

    Starting points (``posterior_sampler_initialization.m``): with one chain the mode (the prior mode for the slice
    sampler unless ``slice_initialize_with_mode``); with several, draws around the mode as for the random-walk sampler
    (``mh_init_scale_factor``), or for the slice sampler draws from the truncated priors. ``x0`` overrides them.
    """
    from . import posterior as _post
    o = est.options
    method = method or o["posterior_sampling_method"]
    if method in _NOT_PORTED:
        raise NotImplementedError(f"posterior_sampling_method={method!r}: {_NOT_PORTED[method]}")
    bad = [k for k, v in (("innovations", innovations), ("uniforms", uniforms), ("init_innovations", init_innovations))
           if v is not None]
    if bad:
        raise SamplerError(f"{bad} are arguments of the default random-walk sampler; here use random=[ArrayRandom(...), "
                           "...] (pydyn.octave_rng) to supply random numbers")
    target = o["mh_tune_jscale"] if tune_jscale is None else tune_jscale
    if target is not None and target is not False:
        if method != RWMH:
            warnings.warn("mh_tune_jscale is only available with Random Walk Metropolis Hastings!")
        else:
            raise NotImplementedError("mh_tune_jscale with a Student proposal or rng='octave' is not implemented")
    spec = o["posterior_sampler_options"] if sampler_options is None else sampler_options
    opts = globals()["sampler_options"](method, spec, est)
    if method in SMC:
        if load is not None or save is not None:
            raise SamplerError(f"{method}: load and save are not supported for SMC samplers")
        fn = hssmc if method == "hssmc" else dsmh
        return fn(est, opts=opts, seed=seed, rng=rng, random=random, verbose=verbose)
    if not est.prior.is_bayesian:
        raise SamplerError("the sampler needs priors")
    log = print if verbose else (lambda *_: None)
    replic = int(o["mh_replic"] if replic is None else replic)
    drop = o["mh_drop"] if drop is None else drop
    trunc = o["prior_trunc"]
    mlb, mub = est.prior.bounds(trunc) if kernel_bounds is None else (np.asarray(kernel_bounds[0], float),
                                                                       np.asarray(kernel_bounds[1], float))
    slb, sub = ((np.array(est.lb, float), np.array(est.ub, float)) if sampler_bounds is None else
                (np.asarray(sampler_bounds[0], float), np.asarray(sampler_bounds[1], float)))   # sampler_options.bounds
    if method == SLICE and (np.any(np.isinf(slb)) or np.any(np.isinf(sub))):
        raise SamplerError('The option "slice" is inconsistent with prior_trunc=0.')

    def logpost(x):
        return est.log_posterior(x, bounds=(mlb, mub))

    previous = None
    if load is not None:
        previous = load if hasattr(load, "draws") else _post.load_chains(load)
        if list(previous.names) != list(est.names):
            raise SamplerError("load: the stored chains are for other parameters")
        nblocks = previous.draws.shape[0]
        if replic == 0:
            return previous
    nblocks = int(o["mh_nblck"] if nblocks is None else nblocks)
    chain_rand, init_rand, gen = _sources(rng, random, seed, nblocks, 0 if previous is None else previous.draws.shape[1])
    # jscale (set_prior.m, dynare_estimation_init.m) and scale_file
    js, base = est.jscale(jscale)
    if method in (RWMH, TARB) and _has_key(spec, "scale_file"):
        sc = _scale_file(dict(_pairs(spec))["scale_file"], est)
        js, base = np.full(est.n, sc), sc
    # the proposal covariance (invhess after set_mcmc_jumping_covariance.m) and xparam1
    xmode = np.asarray(mode.x if hasattr(mode, "x") else (est.x0 if mode is None else mode), float)
    inv_h = getattr(mode, "inv_hessian", None)
    use_mh_cov = bool(opts.get("use_mh_covariance_matrix", 0))
    kind = o["mcmc_jumping_covariance"] if jumping_covariance is None else jumping_covariance

    def mode_covariance():
        return np.asarray(cov, float) if cov is not None else _post.jumping_covariance(est, kind, inv_h)
    V = None
    prior_start = False                           # slice: start from the prior mode / prior draws
    V1 = WR = None
    if method == SLICE:
        if opts["mode_files"] is not None:
            opts["mode"] = _load_modes(opts["mode_files"], est)
            opts["rotated"], opts["WR"] = 1, None
        multimodal = bool(opts.get("mode"))
        with_mode = bool(opts["slice_initialize_with_mode"]) and previous is None   # reset to 0 by load_mh_file
        if with_mode and mode is None and cov is None:
            raise SamplerError('check_posterior_sampler_options:: The option "slice_initialize_with_mode" is inconsistent '
                               "with mode_compute=0 or empty mode_file.")
        if not with_mode and previous is None:
            # dynare_estimation_1.m with mh_posterior_mode_estimation: start at the prior mode (the prior mean for
            # uniform priors and where the mode is outside the bounds); invhess is the diagonal of the prior variances
            prior_start = True
            xmode = prior_mode(est.prior)
            k = np.isnan(xmode)
            xmode[k] = est.prior.p1[k]
            out = (xmode < slb) | (xmode > sub)
            xmode[out] = est.prior.p1[out]
        if previous is not None and use_mh_cov:
            V = _mh_covariance(previous, drop)[1]
        elif with_mode:
            V = mode_covariance()
        elif multimodal:
            var = np.asarray(est.prior.p2, float) ** 2
            var[np.isinf(var)] = 1.0
            V = np.diag(var)
        if opts["rotated"] and not multimodal:
            if rotation is not None:
                V1 = np.asarray(rotation[0], float)
                WR = None if rotation[1] is None else np.asarray(rotation[1], float)
            elif V is None:
                raise SamplerError("check_posterior_sampler_options:: Rotated slice cannot start: there is no previous "
                                   "MCMC to load (with use_mh_covariance_matrix) or the Hessian at the mode is not "
                                   "computed (use slice_initialize_with_mode)")
            else:
                V1, WR = rotation_from_covariance(V)
        if not opts["rotated"] and not with_mode and previous is None:
            V = None                              # invhess = []: several chains start from prior draws
    elif previous is not None and use_mh_cov:
        V = _mh_covariance(previous, drop)[1]
    elif method in (RWMH, IMH, TARB) and previous is None:
        V = mode_covariance()                     # TaRB: only for the starting points of several chains
    d = None
    if previous is not None and method in (RWMH, IMH) and not use_mh_cov:
        if previous.proposal_chol is None:
            raise SamplerError("load: the stored chains carry no proposal density")
        d, js, base = previous.proposal_chol, np.asarray(previous.jscale_vector, float), previous.jscale
    elif V is not None:
        d = np.linalg.cholesky(V).T
    if method == IMH and previous is not None and mode is None:
        if getattr(previous, "proposal_center", None) is None:
            raise SamplerError("load: the independent sampler needs the mode (proposal centre): pass mode=...")
        xmode = np.asarray(previous.proposal_center, float)
    W1 = opts.get("initial_step_size", 0.8) * (sub - slb) if method == SLICE else None
    # starting points
    starts, lp0 = np.empty((nblocks, est.n)), np.empty(nblocks)
    if previous is not None:
        starts, lp0 = previous.draws[:, -1, :].copy(), previous.log_posterior[:, -1].copy()
    elif x0 is not None:
        x0 = np.asarray(x0, float).reshape(nblocks, est.n)
        for j in range(nblocks):
            starts[j], lp0[j] = x0[j], logpost(x0[j])
            if not np.isfinite(lp0[j]):
                raise SamplerError(f"chain {j}: the log posterior is not finite at the starting point")
    elif nblocks == 1:
        if not (np.all(xmode >= mlb) and np.all(xmode <= mub)):
            raise SamplerError("Initialization failed: the " + ("prior" if prior_start else "posterior")
                               + " mode lies outside the prior bounds")
        starts[0], lp0[0] = xmode, logpost(xmode)
    else:
        factor = o["mh_init_scale_factor"] if init_scale_factor is None else init_scale_factor
        fixed = o["mh_init_scale"] if init_scale_factor is None else None
        for j in range(nblocks):
            validate, init_iter, trial = False, 0, 1
            while not validate and trial <= 10:
                if d is None:
                    cand = prior_draw(est.prior, mlb, mub, init_rand, gen)
                else:
                    scale = fixed if fixed is not None else factor * base
                    cand = rand_multivariate_normal(init_rand, xmode, d * scale)
                if np.all(cand >= mlb) and np.all(cand <= mub):
                    lp = logpost(cand)
                    if np.isfinite(lp):
                        starts[j], lp0[j], validate = cand, lp, True
                init_iter += 1
                if not validate and init_iter > 100:
                    if fixed is not None:
                        fixed *= 0.9
                    else:
                        factor *= 0.9
                    trial += 1
            if not validate:
                raise SamplerError(f"I'm unable to find a starting value for block {j + 1}.")
    # TaRB optimiser
    if method == TARB:
        optimizer = optimizer or _tarb_optimizer(est, opts)
    chol_prop = None if d is None else d * js[None, :]
    student = opts.get("proposal_distribution") == "rand_multivariate_student"
    if method == IMH and student and "student_degrees_of_freedom" not in opts:
        raise SamplerError("independent_metropolis_hastings with rand_multivariate_student: set "
                           "student_degrees_of_freedom (Dynare 7.2 has no default for this sampler and stops)")
    dof = opts.get("student_degrees_of_freedom")
    draws = np.empty((nblocks, replic, est.n))
    logpo = np.empty((nblocks, replic))
    acc = np.empty((nblocks, replic), float if method == TARB else bool)
    neval = np.empty((nblocks, replic))
    prior_flags = [bool(opts.get("use_prior_draws", False))] * nblocks
    for j in range(nblocks):
        r = chain_rand[j]
        x, lp = starts[j].copy(), float(lp0[j])

        def pdraw(r=r):
            return prior_draw(est.prior, mlb, mub, r, gen)
        for t in range(replic):
            if method == SLICE:
                if opts["rotated"]:
                    if opts.get("mode"):
                        Vm = multimodal_directions(x, opts["mode"], r)
                        x, lp, nev = rotated_slice_iteration(logpost, x, slb, sub, Vm, None, r)
                        x, lp, nev2, prior_flags[j] = slice_iteration(logpost, x, slb, sub, W1, r, pdraw,
                                                                      prior_flags[j], lp, log)
                        nev = np.sum(nev) + np.sum(nev2)
                    else:
                        x, lp, nev = rotated_slice_iteration(logpost, x, slb, sub, V1, WR, r)
                else:
                    x, lp, nev, prior_flags[j] = slice_iteration(logpost, x, slb, sub, W1, r, pdraw, prior_flags[j], lp,
                                                                 log)
                a, ne = True, float(np.sum(nev))
            elif method == TARB:
                x, lp, a, ne = tarb_iteration(est, x, lp, slb, sub, mlb, mub, js, opts, r, optimizer)
            else:
                center = x if method == RWMH else xmode
                if student:
                    par = rand_multivariate_student(r, center, chol_prop, dof)
                else:
                    par = rand_multivariate_normal(r, center, chol_prop)
                if np.all(par > slb) and np.all(par < sub):
                    lpp = _safe(logpost, par)
                else:
                    lpp = -np.inf
                if method == RWMH:
                    ratio = lpp - lp
                else:
                    if student:
                        qo, qn = (multivariate_student_pdf(x, xmode, chol_prop, dof),
                                  multivariate_student_pdf(par, xmode, chol_prop, dof))
                    else:
                        qo, qn = (multivariate_normal_pdf(x, xmode, chol_prop, est.n),
                                  multivariate_normal_pdf(par, xmode, chol_prop, est.n))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = lpp - lp + np.log(qo) - np.log(qn)
                a = bool(lpp > -np.inf and math.log(r.rand()) < ratio)
                if a:
                    x, lp = par, lpp
                ne = 1.0
            draws[j, t], logpo[j, t], acc[j, t], neval[j, t] = x, lp, a, ne
    cls = _result_cls()
    x_init = starts
    if previous is not None:
        draws = np.concatenate([previous.draws, draws], axis=1)
        logpo = np.concatenate([previous.log_posterior, logpo], axis=1)
        if previous.accepted is not None:
            acc = np.concatenate([np.asarray(previous.accepted, acc.dtype), acc], axis=1)
        pn = getattr(previous, "neval", None)
        neval = np.concatenate([pn if pn is not None else np.ones(previous.log_posterior.shape), neval], axis=1)
        x_init = previous.x0
    out = cls(draws=draws, log_posterior=logpo, acceptance_rate=np.asarray(acc, float).mean(axis=1), names=list(est.names),
              x0=x_init, drop=drop, accepted=acc, jscale=float(base), jscale_vector=np.array(js, float),
              proposal_chol=None if d is None else np.array(d), method=method, neval=neval, sampler_options=opts,
              proposal_center=xmode.copy() if method == IMH else None, rotation=V1, rotation_widths=WR)
    if save is not None:
        _post.save_chains(save, out)
    return out


def _tarb_optimizer(est, opts):
    """The block mode search of TaRB: ``mode_compute`` of the sampler options (default 4, csminwel with Dynare's
    defaults and ``optim`` overrides, silent). The random numbers csminwel uses at a "cliff" come from the chain's
    uniform stream, as in Dynare."""
    from .estimation import _parse_optim, csminwel
    mc = opts["mode_compute"]
    if mc != 4:
        raise NotImplementedError(f"TaRB with mode_compute={mc}: only 4 (csminwel, Dynare's default for TaRB) is "
                                  "available for the block mode search")
    opt = _parse_optim(opts.get("optim_opt") or est.options["optim"])
    H0 = opt.get("InitialInverseHessian")
    crit = float(opt.get("TolFun", 1e-7))
    nit = int(opt.get("MaxIter", 1000))
    method = int(opt.get("NumgradAlgorithm", est.options["gradient_method"]))
    epsilon = float(opt.get("NumgradEpsilon", est.options["gradient_epsilon"]))

    def run(objective, x0, idx, r, counter):
        def counted(v):
            counter[0] += 1
            return objective(v)
        h0 = None
        if isinstance(H0, str):
            mt = re.fullmatch(r"\s*([-+0-9.eE]+)\s*\*\s*eye\(\s*(\d+)\s*\)\s*", H0)
            h0 = float(mt.group(1)) * np.eye(len(x0)) if mt else None
        res = csminwel(counted, x0, H0=h0, crit=crit, nit=nit, method=method, epsilon=epsilon,
                       rand=lambda n: np.atleast_1d(r.rand(n)))
        if res["retcode"] == -1:
            raise SamplerError("TaRB: csminwel: bad initial parameter")
        return res["x"], dict(fopt=res["f"], iterations=res["iterations"], fcount=res["fcount"], retcode=res["retcode"])
    return run


# ====================================================================== sequential Monte Carlo
def _exp(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.exp(x))


def kitagawa(weights, noise, m=None):
    """Systematic resampling (``kitagawa.m``): 0-based indices of ``m`` (default: as many as weights) particles."""
    n = len(weights) if m is None else int(m)
    cw = np.cumsum(weights)
    ww = (np.arange(n) + noise) * (1.0 / n)
    out = np.empty(n, int)
    j = 0
    for i in range(n):
        while ww[i] > cw[j]:
            j += 1
        out[i] = j
    return out


def _smc_setup(est, rng, random, seed, n, name):
    trunc = est.options["prior_trunc"]
    lb, ub = est.prior.bounds(trunc)

    def tempered(x, lam):
        lpk = _safe(lambda y: est.log_posterior(y, bounds=(lb, ub)), x)
        lprior = est.prior.logpdf(x)
        ll = lpk - lprior
        return lam * ll + lprior, ll
    if random is not None:
        r = random
        gen = np.random.default_rng(seed)
    elif rng == "octave":
        r, gen = OctaveRandom(int(seed) + n), np.random.default_rng(seed)
    else:
        gen = np.random.default_rng(seed)
        r = NumpyRandom(np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]))
    return lb, ub, tempered, r, gen


def _smc_init(est, n, lb, ub, tempered, gen, seed, rng, initial):
    """``smc_samplers_initialization.m``: particle j drawn from the truncated priors (generator seeded with
    ``seed + j`` under rng='octave'), redrawn until the likelihood is finite. ``initial``: given particles (n, npar)."""
    npar = est.n
    P = np.empty((n, npar))
    tlp, ll = np.empty(n), np.empty(n)
    for j in range(n):
        if initial is not None:
            P[j] = initial[j]
            tlp[j], ll[j] = tempered(P[j], 0.0)
            if not np.isfinite(ll[j]):
                raise SamplerError(f"initial particle {j}: the likelihood is not finite")
            continue
        rj = OctaveRandom(int(seed) + j + 1) if rng == "octave" else NumpyRandom(gen)
        while True:
            cand = prior_draw(est.prior, lb, ub, rj, gen)
            if np.all(cand >= lb) and np.all(cand <= ub):
                P[j] = cand
                tlp[j], ll[j] = tempered(cand, 0.0)
                if np.isfinite(ll[j]):
                    break
    return P, tlp, ll


def hssmc(est, opts=None, seed=0, rng=None, random=None, initial=None, verbose=False):
    """Herbst and Schorfheide's sequential Monte Carlo sampler (``hssmc.m``): ``particles`` draws from the priors
    tempered towards the posterior in ``steps`` stages (``phi_i = ((i-1)/(steps-1))^lambda``), resampled (systematic)
    when the effective sample size falls below half the particles, mutated by one random-walk Metropolis step with the
    weighted particle covariance scaled adaptively towards the acceptance ``target``.

    ``initial`` (particles x parameters) replaces the prior draws of the first stage. With ``rng='octave'`` the random
    numbers after the initialisation are Dynare's (Octave seeded with ``seed + particles``)."""
    opts = opts or sampler_options("hssmc", est.options["posterior_sampler_options"], est)
    log = print if verbose else (lambda *_: None)
    N, steps = int(opts["particles"]), int(opts["steps"])
    lb, ub, tempered, r, gen = _smc_setup(est, rng, random, seed, N, "hssmc")
    phi = (np.arange(steps) / (steps - 1)) ** opts["lambda"]
    P, tlp, ll = _smc_init(est, N, lb, ub, tempered, gen, seed, rng, initial)
    mdd = 0.0
    scl, ess, acpt_hist = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    weights = np.ones(N) / N
    scale, acpt = float(opts["scale"]), float(opts["acpt"])
    target = float(opts["target"])
    stages = [dict(particles=P.copy(), log_posterior=tlp.copy(), log_likelihood=ll.copy())]
    resampled = False
    for i in range(1, steps):
        weights = weights * np.exp((phi[i] - phi[i - 1]) * ll)
        sw = float(np.sum(weights))
        weights = weights / sw
        mdd += math.log(sw)
        ess[i] = 1.0 / float(np.sum(weights ** 2))
        if 2 * ess[i] < N:
            resampled = True
            k = kitagawa(weights, r.rand())
            P, ll, tlp = P[k], ll[k], tlp[k]
            weights = np.ones(N) / N
        scale = scale * (0.95 + 0.1 / (1 + math.exp(-16 * (acpt - target))))
        scl[i] = scale
        mu = P.T @ weights
        z = P.T - mu[:, None]
        Rm = z @ (z.T * weights[:, None])
        Rm = 0.5 * (Rm + Rm.T)
        RR = np.linalg.cholesky(Rm)
        tlp = tlp + (phi[i] - phi[i - 1]) * ll
        accepted = np.zeros(N, bool)
        for j in range(N):
            cand = P[j] + scale * (RR @ np.atleast_1d(r.randn(est.n)))
            if np.all(cand >= lb) and np.all(cand <= ub):
                tl, lk = tempered(cand, phi[i])
                if np.isfinite(lk):
                    with np.errstate(over="ignore", invalid="ignore"):
                        ok = r.rand() < np.exp(tl - tlp[j])
                    if ok:
                        accepted[j] = True
                        P[j], ll[j], tlp[j] = cand, lk, tl
        acpt = float(np.sum(accepted)) / N
        acpt_hist[i] = acpt
        log(f"{i + 1:3d}  {phi[i]:5.4f}  {ess[i]:9.5E}  {acpt:5.4f}  {scale:4.3f}  {'yes' if resampled else 'no'}")
        if i == steps - 1 and not resampled:
            k = kitagawa(weights, r.rand())
            P = P[k]
        stage = dict(particles=P.copy(), log_posterior=tlp.copy(), log_likelihood=ll.copy())
        if not resampled:
            stage["weights"] = weights.copy()
        stages.append(stage)
        resampled = False
    return SMCResult(method="hssmc", particles=P, log_posterior=tlp, log_likelihood=ll, log_marginal_density=mdd,
                     names=list(est.names), tempering=phi, ess=ess, acceptance=acpt_hist, scale=scl, weights=weights,
                     stages=stages)


def dsmh(est, opts=None, seed=0, rng=None, random=None, initial=None, verbose=False):
    """Dynamic striated Metropolis-Hastings (``dsmh.m``): ``N*G`` particles from the priors, ``H`` tempering stages
    (``lambda_i`` geometric from ``1/(10 npar T)`` to 1), each with importance weights, a tuning of the random-walk scale
    ``c`` on ``G`` resampled particles (``K`` steps each) until the acceptance is in [alpha0, alpha1], and a striated
    mutation (``N*tau`` sweeps over ``G`` resampled particles mixing random-walk moves and jumps between ``M`` strata of
    the previous kernel). Returns an :class:`SMCResult` whose ``log_marginal_density`` is Dynare's ``zhat``."""
    opts = opts or sampler_options("dsmh", est.options["posterior_sampler_options"], est)
    log = print if verbose else (lambda *_: None)
    H, M, N, G, K, tau = (int(opts[k]) for k in ("H", "M", "N", "G", "K", "tau"))
    a0, a1 = float(opts["alpha0"]), float(opts["alpha1"])
    npar = est.n
    lb, ub, tempered, r, gen = _smc_setup(est, rng, random, seed, N * G, "dsmh")
    T = est.nobs
    lambda1 = 1 / (10 * npar * T)
    lam = np.exp((H - np.arange(1, H + 1)) / (H - 1) * math.log(lambda1))
    c = 1.0
    param, tlp_prev, ll = _smc_init(est, N * G, lb, ub, tempered, gen, seed, rng, initial)
    ess, acc_hist, c_hist = np.zeros(H), np.zeros(H), np.zeros(H)
    zhat = 0.0
    stages = []
    for i in range(H):
        order = np.argsort(tlp_prev, kind="stable")          # sortrows
        tlp_prev, ll, param = tlp_prev[order], ll[order], param[order]
        dl = lam[i] if i == 0 else lam[i] - lam[i - 1]
        tlp_i = tlp_prev + ll * dl
        weights = np.exp(tlp_i - tlp_prev)
        zhat += math.log(float(np.mean(weights)))
        weights = weights / np.sum(weights)
        ess[i] = 1 / float(np.sum(weights ** 2))
        mu = param.T @ weights
        z = param.T - mu[:, None]
        Om = (z * weights[None, :]) @ z.T
        Oc = np.linalg.cholesky((Om + Om.T) / 2)
        # tune_c
        lower, upper = (0.5 * (a0 + a1)) ** 5, (0.5 * (a0 + a1)) ** (1 / 5)
        while True:
            acpt = 0.0
            k = kitagawa(weights, r.rand(), G)
            p0, t0 = param[k].copy(), tlp_i[k].copy()
            for j in range(G):
                for _ in range(K):
                    cand = p0[j] + math.sqrt(c) * (Oc @ np.atleast_1d(r.randn(npar)))
                    if np.all(cand >= lb) and np.all(cand <= ub):
                        tx, lx = tempered(cand, lam[i])
                        if np.isfinite(lx) and r.rand() < _exp(tx - t0[j]):
                            acpt += 1 / (G * K)
                            p0[j], t0[j] = cand, tx
            if a0 <= acpt <= a1:
                break
            if acpt < lower:
                c = c / 5
            elif lower <= acpt <= upper:
                c = c * math.log(0.5 * (a0 + a1)) / math.log(acpt)
            else:
                c = 5 * c
        # mutation_DSMH
        idx_levels = (np.arange(1, M) * N * G // M) - 1
        p = 0.1 / tau
        levels = np.concatenate([[-np.inf], tlp_prev[idx_levels]])
        out_param, out_tlp, out_ll = param.copy(), tlp_i.copy(), ll.copy()
        k = kitagawa(weights, r.rand(), G)
        p0, tm0, ti0, l0 = param[k].copy(), tlp_prev[k].copy(), tlp_i[k].copy(), ll[k].copy()
        for l_ in range(1, N * tau + 1):
            for j in range(G):
                u1, u2 = r.rand(), r.rand()
                if u1 < p:
                    kk = M
                    for m_ in range(1, M):
                        if tm0[j] < levels[m_]:
                            kk = m_
                            break
                    indx = int(math.floor((kk - 1) * N * G / M + 1 + u2 * (N * G / M - 1))) - 1
                    alp = (ll[indx] - l0[j]) * dl
                    if u2 < _exp(alp):
                        p0[j], ti0[j], l0[j], tm0[j] = param[indx], tlp_i[indx], ll[indx], tlp_prev[indx]
                else:
                    cand = p0[j] + math.sqrt(c) * (Oc @ np.atleast_1d(r.randn(npar)))
                    if np.all(cand >= lb) and np.all(cand <= ub):
                        tx, lx = tempered(cand, lam[i])
                        if np.isfinite(lx) and u2 < _exp(tx - ti0[j]):
                            p0[j], ti0[j], l0[j] = cand, tx, lx
                            tm0[j] = tx - lx * dl
            if l_ % tau == 0:
                rng_ = slice((l_ // tau - 1) * G, l_ * G // tau)
                out_param[rng_], out_tlp[rng_], out_ll[rng_] = p0, ti0, l0
        param, tlp_prev, ll = out_param, out_tlp, out_ll
        acc_hist[i], c_hist[i] = acpt, c
        stages.append(dict(particles=param.copy(), log_posterior=tlp_prev.copy(), log_likelihood=ll.copy()))
        log(f"{i + 1:4d}  {lam[i]:5.4f}  {ess[i]:9.5E}  {c:5.4f}  {acpt:5.4f}  {zhat:+5.4f}")
    return SMCResult(method="dsmh", particles=param, log_posterior=tlp_prev, log_likelihood=ll, log_marginal_density=zhat,
                     names=list(est.names), tempering=lam, ess=ess, acceptance=acc_hist, scale=c_hist, stages=stages)

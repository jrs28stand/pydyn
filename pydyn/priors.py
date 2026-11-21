# This file is part of pydyn.
#
# It follows, and in places is a Python port of, matlab/estimation/{set_prior, priordens}.m and matlab/distributions/*.m from Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Prior distributions with Dynare's parameterisations.

Shape codes are those of the preprocessor / ``bayestopt_.pshape``:
1 beta, 2 gamma, 3 normal, 4 inverse gamma type 1, 5 uniform, 6 inverse gamma type 2, 8 Weibull
(0 means "no prior": maximum likelihood).

The hyper-parameter mapping follows ``set_prior.m`` and the ``*_specification.m`` helpers of
Dynare 7.2; the log density follows ``priordens.m`` and the ``lpdf*.m`` files.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.special import betaln, gammaln

SHAPE_NAMES = {0: "none", 1: "beta", 2: "gamma", 3: "normal", 4: "inv_gamma1", 5: "uniform",
               6: "inv_gamma2", 8: "weibull"}
SHAPE_CODES = {v: k for k, v in SHAPE_NAMES.items()}
SHAPE_CODES.update(inv_gamma=4, invgamma=4, invgamma1=4, invgamma2=6, beta_pdf=1, gamma_pdf=2,
                   normal_pdf=3, inv_gamma_pdf=4, inv_gamma1_pdf=4, uniform_pdf=5, inv_gamma2_pdf=6,
                   weibull_pdf=8)


class PriorError(ValueError):
    pass


# ------------------------------------------------------------------ (mean, variance) -> hyper-parameters
def beta_specification(mu, sigma2, lb=0.0, ub=1.0):
    """Shape parameters (a, b) of a beta distribution on [lb, ub]."""
    if mu < lb or mu > ub:
        raise PriorError(f"beta prior: mean {mu} outside [{lb}, {ub}]")
    length = ub - lb
    mu = (mu - lb) / length
    sigma2 = sigma2 / (length * length)
    if sigma2 > (1 - mu) * mu:
        raise PriorError(f"beta prior: the std has to be smaller than {np.sqrt((1 - mu) * mu) * length}")
    a = (1 - mu) * mu * mu / sigma2 - mu
    b = a * (1 / mu - 1)
    return a, b


def gamma_specification(mu, sigma2, lb=0.0):
    """Shape a and scale b of a gamma distribution shifted by lb."""
    if mu < lb:
        raise PriorError(f"gamma prior: mean {mu} below the lower bound {lb}")
    if np.isinf(sigma2):
        raise PriorError("gamma prior: the variance has to be finite")
    mu = mu - lb
    b = sigma2 / mu
    return mu / b, b


def _ig1fun(nu, mu2, sigma2):
    return np.log(2 * mu2) - np.log((sigma2 + mu2) * (nu - 2)) + 2 * (gammaln(nu / 2) - gammaln((nu - 1) / 2))


def inverse_gamma_specification(mu, sigma2, lb=0.0, type=1):
    """(s, nu) of an inverse gamma distribution of type 1 or 2, shifted by lb."""
    if sigma2 <= 0:
        raise PriorError("inverse gamma prior: the variance must be positive")
    if mu <= lb:
        raise PriorError(f"inverse gamma prior: mean {mu} must be above the lower bound {lb}")
    sigma = np.sqrt(sigma2)
    mu2 = (mu - lb) * (mu - lb)
    if type == 2:
        nu = 2 * (2 + mu2 / sigma2)
        s = 2 * mu * (1 + mu2 / sigma2)          # sic: Dynare uses mu here, not mu-lb
        return s, nu
    if type != 1:
        raise PriorError("inverse gamma prior: type must be 1 or 2")
    if not np.isfinite(sigma2):
        return 2 * mu2 / np.pi, 2.0
    # Dynare's bracketing bisection
    nu = np.sqrt(2 * (2 + mu2 / sigma2))
    nu2, nu1 = 2 * nu, 2.0
    err, err2 = _ig1fun(nu, mu2, sigma2), _ig1fun(nu2, mu2, sigma2)
    if err2 > 0:
        while nu2 < 1e12:
            nu1, nu2 = nu2, nu2 * 2
            err2 = _ig1fun(nu2, mu2, sigma2)
            if err2 < 0:
                break
        if err2 > 0:
            raise PriorError("inverse gamma prior: no interval with a sign change; the prior variance "
                             "is probably too small compared with the prior mean")
    while abs(nu2 / nu1 - 1) > 1e-14:
        if err > 0:
            nu1 = nu
            if nu < nu2:
                nu = nu2
            else:
                nu = 2 * nu
                nu2 = nu
        else:
            nu2 = nu
        nu = (nu1 + nu2) / 2
        err = _ig1fun(nu, mu2, sigma2)
    s = (sigma2 + mu2) * (nu - 2)
    if abs(np.log(mu - lb) - np.log(np.sqrt(s / 2)) - gammaln((nu - 1) / 2) + gammaln(nu / 2)) > 1e-7 \
            or abs(sigma - np.sqrt(s / (nu - 2) - (mu - lb) * (mu - lb))) > 1e-7:
        raise PriorError("inverse gamma prior: failed in solving for the hyper-parameters")
    return s, nu


def weibull_specification(mu, sigma2, lb=0.0):
    """(shape, scale) of a Weibull distribution shifted by lb."""
    if mu < lb:
        raise PriorError(f"Weibull prior: mean {mu} below the lower bound {lb}")
    if np.isinf(sigma2):
        raise PriorError("Weibull prior: the variance has to be finite")
    mu = mu - lb
    target = np.log1p(sigma2 / (mu * mu))

    def eqn(k):
        return gammaln(1 + 2 / k) - 2 * gammaln(1 + 1 / k) - target

    lo, hi = 1e-2, 1e2
    while eqn(hi) > 0 and hi < 1e8:
        hi *= 10
    shape = optimize.brentq(eqn, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    scale = mu / np.exp(gammaln(1 + 1 / shape))
    return shape, scale


def uniform_specification(m, s, p3, p4):
    """Return (mean, std, lower, upper)."""
    if not (np.isnan(p3) or np.isnan(p4)):
        if not (np.isnan(m) and np.isnan(s)):
            raise PriorError("uniform prior: with lower and upper bounds given, leave mean and std empty")
        return (p3 + p4) / 2, (p4 - p3) / np.sqrt(12), p3, p4
    if not (np.isfinite(m) and np.isfinite(s)):
        raise PriorError("uniform prior: mean or standard deviation is not finite")
    return m, s, m - s * np.sqrt(3), m + s * np.sqrt(3)


# ------------------------------------------------------------------ log densities (lpdf*.m)
def lpdf_beta(x, a, b, aa=0.0, bb=1.0):
    if not (x - aa > 0 and x - bb < 0):
        return -np.inf
    return -betaln(a, b) + (a - 1) * np.log(x - aa) + (b - 1) * np.log(bb - x) - (a + b - 1) * np.log(bb - aa)


def lpdf_gamma(x, a, b):
    if not x > 0:
        return -np.inf
    return -gammaln(a) - a * np.log(b) + (a - 1) * np.log(x) - x / b


def lpdf_normal(x, a, b):
    return -np.log(b) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - a) / b) * ((x - a) / b)


def lpdf_ig1(x, s, nu):
    if not x > 0:
        return -np.inf
    return np.log(2) - gammaln(0.5 * nu) - 0.5 * nu * (np.log(2) - np.log(s)) - (nu + 1) * np.log(x) - 0.5 * s / (x * x)


def lpdf_ig2(x, s, nu):
    if not x > 0:
        return -np.inf
    return -gammaln(0.5 * nu) - (0.5 * nu) * (np.log(2) - np.log(s)) - 0.5 * (nu + 2) * np.log(x) - 0.5 * s / x


def lpdf_weibull(x, a, b, c=0.0):
    """Shape a, scale b, location c. Reproduces ``lpdfgweibull.m``, including its value of 1.0
    (instead of ``-log(b)``) at ``x == c`` when the shape is one."""
    if not x - c >= 0:
        return -np.inf
    if abs(a - 1) < 1e-12 and x - c < 1e-12:
        return 1.0
    x0 = x - c
    with np.errstate(divide="ignore"):
        return np.log(a) - a * np.log(b) + (a - 1) * np.log(x0) - (x0 / b) ** a


# ------------------------------------------------------------------ a set of independent priors
class Prior:
    """Independent priors for a parameter vector, as in Dynare's ``bayestopt_``.

    ``shape`` holds the integer codes, ``p1``/``p2`` prior mean and standard deviation, ``p3``/``p4``
    the bounds of the generalised distributions (NaN for the defaults). After construction
    ``p6``/``p7`` hold the hyper-parameters exactly as ``set_prior.m`` computes them:

    ======== ===================== =====================
    shape    p6                    p7
    ======== ===================== =====================
    beta     a                     b
    gamma    shape                 scale
    normal   mean                  standard deviation
    inv_g 1  s                     nu
    uniform  lower bound           upper bound
    inv_g 2  s                     nu
    weibull  shape                 scale
    ======== ===================== =====================
    """

    def __init__(self, shape, p1, p2, p3=None, p4=None, names=None):
        n = len(shape)
        self.shape = np.array([SHAPE_CODES[s] if isinstance(s, str) else int(s) for s in shape], dtype=int)
        self.p1 = np.array(p1, float)
        self.p2 = np.array(p2, float)
        self.p3 = np.full(n, np.nan) if p3 is None else np.array(p3, float)
        self.p4 = np.full(n, np.nan) if p4 is None else np.array(p4, float)
        self.p6 = np.full(n, np.nan)
        self.p7 = np.full(n, np.nan)
        self.names = list(names) if names is not None else [f"p{i}" for i in range(n)]
        if np.any((self.shape != 0) & (self.p2 == 0)):
            raise PriorError("a prior standard deviation of zero is not allowed")
        for i, sh in enumerate(self.shape):
            try:
                self._specify(i, sh)
            except PriorError as exc:
                raise PriorError(f"{self.names[i]}: {exc}") from None

    def _specify(self, i, sh):
        p1, p2, p3, p4 = self.p1[i], self.p2[i], self.p3[i], self.p4[i]
        if sh == 0:
            return
        if sh == 1:
            p3 = 0.0 if np.isnan(p3) else p3
            p4 = 1.0 if np.isnan(p4) else p4
            self.p6[i], self.p7[i] = beta_specification(p1, p2 ** 2, p3, p4)
        elif sh == 2:
            p3 = 0.0 if np.isnan(p3) else p3
            p4 = np.inf if np.isnan(p4) else p4
            self.p6[i], self.p7[i] = gamma_specification(p1, p2 ** 2, p3)
        elif sh == 3:
            p3 = -np.inf if np.isnan(p3) else p3
            p4 = np.inf if np.isnan(p4) else p4
            self.p6[i], self.p7[i] = p1, p2
        elif sh in (4, 6):
            p3 = 0.0 if np.isnan(p3) else p3
            p4 = np.inf if np.isnan(p4) else p4
            self.p6[i], self.p7[i] = inverse_gamma_specification(p1, p2 ** 2, p3, 1 if sh == 4 else 2)
        elif sh == 5:
            self.p1[i], self.p2[i], self.p6[i], self.p7[i] = uniform_specification(p1, p2, p3, p4)
            p3, p4 = self.p6[i], self.p7[i]
        elif sh == 8:
            p3 = 0.0 if np.isnan(p3) else p3
            p4 = np.inf if np.isnan(p4) else p4
            self.p6[i], self.p7[i] = weibull_specification(p1, p2 ** 2, p3)
        else:
            raise PriorError(f"unknown prior shape {sh}")
        self.p3[i], self.p4[i] = p3, p4

    def __len__(self):
        return len(self.shape)

    @property
    def is_bayesian(self):
        return bool(np.any(self.shape > 0))

    def logpdf_each(self, x):
        """Log density of every element (``-inf`` outside the support)."""
        x = np.asarray(x, float)
        out = np.zeros(len(self))
        for i, sh in enumerate(self.shape):
            xi, p3, p4, p6, p7 = x[i], self.p3[i], self.p4[i], self.p6[i], self.p7[i]
            if sh == 1:
                out[i] = lpdf_beta(xi, p6, p7, p3, p4)
            elif sh == 2:
                out[i] = lpdf_gamma(xi - p3, p6, p7)
            elif sh == 3:
                out[i] = lpdf_normal(xi, p6, p7)
            elif sh == 4:
                out[i] = lpdf_ig1(xi - p3, p6, p7)
            elif sh == 5:
                out[i] = -np.inf if (xi - p3 < 0 or xi - p4 > 0) else np.log(1.0 / (p4 - p3))
            elif sh == 6:
                out[i] = lpdf_ig2(xi - p3, p6, p7)
            elif sh == 8:
                out[i] = lpdf_weibull(xi, p6, p7, p3)
        return out

    def logpdf(self, x):
        """Joint log prior density, as ``priordens.m`` (0 for a model without priors)."""
        return float(np.sum(self.logpdf_each(x)))

    def bounds(self, prior_trunc=0.0):
        """Support of each prior (``prior_bounds.m``); with ``prior_trunc`` > 0 the quantiles
        ``prior_trunc`` and ``1 - prior_trunc``."""
        n = len(self)
        lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
        q = prior_trunc
        for i, sh in enumerate(self.shape):
            p3, p4, p6, p7 = self.p3[i], self.p4[i], self.p6[i], self.p7[i]
            if sh == 1:
                lb[i], ub[i] = (p3, p4) if q == 0 else (stats.beta.ppf(q, p6, p7) * (p4 - p3) + p3,
                                                        stats.beta.ppf(1 - q, p6, p7) * (p4 - p3) + p3)
            elif sh == 2:
                lb[i], ub[i] = (p3, np.inf) if q == 0 else (stats.gamma.ppf(q, p6, scale=p7) + p3,
                                                            stats.gamma.ppf(1 - q, p6, scale=p7) + p3)
            elif sh == 3:
                lb[i], ub[i] = (p3, p4) if q == 0 else (max(stats.norm.ppf(q, p6, p7), p3),
                                                        min(stats.norm.ppf(1 - q, p6, p7), p4))
            elif sh == 4:
                lb[i], ub[i] = (p3, np.inf) if q == 0 else (
                    1 / np.sqrt(stats.gamma.ppf(1 - q, p7 / 2, scale=2 / p6)) + p3,
                    1 / np.sqrt(stats.gamma.ppf(q, p7 / 2, scale=2 / p6)) + p3)
            elif sh == 5:
                lb[i], ub[i] = (p6, p7) if q == 0 else (p6 + (p7 - p6) * q, p7 - (p7 - p6) * q)
            elif sh == 6:
                lb[i], ub[i] = (p3, np.inf) if q == 0 else (
                    1 / stats.gamma.ppf(1 - q, p7 / 2, scale=2 / p6) + p3,
                    1 / stats.gamma.ppf(q, p7 / 2, scale=2 / p6) + p3)
            elif sh == 8:
                lb[i], ub[i] = (p3, np.inf) if q == 0 else (
                    p3 + stats.weibull_min.ppf(q, p6, scale=p7),
                    p3 + stats.weibull_min.ppf(1 - q, p6, scale=p7))
        return lb, ub

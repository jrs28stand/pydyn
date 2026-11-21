# This file is part of pydyn.
#
# It follows, and in places is a Python port of, these files of Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org; GPL-3.0-or-later), modified in 2026:
#   matlab/estimation/{posterior_moments, GetPosteriorParametersStatistics, GetAllPosteriorDraws, marginal_density,
#   compute_mh_covariance_matrix, recursive_moments, model_comparison, calibrate_mh_scale_parameter, CutSample,
#   get_posterior_subsample, set_number_of_subdraws, PosteriorIRF, PosteriorIRF_core1, mode_check,
#   draw_prior_density, plot_priors}.m,
#   matlab/convergence_diagnostics/{mcmc_diagnostics, mcmc_diagnostics_core, geweke_moments, geweke_chi2_test,
#   mcmc_ifac, raftery_lewis}.m, matlab/kernel_density_estimate.m, matlab/mh_optimal_bandwidth.m,
#   matlab/utilities/estimation/set_mcmc_jumping_covariance.m, matlab/stochastic_solver/{irf, getIrfShocksIndx}.m
#   and matlab/get_lower_cholesky_covariance.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Analysis of Metropolis-Hastings output, following Dynare 7.2.

Everything here works on the draws returned by :meth:`pydyn.estimation.Estimation.sample` (an
``MCMCResult``) or on plain arrays shaped (chains, draws, parameters) with log posteriors shaped
(chains, draws)::

    mc = est.sample(mode=mode)
    st = posterior.posterior_statistics(mc, est)          # mean, median, std, HPD interval, deciles, densities
    print(st.table())
    mhm = posterior.modified_harmonic_mean(mc)            # log marginal data density (Geweke 1999)
    posterior.model_comparison([mhm.value, other])        # posterior odds
    cd = posterior.convergence_diagnostics(mc)            # inefficiency factors, Geweke, Brooks-Gelman
    irfs = posterior.posterior_irf(est, mc, periods=20)   # mean, median and HPD bands of first-order IRFs
    posterior.save_chains("run.npz", mc); mc = posterior.load_chains("run.npz")

Burn-in: as in Dynare the first ``floor(mh_drop * replic)`` draws of every chain are discarded by
the statistics, and kept draws of several chains are pooled chain after chain.

Conventions of Dynare that are reproduced on purpose (they are easy to trip over):

* The "posterior mode" of the draws and the scaling constant of the modified harmonic mean use
  the highest posterior kernel over *all* draws, burn-in included.
* The posterior mean and standard deviation reported by the Geweke test are those of the whole
  chain, burn-in included; only the two sub-samples that are compared exclude it.
* The Raftery-Lewis diagnostic is computed on the whole chain, burn-in included (Dynare uses the
  last file of draws of each chain, which is the whole chain unless it is very long).
* The inefficiency factors are ``1 + 2 sum_{i=0}^{Nc} w((i+1)/Nc) rho_i``: the Parzen window is
  shifted by one lag and the sum includes lag 0, so they are close to the textbook factor plus 2
  (about 3 for independent draws).
* The HPD interval is the shortest among intervals spanning ``N - round((1 - p) N) - 1`` order
  statistics, which is one less than a ``p`` share of the draws.
* The covariance used by the modified harmonic mean divides by N, not N - 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["kept_draws", "pooled_draws", "posterior_moments", "posterior_statistics", "PosteriorStatistics",
           "kernel_density_estimate", "optimal_bandwidth", "mh_mode", "modified_harmonic_mean", "MarginalDensity",
           "model_comparison", "inefficiency_factors", "mcmc_ifac", "geweke_moments", "geweke", "raftery_lewis",
           "raftery_lewis_diagnostics", "brooks_gelman", "convergence_diagnostics", "jumping_covariance",
           "tune_jscale", "save_chains", "load_chains", "posterior_subsample", "number_of_subdraws", "irf_shocks",
           "irf_at", "posterior_irf", "PosteriorIRF", "moments_along_draws", "lower_cholesky_covariance",
           "mode_check", "mode_check_grid", "prior_density"]


def _mround(x):
    """MATLAB's round(): halves away from zero."""
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


def _iround(x) -> int:
    return int(_mround(x))


def _seqsum(a):
    """Sum along the last axis in strict left-to-right order (numpy's sum is pairwise)."""
    a = np.asarray(a, float)
    if a.shape[-1] == 0:
        return np.zeros(a.shape[:-1])
    return np.add.accumulate(a, axis=-1)[..., -1]


# ====================================================================== draws
def _unpack(mc, log_posterior=None):
    """(draws (chains, T, n), log posterior (chains, T) or None, default drop share)."""
    if hasattr(mc, "draws"):
        lp = mc.log_posterior if log_posterior is None else log_posterior
        return np.asarray(mc.draws, float), (None if lp is None else np.asarray(lp, float)), float(mc.drop)
    d = np.asarray(mc, float)
    if d.ndim == 2:
        d = d[None, :, :]
    if d.ndim != 3:
        raise ValueError("draws must be shaped (chains, draws, parameters)")
    lp = None if log_posterior is None else np.asarray(log_posterior, float).reshape(d.shape[0], d.shape[1])
    return d, lp, 0.5


def first_kept(replic: int, drop: float) -> int:
    """0-based index of the first draw that is kept (``CutSample.m``: ``floor(mh_drop * replic)``)."""
    return max(0, int(np.floor(drop * replic)))


def kept_draws(mc, drop=None):
    """Draws after the burn-in, shaped (chains, kept draws, parameters)."""
    d, _, d0 = _unpack(mc)
    drop = d0 if drop is None else drop
    return d[:, first_kept(d.shape[1], drop):, :]


def pooled_draws(mc, drop=None):
    """Kept draws of all chains, one chain after the other (``GetAllPosteriorDraws.m``)."""
    k = kept_draws(mc, drop)
    return k.reshape(-1, k.shape[2])


# ====================================================================== kernel density
_SQ2PI = np.sqrt(2 * np.pi)
_KERNELS = {
    "gaussian": lambda x: (1 / _SQ2PI) * np.exp(-0.5 * x ** 2),
    "uniform": lambda x: 0.5 * (np.abs(x) <= 1),
    "triangle": lambda x: (1 - np.abs(x)) * (np.abs(x) <= 1),
    "epanechnikov": lambda x: 0.75 * (1 - x ** 2) * (np.abs(x) <= 1),
    "quartic": lambda x: 0.9375 * ((1 - x ** 2) ** 2) * (np.abs(x) <= 1),
    "triweight": lambda x: 1.09375 * ((1 - x ** 2) ** 3) * (np.abs(x) <= 1),
    "cosinus": lambda x: (np.pi / 4) * np.cos((np.pi / 2) * x) * (np.abs(x) <= 1),
}
# (mu02, mu21) of mh_optimal_bandwidth.m
_KERNEL_CONSTANTS = {"gaussian": (1 / (2 * np.sqrt(np.pi)), 1.0), "uniform": (0.5, 1 / 3), "triangle": (2 / 3, 1 / 6),
                     "epanechnikov": (3 / 5, 1 / 5), "quartic": (15 / 21, 1 / 7), "triweight": (350 / 429, 1 / 9),
                     "cosinus": (np.pi ** 2 / 16, (np.pi ** 2 - 8) / np.pi ** 2)}
_K4 = {"gaussian": lambda x: (1 / _SQ2PI) * (3 - 6 * x ** 2 + x ** 4) * np.exp(-0.5 * x ** 2),
       "triweight": lambda x: (-1575 / 4 * x ** 2 + 315 / 4) * (np.abs(x) <= 1),
       "cosinus": lambda x: (1 / 64 * np.cos(np.pi * x / 2) * np.pi ** 5) * (np.abs(x) <= 1)}
_K6 = {"gaussian": lambda x: (1 / _SQ2PI) * (-15 + 45 * x ** 2 - 15 * x ** 4 + x ** 6) * np.exp(-0.5 * x ** 2),
       "triweight": lambda x: (-1575 / 2) * (np.abs(x) <= 1),
       "cosinus": lambda x: (-1 / 256 * np.cos(np.pi * x / 2) * np.pi ** 7) * (np.abs(x) <= 1)}


def _repeated_draws_correction(sorted_draws) -> float:
    """``correction_for_repeated_draws`` of mh_optimal_bandwidth.m: mean of ``2 m_i - 1`` where
    ``m_i`` counts the draws from position i on that equal draw i (to within 2 eps)."""
    d = np.asarray(sorted_draws, float)
    n = d.size
    tol = 2 * np.finfo(float).eps
    j = np.searchsorted(d, d + tol, side="left")          # first j with d[j] >= d[i] + tol (approximately)
    idx = np.arange(n)
    for _ in range(4):                                    # settle the comparison exactly as written in Dynare
        up = (j < n) & ((d[np.minimum(j, n - 1)] - d) < tol)
        down = (j > idx + 1) & ~((d[np.maximum(j - 1, 0)] - d) < tol)
        if not (up.any() or down.any()):
            break
        j = j + up - down
    return float(np.sum(2 * (j - idx) - 1) / n)


def optimal_bandwidth(data, bandwidth=0, kernel_function="gaussian") -> float:
    """Bandwidth of the kernel density estimator (``mh_optimal_bandwidth.m``); ``data`` must be sorted.

    ``bandwidth=0``: Silverman's rule of thumb; ``-1``: Sheather and Jones plug-in (quadratic cost in
    the number of draws); both corrected for the repeated draws of a Metropolis-Hastings chain
    (Skold and Roberts 2003). ``-2`` (local bandwidths) cannot run in Dynare 7.2 (it refers to an
    undefined variable) and is not provided."""
    kern = kernel_function.lower()
    if kern not in _KERNELS:
        raise ValueError(f"kernel function {kernel_function!r} is not implemented")
    d = np.asarray(data, float).ravel()
    n = d.size
    mu02, mu21 = _KERNEL_CONSTANTS[kern]
    if bandwidth not in (0, -1):
        if bandwidth == -2:
            raise NotImplementedError("bandwidth=-2 (local bandwidths) fails in Dynare 7.2 itself and is not ported")
        raise ValueError("bandwidth must be 0 or -1")
    correction = _repeated_draws_correction(d)
    sigma = np.std(d, ddof=1)
    if bandwidth == 0:
        h = 2 * sigma * (np.sqrt(np.pi) * mu02 / (12 * mu21 ** 2 * n)) ** (1 / 5)
        return float(h * correction ** (1 / 5))
    if kern not in _K4:
        raise ValueError("the plug-in bandwidth needs the gaussian, triweight or cosinus kernel")
    k4, k6 = _K4[kern], _K6[kern]

    def double_sum(fun, g):
        tot = 0.0
        for start in range(0, n, 512):
            tot += float(np.sum(fun((d[start:start + 512, None] - d[None, :]) / g)))
        return tot
    itilda4 = 8 * 7 * 6 * 5 / (((2 * sigma) ** 9) * np.sqrt(np.pi))
    g3 = abs(2 * correction * float(k6(np.zeros(1))[0]) / (mu21 * itilda4 * n)) ** (1 / 9)
    ihat3 = -double_sum(k6, g3) / (n ** 2 * g3 ** 7)
    g2 = abs(2 * correction * float(k4(np.zeros(1))[0]) / (mu21 * ihat3 * n)) ** (1 / 7)
    ihat2 = double_sum(k4, g2) / (n ** 2 * g2 ** 5)
    return float((correction * mu02 / (n * ihat2 * mu21 ** 2)) ** (1 / 5))


def kernel_density_estimate(data, bandwidth, number_of_grid_points=2 ** 9, kernel_function="gaussian"):
    """Kernel density estimate on a regular grid by fast Fourier transform (``kernel_density_estimate.m``).
    Returns ``(abscissa, density)``; the grid extends a third of the data range beyond both extremes."""
    d = np.asarray(data, float).ravel()
    ng = int(number_of_grid_points)
    if ng < 2 or ng & (ng - 1):
        raise ValueError("the number of grid points must be a power of 2")
    kernel = _KERNELS[kernel_function.lower()]
    n = d.size
    lo, hi = d.min() - (d.max() - d.min()) / 3, d.max() + (d.max() - d.min()) / 3
    abscissa = np.linspace(lo, hi, ng)
    inc = abscissa[1] - abscissa[0]
    xa = (d - lo) / (hi - lo) * ng
    indx = np.floor(xa).astype(int)
    temp = xa - indx
    xi = np.zeros(ng)
    np.add.at(xi, indx, 1 - temp)
    np.add.at(xi, indx + 1, temp)
    xk = np.arange(-ng, ng) * inc
    kk = kernel(xk / bandwidth)
    kk = kk / (np.sum(kk) * inc * n)
    f = np.fft.ifft(np.fft.fft(np.fft.fftshift(kk)) * np.fft.fft(np.concatenate([xi, np.zeros(ng)])))
    return abscissa, np.real(f[:ng])


# ====================================================================== posterior statistics
def posterior_moments(draws, mh_conf_sig=0.9, density=False, kernel_options=None) -> dict:
    """Moments of one series of draws as computed by Dynare (``posterior_moments.m``).

    Returns ``mean``, ``median``, ``variance`` (N - 1), ``hpd`` (lower, upper: the shortest interval
    by Dynare's rule, NaN with too few draws), ``deciles`` (9) and, when ``density`` is true,
    ``density`` (grid points x 2: abscissa, kernel density estimate; NaN if the variance is below
    1e-12). ``kernel_options``: dict with ``gridpoints`` (512), ``bandwidth`` (0) and
    ``kernel_function`` ('gaussian')."""
    xx = np.sort(np.asarray(draws, float).ravel())
    n = xx.size
    out = {"mean": float(np.mean(xx)), "median": float(np.median(xx)),
           "variance": float(np.var(xx, ddof=1)) if n > 1 else np.nan}
    hpd_draws = _iround((1 - mh_conf_sig) * n)
    if hpd_draws > 2:
        # kk(ii) = xx(jj) - xx(ii), jj = n - hpd_draws + ii - 1 (1-based)
        ii = np.arange(hpd_draws)
        kk = xx[n - hpd_draws - 1 + ii] - xx[ii]
        idx = int(np.argmin(kk))
        out["hpd"] = np.array([xx[idx], xx[idx] + kk[idx]])
    else:
        out["hpd"] = np.full(2, np.nan)
    if n > 9:
        out["deciles"] = xx[np.array([_iround(q * n) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]) - 1]
    else:
        out["deciles"] = np.full(9, np.nan)
    if density:
        ko = {"gridpoints": 2 ** 9, "bandwidth": 0, "kernel_function": "gaussian"}
        ko.update(kernel_options or {})
        if out["variance"] > 1e-12:
            h = optimal_bandwidth(xx, ko["bandwidth"], ko["kernel_function"])
            out["bandwidth"] = h
            out["density"] = np.column_stack(kernel_density_estimate(xx, h, ko["gridpoints"], ko["kernel_function"]))
        else:
            out["density"] = np.full((int(ko["gridpoints"]), 2), np.nan)
    return out


_PRIOR_NAMES = {0: "", 1: "beta", 2: "gamm", 3: "norm", 4: "invg", 5: "unif", 6: "invg2", 8: "weibl"}


@dataclass
class PosteriorStatistics:
    """Posterior moments of the estimated parameters (Dynare's ``oo_.posterior_mean`` etc.)."""
    names: list
    mean: np.ndarray
    median: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    hpd_inf: np.ndarray
    hpd_sup: np.ndarray
    deciles: np.ndarray                         # (parameters, 9)
    mode: np.ndarray                            # draw with the highest posterior kernel
    mode_log_posterior: float
    mh_conf_sig: float
    density: np.ndarray = field(default=None, repr=False)      # (parameters, grid points, 2) or None
    prior_mean: np.ndarray = None
    prior_std: np.ndarray = None
    prior_shape: np.ndarray = None

    def as_dict(self, what="mean"):
        return dict(zip(self.names, np.asarray(getattr(self, what)).tolist()))

    def table(self) -> str:
        """Prior and posterior table in the layout of Dynare's ``ESTIMATION RESULTS``."""
        width = max(len(s) for s in self.names) + 2
        hpd = f"{self.mh_conf_sig * 100:g}% HPD interval"
        lines = [f"{'':<{width}} {'prior mean':>12} {'post. mean':>12} {hpd:>23} {'prior':>8} {'pstdev':>12}"]
        for i, name in enumerate(self.names):
            pm = np.nan if self.prior_mean is None else self.prior_mean[i]
            ps = np.nan if self.prior_std is None else self.prior_std[i]
            sh = "" if self.prior_shape is None else _PRIOR_NAMES.get(int(self.prior_shape[i]), "")
            lines.append(f"{name:<{width}} {pm:12.3f} {self.mean[i]: 12.4f} {self.hpd_inf[i]:11.4f} "
                         f"{self.hpd_sup[i]:11.4f} {sh:>8} {ps:12.4f}")
        return "\n".join(lines)


def mh_mode(mc, log_posterior=None):
    """Draw with the highest posterior kernel and that kernel value. As Dynare
    (``compute_mh_covariance_matrix.m``) the search covers all draws, burn-in included; ties go to the
    first chain and the first draw."""
    d, lp, _ = _unpack(mc, log_posterior)
    if lp is None:
        raise ValueError("the log posterior of the draws is needed")
    best, x = -np.inf, np.full(d.shape[2], np.nan)
    for b in range(d.shape[0]):
        k = int(np.argmax(lp[b]))
        if lp[b, k] > best:
            best, x = float(lp[b, k]), d[b, k].copy()
    return x, best


def posterior_statistics(mc, est=None, mh_conf_sig=None, drop=None, density=True, names=None,
                         log_posterior=None) -> PosteriorStatistics:
    """Posterior mean, median, standard deviation, HPD interval, deciles and kernel density of every
    parameter from the kept draws of all chains (``GetPosteriorParametersStatistics.m``).

    ``est`` (an ``Estimation``) supplies the prior columns of the table and the default
    ``mh_conf_sig`` (the ``.mod`` option, else 0.9)."""
    d, lp, _ = _unpack(mc, log_posterior)
    if mh_conf_sig is None:
        mh_conf_sig = float(est.options.get("mh_conf_sig", 0.9)) if est is not None else 0.9
    x = pooled_draws(mc, drop)
    n = x.shape[1]
    res = [posterior_moments(x[:, j], mh_conf_sig, density=density) for j in range(n)]
    names = list(names or getattr(mc, "names", None) or (est.names if est is not None else [f"p{j}" for j in range(n)]))
    mode, mlp = mh_mode(mc, lp) if lp is not None else (np.full(n, np.nan), np.nan)
    var = np.array([r["variance"] for r in res])
    out = PosteriorStatistics(
        names=names, mean=np.array([r["mean"] for r in res]), median=np.array([r["median"] for r in res]),
        std=np.sqrt(var), variance=var, hpd_inf=np.array([r["hpd"][0] for r in res]),
        hpd_sup=np.array([r["hpd"][1] for r in res]), deciles=np.array([r["deciles"] for r in res]),
        mode=mode, mode_log_posterior=mlp, mh_conf_sig=float(mh_conf_sig),
        density=np.array([r["density"] for r in res]) if density else None)
    if est is not None:
        out.prior_mean, out.prior_std, out.prior_shape = est.prior.p1.copy(), est.prior.p2.copy(), est.prior.shape.copy()
    return out


# ====================================================================== marginal data density
@dataclass
class MarginalDensity:
    value: float                 # log marginal data density: mean over the nine truncation probabilities
    table: np.ndarray            # (9, 2): truncation probability, estimate
    increase: float              # factor applied to the covariance of the weighting density (1 normally)
    converged: bool              # False if Dynare would print "There's probably a problem with the ... estimator"
    mean: np.ndarray = field(default=None, repr=False)
    covariance: np.ndarray = field(default=None, repr=False)
    log_posterior_max: float = np.nan

    def __float__(self):
        return float(self.value)


def _recursive_moments(x):
    """Mean and covariance (divisor N) by the recursion of ``recursive_moments.m``. The recursion is kept
    because its rounding error (of the order of eps times the squared means) is larger than that of the
    direct formula and shows in the result at the 1e-9 level."""
    m0 = np.zeros(x.shape[1])
    s0 = np.zeros((x.shape[1], x.shape[1]))
    for t, row in enumerate(x, start=1):
        m1 = m0 + (row - m0) / t
        qq = np.outer(m1, m1)
        s0 = s0 + ((np.outer(row, row) - qq - s0) + (t - 1) * (np.outer(m0, m0) - qq.T)) / t
        m0 = m1
    return m0, s0


def modified_harmonic_mean(mc, log_posterior=None, drop=None, tolerance=0.01) -> MarginalDensity:
    """Log marginal data density by Geweke's (1999) modified harmonic mean, as ``marginal_density.m``.

    The weighting density is a normal with the mean and covariance (divisor N) of the kept draws,
    truncated to the ellipsoid of probability p, for p = 0.1, ..., 0.9; the result is the mean of the
    nine estimates. If the first and the last differ by more than ``tolerance`` in relative terms the
    covariance is scaled up by 1.2 and everything is recomputed, repeatedly until the factor exceeds
    20 (then ``converged`` is False and the last table is used, as in Dynare)."""
    d, lp, d0 = _unpack(mc, log_posterior)
    if lp is None:
        raise ValueError("the log posterior of the draws is needed")
    drop = d0 if drop is None else drop
    first = first_kept(d.shape[1], drop)
    x = d[:, first:, :].reshape(-1, d.shape[2])
    logpo = lp[:, first:].reshape(-1)
    ntot, npar = x.shape
    lpost_mode = float(np.max(lp))                      # over all draws, burn-in included (see module notes)
    mu, sigma = _recursive_moments(x)
    xc = x - mu
    try:
        logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(sigma))))
    except np.linalg.LinAlgError:
        logdet = np.nan
    inv_sigma = np.linalg.inv(sigma)
    dev0 = np.einsum("ij,jk,ik->i", xc, inv_sigma, xc)
    probs = np.arange(1, 10) / 10
    increase, table, converged = 1.0, np.zeros((9, 2)), True
    while True:
        deviation = dev0 / increase
        for k, p in enumerate(probs):
            critval = stats.chi2.ppf(p, npar)
            inside = deviation <= critval
            terms = (-np.log(p) - (npar * np.log(2 * np.pi) + (npar * np.log(increase) + logdet) + deviation[inside]) / 2
                     - logpo[inside] + lpost_mode)
            with np.errstate(over="ignore", divide="ignore"):
                tmp = np.sum(np.exp(terms))
                norm_proba = 0.0
                if np.isinf(tmp):
                    norm_proba = np.max(terms)
                    tmp = np.sum(np.exp(terms - norm_proba))
                table[k] = p, lpost_mode - np.log(tmp / ntot) - norm_proba
        with np.errstate(invalid="ignore", divide="ignore"):
            bad = abs((table[8, 1] - table[0, 1]) / table[8, 1]) > tolerance or np.any(np.isinf(table[:, 1]))
        if not bad:
            break
        increase *= 1.2
        if increase > 20 and increase / 1.2 != 1.0:
            converged = False
            break
    return MarginalDensity(value=float(np.mean(table[:, 1])), table=table.copy(), increase=increase if converged else increase / 1.2,
                           converged=converged, mean=mu, covariance=sigma, log_posterior_max=lpost_mode)


def model_comparison(log_marginal_densities, priors=None, names=None) -> dict:
    """Posterior odds and model probabilities from log marginal data densities (``model_comparison.m``).

    Returns a dict of arrays, one entry per model: ``prior`` (rescaled to sum to one),
    ``log_marginal_density``, ``bayes_ratio`` (against the first model; with user priors Dynare
    reports the prior-weighted ratio under this name), ``posterior_odds_ratio`` (against the first
    model) and ``posterior_model_probability``; plus ``names``."""
    mld = np.asarray([float(v) for v in log_marginal_densities], float)
    m = mld.size
    user_prior = priors is not None
    pr = np.full(m, 1.0 / m) if priors is None else np.asarray(priors, float)
    if pr.size != m:
        raise ValueError("one prior probability per model is needed")
    if abs(pr.sum() - 1) > 1e-6:
        pr = pr / pr.sum()
    lmpd = np.log(pr) + mld
    elmpd = np.exp(lmpd - lmpd.max())
    odds = np.exp(lmpd - lmpd[0])
    return {"names": list(names) if names is not None else [f"model {i + 1}" for i in range(m)],
            "prior": pr, "log_marginal_density": mld,
            "bayes_ratio": odds if user_prior else np.exp(mld - mld[0]),
            "posterior_odds_ratio": odds, "posterior_model_probability": elmpd / elmpd.sum()}


# ====================================================================== convergence diagnostics
def mcmc_ifac(x, nc) -> float:
    """Inefficiency factor of one series with a Parzen window of ``nc`` lags, exactly as ``mcmc_ifac.m``:
    its sum starts at lag 0 with the window shifted by one lag, which gives about 3 (not 1) for
    independent draws."""
    x = np.asarray(x, float).ravel()
    nc = int(np.floor(min(nc, x.size / 2)))
    if nc % 2:
        nc -= 1
    m, sd = x.mean(), x.std()
    xc = x - m
    acf = np.ones(nc + 1)
    full = np.correlate(xc, xc, mode="full")[x.size - 1:x.size + nc] if x.size * nc < 5e7 else None
    if full is not None:
        acf[1:] = full[1:] / (x.size * sd ** 2)
    else:
        for i in range(1, nc + 1):
            acf[i] = xc[i:] @ xc[:-i] / (x.size * sd ** 2)
    i = np.arange(1, nc + 2, dtype=float)
    parzen = np.where(i <= nc / 2, 1 - 6 * (i / nc) ** 2 + 6 * (i / nc) ** 3, 2 * (1 - i / nc) ** 3)
    return float(1 + 2 * np.sum(parzen * acf))


def inefficiency_factors(mc, drop=None) -> np.ndarray:
    """Inefficiency factor of every parameter in every chain, shaped (chains, parameters), from the kept
    draws with ``min(1000, kept/2)`` lags (``mcmc_diagnostics.m``)."""
    k = kept_draws(mc, drop)
    nc = min(1000, k.shape[1] / 2)
    return np.array([[mcmc_ifac(k[b, :, j], nc) for j in range(k.shape[2])] for b in range(k.shape[0])])


def geweke_moments(draws, taper_steps=(4, 8, 15)) -> dict:
    """Mean, standard deviation, numerical standard errors and relative numerical efficiencies of one
    series (``geweke_moments.m``): 100 batch means, autocovariances tapered with the given steps."""
    x = np.asarray(draws, float).ravel()
    ng = 100
    ns = x.size // ng
    used = ns * ng
    blocks = x[:used].reshape(ng, ns)
    # Dynare computes the variance as E[x^2] - E[x]^2, which loses up to ten digits for a parameter with a
    # small coefficient of variation; summing in its order (left to right) keeps the two results together
    wm = _seqsum(blocks) / ns
    wv = _seqsum(blocks ** 2) / ns
    mean = _seqsum(wm) / ng
    var = _seqsum(wv) / ng - mean ** 2
    nse = np.std(x[:used]) / np.sqrt(used)
    out = {"posteriormean": float(mean), "posteriorstd": float(np.sqrt(var)), "nse_iid": float(nse),
           "rne_iid": float(var / (used * nse ** 2))}
    c = wm - mean
    acov = np.array([c[lag:] @ c[:ng - lag] / ng for lag in range(ng)])
    for t in taper_steps:
        t = int(t)
        lags = np.arange(1, t)
        s = acov[0] + np.sum(2 * (1 - lags / t) * acov[lags])
        nse_t = np.sqrt(s / ng)
        out[f"nse_taper_{t}"] = float(nse_t)
        out[f"rne_taper_{t}"] = float(var / (used * nse_t ** 2))
    return out


def geweke(mc, geweke_interval=(0.2, 0.5), taper_steps=(4, 8, 15), drop=None) -> dict:
    """Geweke (1992) convergence test for every chain and parameter (``mcmc_diagnostics.m``).

    The mean over the first ``geweke_interval[0]`` share of the kept draws is compared with the mean
    over the share after ``geweke_interval[1]``. Returns arrays shaped (chains, parameters, ...):
    ``moments`` {name: array} for the whole chain (burn-in included, as Dynare reports them),
    ``pooled_mean``, ``pooled_nse`` and ``prob_chi2_test`` with one column without taper and one per
    taper step, and the 1-based sample limits ``samples``."""
    d, _, d0 = _unpack(mc)
    drop = d0 if drop is None else drop
    gi = np.asarray(geweke_interval, float).ravel()
    if gi.size != 2 or np.any(gi < 0) or np.any(gi > 1) or gi[1] - gi[0] < 0:
        gi = np.array([0.2, 0.5])                         # Dynare falls back to its default
    nb, T, n = d.shape
    first = max(1, int(np.ceil(drop * T)))
    last_begin = first + _iround(gi[0] * T * (1 - drop))
    first_end = first + _iround(gi[1] * T * (1 - drop))
    taper_steps = [int(t) for t in np.atleast_1d(taper_steps)]
    keys = ["posteriormean", "posteriorstd", "nse_iid", "rne_iid"] + [f"{a}_taper_{t}" for t in taper_steps for a in ("nse", "rne")]
    moments = {k: np.zeros((nb, n)) for k in keys}
    cols = 1 + len(taper_steps)
    pooled_mean, pooled_nse, prob = (np.zeros((nb, n, cols)) for _ in range(3))
    nse_keys = ["nse_iid"] + [f"nse_taper_{t}" for t in taper_steps]
    for b in range(nb):
        for j in range(n):
            x = d[b, :, j]
            full = geweke_moments(x, taper_steps)
            for k in keys:
                moments[k][b, j] = full[k]
            r1 = geweke_moments(x[first - 1:last_begin], taper_steps)
            r2 = geweke_moments(x[first_end - 1:], taper_steps)
            for c, nk in enumerate(nse_keys):
                nse = np.array([r1[nk], r2[nk]])
                means = np.array([r1["posteriormean"], r2["posteriormean"]])
                w = np.sum(1 / nse ** 2)
                pooled_mean[b, j, c] = np.sum(means / nse ** 2) / w
                pooled_nse[b, j, c] = 1 / np.sqrt(w)
                prob[b, j, c] = 1 - stats.chi2.cdf((means[0] - means[1]) ** 2 / np.sum(nse ** 2), 1)
    return {"moments": moments, "pooled_mean": pooled_mean, "pooled_nse": pooled_nse, "prob_chi2_test": prob,
            "samples": (first, last_begin, first_end, T), "taper_steps": taper_steps}


def _g2_second_order(dch):
    n = dch.size
    tran = np.zeros((2, 2, 2))
    if n >= 3:
        np.add.at(tran, (dch[:-2], dch[1:-1], dch[2:]), 1)
    g2 = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if tran[a, b, c] != 0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fitted = (tran[a, b, 0] + tran[a, b, 1]) * (tran[0, b, c] + tran[1, b, c]) / \
                            (tran[0, b, 0] + tran[0, b, 1] + tran[1, b, 0] + tran[1, b, 1])
                        g2 += np.log(tran[a, b, c] / fitted) * tran[a, b, c]
    g2 *= 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return g2, g2 - np.log(n - 2) * 2


def _g2_independence(dch):
    n = dch.size
    trans = np.zeros((2, 2))
    if n >= 2:
        np.add.at(trans, (dch[:-1], dch[1:]), 1)
    dcm1 = n - 1
    g2 = 0.0
    for a in range(2):
        for b in range(2):
            if trans[a, b] != 0:
                fitted = (trans[a, 0] + trans[a, 1]) * (trans[0, b] + trans[1, b]) / dcm1
                g2 += np.log(trans[a, b] / fitted) * trans[a, b]
    g2 *= 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return g2, g2 - np.log(dcm1)


def raftery_lewis(runs, q=0.025, r=0.005, s=0.95) -> dict:
    """Raftery and Lewis (1992) diagnostic for every column of ``runs`` (``raftery_lewis.m``): burn-in
    ``M_burn``, draws needed ``N_prec``, their sum ``N_total``, thinning ``k_thin`` for a first-order
    chain, ``k_ind`` for independence, ``I_stat``, and ``N_min`` (draws needed with independent draws)
    to estimate the ``q`` quantile to within ``r`` with probability ``s``."""
    runs = np.asarray(runs, float)
    if runs.ndim == 1:
        runs = runs[:, None]
    if not 0 < q < 1:
        raise ValueError("Quantile must be between 0 and 1")
    n_runs, n_vars = runs.shape
    phi = stats.norm.ppf((s + 1) / 2)
    out = {k: np.full(n_vars, np.nan) for k in ("M_burn", "N_prec", "k_thin", "k_ind", "I_stat", "N_total")}
    out["N_min"] = float(np.trunc(phi ** 2 * (1 - q) * q / r ** 2 + 1))
    epss = 0.001
    for nv in range(n_vars):
        work = (runs[:, nv] <= np.quantile(runs[:, nv], q, method="hazen")).astype(int)
        k_thin, bic = 1, 1.0
        while bic > 0:
            thinned = work[::k_thin]
            _, bic = _g2_second_order(thinned)
            k_thin += 1
        k_thin -= 1
        tm = np.zeros((2, 2))
        np.add.at(tm, (thinned[:-1], thinned[1:]), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = tm[0, 1] / (tm[0, 0] + tm[0, 1])
            beta = tm[1, 0] / (tm[1, 0] + tm[1, 1])
            kmind = k_thin
            _, bic = _g2_independence(thinned)
            while bic > 0:
                thinned = work[::kmind]
                _, bic = _g2_independence(thinned)
                kmind += 1
            m_star = np.log((alpha + beta) * epss / max(alpha, beta)) / np.log(abs(1 - alpha - beta))
            out["M_burn"][nv] = np.trunc((m_star + 1) * k_thin)
            n_star = (2 - (alpha + beta)) * alpha * beta * phi ** 2 / ((alpha + beta) ** 3 * r ** 2)
            out["N_prec"][nv] = np.trunc(n_star + 1) * k_thin
            out["I_stat"][nv] = (out["M_burn"][nv] + out["N_prec"][nv]) / out["N_min"]
            out["k_ind"][nv] = max(np.trunc(out["I_stat"][nv] + 1), kmind)
            out["k_thin"][nv] = k_thin
            out["N_total"][nv] = out["M_burn"][nv] + out["N_prec"][nv]
    return out


def raftery_lewis_diagnostics(mc, qrs=(0.025, 0.005, 0.95)) -> list:
    """Raftery-Lewis diagnostics of every chain (list of dicts). As in Dynare the whole chain is
    used, burn-in included; invalid ``qrs`` fall back to (0.025, 0.005, 0.95)."""
    d, _, _ = _unpack(mc)
    qrs = np.asarray(qrs, float).ravel()
    if qrs.size != 3 or np.any(qrs < 0) or np.any(qrs > 1) or qrs[0] - qrs[1] <= 0:
        qrs = np.array([0.025, 0.005, 0.95])
    return [raftery_lewis(d[b], *qrs) for b in range(d.shape[0])]


def _bg_series(tmp, nb, origin, step, drop):
    """Six Brooks-Gelman series for one scalar series ``tmp`` (chains, draws)."""
    T = tmp.shape[1]
    alpha = 0.2
    iters = np.arange(origin, T + 1, step)
    out = np.zeros((iters.size, 6))
    for w, it in enumerate(iters):
        linea = int(np.ceil(drop * it))
        n = it - linea + 1
        cinf, csup = max(1, _iround(n * alpha / 2)), _iround(n * (1 - alpha / 2))
        CINF, CSUP = max(1, _iround(nb * n * alpha / 2)), _iround(nb * n * (1 - alpha / 2))
        win = tmp[:, max(linea, 1) - 1:it]                 # draws linea..iter (1-based) of every chain
        pooled = np.sort(win.ravel())
        out[w, 0] = pooled[CSUP - 1] - pooled[CINF - 1]
        pm = pooled.mean()
        out[w, 2] = np.sum((pooled - pm) ** 2) / (nb * n - 1)
        out[w, 4] = np.sum(np.abs(pooled - pm) ** 3) / (nb * n - 1)
        for b in range(nb):
            pmet = np.sort(win[b])
            out[w, 1] += pmet[csup - 1] - pmet[cinf - 1]
            wm = pmet.mean()
            out[w, 3] += np.sum((pmet - wm) ** 2) / (n - 1)
            out[w, 5] += np.sum(np.abs(pmet - wm) ** 3) / (n - 1)
    out[:, [1, 3, 5]] /= nb
    return iters, out


def brooks_gelman(mc, log_posterior=None, drop=None) -> dict:
    """Brooks and Gelman (1998) convergence diagnostics as computed for Dynare's figures
    (``mcmc_diagnostics.m``, ``mcmc_diagnostics_core.m``).

    Evaluated at draws 1000, 1000 + step, ... with ``step = ceil((replic - 1000) / 100)``; at each
    point only the draws after the first ``drop`` share up to that point are used. Returns ``x``
    (the evaluation points), ``univariate`` shaped (points, 6, parameters) and ``multivariate``
    shaped (points, 6), the latter from the posterior kernel. The six columns are: length of the
    80% interval, second and third central absolute moments, each for the pooled draws and then as
    the mean over chains ("within"). Needs at least two chains and more than 2000 draws per chain
    (Dynare's conditions for the random-walk sampler); otherwise None is returned for the series."""
    d, lp, d0 = _unpack(mc, log_posterior)
    drop = d0 if drop is None else drop
    nb, T, n = d.shape
    if nb < 2 or T <= 2000:
        return {"x": np.zeros(0), "univariate": None, "multivariate": None}
    origin = 1000
    step = int(np.ceil((T - origin) / 100))
    uni = []
    for j in range(n):
        x, s = _bg_series(d[:, :, j], nb, origin, step, drop)
        uni.append(s)
    multi = None if lp is None else _bg_series(lp, nb, origin, step, drop)[1]
    return {"x": x, "univariate": np.stack(uni, axis=2), "multivariate": multi}


def convergence_diagnostics(mc, est=None, drop=None, geweke_interval=None, taper_steps=None,
                            raftery_lewis_qrs=None, log_posterior=None) -> dict:
    """All diagnostics Dynare computes after Metropolis-Hastings (``oo_.convergence`` and the
    Brooks-Gelman figures): ``inefficiency_factors``, ``geweke``, ``brooks_gelman`` and, if
    ``raftery_lewis_qrs`` is given or the ``.mod`` file asks for ``raftery_lewis_diagnostics``,
    ``raftery_lewis``. Options default to those of the ``estimation`` command of ``est``."""
    o = est.options if est is not None else {}
    d, _, _ = _unpack(mc)
    gi = geweke_interval if geweke_interval is not None else o.get("geweke_interval", (0.2, 0.5))
    ts = taper_steps if taper_steps is not None else o.get("taper_steps", (4, 8, 15))
    out = {"inefficiency_factors": inefficiency_factors(mc, drop)}
    if d.shape[1] > 2000:                                # Dynare skips the rest for short chains
        out["geweke"] = geweke(mc, gi, ts, drop)
        qrs = raftery_lewis_qrs
        if qrs is None and o.get("raftery_lewis_diagnostics"):
            qrs = o.get("raftery_lewis_qrs", (0.025, 0.005, 0.95))
        if qrs is not None:
            out["raftery_lewis"] = raftery_lewis_diagnostics(mc, qrs)
        out["brooks_gelman"] = brooks_gelman(mc, log_posterior, drop)
    return out


# ====================================================================== sampler set-up and chain files
def jumping_covariance(est, kind="hessian", inv_hessian=None, as_dynare=False) -> np.ndarray:
    """Covariance of the random-walk proposal before scaling (``set_mcmc_jumping_covariance.m``).

    ``kind``: ``'hessian'`` (the inverse Hessian at the mode, ``inv_hessian``), ``'prior_variance'``
    (diagonal matrix of the prior variances; an error if one is infinite), ``'identity_matrix'``, a
    square matrix, or the name of a file holding a matrix called ``jumping_covariance`` (``.npz``,
    ``.npy`` or ``.mat``, looked up next to the ``.mod`` file too).

    Dynare 7.2 treats a matrix loaded from a file as a *Hessian*: it uses its inverse as the
    covariance, although the manual describes the file as holding the covariance matrix. pydyn
    follows the manual and uses the matrix as the covariance; ``as_dynare=True`` reproduces
    Dynare's inversion."""
    n = est.n
    if isinstance(kind, str) and kind == "hessian":
        if inv_hessian is None:
            raise ValueError("mcmc_jumping_covariance='hessian' needs the inverse Hessian: pass cov=... or a "
                             "ModeResult with an inverse Hessian")
        return np.asarray(inv_hessian, float)
    if isinstance(kind, str) and kind == "prior_variance":
        p2 = np.asarray(est.prior.p2, float)
        if np.any(~np.isfinite(p2)):
            raise ValueError("Infinite prior variances detected. You cannot use the prior variances as the proposal "
                             "density, if some variances are Inf.")
        return np.diag(p2 ** 2)
    if isinstance(kind, str) and kind == "identity_matrix":
        return np.eye(n)
    if isinstance(kind, (str, bytes)) or hasattr(kind, "__fspath__"):
        from pathlib import Path
        base = est.model.json_dir.parents[2] if hasattr(est.model, "json_dir") else Path(".")
        cands = [Path(kind)] + [Path(str(kind) + e) for e in (".npz", ".npy", ".mat")]
        cands += [base / c for c in cands if not c.is_absolute()]
        found = next((c for c in cands if c.is_file()), None)
        if found is None:
            raise ValueError(f"mcmc_jumping_covariance: no file {kind} (.npz, .npy or .mat)")
        if found.suffix == ".npy":
            mat = np.load(found)
        elif found.suffix == ".npz":
            with np.load(found) as z:
                if "jumping_covariance" not in z.files:
                    raise ValueError(f"No matrix named 'jumping_covariance' could be found in {found}")
                mat = z["jumping_covariance"]
        else:
            from scipy.io import loadmat
            data = loadmat(found)
            if "jumping_covariance" not in data:
                raise ValueError(f"No matrix named 'jumping_covariance' could be found in {found}")
            mat = data["jumping_covariance"]
    else:
        mat = kind
    mat = np.atleast_2d(np.asarray(mat, float))
    if mat.shape != (n, n):
        raise ValueError(f"'jumping_covariance' matrix must be square and have {n} rows and columns")
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        raise ValueError("Specified 'MCMC_jumping_covariance' is not positive definite") from None
    if as_dynare:
        hsd = np.sqrt(np.diag(mat))
        s = np.outer(hsd, hsd)
        return np.linalg.inv(mat / s) / s
    return mat


def tune_jscale(est, mode, cov, target=0.33, guess=None, maxiter=200000, stepsize=1000, rho=0.7, c1=0.02, c2=0.05,
                c3=4, rng=None, innovations=None, uniforms=None, bounds=None, return_info=False):
    """Scale of the random-walk proposal that gives the ``target`` acceptance rate
    (``mh_tune_jscale``; ``calibrate_mh_scale_parameter.m``).

    A chain is run from the mode with proposal ``last + N(0, scale^2 cov)``; every ``stepsize``
    draws the scale is multiplied by a damped ratio of the acceptance rate over the last
    ``stepsize`` draws to the target, and the search stops when the overall rate is within ``c2``
    and the recent rate within ``c1`` of the target ``c3 + 1`` times in a row, or after ``maxiter``
    draws. The defaults are Dynare's. ``guess``: initial scale (default ``2.38 / sqrt(n)``).
    The per-parameter ``jscale`` values of the ``estimated_params`` block play no role here, as in
    Dynare. ``innovations`` (rows of standard normals) and ``uniforms`` (a stream, one used per
    proposal with a finite posterior) replace the random numbers of ``rng``."""
    x = np.asarray(mode, float).copy()
    n = x.size
    lb, ub = est.prior.bounds(est.options["prior_trunc"]) if bounds is None else bounds
    rng = np.random.default_rng(0) if rng is None else rng
    dd = np.linalg.cholesky(np.asarray(cov, float)).T
    scale = 2.38 / np.sqrt(n) if guess is None else float(guess)
    logpo1 = est.log_posterior(x, bounds=(lb, ub))
    correction = 1.0
    j, jj, isux, jsux, i, used = 1, 1, 0, 0, 0, 0
    while j <= maxiter:
        z = rng.standard_normal(n) if innovations is None else np.asarray(innovations, float)[j - 1]
        prop = x + z @ (scale * dd)
        if np.all(prop > lb) and np.all(prop < ub):
            logpo0 = est.log_posterior(prop, bounds=(lb, ub))
        else:
            logpo0 = -np.inf
        if logpo0 > -np.inf:
            u = rng.random() if uniforms is None else uniforms[used]
            used += 1
            if np.log(u) < logpo0 - logpo1:
                x, logpo1 = prop, logpo0
                isux += 1
                jsux += 1
        if j % stepsize == 0:
            r1 = jsux / jj                      # acceptance over the last jj draws (jj starts at 1, then 0: as Dynare)
            r2 = isux / j
            ratio = r1 / target
            correction = correction ** rho * ratio ** (1 - rho) if abs(ratio - 1) > 0.05 else ratio
            scale = scale * correction if ratio > 0 else scale / 10
            jsux, jj = 0, 0
            i = i + 1 if (abs(r2 - target) < c2 and abs(r1 - target) < c1) else 0
            if i > c3:
                break
        j += 1
        jj += 1
    if return_info:
        return float(scale), {"draws": min(j, maxiter), "acceptance": isux / max(min(j, maxiter), 1), "uniforms_used": used}
    return float(scale)


def save_chains(path, mc) -> None:
    """Write the chains of an ``MCMCResult`` to a compressed ``.npz`` file.

    Arrays: ``draws`` (chains, draws, parameters), ``log_posterior`` (chains, draws), ``accepted``
    (chains, draws; booleans), ``names`` (parameter names), ``x0`` (starting points), ``drop``
    (burn-in share), ``jscale`` (scale of the proposal), ``jscale_vector`` (scale per parameter) and
    ``proposal_chol`` (upper Cholesky factor of the jumping covariance, so that the proposal of draw
    t is ``draws[t-1] + z @ (proposal_chol * jscale_vector)``); ``format`` is 1. Everything needed
    to continue the chains with ``Estimation.sample(load=path)`` is in the file."""
    data = {"format": np.array(1), "draws": np.asarray(mc.draws, float), "log_posterior": np.asarray(mc.log_posterior, float),
            "names": np.array(list(mc.names)), "x0": np.asarray(mc.x0, float), "drop": np.array(float(mc.drop))}
    for key in ("accepted", "jscale", "jscale_vector", "proposal_chol"):
        v = getattr(mc, key, None)
        if v is not None:
            data[key] = np.asarray(v)
    np.savez_compressed(path, **data)


def load_chains(path):
    """Read a file written by :func:`save_chains`; returns an ``MCMCResult``."""
    from .estimation import MCMCResult
    from pathlib import Path
    p = Path(path)
    if not p.is_file() and Path(str(path) + ".npz").is_file():
        p = Path(str(path) + ".npz")
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "draws" not in d or "log_posterior" not in d:
        raise ValueError(f"{p} is not a chain file written by pydyn.posterior.save_chains")
    acc = d.get("accepted")
    rate = acc.mean(axis=1) if acc is not None else np.array(
        [np.mean(np.any(np.diff(c, axis=0) != 0, axis=1)) for c in d["draws"]])
    return MCMCResult(draws=d["draws"], log_posterior=d["log_posterior"], acceptance_rate=rate,
                      names=[str(s) for s in d["names"]], x0=d["x0"], drop=float(d["drop"]), accepted=acc,
                      jscale=float(d["jscale"]) if "jscale" in d else None, jscale_vector=d.get("jscale_vector"),
                      proposal_chol=d.get("proposal_chol"))


# ====================================================================== posterior impulse responses
def posterior_subsample(mc, size, drop=None, rng=None):
    """``size`` parameter vectors drawn with replacement from the kept draws of all chains
    (``get_posterior_subsample.m``: a chain and a draw are picked uniformly, then the sample is
    ordered by chain). Returns ``(draws, log_posterior or None)``."""
    d, lp, d0 = _unpack(mc)
    drop = d0 if drop is None else drop
    rng = np.random.default_rng(0) if rng is None else rng
    first = first_kept(d.shape[1], drop)
    nb, nkept = d.shape[0], d.shape[1] - first
    u = rng.random((int(size), 2))
    chain = np.ceil(u[:, 0] * nb).astype(int).clip(1, nb) - 1
    draw = np.ceil(u[:, 1] * nkept).astype(int).clip(1, nkept) - 1
    order = np.argsort(chain, kind="stable")
    chain, draw = chain[order], draw[order]
    return d[chain, first + draw, :], (None if lp is None else lp[chain, first + draw])


def number_of_subdraws(mc, est=None, sub_draws=None, drop=None) -> int:
    """``sub_draws`` if given (it may not exceed the number of kept draws), else
    ``min(posterior_max_subsample_draws, kept draws of all chains)`` (``set_number_of_subdraws.m``)."""
    k = kept_draws(mc, drop)
    total = k.shape[0] * k.shape[1]
    o = est.options if est is not None else {}
    sub_draws = o.get("sub_draws") if sub_draws is None else sub_draws
    if sub_draws is None:
        return int(min(o.get("posterior_max_subsample_draws", 1200), total))
    if sub_draws > total * k.shape[0]:        # Dynare compares with draws x chains x chains
        raise ValueError(f"The value of option sub_draws ({sub_draws}) is greater than the number of available "
                         f"draws in the MCMC ({total})!")
    return int(sub_draws)


def lower_cholesky_covariance(Sigma_e, add_tiny_number=1e-14):
    """Lower Cholesky factor of the shock covariance matrix with zero rows and columns for shocks
    of zero variance (``get_lower_cholesky_covariance.m``)."""
    S = np.asarray(Sigma_e, float)
    nz = np.flatnonzero(np.sqrt(np.diag(S)) != 0)
    out = np.zeros_like(S)
    if nz.size:
        sub = S[np.ix_(nz, nz)]
        try:
            out[np.ix_(nz, nz)] = np.linalg.cholesky(sub)
        except np.linalg.LinAlgError:
            out[np.ix_(nz, nz)] = np.linalg.cholesky(sub + add_tiny_number * np.eye(nz.size))
    return out


def irf_shocks(est, irf_shocks=None) -> list:
    """Indices of the shocks for which IRFs are computed (``getIrfShocksIndx.m``): all, or those
    named in ``irf_shocks`` (default: the ``.mod`` option), in the order of declaration."""
    names = est.options.get("irf_shocks") if irf_shocks is None else irf_shocks
    exo = list(est.model.exo_names)
    if not names:
        return list(range(len(exo)))
    if isinstance(names, str):
        names = [names]
    unknown = [s for s in names if s not in exo]
    if unknown:
        raise ValueError(f"irf_shocks: unknown shock(s) {unknown}")
    return sorted({exo.index(s) for s in names})


def irf_at(est, theta, periods=40, relative_irf=False, shocks=None, variables=None) -> np.ndarray:
    """First-order impulse responses at ``theta`` with Dynare's conventions (``PosteriorIRF_core1.m``):
    array (periods, variables, shocks in declaration order).

    The impulse to shock i is column i of the lower Cholesky factor of the shock covariance matrix
    (one standard deviation, with the contemporaneous effect on shocks ordered after it when shocks
    are correlated). With ``relative_irf`` the responses are multiplied by 100 and divided by that
    standard deviation. Shocks with a standard deviation of at most 5e-7 and responses that vary
    by less than 1e-12 over the horizon are left at zero, as in Dynare. Under ``loglinear`` the
    responses are those of the logs. ``shocks``: indices to compute (others stay zero)."""
    m = est.model
    theta = np.asarray(theta, float)
    Q, _ = est.covariances(theta)
    ys, G, H, _ = est.solve(theta)
    SS = lower_cholesky_covariance(Q)
    idx = list(range(m.n_endo)) if variables is None else [m.endo_names.index(v) for v in variables]
    shocks = range(len(m.exo_names)) if shocks is None else shocks
    out = np.zeros((int(periods), len(idx), len(m.exo_names)))
    for i in shocks:
        if SS[i, i] > 5e-7:
            y = np.empty((int(periods), m.n_endo))
            cur = H @ SS[:, i]
            for t in range(int(periods)):
                y[t] = cur
                cur = G @ cur
            if relative_irf:
                y = 100 * y / SS[i, i]
            y = y[:, idx]
            keep = (y.max(axis=0) - y.min(axis=0)) > 1e-12
            out[:, keep, i] = y[:, keep]
    return out


@dataclass
class PosteriorIRF:
    """Posterior distribution of impulse responses (Dynare's ``oo_.PosteriorIRF.dsge``).
    ``names`` are ``variable_shock``; the arrays have one column per name."""
    names: list
    mean: np.ndarray             # (periods, series)
    median: np.ndarray
    var: np.ndarray
    hpd_inf: np.ndarray
    hpd_sup: np.ndarray
    deciles: np.ndarray          # (periods, 9, series)
    mh_conf_sig: float
    variables: list
    shocks: list
    params: np.ndarray = field(default=None, repr=False)      # parameter draws used
    draws: np.ndarray = field(default=None, repr=False)       # (draws, periods, variables, shocks in declaration order)

    def __getitem__(self, name) -> dict:
        k = self.names.index(name)
        return {"Mean": self.mean[:, k], "Median": self.median[:, k], "Var": self.var[:, k],
                "HPDinf": self.hpd_inf[:, k], "HPDsup": self.hpd_sup[:, k], "deciles": self.deciles[:, :, k]}


def moments_along_draws(x, mh_conf_sig=0.9) -> dict:
    """:func:`posterior_moments` (without the density) for every series of ``x`` (draws on axis 0)."""
    xx = np.sort(np.asarray(x, float), axis=0)
    n = xx.shape[0]
    out = {"mean": xx.mean(axis=0), "median": np.median(xx, axis=0),
           "variance": xx.var(axis=0, ddof=1) if n > 1 else np.full(xx.shape[1:], np.nan)}
    h = _iround((1 - mh_conf_sig) * n)
    if h > 2:
        kk = xx[n - h - 1:n - 1] - xx[:h]
        idx = np.argmin(kk, axis=0)
        lo = np.take_along_axis(xx, idx[None], axis=0)[0]
        out["hpd_inf"], out["hpd_sup"] = lo, lo + np.take_along_axis(kk, idx[None], axis=0)[0]
    else:
        out["hpd_inf"] = out["hpd_sup"] = np.full(xx.shape[1:], np.nan)
    if n > 9:
        out["deciles"] = xx[np.array([_iround(q * n) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]) - 1]
    else:
        out["deciles"] = np.full((9,) + xx.shape[1:], np.nan)
    return out


def posterior_irf(est, mc=None, periods=None, relative_irf=None, irf_shocks=None, variables=None, sub_draws=None,
                  mh_conf_sig=None, drop=None, params=None, rng=None, keep_draws=False) -> PosteriorIRF:
    """Posterior mean, median, variance, deciles and HPD bands of first-order impulse responses
    (``bayesian_irf``; ``PosteriorIRF.m``).

    The model is solved at ``sub_draws`` parameter vectors drawn from the kept draws of ``mc``
    (:func:`posterior_subsample`; ``params`` supplies the vectors directly, one per row) and
    :func:`irf_at` is evaluated at each; the moments over draws are Dynare's
    (:func:`posterior_moments`). Defaults come from the ``estimation`` command: ``irf`` (40),
    ``relative_irf``, ``irf_shocks``, ``mh_conf_sig``, ``sub_draws``, and for ``variables`` the list
    after the command, else the observed variables. A draw at which the model cannot be solved
    raises an error, as in Dynare."""
    o = est.options
    periods = int(o["irf"] or 40) if periods is None else int(periods)
    relative_irf = bool(o["relative_irf"]) if relative_irf is None else bool(relative_irf)
    mh_conf_sig = float(o["mh_conf_sig"]) if mh_conf_sig is None else float(mh_conf_sig)
    variables = list(variables or est.var_list or est.varobs)
    unknown = [v for v in variables if v not in est.model.endo_names]
    if unknown:
        raise ValueError(f"posterior_irf: {unknown} are not declared endogenous variables")
    shocks = globals()["irf_shocks"](est, irf_shocks)
    if params is None:
        if mc is None:
            raise ValueError("posterior_irf needs the chains (mc) or parameter draws (params)")
        size = number_of_subdraws(mc, est, sub_draws, drop)
        params, _ = posterior_subsample(mc, size, drop, rng)
    params = np.atleast_2d(np.asarray(params, float))
    if _iround((1 - mh_conf_sig) * params.shape[0]) < 2:
        import warnings
        warnings.warn("posterior_irf: mh_conf_sig times the number of draws is too small to generate HPD intervals")
    stock = np.empty((params.shape[0], periods, len(variables), len(est.model.exo_names)))
    for k, theta in enumerate(params):
        try:
            stock[k] = irf_at(est, theta, periods, relative_irf, shocks, variables)
        except Exception as exc:
            raise RuntimeError(f"PosteriorIRF: unable to solve the model at draw {k} ({exc})") from exc
    sel = stock[:, :, :, shocks]
    mom = moments_along_draws(sel, mh_conf_sig)
    exo = est.model.exo_names
    names = [f"{v}_{exo[i]}" for i in shocks for v in variables]

    def flat(a):                                  # (periods, variables, shocks) -> (periods, series), shock-major
        return a.transpose(0, 2, 1).reshape(a.shape[0], -1)
    dec = mom["deciles"].transpose(1, 0, 3, 2).reshape(periods, 9, -1)
    return PosteriorIRF(names=names, mean=flat(mom["mean"]), median=flat(mom["median"]), var=flat(mom["variance"]),
                        hpd_inf=flat(mom["hpd_inf"]), hpd_sup=flat(mom["hpd_sup"]), deciles=dec, mh_conf_sig=mh_conf_sig,
                        variables=variables, shocks=[exo[i] for i in shocks], params=params,
                        draws=stock if keep_draws else None)


# ====================================================================== mode check and prior densities
def _octave_range(base, inc, limit):
    """``base:inc:limit`` as Octave builds it (elements ``base + i*inc``, the last one clipped to the
    limit; empty for a zero or wrongly signed increment)."""
    if inc == 0 or not np.isfinite(inc) or (limit - base) / inc < 0:
        return np.zeros(0)
    ct = 3.0 * np.finfo(float).eps
    tmp = (limit - base) / inc
    n = int(np.floor(tmp + max(abs(tmp), 1.0) * ct)) + 1 if np.isfinite(tmp) else 0
    n = max(n, 0)
    out = base + np.arange(n) * inc
    if n:
        out[-1] = min(out[-1], limit) if inc > 0 else max(out[-1], limit)
    return out


def mode_check_grid(x, lb, ub, p2, neighbourhood_size=0.5, symmetric_plots=True, number_of_points=20, huge_number=1e7,
                    as_dynare=True):
    """Grid of ``mode_check.m`` for one parameter with mode ``x``, bounds ``lb``/``ub`` and prior
    standard deviation ``p2``. Returns ``(z, binding_lower, binding_upper)``.

    With finite bounds and a non-zero mode the grid covers ``x (1 -/+ neighbourhood_size)`` cut at
    the bounds, made symmetric around the mode if asked and possible. Otherwise Dynare 7.2 builds
    the interval from the *opposite* bounds, around zero instead of the mode: for a parameter with
    infinite bounds and ``p2 < 10`` the grid runs from ``-p2`` up to the mode and from the mode
    *down* to ``-1e-6 * huge_number`` (= -10). That is reproduced when ``as_dynare`` is true;
    ``as_dynare=False`` uses ``x -/+ p2`` cut at the bounds instead."""
    ll = float(neighbourhood_size)
    tol_bounds = 1e-8
    if np.isinf(ll):
        symmetric_plots = False
    m1 = False
    if x != 0 and not np.isinf(lb) and not np.isinf(ub):
        l1 = max(lb, (1 - np.sign(x) * ll) * x)
        l2 = min(ub, (1 + np.sign(x) * ll) * x)
    elif as_dynare:
        upper_bound = lb if not np.isinf(lb) else -1e-6 * huge_number
        lower_bound = ub if not np.isinf(ub) else -1e-6 * huge_number
        l1 = max(lower_bound, -p2)
        l2 = min(upper_bound, p2)
    else:
        l1, l2 = max(lb, x - p2), min(ub, x + p2)
    binding_lower = abs(x - lb) < tol_bounds
    binding_upper = (not binding_lower) and abs(x - ub) < tol_bounds
    if symmetric_plots and not binding_lower and not binding_upper:
        if l2 < (1 + ll) * x:
            l1 = x - (l2 - x)
            m1 = True
        if not m1 and l1 > (1 - ll) * x and x + (x - l1) < ub:
            l2 = x + (x - l1)
    half = number_of_points / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = _octave_range(l1, (x - l1) / half, x)
        z2 = _octave_range(x, (l2 - x) / half, l2)
    return np.unique(np.concatenate([z1, z2])), bool(binding_lower), bool(binding_upper)


def mode_check(est, mode, neighbourhood_size=None, symmetric_plots=None, number_of_points=None, as_dynare=True) -> dict:
    """Data of Dynare's mode-check plots (``mode_check.m``): for every parameter the log posterior
    and the log-likelihood kernel on a grid around the mode, the other parameters held at the mode.

    Returns ``{name: array (points, 3)}`` with columns: parameter value, log posterior kernel (NaN
    where it is not defined, including outside the bounds), and log posterior minus log prior plus
    the log prior at the mode ("log-lik kernel": the likelihood shifted to meet the posterior at
    the mode). ``mode`` is a vector or a ``ModeResult``. Options default to those of the
    ``estimation`` command (``mode_check_neighbourhood_size`` 0.5, ``mode_check_symmetric_plots``,
    ``mode_check_number_of_points`` 20). See :func:`mode_check_grid` for the grid and for a defect of
    Dynare's grid when a parameter has an infinite bound."""
    o = est.options
    x = np.asarray(getattr(mode, "x", mode), float)
    ll = o["mode_check_neighbourhood_size"] if neighbourhood_size is None else neighbourhood_size
    sym = bool(o["mode_check_symmetric_plots"]) if symmetric_plots is None else bool(symmetric_plots)
    npts = o["mode_check_number_of_points"] if number_of_points is None else number_of_points
    dy = est.log_prior(x)
    out = {}
    for k, name in enumerate(est.names):
        z, _, _ = mode_check_grid(x[k], est.lb[k], est.ub[k], est.prior.p2[k], ll, sym, npts, o["huge_number"], as_dynare)
        y = np.full((z.size, 3), np.nan)
        y[:, 0] = z
        for i, zi in enumerate(z):
            xx = x.copy()
            xx[k] = zi
            lp = est.log_posterior(xx)
            if np.isfinite(lp):
                y[i, 1] = lp
                with np.errstate(all="ignore"):
                    y[i, 2] = lp - est.log_prior(xx) + dy
        out[name] = y
    return out


def prior_density(est_or_prior, index=None, steps=200):
    """Prior density curves as drawn by Dynare (``draw_prior_density.m``, ``plot_priors``).

    For parameter ``index`` returns ``(x, f)``: ``steps`` points between the 0.1% and 99.9% quantiles
    of the prior (1% and 99% for the inverse gamma families, the support for the uniform) and the
    density there; where the density peaks at an end of the grid values above 10 are replaced by
    NaN. Without ``index``: ``(X, F)`` with one row per parameter."""
    from scipy import special
    from . import priors as _pr
    pr = getattr(est_or_prior, "prior", est_or_prior)
    if index is None:
        res = [prior_density(pr, i, steps) for i in range(len(pr.shape))]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])
    i = int(index)
    sh, p3, p4, p6, p7 = int(pr.shape[i]), pr.p3[i], pr.p4[i], pr.p6[i], pr.p7[i]
    trunc = 1e-3
    with np.errstate(all="ignore"):
        if sh == 1:
            lo, hi = (stats.beta.ppf(q, p6, p7) * (p4 - p3) + p3 for q in (trunc, 1 - trunc))
            x = np.linspace(lo, hi, steps)
            dens = stats.beta.pdf((x - p3) / (p4 - p3), p6, p7) / (p4 - p3)
        elif sh == 2:
            lo, hi = (stats.gamma.ppf(q, p6, scale=p7) + p3 for q in (trunc, 1 - trunc))
            x = np.linspace(lo, hi, steps)
            dens = stats.gamma.pdf(x - p3, p6, scale=p7)
        elif sh == 3:
            lo = max(p3, stats.norm.ppf(trunc, p6, p7))
            hi = min(p4, stats.norm.ppf(1 - trunc, p6, p7))
            x = np.linspace(lo, hi, steps)
            dens = stats.norm.pdf(x, p6, p7)
        elif sh in (4, 6):
            g_hi = special.gammaincinv(p7 / 2, 1 - 10 * trunc) * (2 / p6)
            g_lo = special.gammaincinv(p7 / 2, 10 * trunc) * (2 / p6)
            if sh == 4:
                lo, hi = 1 / np.sqrt(g_hi) + p3, 1 / np.sqrt(g_lo) + p3
                x = np.linspace(lo, hi, steps)
                dens = np.exp([_pr.lpdf_ig1(v, p6, p7) for v in x - p3])
            else:
                lo, hi = 1 / g_hi + p3, 1 / g_lo + p3
                x = np.linspace(lo, hi, steps)
                dens = np.exp([_pr.lpdf_ig2(v, p6, p7) for v in x - p3])
        elif sh == 5:
            x = np.linspace(p6, p7, steps)
            dens = np.ones(steps) / (p7 - p6)
        elif sh == 8:
            lo, hi = (p3 + stats.weibull_min.ppf(q, p6, scale=p7) for q in (trunc, 1 - trunc))
            x = np.linspace(lo, hi, steps)
            dens = np.exp([_pr.lpdf_weibull(v, p6, p7, p3) for v in x])
        else:
            raise ValueError(f"prior_density: unknown distribution shape (index {i}, type {sh})")
        dens = np.array(dens, float)
        if sh != 5:
            k1 = int(np.nanargmax(dens)) if np.any(np.isfinite(dens)) else 0
            if k1 == 0 or k1 == dens.size - 1:
                dens[dens > 10] = np.nan
    return x, dens

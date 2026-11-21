# This file is part of pydyn.
#
# The figures follow those of Dynare 7.2 (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org;
# GPL-3.0-or-later): matlab/estimation/{mode_check, plot_priors, PlotPosteriorDistributions, trace_plot}.m and
# matlab/convergence_diagnostics/mcmc_diagnostics.m. Modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Figures for estimation results, drawn with matplotlib (imported only when a plot is made).

Every ``plot_*`` function has a ``*_data`` counterpart (here or in :mod:`pydyn.posterior`) that
returns the arrays it draws; those arrays are what is validated against Dynare. All functions
return a list of figures (at most nine panels per figure, laid out as Dynare's ``pltorg``); use
``pydyn.plotting.save`` to write them. ``style`` is ``"pydyn"`` or ``"dynare"`` as in
:mod:`pydyn.plotting`.

    plot_mode_check(est, mode)             log posterior and log-likelihood kernel around the mode
    plot_priors(est)                       prior densities
    plot_posteriors(est, mc, mode=None)    prior density, posterior kernel density, posterior mode
    plot_trace(mc)                         draws of every chain
    plot_brooks_gelman(mc, est)            univariate and multivariate convergence diagnostics
    plot_posterior_series(sd)              posterior mean and deciles of smoothed / filtered series (pm3.m)
"""
from __future__ import annotations

import numpy as np

from . import posterior
from .plotting import DECOMP_COLORS, STYLES, layout, save  # noqa: F401  (save re-exported for convenience)

__all__ = ["plot_mode_check", "plot_priors", "plot_posteriors", "posterior_density_data", "plot_trace", "trace_data",
           "plot_brooks_gelman", "brooks_gelman_data", "posterior_series_data", "plot_posterior_series", "save"]


def _grid(panels, style, title, legend=None, per_figure=9, shape=None):
    """``panels``: list of (panel title, [(x, y, role, label)], [vertical line positions]) where role is
    'primary', 'secondary', 'muted' or an integer (categorical colour in fixed order)."""
    import matplotlib.pyplot as plt

    st = STYLES[style]
    figs = []
    for start in range(0, len(panels), per_figure):
        chunk = panels[start:start + per_figure]
        nr, nc = shape or layout(len(chunk))
        fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc + 0.4, 2.5 * nr + 0.6), squeeze=False, facecolor=st["surface"])
        handles = {}
        for ax, (name, lines, vlines) in zip(axes.ravel(), chunk):
            ax.set_facecolor(st["surface"])
            for x, y, role, label in lines:
                if isinstance(role, dict):
                    kw = dict(role)
                elif isinstance(role, int):
                    kw = dict(color=DECOMP_COLORS[role], ls="-", lw=1.0)
                elif role == "muted":
                    kw = dict(color="#a3a29d" if style == "pydyn" else (0.7, 0.7, 0.7), ls="-", lw=2.0)
                else:
                    kw = dict(st[role])
                h, = ax.plot(x, y, label=label, zorder=3, **kw)
                handles.setdefault(label, h)
            for v in vlines:
                ax.axvline(v, color=st["muted"] if style == "pydyn" else "g", ls="--", lw=1.0, zorder=2)
            ax.set_title(name, fontsize=10, color=st["ink"])
            ax.tick_params(labelsize=8, colors=st["muted"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(style == "dynare")
            for side in ("left", "bottom"):
                ax.spines[side].set_color(st["muted"])
            if st["grid"]:
                ax.grid(True, color="#e4e3df", lw=0.6, zorder=0)
            ax.margins(x=0)
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        labels = [k for k in handles if k]
        if len(labels) > 1 and legend is not False:
            fig.legend([handles[k] for k in labels], labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
                       fontsize=8, labelcolor=st["ink"])
        if title:
            part = f", figure {start // per_figure + 1}" if len(panels) > per_figure else ""
            fig.suptitle(title + part, fontsize=11, color=st["ink"])
        fig.tight_layout(rect=(0, 0.05 if len(labels) > 1 else 0, 1, 1))
        figs.append(fig)
    return figs


def plot_mode_check(est, mode, style="pydyn", title="Mode check plots", **options):
    """Dynare's mode-check figure from :func:`pydyn.posterior.mode_check` (``options`` are passed on):
    log posterior and log-likelihood kernel around the mode of every parameter, with a vertical line
    at the mode. Points where the posterior is not defined are left out (Dynare marks them in red)."""
    data = posterior.mode_check(est, mode, **options)
    x = np.asarray(getattr(mode, "x", mode), float)
    panels = [(name, [(d[:, 0], d[:, 1], "primary", "log-post"), (d[:, 0], d[:, 2], "secondary", "log-lik kernel")], [x[k]])
              for k, (name, d) in enumerate(data.items())]
    return _grid(panels, style, title)


def plot_priors(est, style="pydyn", title="Priors"):
    """Prior density of every estimated parameter (:func:`pydyn.posterior.prior_density`)."""
    X, F = posterior.prior_density(est)
    return _grid([(name, [(X[k], F[k], "muted", "prior")], []) for k, name in enumerate(est.names)], style, title)


def posterior_density_data(est, mc, mode=None, drop=None) -> dict:
    """What Dynare's prior-posterior figure shows for every parameter: ``{name: {"prior": (x, f),
    "posterior": (x, f), "mode": value or None}}``; the posterior density is Dynare's kernel
    estimate (:func:`pydyn.posterior.posterior_moments`)."""
    X, F = posterior.prior_density(est)
    xm = None if mode is None else np.asarray(getattr(mode, "x", mode), float)
    k_all = posterior.pooled_draws(mc, drop)
    out = {}
    for k, name in enumerate(est.names):
        dens = posterior.posterior_moments(k_all[:, k], density=True)["density"]
        out[name] = {"prior": (X[k], F[k]), "posterior": (dens[:, 0], dens[:, 1]), "mode": None if xm is None else xm[k]}
    return out


def plot_posteriors(est, mc, mode=None, style="pydyn", title="Priors and posteriors", drop=None):
    """Prior density, posterior kernel density and (if ``mode`` is given) the posterior mode."""
    data = posterior_density_data(est, mc, mode, drop)
    panels = [(name, [(d["prior"][0], d["prior"][1], "muted", "prior"),
                      (d["posterior"][0], d["posterior"][1], "primary", "posterior")],
               [] if d["mode"] is None else [d["mode"]]) for name, d in data.items()]
    return _grid(panels, style, title)


def trace_data(mc, names=None) -> dict:
    """``{name: array (chains, draws)}`` of all draws, burn-in included."""
    d, _, _ = posterior._unpack(mc)
    names = list(names or getattr(mc, "names", None) or [f"p{j}" for j in range(d.shape[2])])
    return {name: d[:, :, j] for j, name in enumerate(names)}


def plot_trace(mc, names=None, style="pydyn", title="Trace plots"):
    """Draws of every chain against the iteration number, one panel per parameter (at most eight
    chains: the palette is not cycled)."""
    data = trace_data(mc, names)
    nb = next(iter(data.values())).shape[0]
    if nb > len(DECOMP_COLORS):
        raise ValueError(f"plot_trace draws at most {len(DECOMP_COLORS)} chains; select chains first")
    panels = []
    for name, x in data.items():
        t = np.arange(1, x.shape[1] + 1)
        panels.append((name, [(t, x[b], b, f"chain {b + 1}") for b in range(nb)], []))
    return _grid(panels, style, title)


def brooks_gelman_data(mc, est=None, drop=None, log_posterior=None) -> dict:
    """Series of Dynare's convergence figures: ``{"x": points, "<name> (Interval)": (pooled, within),
    "<name> (m2)": ..., "<name> (m3)": ..., "Interval": ..., "m2": ..., "m3": ...}``; the last three
    are the multivariate diagnostic."""
    bg = posterior.brooks_gelman(mc, log_posterior, drop)
    out = {"x": bg["x"]}
    if bg["univariate"] is None:
        return out
    names = list(getattr(mc, "names", None) or (est.names if est is not None else
                                                [f"p{j}" for j in range(bg["univariate"].shape[2])]))
    for j, name in enumerate(names):
        for c, crit in enumerate(("Interval", "m2", "m3")):
            out[f"{name} ({crit})"] = (bg["univariate"][:, 2 * c, j], bg["univariate"][:, 2 * c + 1, j])
    if bg["multivariate"] is not None:
        for c, crit in enumerate(("Interval", "m2", "m3")):
            out[crit] = (bg["multivariate"][:, 2 * c], bg["multivariate"][:, 2 * c + 1])
    return out


def plot_brooks_gelman(mc, est=None, style="pydyn", plotrows=None, drop=None):
    """Brooks and Gelman (1998) diagnostics as Dynare draws them: for every parameter three panels
    (80% interval, second and third moments), pooled draws against the mean within chains;
    ``plotrows`` parameters per figure (``brooks_gelman_plotrows``, default 3); the last figure is the
    multivariate diagnostic. Returns [] when Dynare would not compute them (one chain or at most 2000
    draws)."""
    data = brooks_gelman_data(mc, est, drop)
    x = data.pop("x")
    if not data:
        return []
    rows = int(plotrows or (est.options["brooks_gelman_plotrows"] if est is not None else 3))
    uni = [(k, [(x, v[0], "primary", "pooled"), (x, v[1], "secondary", "within")], []) for k, v in data.items() if "(" in k]
    multi = [(k, [(x, v[0], "primary", "pooled"), (x, v[1], "secondary", "within")], []) for k, v in data.items() if "(" not in k]
    figs = _grid(uni, style, "MCMC univariate convergence diagnostic (Brooks and Gelman, 1998)", per_figure=3 * rows,
                 shape=(rows, 3))
    if multi:
        figs += _grid(multi, style, "Multivariate convergence diagnostic", shape=(3, 1))
    return figs


def posterior_series_data(sd, names=None) -> dict:
    """What Dynare's ``pm3`` figures draw for each series of a :class:`pydyn.posterior_moments.SeriesDistribution`
    (smoothed variables, shocks, filtered variables, ...): ``{name: {"mean": (periods,), "deciles": (9, periods)}}``.
    Series whose mean is NaN in every period (for instance the trend of an unobserved variable) get no panel, as in
    ``pm3_core.m``."""
    out = {}
    for name in (names or sd.names):
        k = sd.names.index(name)
        if np.all(np.isnan(sd.mean[:, k])):
            continue
        out[name] = {"mean": sd.mean[:, k], "deciles": sd.deciles[:, :, k]}
    return out


def plot_posterior_series(sd, title="Posterior distribution", style="pydyn", names=None):
    """Posterior mean and the nine deciles of every series over the sample, nine panels per figure (``pm3_core.m``:
    deciles in green, mean in black in Dynare's own style)."""
    data = posterior_series_data(sd, names)
    panels = []
    for name, d in data.items():
        t = np.arange(1, len(d["mean"]) + 1)
        dec = dict(color="g", ls="-", lw=0.5) if style == "dynare" else dict(color="#8fb8e8", ls="-", lw=0.8)
        lines = [(t, d["deciles"][q], dec, "deciles" if q == 0 else "") for q in range(9)]
        lines.append((t, d["mean"], dict(color="k", ls="-", lw=1.0) if style == "dynare" else dict(color="#2a78d6", ls="-", lw=2.0),
                      "mean"))
        panels.append((name, lines, []))
    figs = _grid(panels, style, title)
    return figs

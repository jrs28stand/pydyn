# This file is part of pydyn.
#
# The figure code below transliterates the plotting parts of matlab/stochastic_solver/stoch_simul.m, matlab/pltorg.m,
# and the other routines named at each function, of Dynare 7.2 (Copyright (C) 2003-2026 Dynare Team,
# https://www.dynare.org; GPL-3.0-or-later), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare's figures for :func:`pydyn.dynare` (``nograph=False``): the figures, panels, series, titles and file names of
Dynare 7.2's commands, drawn through :class:`pydyn.gfx.Gfx` (which records every call; matplotlib is imported only when a
figure file is written).  Each function names the Dynare routine it transliterates; the calls it makes are compared
with the calls Dynare makes under Octave (``reference/fig``, ``tests/test_figures.py``)."""
from __future__ import annotations

import numpy as np


def _R_():
    from . import results
    return results


def pltorg(number):
    """``pltorg.m``: (nbplt, nr, nc, lr, lc, nstar), the number of figures and the grid of the full and of the last one."""
    nrstar, ncstar = 3, 3
    nstar = nrstar * ncstar
    nbplt = nr = nc = lr = lc = 0
    small = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2), 7: (3, 3), 8: (3, 3), 9: (3, 3)}
    if number in small:
        nbplt = 1
        nr, nc = small[number]
    elif number > 9:
        nr, nc = nrstar, ncstar
        if number / nstar == round(number / nstar):
            nbplt = number // nstar
            lr, lc = nr, nc
        else:
            nbplt = int(np.ceil(number / nstar))
            reste = number - (nbplt - 1) * nstar
            lr, lc = small[reste]
    return nbplt, nr, nc, lr, lc, nstar


class FiguresMixin:
    """Methods of :class:`pydyn.results.Results` that draw Dynare's figures (every one returns at once under ``nograph``)."""

    # ------------------------------------------------------------------ plumbing
    def fig_on(self):
        return not self.opt_flag("nograph")

    def _graph_stem(self, suffix):
        """``[M_.dname '/graphs/' M_.fname suffix]``"""
        return f"{self.name}/graphs/{self.name}{suffix}"

    def _fig_save(self, stem, fmt=None):
        fmt = self.opt("graph_format", "eps") if fmt is None else fmt
        self.gfx.saveas(stem, [fmt] if isinstance(fmt, str) else [str(f) for f in fmt])

    # ------------------------------------------------------------------ stoch_simul: impulse responses
    def fig_irf(self, shock, rows):
        """The figures of one shock's IRFs (``stoch_simul.m``): ``rows`` is the list of (variable name, response) of the
        variables whose largest absolute response is at least options_.impulse_responses.plot_threshold, in the order of
        ``var_list``."""
        if not self.fig_on() or not rows:
            return
        g = self.gfx
        irf = int(self.opt("irf", 40))
        relative = self.opt_flag("relative_irf")
        n = len(rows)
        nbplt, nr, nc, lr, lc, nstar = pltorg(n)
        if nbplt == 0:
            return
        x = np.arange(1, irf + 1)

        def panel(k, row):
            g.subplot(*k)
            g.plot(x, row[1], "-k", linewidth=1)
            g.hold()
            g.plot([1, irf], [0, 0], "-r", linewidth=0.5)
            g.xlim([1, irf])
            g.title(row[0])
        if nbplt == 1:
            g.figure(("Relative response to orthogonalized shock to " if relative else "Orthogonalized shock to ") + shock)
            for j in range(n):
                panel((nr, nc, j + 1), rows[j])
            self._fig_save(self._graph_stem(f"_IRF_{shock}"))
            return
        for fig in range(1, nbplt):
            g.figure((f"Relative response to orthogonalized shock to {shock} figure {fig}" if relative
                      else f"Orthogonalized shock to {shock} figure {fig}"))
            for plt_ in range(1, nstar + 1):
                panel((nr, nc, plt_), rows[(fig - 1) * nstar + plt_ - 1])
            self._fig_save(self._graph_stem(f"_IRF_{shock}{fig}"))
        g.figure(f"Orthogonalized shock to {shock} figure {nbplt}.")
        for m in range(1, n - (nbplt - 1) * nstar + 1):
            panel((lr, lc, m), rows[(nbplt - 1) * nstar + m - 1])
        self._fig_save(self._graph_stem(f"_IRF_{shock}{nbplt}"))

    # ------------------------------------------------------------------ forecasts
    def fig_forecast(self, fc):
        """``+forecasts/graph.m``: 3 x 4 panels of the mean forecast and its HPD band per figure, then the same including
        measurement error for the observed variables."""
        if not self.fig_on() or not fc.Mean:
            return
        g = self.gfx
        nc, nr = 4, 3

        def block(names, mean, lo, hi, title, stem):
            m, n_fig = 1, 1
            g.figure(f"{title} (I)")
            for vn in names:
                if m > nc * nr:
                    self._fig_save(f"{self.name}/graphs/{stem}{n_fig}")
                    n_fig += 1
                    g.figure(f"{title} ({n_fig})")
                    m = 1
                g.subplot(nr, nc, m)
                g.plot(np.asarray(mean[vn], float).ravel(), None, "b-")
                g.hold()
                g.plot(np.asarray(lo[vn], float).ravel(), None, "b-")
                g.hold()
                g.plot(np.asarray(hi[vn], float).ravel(), None, "b-")
                g.title(vn)
                g.xlim([1, len(np.asarray(hi[vn]).ravel())])
                m += 1
            if m > 1:
                self._fig_save(f"{self.name}/graphs/{stem}{n_fig}")

        block(list(fc.Mean), fc.Mean, fc.HPDinf, fc.HPDsup, "Forecasts", "forcst")
        if fc.HPDinf_ME:
            block(list(fc.HPDinf_ME), fc.Mean, fc.HPDinf_ME, fc.HPDsup_ME, "Forecasts including ME", "forcst_ME")

    # ------------------------------------------------------------------ conditional forecasts
    def fig_conditional_forecast(self, cf, names, periods):
        """``conditional_forecasts.plot``: per variable the conditional mean and band (shaded), the unconditional mean and
        band (dashed)."""
        if not self.fig_on() or not names:
            return
        g = self.gfx
        fper = len(np.asarray(cf["cond"]["Mean"][names[0]], float).ravel())
        if periods is None:
            periods = fper
        else:
            periods += 1
        if periods > fper:
            periods = fper
        title = str(cf["graph"]["title"])
        fname = str(cf["graph"]["fname"])
        for v in names:
            ci1 = np.atleast_2d(np.asarray(cf["cond"]["ci"][v], float))[:, :periods]
            m1 = np.asarray(cf["cond"]["Mean"][v], float).ravel()[:periods]
            ci2 = np.atleast_2d(np.asarray(cf["uncond"]["ci"][v], float))[:, :periods]
            m2 = np.asarray(cf["uncond"]["Mean"][v], float).ravel()[:periods]
            g.figure(f"Conditional forecast ({title}): {v}.")
            H = len(m1)
            x = np.arange(1, H + 1)
            base = min(np.min(ci1[0, :]), np.min(ci2[0, :]))
            g.area(x, ci1[1, :H], BaseValue=base, FaceColor=[.9, .9, .9])
            g.hold()
            g.area(x, ci1[0, :H], BaseValue=base, FaceColor=[1, 1, 1])
            g.plot(x, m1, "-k", linewidth=3)
            g.plot(x, m2, "--k", linewidth=3)
            g.plot(x, ci2[0, :], "--k", linewidth=1)
            g.plot(x, ci2[1, :], "--k", linewidth=1)
            self._fig_save(f"{fname}/graphs/Conditional_forecast_{title.rstrip().replace(' ', '_')}_{v}")

    # ------------------------------------------------------------------ shock decompositions
    def _decomp_initial_date(self, res):
        """options_.initial_date as the smoother of ``evaluate_smoother`` returns it: the first date of the sample (dates
        '1Y' when the data carry none)."""
        from .dates import Date, data_start_date
        d0 = None
        opts = getattr(res, "options", None) or {}
        if isinstance(opts.get("datafile"), str):
            d0 = data_start_date(opts["datafile"], self.model.json_dir.parents[2])
        d0 = Date(1, 1) if d0 is None else d0
        return d0 + (int(getattr(res, "first_obs", 1)) - 1)

    def fig_shock_decomposition(self, i_var, init_cond=False, forecast_type=None):
        """``graph_decomp.m`` for the plain case of ``plot_shock_decomposition`` (one stacked-bar figure per variable)."""
        from .matlab import farr, arrn
        if not self.fig_on():
            return
        R = _R_()
        refuse = {"plot_shock_decomp.use_shock_groups": "use_shock_groups", "plot_shock_decomp.type": "type",
                  "plot_shock_decomp.detail_plot": "detail_plot", "plot_shock_decomp.plot_init_date": "plot_init_date",
                  "plot_shock_decomp.plot_end_date": "plot_end_date", "plot_shock_decomp.write_xls": "write_xls",
                  "plot_shock_decomp.interactive": "interactive", "plot_shock_decomp.forecast_length": "forecast_length",
                  "plot_shock_decomp.realtime": "realtime", "plot_shock_decomp.init2shocks": "init2shocks"}
        for key, nm in refuse.items():
            v = self.opt(key, None)
            if v is not None and not (isinstance(v, np.ndarray) and v.size == 0) and v is not False and v != "" \
                    and not (isinstance(v, (int, float, np.number, np.bool_)) and float(v) == 0):
                raise R.ResultsError(f"plot_shock_decomposition: the option {nm} is not supported by pydyn.dynare() "
                                     "(its figures are not written)")
        ft = forecast_type
        if ft is None:
            v = self.opt("shock_decomp.forecast_type", None)
            if isinstance(v, str) and v:
                raise R.ResultsError("plot_shock_decomposition: forecast_type is not supported by pydyn.dynare()")
        flip = self.opt_flag("plot_shock_decomp.flip")
        diff = self.opt_flag("plot_shock_decomp.diff")
        steadystate = self.opt_flag("plot_shock_decomp.steadystate")
        fig_name = str(self.opt("plot_shock_decomp.fig_name", "") or "")
        fig_name = (" " + fig_name) if fig_name else ""
        if ft:
            fig_name += f" {ft} forecast"
        if flip:
            fig_name += " flip"
        if diff:
            fig_name += " diff"
        z = np.array(arrn(self.oo_["forecast_shock_decomposition"][ft] if ft else
                          self.oo_["initval_decomposition" if init_cond else "shock_decomposition"]), float)
        if z.ndim == 2:
            z = z[:, :, None]
        info = self.oo_["shock_decomposition_info"]
        parameter_set = str(info["forecast"][ft]["parameter_set"] if ft else
                            info["initval_decomposition"]["parameter_set"] if init_cond else info["parameter_set"])
        dr = self.oo_.get("dr")
        ss = dr["ys"] if hasattr(dr, "keys") and "ys" in dr else self.oo_["steady_state"]
        ss = np.asarray(farr(ss), float).ravel()
        if diff:
            z[:, :, 1:] = z[:, :, 1:] - z[:, :, :-1]
            z[:, :, 0] = np.nan
            ss = ss * 0
        if flip:
            z = -z
            ss = -ss
        endo_names = [str(s) for s in self.M_["endo_names"].items]
        shock_names = endo_names if init_cond else [str(s) for s in self.M_["exo_names"].items]
        comp_nbr = z.shape[1] - 1
        if init_cond and comp_nbr > 18:
            raise R.ResultsError("initial_condition_decomposition: the screening of more than 18 components (screen_shocks) "
                                 "is not supported by pydyn.dynare()")
        fig_name_long = fig_name
        stem_name = fig_name_long
        for a, b in ((" (given ", ""), ("(vintage ", ""), (" ", "_"), (".", ""), ("-", ""), (")", ""), ("(", "")):
            stem_name = stem_name.replace(a, b)
        stem_name = stem_name + "_" + parameter_set
        pset_string = {"calibration": "Calibration", "prior_mode": "Prior Mode", "prior_mean": "Prior Mean",
                       "posterior_mode": "Posterior Mode", "posterior_mean": "Posterior Mean",
                       "posterior_median": "Posterior Median", "mle_mode": "MLE Mode"}.get(parameter_set, parameter_set)
        preamble_txt = "Initial condition decomposition" if init_cond else "Historical shock decomposition"
        preamble_fig = "_initval_decomposition_" if init_cond else "_shock_decomposition_"
        gend = z.shape[2]
        from .dates import Date
        d0 = getattr(self, "_sd_initial_date", None) or Date(1, 1)
        if d0.freq not in (1, 2, 4, 12):
            raise R.ResultsError("plot_shock_decomposition: the date axis of a weekly or daily sample is not supported")
        freq = d0.freq
        initial_period = d0.year + (d0.subperiod - 1) / freq if freq != 1 else float(d0.time)
        x = initial_period - 1 / freq + np.arange(gend + 1) / freq
        labels = shock_names + ["Initial values"]
        g = self.gfx
        for j in np.asarray(i_var, int).ravel():
            z1 = z[j - 1, :, :]
            xmin, xmax = x[0], x[-1] + 1 / freq
            ymax = np.max(np.sum(np.where(z1[:comp_nbr, :] > 0, z1[:comp_nbr, :], 0), axis=0)) * 1.1
            ymin = np.min(np.sum(np.where(z1[:comp_nbr, :] < 0, z1[:comp_nbr, :], 0), axis=0)) * 1.1
            if ymax - ymin < 1e-6:
                continue
            g.figure(f"{preamble_txt}{fig_name_long}: {endo_names[j - 1]} ({pset_string}).")
            g.axes([0.1, 0.1, 0.6, 0.8])
            g.hold()
            for i in range(gend):
                yp = ym = 0.0
                for k in range(comp_nbr):
                    zz = z1[k, i]
                    xs = np.array([x[i], x[i], x[i + 1], x[i + 1]]) + 1 / freq / 2
                    if zz > 0:
                        g.fill(xs, [yp, yp + zz, yp + zz, yp], k + 1)
                        yp += zz
                    else:
                        g.fill(xs, [ym, ym + zz, ym + zz, ym], k + 1)
                        ym += zz
            g.plot(x[1:], z1[-1, :], "k-", linewidth=2)
            if steadystate:
                g.plot([xmin, xmax], [0, 0], "--", linewidth=1, color=[0.7, 0.7, 0.7])
                if ymin + ss[j - 1] < 0 and ymax + ss[j - 1] > 0:
                    g.plot([xmin, xmax], ss[j - 1] * np.array([-1.0, -1.0]), "k--", linewidth=1)
            if ft:               # graph_decomp.m: the red line where the forecast starts (options_.plot_shock_decomp.forecast_length)
                fid = gend - getattr(self, "_sd_forecast_length", 0) + 1.5
                g.plot([fid, fid], [ymin, ymax], "-r", linewidth=.5)       # ylim of the axes: the stacked bars' extent
            g.set_xlim([xmin, xmax])
            g.axes([0.75, 0.1, 0.2, 0.8])
            g.axis_off()
            y1, height = 0.0, 1 / comp_nbr
            for i in range(comp_nbr, 0, -1):
                g.fill([0, 0, 0.2, 0.2], [y1, y1 + 0.7 * height, y1 + 0.7 * height, y1], i)
                g.text(0.3, y1 + 0.3 * height, labels[i - 1])
                y1 += height
            self._fig_save(f"{self.name}/graphs/{self.name}{preamble_fig}{endo_names[j - 1]}{stem_name}",
                           self.opt("plot_shock_decomp.graph_format", "eps"))

    # ------------------------------------------------------------------ estimation
    def fig_priors(self, est, title="Priors"):
        """``plot_priors.m``: the prior densities of the estimated parameters, 3 x 3 per figure."""
        from .posterior import prior_density
        from .results_post import _dynare_parameter_name
        if not self.fig_on():
            return
        g = self.gfx
        npar = len(est.names)
        nbplt, nr, nc, lr, lc, nstar = pltorg(npar)
        for plt in range(1, nbplt + 1):
            g.figure(title)
            nstar0 = min(nstar, npar - (plt - 1) * nstar)
            for index in range(1, nstar0 + 1):
                i = (plt - 1) * nstar + index
                x, f = prior_density(est, i - 1)
                g.subplot(nr, nc, index)
                g.plot(x, f, "-k", linewidth=2)
                g.title(_dynare_parameter_name(est, est.names[i - 1]))
            self._fig_save(f"{self.name}/graphs/{self.name}_Priors{plt}")

    def fig_classical_smoother(self, res, nvn):
        """``plot_classical_smoother_results.m``: smoothed shocks, smoothed measurement errors, and historical against
        smoothed observed variables."""
        if not self.fig_on():
            return
        g = self.gfx
        m = self.model
        gend = res.nobs
        varobs = list(res.varobs)
        n_varobs = len(varobs)
        exo = list(m.exo_names)
        oo = self.oo_
        mk = ("-ro", "-ko") if gend == 1 else ("-r", "-k")

        def zero_line_panels(names, series, stem, figname, count):
            nbplt, nr, nc, lr, lc, nstar = pltorg(count)
            for plt in range(1, nbplt + 1):
                g.figure(figname)
                nstar0 = min(nstar, count - (plt - 1) * nstar)
                for i in range(1, nstar0 + 1):
                    k = (plt - 1) * nstar + i
                    g.subplot(nr, nc, i)
                    g.plot([1, gend], [0, 0], mk[0], linewidth=0.5)
                    g.hold()
                    g.plot(np.arange(1, gend + 1), series[k - 1], mk[1], linewidth=1)
                    g.hold()
                    if gend > 1:
                        g.xlim([1, gend])
                    g.title(names[k - 1])
                self._fig_save(f"{self.name}/graphs/{self.name}_{stem}{plt}")
        from .matlab import farr
        zero_line_panels(exo, [np.asarray(farr(oo["SmoothedShocks"][v]), float).ravel() for v in exo],
                         "SmoothedShocks", "Smoothed shocks", len(exo))
        if nvn:
            tol = float(self.opt("ME_plot_tol", 1e-6))
            sme = oo["SmoothedMeasurementErrors"]
            idx = [v for v in varobs if np.max(np.abs(np.asarray(farr(sme[v]), float))) > tol]
            # the reference is made after the draws of the data; the panels use the same order as options_.varobs
            zero_line_panels(idx, [np.asarray(farr(sme[v]), float).ravel() for v in idx],
                             "SmoothedObservationErrors", "Smoothed observation errors", len(idx))
        nbplt, nr, nc, lr, lc, nstar = pltorg(n_varobs)
        raw = np.asarray(res.raw, float)[res.first_obs - 1:res.first_obs - 1 + gend, :]
        if res.loglinear and not res.options.get("logdata"):
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = np.log(raw)
        mk2 = ("-ro", "--ko") if gend == 1 else ("-r", "--k")
        for plt in range(1, nbplt + 1):
            g.figure("Historical and smoothed variables")
            nstar0 = min(nstar, n_varobs - (plt - 1) * nstar)
            for i in range(1, nstar0 + 1):
                k = (plt - 1) * nstar + i
                name = varobs[k - 1]
                g.subplot(nr, nc, i)
                g.plot(np.arange(1, gend + 1), np.asarray(farr(oo["SmoothedVariables"][name]), float).ravel(), mk2[0],
                       linewidth=1)
                g.hold()
                g.plot(np.arange(1, gend + 1), raw[:, k - 1], mk2[1], linewidth=1)
                g.hold()
                if gend > 1:
                    g.xlim([1, gend])
                g.title(name)
            self._fig_save(f"{self.name}/graphs/{self.name}_HistoricalAndSmoothedVariables{plt}")

    def fig_mode_check(self, items, nolik, mode):
        """The plots of ``mode_check.m``: ``items`` is a list of (name, z, y (points x 1 or 2), binding lower, binding upper,
        bound value)."""
        if not self.fig_on():
            return
        g = self.gfx
        n = len(items)
        nbplt, nr, nc, lr, lc, nstar = pltorg(n)
        for plt in range(1, nbplt + 1):
            g.figure("Mode check plots")
            for k in range(1, min(nstar, n - (plt - 1) * nstar) + 1):
                name, z, y, bl, bu, bound = items[(plt - 1) * nstar + k - 1]
                g.subplot(nr, nc, k)
                g.plot(z, y if y.shape[1] > 1 else y[:, 0])
                g.hold()
                fin = y[np.isfinite(y)]
                yl = [float(fin.min()), float(fin.max())] if fin.size else [0.0, 1.0]
                if yl[0] == yl[1]:
                    yl = [yl[0] - 0.5, yl[1] + 0.5]
                g.plot([mode[name]] * 2, yl, "c", linewidth=1)
                nan_idx = np.flatnonzero(np.isnan(y[:, 0]))
                g.plot(z[nan_idx], yl[0] * np.ones(nan_idx.size), "o", MarkerEdgeColor="r", MarkerFaceColor="r", MarkerSize=6)
                g.title(name)
                if bl or bu:
                    xl = [float(np.min(z)), float(np.max(z))]
                    g.plot([bound, bound], yl, "r--", linewidth=1)
                    g.xlim([xl[0] - 0.5 * bl * (xl[1] - xl[0]), xl[1] + 0.5 * bu * (xl[1] - xl[0])])
            if not nolik:
                g.axes([0.3, 0.93, 0.42, 0.07])
                g.plot([0.48, 0.68], [0.5, 0.5], color="r")
                g.hold()
                g.plot([0.04, 0.24], [0.5, 0.5], color="b")
                g.set_xlim([0, 1])
                g.text(0.25, 0.5, "log-post")
                g.text(0.69, 0.5, "log-lik kernel")
            self._fig_save(f"{self.name}/graphs/{self.name}_CheckPlots{plt}")

    def fig_prior_posterior(self, est):
        """``PlotPosteriorDistributions.m``: prior and posterior densities (and the posterior mode), 3 x 3 per figure; fills
        oo_.prior_density as the routine does."""
        from .posterior import prior_density
        from .results_post import _dynare_parameter_name
        from .matlab import farr
        if not self.fig_on():
            return
        R = _R_()
        g = self.gfx
        npar = len(est.names)
        prior = self._param_tree(est, [0.0] * npar, "oo_.prior_density")
        fig_no = sub = 0
        for i, name in enumerate(est.names):
            sub += 1
            if sub == 1:
                fig_no += 1
                g.figure("Priors and posteriors")
            x2, f2 = prior_density(est, i)
            nam = _dynare_parameter_name(est, name)
            self._put_param(prior, name, R.mat(np.column_stack([x2, f2])))
            top2 = np.nanmax(f2)
            tree = lambda fld: self._get_param(self.oo_[fld], name)      # noqa: E731
            dens = np.asarray(farr(tree("posterior_density")), float)
            x1, f1 = dens[:, 0], dens[:, 1]
            pmod = float(tree("posterior_mode"))
            top0 = max(np.nanmax(f1), top2)
            borneinf = min(x1[0], x2[0])
            bornesup = max(x1[-1], x2[-1])
            g.subplot(3, 3, sub)
            g.plot(x2, f2, "-k", linewidth=2, color=[0.7, 0.7, 0.7])
            g.hold()
            g.plot(x1, f1, "-k", linewidth=2)
            g.plot([pmod, pmod], [0.0, 1.1 * top0], "--g", linewidth=2)
            g.axis([borneinf, bornesup, 0, 1.1 * top0])
            g.title(nam)
            if sub == 9 or i == npar - 1:
                self._fig_save(f"{self.name}/graphs/{self.name}_PriorsAndPosteriors{fig_no}")
                sub = 0
        self.set_oo("prior_density", prior)

    def fig_posterior_irf(self, mom, shocks, variables, periods):
        """``PosteriorIRF_core2.m``: the posterior IRFs with their HPD bands, 3 x 3 per figure, in ``<dname>/Output``.
        ``mom`` is posterior.moments_along_draws' dict over (periods, variables, shocks selected)."""
        if not self.fig_on() or self.opt_flag("no_graph.posterior"):
            return
        g = self.gfx
        m = self.model
        relative = self.opt_flag("relative_irf")
        thr = float(self.opt("impulse_responses.plot_threshold", 1e-10))
        sel = {int(i): a for a, i in enumerate(shocks)}
        nvar = len(variables)
        x = np.arange(1, periods + 1)
        subplotnum = 0
        for i in range(len(m.exo_names)):
            figunumber = 0
            for j in range(nvar):
                mean = mom["mean"][:, j, sel[i]] if i in sel else np.zeros(periods)
                if np.max(np.abs(mean)) >= thr:
                    subplotnum += 1
                    if subplotnum == 1:
                        g.figure(("Relative response to orthogonalized shock to " if relative
                                  else "Orthogonalized shock to ") + m.exo_names[i])
                    g.subplot(3, 3, subplotnum)
                    lo, hi = mom["hpd_inf"][:, j, sel[i]], mom["hpd_sup"][:, j, sel[i]]
                    if not np.isnan(np.min(lo)):
                        base = float(np.min(lo))
                        g.area(x, hi, FaceColor=[.9, .9, .9], BaseValue=base)
                        g.hold()
                        g.area(x, lo, FaceColor=[1, 1, 1], BaseValue=base)
                    g.plot(x, mean, "-k", linewidth=3)
                    g.xlim([1, periods])
                    g.hold()
                    g.title(variables[j])
                if subplotnum == 9 or (j == nvar - 1 and subplotnum > 0):
                    figunumber += 1
                    self._fig_save(f"{self.name}/Output/{self.name}_Bayesian_IRF_{m.exo_names[i]}_{figunumber}")
                    subplotnum = 0

    #: pm3's figure titles (tit1) by the name of the oo_ field (name3); the others (trend coefficients, initial state) draw nothing
    _PM3_TITLES = {"SmoothedVariables": "Smoothed variables", "SmoothedShocks": "Smoothed shocks",
                   "Constant": "Smoothed constant", "Trend": "Smoothed trend", "UpdatedVariables": "Updated Variables",
                   "SmoothedMeasurementErrors": "Smoothed measurement errors",
                   "FilteredVariables": "One step ahead forecast (filtered variables)",
                   "MeanForecast": "Forecasted variables (mean)", "PointForecast": "Forecasted variables (point)",
                   "PointForecastME": "Forecasted variables (point) with ME"}

    def fig_pm3(self, name, s, names):
        """``pm3_core.m``: the posterior mean and deciles of a smoothed or forecast object, 3 x 3 per figure."""
        from .matlab import farr
        if not self.fig_on() or self.opt_flag("no_graph.posterior") or name not in self._PM3_TITLES:
            return
        g = self.gfx
        tit1 = self._PM3_TITLES[name]
        nvar = len(names)
        means = [np.asarray(farr(s["Mean"][v]), float).ravel() for v in names]
        if all(np.all(np.isnan(mu)) for mu in means):
            return
        n2 = means[0].size
        figunumber = subplotnum = 0
        g.figure(f"{tit1} {figunumber + 1}")
        for i, v in enumerate(names):
            if not np.all(np.isnan(means[i])):
                subplotnum += 1
                g.subplot(3, 3, subplotnum)
                g.hold()
                dec = np.asarray(farr(s["deciles"][v]), float).reshape(9, -1)
                for k in range(9):
                    g.plot(np.arange(1, n2 + 1), dec[k, :], "-g", linewidth=0.5)
                g.plot(np.arange(1, n2 + 1), means[i], "-k", linewidth=1)
                g.xlim([1, n2])
                g.title(v)
            if subplotnum == 9 or i == nvar - 1:
                self._fig_save(f"{self.name}/graphs/{self.name}_{name}_{v}")
                subplotnum = 0
                figunumber += 1
                if i != nvar - 1:
                    g.figure(f"{name} {figunumber + 1}")

    def fig_brooks_gelman(self, est, mc, names):
        """The convergence figures of ``mcmc_diagnostics.m`` (Brooks and Gelman, 1998; only for more than one chain):
        ``<fname>_udiag<k>`` (``brooksgelman.plotrows`` parameters per figure) and ``<fname>_mdiag``."""
        from . import posterior as _post
        if not self.fig_on():
            return
        g = self.gfx
        bg = _post.brooks_gelman(mc, drop=float(self.opt("mh_drop", 0.5)))
        if bg["univariate"] is None:
            return
        xx = np.asarray(bg["x"], float)
        uni = bg["univariate"]
        npar = uni.shape[2]
        npardisp = int(self.opt("convergence.brooksgelman.plotrows", 3))
        crits = ("Interval", "m2", "m3")

        def panels(k_list, nr, nc):
            boxplot = 1
            for k in k_list:
                for c, crit in enumerate(crits):
                    g.subplot(nr, nc, boxplot)
                    g.plot(xx, uni[:, 2 * c, k], "-b")
                    g.hold()
                    g.plot(xx, uni[:, 2 * c + 1, k], "-r")
                    g.xlim([xx[0], xx[-1]])
                    g.title(f"{names[k]} ({crit})")
                    boxplot += 1
        pages = npar // npardisp
        k = 0
        for i in range(1, pages + 1):
            g.figure("MCMC univariate convergence diagnostic (Brooks and Gelman,1998)")
            panels(range(k, k + npardisp), npardisp, 3)
            k += npardisp
            self._fig_save(f"{self.name}/graphs/{self.name}_udiag{i}")
        reste = npar - k
        if reste:
            nr, nc = (3, 1) if reste == 1 else (npardisp, 3)
            g.figure("MCMC univariate convergence diagnostic (Brooks and Gelman, 1998)")
            panels(range(k, npar), nr, nc)
            self._fig_save(f"{self.name}/graphs/{self.name}_udiag{pages + 1}")
        multi = bg["multivariate"]
        if multi is not None:
            g.figure("Multivariate convergence diagnostic")
            for c, crit in enumerate(crits):
                g.subplot(3, 1, c + 1)
                g.plot(xx, multi[:, 2 * c], "-b")
                g.hold()
                g.plot(xx, multi[:, 2 * c + 1], "-r")
                g.xlim([xx[0], xx[-1]])
                g.title(crit)
            self._fig_save(f"{self.name}/graphs/{self.name}_mdiag")

    # ------------------------------------------------------------------ OccBin
    def fig_occbin(self, var_list, steady_state):
        """``+occbin/graph.m``: the piecewise-linear and linear paths of the variables (and shocks) listed, in deviation from
        the steady state unless ``steady_state``."""
        from .matlab import farr
        if not self.fig_on():
            return
        R = _R_()
        m = self.model
        occ = self.oo_.get("occbin")
        if not hasattr(occ, "keys") or "simul" not in occ:
            raise R.ResultsError("occbin_graph: oo_.occbin.simul is empty: run occbin_solver first")
        sim = occ["simul"]
        orig = int(self.M_["orig_endo_nbr"])
        names = list(var_list) or list(m.endo_names[:orig])
        endo = [v for v in dict.fromkeys(names) if v in m._ie]
        exo = [v for v in dict.fromkeys(names) if v in m.exo_names]
        i_var = [m._ie[v] for v in endo]
        pw = np.asarray(farr(sim["piecewise"]), float)
        nper = pw.shape[0]
        data = [pw[:, i_var]]
        legend_list = ["Piecewise Linear"]
        has_lin = "linear" in sim
        if has_lin:
            data.append(np.asarray(farr(sim["linear"]), float)[:, i_var])
            legend_list.append("Linear")
        ys = np.asarray(farr(sim["ys"]), float).ravel(order="F")
        if not steady_state:
            data = [d - ys[i_var][None, :] for d in data]
        titles = list(endo)
        if exo:
            exo_pos = np.atleast_1d(np.asarray(farr(sim["exo_pos"]), float)).ravel(order="F").astype(int)
            seq = np.asarray(farr(sim["shocks_sequence"]), float)
            if seq.ndim == 1:
                seq = seq.reshape(-1, 1)
            cols = []
            for v in exo:
                hit = np.flatnonzero(exo_pos == m.exo_names.index(v) + 1)
                if hit.size == 0:
                    raise R.ResultsError(f"occbin_graph: {v} was not part of the shocks for OccBin.")
                cols.append(int(hit[0]))
            ex = np.vstack([seq[:, cols], np.zeros((nper - seq.shape[0], len(cols)))])
            data[0] = np.column_stack([data[0], ex])
            if has_lin:
                data[1] = np.column_stack([data[1], np.full((nper, len(cols)), np.nan)])
            titles += exo
        n_endo, n_exo = len(endo), len(exo)
        ndim = len(data)
        g = self.gfx
        nbplt, nr, nc, lr, lc, nstar = pltorg(n_endo + n_exo)
        for fig in range(1, nbplt + 1):
            g.figure(f"OccBin simulated paths, figure {fig}")
            for plt in range(1, nstar + 1):
                k = (fig - 1) * nstar + plt
                if fig == nbplt and lr != 0:
                    g.subplot(lr, lc, plt)
                else:
                    g.subplot(nr, nc, plt)
                g.plot([1, nper], [0, 0], "--k", linewidth=0.5)
                g.hold()
                x = np.arange(1, nper + 1)
                g.plot(x, data[0][:, k - 1], "b-", linewidth=2)
                if ndim == 2 and k <= n_endo:
                    g.plot(x, data[1][:, k - 1], "r--", linewidth=2)
                vals = np.concatenate([d[:, k - 1] for d in data])
                max_y = np.nanmax(vals) if np.any(~np.isnan(vals)) else np.nan
                min_y = np.nanmin(vals) if np.any(~np.isnan(vals)) else np.nan
                y_bottom = min_y - .01 * abs(min_y)
                y_top = max_y + 0.01 * abs(max_y)
                if y_bottom == y_top:
                    y_top = y_bottom + 1
                g.axis([1, nper, y_bottom, y_top])
                if plt == 1:
                    g.legend(legend_list, box="off")
                g.title(titles[k - 1])
                if n_endo + n_exo == k:
                    break
            self._fig_save(f"{self.name}/graphs/{self.name}_occbin_{fig}")

    # ------------------------------------------------------------------ rplot
    def fig_rplot(self, s1):
        """``rplot.m``: the simulated trajectories of oo_.endo_simul / oo_.exo_simul (``options_.rplottype``: 0 all in one
        plot, 1 one figure each, 2 subplots)."""
        from .matlab import farr
        R = _R_()
        es = self.oo_.get("endo_simul")
        if es is None or np.asarray(farr(es)).size == 0:
            raise R.ResultsError("MATLAB error in the .mod file: rplot: oo_.endo_simul is empty.")
        if not self.fig_on():
            return
        smpl = self.opt("smpl", 0)
        if not (np.ndim(smpl) == 0 and float(smpl) == 0):
            raise R.ResultsError("rplot: options_.smpl (dsample) is not supported by pydyn.dynare()")
        m = self.model
        endo_simul = np.asarray(farr(es), float)
        ml = int(self.M_["maximum_lag"])
        ix = np.arange(1 - ml, endo_simul.shape[1] - ml + 1)
        rows = []
        for v in s1:
            if v in m._ie:
                rows.append(endo_simul[m._ie[v], :])
            elif v in m.exo_names:
                rows.append(np.asarray(farr(self.oo_["exo_simul"]), float)[:, m.exo_names.index(v)])
            else:
                raise R.ResultsError("MATLAB error in the .mod file: rplot: One of the variables specified does not exist")
        y = np.vstack(rows)
        i = np.arange(max(1, ml), endo_simul.shape[1] + 1)            # 1-based columns
        x = ix[i - 1]
        g = self.gfx
        rt = int(self.opt("rplottype", 0) or 0)
        stem = lambda v: f"{self.name}/graphs/SimulatedTrajectory_{v}"      # noqa: E731
        if rt == 0:
            g.figure("Simulated Trajectory")
            g.plot(x, y[:, i - 1])
            g.title("Plot of " + " ".join(s1))
            g.xlabel("Periods")
            g.xlim([x.min(), x.max()])
            if len(s1) > 1:
                g.legend(s1)
            self._fig_save(stem(s1[0]))
        elif rt == 1:
            for j, v in enumerate(s1):
                g.figure("Simulated Trajectory")
                g.plot(x, y[j, i - 1])
                g.xlim([x.min(), x.max()])
                g.title("Plot of " + v)
                g.xlabel("Periods")
                self._fig_save(stem(v))
        elif rt == 2:
            g.figure("Simulated Trajectory")
            nl = max(1, int(np.fix(y.shape[0] / 4)))
            nc = int(np.ceil(y.shape[0] / nl))
            for j, v in enumerate(s1):
                g.subplot(nl, nc, j + 1)
                g.plot(x, y[j, i - 1])
                g.hold()
                if v in m._ie:
                    ss = np.asarray(farr(self.oo_["steady_state"]), float).ravel()[m._ie[v]]
                else:
                    ss = np.asarray(farr(self.oo_["exo_steady_state"]), float).ravel()[m.exo_names.index(v)]
                g.plot(x, ss * np.ones(i.size), "r:")
                g.xlabel("Periods")
                g.ylabel(v)
                g.title("Plot of " + v)
            self._fig_save(stem(s1[0]))

# This file is part of pydyn.
#
# The decomposition figures follow matlab/shock_decomposition/{plot_shock_decomposition, graph_decomp,
# graph_decomp_detail}.m of Dynare 7.2 (Copyright (C) Dynare Team, https://www.dynare.org), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Figures in the layout Dynare uses, drawn with matplotlib (imported only when a plot is made)."""
from __future__ import annotations

import numpy as np

#: validated two-series palette (blue, orange) on a light surface, plus neutral inks
STYLES = {
    "pydyn": {"primary": dict(color="#2a78d6", ls="-", lw=2.0), "secondary": dict(color="#eb6834", ls=(0, (5, 2)), lw=2.0),
              "zero": dict(color="#8a8a86", ls="-", lw=0.8), "grid": True, "ink": "#2b2b29", "muted": "#6b6b67",
              "surface": "#fcfcfb"},
    # the colours and line styles of Dynare's own figures
    "dynare": {"primary": dict(color="b", ls="-", lw=2.0), "secondary": dict(color="r", ls="--", lw=2.0),
               "zero": dict(color="k", ls="--", lw=0.5), "grid": False, "ink": "k", "muted": "k", "surface": "w"},
}


def layout(n: int):
    """Rows and columns per figure as in Dynare's ``pltorg`` (at most 3 x 3 panels per figure)."""
    table = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2), 7: (3, 3), 8: (3, 3), 9: (3, 3)}
    return table[min(n, 9)]


def _panels(series: dict, labels, title, style, x0=1, legend=True):
    """``series``: {panel title: [primary array, secondary array or None]}. Returns a list of figures."""
    import matplotlib.pyplot as plt

    st = STYLES[style]
    names = list(series)
    figs = []
    for start in range(0, len(names), 9):
        chunk = names[start:start + 9]
        nr, nc = layout(len(chunk))
        fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc + 0.4, 2.5 * nr + 0.6), squeeze=False,
                                 facecolor=st["surface"])
        for ax, name in zip(axes.ravel(), chunk):
            first, second = series[name]
            T = len(first)
            t = np.arange(x0, x0 + T)
            ax.set_facecolor(st["surface"])
            ax.plot([t[0], t[-1]], [0, 0], zorder=1, **st["zero"])
            h1, = ax.plot(t, first, zorder=3, label=labels[0], **st["primary"])
            handles = [h1]
            if second is not None:
                h2, = ax.plot(t, second, zorder=2, label=labels[1], **st["secondary"])
                handles.append(h2)
            both = np.concatenate([np.asarray(first, float)] + ([np.asarray(second, float)] if second is not None else []))
            lo, hi = np.nanmin(both), np.nanmax(both)
            lo, hi = lo - 0.01 * abs(lo), hi + 0.01 * abs(hi)          # Dynare's axis rule
            if lo == hi:
                hi = lo + 1
            if style == "pydyn":
                pad = 0.04 * (hi - lo)
                lo, hi = lo - pad, hi + pad
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(lo, hi)
            ax.set_title(name, fontsize=10, color=st["ink"])
            ax.tick_params(labelsize=8, colors=st["muted"])
            ax.xaxis.get_major_locator().set_params(integer=True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(style == "dynare")
            for side in ("left", "bottom"):
                ax.spines[side].set_color(st["muted"])
            if st["grid"]:
                ax.grid(True, color="#e4e3df", lw=0.6, zorder=0)
            if legend and ax is axes.ravel()[0] and len(handles) > 1:
                ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=st["ink"])
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        if title:
            part = f", figure {start // 9 + 1}" if len(names) > 9 else ""
            fig.suptitle(title + part, fontsize=11, color=st["ink"])
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_occbin(result, model, var_list=None, steady_state=False, style="pydyn", title="OccBin simulated paths"):
    """Dynare's ``occbin_graph``: piecewise-linear against linear simulated paths, one panel per variable.

    Paths are shown as deviations from the steady state unless ``steady_state`` is true. Exogenous
    variables named in ``var_list`` are drawn from the shock sequence. Returns a list of figures.
    """
    var_list = list(var_list) if var_list else model.orig_endo_names
    endo = [v for v in var_list if v in model.endo_names]
    exo = [v for v in var_list if v in model.exo_names]
    unknown = set(var_list) - set(endo) - set(exo)
    if unknown:
        raise ValueError(f"unknown variables: {sorted(unknown)}")
    series = {}
    for v in endo:                                   # Dynare draws endogenous first, then exogenous
        i = model.endo_names.index(v)
        off = 0.0 if steady_state else result.ys[i]
        series[v] = [result.piecewise[:, i] - off, result.linear[:, i] - off]
    T = result.piecewise.shape[0]
    for v in exo:
        path = np.zeros(T)
        path[: result.shocks.shape[0]] = result.shocks[:, model.exo_names.index(v)]
        series[v] = [path, None]
    return _panels(series, ("Piecewise Linear", "Linear"), title, style)


def plot_irf(responses, names=None, var_list=None, compare=None, labels=("Response", "Comparison"),
             style="pydyn", title=None):
    """Impulse responses, one panel per variable.

    ``responses`` is a (periods, n) array with column names ``names``, or a dict of name -> array.
    ``compare`` is an optional second set in the same form (e.g. the linear response next to a
    piecewise-linear or pruned one). Returns a list of figures.
    """
    def as_dict(r):
        if r is None or isinstance(r, dict):
            return r
        if names is None:
            raise ValueError("names are needed for an array of responses")
        return {n: np.asarray(r)[:, i] for i, n in enumerate(names)}

    a, b = as_dict(responses), as_dict(compare)
    keep = list(var_list) if var_list else list(a)
    series = {v: [a[v], None if b is None else b[v]] for v in keep}
    return _panels(series, labels, title, style)


#: categorical palette of the decomposition figures: a fixed order of hues, never cycled by rank
#: (adjacent pairs checked for colour-vision-deficiency separation on the light surface); components
#: beyond the eighth reuse the hues with a hatch as second encoding
DECOMP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: neutral fills: the initial conditions (last component) and the "Others" bin of screened shocks
DECOMP_NEUTRAL = {"initial": "#a3a29d", "others": "#d5d4cf", "residual": "#c4c3be"}
_DECOMP_HATCHES = ["", "////", "....", "xxxx"]
_PARAMETER_SET_NAMES = {"calibration": "Calibration", "posterior_mode": "Posterior Mode",
                        "posterior_mean": "Posterior Mean", "posterior_median": "Posterior Median",
                        "mle_mode": "MLE Mode", "prior_mode": "Prior Mode", "prior_mean": "Prior Mean",
                        "theta": "given parameter vector"}


def decomposition_colors(labels):
    """``(colour, hatch)`` for every component of a decomposition figure: ``DECOMP_COLORS`` in
    order for the shocks or groups, a neutral grey for the last component (the initial conditions)
    and a lighter one for the ``"Others"`` bin."""
    out, k = [], 0
    for i, lab in enumerate(labels):
        if i == len(labels) - 1:
            out.append((DECOMP_NEUTRAL["initial"], ""))
        elif lab == "Others":
            out.append((DECOMP_NEUTRAL["others"], ""))
        else:
            out.append((DECOMP_COLORS[k % len(DECOMP_COLORS)],
                        _DECOMP_HATCHES[(k // len(DECOMP_COLORS)) % len(_DECOMP_HATCHES)]))
            k += 1
    return out


def _decomp_axes(ax, st, style, periods, lo, hi):
    ax.set_facecolor(st["surface"])
    ax.set_xlim(periods[0] - 0.5, periods[-1] + 0.5)
    if hi > lo:
        ax.set_ylim(lo, hi)
    ax.tick_params(labelsize=8, colors=st["muted"])
    ax.xaxis.get_major_locator().set_params(integer=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(style == "dynare")
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st["muted"])
    if st["grid"]:
        ax.grid(True, axis="y", color="#e4e3df", lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def _draw_stacked(ax, panel, st, style, type_, steadystate, colors):
    """Stacked bars of one variable with the total as a line. Returns the legend handles."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from . import smoother as _smo

    contrib, total, periods = panel["contributions"], panel["total"], np.asarray(panel["periods"], float)
    bottom, top = _smo.stack_bars(contrib)
    width = (0.7 if type_ == "aoa" else 1.0) if style == "dynare" else (0.7 if type_ == "aoa" else 0.86)
    edge = dict(edgecolor="k", linewidth=0.3) if style == "dynare" else dict(edgecolor=st["surface"], linewidth=0.4)
    handles = []
    for k, (color, hatch) in enumerate(colors):
        h = np.nan_to_num(contrib[k])
        bars = ax.bar(periods, h, width, bottom=bottom[k], color=color, hatch=hatch or None, zorder=2,
                      label=panel["labels"][k], **edge)
        handles.append(bars)
    clean = np.nan_to_num(contrib)
    hi = np.max(np.sum(np.where(clean > 0, clean, 0), axis=0)) * 1.1        # Dynare's axis rule
    lo = np.min(np.sum(np.where(clean < 0, clean, 0), axis=0)) * 1.1
    ax.plot([periods[0] - 0.5, periods[-1] + 0.5], [0, 0], zorder=3, **st["zero"])
    line, = ax.plot(periods, total, color="k", lw=2.0, zorder=4, label="Smoothed variable")
    _decomp_axes(ax, st, style, periods, lo, hi)
    if steadystate:
        ss = panel["steady_state"]
        # ticks at round values of the level (deviation plus steady state), labelled with the level
        levels = MaxNLocator(nbins=7).tick_values(lo + ss, hi + ss)
        ax.set_yticks([v - ss for v in levels if lo <= v - ss <= hi])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos, ss=ss: f"{v + ss + 0.0:.4g}"))
        if lo + ss < 0 < hi + ss:                      # the level zero is inside the figure
            ax.plot([periods[0] - 0.5, periods[-1] + 0.5], [-ss, -ss], color="k", ls="--", lw=1.0, zorder=3)
    return handles, line


def _decomp_title(data, name, detail=False):
    pset = _PARAMETER_SET_NAMES.get(data.parameter_set, str(data.parameter_set))
    kind = f" {data.type}" if data.type else ""
    return f"{data.title}{kind}: {name} ({pset})" + (", detail" if detail else "")


def _legend_right(fig, handles, line, labels, st, line_label):
    # components listed top to bottom in the order they are stacked upwards, the line last
    fig.legend(handles=list(handles)[::-1] + [line], labels=list(labels)[::-1] + [line_label], loc="center left",
               bbox_to_anchor=(0.76, 0.5), frameon=False, fontsize=9, labelcolor=st["ink"], handlelength=1.4,
               borderaxespad=0.0)


def plot_shock_decomposition(decomp, model=None, var_list=None, style="pydyn", detail_plot=None, max_nrows=None,
                             panels=False, statement=-1, **options):
    """Dynare's ``plot_shock_decomposition``: stacked bars of the contributions of the shocks (or
    groups of shocks) with the initial conditions as their own component, and the smoothed
    deviation from the steady state as a black line. Returns a list of matplotlib figures, one per
    variable as in Dynare (``panels=True``: up to nine variables per figure in Dynare's panel
    layout, with one legend).

    ``decomp``: a ``smoother.ShockDecomposition`` (from ``shock_decomposition``, ``group_shocks``,
    ``initial_condition_decomposition`` or ``RealtimeDecomposition.select``), a
    ``RealtimeDecomposition`` (with ``realtime=`` and ``vintage=``), or the output of
    ``smoother.decomposition_plot_data``, which computes the numbers drawn and takes the options:
    ``use_shock_groups``, ``groups``, ``expand``, ``init2shocks`` (with ``icd``), ``type``
    (``"qoq"``, ``"yoy"``, ``"aoa"``), ``diff``, ``flip``, ``steadystate``, ``plot_init_date``,
    ``plot_end_date`` (integer periods), ``screen_shocks``, ``steady_state``.
    ``detail_plot``: one panel per component (the component against the rest), three columns and
    at most ``max_nrows`` rows per figure (default 6).

    With ``model``, the options and the variable list default to those of a
    ``plot_shock_decomposition`` statement of the ``.mod`` file (``statement``: its index among
    such statements, default the last; ``None`` to ignore the file); without such a statement the
    ``use_shock_groups`` option of ``shock_decomposition`` is used. An initial-condition
    decomposition takes them from the ``initial_condition_decomposition`` statement.

    Differences from Dynare, on purpose: with ``steadystate`` the steady state used is the one of
    the parameter set of the decomposition (Dynare uses ``oo_.dr.ys``, i.e. whatever was solved
    last; pass ``steady_state=`` to override); the remainder of an initial-condition decomposition
    is labelled "Shocks" (Dynare's figure calls it "Initial values").
    """
    import matplotlib.pyplot as plt
    from . import smoother as _smo

    st = STYLES[style]
    if isinstance(decomp, _smo.DecompositionPlotData):
        if options or var_list is not None:
            raise TypeError("options and var_list belong to decomposition_plot_data when its output is passed")
        data = decomp
    else:
        opts = {}
        if model is not None and statement is not None:
            is_icd = getattr(decomp, "kind", "") == "initial_conditions"
            kind = "initial_condition_decomposition" if is_icd else "plot_shock_decomposition"
            fo, symbols = _smo._decomp_statement_options(model, kind, which=statement)
            if not is_icd and not _smo._statements(model, kind):
                sd, _ = _smo._decomp_statement_options(model, "shock_decomposition")
                fo = {k: v for k, v in sd.items() if k == "use_shock_groups"}
            honoured = {"use_shock_groups", "steadystate", "type", "diff", "flip", "screen_shocks", "max_nrows",
                        "detail_plot", "init2shocks"}
            if isinstance(decomp, _smo.RealtimeDecomposition):
                honoured |= {"realtime", "vintage"}
            if fo.get("realtime") == 0:
                honoured |= {"realtime"}
            if fo.get("vintage") == 0:
                honoured |= {"vintage"}
            _smo._report_unused(model, kind, honoured, "plot_shock_decomposition", fo)
            opts = {k: v for k, v in fo.items() if k in honoured}
            if var_list is None and symbols:
                var_list = symbols
        opts.update(options)
        if detail_plot is None:
            detail_plot = bool(opts.pop("detail_plot", False))
        if max_nrows is None:
            max_nrows = opts.pop("max_nrows", None)
        opts.pop("detail_plot", None)
        opts.pop("max_nrows", None)
        realtime, vintage = opts.pop("realtime", None), opts.pop("vintage", 0)
        if isinstance(decomp, _smo.RealtimeDecomposition):
            decomp = decomp.select(realtime or 1, vintage or 0)
        elif realtime:
            raise TypeError("realtime= needs a RealtimeDecomposition (smoother.realtime_shock_decomposition)")
        data = _smo.decomposition_plot_data(decomp, var_list, model=model, **opts)
    max_nrows = 6 if max_nrows is None else int(max_nrows)
    line_label = "Smoothed variable" if data.steadystate else "Smoothed deviation"
    figs = []

    if detail_plot:
        for panel in data.panels:
            contrib, total, periods = panel["contributions"], panel["total"], np.asarray(panel["periods"], float)
            ncomp = contrib.shape[0]
            nrow = min(int(np.ceil(ncomp / 3)), max_nrows)
            rest = total[None, :] - contrib
            both = np.nan_to_num(np.stack([contrib, rest]))
            hi = np.max(np.sum(np.where(both > 0, both, 0), axis=0))
            lo = np.min(np.sum(np.where(both < 0, both, 0), axis=0))
            pad = 0.05 * (hi - lo)
            for start in range(0, ncomp, nrow * 3):
                chunk = range(start, min(start + nrow * 3, ncomp))
                fig, axes = plt.subplots(nrow, 3, figsize=(10.5, 2.3 * nrow + 1.3), squeeze=False,
                                         facecolor=st["surface"])
                for ax, k in zip(axes.ravel(), chunk):
                    pair = np.vstack([contrib[k], rest[k]])
                    bottom, _ = _smo.stack_bars(pair)
                    edge = dict(edgecolor="k", linewidth=0.3) if style == "dynare" else {}
                    h1 = ax.bar(periods, np.nan_to_num(pair[0]), 1.0 if style == "dynare" else 0.86, bottom=bottom[0],
                                color=DECOMP_COLORS[0], zorder=2, **edge)
                    h2 = ax.bar(periods, np.nan_to_num(pair[1]), 1.0 if style == "dynare" else 0.86, bottom=bottom[1],
                                color=DECOMP_NEUTRAL["residual"], zorder=2, **edge)
                    ax.plot([periods[0] - 0.5, periods[-1] + 0.5], [0, 0], zorder=3, **st["zero"])
                    line, = ax.plot(periods, total, color="k", lw=1.6, zorder=4)
                    _decomp_axes(ax, st, style, periods, lo - pad, hi + pad)
                    ax.set_title(panel["labels"][k], fontsize=10, color=st["ink"])
                for ax in axes.ravel()[len(chunk):]:
                    ax.set_visible(False)
                fig.suptitle(_decomp_title(data, panel["name"], detail=True), fontsize=11, color=st["ink"])
                fig.legend(handles=[h1, h2, line], labels=["Individual contrib.", "Residual contrib.", line_label],
                           loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=st["ink"])
                fig.tight_layout(rect=(0, 0.06, 1, 0.97))
                figs.append(fig)
        return figs

    if panels and data.panels:
        for start in range(0, len(data.panels), 9):
            chunk = data.panels[start:start + 9]
            nr, nc = layout(len(chunk))
            fig, axes = plt.subplots(nr, nc, figsize=(3.9 * nc + 3.2, 2.7 * nr + 0.9), squeeze=False,
                                     facecolor=st["surface"])
            for ax, panel in zip(axes.ravel(), chunk):
                handles, line = _draw_stacked(ax, panel, st, style, data.type, data.steadystate,
                                              decomposition_colors(panel["labels"]))
                ax.set_title(panel["name"], fontsize=10, color=st["ink"])
            for ax in axes.ravel()[len(chunk):]:
                ax.set_visible(False)
            pset = _PARAMETER_SET_NAMES.get(data.parameter_set, str(data.parameter_set))
            fig.suptitle(f"{data.title}{' ' + data.type if data.type else ''} ({pset})", fontsize=11, color=st["ink"])
            same = all(p["labels"] == chunk[0]["labels"] for p in chunk)
            if same:
                fig.tight_layout(rect=(0, 0, 0.75, 0.96))
                _legend_right(fig, handles, line, chunk[0]["labels"], st, line_label)
            else:                                     # screened shocks differ by variable: a legend per panel
                for ax in axes.ravel()[:len(chunk)]:
                    ax.legend(frameon=False, fontsize=6, ncol=2, labelcolor=st["ink"])
                fig.tight_layout(rect=(0, 0, 1, 0.96))
            figs.append(fig)
        return figs

    for panel in data.panels:
        fig, ax = plt.subplots(figsize=(11.0, 5.2), facecolor=st["surface"])
        handles, line = _draw_stacked(ax, panel, st, style, data.type, data.steadystate,
                                      decomposition_colors(panel["labels"]))
        ax.set_title(_decomp_title(data, panel["name"]), fontsize=11, color=st["ink"])
        ax.set_xlabel("year" if data.type == "aoa" else "period", fontsize=9, color=st["muted"])
        ax.tick_params(labelsize=9)
        fig.tight_layout(rect=(0, 0, 0.75, 1))
        _legend_right(fig, handles, line, panel["labels"], st, line_label)
        figs.append(fig)
    return figs


def plot_initial_condition_decomposition(decomp, model=None, var_list=None, style="pydyn", **options):
    """Figure of Dynare's ``initial_condition_decomposition``: the contribution of every variable's
    smoothed value in period 1, with the effect of all later shocks as the last component
    ("Shocks"). Options as ``plot_shock_decomposition``; with more than 18 variables the 16 largest
    are kept for each figure and the rest summed in "Others", as in Dynare."""
    if getattr(decomp, "kind", "initial_conditions") != "initial_conditions":
        raise TypeError("plot_initial_condition_decomposition needs smoother.initial_condition_decomposition output")
    return plot_shock_decomposition(decomp, model=model, var_list=var_list, style=style, **options)


def save(figs, basename, fmt="png", dpi=150):
    """Save figures as ``basename_1.png``, ``basename_2.png``, ... and return the paths."""
    out = []
    for i, fig in enumerate(figs, 1):
        path = f"{basename}_{i}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        out.append(path)
    return out

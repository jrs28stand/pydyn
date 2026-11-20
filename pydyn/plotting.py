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


def save(figs, basename, fmt="png", dpi=150):
    """Save figures as ``basename_1.png``, ``basename_2.png``, ... and return the paths."""
    out = []
    for i, fig in enumerate(figs, 1):
        path = f"{basename}_{i}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        out.append(path)
    return out

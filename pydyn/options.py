# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Helpers for options written in the .mod file: defaults taken from it, and warnings for options
that a pydyn function does not act on (so that nothing that would change results is dropped silently)."""
from __future__ import annotations

import warnings

#: options that only affect what Dynare prints, plots or stores
COSMETIC = {
    "nograph", "nodisplay", "graph_format", "noprint", "nocorr", "nomoments", "nofunctions", "tex",
    "nodecomposition", "contemporaneous_correlation", "TeX", "no_graph.posterior", "no_graph.shock_decomposition",
    "nodiagnostic", "console_mode", "silent_optimizer", "debug", "plot_priors",
    "no_graph.plot_shock_decomposition",
    "graph_save_formats", "irf_plot_threshold", "dirname", "verbosity", "simul.debug",
}


class UnsupportedOptionWarning(UserWarning):
    pass


#: statements whose options accumulate in Dynare's options_ in file order: a later statement
#: (of any of these kinds) only overrides the options it sets
ACCUMULATED_OPTIONS = ("steady", "check", "stoch_simul", "estimation", "calib_smoother")

_MISSING = object()


def accumulated(model, kinds) -> dict:
    """Dynare's options_ as the statements of ``kinds`` leave it: their options merged in file
    order (model.options_sequence). Options changed in ``model.options[kind]`` after the model was
    loaded take precedence; a kind whose options were emptied (``model.options[kind] = {}``) does
    not contribute."""
    seq = getattr(model, "options_sequence", None)
    if seq is None:                                       # a model built without the statement list
        out = {}
        for kind in kinds:
            out.update(model.options.get(kind, {}))
        return out
    parsed = getattr(model, "_options_parsed", {})
    out = {}
    for kind, opts in seq:
        if kind in kinds and model.options.get(kind):
            out.update(opts)
    for kind in kinds:
        cur, base = model.options.get(kind) or {}, parsed.get(kind, {})
        out.update({k: v for k, v in cur.items() if base.get(k, _MISSING) is not v})
    return out


def stoch_options(model) -> dict:
    """Options of the model's solution commands as Dynare's options_ holds them after the .mod file:
    those of check, stoch_simul, estimation and calib_smoother accumulated in file order."""
    return accumulated(model, ("check", "stoch_simul", "estimation", "calib_smoother"))


def steady_options(model) -> dict:
    """Options of the steady-state computation: those of steady and check (which takes the steady
    options) accumulated in file order, as in Dynare's options_."""
    return accumulated(model, ("steady", "check"))


def qz_criterium(model, value=None, default=1.0 + 1e-6) -> float:
    if value is not None:
        return float(value)
    return float(stoch_options(model).get("qz_criterium", default))


def warn_once(model, message: str, stacklevel: int = 3) -> None:
    seen = model.__dict__.setdefault("_warned", set())
    if message not in seen:
        seen.add(message)
        warnings.warn(message, UnsupportedOptionWarning, stacklevel=stacklevel)


def warn_unused(model, statement: str, honoured, what: str, extra=()) -> list:
    """Warn (once per model) about options of ``statement`` that ``what`` does not act on."""
    opts = dict(model.options.get(statement, {}))
    unused = sorted(k for k in opts if k not in set(honoured) | COSMETIC | set(extra))
    if unused:
        warn_once(model, f"{what} does not act on these {statement} options from the .mod file: "
                         f"{', '.join(unused)}", stacklevel=4)
    return unused


def refuse_occbin(model, what: str) -> None:
    if getattr(model, "occbin_constraints", None):
        raise NotImplementedError(
            f"{what} is linear and would ignore the model's occbin_constraints (it would use the reference "
            "regime only). Use pydyn.occbin_estimation.OccbinEstimation for estimation and "
            "pydyn.occbin_filter.occbin_smoother for smoothing. To run the linear version deliberately, "
            "set model.occbin_constraints = [] first.")

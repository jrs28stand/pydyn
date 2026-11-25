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
    v = stoch_options(model).get("qz_criterium")
    if v is None:                                         # options_.qz_criterium = ... written by MATLAB lines of the .mod file
        v = matlab_option(model, "qz_criterium", default=default)
    return float(v)


def qz_zero_threshold(model, default=None):
    """options_.qz_zero_threshold as the .mod file's statements (or MATLAB lines) left it; None: Dynare's default 1e-6."""
    v = stoch_options(model).get("qz_zero_threshold")
    if v is None:
        v = matlab_option(model, "qz_zero_threshold", default=default)
    return None if v is None else float(v)


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


# ------------------------------------------------------------------ options_ written by MATLAB code in the .mod file
#: ``options_.<path> = value;`` lines of the .mod file are stored in ``model.matlab_options`` (with the position of each
#: write in ``model.matlab_options_log``). These dotted paths are the ones pydyn acts on, with the function that does;
#: any other non-cosmetic field written by MATLAB code gets an UnsupportedOptionWarning when the file is loaded.
HONOURED_MATLAB_OPTIONS = {
    "stack_solve_algo": "perfect_foresight", "solve_algo": "perfect_foresight", "dynatol.f": "perfect_foresight",
    "dynatol.x": "perfect_foresight", "simul.maxit": "perfect_foresight", "lmmcp.TolFun": "perfect_foresight",
    "solve_tolf": "steady_state", "solve_tolx": "steady_state",
    "qz_criterium": "first_order, moments", "qz_zero_threshold": "first_order", "forecast_replic": "forecast", "instruments": "steady-state files, ramsey",
}
#: fields that only select what is printed, plotted or saved, or that the .mod file's own MATLAB code reads back
MATLAB_COSMETIC = COSMETIC | {"nograph", "nodisplay", "console_mode", "TeX", "tex", "dirname"}


def _leaves(prefix, value):
    """Dotted paths of the leaves of a (nested) struct value."""
    if isinstance(value, dict) and value:
        for k, v in value.items():
            yield from _leaves(f"{prefix}.{k}", v)
    else:
        yield prefix


def matlab_option(model, path, before=None, default=None):
    """Value of ``options_.<path>`` as the MATLAB lines of the .mod file left it, taking the writes made before the
    Dynare statement with index ``before`` in ``model.statements`` (None: all writes; the position of a write is the
    index of the last Dynare statement run before it, so a write between ``perfect_foresight_setup`` and
    ``perfect_foresight_solver`` counts for the solver and one after the solver does not)."""
    head, *rest = path.split(".")
    value, found = None, False
    for k, name, v in getattr(model, "matlab_options_log", ()):
        if name == head and (before is None or k < before):
            value, found = v, True
    if not found:
        return default
    for r in rest:
        if isinstance(value, dict) and r in value:
            value = value[r]
        else:
            return default
    return default if value is None else value


def last_statement_index(model, *names):
    """Index in ``model.statements`` of the last statement called one of ``names`` (None if there is none)."""
    idx = None
    for i, st in enumerate(model.statements):
        if st.get("statementName") in names:
            idx = i
    return idx


def warn_unhonoured_matlab_options(model) -> list:
    """Warn (once per model) about fields of options_ written by MATLAB code of the .mod file that no pydyn function acts
    on: Dynare's commands would have used them, pydyn's would not. Returns the sorted list of dotted paths."""
    final = {}
    for _, name, v in getattr(model, "matlab_options_log", ()):
        final[name] = v
    bad = set()
    for name, v in final.items():
        if name in MATLAB_COSMETIC:
            continue
        for leaf in _leaves(name, v):
            if leaf in HONOURED_MATLAB_OPTIONS or leaf in MATLAB_COSMETIC:
                continue
            if any(leaf.startswith(h + ".") for h in HONOURED_MATLAB_OPTIONS):
                continue
            bad.add(leaf)
    bad = sorted(bad)
    if bad:
        warn_once(model, "MATLAB code in the .mod file sets " + ", ".join("options_." + b for b in bad) + ": pydyn does "
                  "not act on these fields (it keeps the values for MATLAB code that reads them back), so Dynare commands "
                  "that would use them run with their defaults or their own options", stacklevel=6)
    return bad

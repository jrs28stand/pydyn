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
    "graph_save_formats", "sub_draws", "irf_plot_threshold", "dirname", "verbosity", "simul.debug",
}


class UnsupportedOptionWarning(UserWarning):
    pass


def stoch_options(model) -> dict:
    """Options of the model's solution commands, later commands taking precedence as in Dynare."""
    out = {}
    for kind in ("check", "stoch_simul", "estimation", "calib_smoother"):
        out.update(model.options.get(kind, {}))
    return out


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
            "regime only). OccBin filtering and estimation are not implemented. To run the linear version "
            "deliberately, set model.occbin_constraints = [] first.")

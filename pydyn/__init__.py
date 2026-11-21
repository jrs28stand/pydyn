# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""pydyn: a Python front end for Dynare models, built on the Dynare preprocessor's JSON output."""
from .preprocessor import run_preprocessor
from .model import Model, ModelError
from .steady import steady_state, SteadyStateError
from .simulate import simulate, irf
from .perturbation import first_order, FirstOrderSolution
from . import estimation, kalman, korder, matlab, moments, optimal_policy, posterior, posterior_plots, priors, smoother
from .korder import k_order
from .perfect_foresight import (perfect_foresight, PerfectForesightError, PerfectForesightResult,
                                unanticipated_shocks)
from .occbin import Occbin, OccbinResult, OccbinError
from . import occbin_filter
from .occbin_estimation import OccbinEstimation
from .nonlinear_estimation import NonlinearEstimation
# the other public modules, so that ``import pydyn`` is enough to reach ``pydyn.<module>``
# (``pydyn.simulate`` and ``pydyn.perfect_foresight`` are the functions of the same name)
from . import (dates, dsge_var, endogenous_prior, epilogue, external, heteroskedastic, newrat, nonlinear_filters,
               octave_rng, plotting, samplers, simplex, varexo_det)

__all__ = ["run_preprocessor", "Model", "ModelError", "steady_state", "first_order", "FirstOrderSolution",
           "k_order", "simulate", "irf", "SteadyStateError", "perfect_foresight", "PerfectForesightError",
           "PerfectForesightResult", "unanticipated_shocks", "Occbin", "OccbinResult", "OccbinError", "OccbinEstimation",
           "NonlinearEstimation",
           # modules
           "dates", "dsge_var", "endogenous_prior", "epilogue", "estimation", "external", "heteroskedastic", "kalman",
           "korder", "matlab", "moments", "newrat", "nonlinear_filters", "occbin_filter", "octave_rng",
           "optimal_policy", "plotting", "posterior", "posterior_plots", "priors", "samplers", "simplex", "smoother",
           "varexo_det"]

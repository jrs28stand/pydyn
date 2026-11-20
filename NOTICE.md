# Licence and provenance

pydyn is free software, released under the GNU General Public License,
version 3 or (at your option) any later version; see `LICENSE`. It comes with no warranty.

It is an independent project and is not affiliated with or endorsed by the Dynare Team.

The repository contains no copy of Dynare's or Octave's program code, but parts of it are derived
from Dynare 7.2 (https://www.dynare.org), which is distributed under the GPL, version 3 or later
(Copyright 1996-2026 Dynare Team, see Dynare's `license.txt`):

| File here | Derived from (Dynare 7.2) | Upstream terms |
|---|---|---|
| `pydyn/simulate.py` | `matlab/stochastic_solver/simult_.m`, `irf.m` | GPL-3.0-or-later, Dynare Team |
| `korder/korder_capi.cc` | `mex/sources/k_order_perturbation/k_order_perturbation.cc`; built against `mex/sources/libkorder` (not included) | GPL-3.0-or-later, Dynare Team |
| `pydyn/occbin.py` | `matlab/+occbin/solve_one_constraint.m`, `solve_two_constraints.m`, `mkdatap_anticipated_dyn.m`, `mkdatap_anticipated_2constraints_dyn.m`, `map_regime.m`, `mkdata.m` | public domain (original authors Luca Guerrieri and Matteo Iacoviello, adapted by the Dynare Team) |
| `pydyn/smoother.py`, decomposition figures in `pydyn/plotting.py` | `matlab/kalman/DsgeSmoother.m`, `evaluate_smoother.m`, `store_smoother_results.m`, `missing_DiffuseKalmanSmootherH1_Z.m`, `missing_DiffuseKalmanSmootherH3_Z.m`, `matlab/estimation/compute_Pinf_Pstar.m`, `get_posterior_parameters.m`, `set_prior.m`, `matlab/compute_trend_coefficients.m`, `matlab/distributions/compute_prior_mode.m`, `matlab/shock_decomposition/{shock_decomposition, initial_condition_decomposition, realtime_shock_decomposition, squeeze_shock_decomposition, plot_shock_decomposition, expand_group, graph_decomp, graph_decomp_detail}.m` | GPL-3.0-or-later, Dynare Team |
| `pydyn/moments.py` | `matlab/moments/*.m` (theoretical, simulated and pruned moments, variance decompositions), `matlab/matrix_solver/lyapunov_symm.m`, `matlab/+pruned_SS/pruned_state_space_system.m`, the HP and Baxter-King filters, `mex/sources/disclyap_fast` | GPL-3.0-or-later, Dynare Team |
| `pydyn/estimation.py`, `pydyn/kalman.py`, `pydyn/priors.py` | `matlab/estimation/*.m` (likelihood, priors, mode and sampler logic), `matlab/kalman/likelihood/*.m`, `matlab/distributions/*.m`, `matlab/optimization/hessian.m` | GPL-3.0-or-later, Dynare Team |
| `reference/est_schorfheide_data.m` | `examples/estimation/schorfheide_2000_data.m`, copied unchanged | CC-BY-4.0, Copyright 2000-2022 Frank Schorfheide, 2023-2025 Dynare Team (replication data for Schorfheide, 2000, Journal of Applied Econometrics; http://dx.doi.org/10.15456/jae.2022314.0708799949) |
| `pydyn/perfect_foresight.py` | `matlab/perfect-foresight-models/*.m` (set-up, stacked and period-by-period solvers, homotopy, expectation errors) and the preprocessor's handling of `initval`, `endval`, `histval`, `shocks`, `mshocks` | GPL-3.0-or-later, Dynare Team |
| `pydyn/perfect_foresight.py` (complementarity solver) | `matlab/lmmcp/lmmcp.m` | Copyright 2005 Christian Kanzow and Stefania Petra, 2013 Christophe Gouel, 2014-2022 Dynare Team; permissive terms ("Unlimited permission is granted to everyone to use, copy, modify or distribute this software") |
| `examples/nk_baseline_steadystate.py` | `examples/stochastic_simulations/nk_baseline_steadystate.m` | GPL-3.0-or-later, Dynare Team |
| `reference/*.mod` | Dynare's example and test models, shortened or with changed commands; original headers kept where the source had them | GPL-3.0-or-later, Dynare Team |
| `reference/*.mat` | numerical output of Dynare 7.2 run under GNU Octave | not program code |

All of these were modified in 2026. Everything else was written for this project.

The OccBin authors ask that use of the algorithm be acknowledged by citing: Luca Guerrieri and
Matteo Iacoviello (2015), "OccBin: A toolkit for solving dynamic models with occasionally binding
constraints easily", Journal of Monetary Economics 70, 22-38.

This file records provenance; it is not legal advice.

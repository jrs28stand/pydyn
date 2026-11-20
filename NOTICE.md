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
| `examples/nk_baseline_steadystate.py` | `examples/stochastic_simulations/nk_baseline_steadystate.m` | GPL-3.0-or-later, Dynare Team |
| `reference/*.mod` | Dynare's example and test models, shortened or with changed commands; original headers kept where the source had them | GPL-3.0-or-later, Dynare Team |
| `reference/*.mat` | numerical output of Dynare 7.2 run under GNU Octave | not program code |

All of these were modified in 2026. Everything else was written for this project.

The OccBin authors ask that use of the algorithm be acknowledged by citing: Luca Guerrieri and
Matteo Iacoviello (2015), "OccBin: A toolkit for solving dynamic models with occasionally binding
constraints easily", Journal of Monetary Economics 70, 22-38.

This file records provenance; it is not legal advice.

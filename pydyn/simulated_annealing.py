# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Port of Dynare 7.2 matlab/optimization/simulated_annealing.m (Copyright (C) Dynare Team, GPL-3.0-or-later; after
# Goffe, Ferrier and Rogers' simann).
"""Dynare's simulated annealing (``mode_compute=2``, ``pac.estimate.nls(..., 'annealing')``), drawing its uniforms from an
:class:`pydyn.octave_rng.OctaveRandom` stream, so that its path is Dynare's under Octave."""
from __future__ import annotations

import numpy as np

#: options_.saopt (default_option_values.m)
SAOPT = dict(neps=10, maximizer_indicator=0, rt=0.10, MaxIter=100000, verbosity=1, TolFun=1.0e-8, initial_temperature=15,
             ns=10, nt=10, step_length_c=0.1, initial_step_length=1)


def simulated_annealing(fcn, x, optim, lb, ub, rng):
    """``[xopt, fopt, exitflag, n_accepted, n_total, n_out_of_bounds, ..., message]``; ``fcn`` returns the objective."""
    x = np.array(x, float).ravel()
    lb, ub = np.asarray(lb, float).ravel(), np.asarray(ub, float).ravel()
    c = np.asarray(optim["step_length_c"], float) * np.ones(x.size)
    t = float(optim["initial_temperature"])
    vm = np.asarray(optim["initial_step_length"], float) * np.ones(x.size)
    n = x.size
    xp = np.zeros(n)
    n_acc = n_oob = n_tot = 0
    xopt = x.copy()
    nacp = np.zeros(n)
    neps, ns, nt = int(optim["neps"]), int(optim["ns"]), int(optim["nt"])
    fstar = 1e20 * np.ones(neps)
    sign = -1.0 if int(optim["maximizer_indicator"]) == 0 else 1.0
    if t <= 0.0:
        return xopt, 0.0, 3, 0, 0, 0, "The initial temperature is not positive. Reset the variable t"
    if np.sum(x > ub) + np.sum(x < lb) > 0:
        return xopt, 0.0, 2, 0, 0, 0, "Initial condition out of bounds"
    f = sign * fcn(x)
    n_tot += 1
    fopt = f
    fstar[0] = f
    max_iter = float(optim["MaxIter"])
    tolfun = float(optim["TolFun"])
    while True:
        for _ in range(nt):
            for _ in range(ns):
                for h in range(n):
                    for i in range(n):
                        if i == h:
                            xp[i] = x[i] + (float(np.ravel(rng.rand())[0]) * 2.0 - 1.0) * vm[i]
                        else:
                            xp[i] = x[i]
                        if xp[i] < lb[i] or xp[i] > ub[i]:
                            xp[i] = lb[i] + (ub[i] - lb[i]) * float(np.ravel(rng.rand())[0])
                            n_oob += 1
                    fp = sign * fcn(xp)
                    n_tot += 1
                    if n_tot >= max_iter:
                        return xopt, sign * fopt, 1, n_acc, n_tot, n_oob, (
                            "Too many function evaluations; consider increasing optim.MaxIter or optim.TolFun or decreasing "
                            "optim.nt or optim.rt. These results are likely to be poor")
                    if fp >= f:
                        x = xp.copy()
                        f = fp
                        n_acc += 1
                        nacp[h] += 1
                        if fp > fopt:
                            xopt = xp.copy()
                            fopt = fp
                    else:
                        p = np.exp((fp - f) / t)
                        pp = float(np.ravel(rng.rand())[0])
                        if pp < p:
                            x = xp.copy()
                            f = fp
                            n_acc += 1
                            nacp[h] += 1
            for i in range(n):
                ratio = nacp[i] / ns
                if ratio > .6:
                    vm[i] = vm[i] * (1. + c[i] * (ratio - .6) / .4)
                elif ratio < .4:
                    vm[i] = vm[i] / (1. + c[i] * ((.4 - ratio) / .4))
                if vm[i] > (ub[i] - lb[i]):
                    vm[i] = ub[i] - lb[i]
            nacp = np.zeros(n)
        fstar[0] = f
        quit_ = (fopt - fstar[0]) <= tolfun
        if np.sum(np.abs(f - fstar) > tolfun) > 0:
            quit_ = False
        if quit_:
            return xopt, sign * fopt, 0, n_acc, n_tot, n_oob, "SA achieved termination criteria."
        t = float(optim["rt"]) * t
        for i in range(neps - 1, 0, -1):
            fstar[i] = fstar[i - 1]
        f = fopt
        x = xopt.copy()

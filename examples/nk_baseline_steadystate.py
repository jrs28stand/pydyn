# This file is part of pydyn.
#
# It is a Python port of examples/stochastic_simulations/nk_baseline_steadystate.m from Dynare 7.2
# (Copyright (C) 2013-2024 Dynare Team, https://www.dynare.org), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Python counterpart of Dynare's examples/stochastic_simulations/nk_baseline_steadystate.m.

A steady-state function receives the model, the current parameter vector and the exogenous
steady state, and returns the steady state (original variables are enough; auxiliary variables are
filled in automatically) together with the possibly updated parameters.
"""
import numpy as np
from scipy.optimize import newton

from pydyn import SteadyStateError


def steady_state(model, params, exo):
    p = dict(zip(model.param_names, params))
    betta, h, alppha = p["betta"], p["h"], p["alppha"]
    thetap, thetaw, chi, chiw = p["thetap"], p["thetaw"], p["chi"], p["chiw"]
    epsilon, eta, gammma, varpsi, Phi = p["epsilon"], p["eta"], p["gammma"], p["varpsi"], p["Phi"]
    LambdaA, Lambdamu = p["LambdaA"], p["Lambdamu"]
    delta = p["delta"]

    PI = p["PIbar"]
    u = q = d = phi = 1.0
    LambdaYd = (LambdaA + alppha * Lambdamu) / (1 - alppha)
    mu_z, mu_I, mu_A = np.exp(LambdaYd), np.exp(Lambdamu), np.exp(LambdaA)
    Lambdax = mu_z
    gammma1 = mu_z * mu_I / betta - (1 - delta)
    if gammma1 < 0:
        raise SteadyStateError("parameter restriction violated: gammma1 < 0")
    r = gammma1
    R = 1 + (PI * mu_z / betta - 1)
    Rbar = R
    PIstar = ((1 - thetap * PI ** (-(1 - epsilon) * (1 - chi))) / (1 - thetap)) ** (1 / (1 - epsilon))
    PIstarw = ((1 - thetaw * PI ** (-(1 - chiw) * (1 - eta)) * mu_z ** (-(1 - eta))) / (1 - thetaw)) ** (1 / (1 - eta))
    mc = ((epsilon - 1) / epsilon * (1 - betta * thetap * PI ** ((1 - chi) * epsilon))
          / (1 - betta * thetap * PI ** (-(1 - epsilon) * (1 - chi))) * PIstar)
    w = (1 - alppha) * (mc * (alppha / r) ** alppha) ** (1 / (1 - alppha))
    wstar = w * PIstarw
    vp = (1 - thetap) / (1 - thetap * PI ** ((1 - chi) * epsilon)) * PIstar ** (-epsilon)
    vw = (1 - thetaw) / (1 - thetaw * PI ** ((1 - chiw) * eta) * mu_z ** eta) * PIstarw ** (-eta)
    tempvaromega = alppha / (1 - alppha) * w / r * mu_z * mu_I
    cons_per_ld = mu_A / mu_z / vp * tempvaromega ** alppha - tempvaromega * (1 - (1 - delta) / (mu_z * mu_I))

    def labour_eq(ld):
        return ((1 - betta * thetaw * mu_z ** (eta - 1) * PI ** (-(1 - chiw) * (1 - eta)))
                / (1 - betta * thetaw * mu_z ** (eta * (1 + gammma)) * PI ** (eta * (1 - chiw) * (1 + gammma)))
                - (eta - 1) / eta * wstar / (varpsi * PIstarw ** (-eta * gammma) * ld ** gammma)
                * ((1 - h / mu_z) ** (-1) - betta * h / (mu_z - h))
                / (cons_per_ld * ld - Phi / vp))

    try:
        ld = newton(labour_eq, 0.25, tol=1e-15, maxiter=200)
        if not np.isfinite(ld) or abs(labour_eq(ld)) > 1e-10:
            raise RuntimeError
    except RuntimeError:
        raise SteadyStateError("no solution for steady-state labour") from None

    l = vw * ld
    k = tempvaromega * ld
    x = (1 - (1 - delta) / (mu_z * mu_I)) * k
    yd = (mu_A / mu_z * k ** alppha * ld ** (1 - alppha) - Phi) / vp
    c = cons_per_ld * ld - Phi / vp
    lambda_ = (1 - h * betta / mu_z) / (1 - h / mu_z) / c
    F = yd - 1 / (1 - alppha) * w * ld
    f = ((eta - 1) / eta * wstar * PIstarw ** (-eta) * lambda_ * ld
         / (1 - betta * thetaw * mu_z ** (eta - 1) * PI ** (-(1 - chiw) * (1 - eta))))
    f2 = (varpsi * d * phi * PIstarw ** (-eta * (1 + gammma)) * ld ** (1 + gammma)
          / (1 - betta * thetaw * (PI ** chiw / PI) ** (-eta * (1 + gammma)) * mu_z ** (eta * (1 + gammma))))
    g1 = lambda_ * mc * yd / (1 - betta * thetap * PI ** ((1 - chi) * epsilon))
    g2 = epsilon / (epsilon - 1) * g1

    local = dict(locals())
    local["lambda"] = lambda_
    # parameters set in this file (Rbar and delta) are written back, as the .m file does
    new_params = np.array([local.get(nm, p[nm]) for nm in model.param_names], float)
    n_orig = model.raw["orig_endo_nbr"]
    ys = np.array([local[nm] for nm in model.endo_names[:n_orig]], float)
    return ys, new_params

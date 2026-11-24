# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
#
# Port of Octave 8.4's fminsearch.m (scripts/optimization/fminsearch.m, Copyright (C) 2003-2023 The Octave Project Developers,
# GPL-3.0-or-later), whose inner routine nmsmax is the Nelder-Mead code of N. J. Higham's Matrix Toolbox (Copyright (C) 2002,
# 2013 N. J. Higham; modified for Octave by A. Adler, 2003), distributed with Octave under the GPL.  Modified in 2026 for
# Python (this file keeps the algorithm, stopping rules and counts of the original).
"""Dynare's ``mode_compute=7``: ``fminsearch`` as Octave runs it (the reference pydyn is validated against runs Dynare under
Octave 8.4).  MATLAB's own ``fminsearch`` is a different implementation of Nelder-Mead and takes other steps."""
from __future__ import annotations

import numpy as np


def fminsearch(fcn, x0, tol_x=1e-4, tol_f=1e-4, max_fun_evals=None, max_iter=None):
    """Minimise ``fcn`` from ``x0`` with Octave's ``nmsmax`` (regular initial simplex, direction -1).  Returns
    ``(x, fval, exitflag, output)`` with ``output`` a dict of ``iterations``, ``funcCount`` and ``message``; ``fval`` is one
    more call of ``fcn`` at ``x``, as ``fminsearch`` makes."""
    x0 = np.array(x0, float).ravel()
    n = x0.size
    max_fun_evals = 200 * n if max_fun_evals is None else max_fun_evals
    max_iter = 200 * n if max_iter is None else max_iter
    dirn = -1.0

    def F(v):
        return dirn * float(np.real(fcn(np.array(v, float))))

    V = np.zeros((n, n + 1))
    f = np.zeros(n + 1)
    V[:, 0] = x0
    f[0] = F(x0)
    fmax_old = f[0]
    nf = 1
    scale = max(np.max(np.abs(x0)), 1.0)
    alpha = scale / (n * np.sqrt(2)) * np.array([np.sqrt(n + 1) - 1 + n, np.sqrt(n + 1) - 1])
    V[:, 1:] = (x0 + alpha[1])[:, None] * np.ones((1, n))
    for j in range(1, n + 1):
        V[j - 1, j] = x0[j - 1] + alpha[0]
        f[j] = F(V[:, j])
    nf += n

    def reorder(V, f):
        j = np.argsort(f, kind="stable")[::-1]          # [~, j] = sort (f); j = j(n+1:-1:1)
        return V[:, j], f[j]

    V, f = reorder(V, f)
    exitflag = 0
    msg = ""
    k = 0
    a, b, g = 1.0, 0.5, 2.0
    while exitflag != -1:
        k += 1
        if k > max_iter:
            msg = "Exceeded maximum iterations\n"
            break
        fmax = f[0]
        fmax_old = fmax
        if nf >= max_fun_evals:
            msg = "Exceeded maximum number of function evaluations\n"
            exitflag = 0
            break
        v1 = V[:, 0]
        size_simplex = np.linalg.norm(V[:, 1:] - v1[:, None], 1) / max(1.0, np.linalg.norm(v1, 1))
        with np.errstate(invalid="ignore"):
            step_f = np.max(np.abs(f[0] - f[1:]))
        if size_simplex <= tol_x and step_f <= tol_f:
            msg = "Algorithm converged.\n"
            exitflag = 1
            break
        vbar = np.sum(V[:, :n], axis=1) / n
        vr = (1 + a) * vbar - a * V[:, n]
        fr = F(vr)
        nf += 1
        vk, fk = vr, fr
        if fr > f[n - 1]:
            if fr > f[0]:
                ve = g * vr + (1 - g) * vbar
                fe = F(ve)
                nf += 1
                if fe > f[0]:
                    vk, fk = ve, fe
        else:
            vt, ft = V[:, n], f[n]
            if fr > ft:
                vt, ft = vr, fr
            vc = b * vt + (1 - b) * vbar
            fc = F(vc)
            nf += 1
            if fc > f[n - 1]:
                vk, fk = vc, fc
            else:
                for j in range(1, n):
                    V[:, j] = (V[:, 0] + V[:, j]) / 2
                    f[j] = F(V[:, j])
                nf += n - 1
                vk = (V[:, 0] + V[:, n]) / 2
                fk = F(vk)
                nf += 1
        V[:, n] = vk
        f[n] = fk
        V, f = reorder(V, f)
    x = V[:, 0].copy()
    fval = float(np.real(fcn(x.copy())))
    return x, fval, exitflag, {"iterations": k, "funcCount": nf, "algorithm": "Nelder-Mead simplex direct search",
                               "message": msg}

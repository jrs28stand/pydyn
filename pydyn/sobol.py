"""Dynare's ``qmc_sequence`` mex file (Sobol sequences): Burkardt's ``i8_sobol`` (Bratley and Fox, with
the Joe-Kuo direction numbers for up to 1111 dimensions) and the Gaussian transformation of
``mex/sources/sobol/gaussian.hh``, reproduced bit for bit (the direction numbers are in
``data/sobol_tables.json``, extracted from Dynare's ``sobol.f08``)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

DIM_MAX = 1111
LOG_MAX = 62

__all__ = ["qmc_sequence", "icdf", "sobol_points"]


@lru_cache(maxsize=1)
def _tables():
    d = json.loads((Path(__file__).parent / "data" / "sobol_tables.json").read_text())
    return d["v"], d["poly"]


@lru_cache(maxsize=8)
def _direction_numbers(dim: int):
    """V (dim x maxcol) of i8_sobol after its initialisation (columns already scaled by powers of two) and
    ``recipd``."""
    vi, poly = _tables()
    maxcol = LOG_MAX
    V = [[0] * (maxcol + 1) for _ in range(dim + 1)]            # 1-based
    for j in range(1, maxcol + 1):
        V[1][j] = 1
    for i in range(2, dim + 1):
        for j in range(1, 14):
            V[i][j] = vi[j - 1][i - 1]
    for i in range(2, dim + 1):
        j = poly[i - 1]
        m = 0
        while True:
            j //= 2
            if j <= 0:
                break
            m += 1
        j = poly[i - 1]
        includ = [False] * (m + 1)
        for k in range(m, 0, -1):
            j2 = j // 2
            includ[k] = j != 2 * j2
            j = j2
        for j in range(m + 1, maxcol + 1):
            newv = V[i][j - m]
            lval = 1
            for k in range(1, m + 1):
                lval *= 2
                if includ[k]:
                    newv ^= lval * V[i][j - k]
            V[i][j] = newv
    lval = 1
    for j in range(maxcol - 1, 0, -1):
        lval *= 2
        for i in range(1, dim + 1):
            V[i][j] *= lval
    recipd = 0.5 / float(lval)
    arr = np.array([[V[i][j] for j in range(1, maxcol + 1)] for i in range(1, dim + 1)], dtype=np.int64)
    return arr, recipd


def _bit_lo0(n: int) -> int:
    bit = 0
    while True:
        bit += 1
        i2 = n // 2
        if n == 2 * i2:
            return bit
        n = i2


def sobol_points(dim: int, seed: int, n: int) -> tuple[np.ndarray, int]:
    """``n`` successive points of the sequence from index ``seed`` (``i8_sobol``): (dim x n), new seed."""
    if not 1 <= dim <= DIM_MAX:
        raise ValueError(f"qmc_sequence: the dimension must be between 1 and {DIM_MAX}")
    seed = max(int(seed), 0)
    V, recipd = _direction_numbers(dim)
    lastq = np.zeros(dim, dtype=np.int64)
    for s in range(seed):
        lastq ^= V[:, _bit_lo0(s) - 1]
    out = np.empty((dim, n))
    for k in range(n):
        out[:, k] = lastq.astype(np.float64) * recipd
        lastq ^= V[:, _bit_lo0(seed + k) - 1]
    return out, seed + n


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
_LB, _UB = .02425, .97575


def icdf(u: float) -> float:
    """``icdf`` of gaussian.hh: Acklam's rational approximation and one step of Halley's refinement."""
    g = 0.0
    if 0 < u < _LB:
        t = math.sqrt(-2 * math.log(u))
        g = (((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]) / \
            ((((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1)
    elif _LB <= u <= _UB:
        t = u - .5
        T = t * t
        g = (((((_A[0] * T + _A[1]) * T + _A[2]) * T + _A[3]) * T + _A[4]) * T + _A[5]) * t / \
            (((((_B[0] * T + _B[1]) * T + _B[2]) * T + _B[3]) * T + _B[4]) * T + 1)
    elif _UB < u < 1:
        t = math.sqrt(-2 * math.log(1 - u))
        g = -(((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]) / \
            ((((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1)
    if 0 < u < 1:
        t = .5 * math.erfc(-g / math.sqrt(2.0)) - u
        t2 = t * math.sqrt(2 * math.pi) * math.exp(.5 * g * g)
        g = g - t2 / (1 + .5 * g * t2)
    if u == 0:
        g = -math.inf
    if u == 1:
        g = math.inf
    return g


def qmc_sequence(d: int, seed: int, kind: int, n: int, lu=None):
    """``[a, seed] = qmc_sequence(d, int64(seed), kind, n)``: (d x n) matrix of Sobol numbers, ``kind`` 0 uniform
    (``lu``: d x 2 bounds), 1 Gaussian (``lu``: d x d lower Cholesky factor of the covariance matrix), 2 uniform
    on the hypersphere (``lu``: radius)."""
    if kind not in (0, 1, 2):
        raise ValueError("qmc_sequence: the type must be 0, 1 or 2")
    if kind == 2 and d < 2:
        raise ValueError("qmc_sequence: the dimension must be greater than 1 for a uniform QMC on an hypersphere")
    a, seed_out = sobol_points(d, seed, n)
    if kind == 0 and lu is not None:
        lu = np.asarray(lu, float)
        a = lu[:, [0]] + (lu[:, 1] - lu[:, 0])[:, None] * a
    elif kind in (1, 2):
        flat = a.ravel(order="F")
        a = np.array([icdf(float(x)) for x in flat]).reshape(a.shape, order="F")
        if kind == 1 and lu is not None:
            a = np.asarray(lu, float) @ a
        elif kind == 2:
            nrm = np.sqrt(np.sum(a * a, axis=0))
            a = a / nrm * (1.0 if lu is None else float(lu))
    return a, seed_out

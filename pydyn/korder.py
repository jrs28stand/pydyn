"""Higher-order perturbation through Dynare's own C++ k-order library (libkorder).

The library is compiled as a plain shared library (see ``korder/build.sh``) and called with
ctypes. Model derivatives are evaluated in Python from the preprocessor's JSON output and
passed in; no MATLAB or Octave is involved.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import numpy as np

from .perturbation import QZ_CRITERIUM, dr_ordering
from .steady import steady_state

_NAMES = ["gy", "gu", "gyy", "gyu", "guu", "gss", "gyyy", "gyyu", "gyuu", "guuu", "gyss", "guss"]
#: Dynare's names for the same objects in oo_.dr
DYNARE_NAMES = {"gy": "ghx", "gu": "ghu", "gyy": "ghxx", "gyu": "ghxu", "guu": "ghuu", "gss": "ghs2",
                "gyyy": "ghxxx", "gyyu": "ghxxu", "gyuu": "ghxuu", "guuu": "ghuuu",
                "gyss": "ghxss", "guss": "ghuss"}
_lib = None
_lock = threading.Lock()      # the C++ library keeps process-wide state
_pid = None


class KOrderError(Exception):
    pass


def load_library(path: str | os.PathLike | None = None):
    global _lib
    if _lib is not None and path is None:
        return _lib
    cands = [path, os.environ.get("PYDYN_KORDER_LIB"),
             Path(__file__).resolve().parent.parent / "korder" / "libpydyn_korder.so"]
    for c in cands:
        if c and Path(c).exists():
            lib = ctypes.CDLL(str(c))
            break
    else:
        raise FileNotFoundError("libpydyn_korder.so not found; build it with korder/build.sh "
                                "or set PYDYN_KORDER_LIB")
    dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.pydyn_korder_solve.restype = ctypes.c_void_p
    lib.pydyn_korder_solve.argtypes = [ctypes.c_int] * 7 + [dp, dp, dp, ip, ip, ip, ip, dp,
                                                            ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.pydyn_korder_error.restype = ctypes.c_char_p
    lib.pydyn_korder_error.argtypes = [ctypes.c_void_p]
    lib.pydyn_korder_dims.restype = ctypes.c_int
    lib.pydyn_korder_dims.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ip, ip]
    lib.pydyn_korder_copy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, dp]
    lib.pydyn_korder_free.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


def k_order(model, order=None, ys=None, params=None, pruning=None, threads=1, library=None,
            qz_criterium=None, loglinear=None) -> dict:
    """Solve the model by perturbation at ``order`` (1 to 3) with Dynare's C++ solver.

    Returns a dict with Dynare's decision-rule arrays (``ghx``, ``ghu``, ``ghxx``, ``ghxu``,
    ``ghuu``, ``ghs2``, ... as in ``oo_.dr``), the folded tensors ``g_0`` ... ``g_k``, plus
    ``ys``, ``order_var`` and ``state_var`` (0-based declaration indices).
    Rows are in decision-rule order; state columns are (state variables, shocks).
    """
    from . import options as _opt
    so = _opt.stoch_options(model)
    if order is None:                      # as in the .mod file; Dynare's own default is 2
        order = int(so.get("order", 2))
    if pruning is None:
        pruning = bool(so.get("pruning", False))
    if loglinear is None:
        loglinear = bool(so.get("loglinear", False))
    if loglinear and order > 1:
        raise KOrderError("Loglinear options currently only works at order 1")      # as Dynare
    qz = _opt.qz_criterium(model, qz_criterium)
    if order > model.max_deriv_order:
        raise KOrderError(f"the preprocessor output only has derivatives up to order "
                          f"{model.max_deriv_order}; rerun it with output=third or set order in the .mod file")
    global _pid
    lib = load_library(library)
    if _pid is None:
        _pid = os.getpid()
    elif _pid != os.getpid():
        raise KOrderError("the k-order library was already used in the parent process; its worker "
                          "threads do not survive fork(). Use the 'spawn' start method.")
    if ys is None:
        ys, params = steady_state(model, params)
    params = np.ascontiguousarray(model.params if params is None else params, dtype=float)
    ys = np.ascontiguousarray(ys, dtype=float)
    order_var, state_var, cnt = dr_ordering(model.lead_lag_incidence)

    if not (np.all(np.isfinite(ys)) and np.all(np.isfinite(params)) and np.all(np.isfinite(model.Sigma_e))):
        raise KOrderError("steady state, parameters or Sigma_e contain NaN or Inf")
    yy, x0 = np.tile(ys, 3), model.exo_steady()
    nnz, rows, cols, vals = [], [], [], []
    for o in range(1, order + 1):
        r, c, v = model.dynamic_sparse_derivs(o, yy, x0, params, ys)
        if o > 1 and len(r):
            key = np.column_stack([r, np.sort(c, axis=1)])
            if len(np.unique(key, axis=0)) != len(key):
                raise KOrderError("duplicate symmetric entries in the derivative list")
        nnz.append(len(r)); rows.append(r); cols.append(c.ravel()); vals.append(v)
    nnz = np.array(nnz, dtype=np.int32)
    rows = np.ascontiguousarray(np.concatenate(rows), dtype=np.int32)
    cols = np.ascontiguousarray(np.concatenate(cols), dtype=np.int32)
    vals = np.ascontiguousarray(np.concatenate(vals), dtype=float)
    sigma = np.asfortranarray(model.Sigma_e, dtype=float)
    dr = np.ascontiguousarray(order_var, dtype=np.int32)

    dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    as_d = lambda a: a.ctypes.data_as(dp)  # noqa: E731
    as_i = lambda a: a.ctypes.data_as(ip)  # noqa: E731
    with _lock:
        h = lib.pydyn_korder_solve(order, cnt["nstatic"], cnt["npred"], cnt["nboth"], cnt["nfwrd"],
                                   model.n_exo, len(params), as_d(ys), as_d(sigma), as_d(params),
                                   as_i(dr), as_i(nnz), as_i(rows), as_i(cols), as_d(vals),
                                   qz, int(bool(pruning)), int(threads))
    try:
        err = lib.pydyn_korder_error(h).decode()
        if err:
            raise KOrderError(err)

        def get(name):
            nr, nc = ctypes.c_int(), ctypes.c_int()
            if not lib.pydyn_korder_dims(h, name.encode(), ctypes.byref(nr), ctypes.byref(nc)):
                return None
            out = np.empty(nr.value * nc.value)
            lib.pydyn_korder_copy(h, name.encode(), as_d(out))
            return out.reshape((nr.value, nc.value), order="F")

        res = {"ys": ys, "order_var": order_var, "state_var": state_var, "order": order, "pruning": bool(pruning),
               "params": params.copy()}
        for nm in _NAMES:
            a = get(nm)
            if a is not None:
                res[DYNARE_NAMES[nm]] = a
        for i in range(order + 1):
            res[f"g_{i}"] = get(f"g_{i}")
            if pruning:
                res[f"pruning_g_{i}"] = get(f"pruning_g_{i}")
        if loglinear:                                    # order 1 only (checked above)
            from .perturbation import log_mask
            mask = log_mask(model)
            d = np.where(mask, ys, 1.0)
            if np.any(ys[mask] <= 0):
                raise KOrderError("The loglinearization of the model cannot be performed, because the "
                                  "steady state is not strictly positive")
            res["ghx"] = res["ghx"] * d[state_var][None, :] / d[order_var][:, None]
            res["ghu"] = res["ghu"] / d[order_var][:, None]
            res["steady_state"] = ys.copy()
            res["ys"] = np.where(mask, np.log(d), ys)
            res["loglinear"] = True
            for i in range(order + 1):                   # the folded tensors are in levels
                res.pop(f"g_{i}", None)
        return res
    finally:
        lib.pydyn_korder_free(h)

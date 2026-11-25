# This file is part of pydyn.
#
# It follows matlab/+identification/run.m of Dynare 7.2 (Copyright (C) 2010-2026 Dynare Team, GPL-3.0-or-later), modified in
# 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""The ``identification`` statement of :func:`pydyn.dynare` (first order: :mod:`pydyn.identification`).

Dynare's ``identification.run`` changes none of ``M_``, ``oo_`` and ``options_`` (except ``options_.nodisplay`` and
``options_.nograph`` when the statement gives them) and writes its results to ``<name>/identification/<name>_identif.mat``
and ``<name>/identification/<name>_<parameter set>_identif.mat`` (``ide_moments_point``, ``ide_spectrum_point``,
``ide_minimal_point``, ``ide_hess_point``, ``ide_reducedform_point``, ``ide_dynamic_point``, ``store_options_ident``).
pydyn writes the same files under the rule of ``save``: the folder of the .mod file only, and a file that exists is
overwritten only if pydyn.dynare() wrote it."""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np

_COLUMN_FIELDS = {"DYNAMIC", "REDUCEDFORM", "MOMENTS", "S", "Mco"}


def _R():
    from . import results
    return results


def _unsupported(msg, kind):
    from .matlab import Unsupported
    return Unsupported(msg, kind)


def _mat(v, key=None):
    """A value of the identification structures as scipy.io.savemat writes it (row vectors for 1-d arrays, as Dynare's)."""
    if isinstance(v, dict):
        return {k: _mat(x, k) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return np.array([[bool(v)]])
    if isinstance(v, (int, float, np.floating, np.integer)):
        return np.array([[float(v)]])
    if isinstance(v, str):
        return v
    a = np.asarray(v)
    if a.dtype == object:
        out = np.empty(a.shape, dtype=object)
        for ix in np.ndindex(a.shape):
            out[ix] = _mat(a[ix])
        return out
    if a.dtype.kind in "fiub":
        a = a.astype(float)
        if a.ndim == 1 and a.size == 0:
            return np.zeros((0, 1)) if key in _COLUMN_FIELDS else np.zeros((0, 0))
        if a.ndim == 1:
            a = a.reshape(-1, 1) if key in _COLUMN_FIELDS else a.reshape(1, -1)
        return a
    raise TypeError(f"cannot write {type(v)}")


class IdentificationMixin:
    def _functions_ident(self):
        return {"rank": self.fn_rank, "lyapunov_symm": self.fn_lyapunov_symm,
                "identification_get_minimal_state_representation": self.fn_identification_get_minimal_state_representation}

    # ------------------------------------------------------------------ MATLAB functions of .mod files
    def fn_rank(self, args, nargout):
        """``rank(A)`` / ``rank(A, tol)`` (Octave's: singular values above ``max(size(A)) eps(max sv)`` or above ``tol``)."""
        from .matlab import farr, _F
        from .identification import rank
        if not 1 <= len(args) <= 2:
            raise _R().MatlabError("Invalid call to rank")
        A = np.asarray(farr(args[0]), float)
        tol = None if len(args) == 1 else float(np.asarray(farr(args[1]), float).ravel()[0])
        return [_F(float(rank(A, tol)))]

    def fn_lyapunov_symm(self, args, nargout):
        """``[X, u] = lyapunov_symm(A, B, fixed_point_tol, qz_criterium, complex_threshold, method, debug)``: the solution of
        ``X = A X A' + B`` on the stable Schur subspace (pydyn.moments.lyapunov_symm; the method argument only selects
        Dynare's algorithm, the solution is the same)."""
        from .matlab import farr
        from .moments import lyapunov_symm
        R = _R()
        if len(args) < 2:
            raise R.MatlabError("Invalid call to lyapunov_symm")
        kw = {}
        if len(args) > 3 and np.asarray(farr(args[3])).size:
            kw["qz_criterium"] = float(np.asarray(farr(args[3]), float).ravel()[0])
        if len(args) > 4 and np.asarray(farr(args[4])).size:
            kw["complex_threshold"] = float(np.asarray(farr(args[4]), float).ravel()[0])
        x, u = lyapunov_symm(np.asarray(farr(args[0]), float), np.asarray(farr(args[1]), float), **kw)
        return [R.mat(x), R.mat(u)] if u.size else [R.mat(x), np.zeros((0, 0))]

    def fn_identification_get_minimal_state_representation(self, args, nargout):
        """``[CheckCO, minns, minSYS] = identification.get_minimal_state_representation(SYS, derivs_flag)``."""
        from .matlab import farr, _F
        from .identification import get_minimal_state_representation
        R = _R()
        if not 1 <= len(args) <= 2 or not hasattr(args[0], "items"):
            raise R.MatlabError("identification.get_minimal_state_representation: SYS must be a struct with A, B, C, D")
        sys_in = {k: np.asarray(farr(args[0][k]), float) for k in ("A", "B", "C", "D")}
        derivs = len(args) == 2 and bool(np.asarray(farr(args[1]), float).ravel()[0])
        if derivs:
            raise _unsupported("identification.get_minimal_state_representation with derivs_flag", "function")
        ok, minns, out = get_minimal_state_representation(sys_in, False)
        s = R.rstruct("minSYS")
        if out is not None:
            for k in ("A", "B", "C", "D"):
                s[k] = R.mat(np.atleast_2d(out[k]))
        return [np.bool_(ok), _F(float(minns)) if minns is not None else R.mat(np.zeros((0, 0))), s]

    def st_identification(self, st):
        R = _R()
        from .identification import Identification
        opts = dict(st.get("options") or {})
        self.write_options({k: opts[k] for k in ("nodisplay", "nograph", "graph_format") if k in opts})
        self.pull()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ident = Identification(self.model, opts, rng=self.rng)
                out = ident.run_point()
        except (NotImplementedError, ValueError, np.linalg.LinAlgError) as exc:
            raise R.ResultsError(f"identification: {exc}") from None
        if out is None:
            return
        label, variables, messages = out
        for msg in messages:
            self.warnings.append(f"identification: {msg}")
        name = self.name
        self._write_identification_mat(f"{name}_identif.mat", variables)
        self._write_identification_mat(f"{name}_{label}_identif.mat", variables)

    def _write_identification_mat(self, fname, variables):
        """``save([IdentifDirectoryName '/' fname], 'ide_...')`` into ``<mod folder>/<name>/identification``."""
        from scipy.io import savemat
        R = _R()
        folder = self.model.json_dir.parents[1] / "identification"
        path = folder / fname
        rec_path = self.model.json_dir.parents[1] / self._WRITTEN
        key = f"identification/{fname}"
        try:
            rec = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            rec = {}
        if path.exists():
            ok = path.is_file() and not path.is_symlink() and rec.get(key) == hashlib.sha256(path.read_bytes()).hexdigest()
            if not ok:
                raise R.ResultsError(f"identification: {key} exists and was not written by pydyn.dynare(); it is not "
                                     "overwritten")
        folder.mkdir(parents=True, exist_ok=True)
        savemat(str(path), {k: _mat(v, k) for k, v in variables.items()}, do_compression=False, oned_as="row",
                long_field_names=True)
        rec[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        rec_path.write_text(json.dumps(rec, indent=0))

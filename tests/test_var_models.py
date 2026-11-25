# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""var_model, trend_component_model, var_expectation_model and the PAC machinery against Dynare 7.2 run under Octave.

reference/var/<case>.mod are Dynare's own test files (tests/var-expectations, tests/pac, tests/trend-component-and-var-models,
named in the table of reference/var/r_var.m, which also made reference/var/<case>.mat: M_.params, the sub-model structures
of M_ and oo_, the last 12 rows of the dseries and the numeric variables of the workspace, and the .inc files that
pac.print / var_expectation.print wrote). The preprocessor must be the one of Dynare 7.2 (DYNARE_PREPROCESSOR)."""
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import dseries, matlab, pac, submodels, var_model

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference"))) / "var"
CASES = re.findall(r"^ '(\w+)',", (REF / "r_var.m").read_text(), re.M)

#: relative tolerance of the comparisons; the quantities are sums of products of a few dozen terms evaluated in the same
#: order as in Dynare, and agree to 1e-15 in practice
RTOL, ATOL = 1e-11, 1e-12
#: cases whose simulated data differ from Dynare's (see NOTES.md): the backward simulation of the unary-operator
#: auxiliary variables of pac/trend-component-11 is outside the sub-model code
_runs = {}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    def _run(case):
        if case not in _runs:
            d = tmp_path_factory.mktemp(case)
            shutil.copy(REF / f"{case}.mod", d)
            _runs[case] = pydyn.dynare(d / f"{case}.mod"), d
        return _runs[case]
    return _run


def _mat(case):
    return loadmat(REF / f"{case}.mat", squeeze_me=False, struct_as_record=True, chars_as_strings=True)


def _diffs(py, ref, path, out):
    """Collect the differences between a pydyn value and a loadmat value (recursively)."""
    if isinstance(ref, np.ndarray) and ref.dtype.names:
        els = list(ref.ravel(order="F"))
        pyel = list(py.elems) if hasattr(py, "elems") else [py]
        if len(els) != len(pyel):
            out.append(f"{path}: struct array of {len(pyel)} != {len(els)}"); return
        for k, (pe, re_) in enumerate(zip(pyel, els)):
            if not isinstance(pe, dict):
                out.append(f"{path}: not a struct"); return
            for nm in ref.dtype.names:
                if nm not in pe:
                    # a struct array of Octave has the fields of all its elements, empty where the element lacks them
                    if not (isinstance(re_[nm], np.ndarray) and re_[nm].size == 0 and len(els) > 1):
                        out.append(f"{path}.{nm}: missing")
                    continue
                _diffs(pe[nm], re_[nm], f"{path}{'' if len(els) == 1 else f'({k})'}.{nm}", out)
        return
    if isinstance(ref, np.ndarray) and ref.dtype == object:           # cell
        items = list(ref.ravel(order="F"))
        pyi = list(py.items) if isinstance(py, matlab.Cell) else list(np.asarray(py, dtype=object).ravel(order="F"))
        if len(items) != len(pyi):
            out.append(f"{path}: cell of {len(pyi)} != {len(items)}"); return
        for k, (a, b) in enumerate(zip(pyi, items)):
            _diffs(a, b, f"{path}{{{k}}}", out)
        return
    if isinstance(ref, np.ndarray) and ref.dtype.kind in "US":
        s = "".join(ref.ravel()) if ref.size else ""
        if str(py) != s:
            out.append(f"{path}: {py!r} != {s!r}")
        return
    a = np.asarray(py if isinstance(py, np.ndarray) else (matlab.farr(py) if not isinstance(py, (bool, np.bool_)) else float(py)),
                   float).ravel(order="F")
    b = np.asarray(ref, float).ravel(order="F")
    if a.size != b.size:
        out.append(f"{path}: size {a.size} != {b.size}"); return
    nan = np.isnan(a) | np.isnan(b)
    if (np.isnan(a) != np.isnan(b)).any():
        out.append(f"{path}: NaN pattern"); return
    if not np.allclose(a[~nan], b[~nan], rtol=RTOL, atol=ATOL):
        out.append(f"{path}: max abs diff {np.max(np.abs(a[~nan] - b[~nan])):.2e}")


@pytest.mark.parametrize("case", CASES)
def test_parameters_and_submodels(run, case):
    ws, _ = run(case)
    ref = _mat(case)
    out = []
    _diffs(ws.M_["params"], ref["params"], "M_.params", out)
    for key, tgt in (("M_var", ("M_", "var")), ("M_tc", ("M_", "trend_component")), ("M_ve", ("M_", "var_expectation")),
                     ("M_pac", ("M_", "pac")), ("oo_var", ("oo_", "var")), ("oo_tc", ("oo_", "trend_component"))):
        if key in ref:
            _diffs(getattr(ws, tgt[0])[tgt[1]], ref[key], ".".join(tgt), out)
    assert not out, "\n".join(out[:20])


@pytest.mark.parametrize("case", CASES)
def test_series_and_scalars(run, case):
    ws, _ = run(case)
    ref = _mat(case)
    out = []
    for k in ref:
        if k.startswith("ds_"):
            v = ws.vars.get(k[3:])
            assert hasattr(v, "data") and hasattr(v, "names"), k
            r = ref[k]
            a = r["data"][0, 0]
            assert v.data.shape[0] == int(r["nobs"][0, 0][0, 0]) and v.data.shape[1] == a.shape[1], k
            assert list(v.names) == [str(x[0]) for x in r["name"][0, 0].ravel()], k
            _diffs(v.data[-a.shape[0]:], a, k, out)
        elif k.startswith("v_") and k[2:] in ws.vars and isinstance(ws.vars[k[2:]], (int, float, np.ndarray, np.floating)):
            _diffs(ws.vars[k[2:]], ref[k], k, out)
    assert not out, "\n".join(out[:20])


@pytest.mark.parametrize("case", CASES)
def test_inc_files(run, case):
    ws, d = run(case)
    ref = _mat(case)
    for k in ref:
        if k.startswith("inc_"):
            name = k[4:]
            files = [p for p in (d / case).rglob("*.inc") if re.sub("[^A-Za-z0-9]", "_", p.name) == name]
            assert files, name
            text = files[0].read_text()
            text = text[text.index("\n") + 1:]                     # the first line holds the time stamp
            assert _numbers_close(text, "".join(ref[k].ravel())), name


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers_close(a, b, rtol=1e-12):
    """The text of two .inc files: the same but for the numbers, which agree to rtol (they are printed with 16 digits)."""
    if _NUM.sub("#", a) != _NUM.sub("#", b):
        return False
    x, y = np.array(_NUM.findall(a), float), np.array(_NUM.findall(b), float)
    return x.shape == y.shape and np.allclose(x, y, rtol=rtol, atol=0)


# ------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("call, text", [
    ("pac.estimate.nls('eq:pac', struct('a', 1), 'csminwel');", "Not enough input arguments"),       # estimation itself: test_pac_estimate.py
    ("pac.estimate.iterative_ols('eq:pac', {'beta'}, 'x');", "Not enough input arguments"),
    ("equation.evaluate('eq:pac');", "equation package"),
    ("pac.expectations.evaluate(1);", "evaluation functions"),
])
def test_unsupported_functions_are_refused(tmp_path, call, text):
    (tmp_path / "pv1.mod").write_text((REF / "pv1.mod").read_text() + "\n" + call + "\n")
    with pytest.raises(pydyn.ResultsError, match=text):
        pydyn.dynare(tmp_path / "pv1.mod")


# ------------------------------------------------------------------------------------------------------ helpers
def test_a2alpha_by_hand():
    # a2alpha.m: alpha_m = a_m, alpha_i = a_i - a_{i+1} (1 < i < m), alpha_1 = a_1 - a_2 - 1
    np.testing.assert_allclose(pac.a2alpha([0.5, 0.2, -0.1]), [-0.7, 0.3, -0.1], rtol=0, atol=1e-16)
    with pytest.raises(pac.PacError):
        pac.a2alpha([0.5])


def test_build_G_checks_the_discount_factor():
    with pytest.raises(pac.PacError, match="discount"):
        pac.build_G([0.5, 0.2, 1.5])
    G, alpha, beta = pac.build_G([0.5, 0.2, 0.9])
    assert G.shape == (2, 2) and beta == 0.9 and alpha.shape == (2,)

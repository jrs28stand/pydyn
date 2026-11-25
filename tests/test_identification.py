"""``identification`` (order 1, prior_mc=1, advanced=0) against Dynare 7.2 under Octave 8.4.

reference/ident/<name>_identif.mat is the file Dynare writes into <name>/identification/.  pydyn must write the same
variables.  Fields that are rounding noise of exactly singular problems are not compared: ``cond`` of rank-deficient
matrices, the multicorrelations ``Mco``/``Pco`` of collinear columns, the arbitrary null-space basis ``V`` and the
null-space based ``indno`` of the Hessian."""
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn

REF = Path(__file__).resolve().parent.parent / "reference" / "ident"
CASES = ["BrockMirman", "fl1", "cgg_criteria_differ", "as2007_QT"]
NOISE = {"cond", "Mco", "Pco", "V"}
NOISE_PATH = {"ide_hess_point.indno"}


def _cmp(a, b, path, bad, tol):
    if hasattr(a, "_fieldnames"):
        assert set(a._fieldnames) == set(b._fieldnames), (path, a._fieldnames, b._fieldnames)
        for k in a._fieldnames:
            _cmp(getattr(a, k), getattr(b, k), f"{path}.{k}", bad, tol)
        return
    if path.rsplit(".", 1)[-1] in NOISE or path in NOISE_PATH:
        return
    a = a.toarray() if hasattr(a, "toarray") else a
    b = b.toarray() if hasattr(b, "toarray") else b
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == object:
        assert a.shape == b.shape, path
        for i, (x, y) in enumerate(zip(a.ravel(order="F"), b.ravel(order="F"))):
            _cmp(x, y, f"{path}{{{i}}}", bad, tol)
        return
    if a.dtype.kind in "US":
        return
    a, b = a.astype(float), b.astype(float)
    if a.shape != b.shape and a.size == b.size and 1 in a.shape and 1 in b.shape:
        b = b.reshape(a.shape)
    if a.shape != b.shape:
        bad.append(f"{path}: shape {a.shape} vs {b.shape}")
        return
    nan = np.isnan(a)
    if np.any(nan != np.isnan(b)):
        bad.append(f"{path}: NaN pattern")
        return
    a, b = np.where(nan, 0, a), np.where(nan, 0, b)
    scale = max(1.0, np.max(np.abs(a)) if a.size else 1.0)
    if a.size and np.max(np.abs(a - b)) > tol * scale:
        bad.append(f"{path}: maxdiff {np.max(np.abs(a - b)):.3g} (scale {scale:.3g})")


@pytest.mark.parametrize("case", CASES)
def test_against_octave(case, tmp_path):
    for f in list(REF.glob("*.inc")) + list(REF.glob("*_QT.mat")):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(REF / f"{case}.mod", tmp_path / f"{case}.mod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pydyn.dynare(tmp_path / f"{case}.mod")
    out = tmp_path / case / "identification"
    ref = loadmat(REF / f"{case}_identif.mat", struct_as_record=False)
    files = sorted(out.glob("*_identif.mat"))
    assert files, "no identification files written"
    assert (out / f"{case}_identif.mat").exists()
    for f in files:
        got = loadmat(f, struct_as_record=False)
        bad = []
        for k in ref:
            if k.startswith("__") or k == "store_options_ident":
                continue
            assert k in got, f"{f.name}: {k} missing"
            _cmp(ref[k][0, 0], got[k][0, 0], k, bad, 1e-8)
        assert not bad, f"{f.name}: " + "; ".join(bad)


REFUSE = [
    ("identification(order=2);", "order"),
    ("identification(prior_mc=5);", "prior_mc"),
    ("identification(advanced=1);", "advanced"),
    ("identification(analytic_derivation_mode=-1);", "analytic_derivation_mode"),
]


@pytest.mark.parametrize("stmt,word", REFUSE)
def test_refusals(stmt, word, tmp_path):
    lines = (REF / "as2007_QT.mod").read_text().split("\n")
    text = "\n".join(lines[: next(i for i, ln in enumerate(lines) if ln.startswith("identification("))])
    (tmp_path / "r.mod").write_text(text + "\n" + stmt + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as e:
            pydyn.dynare(tmp_path / "r.mod")
    assert word in str(e.value)

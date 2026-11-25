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
CASES = ["BrockMirman", "fl1", "cgg_criteria_differ", "as2007_QT", "forward_looking_varobs_x", "au5", "au6", "kim2",
         "fs2000_corr_pt"]
# (file name stem) -> paths whose values depend on a basis chosen arbitrarily by an SVD of a rank-deficient problem
BASIS_PATHS = {"fs2000_corr_pt": ("ide_minimal_point.dMINIMAL", "ide_minimal_point.norm_dMINIMAL")}
# kim2: the Hessian is singular to 1e-17 and Octave's A\\B then returns an LU solution of rounding noise
SINGULAR_PATHS = {"kim2": ("si_dMOMENTSnorm", "si_dREDUCEDFORMnorm", "si_dDYNAMICnorm")}
NOISE = {"cond", "Mco", "Pco", "V"}
NOISE_PATH = {"ide_hess_point.indno"}


def _cmp(a, b, path, bad, tol, skip=()):
    if hasattr(a, "_fieldnames"):
        assert set(a._fieldnames) == set(b._fieldnames), (path, a._fieldnames, b._fieldnames)
        for k in a._fieldnames:
            _cmp(getattr(a, k), getattr(b, k), f"{path}.{k}", bad, tol, skip)
        return
    if path.rsplit(".", 1)[-1] in NOISE or path in NOISE_PATH or path in skip or path.rsplit(".", 1)[-1] in skip:
        return
    a = a.toarray() if hasattr(a, "toarray") else a
    b = b.toarray() if hasattr(b, "toarray") else b
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == object:
        assert a.shape == b.shape, path
        for i, (x, y) in enumerate(zip(a.ravel(order="F"), b.ravel(order="F"))):
            _cmp(x, y, f"{path}{{{i}}}", bad, tol, skip)
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


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


@pytest.mark.parametrize("case", CASES)
def test_against_octave(case, tmp_path):
    for f in list(REF.glob("*.inc")) + list(REF.glob("*_QT.mat")):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(REF / f"{case}.mod", tmp_path / f"{case}.mod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pydyn.dynare(tmp_path / f"{case}.mod")
    out = tmp_path / case / "identification"
    skip = BASIS_PATHS.get(case, ()) + SINGULAR_PATHS.get(case, ())
    stems = [f"{case}_identif"] + [p.stem for p in sorted(REF.glob(f"{case}_identif_*.mat"))]
    for stem in stems:
        ref = loadmat(REF / f"{stem}.mat", struct_as_record=False)
        got = loadmat(out / f"{stem}.mat", struct_as_record=False)
        bad = []
        for k in ref:
            if k.startswith("__") or k == "store_options_ident":
                continue
            assert k in got, f"{stem}: {k} missing"
            if k.startswith("STO_"):
                a, b = _dense(ref[k]), _dense(got[k])
                assert a.shape == b.shape, (stem, k, a.shape, b.shape)
                if a.dtype != object:
                    assert np.max(np.abs(a - b)) <= 1e-7 * max(1.0, np.max(np.abs(a))), (stem, k)
                continue
            _cmp(ref[k][0, 0], got[k][0, 0], k, bad, 1e-7 if case in ("kim2", "au5", "au6") else 1e-8, skip)
        assert not bad, f"{stem}: " + "; ".join(bad)
    # the label file holds the point variables (Dynare writes it before the Monte Carlo ones are appended)
    labelled = [p for p in out.glob(f"{case}_*_identif.mat")]
    assert labelled
    if case == "fs2000_corr_pt":
        # a non-minimal system: dMINIMAL is only defined up to an orthogonal change of the basis of the uncontrollable
        # states; the singular values of its parameter block do not depend on it
        r = loadmat(REF / f"{stems[0]}.mat", struct_as_record=False)["ide_minimal_point"][0, 0].dMINIMAL
        g = loadmat(out / f"{stems[0]}.mat", struct_as_record=False)["ide_minimal_point"][0, 0].dMINIMAL
        npar = 4
        assert np.allclose(np.linalg.svd(r[:, :npar], compute_uv=False), np.linalg.svd(g[:, :npar], compute_uv=False),
                           rtol=1e-8, atol=1e-10)


REFUSE = [
    ("identification(no_identification_strength,order=2);", "order"),
    ("identification(no_identification_strength,prior_mc=5,checks_via_subsets=1);", "checks_via_subsets"),
    ("identification(no_identification_strength,analytic_derivation_mode=-1);", "analytic_derivation_mode"),
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

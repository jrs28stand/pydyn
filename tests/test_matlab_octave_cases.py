"""pydyn.matlab against Octave 8.4 on small MATLAB snippets (reference/matlab_octave_cases).

matlab_extra_*: the 376 cases of JDyn's review of pydyn's interpreter; pydyn_extra_*: pydyn's own
further cases; pydyn_nd_*: N-dimensional arrays; pydyn_rng_*: Octave's rand/randn/randi/randperm/rng (same
format; every case seeds the generators first); pydyn_blk1_*: Dynare's dates class (Octave with Dynare's dseries code,
reference/matlab_octave_cases/r_blk1.m), corr and the libm functions.  Each snippet sets ``r``; Octave's ``double(r)`` is stored (char results
as character codes, failures as 'ERROR: ...').  pydyn must either give Octave's value (an Octave
error must be a MatlabError) or refuse with Unsupported: never a different value, never a raw
Python exception.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pydyn import matlab as M

DIR = Path(__file__).resolve().parent.parent / "reference" / "matlab_octave_cases"
SETS = {"matlab_extra": ("matlab_extra_cases.txt", "matlab_extra.mat"),
        "pydyn_extra": ("pydyn_extra_cases.txt", "pydyn_extra.mat"),
        "pydyn_nd": ("pydyn_nd_cases.txt", "pydyn_nd.mat"),
        "pydyn_rng": ("pydyn_rng_cases.txt", "pydyn_rng.mat"),
        "pydyn_blk1": ("pydyn_blk1_cases.txt", "pydyn_blk1.mat"),
        "pydyn_fn": ("pydyn_fn_cases.txt", "pydyn_fn.mat"),
        "pydyn_mi": ("pydyn_mi_cases.txt", "pydyn_mi.mat"),
        "pydyn_gfx": ("pydyn_gfx_cases.txt", "pydyn_gfx.mat"),
        "pydyn_char": ("pydyn_char_cases.txt", "pydyn_char.mat")}


def _load(name):
    cases_file, mat_file = SETS[name]
    cases = (DIR / cases_file).read_text().split("\n%%\n")
    ref = loadmat(DIR / mat_file)["results"].ravel()
    assert len(cases) == len(ref)
    return cases, ref


DATA = {name: _load(name) for name in SETS}
PARAMS = [(name, k) for name in SETS for k in range(len(DATA[name][0]))]


def _unwrap(w):
    while isinstance(w, np.ndarray) and w.dtype == object and w.size == 1:
        w = w.ravel()[0]
    return w


def _text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind in "US":
        return "".join(x.ravel().tolist())
    return None


def _same(got, want):
    """Octave's double(r) against pydyn's r (char results compared by character code)."""
    if getattr(got, "obj_kind", None) == "char matrix":          # pydyn.mlchar.CharMat: by character code, with its shape
        w = np.atleast_2d(np.asarray(want, float))
        return got.a.shape == w.shape and bool(np.all(got.a == w)), f"{got.a!r} vs Octave {want!r}"
    gt = _text(got)
    if gt is not None:              # char results: by character code (pydyn's '' does not say 0x0 or 1x0)
        codes = [float(ord(c)) for c in gt]
        return codes == np.asarray(want, float).ravel().tolist(), f"{gt!r} vs Octave {want!r}"
    else:
        g = np.asarray(got)
        if g.dtype == object:
            return False, f"pydyn returned {got!r}"
        g = np.atleast_2d(g.astype(float))
    w = np.atleast_2d(np.asarray(want, float))
    if g.shape != w.shape and not (g.size == 1 and w.size == 1):
        return False, f"shape {g.shape} vs Octave {w.shape}"
    g, w = g.ravel(), w.ravel()
    with np.errstate(invalid="ignore"):
        ok = np.all((np.isnan(g) & np.isnan(w)) | (g == w) | (np.abs(g - w) <= 1e-12 + 1e-9 * np.abs(w)))
    return bool(ok), f"{g[:8]} vs Octave {w[:8]}"


@pytest.mark.parametrize("name,k", PARAMS, ids=[f"{n}-{k + 1}" for n, k in PARAMS])
def test_against_octave(name, k):
    cases, ref = DATA[name]
    code, want = cases[k], _unwrap(ref[k])
    wt = _text(want)
    octave_error = wt is not None and wt.startswith("ERROR")
    try:
        got = M.run(code)["r"]
    except M.Unsupported:
        return                                   # a refusal is acceptable
    except M.MatlabError as exc:
        assert octave_error, f"pydyn raised a MATLAB error ({exc}); Octave gives {want!r}\n{code}"
        return
    assert not octave_error, f"Octave fails ({wt}); pydyn returned {got!r}\n{code}"
    ok, why = _same(got, want)
    assert ok, f"{why}\n{code}"


def test_refusals_are_few():
    """The cases are meant to be answered: refusals stay the exception (60 of 376 at the time of the
    review, mostly functions pydyn does not provide)."""
    refused = 0
    for name, k in PARAMS:
        if name in ("pydyn_nd", "pydyn_rng", "pydyn_blk1"):
            continue
        try:
            M.run(DATA[name][0][k])
        except M.Unsupported:
            refused += 1
        except M.MatlabError:
            pass
    assert refused <= 70


def test_blk1_refusals_are_the_documented_ones():
    """pydyn_blk1 (dates/dseries, addpath-era elementary functions): 244 cases, of which exactly the cases Octave
    answers with int8/single types, complex results, unary minus or * on dates, unknown members, dates(1990) and
    arrayfun over dates are refused (12 when written); a refusal is never a wrong answer (test above)."""
    refused = 0
    for k in range(len(DATA["pydyn_blk1"][0])):
        try:
            M.run(DATA["pydyn_blk1"][0][k])
        except M.Unsupported:
            refused += 1
        except M.MatlabError:
            pass
    assert refused <= 12


def test_nd_refusals_are_few():
    """N-d cases: refusals only where MATLAB and Octave differ or for functions pydyn lacks."""
    refused = 0
    for k in range(len(DATA["pydyn_nd"][0])):
        try:
            M.run(DATA["pydyn_nd"][0][k])
        except M.Unsupported:
            refused += 1
        except M.MatlabError:
            pass
    assert refused <= 12


def test_rng_refusals_are_few():
    """Random-number cases: refusals only for Octave's old generators, entropy seeding and other classes."""
    refused = 0
    for k in range(len(DATA["pydyn_rng"][0])):
        try:
            M.run(DATA["pydyn_rng"][0][k])
        except M.Unsupported:
            refused += 1
        except M.MatlabError:
            pass
    assert refused <= 8


def test_refusal_is_not_caught_by_try():
    """A refusal must not be taken for a MATLAB error by the user's try/catch."""
    with pytest.raises(M.Unsupported):
        M.run("r = 0; try, x = magic(3); catch, r = 1; end")
    with pytest.raises(M.Unsupported):
        M.run("r = 0; try, x = 2; x(1, 2) = []; catch, r = 1; end")


def test_python_errors_become_matlab_errors():
    """numpy's exceptions inside a built-in are MATLAB errors: try/catch catches them."""
    out = M.run("r = 0; try, x = chol([1 2; 2 1]); catch err, r = 1; m = err.message; end")
    assert out["r"] == 1 and "chol" in out["m"]
    out = M.run("r = 0; try, x = 1:3; x(5) = []; catch err, r = 1; end")
    assert out["r"] == 1
    with pytest.raises(M.MatlabError):
        M.run("x = [1 2; 3 4]; x(3, :) = [];")


def test_error_identifier():
    out = M.run("try, error('pkg:sub:id', 'value %d', 3); catch e, m = e.message; i = e.identifier; end")
    assert out["m"] == "value 3" and out["i"] == "pkg:sub:id"
    out = M.run("try, try, error('a:b', 'x'); catch e, rethrow(e); end; catch e2, i = e2.identifier; end")
    assert out["i"] == "a:b"



# ---- file-system queries: the cases of pydyn_files_cases.txt run in a folder with f.txt and sub/g.mat (see r_pydyn_files.m)
def _mk(tmp_path, k):
    d = tmp_path / f"d{k}"
    d.mkdir()
    return d


def _files_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("x\n")
    (tmp_path / "sub" / "g.mat").write_text("y\n")
    return tmp_path


FILES_CASES = (DIR / "pydyn_files_cases.txt").read_text().split("\n%%\n")
FILES_REF = loadmat(DIR / "pydyn_files.mat")["results"].ravel()


@pytest.mark.parametrize("k", range(len(FILES_CASES)))
def test_files_against_octave(k, tmp_path):
    """isfile/isfolder/isdir/exist/fullfile/fileparts read the model's own folder only; a path that leaves it is refused."""
    code, want = FILES_CASES[k], _unwrap(FILES_REF[k])
    wt = _text(want)
    octave_error = wt is not None and wt.startswith("ERROR")
    try:
        got = M.run(code, directory=_files_dir(tmp_path))["r"]
    except M.Unsupported:
        return
    except M.MatlabError as exc:
        assert octave_error, f"pydyn raised a MATLAB error ({exc}); Octave gives {want!r}\n{code}"
        return
    assert not octave_error, f"Octave fails ({wt}); pydyn returned {got!r}\n{code}"
    ok, why = _same(got, want)
    assert ok, f"{why}\n{code}"


def test_files_refusals_are_the_documented_ones(tmp_path):
    """Only paths that leave the folder (../, absolute) are refused, and exist() without a file name or a 'dir'."""
    refused = []
    for k, code in enumerate(FILES_CASES):
        try:
            M.run(code, directory=_files_dir(_mk(tmp_path, k)))
        except M.Unsupported:
            refused.append(k + 1)
        except M.MatlabError:
            pass
    assert refused == [27, 28, 29, 30], refused

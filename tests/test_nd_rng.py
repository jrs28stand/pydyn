# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""N-d arrays, Octave's random numbers and the sandboxed ``save`` of the MATLAB interpreter and results mode.

The values of the interpreter are checked against Octave 8.4 in tests/test_matlab_octave_cases.py (sets
pydyn_nd and pydyn_rng) and against Dynare 7.2 under Octave in tests/test_results.py (rng_oo, nd_oo);
these tests cover what those cannot: Dynare's set_dynare_seed (checked by hand against Dynare 7.2's
set_dynare_seed_local_options.m under Octave 8.4), the refusals and the file sandbox."""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import matlab as M

REF = Path(__file__).resolve().parent.parent / "reference" / "oo"


def test_set_dynare_seed_is_both_octave_states():
    # Dynare 7.2 under Octave 8.4: set_dynare_seed(4); a = randn(1,2); b = rand(1,2); set_dynare_seed('reset');
    # c = randn(1,2); x = rand; set_dynare_seed('default'); d = [rand randn]; set_dynare_seed(17);
    # set_dynare_seed('reset'); e = [rand randn]
    out = M.run("set_dynare_seed(4); a = randn(1,2); b = rand(1,2); set_dynare_seed('reset'); c = randn(1,2); x = rand;"
                "set_dynare_seed('default'); d = [rand randn]; set_dynare_seed(17); set_dynare_seed('reset'); e = [rand randn];")
    np.testing.assert_allclose(out["a"].ravel(), [-1.49614796e+00, -7.05420226e-01], rtol=1e-8)
    np.testing.assert_allclose(out["b"].ravel(), [2.36048090e-01, 1.03166034e-01], rtol=1e-8)
    np.testing.assert_array_equal(out["c"], out["a"])
    np.testing.assert_allclose(out["d"].ravel(), [8.44421852e-01, -1.22483653e+00], rtol=1e-8)
    np.testing.assert_allclose(out["e"].ravel(), [5.21983910e-01, -1.42644106e-01], rtol=1e-8)
    with pytest.raises(M.MatlabError):
        M.run("set_dynare_seed(1.5);")
    with pytest.raises(M.Unsupported):
        M.run("set_dynare_seed('mt19937ar', 3);")


def test_default_stream_is_dynares_and_per_interpreter():
    a = M.run("x = rand(1, 3);")["x"]
    b = M.run("rand('state', 0); x = rand(1, 3);")["x"]
    np.testing.assert_array_equal(a, b)
    assert M.run("x = rand;")["x"] == a[0, 0]          # a new interpreter starts again from Dynare's seed


def test_random_numbers_are_refused_where_dynare_draws_at_each_call(tmp_path):
    (tmp_path / "f.m").write_text("function y = f(x)\ny = x + rand;\nend\n")
    fn = M.MatlabFunction(tmp_path / "f.m")
    with pytest.raises(M.Unsupported, match="global stream"):
        fn(1.0)
    with pytest.raises(M.Unsupported):
        M.run("rand('seed', 3);")
    with pytest.raises(M.Unsupported):
        M.run("x = gamrnd(1, 1);")


def test_nd_refusals_and_errors():
    with pytest.raises(M.Unsupported):
        M.run("a = zeros(2, 2, 2); b = inv(a);")             # matrix-only functions refuse N-d arrays
    with pytest.raises(M.Unsupported):
        M.run("a = zeros(2, 3, 4); a(:, 1) = [];")         # MATLAB and Octave delete differently
    with pytest.raises(M.MatlabError):
        M.run("a = zeros(2, 2, 2); b = a';")
    with pytest.raises(M.MatlabError):
        M.run("a = zeros(2, 3, 4); a(2, 15) = 1;")         # growth through fewer subscripts: ambiguous
    out = M.run("a = zeros(2, 3); a(5) = 7; b = a(:)';")     # linear scalar assignment into a matrix
    assert out["b"].tolist() == [[0, 0, 0, 0, 7, 0]]


def _copy(tmp_path, name, extra=()):
    shutil.copy(REF / f"{name}.mod", tmp_path / f"{name}.mod")
    for f in extra:
        shutil.copy(REF / f, tmp_path / f)
    return tmp_path / f"{name}.mod"


SAVE_MOD = """var y; varexo e; parameters rho; rho = 0.5;
model; y = rho*y(-1) + e; end;
shocks; var e; stderr 0.1; end;
steady;
stoch_simul(order=1, irf=0, periods=30, drop=0, nograph);
w = oo_.endo_simul'; c = {'a', [1 2]}; s.f = cat(3, 1, 2);
save data_s.mat w c s;
q = load('data_s');
"""


def test_save_writes_a_mat_file_in_the_model_folder(tmp_path):
    (tmp_path / "sv.mod").write_text(SAVE_MOD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pydyn.dynare(tmp_path / "sv.mod")
    d = loadmat(tmp_path / "data_s.mat")
    np.testing.assert_array_equal(d["w"], r.vars["w"])
    assert d["s"]["f"][0, 0].shape == (1, 1, 2)
    np.testing.assert_array_equal(r.vars["q"]["w"], r.vars["w"])
    rec = json.loads((tmp_path / "sv" / "pydyn_written_files.json").read_text())
    assert set(rec) == {"data_s.mat"}
    with warnings.catch_warnings():                       # pydyn's own file may be written again
        warnings.simplefilter("ignore")
        pydyn.dynare(tmp_path / "sv.mod")


def test_save_never_overwrites_other_files_or_leaves_the_folder(tmp_path):
    (tmp_path / "data_s.mat").write_bytes(b"not pydyn's")
    (tmp_path / "sv.mod").write_text(SAVE_MOD)
    with pytest.raises(pydyn.ResultsError, match="not overwritten"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pydyn.dynare(tmp_path / "sv.mod")
    assert (tmp_path / "data_s.mat").read_bytes() == b"not pydyn's"
    for line in ("save ../x.mat w;", "save('/tmp/x.mat', 'w');", "save x.txt w;", "save -ascii x.mat w;"):
        d = tmp_path / str(abs(hash(line)))
        d.mkdir()
        (d / "sv.mod").write_text(SAVE_MOD.replace("save data_s.mat w c s;\nq = load('data_s');", line))
        with pytest.raises(pydyn.ResultsError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pydyn.dynare(d / "sv.mod")
        assert not (tmp_path / "x.mat").exists() and not (d / "x.txt").exists() and not (d / "x.mat").exists()


def _run_mod(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.dynare(path)


_TINY = "var y; varexo e; parameters rho; rho = 0.5;\nmodel; y = rho*y(-1) + e; end;\n"


def test_functions_behind_addpath_run_and_write_next_to_the_mod_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "gen.m").write_text("function gen(n)\nx = (1:n)' * 2;\nsave('gen_out.mat', 'x');\nend\n")
    (tmp_path / "sub" / "a.mod").write_text(_TINY + "addpath('..');\ngen(3);\n")
    _run_mod(tmp_path / "sub" / "a.mod")
    assert (tmp_path / "sub" / "gen_out.mat").exists() and not (tmp_path / "gen_out.mat").exists()


def test_addpath_refuses_to_overwrite_a_file_pydyn_did_not_write(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "gen.m").write_text("function gen(n)\nx = n;\nsave('gen_out.mat', 'x');\nend\n")
    (tmp_path / "sub" / "gen_out.mat").write_bytes(b"precious")
    (tmp_path / "sub" / "a.mod").write_text(_TINY + "addpath('..');\ngen(3);\n")
    with pytest.raises(pydyn.ResultsError):
        _run_mod(tmp_path / "sub" / "a.mod")
    assert (tmp_path / "sub" / "gen_out.mat").read_bytes() == b"precious"


def test_addpath_refuses_a_file_that_shadows_a_builtin(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "exp.m").write_text("function y = exp(x)\ny = 0;\nend\n")
    (tmp_path / "sub" / "a.mod").write_text(_TINY + "addpath('..');\nz = exp(1);\n")
    with pytest.raises(pydyn.ResultsError):
        _run_mod(tmp_path / "sub" / "a.mod")


def _build(tmp_path, text, name="m"):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_matlab import build
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(tmp_path, text, name=name)


def test_model_loading_draws_from_dynares_stream(tmp_path):
    base = "var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\n"
    m = _build(tmp_path, base + "rho = 0.1 + 0.5*rand;\n")
    assert m.get_param("rho") == 0.1 + 0.5 * M.run("x = rand;")["x"]
    # code that was skipped may have drawn in Dynare: later draws are refused, until both generators are seeded
    with pytest.raises(pydyn.ModelError, match="random stream is unknown"):
        _build(tmp_path, base + "z = some_unknown_function(1);\nrho = rand;\n", name="m2")
    m = _build(tmp_path, base + "z = some_unknown_function(1);\nset_dynare_seed(3);\nrho = rand;\n", name="m3")
    assert m.get_param("rho") == M.run("set_dynare_seed(3); x = rand;")["x"]

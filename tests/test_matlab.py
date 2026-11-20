"""pydyn.matlab: MATLAB code in .mod files, steady-state files and helper functions.

Expected values come from Octave 8.4 and Dynare 7.2 (reference/r_matlab.m, tools/sweep_matlab.py):
- reference/mat_semantics.mat: value of ``r`` after each snippet of reference/mat_cases.txt;
- reference/mat_flow.mat: M_.params, M_.Sigma_e and workspace variables after ``dynare mat_flow``;
- reference/mat_corpus.mat: M_.params, M_.Sigma_e and oo_.steady_state at the end of Dynare's run of
  .mod files of the Dynare test suite (the files themselves are read from the Dynare source tree).
"""
import importlib.util
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import matlab as M

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
DYNARE_SRC = Path(os.environ.get("PYDYN_DYNARE_SRC", str(EXAMPLES.parent)))
SS = EXAMPLES / "stochastic_simulations"


def run(code, **variables):
    with np.errstate(invalid="raise", divide="ignore", over="ignore"):
        return M.run(code, variables)


def build(tmp_path, text, name="m", files=None, **kw):
    for fname, content in (files or {}).items():
        (tmp_path / fname).write_text(content)
    f = tmp_path / f"{name}.mod"
    f.write_text(text)
    return pydyn.Model(pydyn.run_preprocessor(f), **kw)


def as_matrix(v):
    if isinstance(v, str):
        return np.array([[float(ord(c)) for c in v]]) if v else np.zeros((0, 0))
    return np.atleast_2d(np.asarray(v, float))


# ------------------------------------------------------------------ (a) semantics against Octave
CASES = (REF / "mat_cases.txt").read_text().split("\n%%\n")
EXPECTED = loadmat(REF / "mat_semantics.mat")["results"].ravel()


def test_case_file_and_reference_agree():
    assert len(CASES) == len(EXPECTED) >= 60
    assert all(e.dtype.kind not in "US" for e in EXPECTED), "an Octave reference case failed"


@pytest.mark.parametrize("k", range(len(CASES)))
def test_semantics_against_octave(k):
    got = as_matrix(run(CASES[k])["r"])
    want = np.atleast_2d(EXPECTED[k].astype(float))
    assert got.shape == want.shape, CASES[k]
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0, equal_nan=True, err_msg=CASES[k])


def test_matlab_specific_behaviour():
    """Points where MATLAB and Octave differ: pydyn follows MATLAB (values from the MATLAB documentation)."""
    assert run("r = sprintf('%d', 1.5);")["r"] == "1.500000e+00"       # Octave prints 1.5
    assert run("r = isoctave;")["r"] == False                           # noqa: E712
    with pytest.raises(M.Unsupported):                                  # "..." strings are string objects in MATLAB
        run('r = ["a" "b"];')


@pytest.mark.parametrize("code", ["r = sqrt(-1);", "r = (-8)^(1/3);", "r = log(-1);", "r = 0/0;", "r = Inf - Inf;",
                                  "r = acos(2);"])
def test_complex_or_indeterminate_results_are_errors(code):
    """Complex numbers are outside the subset: such operations raise instead of returning NaN."""
    with pytest.raises(M.MATLAB_ERRORS):
        run(code)


@pytest.mark.parametrize("code,exc", [
    ("a = [1 2 3]; r = a(4);", M.MatlabError), ("a = [1 2 3]; r = a(0);", M.MatlabError),
    ("a = [1 2 3]; r = a(1.5);", M.MatlabError), ("r = [1 2] + [1 2 3];", M.MatlabError),
    ("r = [1 2; 3];", M.MatlabError), ("r = [1 2 3] * [1 2 3];", M.MatlabError),
    ("r = undefined_thing + 1;", M.Unsupported), ("r = rand(2);", M.Unsupported),
    ("r = 1i;", M.Unsupported), ("x = dseries(1);", M.Unsupported), ("r = evalin('base', '1');", M.Unsupported),
    ("r = system('ls');", M.Unsupported), ("fid = fopen('x.txt', 'w');", M.Unsupported),
    ("s = struct('a', {1, 2});", M.Unsupported), ("a = zeros(2, 2, 2);", M.Unsupported),
    ("if [1 2] && 1, r = 1; end", M.MatlabError), ("error('my:id', 'value %d', 4)", M.MatlabError),
])
def test_errors_and_refusals(code, exc):
    with pytest.raises(exc):
        run(code)


def test_no_access_to_python_from_matlab_text():
    """Names resolve only against MATLAB variables and the built-in table."""
    for code in ["r = __import__('os');", "r = open('/etc/passwd');", "r = eval('__import__(''os'').getcwd()');",
                 "x.__class__ = 1;", "r = getattr(1, 'real');", "r = exec('1');", "r = globals();"]:
        with pytest.raises((M.Unsupported, M.MatlabError, M.ParseError)):
            run(code)


def test_endless_loop_is_stopped(monkeypatch):
    monkeypatch.setattr(M, "MAX_STEPS", 5000)
    with pytest.raises(M.Unsupported, match="step limit"):
        run("k = 0; while true, k = k + 1; end")


def test_function_files_and_scripts(tmp_path):
    (tmp_path / "addmul.m").write_text("function [s, p] = addmul(a, b)\n% help text\ns = a + b;\nif nargout > 1\n  p = helper(a) * b;\nend\nend\n"
                                       "function y = helper(x)\n  y = x;\nend\n")
    (tmp_path / "setup_script.m").write_text("zz = 4;\nww = zz * 2;\n")
    with np.errstate(invalid="raise", divide="ignore", over="ignore"):
        ws = M.run("[s, p] = addmul(2, 3); t = addmul(1, 1); setup_script; q = ww + zz;", directory=tmp_path)
    assert (ws["s"], ws["p"], ws["t"], ws["q"]) == (5.0, 6.0, 2.0, 12.0)


# ------------------------------------------------------------------ (b) .mod files
def test_control_flow_with_dynare_statements_against_dynare(tmp_path):
    """if/for/while/switch/try around parameter assignments, set_param_value, M_.params(i), M_.Sigma_e,
    function handles, fzero: same calibration as Dynare (reference/mat_flow.mod, run under Octave)."""
    shutil.copy(REF / "mat_flow.mod", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # nothing is skipped in this file
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "mat_flow.mod"))
    r = loadmat(REF / "mat_flow.mat", squeeze_me=True)
    assert m.param_names == r["param_names"].split(",")
    np.testing.assert_allclose(m.params, r["params"], rtol=1e-12)
    np.testing.assert_allclose(m.Sigma_e, r["Sigma_e"], rtol=1e-12)
    assert m.get_param("s") == 5.0 and m._matlab.sc.vars["s"] == 99.0      # one-line if: workspace only
    for name in ("x1", "x2", "x3", "k0"):
        np.testing.assert_allclose(m.workspace[name], r[name], rtol=1e-12)
    np.testing.assert_allclose(np.ravel(m._matlab.sc.vars["w"]), r["w"], rtol=1e-12)
    assert m.matlab_options["irf"] == 7 and m.matlab_skipped == []
    # the stale workspace copy: b1 was changed through M_.params(6), the MATLAB variable b1 was not
    assert m.workspace["x1"] == m.workspace["x2"] != m.get_param("b1")


FLOW = ("var c; varexo e; parameters rho s; rho=0.5; s=2;\ncalib = {calib};\nif calib == 1\n rho = 0.9; s = 5;\nend\n"
        "model; c = rho*c(-1) + s*e; end;")


@pytest.mark.parametrize("calib,rho,s", [(0, 0.5, 2.0), (1, 0.9, 5.0)])
def test_matlab_control_flow_is_executed(tmp_path, calib, rho, s):
    """Replaces tests/test_review.py::test_matlab_control_flow_is_an_error: the block is now executed
    (Dynare gives rho=0.5, s=2 for calib=0 and rho=0.9, s=5 for calib=1)."""
    m = build(tmp_path, FLOW.format(calib=calib))
    assert (m.get_param("rho"), m.get_param("s")) == (rho, s)


def test_block_with_unknown_condition_around_parameters_is_an_error(tmp_path):
    with pytest.raises(pydyn.ModelError, match="param_init"):
        build(tmp_path, FLOW.format(calib="some_function_pydyn_does_not_have(3)"))


def test_unknown_condition_without_calibration_is_skipped_with_one_warning(tmp_path):
    text = ("var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\nshocks; var e; stderr 1; end;\n"
            "stoch_simul(order=1, irf=5);\nfigure; plot(c_e); title('irf');\n"
            "if max(abs(oo_.dr.ghx)) > 1\n  error('unstable');\nend\nx = oo_.steady_state(1) + 1;\ny = 2*x;\ndisp(y)\n")
    with pytest.warns(UserWarning, match="statement") as rec:
        m = build(tmp_path, text)
    assert sum("MATLAB code" in str(w.message) for w in rec) == 1
    kinds = [k for _, k, _ in m.matlab_skipped]
    assert "plot" in kinds and "block" in kinds and "results" in kinds and "tainted" in kinds
    assert m.get_param("rho") == 0.5 and "y" not in m.workspace


@pytest.mark.parametrize("line,match", [
    ("M_.params(1) = oo_.steady_state(1);", "assigns to M_"),
    ("set_param_value('rho', some_unknown_function(1));", "set_param_value"),
    ("M_.Sigma_e = cov(randn(10, 1));", "assigns to M_"),
    ("M_.H = 1;", "assigns to M_"),
    ("oo_.steady_state(1) = 3;", "oo_.steady_state"),
    ("eval(some_unknown_function(1));", "eval"),
    ("x = some_unknown_function(1);\nrho = x;", "could not compute"),
    ("x = rand;\nshocks; var e; stderr x; end;", "could not compute"),
    ("error('calibration is inconsistent');", "calibration is inconsistent"),
    ("M_.params(5) = 1;", "must not change size|out of bound"),
    ("load_params_from_somewhere;\nif exist('q', 'var')\n rho = 0.1;\nend", "param_init"),
])
def test_code_that_changes_the_model_and_cannot_be_followed_is_an_error(tmp_path, line, match):
    text = ("var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\n" + line + "\n")
    with pytest.raises(pydyn.ModelError, match=match):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(tmp_path, text)


def test_parameters_changed_by_the_steady_state_are_unknown_afterwards(tmp_path):
    """After `steady`, Dynare's M_.params holds values set by the steady_state_model block; pydyn does not
    compute the steady state while loading, so MATLAB code reading such a parameter is refused."""
    base = ("var c; varexo e; parameters rho cbar; rho=0.5;\nmodel; c = cbar + rho*(c(-1)-cbar) + e; end;\n"
            "steady_state_model; cbar = 2*rho; c = cbar; end;\nsteady;\n")
    with pytest.raises(pydyn.ModelError, match="assigns to M_|set_param_value"):
        build(tmp_path, base + "set_param_value('rho', get_param_by_name('cbar')/4);\n")
    m = build(tmp_path, base + "set_param_value('rho', get_param_by_name('rho')/2);\n")       # rho is not touched by the block
    assert m.get_param("rho") == 0.25


def test_try_block_with_unknown_statement_is_not_guessed(tmp_path):
    """A statement pydyn cannot run inside try might have raised in MATLAB: the block is undecided."""
    text = ("var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\nflag = 0;\n"
            "try\n  some_unknown_function(1);\n  flag = 1;\ncatch\n  flag = 2;\nend\n")
    with pytest.warns(UserWarning):
        m = build(tmp_path, text)
    assert "flag" not in m.workspace
    with pytest.raises(pydyn.ModelError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(tmp_path, text + "rho = flag;\n", name="m2")


def test_load_of_mat_file_next_to_the_mod_file(tmp_path):
    from scipy.io import savemat
    savemat(tmp_path / "calib.mat", {"rho_data": 0.7, "vec": np.array([[1.0, 2.0, 3.0]])})
    m = build(tmp_path, "var c; varexo e; parameters rho s;\nload calib\nrho = rho_data; s = vec(3);\n"
                        "model; c = rho*c(-1) + s*e; end;")
    assert (m.get_param("rho"), m.get_param("s")) == (0.7, 3.0)
    with pytest.raises(pydyn.ModelError):                 # a file that does not exist yet: the variables are unknown
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(tmp_path, "var c; varexo e; parameters rho;\nload not_there\nrho = rho_data;\n"
                            "model; c = rho*c(-1) + e; end;", name="m2")


def test_script_and_function_next_to_the_mod_file(tmp_path):
    files = {"calibration.m": "rho_s = 0.3;\nset_param_value('s', 4);\n",
             "twice.m": "function y = twice(x)\ny = 2*x;\n"}
    m = build(tmp_path, "var c; varexo e; parameters rho s; s = 1;\ncalibration;\nrho = twice(rho_s);\n"
                        "model; c = rho*c(-1) + s*e; end;", files=files)
    assert (m.get_param("rho"), m.get_param("s")) == (0.6, 4.0)


# ------------------------------------------------------------------ steady-state files
def load(mod, tmp_path, extra=(), **kw):
    for f in (mod, *extra):
        shutil.copy(f, tmp_path)
    return pydyn.Model(pydyn.run_preprocessor(tmp_path / Path(mod).name), **kw)


def nk(tmp_path, **kw):
    return load(SS / "nk_baseline.mod", tmp_path, [SS / "nk_baseline_steadystate.m"], **kw)


def test_nk_baseline_with_its_matlab_steady_state_file(tmp_path):
    """(c) No Python function supplied: the .m file is used automatically (and translated)."""
    m = nk(tmp_path)
    r = loadmat(REF / "clean_nk_baseline.mat", squeeze_me=True)
    sol = pydyn.first_order(m)
    assert m._steady_state_file_fn.mode == "compiled"
    np.testing.assert_allclose(sol.ys, r["ys"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol.params, r["params"], rtol=1e-10)
    np.testing.assert_allclose(sol.ghx, r["ghx"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sol.ghu, r["ghu"], rtol=1e-6, atol=1e-8)
    m.set_param("betta", 0.995)                           # the file updates Rbar
    sol_b = pydyn.first_order(m)
    np.testing.assert_allclose(sol_b.params, r["params_b"], rtol=1e-10)
    np.testing.assert_allclose(sol_b.ys, r["ys_b"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol_b.ghx, r["ghx_b"], rtol=1e-6, atol=1e-8)


def test_steady_state_file_replaces_the_old_error(tmp_path):
    """Replaces tests/test_review.py::test_steady_state_file_is_reported (which expected SteadyStateError
    because .m files could not be run)."""
    m = nk(tmp_path)
    ys, params = pydyn.steady_state(m)
    assert np.max(np.abs(m.static_resid(ys, params=params))) < 1e-10
    assert np.isnan(m.get_param("Rbar")) and np.isfinite(params[m._ip["Rbar"]])


def test_steady_state_file_failure_flag(tmp_path):
    m = nk(tmp_path)
    m.set_param("betta", 50.0)                            # gammma1 < 0: the file sets check = 1 and returns
    with pytest.raises(pydyn.SteadyStateError, match="signalled failure"):
        pydyn.steady_state(m)


def test_interpreted_and_translated_steady_state_agree(tmp_path):
    m = nk(tmp_path)
    fn = pydyn.steady.steady_state_file_fn(m)
    x = np.zeros(m.n_exo)
    ys_c, p_c = fn(m, m.params.copy(), x)
    assert fn.mode == "compiled"
    with np.errstate(invalid="raise", divide="ignore", over="ignore"):
        ys_i, p_i = fn._interpret(m, m.params.copy(), x)
    np.testing.assert_allclose(ys_i, ys_c, rtol=1e-13)
    np.testing.assert_allclose(p_i, p_c, rtol=1e-13)
    hand = _hand_port()
    ys_h, p_h = hand(m, m.params.copy(), x)
    np.testing.assert_allclose(ys_c[:len(ys_h)], ys_h, rtol=1e-10)
    np.testing.assert_allclose(p_c, p_h, rtol=1e-12)


def _hand_port():
    spec = importlib.util.spec_from_file_location("nk_ss", ROOT / "examples" / "nk_baseline_steadystate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.steady_state


def test_translate_steady_state_gives_importable_python(tmp_path):
    src = M.translate_steady_state(SS / "nk_baseline_steadystate.m")
    assert "def steady_state(model, params_in, exo_in):" in src and "lambda_ = " in src and "eval(" not in src
    f = tmp_path / "nk_translated.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("nk_translated", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # a normal module: "from pydyn.matlab import rt"
    m = nk(tmp_path, steady_state_fn=mod.steady_state)
    r = loadmat(REF / "clean_nk_baseline.mat", squeeze_me=True)
    ys, params = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, r["ys"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(params, r["params"], rtol=1e-10)


def test_user_functions_take_precedence(tmp_path):
    calls = []

    def fn(model, params, exo):
        calls.append(1)
        return _hand_port()(model, params, exo)
    m = nk(tmp_path, steady_state_fn=fn)
    pydyn.steady_state(m)
    assert calls and getattr(m, "_steady_state_file_fn", None) is None
    sub = tmp_path / "c"
    sub.mkdir()
    m2 = load(SS / "collard_2001_analytical_steady_state.mod", sub,
              [SS / "collard_2001_analytical_steady_state_helper.m"],
              functions={"collard_2001_analytical_steady_state_helper": lambda *a: 0.25})
    assert m2.matlab_functions == {}
    ys, _ = pydyn.steady_state(m2, check=False)
    assert ys[m2._ie["h"]] == 0.25


def test_helper_function_file_in_steady_state_model(tmp_path):
    """(c) collard_2001_analytical_steady_state with its original .m helper (fsolve inside)."""
    m = load(SS / "collard_2001_analytical_steady_state.mod", tmp_path,
             [SS / "collard_2001_analytical_steady_state_helper.m"])
    assert list(m.matlab_functions) == ["collard_2001_analytical_steady_state_helper"]
    r = loadmat(REF / "clean_collard_2001_analytical_steady_state.mat", squeeze_me=True)
    sol = pydyn.first_order(m)
    np.testing.assert_allclose(m.Sigma_e, r["Sigma_e"], rtol=1e-12)
    # Dynare's fsolve stops at TolFun = 1e-10; pydyn solves the helper's equation to machine precision
    np.testing.assert_allclose(sol.ys, r["ys"], rtol=1e-7, atol=1e-9)
    at_ref = pydyn.first_order(m, ys=r["ys"].astype(float), params=r["params"].astype(float))
    np.testing.assert_allclose(at_ref.ghx, r["ghx"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(at_ref.ghu, r["ghu"], rtol=1e-9, atol=1e-11)


SSMODEL = ("var c k; varexo e; parameters rho cbar kbar;\nrho = 0.5; cbar = 2;\n"
           "model; c = cbar + rho*(c(-1)-cbar) + e; k = kbar + rho*(k(-1)-kbar); end;\n")


def test_helper_with_several_outputs(tmp_path):
    files = {"two_out.m": "function [a, b, info] = two_out(x)\ninfo = 0;\na = x;\nif x > 1\n  b = 3*x;\nelse\n  b = -1;\nend\n"}
    m = build(tmp_path, SSMODEL + "steady_state_model;\n[c, kk, info] = two_out(cbar);\nkbar = kk;\nk = kbar;\nend;\n", files=files)
    ys, params = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, [2.0, 6.0])
    assert params[m._ip["kbar"]] == 6.0 and m.matlab_functions["two_out"].mode == "compiled"


OLD_STYLE = """function [ys, check] = old_steadystate(ys, exo)
% steady-state file in the old two-output form: parameters come from the global M_
global M_
check = 0;
rho = M_.params(1);
cbar = M_.params(2);
if rho >= 1
    check = 1;
    return
end
M_.params(3) = 3*cbar;
ys = [cbar; M_.params(3)];
"""


def test_old_two_output_steady_state_file(tmp_path):
    """The two-output form does not occur in the Dynare 7.2 corpus; this is a hand-written example."""
    m = build(tmp_path, SSMODEL, name="old", files={"old_steadystate.m": OLD_STYLE})
    ys, params = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, [2.0, 6.0])
    assert params[m._ip["kbar"]] == 6.0 and m._steady_state_file_fn.mode == "interpreted"
    m.set_param("rho", 1.5)
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(m)


def test_steady_state_file_that_cannot_be_interpreted_is_reported(tmp_path):
    bad = "function [ys, params, check] = bad_steadystate(ys, exo, M_, options_)\nparams = M_.params;\ncheck = 0;\nys = some_toolbox_function(ys);\n"
    m = build(tmp_path, SSMODEL, name="bad", files={"bad_steadystate.m": bad})
    with pytest.raises(pydyn.SteadyStateError, match="bad_steadystate.m cannot be interpreted"):
        pydyn.steady_state(m)


def test_steady_state_timing(tmp_path, capsys):
    """Per-call cost of the automatic path against the hand-written port (recorded, loosely bounded)."""
    m = nk(tmp_path)
    fn, hand, x = pydyn.steady.steady_state_file_fn(m), _hand_port(), np.zeros(m.n_exo)

    def bench(f, n):
        f(m, m.params.copy(), x)
        t = time.perf_counter()
        for _ in range(n):
            f(m, m.params.copy(), x)
        return (time.perf_counter() - t) / n
    auto, port = bench(fn, 100), bench(hand, 100)
    with np.errstate(invalid="raise", divide="ignore", over="ignore"):
        interp = bench(fn._interpret, 5)
    with capsys.disabled():
        print(f"\n  nk_baseline steady state per call: translated {auto * 1e6:.0f} us, hand-written port {port * 1e6:.0f} us, "
              f"interpreted {interp * 1e6:.0f} us")
    assert auto < 5 * port and auto < interp


# ------------------------------------------------------------------ (b) corpus files against Dynare
CORPUS = loadmat(REF / "mat_corpus.mat", squeeze_me=True, struct_as_record=False) if (REF / "mat_corpus.mat").exists() else {}
CORPUS_FILES = [str(f) for f in np.atleast_1d(CORPUS["files"])] if CORPUS else []


@pytest.mark.skipif(not CORPUS_FILES, reason="reference/mat_corpus.mat missing")
def test_corpus_reference_is_large_enough():
    assert len(CORPUS_FILES) >= 40


@pytest.mark.parametrize("k", range(len(CORPUS_FILES)))
def test_corpus_calibration_against_dynare(k, tmp_path):
    """Parameters and shock covariance after loading equal Dynare's M_ at the end of its run (rtol 1e-10);
    the steady state is compared for the files flagged in the reference."""
    rel = CORPUS_FILES[k]
    src = DYNARE_SRC / rel
    if not src.exists():
        pytest.skip("Dynare source tree not available")
    for f in src.parent.iterdir():
        if f.is_file() and f.stat().st_size < 3_000_000:
            shutil.copy(f, tmp_path)
    M.ISOCTAVE = True                                     # the reference run used Octave
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pydyn.Model(pydyn.run_preprocessor(tmp_path / src.name))
            ref = getattr(CORPUS["refs"], f"f{k}")
            params = m.params
            if int(ref.after_steady):
                params = pydyn.steady_state(m, check=False)[1]
            np.testing.assert_allclose(params, np.atleast_1d(ref.params), rtol=1e-10, atol=0, equal_nan=True)
            np.testing.assert_allclose(m.Sigma_e, np.reshape(ref.Sigma_e, m.Sigma_e.shape), rtol=1e-10, atol=0)
            if int(ref.check_ys):
                ys, _ = pydyn.steady_state(m)
                # rtol: Dynare's own steady-state solver stops at 1e-6 on the residuals for numerical steady states
                np.testing.assert_allclose(ys, np.atleast_1d(ref.ys), rtol=float(ref.ys_rtol), atol=float(ref.ys_rtol))
    finally:
        M.ISOCTAVE = False

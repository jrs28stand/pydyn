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


@pytest.mark.parametrize("code", ["r = sqrt(-1);", "r = (-8)^(1/3);", "r = log(-1);", "r = acos(2);", "r = [4 -4].^0.5;"])
def test_complex_results_are_refused_not_matlab_errors(code):
    """Complex numbers are outside the subset.  An operation whose MATLAB result is complex is refused
    (Unsupported): it is not a MATLAB error, so a MATLAB try/catch does not catch it (review A6/B4;
    this replaces test_complex_or_indeterminate_results_are_errors, which expected a MATLAB error
    also for 0/0)."""
    with pytest.raises(M.Unsupported, match="complex"):
        run(code)
    with pytest.raises(M.Unsupported, match="complex"):
        run("try, " + code + " catch, r = 5; end")


def test_indeterminate_results_are_nan_as_in_matlab():
    """0/0, Inf-Inf, 0*Inf, sin(Inf) are NaN (no error), 1/0 is Inf, 10^400 is Inf (Octave values)."""
    ws = run("a = 0/0; b = Inf - Inf; c = 0*Inf; d = sin(Inf); e1 = 1/0; f = 10^400; g = -1/0; h = mod(5, 0);")
    assert all(np.isnan(ws[k]) for k in "abcd") and ws["e1"] == np.inf and ws["f"] == np.inf and ws["g"] == -np.inf
    assert run("try, x = 0/0; k = 5; catch, k = 6; end")["k"] == 5


@pytest.mark.parametrize("code,exc", [
    ("a = [1 2 3]; r = a(4);", M.MatlabError), ("a = [1 2 3]; r = a(0);", M.MatlabError),
    ("a = [1 2 3]; r = a(1.5);", M.MatlabError), ("r = [1 2] + [1 2 3];", M.MatlabError),
    ("r = [1 2; 3];", M.MatlabError), ("r = [1 2 3] * [1 2 3];", M.MatlabError),
    ("r = undefined_thing + 1;", M.Unsupported), ("r = gamrnd(2, 1);", M.Unsupported),
    ("r = 1i;", M.Unsupported), ("x = dseries(1);", M.Unsupported), ("r = evalin('base', '1');", M.Unsupported),
    ("r = system('ls');", M.Unsupported), ("fid = fopen('x.txt', 'w');", M.Unsupported),
    ("s = struct('a', {1, 2});", M.Unsupported), ("a = zeros(2, 2, 2); b = inv(a);", M.Unsupported),
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
        # ... but options_.irf = 7 is written by MATLAB code and nothing acts on it: reported (UnsupportedOptionWarning)
        with pytest.warns(pydyn.options.UnsupportedOptionWarning, match=r"options_\.irf"):
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
            "if max(abs(oo_.dr.ghx)) > 1\n  error('unstable');\nend\nx = oo_.dr.ys(1) + 1;\ny = 2*x;\ndisp(y)\n")
    with pytest.warns(UserWarning, match="statement") as rec:
        m = build(tmp_path, text)
    assert sum("MATLAB code" in str(w.message) for w in rec) == 1
    kinds = [k for _, k, _ in m.matlab_skipped]
    assert "plot" in kinds and "block" in kinds and "results" in kinds and "tainted" in kinds
    assert m.get_param("rho") == 0.5 and "y" not in m.workspace


@pytest.mark.parametrize("line,match", [
    ("M_.params(1) = oo_.dr.ghx(1);", "assigns to M_"),
    ("set_param_value('rho', some_unknown_function(1));", "set_param_value"),
    ("M_.Sigma_e = cov(randn(10, 1));", "assigns to M_"),
    ("M_.H = 1;", "assigns to M_"),
    ("oo_.steady_state(1) = oo_.dr.ys(1);\nsteady;", "oo_.steady_state"),   # computable writes are followed (see below)
    ("eval(some_unknown_function(1));", "eval"),
    ("x = some_unknown_function(1);\nrho = x;", "could not compute"),
    ("x = gamrnd(1, 1);\nshocks; var e; stderr x; end;", "could not compute"),
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


def test_parameters_set_by_the_steady_state_are_seen_by_later_statements(tmp_path):
    """After `steady`, Dynare's M_.params holds the values set by the steady_state_model block; pydyn
    computes them at that statement, so later MATLAB code and Dynare statements read them."""
    base = ("var c; varexo e; parameters rho cbar; rho=0.5;\nmodel; c = cbar + rho*(c(-1)-cbar) + e; end;\n"
            "steady_state_model; cbar = 2*rho; c = cbar; end;\nsteady;\n")
    m = build(tmp_path, base + "set_param_value('rho', get_param_by_name('cbar')/4);\n")
    assert (m.get_param("rho"), m.get_param("cbar")) == (0.25, 1.0)
    m = build(tmp_path, base + "set_param_value('rho', get_param_by_name('rho')/2);\n", name="m2")
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
    # like Dynare's M_.params after `steady`, the loaded parameters include those the file sets
    assert m.get_param("Rbar") == params[m._ip["Rbar"]] and np.isfinite(m.get_param("Rbar"))


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


def test_old_two_output_steady_state_file_is_an_error_as_in_dynare(tmp_path):
    """Review A11: Dynare 7.2 calls [ys, params, info] = f(ys, exo, M_, options_); a file with the old
    [ys, check] = f(ys, exo) signature fails there ("called with too many inputs"), and here too."""
    m = build(tmp_path, SSMODEL, name="old", files={"old_steadystate.m": OLD_STYLE})
    with pytest.raises(pydyn.SteadyStateError, match="too many"):
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


# ------------------------------------------------------------------ independent review (findings A1-A11, B1-B6)
# Inputs are the reviewer's cases; expected values are those Dynare 7.2 printed under Octave 8.4
# (M_.params as [p q r] unless stated).  "refused" means pydyn must raise ModelError.
RV_HEAD = ("var y c; varexo e u; parameters p q r;\n"
           "model; y = 0.5*y(-1)+e+0*(p+q+r); c = 0.3*c(-1) + u; end;\n")
RV_P = "p=1;q=2;r=3;\n"
RV_RBC = ("var y c k; varexo e; parameters alpha beta delta A;\n"
          "model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;\n"
          "alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;\n")
RV_SSM_A = "steady_state_model;\nk = 10;\nA = (1/beta - 1 + delta)/(alpha*k^(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\nend;\n"
RV_SSM = "steady_state_model;\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\nend;\n"
RV_SSFILE = """function [ys,params,check] = {name}_steadystate(ys,exo,M_,options_)
NumberOfParameters = M_.param_nbr;
for ii = 1:NumberOfParameters
  paramname = M_.param_names{{ii}};
  eval([ paramname ' = M_.params(' int2str(ii) ');']);
end
check = 0;
{body}
params=NaN(NumberOfParameters,1);
for iter = 1:length(M_.params)
  eval([ 'params(' num2str(iter) ') = ' M_.param_names{{iter}} ';' ])
end
NumberOfEndogenousVariables = M_.orig_endo_nbr;
for ii = 1:NumberOfEndogenousVariables
  varname = M_.endo_names{{ii}};
  eval(['ys(' int2str(ii) ') = ' varname ';']);
end
end
"""
SETGET = {"setget.m": "function y = setget(x)\nglobal M_\nM_.params(1) = x;\ny = x;\nend\n"}
REFUSED = "refused"


def rv(tmp_path, body, files=None, head=RV_HEAD, name="m"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(tmp_path, head + body, name=name, files=files)


RV_CASES = {
    # A1: one unparsable statement must not hide the other statements of its line
    "A1_u01_complex_literal": (RV_P + "z = abs(3+4i); M_.params(1) = z;", None, REFUSED),          # Dynare [5 2 3]
    "A1_u10_format_long": (RV_P + "x = 1; format long; M_.params(1) = 5;", None, [5, 2, 3]),
    "A1_u11_close_all": (RV_P + "x = 1; close all; M_.params(1) = 5;", None, [5, 2, 3]),
    "A1_u16_hex": (RV_P + "x = 0x10; M_.params(1) = x;", None, REFUSED),                            # Dynare [16 2 3]
    "A1_u05_plus_equal": (RV_P + "x = 3; x += 1; M_.params(1) = x;", None, REFUSED),                # Octave only: [4 2 3]
    "A1_u14_increment": (RV_P + "x = 3; x++; M_.params(1) = x;", None, REFUSED),
    "A1_u06_inside_if": (RV_P + "if 1, x = 2i; M_.params(1) = 3; end", None, [3, 2, 3]),
    "A1_u02_stale_workspace": (RV_P + "w = 1;\nx = 1; clear x; w = 2;\nM_.params(1) = w;", None, [2, 2, 3]),
    "A1_u07_param_copy": (RV_P + "x = 2i; p = 3;\nM_.params(2) = p;", None, [1, 3, 3]),
    # A4: M_ and options_ are values
    "A4_m01_copy": (RV_P + "M2 = M_;\nM2.params(1) = 100;\nx = M_.params(1);\nq = x;", None, [1, 1, 3]),
    "A4_k16_restore": (RV_P + "M_bak = M_;\np = 9;\nM_ = M_bak;", None, [1, 2, 3]),
    "A4_m05_restore_after_loop": (RV_P + "M_bak = M_;\nfor k = 1:2\n set_param_value('p', k*10);\nend\nM_ = M_bak;", None, [1, 2, 3]),
    "A4_m06_struct_field": (RV_P + "s.M = M_;\np = 9;\nzz = s.M.params(1);\nM_.params(3) = zz;", None, [9, 2, 1]),
    "A4_m08_handle_capture": (RV_P + "f = @() M_.params(1);\np = 9;\nzz = f();\nM_.params(3) = zz;", None, [9, 2, 1]),
    "A4_m16_options": (RV_P + "opt_bak = options_;\noptions_.order = 3;\noptions_ = opt_bak;\nx = options_.order;\nq = x;", None, [1, 2, 3]),
    # A5: arguments of display functions are evaluated
    "A5_k30_disp": (RV_P + "disp(setget(3));", SETGET, [3, 2, 3]),
    "A5_k31_fprintf": (RV_P + "fprintf('%d\\n', setget(3));", SETGET, [3, 2, 3]),
    "A5_y03_warning": (RV_P + "warning('value %d', setget(3));", SETGET, [3, 2, 3]),
    "A5_y07_display": (RV_P + "x = 1;\ndisplay(setget(3));", SETGET, [3, 2, 3]),
    "A5_y04_evalc": (RV_P + "x = 5;\ndisp(evalc('x = 7;'));\nM_.params(2) = x;", None, REFUSED),   # Dynare [1 7 3]
    # A6 / B4: NaN-producing operations are not errors
    "A6_t01_try": (RV_P + "try\n x = 0/0;\n M_.params(1) = 5;\ncatch\n M_.params(1) = 6;\nend", None, [5, 2, 3]),
    "B4_t02_nan": (RV_P + "x = 0/0;\nM_.params(1) = isnan(x);", None, [1, 2, 3]),
    "B4_t03_inf_minus_inf": (RV_P + "x = Inf - Inf;\nif isnan(x)\n p = 7;\nend", None, [7, 2, 3]),
    "B4_u48_roundoff": (RV_P + "x = 0.1*3; y1 = 0.3; M_.params(1) = (x == y1) + 10*(abs(x-y1) < eps) + 100*(x >= y1); "
                               "M_.params(2) = 1e-320*1e10; M_.params(3) = realmax*10 + (-realmax*10);", None,
                        [110, 9.9998886718268301e-311, np.nan]),
    # A7: for over an empty range empties the loop variable
    "A7_t06": (RV_P + "k = 5;\nfor k = 1:0\nend\nif isempty(k)\n p = 7;\nelse\n p = 8;\nend", None, [7, 2, 3]),
    "A7_c29": (RV_P + "k = 10;\nfor k = 1:0\n p = 5;\nend\nq = isempty(k);\nr = numel(k);", None, [1, 1, 0]),
    "A7_y12": (RV_P + "k = 3;\nfor k = zeros(1,0)\nend\nM_.params(2) = numel(k);", None, [1, 0, 3]),
    # A8: mod with a non-integer divisor
    "A8_t04": (RV_P + "if mod(1, 0.1) == 0\n p = 7;\nelse\n p = 8;\nend", None, [7, 2, 3]),
    "A8_w04": ("x = 0.1; z = 1;\np = mod(z, x); q = 1; r = 1;", None, [0, 1, 1]),
    # A9: max/min with NaN in Dynare statements
    "A9_y01": ("x = NaN;\np = max(x, 1); q = min(1, x); r = 1;", None, [1, 1, 1]),
    # B3: Inf in Dynare statements
    "B3_v19": ("flag = true; x = -1; z = 0;\np = flag*2 + flag; q = (x < z) + (x == -1) + (x >= z)*10; r = 1/z;", None, [3, 2, np.inf]),
    "B3_w02": ("x = 10; z = -1;\np = x^400; q = 1; r = 1;", None, [np.inf, 1, 1]),
    # B6: a parameter without a value is not a MATLAB variable
    "B6_d19": ("p = 1;\nzz = q;\n", None, REFUSED),                                                 # Dynare: 'q' undefined
    # section C (relaxed refusals)
    "C_m18_row_params": (RV_P + "M_.params = [7 8 9];", None, [7, 8, 9]),
    "C_k12_run": (RV_P + "run('myscript.m');", {"myscript.m": "set_param_value('p', 33);\nzz = 17;\n"}, [33, 2, 3]),
    "C_a29_beta_function": ("p = beta(2,3); q = 1; r = lgamma(5);", None, [1 / 12, 1, 3.1780538303479458]),
    "C_k45_options_default": (RV_P + "x = options_.order;\nq = x;", None, [1, 2, 3]),
    "C_n07_resid_keeps_params": (RV_P + "resid;\nq = p/2;", None, [1, 0.5, 3]),
    # second pass by the author (values from Dynare 7.2 under Octave, cases p03-p40 of the pass)
    "P2_p03_condition_side_effect": (RV_P + "if setget(3) > 2, zz = 9; end\nM_.params(2) = zz;", SETGET, [3, 9, 3]),
    "P2_p05_cell_copy_of_M": (RV_P + "cc = {M_};\np = 9;\nM_.params(3) = cc{1}.params(1);", None, [9, 2, 1]),
    "P2_p06_M_passed_by_value": (RV_P + "r0 = getp(M_);\nM_.params(2) = r0;",
                                 {"getp.m": "function v = getp(M)\nM.params(1) = 50;\nv = M.params(1);\nend\n"}, [1, 50, 3]),
    "P2_p07_global_variable": (RV_P + "global gx\ngx = 4;\nsetg();\nM_.params(1) = gx;",
                               {"setg.m": "function setg()\nglobal gx\ngx = gx + 1;\nend\n"}, [5, 2, 3]),
    "P2_p10_oo_steady_state_before_steady": (RV_P + "stoch_simul(order=1,irf=0,noprint);\nx = oo_.steady_state(1) + 5;\nq = x;", None, [1, 5, 3]),
    "P2_p20_option_changed_by_command": (RV_P + "options_.order = 1;\nstoch_simul(order=2,irf=0,noprint);\nx = options_.order;\nq = x;",
                                         None, REFUSED),                                           # Dynare q = 2
    "P2_p21_option_not_touched_by_command": (RV_P + "options_.irf = 7;\nstoch_simul(order=1,noprint,nograph);\nx = options_.irf;\nq = x;",
                                             None, [1, 7, 3]),
    "P2_p40_oo_from_initval": (RV_P + "initval; y = 2; end;\nx = oo_.steady_state(1) + oo_.exo_steady_state(2);\nq = x;", None, [1, 2, 3]),
    "P2_unterminated_string": (RV_P + "x = 'abc; M_.params(1) = 5;", None, REFUSED),               # a syntax error in MATLAB
}


@pytest.mark.parametrize("case", sorted(RV_CASES))
def test_review_cases(case, tmp_path):
    body, files, expected = RV_CASES[case]
    if expected == REFUSED:
        with pytest.raises(pydyn.ModelError):
            rv(tmp_path, body, files)
        return
    m = rv(tmp_path, body, files)
    np.testing.assert_allclose(m.params, expected, rtol=1e-14, atol=0, equal_nan=True)


def test_review_a1_warning_does_not_overclaim(tmp_path):
    with pytest.warns(UserWarning) as rec:
        m = build(tmp_path, RV_HEAD + RV_P + "x = 2i;\n")
    msg = [str(w.message) for w in rec if "MATLAB code" in str(w.message)]
    assert len(msg) == 1 and "None of them writes" not in msg[0] and "found no write" in msg[0]
    assert m.matlab_skipped[0][0] == "x = 2i" and list(m.params) == [1, 2, 3]


def test_review_a2_parameters_set_by_the_steady_state(tmp_path):
    """A2 (s27, s32, n03, n04): files WITHOUT any MATLAB line.  After steady/stoch_simul Dynare's
    M_.params holds A = 0.4975... set by the steady state; later statements use it."""
    A, delta = 0.49751442178543509, 0.24875721089271755
    m = rv(tmp_path, RV_SSM_A + "steady;\ndelta = A/2;\n", head=RV_RBC, name="s32")
    np.testing.assert_allclose(m.params, [0.33, 0.99, delta, A], rtol=1e-14)
    assert m.matlab_skipped == []
    m = rv(tmp_path, RV_SSM_A + "stoch_simul(order=1,irf=0,noprint);\ndelta = A/2;\n", head=RV_RBC, name="n03")
    np.testing.assert_allclose(m.params, [0.33, 0.99, delta, A], rtol=1e-14)
    m = rv(tmp_path, RV_SSM_A + "steady;\nshocks; var e = A; end;\n", head=RV_RBC, name="n04")
    np.testing.assert_allclose(m.Sigma_e, [[A]], rtol=1e-14)
    np.testing.assert_allclose(m.params, [0.33, 0.99, 0.025, A], rtol=1e-14)
    body = "k = 10;\nA = (1/beta - 1 + delta)/(alpha*k^(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\n"
    m = rv(tmp_path, "steady;\ndelta = A/2;\n", head=RV_RBC, name="s27",
           files={"s27_steadystate.m": RV_SSFILE.format(name="s27", body=body)})
    np.testing.assert_allclose(m.params, [0.33, 0.99, delta, A], rtol=1e-14)
    # s25: the MATLAB variable A is not updated by steady (x = 1), M_.params(A) is
    m = rv(tmp_path, RV_SSM_A + "steady;\nx = A;\nM_.params(3) = x/2;\n", head=RV_RBC, name="s25")
    np.testing.assert_allclose(m.params, [0.33, 0.99, 0.5, A], rtol=1e-14)


def test_review_a3_b1_initval_endval_read_the_current_steady_state_vector(tmp_path):
    """A3 (a47, v01, v02) and B1 (a49, v12, v13)."""
    m = rv(tmp_path, RV_P + "initval; y = 1; c = 2; end;\nendval; y = 3; c = y*2; end;\n", name="a47")
    assert m.initval == {"y": 1.0, "c": 2.0} and m.endval == {"y": 3.0, "c": 6.0}
    m = rv(tmp_path, RV_P + "initval; c = y*2 + 1; y = 4; end;\n", name="a49")          # unset variables are 0
    assert m.initval == {"c": 1.0, "y": 4.0}
    m = rv(tmp_path, RV_P + "initval; y = 1; e = 0.5; end;\nendval; e = 1; c = e*3; y = c + 1; end;\n", name="v12")
    assert m.endval == {"e": 1.0, "c": 3.0, "y": 4.0} and m.initval == {"y": 1.0, "e": 0.5}
    m = rv(tmp_path, RV_P + "endval; e = 1; c = e*3; y = c + 1; end;\n", name="v13")
    assert m.endval == {"e": 1.0, "c": 3.0, "y": 4.0}
    # v02: analytical steady state, then endval refers to it: Dynare k = 56.696838122096885
    m = rv(tmp_path, RV_SSM + "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady;\nendval; k = k*2; c = 1; y = 1; end;\n",
           head=RV_RBC, name="v02")
    np.testing.assert_allclose(m.endval["k"], 56.696838122096885, rtol=1e-14)
    assert m.initval == {"k": 20.0, "c": 1.5, "y": 2.5}
    # v01: numerical steady state.  Dynare's own solver (solve_algo=4, trust region) stops at
    # k = 28.348419047 (residual 1e-10), the exact value is 28.348419061.  pydyn replays a 'steady'
    # command with its port of Dynare's solver, so it stops at the same iterate (measured: identical)
    m = rv(tmp_path, "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady;\ninitval; c = k/10; end;\n", head=RV_RBC, name="v01")
    np.testing.assert_allclose(m.initval["c"], 2.8348419047165483, rtol=1e-13)
    assert m.initval["k"] == 20.0 and m.initval["y"] == 2.5
    assert abs(m.initval["c"] - 2.8348419061048443) > 1e-10          # not the exact root (1/beta-1+delta)/(alpha)


def test_review_a6_a7_in_steady_state_files(tmp_path):
    """x70-x72: translated steady-state files agree with the interpreter and with Dynare
    (oo_.steady_state [y c k] = [50 3 1], [2 3 5], [2 3 5])."""
    head = ("var y c k; varexo e; parameters alpha beta delta A lambda is np;\n"
            "model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta) + 0*(lambda+is+np); y = A*exp(e)*k(-1)^alpha; "
            "k = y - c + (1-delta)*k(-1); end;\nalpha = 0.33; beta = 0.99; delta = 0.025; A = 1; lambda = 2; is = 3; np = 4;\n")
    cases = {"x70": ("k = 1; y = 2; c = 3;\nfor ii = 1:0\n k = 100;\nend\nif isempty(ii), y = 50; end\n", [50, 3, 1]),
             "x71": ("k = 1; y = 2; c = 3; t = 0;\ntry\n t = 1/t;\n k = 5;\ncatch\n k = 6;\nend\n", [2, 3, 5]),
             "x72": ("k = 1; y = 2; c = 3; t = 10;\ntry\n t = t^400;\n k = 5;\ncatch\n k = 6;\nend\n", [2, 3, 5])}
    for name, (body, expected) in cases.items():
        d = tmp_path / name
        d.mkdir()
        m = rv(d, "steady(nocheck);\n", head=head, name=name, files={f"{name}_steadystate.m": RV_SSFILE.format(name=name, body=body)})
        fn = pydyn.steady.steady_state_file_fn(m)
        ys, _ = fn(m, m.params.copy(), np.zeros(1))
        np.testing.assert_allclose(ys[:3], expected)
        with np.errstate(all="ignore"):
            np.testing.assert_allclose(fn._interpret(m, m.params.copy(), np.zeros(1))[0][:3], expected)


def test_review_a9_initval_with_nan(tmp_path):
    m = rv(tmp_path, RV_P + "x = NaN;\ninitval; y = max(x, 2); c = min(x, 2); end;\n")          # w15
    assert m.initval == {"y": 2.0, "c": 2.0}


def test_review_a10_builtins():
    ws = run("a = str2double({'0.5','0.7'}); b = str2double('1,000'); c = round(0.49999999999999994); "
             "d = num2str(2^53); e1 = sprintf('%s', 65); f = round(-2.5); g = mod(10, 0.1) + mod(0.3, 0.1) + rem(1, 0.1);")
    np.testing.assert_array_equal(ws["a"], [[0.5, 0.7]])
    assert (ws["b"], ws["c"], ws["d"], ws["e1"], ws["f"], ws["g"]) == (1000.0, 0.0, "9007199254740992", "A", -3.0, 0.0)


def test_review_a11_two_output_steady_state_file(tmp_path):
    f = ("function [ys,check] = s12_steadystate(ys,exo)\nglobal M_\ncheck = 0;\nalpha = M_.params(1); beta = M_.params(2); "
         "delta = M_.params(3); A = M_.params(4);\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\ny = A*k^alpha;\n"
         "c = y - delta*k;\nys = [y; c; k];\nend\n")
    m = rv(tmp_path, "steady;\n", head=RV_RBC, name="s12", files={"s12_steadystate.m": f})
    with pytest.raises(pydyn.SteadyStateError, match="too many inputs"):       # Dynare: "called with too many inputs"
        pydyn.steady_state(m)


def test_review_b2_block_local_named_x(tmp_path):
    """v04/v05: a steady_state_model local called x next to an exogenous variable."""
    expected = [3.0153277085137282, 2.3066172319875173, 28.348419061048443]
    for name, block in (("v04", "x = (1/beta - 1 + delta)/(alpha*A*exp(e));\nk = x^(1/(alpha-1));\n"),
                        ("v05", "x = (1/beta - 1 + delta)/(alpha*A);\nk = x^(1/(alpha-1));\n")):
        d = tmp_path / name
        d.mkdir()
        m = rv(d, "steady_state_model;\n" + block + "y = A*k^alpha*exp(e);\nc = y - delta*k;\nend;\nsteady(nocheck);\n",
               head=RV_RBC, name=name)
        np.testing.assert_allclose(pydyn.steady_state(m)[0], expected, rtol=1e-14)
        import pydyn.steady as st
        np.testing.assert_allclose(st._from_steady_state_model_interpreted(
            m, m.params.copy(), np.zeros(1), np.zeros(3))[0], expected, rtol=1e-14)


def test_review_b3_inf_in_shocks(tmp_path):
    m = rv(tmp_path, RV_P + "x = 0;\nshocks; var e; stderr 1/x; end;\n")                # w12: Sigma_e(1,1) = Inf
    assert m.Sigma_e[0, 0] == np.inf


def test_review_b5_json_repairs(tmp_path):
    """The preprocessor writes invalid JSON for an equation tag containing a backslash
    (tests/shock_decomposition/fs2000_cal_groups.mod) and for a colormap given as a MATLAB variable."""
    m = rv(tmp_path, "var y; varexo e; parameters p; p = 0.5;\nmodel; [eq='\\#1', name='growth: $\\Delta A$'] y = p*y(-1) + e; end;\n",
           head="")
    assert m.get_param("p") == 0.5


def test_review_c_vector_elements_and_m_fields_in_dynare_statements(tmp_path):
    m = rv(tmp_path, RV_P + "v = [0.1 0.2 0.3];\nshocks; var e; stderr v(2); var u = v(3); end;\n"
                             "initval; y = v(1)*10; end;\nq = M_.Sigma_e(2,2);\n")
    np.testing.assert_allclose(np.diag(m.Sigma_e), [0.04, 0.3])
    assert m.initval == {"y": 1.0} and m.get_param("q") == 0.3


def test_review_c_helper_reading_global_m(tmp_path):
    """s39: Dynare oo_.steady_state = [3.0153277085137282 2.3066172319875173 28.348419061048443]."""
    kglob = ("function k = kglob(dummy)\nglobal M_\nalpha = M_.params(1); beta = M_.params(2); delta = M_.params(3); "
             "A = M_.params(4);\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\nend\n")
    m = rv(tmp_path, "steady_state_model;\nk = kglob(1);\ny = A*k^alpha;\nc = y - delta*k;\nend;\nsteady;\n",
           head=RV_RBC, name="s39", files={"kglob.m": kglob})
    np.testing.assert_allclose(pydyn.steady_state(m)[0], [3.0153277085137282, 2.3066172319875173, 28.348419061048443], rtol=1e-14)
    m.set_param("A", 1.2)                                   # the helper sees the parameters of the call
    assert pydyn.steady_state(m)[0][2] > 30


def test_review_e_dynare_functions_writing_m_through_global(tmp_path):
    """v24 and the scan of Dynare's matlab/ tree: a call to such a function is refused, not skipped."""
    for line in ("set_parameters(xparam1);", "x = 1;\nolsgibbs(ds, 'eq', 1, 2, 3);", "dynare_estimation(var_list_);",
                 "pac.update.parameters('pacman', M_, oo_);"):
        with pytest.raises(pydyn.ModelError):
            rv(tmp_path, RV_P + "xparam1 = 0.7;\n" + line + "\n", name="e" + str(abs(hash(line)) % 10**6))


def test_second_pass_steady_state_parameters_and_copies(tmp_path):
    """p25, p28, p29, p37 of the author's second pass (Dynare's M_.params [alpha beta delta A])."""
    A = 0.49751442178543509
    cases = {"p25": ("steady;\nM_bak = M_;\nA = 3;\nM_ = M_bak;\ndelta = A/2;", [0.33, 0.99, A / 2, A]),
             "p28": ("steady;\nA = 2;\nsteady;\ndelta = A/2;", [0.33, 0.99, A / 2, A]),
             "p29": ("steady;\nf = @() get_param_by_name('A');\nM_.params(3) = f()/2;", [0.33, 0.99, A / 2, A]),
             "p36": ("check;\ndelta = A/2;", [0.33, 0.99, 0.5, 1.0]),
             "p37": ("steady;\nset_param_value('A', 4);\nstoch_simul(order=1,irf=0,noprint);\ndelta = A/2;", [0.33, 0.99, A / 2, A])}
    for name, (body, expected) in cases.items():
        d = tmp_path / name
        d.mkdir()
        m = rv(d, RV_SSM_A + body + "\n", head=RV_RBC, name=name)
        np.testing.assert_allclose(m.params, expected, rtol=1e-14, err_msg=name)


def test_mutator_list_is_complete_for_the_dynare_tree():
    """Review E: every Dynare function that writes the calibration through `global M_` (or returns a
    changed M_) is in pydyn.matlab's refusal lists (scan in tools/sweep_matlab.py)."""
    if not (DYNARE_SRC / "matlab").is_dir():
        pytest.skip("Dynare source tree not available")
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sweep_matlab
    finally:
        sys.path.pop(0)
    assert sweep_matlab.mutators(DYNARE_SRC) == []


# ------------------------------------------------------------------ oo_.steady_state read and written by MATLAB code
WR_HEAD = ("var y c k; varexo e; parameters alpha beta delta A;\n"
           "model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;\n"
           "alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;\ninitval; k = 20; c = 1.5; y = 2.5; end;\n")


def test_reads_and_writes_of_the_steady_state_vector(tmp_path):
    """After a numerical 'steady', MATLAB code reads oo_.steady_state (pydyn replays Dynare's solver,
    solve_algo=4, so the value is Dynare's iterate; model.calibration_tolerance records that it is
    only known to the steady-state tolerance), writes elements of oo_.steady_state and the whole
    oo_.exo_steady_state, and a later initval block reads them. Dynare 7.2 (Octave): params
    [0.33 0.99 0.025 1.0051092360783622], oo_.steady_state [3.0153277082350867 3.01 30], exo 0.01."""
    with pytest.warns(UserWarning, match="computed numerically"):
        m = build(tmp_path, WR_HEAD + "steady;\nx1 = oo_.steady_state(1);\noo_.steady_state(3) = 30;\n"
                  "oo_.exo_steady_state = 0.01;\ninitval; c = k/10 + e; end;\nA = x1/3;\n")
    np.testing.assert_allclose(m.params, [0.33, 0.99, 0.025, 1.0051092360783622], rtol=1e-15)
    assert m.initval == {"k": 30.0, "c": 3.01, "y": pytest.approx(3.0153277082350867, rel=1e-15), "e": 0.01}
    assert m.calibration_tolerance == pytest.approx(6.06e-6, rel=1e-2)
    # the writes are seen by the perfect-foresight replay at their place in the file
    ws = importlib.import_module("pydyn.perfect_foresight")._replay(m, steady=False)
    assert ws.ss[2] == 30.0 and ws.ss[1] == 3.01 and ws.xss[0] == 0.01
    # an analytical steady state is read without the note
    m2 = rv(tmp_path, RV_SSM + "steady;\nx1 = oo_.steady_state(3);\nalpha = x1/100;\n", head=RV_RBC, name="m2")
    assert m2.calibration_tolerance is None
    assert m2.get_param("alpha") == pytest.approx(((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1)) / 100, rel=1e-14)


def test_unknown_writes_to_oo_refuse_only_their_users(tmp_path):
    """oo_.steady_state assigned from something pydyn cannot compute: the file loads as long as nothing
    later starts from it; a later steady, initval reference or perfect_foresight_setup is refused."""
    ok = rv(tmp_path, "oo_ = some_unknown_function(oo_);\noo_.steady_state = oo_.dr.ys;\n", head=WR_HEAD, name="ok")
    assert ok.params[0] == 0.33
    for k, tail in enumerate(("oo_ = some_unknown_function(oo_);\nsteady;\n",
                              "oo_.steady_state = oo_.dr.ys;\ninitval; c = k/10; end;\n",
                              "oo_.steady_state(2) = oo_.dr.ys(2);\nperfect_foresight_setup(periods=10);\n")):
        with pytest.raises(pydyn.ModelError, match="oo_"):
            rv(tmp_path, tail, head=WR_HEAD, name=f"bad{k}")

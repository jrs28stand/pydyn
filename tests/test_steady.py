"""Numerical steady states: random restarts, maxit, solve_algo (pydyn's ports of Dynare's solvers),
exact guesses.

Expected values were printed by Dynare 7.2 under Octave 8.4 (evaluate_steady_state.m called on the
models below, the starting points given; scratch runs /home/claude/run_pf2/algos and
/home/claude/run_pf2/bench of 2026-11-21).
"""
import warnings

import numpy as np
import pytest

import pydyn
from pydyn import steady as S

RBC = """var y c k; varexo e; parameters alpha beta delta A;
model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;
alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;
%s
"""
#: Dynare's oo_.steady_state [y c k] from initval y = 2.5, c = 1.5, k = 20 with solve_algo 4, 9, 13, 14
#: (all four stop at the same Newton iterate, residual below 6e-6)
DYNARE_TR = [3.0153277082350867, 2.306617232055947, 28.348419047165482]
#: the exact root (alpha/(1/beta - 1 + delta))^(1/(1-alpha))
K_EXACT = (0.33 / (1 / 0.99 - 1 + 0.025)) ** (1 / (1 - 0.33))


def build(tmp_path, text, name="m"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / f"{name}.mod").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / f"{name}.mod"))


def test_random_restart_when_the_guess_is_not_finite(tmp_path):
    """1/c at c = 0: Dynare's dynare_solve draws new guesses (rand*10, then randn*10, then -rand*10)
    until the residuals and the Jacobian are finite. pydyn does the same with its own seeded draws
    (deterministic), warns once, and polishes the root; Dynare stopped at k = 28.348324674939885 from
    its own random point (within its tolerance of the root 28.3484190610)."""
    m = build(tmp_path, RBC % "steady;")
    with pytest.warns(UserWarning, match="restarted from random guesses") as rec:
        ys, _ = pydyn.steady_state(m)
        ys2, _ = pydyn.steady_state(m)
    assert sum("restarted" in str(w.message) for w in rec) == 1          # once per model
    np.testing.assert_array_equal(ys, ys2)
    np.testing.assert_allclose(ys[2], K_EXACT, rtol=1e-13)
    assert abs(ys[2] - 28.348324674939885) / K_EXACT < 1e-5                # Dynare: same root, its own stopping point
    # Dynare's algorithm (pydyn's port) from the same kind of restart: a point within the tolerance
    ys4, _ = pydyn.steady_state(m, solve_algo=4)
    assert abs(ys4[2] - K_EXACT) / K_EXACT < 1e-4
    assert np.max(np.abs(m.static_resid(ys4, np.zeros(1), m.params))) < S.DYNARE_TOLF


def test_the_draws_follow_dynare_solve():
    """idx = find(x) (the non-zero entries) unless the guess is zero; 10*len(idx)+1 draws per phase."""
    calls = []
    draws = list(S._random_draws(lambda y: calls.append(y.copy()) or False, np.array([0.0, 2.0, 0.0])))
    assert draws == [] and len(calls) == 3 * (10 * 1 + 1)
    assert all(c[0] == 0 and c[2] == 0 for c in calls)                    # only the non-zero entry is redrawn
    assert all(0 < c[1] < 10 for c in calls[:11]) and any(c[1] < 0 for c in calls[11:22])
    assert all(-10 < c[1] < 0 for c in calls[22:])
    calls.clear()
    next(S._random_draws(lambda y: calls.append(y.copy()) or True, np.zeros(3)))
    assert np.all(calls[0] > 0)                                            # zero guess: every entry


def test_an_exact_guess_is_returned_unchanged(tmp_path):
    """The numerical branch never moves away from a guess that already solves the static model."""
    m = build(tmp_path, RBC % "initval; k = 28; c = 2; y = 3; end;\nsteady;")
    exact, _ = pydyn.steady_state(m)
    for algo in (None, 4, 13):
        again, _ = pydyn.steady_state(m, guess=exact, solve_algo=algo)
        np.testing.assert_array_equal(again, exact)
    # as in dynare_solve, only the non-zero entries of a guess are redrawn: with c = y = 0 kept,
    # 1/c is never finite and the search fails, as Dynare's does
    m0 = build(tmp_path, RBC % "initval; k = 28; end;\nsteady;", "m0")
    with pytest.raises(pydyn.SteadyStateError, match="random restart points"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pydyn.steady_state(m0)


def test_solve_algo_runs_dynare_trust_region(tmp_path):
    """solve_algo 4, 9, 13, 14 run pydyn's ports of trust_region.m and of the block_trust_region MEX
    (with and without the Dulmage-Mendelsohn blocks): Dynare's iterate, to rounding."""
    m = build(tmp_path, RBC % "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady;")
    for algo in (4, 9, 13, 14):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ys, _ = pydyn.steady_state(m, solve_algo=algo)
        np.testing.assert_allclose(ys, DYNARE_TR, rtol=1e-13)
    ys, _ = pydyn.steady_state(m)                                          # pydyn's own search, polished
    np.testing.assert_allclose(ys[2], K_EXACT, rtol=1e-14)
    # from the .mod file
    m2 = build(tmp_path, RBC % "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady(solve_algo=9);", "m2")
    np.testing.assert_allclose(pydyn.steady_state(m2)[0], DYNARE_TR, rtol=1e-13)


def test_other_solve_algo_values(tmp_path):
    """solve_algo values pydyn has not ported: its own search, silently when Newton's method from the
    same guess reaches the same root (Dynare's solve_algo=2 stops at 28.348417798, the same root
    within its tolerance), with a warning otherwise."""
    m = build(tmp_path, RBC % "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady(solve_algo=2);")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ys, _ = pydyn.steady_state(m)
    np.testing.assert_allclose(ys[2], 28.348417798467448, rtol=1e-6)
    # Newton from a guess far away does not converge: the root cannot be confirmed
    m2 = build(tmp_path, RBC % "initval; k = 900; c = 0.1; y = 0.1; end;\nsteady(solve_algo=3);", "m2")
    with pytest.warns(UserWarning, match="solve_algo=3"):
        try:
            pydyn.steady_state(m2)
        except pydyn.SteadyStateError:
            pass
    with pytest.warns(UserWarning, match="complementarity"):
        pydyn.steady_state(m, solve_algo=10)


def test_maxit_is_honoured(tmp_path):
    """steady(maxit=...): Dynare with maxit=2 fails from k = 20 (info 20); so does pydyn, with its own
    search and with the ported trust region (whose default maxit is Dynare's 50)."""
    m = build(tmp_path, RBC % "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady(maxit=2);")
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(m)
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(m, solve_algo=4)
    np.testing.assert_allclose(pydyn.steady_state(m, maxit=50)[0][2], K_EXACT, rtol=1e-13)
    np.testing.assert_allclose(pydyn.steady_state(m, maxit=50, solve_algo=4)[0], DYNARE_TR, rtol=1e-13)


BENCHMARK = """// Dynare 7.2, tests/external_function/benchmark.mod (Copyright (C) Dynare Team, GPL-3.0-or-later).
var y, c, k, a, h, b;
varexo e, u;
parameters beta, rho, alpha, delta, theta, psi, tau;
alpha = 0.36; rho = 0.95; tau = 0.025; beta = 0.99; delta = 0.025; psi = 0; theta = 2.95; phi = 0.1;
model;
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1)^2;
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;
initval;
y = 1.08068253095672; c = 0.80359242014163; h = 0.29175631001732; k = 11.08360443260358;
a = 0; b = 0; e = 0; u = 0;
end;
shocks; var e; stderr 0.009; var u; stderr 0.009; var e, u = phi*0.009*0.009; end;
stoch_simul;
"""


def test_benchmark_external_function_converges_to_dynares_root(tmp_path):
    """The reported 'different root' of tests/external_function/benchmark.mod: its initval is the
    steady state of example1.mod, not of this model (k(-1) is squared: residual -109 in the capital
    equation), and the file has no 'steady' command, so Dynare's oo_.steady_state keeps the initval
    values. The steady state Dynare computes for stoch_simul (oo_.dr.ys, solve_algo=4) is the root pydyn
    finds: pydyn's polished root agrees to 2.7e-11 (Dynare stops within its tolerance), pydyn's port of
    solve_algo=4 to rounding."""
    m = build(tmp_path, BENCHMARK, "benchmark")
    y0 = np.array([m.initval[v] for v in m.endo_names])
    assert abs(m.static_resid(y0, np.zeros(2), m.params)[3]) > 100
    dr_ys = [0.14448332823291504, 0.8035924201416301, 1.4818376455312454, 0.0, 0.03900675868882372, 0.0]
    ys, _ = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, dr_ys, rtol=1e-10, atol=1e-14)
    ys4, _ = pydyn.steady_state(m, solve_algo=4)
    np.testing.assert_allclose(ys4, dr_ys, rtol=1e-13, atol=1e-14)


# ------------------------------------------------------------------ continuum of steady states (unit root)
def _ref_model(tmp_path, name):
    from pathlib import Path
    import shutil
    ref = Path(__file__).resolve().parent.parent / "reference"
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    shutil.copy(ref / f"{name}.mod", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / f"{name}.mod"))


def test_dmperm_is_octaves():
    """The port of CSparse's cs_dmperm returns Octave's [p, q, r, s] (here the structure of the static
    Jacobian of reference/ur_ss.mod at its initval, printed by Octave 8.4: p = [4 3 5 1 2],
    q = [5 1 2 3 4], r = [1 4 5 6], s = [1 5 6 6]; 600 random structures were checked the same way)."""
    J = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0]], bool)
    p, q, r, s = S.dmperm(J)
    assert (p + 1).tolist() == [4, 3, 5, 1, 2] and (q + 1).tolist() == [5, 1, 2, 3, 4]
    assert (r + 1).tolist() == [1, 4, 5, 6] and (s + 1).tolist() == [1, 5, 6, 6]
    assert S.structurally_singular(J) and not S.structurally_singular(np.eye(3) + np.eye(3, k=1))


def test_unit_root_steady_state_is_dynares_point_of_the_continuum(tmp_path):
    """R01. z = z(-1) + ez makes the static model singular (a continuum of steady states indexed by
    z). Dynare's default solve_algo=4 solves the Dulmage-Mendelsohn blocks of the static Jacobian
    from the last to the first, skipping those that already hold: here the unit-root equation and a's
    block hold at initval, and the leading block (rows of the Euler, production and capital
    equations, columns z, y, c, k) is solved, as dynare_solve.m indexes it, on its first three
    columns only, so k keeps its initval 40 and z moves from 0.5 to 0.0516. pydyn's own search
    (Powell's hybrid method on the whole system) used to stop at z = 0.199, k = 46.4 (the z column of
    ghx 16% above Dynare's). Measured now: ys 4e-16, ghx 9e-14. Reference: reference/r_unitroot.m
    (Dynare 7.2, Octave 8.4)."""
    from scipy.io import loadmat
    from pathlib import Path
    r = loadmat(Path(__file__).resolve().parent.parent / "reference" / "clean_ur_ss.mat", squeeze_me=True)
    m = _ref_model(tmp_path, "ur_ss")
    ys, _ = pydyn.steady_state(m)
    assert ys[2] == 40.0                                                  # untouched, as in Dynare
    np.testing.assert_allclose(ys, r["ys"], rtol=1e-12, atol=1e-14)
    ys4, _ = pydyn.steady_state(m, solve_algo=4)
    np.testing.assert_array_equal(ys4, ys)
    sol = pydyn.first_order(m)
    np.testing.assert_array_equal(sol.order_var + 1, r["order_var"])
    np.testing.assert_allclose(sol.ghx, r["ghx"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(sol.ghu, r["ghu"], rtol=1e-10, atol=1e-13)


def test_unit_root_steady_state_fails_where_dynare_fails(tmp_path):
    """reference/ur_ss_fail.mod: a = 0.1 at initval. Dynare 7.2 fails (its block loop gives a's
    equation the column of k, a defect of dynare_solve.m for structurally singular Jacobians);
    pydyn refuses too instead of choosing another point of the continuum."""
    from scipy.io import loadmat
    from pathlib import Path
    r = loadmat(Path(__file__).resolve().parent.parent / "reference" / "clean_ur_ss.mat", squeeze_me=True)
    assert int(r["fail_failed"]) == 1
    m = _ref_model(tmp_path, "ur_ss_fail")
    with pytest.raises(S.SteadyStateError, match="structurally singular"):
        pydyn.steady_state(m)

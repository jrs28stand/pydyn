"""Fixes that came out of the acceptance sweep against Dynare 7.2's test suite (docs/ACCEPTANCE.md). The numbers marked
"Dynare" were produced by Dynare 7.2 under Octave 8.4 for the same .mod text (tests/{expectations/expectation_nested,
deterministic_simulations/homotopy,optimal_policy/RamseyConstraints/test1}.mod)."""
import warnings

import numpy as np
import pytest

import pydyn
from pydyn.options import UnsupportedOptionWarning


def build(tmp_path, text, name="m"):
    f = tmp_path / f"{name}.mod"
    f.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(f))


EXPECT_NESTED = """var y, c, k, a, h, b; varexo e, u;
parameters beta, rho, alpha, delta, theta, psi, tau;
alpha = 0.36; rho = 0.95; tau = 0.025; beta = 0.99; delta = 0.025; psi = 0; theta = 2.95; phi = 0.1;
model;
c*theta*h^(1+psi)=expectation(1)(((1-alpha)*y)+expectation(3)((1-alpha)*y))/2;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;
initval; y = 1.08068253095672; c = 0.80359242014163; h = 0.29175631001732; k = 12; a = 0; b = 0; e = 0; u = 0; end;
shocks; var e; stderr 0.009; var u; stderr 0.009; var e, u = phi*0.009*0.009; end;
stoch_simul(order=1,irf=0);
"""


def test_steady_state_with_nested_expectation_operators(tmp_path):
    """The auxiliary variables of EXPECTATION(1)(.. EXPECTATION(3)(..)) define each other: their equations are not in the row of
    their own index, which left them NaN at the starting point of the search (the search then failed)."""
    m = build(tmp_path, EXPECT_NESTED)
    ys, _ = pydyn.steady_state(m)
    dynare = [1.08068276, 0.80359259, 11.08360683, 0.0, 0.29175631, 0.0,
              1.38327394, 0.69163697, 0.69163697, 1.08068276, 1.08068276]       # oo_.dr.ys (Dynare stops at tolf = 6e-6)
    np.testing.assert_allclose(ys, dynare, rtol=1e-5, atol=1e-8)
    sol = pydyn.first_order(m)
    assert np.all(np.isfinite(sol.ghx))


HOMOTOPY = """var Consumption, Capital, LoggedProductivity; varexo LoggedProductivityInnovation;
parameters beta, alpha, delta, rho; beta = .985; alpha = 1/3; delta = alpha/10; rho = .9;
model;
1/Consumption = beta/Consumption(1)*(alpha*exp(LoggedProductivity(1))*Capital^(alpha-1)+1-delta);
Capital = exp(LoggedProductivity)*Capital(-1)^alpha+(1-delta)*Capital(-1)-Consumption;
LoggedProductivity = rho*LoggedProductivity(-1)+LoggedProductivityInnovation;
end;
steady_state_model;
  LoggedProductivity = LoggedProductivityInnovation/(1-rho);
  Capital = (exp(LoggedProductivity)*alpha/(1/beta-1+delta))^(1/(1-alpha));
  Consumption = exp(LoggedProductivity)*Capital^alpha-delta*Capital;
end;
initval; LoggedProductivityInnovation = 0; end;
steady;
endval; LoggedProductivityInnovation = 1; end;
steady;
perfect_foresight_setup(periods=200, first_simulation_period = 1Q2);
perfect_foresight_solver;
"""


def test_homotopy_to_a_large_permanent_shock_converges(tmp_path):
    """The terminal steady state is 6e6 / 6e7: the residuals cannot go below 2e-9 whatever the iteration, so pydyn's default
    tolf (1e-9) was never met and the homotopy stopped at 0.83. Dynare (tolf 1e-5): 15 homotopy iterations, the last solve
    in 8 iterations with error 1.49e-8."""
    m = build(tmp_path, HOMOTOPY)
    r = pydyn.perfect_foresight(m)
    assert r.status and r.homotopy_completion_share == 1.0
    assert r.homotopy_iterations == 15 and r.iterations == 8                      # Dynare
    assert r.max_residual == pytest.approx(1.49e-8, rel=0.01)
    assert r.max_residual < 1e-5
    # an explicit tolf is still obeyed: the rounding floor is only accepted for pydyn's own default
    with pytest.raises(pydyn.PerfectForesightError):
        pydyn.perfect_foresight(m, tolf=1e-12)


RAMSEY = """var i y pi; varexo e_y e_pi; parameters beta1 beta2 beta3 lambda1 lambda2 pi_bar;
beta1 = 0.6; beta2 = 0.25; beta3 = -0.2; lambda1 = 0.7; lambda2 = 0.1; pi_bar = 2.0;
model;
y = beta1*y(-1) + beta2*y(+1) + beta3*(i-pi(+1)) + e_y;
pi = lambda1*pi(+1) + (1-lambda1)*pi(-1) + lambda2*y + e_pi;
end;
planner_objective (pi-pi_bar)^2 + y^2;
ramsey_model(planner_discount=1.0);
histval; y(0) = -2.0; pi(0) = 1.0; end;
steady;
ramsey_constraints; i > 0; end;
perfect_foresight_setup(periods=50);
%s
perfect_foresight_solver;
%s
"""


def test_options_written_by_matlab_lines_are_honoured_by_perfect_foresight(tmp_path):
    """options_.stack_solve_algo = 7 and options_.solve_algo = 10 (as RamseyConstraints/test1.mod): the stacked problem is the
    mixed complementarity problem, as with perfect_foresight_solver(lmmcp). Before, they were stored and ignored (the path
    differed from Dynare's by 4.3)."""
    ref = pydyn.perfect_foresight(build(tmp_path, RAMSEY % ("", ""), "a"), lmmcp=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedOptionWarning)
        m = build(tmp_path, RAMSEY % ("options_.stack_solve_algo = 7;\noptions_.solve_algo = 10;", ""), "b")
        r = pydyn.perfect_foresight(m)
    assert r.solver == "lmmcp" and r.status
    np.testing.assert_allclose(r.endo, ref.endo, rtol=1e-9, atol=1e-9)
    assert r.endo[:, m.endo_names.index("i")].min() >= -1e-9                       # the constraint i >= 0 binds
    # the position of the assignment matters: after the solver statement it is not an input of that solver
    m2 = build(tmp_path, RAMSEY % ("", "options_.stack_solve_algo = 7;\noptions_.solve_algo = 10;"), "c")
    assert pydyn.perfect_foresight(m2).solver != "lmmcp"
    # stack_solve_algo = 7 with solve_algo = 11 is Dynare's PATH solver: refused, not replaced
    m3 = build(tmp_path, RAMSEY % ("options_.stack_solve_algo = 7;\noptions_.solve_algo = 11;", ""), "d")
    with pytest.raises(NotImplementedError, match="mcppath"):
        pydyn.perfect_foresight(m3)


def test_options_written_by_matlab_lines_that_nothing_acts_on_are_reported(tmp_path):
    f = tmp_path / "m.mod"
    f.write_text(RAMSEY % ("options_.irf = 7;\noptions_.nograph = 1;\noptions_.lmmcp.TolFun = 1e-9;\noptions_.dynatol.f = 1e-6;", ""))
    with pytest.warns(UnsupportedOptionWarning, match=r"options_\.irf") as rec:
        pydyn.Model(pydyn.run_preprocessor(f))
    msg = " ".join(str(w.message) for w in rec if issubclass(w.category, UnsupportedOptionWarning))
    assert "nograph" not in msg and "TolFun" not in msg and "dynatol" not in msg      # cosmetic, or acted on

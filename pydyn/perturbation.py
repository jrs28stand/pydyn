"""First-order perturbation solution."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

QZ_CRITERIUM = 1.0 + 1e-6  # Dynare's default: eigenvalues below this count as stable


class BlanchardKahnError(Exception):
    pass


class LoglinearError(ValueError):
    pass


def log_mask(model) -> np.ndarray:
    """Variables Dynare takes logs of under ``loglinear``: every endogenous variable except the
    auxiliary variables for leads and lags of exogenous variables (types 2 and 3), which stay in levels
    (Dynare's ``log_variable`` / ``indices_lagged_leaded_exogenous_variables``)."""
    mask = np.ones(model.n_endo, bool)
    for a in model.raw.get("aux_vars", []):
        if a.get("type") in (2, 3):
            mask[a["endo_index"] - 1] = False
    return mask


def loglinearize(ys, G, H, mask):
    """Rules in log deviations for the variables in ``mask``: ``D^-1 G D`` and ``D^-1 H`` with
    ``D = diag(ys)`` on logged variables (identity elsewhere), as ``stochastic_solvers.m`` does.
    Returns ``(log_ys, G, H)``, where ``log_ys`` holds logs for logged variables and levels otherwise."""
    ys = np.asarray(ys, float)
    if np.any(ys[mask] <= 0) or not np.all(np.isfinite(ys[mask])):
        bad = np.nonzero(mask & ~(ys > 0))[0]
        raise LoglinearError("The loglinearization of the model cannot be performed, because the steady "
                             f"state is not strictly positive (variables at positions {bad.tolist()})")
    d = np.where(mask, ys, 1.0)
    log_ys = np.where(mask, np.log(np.where(mask, ys, 1.0)), ys)
    return log_ys, G * d[None, :] / d[:, None], H / d[:, None]


@dataclass
class FirstOrderSolution:
    """y_t - ys = G (y_{t-1} - ys) + H u_t, variables in declaration order.

    ``ghx`` / ``ghu`` reproduce Dynare's ``oo_.dr.ghx`` / ``oo_.dr.ghu``: rows in
    decision-rule order (``order_var``), ``ghx`` columns restricted to the state variables.
    """
    ys: np.ndarray
    G: np.ndarray
    H: np.ndarray
    order_var: np.ndarray      # 0-based indices into declaration order
    state_var: np.ndarray      # 0-based declaration indices of states, in DR order
    eigenvalues: np.ndarray
    jacobian: np.ndarray
    params: np.ndarray = None
    loglinear: bool = False                 # rules and ys are in logs (log deviations) for logged variables
    steady_state: np.ndarray = None         # the steady state in levels

    def as_dr(self) -> dict:
        """Decision rule in the dict form used by ``pydyn.simulate`` / ``pydyn.irf``."""
        return {"ys": self.ys, "order_var": self.order_var, "state_var": self.state_var,
                "ghx": self.ghx, "ghu": self.ghu, "order": 1}

    @property
    def ghx(self):
        return self.G[np.ix_(self.order_var, self.state_var)]

    @property
    def ghu(self):
        return self.H[self.order_var]

    def irf(self, shock: np.ndarray, periods: int = 40) -> np.ndarray:
        """Deviations from steady state after a one-off shock vector at t=1; shape (periods, n)."""
        out = np.empty((periods, self.G.shape[0]))
        y = self.H @ np.asarray(shock, float)
        for t in range(periods):
            out[t] = y
            y = self.G @ y
        return out

    def simulate(self, shocks: np.ndarray, y0=None) -> np.ndarray:
        """Simulate deviations from steady state for a (T, n_exo) shock array."""
        y = np.zeros(self.G.shape[0]) if y0 is None else np.asarray(y0, float)
        out = np.empty((len(shocks), y.size))
        for t, u in enumerate(shocks):
            y = self.G @ y + self.H @ u
            out[t] = y
        return out


def dr_ordering(lead_lag_incidence: np.ndarray):
    """Dynare's decision-rule ordering: static, purely backward, mixed, purely forward."""
    lag = lead_lag_incidence[:, 0] > 0
    lead = lead_lag_incidence[:, 2] > 0
    idx = np.arange(len(lag))
    static, pred, both, fwrd = idx[~lag & ~lead], idx[lag & ~lead], idx[lag & lead], idx[~lag & lead]
    order_var = np.concatenate([static, pred, both, fwrd])
    return order_var, np.concatenate([pred, both]), dict(nstatic=len(static), npred=len(pred),
                                                         nboth=len(both), nfwrd=len(fwrd))


#: "reduced" (Dynare's dyn_first_order_solver algorithm: static variables removed by QR, QZ on the
#: dynamic block only) or "full" (the original 2n x 2n pencil). Environment variable
#: PYDYN_FIRST_ORDER overrides the default.
SOLVER = os.environ.get("PYDYN_FIRST_ORDER", "reduced")


def solve_linear_re(fm, f0, fp, fu, qz_criterium=QZ_CRITERIUM, lead_lag_incidence=None, method=None):
    """Solve fp E y(+1) + f0 y + fm y(-1) + fu u = 0 for y = G y(-1) + H u.

    ``lead_lag_incidence`` (n x 3, non-zero where a variable appears with a lag / lead) lets the
    reduced solver use the model's structure; without it the structure is read off the non-zero
    columns of ``fm`` and ``fp``. ``method`` is "reduced" or "full" (default: ``SOLVER``)."""
    method = SOLVER if method is None else method
    if method == "full":
        return _solve_full(fm, f0, fp, fu, qz_criterium)
    if method != "reduced":
        raise ValueError(f"unknown first-order solver {method!r}")
    if lead_lag_incidence is None:
        lag, lead = np.any(fm != 0, axis=0), np.any(fp != 0, axis=0)
    else:
        lag, lead = lead_lag_incidence[:, 0] > 0, lead_lag_incidence[:, 2] > 0
    return _solve_reduced(fm, f0, fp, fu, lag, lead, qz_criterium)


def _solve_reduced(fm, f0, fp, fu, lag, lead, qz_criterium=QZ_CRITERIUM):
    """Port of the QZ branch of Dynare's dyn_first_order_solver.m.

    Variables are put in decision-rule order (static, purely backward, mixed, purely forward).
    The static variables are eliminated with a QR factorisation of their columns of ``f0``; the
    generalised Schur decomposition is then taken of a pencil of size npred + 2*nboth + nfwrd
    instead of 2n."""
    n = f0.shape[0]
    key = (lag.tobytes(), lead.tobytes())
    st = _STRUCT.get(key)
    if st is None:
        idx = np.arange(n)
        static, pred, both, fwrd = idx[~lag & ~lead], idx[lag & ~lead], idx[lag & lead], idx[~lag & lead]
        ov = np.concatenate([static, pred, both, fwrd])
        ns, npred, nboth, nfwrd = len(static), len(pred), len(both), len(fwrd)
        sv = ov[ns:ns + npred + nboth]
        if len(_STRUCT) > 64:
            _STRUCT.clear()
        st = _STRUCT[key] = (ov, sv, ov[ns + npred:], np.ix_(ov, sv), ns, npred, nboth, nfwrd)
    ov, sv, jv, ov_sv, ns, npred, nboth, nfwrd = st
    nspred, nsfwrd, ndyn = npred + nboth, nboth + nfwrd, n - ns
    nd = nspred + nsfwrd
    k = fu.shape[1]
    # [A | B | C | fu]: lagged states, all current variables, led jumpers, shocks (DR order)
    aa = np.empty((n, nspred + n + nsfwrd + k), order="F")
    aa[:, :nspred] = fm[:, sv]
    aa[:, nspred:nspred + n] = f0[:, ov]
    aa[:, nspred + n:nspred + n + nsfwrd] = fp[:, jv]
    aa[:, nspred + n + nsfwrd:] = fu
    if not np.all(np.isfinite(aa)):
        raise BlanchardKahnError("the Jacobian of the dynamic model contains NaN or Inf")
    if ns:
        qr_, tau, _, _ = lapack.dgeqrf(aa[:, nspred:nspred + ns])
        aa, _, info = lapack.dormqr("L", "T", qr_, tau, aa, max(64 * aa.shape[1], 1), overwrite_c=1)
        if info:
            raise BlanchardKahnError("QR factorisation of the static block failed")
    A, B = aa[:, :nspred], aa[:, nspred:nspred + n]
    C, FU = aa[:, nspred + n:nspred + n + nsfwrd], aa[:, nspred + n + nsfwrd:]

    if nd:
        D, E = np.zeros((nd, nd), order="F"), np.zeros((nd, nd), order="F")
        D[:ndyn, :npred] = B[ns:, ns:ns + npred]
        D[:ndyn, nspred:] = C[ns:]
        E[:ndyn, :nspred] = -A[ns:]
        E[:ndyn, nspred:] = -B[ns:, ns + npred:]
        if nboth:
            r = np.arange(nboth)
            D[ndyn + r, npred + r] = 1.0
            E[ndyn + r, nspred + r] = 1.0
        S, T, alpha, beta, w, sdim = _ordered_qz(E, D, qz_criterium)
        with np.errstate(divide="ignore", invalid="ignore"):
            eig = alpha / beta
        if np.any((np.abs(alpha) <= QZ_ZERO_THRESHOLD) & (np.abs(beta) <= QZ_ZERO_THRESHOLD)):
            raise BlanchardKahnError("one of the generalised eigenvalues is close to 0/0 "
                                     "(the model has a singularity: Dynare's error 7)")
    else:
        S = T = w = np.zeros((0, 0))
        eig, sdim = np.zeros(0), 0
    if sdim != nspred:
        kind = "indeterminacy" if sdim > nspred else "no stable solution"
        raise BlanchardKahnError(
            f"Blanchard-Kahn conditions not satisfied ({kind}): {nd - sdim} unstable roots for "
            f"{nsfwrd} forward-looking variables")
    w11, w12, w22 = w[:nspred, :nspred], w[:nspred, nspred:], w[nspred:, nspred:]
    if nsfwrd:
        lu, piv, info = lapack.dgetrf(w22.T)
        rc = lapack.dgecon(lu, np.abs(w22).sum(axis=1).max())[0] if info == 0 else 0.0
        if rc < 1e-9:
            raise BlanchardKahnError("rank condition not satisfied")
        gx = -lapack.dgetrs(lu, piv, w12.T)[0]                     # jumpers on states
    else:
        gx = np.zeros((0, nspred))
    if nspred:
        # hx = w11 T11^-1 S11 w11^-1: transition matrix of the states
        hx = lapack.dtrtrs(T[:nspred, :nspred], w11.T, lower=0, trans=1)[0].T
        hx = hx @ np.linalg.solve(w11.T, S[:nspred, :nspred].T).T
    else:
        hx = np.zeros((0, 0))
    ghx = np.empty((n, nspred))
    ghx[ns:ns + nspred] = hx
    ghx[ns + nspred:] = gx[nboth:]
    Cgx = C @ gx
    if ns:
        rhs = -(Cgx[:ns] @ hx) - A[:ns] - B[:ns, ns:] @ ghx[ns:]
        ghx[:ns] = lapack.dtrtrs(B[:ns, :ns], rhs, lower=0)[0]
    A_ = B.copy()
    A_[:, ns:ns + nspred] += Cgx
    G, H = np.zeros((n, n)), np.empty((n, k))
    G[ov_sv] = ghx
    H[ov] = -np.linalg.solve(A_, FU) if k else np.zeros((n, 0))
    return G, H, eig


QZ_ZERO_THRESHOLD = 1e-6      # Dynare's options_.qz_zero_threshold
_STRUCT: dict = {}            # index sets per lead/lag structure


def _ordered_qz(E, D, qz_criterium):
    """Real generalised Schur form of (E, D) with the eigenvalues below ``qz_criterium`` in modulus
    first. Returns S, T, alpha, beta, Z, sdim.

    This is the LAPACK call Dynare's mjdgges makes (dgges with a selection function). SciPy's
    ``ordqz`` (dgges followed by dtgsen) is the fallback: it gives the same decomposition but costs
    20 to 40% more, and it is used on Windows, where SciPy avoids LAPACK callbacks."""
    c2 = qz_criterium * qz_criterium
    if os.name != "nt":
        S, T, sdim, ar, ai, beta, _, Z, _, info = lapack.dgges(
            lambda re, im, b: re * re + im * im < c2 * b * b, E, D, jobvsl=0, jobvsr=1, sort_t=1,
            overwrite_a=1, overwrite_b=1)
        if info == 0:
            return S, T, (ar + 1j * ai if np.any(ai) else ar), beta, Z, int(sdim)
        if info < 0 or info <= E.shape[0]:
            raise BlanchardKahnError(f"generalised Schur decomposition failed (LAPACK dgges info={info})")
        # info = n+2 / n+3: reordering changed or could not order the eigenvalues; let ordqz decide
    stable = lambda alpha, beta: np.abs(alpha) < qz_criterium * np.abs(beta)  # noqa: E731
    S, T, alpha, beta, _, Z = linalg.ordqz(E, D, sort=stable, output="real", check_finite=False)
    return S, T, alpha, beta, Z, int(np.sum(stable(alpha, beta)))


def _solve_full(fm, f0, fp, fu, qz_criterium=QZ_CRITERIUM):
    """The original solver: QZ of the full 2n x 2n pencil. Kept as a cross-check."""
    n = f0.shape[0]
    I, Z0 = np.eye(n), np.zeros((n, n))
    # A x_{t+1} = B x_t with x_t = [y_{t-1}; y_t]
    A = np.block([[I, Z0], [f0, fp]])
    B = np.block([[Z0, I], [-fm, Z0]])
    stable = lambda alpha, beta: np.abs(alpha) < qz_criterium * np.abs(beta)  # noqa: E731
    S, T, alpha, beta, Q, Z = linalg.ordqz(B, A, sort=stable, output="real")
    with np.errstate(divide="ignore", invalid="ignore"):
        eig = alpha / beta
    n_stable = int(np.sum(np.abs(alpha) < qz_criterium * np.abs(beta)))
    if n_stable != n:
        kind = "indeterminacy" if n_stable > n else "no stable solution"
        raise BlanchardKahnError(
            f"Blanchard-Kahn conditions not satisfied ({kind}): {n_stable} stable roots for {n} variables")
    Z11, Z21 = Z[:n, :n], Z[n:, :n]
    if np.linalg.cond(Z11) > 1e12:
        raise BlanchardKahnError("rank condition not satisfied")
    G = np.linalg.solve(Z11.T, Z21.T).T
    H = -np.linalg.solve(fp @ G + f0, fu)
    return G, H, eig


def first_order(model, ys=None, params=None, qz_criterium=None, loglinear=None) -> FirstOrderSolution:
    """First-order solution. ``qz_criterium`` defaults to the value set in the .mod file
    (``stoch_simul``/``check``/``estimation``), else Dynare's 1 + 1e-6.

    ``loglinear`` (default: the .mod file's ``stoch_simul``/``estimation`` option) returns the rules in
    log deviations as Dynare does: ``ys``, ``ghx``, ``ghu`` and everything computed from them
    (``simulate``, ``irf``, moments) are then in logs, except for the auxiliary variables of leads and
    lags of exogenous variables, which Dynare keeps in levels. ``steady_state`` keeps the levels.
    """
    from . import options as _opt
    from .steady import steady_state

    if loglinear is None:
        loglinear = bool(_opt.stoch_options(model).get("loglinear", False))
    for key in ("partial_information", "aim_solver", "dr", "dr_cycle_reduction_tol", "sylvester"):
        if key in _opt.stoch_options(model):
            _opt.warn_once(model, f"the .mod file sets {key}: pydyn always uses its QZ-based solver")

    if ys is None:
        ys, params = steady_state(model, params)
    params = model.params if params is None else params
    if loglinear:                                        # Dynare checks this before solving (info 26)
        mask = log_mask(model)
        if np.any(~(np.asarray(ys, float)[mask] > 0)):
            bad = [model.endo_names[i] for i in np.nonzero(mask & ~(np.asarray(ys, float) > 0))[0]]
            raise LoglinearError("The loglinearization of the model cannot be performed, because the steady "
                                 f"state is not strictly positive: {', '.join(bad)}")
    n = model.n_endo
    J = model.dynamic_jac(np.tile(ys, 3), model.exo_steady(), params, ys)
    fm, f0, fp, fu = J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:]
    G, H, eig = solve_linear_re(fm, f0, fp, fu, qz_criterium=_opt.qz_criterium(model, qz_criterium),
                                lead_lag_incidence=model.lead_lag_incidence)
    order_var, state_var, _ = dr_ordering(model.lead_lag_incidence)
    # clean numerical noise in columns of non-state variables
    mask = np.ones(n, bool)
    mask[state_var] = False
    G[:, mask] = 0.0
    levels = np.array(ys, float)
    if loglinear:
        ys, G, H = loglinearize(levels, G, H, log_mask(model))
    return FirstOrderSolution(ys=ys, G=G, H=H, order_var=order_var, state_var=state_var,
                              eigenvalues=eig, jacobian=J, params=np.array(params, float),
                              loglinear=bool(loglinear), steady_state=levels)

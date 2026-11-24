# OBC methods in JDyn and pydyn: shared contract

Date: 2026-11-24. Companion to `claude/dynareobc-design-note.md` (background, comparison of methods, tiers). This note fixes what the two ports must agree on so that code and lessons can be swapped between them. Where it differs from the design note's sections 6.2 to 6.4, this note is the current one.

Status: agreed with Jacob in outline; details below are a first version and should be amended (with a dated line) when an implementer finds a problem.

## 1. Scope of this phase

- Order 1 only. Same simulation semantics as OccBin: each period the path is solved with no future shocks expected.
- One user-facing specification: the `occbin_constraints` block and `bind`/`relax` equation tags.
- Three methods behind one switch:

| `obc_method` | Method | First implementation |
|---|---|---|
| `occbin` (default) | Existing piecewise-linear solver, untouched | Done in both ports |
| `minnorm` | MacroModelling.jl's approach: smallest news shocks that enforce the constraint | JDyn (agent), then pydyn |
| `news` | Holden's LCP solved as a mixed-integer programme, with diagnostics on `M` | pydyn (main session), then JDyn |

- Out of scope for now: orders 2 and 3, cubature, estimation, constraints that switch several equations.

**State on 2026-11-24, 13:15, superseding the lines above where they differ** (details in sections 3 to 6, each marked with the date):
- `news` exists in both ports, in two formulations chosen automatically: the LCP when every constraint is one complementary pair, the general regime formulation otherwise (section 3a). Constraints that switch several equations are therefore in scope.
- `minnorm` exists in JDyn only. Decision: it is not ported to pydyn as a solver. In pydyn it becomes an alias for `news` with `obc_selection = min_wedge`, so mod files written for JDyn run; JDyn is to become the same alias once selection is ported there.
- Equilibrium selection is an option, `obc_selection`, default `occbin` (section 5a). In progress in pydyn.
- Estimation through the inversion filter with the news solver: prototype in pydyn, being made an option. Not in JDyn yet.
- Solver dependency: HiGHS in both ports (SciPy in pydyn; `HiGHS_jll` called directly as a hard dependency in JDyn).
- Next phase (decided): orders 2 and 3 and cubature. Design: `claude/obc-higher-order-design.md`.

## 2. What the 7.2 preprocessor gives us (checked on the binary, 2026-11-24)

- `occbin_constraints` appears in the JSON statements with `name`, `bind`, `relax`, `error_bind`, `error_relax` per constraint.
- The two tagged versions of an equation are **merged into one equation** multiplied by a parameter `occbin_<name>_bind`:

  ```
  [name='zlb', relax='zlb']  r = rn;
  [name='zlb', bind='zlb']   r = rbar;
  -->  (r-rn)*(1-occbin_zlb_bind) + occbin_zlb_bind*(r-rbar) = 0
  ```

  So both residuals are recovered by evaluating the model at `occbin_<name>_bind` = 0 and = 1. No text parsing of equations is needed.
- The block is capped at two constraints: a third gives `ERROR: only up to two constraints are supported`. Equation tags naming a third constraint are accepted and create its parameter, so a third constraint could later be declared through a native line. Not in this phase.
- Unknown options are rejected: `occbin_solver(obc_method=news)` and `occbin_setup(obc_method=news)` are syntax errors.
- A native line passes through: `options_.occbin.obc_method = 'news';`. It must be on its own line; anything after it on the same line is swallowed into the native statement.

## 3. The constraint object (output of the post-processor)

OccBin's "relax" regime is the reference regime (the one holding at the steady state). Its "bind" regime is the alternative. The constraint may be slack or binding in the reference regime (Dynare's borrowing-constraint example has it binding there), so this note uses **reference** and **alternative** throughout.

For each constraint the post-processor returns:

| Field | Meaning |
|---|---|
| `name` | Constraint name |
| `param` | Index of `occbin_<name>_bind` |
| `equations` | Indices of the equations that contain that parameter |
| `ref_resid`, `alt_resid` | Residual of each such equation with the parameter at 0 and at 1, as affine functions of the variables at the steady state (constant plus gradient over leads, current, lags) |
| `gap` | `g = s_g * alt_resid`, evaluated along a path. Normalised so that the `bind` condition is equivalent to `g < 0` |
| `wedge` | `y = s_y * ref_resid`. Normalised so that leaving the alternative regime (the `relax` condition, or the negation of `bind` when none is given) is equivalent to `y < 0` |
| `class` | `complementarity`, `multi_equation` or `inconsistent` |
| `why` | For the last two classes, a one-line reason for the user |

With those normalisations every period satisfies `g >= 0`, `y >= 0`, `g*y = 0`: in the reference regime `y = 0` and `g >= 0`; in the alternative regime `g = 0` and `y >= 0`.

Classification rules:

1. More than one equation contains the parameter: `multi_equation`.
2. Entry test. Linearise `d_b = lhs - rhs` of the `bind` condition (written as `d_b < 0`). It must equal `a * alt_resid + b * ref_resid` for scalars with `a != 0`, to tolerance, as affine functions. Then `s_g = sign(a)`. Reason: inside the reference regime `ref_resid = 0`, so the condition is a statement about the gap alone.
3. Exit test. Write the `relax` condition as `d_r > 0` meaning "leave the alternative regime". When no `relax` is given, OccBin leaves when the `bind` condition fails, so `d_r = d_b` (same function, not its negative). `d_r` must equal `c * ref_resid + e * alt_resid` with `c != 0`. Then `s_y = -sign(c)`, so that leaving is `y < 0`. *(Amended 2026-11-24: the first version had `-d_b` and `sign(c)`, both wrong. A constraint without `relax` then gets a negative diagonal in `M`.)*
4. `g` at the steady state must be strictly positive.
5. Any failure of 2 to 4: `inconsistent`, with the reason.
6. Tolerance of the fits in 2 and 3 *(added 2026-11-24)*: exact when the relative residual is below 1e-6. Up to 0.05 the constraint is still `complementarity` but carries `approx` = the residual. This is needed for lower-bound models written as `inom = inomlb*exp(mp)` with `bind inomnot <= inomlb` (Dynare's NKM test files, residual 0.6%): the linearised equations have a term in `mp` that the condition lacks. OccBin switches on the condition and the news methods on the linearised residual, so paths may differ by a term of that order (on Dynare's NKM file they agreed to 1e-14 in the runs made).

7. Orientation test *(added 2026-11-24, from the JDyn implementation; to be added to pydyn)*: with the fits of rules 2 and 3 written as `d_b = a*alt + b*ref` and `d_r = c*ref + e*alt`, require `b*s_y <= 0` and `e*s_g >= 0`. A model whose `relax` condition is the reverse of its `bind` condition passes rules 2 and 3 but has no consistent solution (OccBin cycles on it); this rule classifies it `inconsistent`.

Amendments to the rules above *(2026-11-24)*:
- Rule 1: the switched equations are those whose **linearisation** differs between the regimes, not those whose text contains the parameter. (In `rbc_occbin` an equation contains the INEG parameter but differs only at second order.)
- A further class, `shared_equation`: an equation whose version depends on two constraints jointly. Refused by every news method. No Dynare file is in this class.

`minnorm` runs only on `complementarity` constraints and raises a clear error otherwise. The original plan was the same for `news`; it now handles the other classes through section 3a.

### 3a. The general regime formulation *(added 2026-11-24)*

For `multi_equation` and `inconsistent` constraints (and available for any model):
- Each switched equation gets its own wedge on its reference version. In a period spent in the alternative regime the wedges are whatever makes the alternative versions hold; otherwise they are zero.
- A regime sequence is accepted by OccBin's own test: the `bind` condition false wherever the reference regime is assumed, the `relax` condition false wherever the alternative is assumed, conditions evaluated in levels. The answer is therefore OccBin's piecewise-linear solution, found from precomputed wedge responses with one small linear solve per guess.
- Tied wedges: for one constraint with several equations, if all but one switched equation respond to wedges only in the same period with constant coefficients, and the conditions are exact mirror images of the remaining ("primary") equation, the model is a complementarity problem in one wedge per period, `g = q + M y + G z` (`z` the regime indicator, `G` state-dependent offsets, zero when no shock enters the tied equations). Then `M`'s diagnostics apply, single-spell solutions can be enumerated, and an infeasible linear relaxation (`y >= 0`, `0 <= z <= 1`, `q + M y + G z >= 0`) is a proof that no solution exists within the horizon.
- Order of attempts: guess-and-verify from the no-binding guess; the tied-wedge problem (solve or prove infeasible); a big-M integer programme over regime indicators with a time limit as the last resort.
- Models flagged `approx` (rule 6): the LCP and OccBin can pick different regimes at a knife-edge period (seen on Dynare's NKM file about once in 1,000 periods at large shocks, and once in 40 parameter vectors in a likelihood, where it moved the log posterior by 0.015). The regime formulation uses OccBin's test and agreed with OccBin in every such case. Open choice: default to the regime formulation for `approx` models, at about 3 to 5 times the cost per period.

Worked cases the tests must cover:

| Model | Reference equation | Alternative equation | `g` | `y` |
|---|---|---|---|---|
| Lower bound (`bind rn < rbar`) | `r = rn` | `r = rbar` | `r - rbar` | `r - rn` |
| Dynare borrowing constraint (`bind lb < -lb_ss; relax b > m*y`) | `b = m*y` | `lb = 0` | `lb` (level) | `m*y - b` |

## 4. `M` and `q`

- Horizon `T` (`obc_horizon`, default 40). With `n_c` constraints, stack by constraint then by period.
- Wedge path: `ref_resid_j(t) = s_y * y_j(t)` for `t = 0..T-1`, known to agents from period 0, zero from `T` on. Everything else is the reference regime, and the path returns to the reference steady state.
- `M[(i,t),(j,s)] = d g_i(t) / d y_j(s)`: response of gap `i` at `t` to a unit wedge on constraint `j` at `s`.
- `q[(i,t)]`: gap `i` at `t` with all wedges zero, given the initial state and this period's shocks. At order 1, `q = q0 + Q * [state; shocks]`.
- Constrained path: `g = q + M y`.
- Either construction is acceptable (stacked perfect-foresight linear system, or the first-order solution with anticipated shocks as auxiliary states). Tests must show the chosen one agrees with a stacked-system computation to 1e-10.
- `M` depends only on parameters: cache it per parameter set. `Q` likewise. Precomputing `Q` matters: in pydyn it is what makes the news method about five times faster than OccBin.
- Reference values for `M` and `q` on the lower-bound model: pydyn `reference/obc_zlb.mat` (branch `wip/obc`), made by `reference/r_obc_zlb.m` from Dynare 7.2 under Octave with the news shocks written into the mod file by hand (`obc_zlb_news.mod`). It also holds the wedges from GLPK and Dynare's OccBin path.
- After solving, the gap must also be checked beyond `T` (reuse OccBin's check-ahead length). A violation there means the horizon is too short and is reported as such, not as "no solution".

## 5. Solver interface

One function per method with the same signature, so they are interchangeable:

```
solve(M, q; options) -> (y, status, info)
```

- `status`: `ok`, `no_solution`, `horizon_too_short`, `not_converged`.
- Every solver verifies its own answer before returning `ok`: `y >= -tol`, `q + M y >= -tol`, `|y .* (q + M y)| <= tol`.
- Shortcut shared by all: if `q >= 0`, return `y = 0`.
- Fast path shared by all *(added 2026-11-24)*: an active-set iteration on `M` (start from the periods with `q < 0`, solve the square system on that set, drop negative wedges, add violated periods, repeat). In pydyn it solved every binding period of every test; the method's own solver is then only the fallback.
- The verification step is not optional. GLPK under Octave with its presolver on returned an "optimal" point that violated a constraint by 0.7.
- `minnorm`: follow MacroModelling.jl's formulation (read its source; MIT licence, keep attribution in `NOTICE.md`). Record in the progress log exactly what objective and solver it uses and where JDyn departs from it.
- `news`: integer programme with HiGHS; big-M constants derived from `q` and `M`, not fixed. Reports whether `M` passes Holden's sufficient conditions.

Simulation loop: identical to OccBin's `simul` loop, with the regime iteration replaced by `solve`. The regime history is derived from `y > tol`.

Notes from the implementations *(2026-11-24)*:
- The integer programme is a fallback only. Forced on every binding period it costs 1.8 to 32 ms against 8 to 14 microseconds for the normal path (JDyn); in every test on well-behaved models the active-set iteration solved all binding periods.
- After the integer programme, re-solve the square system on the binding set it reports, then verify.
- Before the integer programme, an infeasible linear programme (`y >= 0`, `q + M y >= 0`) proves there is no solution; it is cheap and should come first wherever failures are expected (the filter's root finder tries implausible shocks).
- Verification tolerance differs between the ports (pydyn: relative to `max(1, |q|)`; JDyn: absolute 1e-10). To be reconciled once the JDyn measurement run reports whether any decision depends on it.
- Diagnostics on `M` cost about 10 ms per parameter set: compute on request, never inside an estimation loop.

### 5a. Equilibrium selection *(decided 2026-11-24; in progress in pydyn)*

Option `obc_selection` for `obc_method = news`:

| Value | Meaning |
|---|---|
| `occbin` (default) | The solution the guess-and-verify / active-set iteration reaches from the no-binding guess. This is what OccBin returns. |
| `shortest` | Among single-spell solutions at the state, the fewest binding periods (ties: earliest start) |
| `longest` | The most binding periods (ties: earliest start) |
| `min_wedge` | The smallest sum of squared wedges |

Non-default values enumerate single-spell solutions in binding periods only; if none exists they fall back to the default and flag the period; with more than one constraint they are refused. `obc_method = minnorm` in pydyn is `news` with `min_wedge`.

## 6. Options and results

- In the mod file, on its own line before `occbin_setup`: `options_.occbin.obc_method = 'minnorm';` Also `options_.occbin.obc_horizon = 40;`.
- On the live workspace: keyword `obc_method` (and `obc_horizon`) on the OccBin simulate and IRF functions in each port.
- Unknown method names are an error.
- `options_.occbin.obc_selection = 'occbin';` and the keyword `obc_selection`, in the same places *(added 2026-11-24)*.
- Estimation *(added 2026-11-24)*: the same options select the solver inside the inversion filter. The piecewise Kalman filter with a news method is refused until it is implemented.
- Results go where OccBin's go (`oo_.occbin.simul.*` in JDyn, the same result object in pydyn), so `irf_occbin`, graphs and tables work unchanged.
- Extra results under `oo_.occbin.news`: `method`, the wedges per period, `status` per period, and for `news` the diagnostics on `M`.

## 7. Tests each implementation must pass

1. Post-processor classification on every OccBin `.mod` file in Dynare 7.2's `tests/` and `examples/`: a table of file, constraint, class, reason. Kept in the repo as a doc and as a regression test.
2. `M` against the stacked-system computation (1e-10).
3. On every `complementarity` model: paths under the new method equal the port's own OccBin paths (which already match Dynare to about 1e-15) to solver tolerance, with identical regime histories, for the file's own shocks and for a seeded random-shock simulation of 200 periods. Any disagreement is recorded and explained, not tuned away: OccBin can converge to a different equilibrium or fail where the new method succeeds.
4. Refusals: `multi_equation` and `inconsistent` models raise the documented error.
5. Timing of the method against OccBin on the lower-bound example and on `rbc_occbin`, recorded in the progress log.

## 8. Working rules for shared repos

- Another session is working on JDyn (`dseries`, `ramsey`, `pac` branches). OBC work goes on its own branch (`obc`), is backed up to `wip/obc`, and is never pushed to `main`. The main session integrates.
- Do not edit `src/matlab/*` in JDyn (owned by the `dseries` stream). Keep changes to shared files (`src/JDyn.jl`, `src/results.jl`, `test/runtests.jl`) small and additive.
- New code goes in new files (`src/obc.jl`, `src/obc_solvers.jl`, `test/test_obc.jl`), calling into `occbin.jl` rather than rewriting it.
- Progress log in the repo (`docs/OBC_PROGRESS.md`): what is done, what was learned, open problems, and a "lessons for the other port" section.

# Failed cubature points: what they are, what the package does, what you can change

This page is for someone who uses the higher-order, cubature option of the occasionally-binding-constraint (OBC) solver (`HigherOrderNews`, `obc_method='news'` with `cubature=True`)
and sees a warning or an error about "failed points". You do not need to know how the solver works inside.

## What cubature does, in one paragraph
With a constraint such as a lower bound on the interest rate, today's behaviour depends on the chance that the bound binds later. The first-order and second-order solutions treat the future as if the
shocks of the coming periods were at their average (zero). **Cubature** corrects this: each period the solver draws a few thousand possible *paths* for the coming periods (random combinations of future
shocks, spread the way the model says they are spread), works out for each path what the bound would do to the economy, and **averages** the answers. Only paths on which the bound would bind
("relevant" paths) need any work; on all others the bound has no effect and contributes zero. The average is what agents are assumed to expect, so it changes today's allocation even in periods in which the bound is
far away. The default is the full average over 4,095 paths (`qmc_level=11`).

## What a "failed point" is
A *point* is one such path. It **fails** when the solver cannot find a consistent answer for it: a way for the constraint to bind and relax over the coming periods that satisfies all the equations of the model.
There are three different reasons, and they mean different things.

| kind | name in the output | what it means | economic reading |
|---|---|---|---|
| (a) | `a`, *proven no solution* | A mathematical test (a linear programme) shows that no solution exists at all for this path, whatever regimes you try within the horizon. | The model has states, which agents give weight to through the shocks they expect, from which **no equilibrium exists**. This is a property of the model or its calibration (typically an expectations trap: no way to meet the constraint and the equations together). |
| (b) | `b`, *horizon too short* | The bound would still be binding when the news horizon (`horizon`, default 40 periods) ends, so the problem has not been solved for long enough. | A numerical limit: a longer horizon may fix it (this is not guaranteed). |
| (c) | `c`, *undecided* | The solver did not find a solution and could not prove there is none: the integer search ran out of its time limit (`cubature_milp_time_limit`), or it found a pattern of binding periods that fails the model's own binding test. | Typically a knife-edge path in the far tail, where the answer depends on rounding-sized differences. Not evidence of anything about the model. |

Only (a) is *established*. (b) and (c) mean "the solver gave up", not "there is no solution".

### Why (c) appears for some models and not for others
For a constraint that the package classifies as *exactly complementary* (for example a plain lower bound on a rate that replaces the rule), the fast method (the "LCP", Holden's formulation) is used. It always returns an answer
when one exists, and (c) hardly arises. For an *approximate* constraint (the two versions of the equation differ by a term that the bound condition does not contain, as in the NKM model of the Dynare test suite), or one that switches
several equations (multi-equation, e.g. the Smets-Wouters model with a lower bound), `obc_formulation='auto'` uses the **regime formulation**. It insists on OccBin's own test: the bound is binding exactly when OccBin's binding condition
says so, evaluated on the path in levels. That is the same solution OccBin would report, but there are a few knife-edge paths on which no regime sequence passes the test, and the solver reports these as (c). The fast method would have returned a
number for them, one that OccBin would not accept. (The differences between the two formulations on approximate-class models are discussed in docs/OBC_ESTIMATION.md.)

## What happens by default
| situation | default | why |
|---|---|---|
| a point of kind (a) in a period | **error** (`CubatureFailure`), at once | A proven absence of equilibrium is information about the model. Averaging over the remaining paths would hide it and give a number that looks like an answer. |
| points of kind (b) or (c) in a period | **dropped**, if there are at most 0.5% of that period's relevant points, and in any case up to 3 points; **error** above that | They are not proven to have no solution; they are numerical tail points. A few of them should not stop a long simulation. |
| the dropped points | the average is taken over the remaining points (the sum of the answers divided by all points minus the dropped ones) | |
| after the simulation | **one warning** with the totals | So that you notice. |

The floor of 3 points exists because a period can have only a few hundred relevant points: 0.5% of 279 is 1.4, so two failures in one period (0.7%) would otherwise stop the run although the failures are rare overall.
Measured on the NKM model with 100 periods (level 11): 26 of 70,412 relevant points (0.04%) fail, spread over 20 periods, never more than 3 in a period.

## Reading the warning and the results
Example (NKM, default settings, 100 periods, level 11):

    cubature: 26 of 70412 relevant perturbed points (0.037%) were dropped in 20 of 100 periods (by class: a=0 proven no solution,
    b=0 horizon too short, c=26 undecided; largest share in one period 1.00%; allowed: 0.5% of a period's relevant points, never fewer than 3).
    Classes (b) and (c) are tail points that the solver could not settle: they are NOT proven to have no solution. The remaining points are
    reweighted by 1/(points - dropped), which biases the average slightly towards benign states. To reduce them: a longer `horizon` for (b);
    a larger `cubature_milp_time_limit` for (c). `cubature_undecided_tolerance=0` makes any failed point an error. See docs/OBC_CUBATURE_FAILURES.md.

"largest share in one period 1.00%" is one point out of 100 relevant points in a period with few relevant points: allowed by the floor of 3. Read the totals: a total share below about 0.1% is the normal level for approximate-class models;
a share of several tenths of a per cent in every period (the `bf` test model shows 0.4%) means the model sits closer to the edge and you may want to raise the tolerance consciously or inspect it.

`simulate` returns, per period: `relevant` (number of relevant points), `dropped` (total dropped), and `dropped_a`, `dropped_b`, `dropped_c` (by kind); the solver object keeps totals in
`stats["relevant_points"]`, `stats["dropped_points"]`, `stats["dropped_periods"]`, `stats["class_a"]`, `stats["class_b"]`, `stats["class_c"]`.

## The consequences of dropping points
Dropped points are tail points: paths with large shocks in which the bound would have bound hard. Removing them and renormalising gives slightly *too little* weight to bad states, so the average is biased a little towards benign
states (the wedge you get is slightly too small). How large is the effect? Measured (regime formulation, simulation of 100 / 60 periods, level 11, all in the units of the model's variables, maximum over periods and variables):

| model | dropped points | effect of the cubature itself | noise from the choice of points (sign patterns of the quasi-random points; standard deviation / range) | difference if dropped points counted as zero instead of renormalising | difference to the fast LCP method | difference with a 5 s time limit (settles more points) |
|---|---|---|---|---|---|---|
| NKM | 26 of 70,412 (0.04%) | 0.098 | 0.0022 / 0.0073 | 1.9e-5 | 2.1e-4 | 1.8e-15 (no point is settled) |
| constructed approximate model (`nkocc`, mismatch 0.5%) | 47 of 87,161 (0.05%) | 0.030 | 0.00043 / 0.0013 | 2.2e-5 | 2.6e-4 | 0 |

So the effect of dropping is about a hundredth of the noise of the integration itself and about a thousandth to a ten-thousandth of the cubature effect. If failures were 5% instead of 0.05% the bias would be roughly 100 times larger
(the effect is proportional to the share dropped); that is why the tolerance exists.

**Raising instead.** With `cubature_undecided_tolerance=0` the simulation stops at the first period with a failed point (on NKM at level 11, period 4). Nothing is biased, but long simulations of approximate-class models are not possible
(on NKM 20 of 100 periods have at least one failing point). Use it to find out where the failures are, not for production runs.

## Changing the settings
| setting | default | effect | when you would change it |
|---|---|---|---|
| `cubature_undecided_tolerance` | 0.005 | Share of a period's relevant points of kind (b) or (c) that may be dropped (floor `cubature_undecided_min_points`=3). 0 = any failed point is an error. | 0 for a strict check or to find the failing periods; 0.01 to 0.02 for models with a regularly larger share (the `bf` model had up to 0.96% in a period, 0.4% overall in a 15-period test). |
| `cubature_undecided_min_points` | 3 | Floor on the number of dropped points per period. | Rarely. |
| `cubature_failure_tolerance` | 0.0 | Share of relevant points of **any** kind that may be dropped, including kind (a). With a positive value a period is acceptable if the (a) points are at most this share *and* all failed points are at most the larger of this share and the undecided allowance. | Only to drop proven-no-solution points *knowingly*, for example to see how widespread they are. Risky: the answer you get is an average over the part of the state space in which an equilibrium exists. |
| `horizon` (and `check_ahead`) | 40 | Longer horizon removes kind (b) points; costs time. | When the warning shows b > 0 (the `borrcon` test model shows only kind (b): 1654 of 30,689 relevant points at horizon 40, level 9, whatever the time limit). |
| `cubature_milp_time_limit` | 0.1 s per point | Time the integer search may spend on one point. A longer limit can only move points from (c) to *solved*, (a) or (b); it never makes a point worse. | Measured with limits from 0.02 s to 1 s on NKM, a constructed model, `borrcon` and `bf` (level 9): the classification is the same from 0.05 s on; a longer limit settled one point of 75 on `bf` and none elsewhere, and made `bf` six times slower (281 to 1736 ms per period at level 9) and the two-constraint model five-fold. Raise it if you want to be sure; it costs time, not accuracy. (On NKM and the constructed model a 5 s limit also left every dropped point unsettled.) |
| `qmc_level` | 11 | Number of points, 2^(level+1) - 1. Fewer points: faster, noisier. | Level 9 for exploration. |

## Worked example 1: NKM (Dynare's `occest_nkm.mod`), approximate class
Run the default: you get the warning above and the run completes. Run with `cubature_undecided_tolerance=0`:

    CubatureFailure: cubature: at least 2 of 279 relevant perturbed points (4095 in total) have no usable solution: a=0 b=0 c=2; cubature_undecided_tolerance=0 ...
    (c) undecided: the integer programme did not settle the point in 0.1 s, or found a regime sequence that fails OccBin's test; try a larger `cubature_milp_time_limit` ...

Interpretation: two paths out of 279 relevant ones in that period could not be settled. They are not proven to have no solution; with a longer time limit the answer for them does not change (5 s: still undecided), so they
are knife-edge paths that OccBin's own iteration does not settle either. Dropping them changes the simulated path by about 2e-5, far below the integration noise.

## Worked example 2: Smets-Wouters model with a lower bound (the repository's `sw` test model), multi-equation
Run `HigherOrderNews(model, order=2, cubature=True, qmc_level=9)` and simulate: the first period raises

    CubatureFailure: cubature: at least 1 of 623 relevant perturbed points (1023 in total) have no usable solution: a=1 b=0 c=0 ... (a) PROVEN no solution: the model has states that agents give weight to ... from which no equilibrium exists. ...

Interpretation: this is kind (a). In an earlier test with the failure tolerance set to 0.9 (to count rather than stop), about 58% of the relevant points failed. Many of the paths that agents would
consider (large combinations of future shocks) lead to a state in which the bound cannot be reconciled with the model's equations (the linear programme proves that the equations plus the bound have no solution there). This is not a numerical problem: a longer horizon or time limit would not remove it. Averaging over only those paths that have a solution would give a number that no longer represents
agents' expectations. The package therefore stops. What to do: check the calibration (is the shock variance realistic? is the model's behaviour at the bound sensible?), use smaller shocks, or use the first or second order
solution without cubature, which does not look at these tail states.

## The existence diagnostic works one way
The package can show that no solution exists (kind (a)). It cannot show that one *does* exist: a point of kind (c) or a path the solver happens to solve is not evidence that the model has an equilibrium at the neighbouring states.
The same holds for the up-front matrix diagnostic (`m_diagnostic`): it can warn that some paths may have no solution, but passing it does not guarantee that none fail.

## Related
The rule is implemented in `HigherOrderNews` (fast method) and `HigherOrderRegime` (regime formulation) identically; technical description in docs/OBC_REGIME_HO.md and docs/OBC_HO_PERF.md; the choice between the two formulations in docs/OBC_ESTIMATION.md.

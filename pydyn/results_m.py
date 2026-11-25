# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Field names and layouts follow Dynare 7.2's M_ (matlab/*.m, Copyright (C) Dynare Team, GPL-3.0-or-later).
"""``M_`` for the results mode (see :mod:`pydyn.results`): built from the model, kept in step by the session."""
from __future__ import annotations

import numpy as np

from .matlab import Cell, Struct, Tainted
from .matlab import _F
from .perturbation import dr_ordering
from .results import cellstr, col, empty, mat, row

#: fields of Dynare's M_ that pydyn does not fill: reading one is refused (it is not "non-existent").  The *_partitions
#: fields are the exception: reading one from MATLAB code loads them on demand (results.getfield -> load_partitions,
#: a second preprocessor run), so they are listed here only so that M_ without them is not "wrong"
KNOWN_UNSUPPORTED = {
    "bvar", "xref1", "xref2", "osr", "instr_id", "endo_partitions", "param_partitions", "heterogeneity_aggregates",
    "database", "histval_dseries", "surprise_shocks", "learnt_shocks", "learnt_endval", "shock_paths", "heteroskedastic_shocks",
    "matched_irfs", "matched_irfs_weights", "perfect_foresight_controlled_paths", "ramsey_orig_eq_nbr",
    "ramsey_orig_endo_nbr", "epilogue_names", "epilogue_var_list_", "mapping", "block_structure", "endo_trends",
    "dynamic_g1_sparse_rowval", "dynamic_g1_sparse_colval", "dynamic_g1_sparse_colptr", "dynamic_g2_sparse_indices", "dynamic_g3_sparse_indices", "ramsey_multipliers_static_g1_sparse_rowval",
    "ramsey_multipliers_static_g1_sparse_colval", "ramsey_multipliers_static_g1_sparse_colptr", "objective_tmp_nbr",
    "objective_g1_sparse_rowval", "objective_g1_sparse_colval", "objective_g1_sparse_colptr", "objective_g2_sparse_indices",
    "dynamic_tmp_nbr", "lhs", "dynamic_mcp_equations_reordering", "static_tmp_nbr", "block_structure_stat",
    "static_g1_sparse_rowval", "static_g1_sparse_colval", "static_g1_sparse_colptr", "static_mcp_equations_reordering",
    "csn", "det_shocks", "exo_det_histval", "static_and_dynamic_models_differ", "has_external_function",
    "parameter_used_with_lead_lag", "xref", "orig_eq_nbr", "equations_tags", "pac", "var", "var_expectation",
    "trend_component", "occbin", "exo_names_orig_ord", "NNZDerivatives", "sigma_e_is_diagonal_unknown",
}


def _row10(a):
    """A row vector; ``zeros(1, 0)`` (not 0 x 0) when empty, as Dynare's M_.state_var of a model without states."""
    a = np.asarray(a, float)
    return np.zeros((1, 0)) if a.size == 0 else row(a)


def build_M(model, mod_name: str) -> Struct:
    """Dynare's M_ for ``model`` before any statement has run (parameters NaN, no shocks)."""
    raw, info = model.raw, model.info
    n, nx = model.n_endo, model.n_exo
    M = Struct()
    M["fname"] = mod_name
    M["dynare_version"] = "7.2"
    M["dname"] = mod_name
    M["params"] = col(np.full(len(model.param_names), np.nan))
    M["endo_histval"] = empty()
    M["exo_histval"] = empty()
    M["Correlation_matrix"] = mat(np.eye(nx)) if nx else empty()
    M["Correlation_matrix_ME"] = mat(np.eye(1))
    for key, fam in (("exo", "exogenous"), ("endo", "endogenous"), ("param", "parameters")):
        items = raw.get(fam, [])
        names = [d["name"] for d in items]
        M[f"{key}_names"] = cellstr(names)
        M[f"{key}_names_tex"] = cellstr([d.get("texName", d["name"]) for d in items])
        M[f"{key}_names_long"] = cellstr([d.get("longName", d["name"]) for d in items])
        if key == "exo":
            det = raw.get("exogenous_deterministic", [])
            if det:
                M["exo_det_names"] = cellstr([d["name"] for d in det])
                M["exo_det_names_tex"] = cellstr([d.get("texName", d["name"]) for d in det])
                M["exo_det_names_long"] = cellstr([d.get("longName", d["name"]) for d in det])
    M["exo_det_nbr"] = _F(model.n_exo_det)
    M["exo_nbr"] = _F(nx)
    M["endo_nbr"] = _F(n)
    if any(st.get("statementName") == "discretionary_policy" for st in model.statements):
        M["discretionary_orig_eq_nbr"] = _F(model.jac_dims[0])          # written by the preprocessor with discretionary_policy
    M["param_nbr"] = _F(len(model.param_names))
    M["orig_endo_nbr"] = _F(raw.get("orig_endo_nbr", n))
    M["aux_vars"] = empty() if not raw.get("aux_vars") else Tainted("M_.aux_vars is a struct array, which pydyn does not represent")
    M["Sigma_e"] = mat(np.zeros((nx, nx)))
    M["Skew_e"] = np.zeros((0, 4))
    M["H"] = _F(0.0)
    M["sigma_e_is_diagonal"] = np.bool_(True)
    M["eq_nbr"] = _F(len(raw.get("model", [])) or n)
    if model.max_deriv_order >= 2:         # the preprocessor output has the Hessian (Dynare sets the field then)
        hess = model._meta.get("dynamic_hess")
        rows = np.unique(np.asarray(hess[0], int)) + 1 if hess is not None else np.zeros(0, int)
        M["nonzero_hessian_eqs"] = mat(rows.astype(float).reshape(1, -1)) if rows.size else mat(np.zeros((0, 0)))
        M["hessian_eq_zero"] = np.bool_(rows.size == 0)
    if not model.n_exo_det:
        # the sparsity pattern of the dynamic Jacobian in compressed-column order, as the preprocessor writes it (the
        # generated <model>.dynamic_g1 builds sparse(rowval, colval, values, endo_nbr, 3*endo_nbr+exo_nbr) from it)
        rws, cls = np.asarray(model.jac_rows, int), np.asarray(model.jac_cols, int)
        order = np.lexsort((rws, cls))
        M["dynamic_g1_sparse_rowval"] = row(rws[order] + 1.0)
        M["dynamic_g1_sparse_colval"] = row(cls[order] + 1.0)
        M["dynamic_g1_sparse_colptr"] = row(1.0 + np.concatenate([[0], np.cumsum(np.bincount(cls, minlength=model.jac_dims[1]))]))
    M["set_auxiliary_variables"] = np.bool_(any(a.get("type") != 6 for a in raw.get("aux_vars", [])))   # no file for Ramsey multipliers
    for k in ("orig_maximum_endo_lag", "orig_maximum_endo_lead", "orig_maximum_exo_lag", "orig_maximum_exo_lead",
              "orig_maximum_exo_det_lag", "orig_maximum_exo_det_lead", "orig_maximum_lag", "orig_maximum_lead",
              "orig_maximum_lag_with_diffs_expanded"):
        if k in info:
            M[k] = _F(info[k])
    lag, lead = int(info["maximum_endo_lag"]), int(info["maximum_endo_lead"])
    lli = model.lead_lag_incidence                       # n x 3 (lag, current, lead), zero-filled
    cols = [c for c, present in ((0, lag > 0), (1, True), (2, lead > 0)) if present]
    M["lead_lag_incidence"] = mat(lli[:, cols].T.astype(float))
    for k in ("nstatic", "nfwrd", "npred", "nboth", "nsfwrd", "nspred", "ndynamic"):
        M[k] = _F(info[k])
    for k in ("maximum_lag", "maximum_lead", "maximum_endo_lag", "maximum_endo_lead"):
        M[k] = _F(info[k])
    order_var, state_var, cnt = dr_ordering(model.lead_lag_incidence)
    inv = np.empty(n, int)
    inv[order_var] = np.arange(n)
    M["state_var"] = Struct(declaration_order=_row10(np.asarray(state_var) + 1), dr_order=_row10(inv[state_var] + 1))
    for k in ("maximum_exo_lag", "maximum_exo_lead"):
        if k in info:
            M[k] = _F(info[k])
    if "nonzero_hessian_eqs" in info:
        M["nonzero_hessian_eqs"] = row(np.asarray(info["nonzero_hessian_eqs"]) + 1)
    return M


def known_missing(name: str) -> bool:
    return name in KNOWN_UNSUPPORTED

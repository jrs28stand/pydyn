# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""tools/accept_compare.py --scope: tests a port parks on purpose are `excluded` only when the port refuses them."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import accept_compare as C  # noqa: E402

ENTRY = {"feature": "feature X", "reason": "parked in the test", "ids": ["grp/a", "pat/*"], "expect": "not ported"}


def _case(tmp_path, tid, port_status, message="", port_has_mat=False):
    ref, port = tmp_path / "ref", tmp_path / "port"
    for d in (ref, port):
        (d / Path(tid).parent).mkdir(parents=True, exist_ok=True)
    (ref / f"{tid}.json").write_text(json.dumps({"status": "ok"}))
    savemat(ref / f"{tid}.mat", {"oo_": {"steady_state": np.array([1.0, 2.0])}})
    (port / f"{tid}.json").write_text(json.dumps({"status": port_status, "message": message, "rng": "octave"}))
    if port_has_mat:
        savemat(port / f"{tid}.mat", {"oo_": {"steady_state": np.array([1.0, 2.0])}})
    return ref, port


def test_refused_listed_test_is_excluded_with_feature_and_reason(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "refused", "ResultsError: orders 2 and 3 are not ported")
    r = C.compare_one("grp/a", port, ref, scope=[ENTRY])
    assert r["verdict"] == "excluded" and r["feature"] == "feature X" and r["reason"] == "parked in the test"
    assert "scope_flag" not in r


def test_pattern_ids_match(tmp_path):
    ref, port = _case(tmp_path, "pat/zzz", "refused", "not ported")
    assert C.compare_one("pat/zzz", port, ref, scope=[ENTRY])["verdict"] == "excluded"


def test_without_scope_the_same_test_is_a_plain_refusal(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "refused", "not ported")
    r = C.compare_one("grp/a", port, ref)
    assert r["verdict"] == "refused"


def test_refusal_for_another_reason_is_reported_not_excluded(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "refused", "ResultsError: something else is missing")
    r = C.compare_one("grp/a", port, ref, scope=[ENTRY])
    assert r["verdict"] == "refused" and r["scope_flag"][0] == "other-reason"


def test_a_crash_is_not_a_refusal(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "error", "KeyError: x")
    r = C.compare_one("grp/a", port, ref, scope=[ENTRY])
    assert r["verdict"] == "port-error" and r["scope_flag"][0] == "not-a-refusal"


def test_a_listed_test_that_runs_is_compared_never_excluded(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "ok", port_has_mat=True)
    r = C.compare_one("grp/a", port, ref, scope=[ENTRY])
    assert r["verdict"] in ("match", "partial") and r["scope_flag"][0] == "now-runs"


def test_heterogeneity_is_a_builtin_excluded_entry(tmp_path):
    ref, port = _case(tmp_path, "heterogeneity/x", "ok", port_has_mat=True)
    r = C.compare_one("heterogeneity/x", port, ref)
    assert r["verdict"] == "excluded" and "heterogeneous" in r["feature"]
    assert C.compare_one("heterogeneity/x", port, ref, include_heterogeneity=True)["verdict"] != "excluded"


def test_report_groups_excluded_by_feature_and_lists_flags(tmp_path):
    ref, port = _case(tmp_path, "grp/a", "refused", "not ported")
    r1 = C.compare_one("grp/a", port, ref, scope=[ENTRY])
    ref2, port2 = _case(tmp_path / "b", "pat/b", "ok", port_has_mat=True)
    r2 = C.compare_one("pat/b", port2, ref2, scope=[ENTRY])
    out = tmp_path / "r.md"
    C.report([r1, r2], {}, ref, "pydyn", out, [ENTRY])
    t = out.read_text()
    assert "### feature X (1 files)" in t and "Reason: parked in the test" in t and "grp/a" in t
    assert "Scope entries to review" in t and "now-runs" in t


def test_shipped_pydyn_scope_file_is_valid():
    ents = C.load_scope(TOOLS / "accept_scope_pydyn.json")
    assert ents and all(e["feature"] and e["reason"] and e["ids"] for e in ents)

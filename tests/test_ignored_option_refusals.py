# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""Estimation options that change what Dynare computes are refused, not silently ignored."""
import shutil
from pathlib import Path

import pytest

import pydyn
from pydyn import ResultsError

REF = Path(__file__).resolve().parent.parent / "reference" / "oo"


def test_estimate_initial_states_endogenous_prior_is_refused(tmp_path):
    """Dynare adds one estimated initial-state parameter per state variable (prior.mean has 13 entries instead of 9 in
    estimation/slice/fs2000_estimate_initial_states) and runs the slice sampler; pydyn used to ignore the option."""
    shutil.copy(REF / "est_conv.mod", tmp_path / "e.mod")
    for f in REF.glob("est_conv*"):
        if f.suffix not in (".mod", ".mat"):
            shutil.copy(f, tmp_path)
    src = (tmp_path / "e.mod").read_text()
    src = src.replace("mode_compute=0,", "mode_compute=0,estimate_initial_states_endogenous_prior,posterior_sampling_method='slice',", 1)
    (tmp_path / "e.mod").write_text(src)
    with pytest.raises(ResultsError, match="estimate_initial_states_endogenous_prior is not supported"):
        pydyn.dynare(tmp_path / "e.mod")

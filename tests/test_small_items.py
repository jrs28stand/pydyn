"""save_params_and_steady_state (the file is byte-identical to Dynare's under Octave 8.4) and the errors of
initial_estimation_checks.m / check_prior_bounds.m (messages recorded from Dynare 7.2 under Octave, reference/misc/rchk.m)."""
import os
import shutil
from pathlib import Path

import pytest

import pydyn
from pydyn import ResultsError

REF = Path(__file__).resolve().parent.parent / "reference" / "misc"
OO = Path(__file__).resolve().parent.parent / "reference" / "oo"
needs_pp = pytest.mark.skipif(not os.environ.get("DYNARE_PREPROCESSOR"), reason="DYNARE_PREPROCESSOR not set")


@needs_pp
def test_save_params_and_steady_state_file_identical(tmp_path):
    shutil.copy(REF / "pss1.mod", tmp_path)
    pydyn.dynare(tmp_path / "pss1.mod")
    assert (tmp_path / "pss1_out.txt").read_bytes() == (REF / "pss1_out.txt").read_bytes()


@needs_pp
def test_save_params_and_steady_state_does_not_overwrite_foreign_file(tmp_path):
    shutil.copy(REF / "pss1.mod", tmp_path)
    (tmp_path / "pss1_out.txt").write_text("mine\n")
    with pytest.raises(ResultsError):
        pydyn.dynare(tmp_path / "pss1.mod")
    assert (tmp_path / "pss1_out.txt").read_text() == "mine\n"


CHECKS = {
    "chk_a": "initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated with a zero variance!",
    "chk_c": "initial_estimation_checks:: mh_drop must be in [0,1).",
    "chk_d": ("Initial value(s) of alp are outside parameter bounds.\nPotentially, you should set prior_trunc = 0. If you used the "
              "mode_file-option, check whether your mode-file is consistent with the priors."),
}


@needs_pp
@pytest.mark.parametrize("case", sorted(CHECKS))
def test_initial_estimation_check_messages(case, tmp_path):
    shutil.copy(REF / f"{case}.mod", tmp_path)
    shutil.copy(OO / "fsdat_simul.m", tmp_path)
    with pytest.raises(ResultsError) as ei:
        pydyn.dynare(tmp_path / f"{case}.mod")
    assert ei.value.matlab_message == CHECKS[case]

# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""``conditional_forecast`` / ``plot_conditional_forecast`` in pydyn.dynare(): what the field-by-field comparison with Dynare
(tests/test_results.py, cases cf_*) does not cover: Dynare's errors, the options that persist in ``options_cond_fcst_``,
the shared random stream and path values given as a vector."""
import re
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn import ResultsError

REF = Path(__file__).resolve().parent.parent / "reference" / "oo"
BASE = (REF / "cf_cal.mod").read_text()
CF = "conditional_forecast(parameter_set=calibration, controlled_varexo=(e_a,e_m)"


def _run(tmp_path, text, name="c"):
    (tmp_path / f"{name}.mod").write_text(text)
    (tmp_path / "fsdat_simul.m").write_text((REF / "fsdat_simul.m").read_text())
    return pydyn.dynare(tmp_path / f"{name}.mod")


def _head(n_cf=0):
    """cf_cal.mod up to and including its first conditional_forecast_paths block."""
    i = BASE.index("conditional_forecast(")
    return BASE[:i]


def test_cf_cal_fields_and_vector_values(tmp_path):
    """The file itself checks that ``values (xx)`` (a vector) gives the paths of the scalar values; it also runs here."""
    ws = _run(tmp_path, BASE)
    cf = ws.oo_["conditional_forecast"]
    assert list(cf) == ["controlled_variables", "instruments", "cond", "controlled_exo_variables", "uncond", "graph"]
    assert np.asarray(cf["controlled_variables"]).shape == (1, 2)          # indexing a row vector: a row
    assert cf["instruments"].shape == (2, 1) and cf["graph"]["title"] == "Calibration"
    assert np.asarray(cf["cond"]["ci"]["gy_obs"]).shape == (2, 41)         # periods = 40
    assert "options_cond_fcst_" in ws.vars


def test_options_persist_between_statements(tmp_path):
    """Dynare's driver only assigns the options a statement gives: the next statement keeps the others."""
    text = _head() + f"{CF}, periods=7, replic=20);\nzz1 = size(oo_.conditional_forecast.cond.ci.gy_obs, 2);\n" \
                     f"{CF});\nzz2 = size(oo_.conditional_forecast.cond.ci.gy_obs, 2);\nzz3 = options_cond_fcst_.replic;\n"
    ws = _run(tmp_path, text)
    assert float(ws.vars["zz1"]) == 8 and float(ws.vars["zz2"]) == 8 and float(ws.vars["zz3"]) == 20


def test_random_stream_is_shared(tmp_path):
    """2 * replic draws of randn(n_exo, periods) (here 2 * 5 * 2 * 5 = 100) are taken from the stream (first the controlled set, then the unconditional
    one); the next randn is the following number."""
    a = _run(tmp_path, _head() + f"{CF}, periods=5, replic=5);\nzz = randn(1, 1);\n", "a").vars["zz"]
    b = _run(tmp_path, _head() + "rn = randn(101, 1);\n", "b").vars["rn"]
    assert float(a) == float(np.asarray(b).ravel()[100])


@pytest.mark.parametrize("text, match", [
    (lambda h: h.replace("stoch_simul(irf=0);", "") + f"{CF});", "stoch_simul before conditional_forecast"),
    (lambda h: h + "conditional_forecast(parameter_set=calibration, controlled_varexo=(e_a));", "number of constrained variables"),
    (lambda h: h + "conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(e_a,e_m));", "needs an estimation"),
    (lambda h: h + f"{CF}, periods=3);", "paths have 5 periods"),
    (lambda h: h + "plot_conditional_forecast gy_obs;", "Can't find conditional forecasts"),
])
def test_dynares_errors(tmp_path, text, match):
    with pytest.raises(ResultsError, match=match):
        _run(tmp_path, text(_head()))


def test_no_paths_block(tmp_path):
    text = BASE[:BASE.index("conditional_forecast_paths")] + f"{CF});"
    with pytest.raises(ResultsError, match="no conditional_forecast_paths"):
        _run(tmp_path, text)


def test_prefilter_is_refused_like_dynare(tmp_path):
    """run.m: ``options_.prefilter`` with an estimated parameter set is an error."""
    text = (REF / "cf_ml.mod").read_text().replace("mode_compute=4", "mode_compute=4,prefilter=1")
    with pytest.raises(ResultsError, match="prefilter"):
        _run(tmp_path, text)

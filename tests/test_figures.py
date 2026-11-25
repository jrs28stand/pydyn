"""Figures of pydyn.dynare(nograph=False) against the drawing calls of Dynare 7.2 under Octave (reference/fig/r_fig.m
records them with stub graphics functions): figure names, panels, plotted arrays, titles and the file stems; styles are not
compared.  Also checks that the files are written (matplotlib, Agg)."""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import figcompare  # noqa: E402

import pydyn  # noqa: E402

REF = Path(__file__).resolve().parent.parent / "reference" / "fig"
OO = Path(__file__).resolve().parent.parent / "reference" / "oo"
needs_pp = pytest.mark.skipif(not os.environ.get("DYNARE_PREPROCESSOR"), reason="DYNARE_PREPROCESSOR not set")

#: case -> tolerance.  mode_compute=4 (fig_est1) finds the mode only to about 1e-7, which moves the mode-check grid.
CASES = {"fig_irf1": 1e-8, "fig_irf2": 1e-8, "fig_irf3": 1e-8, "fig_fc1": 1e-8, "fig_cf1": 1e-8, "fig_sd1": 1e-8,
         "fig_est1": 1e-6, "fig_mh1": 1e-8, "fig_mh2": 1e-8, "fig_conv1": 1e-8, "fig_occ1": 1e-8, "fig_rpl1": 1e-8, "fig_sd2": 1e-8, "fig_sdf1": 1e-8}


def _run(case, tmp_path, **kw):
    shutil.copy(REF / f"{case}.mod", tmp_path)
    shutil.copy(OO / "fsdat_simul.m", tmp_path)
    for aux in REF.glob(f"{case}_*.mat"):
        if not aux.name.endswith("_fig.mat"):
            shutil.copy(aux, tmp_path)
    return pydyn.dynare(tmp_path / f"{case}.mod", nograph=False, **kw)


@needs_pp
@pytest.mark.parametrize("case", sorted(CASES))
def test_figures_match_dynare(case, tmp_path):
    res = _run(case, tmp_path)
    assert figcompare.diff(res, REF / f"{case}_fig.mat", CASES[case]) == []


@needs_pp
def test_files_written(tmp_path):
    pytest.importorskip("matplotlib")
    res = _run("fig_est1", tmp_path)
    g = tmp_path / "fig_est1" / "graphs"
    for stem in ("fig_est1_Priors1", "fig_est1_CheckPlots1", "fig_est1_SmoothedShocks1", "forcst1"):
        assert (g / f"{stem}.eps").stat().st_size > 1000, stem
    assert (g / "fig_est1_check_plot_data.mat").is_file()
    assert res.gfx.written


@needs_pp
def test_nograph_default_draws_nothing(tmp_path):
    shutil.copy(REF / "fig_irf1.mod", tmp_path)
    res = pydyn.dynare(tmp_path / "fig_irf1.mod")
    assert not res.gfx.figures
    assert not list((tmp_path / "fig_irf1").glob("graphs/*"))


@needs_pp
def test_graph_format_pdf_and_none(tmp_path):
    pytest.importorskip("matplotlib")
    shutil.copy(REF / "fig_fc1.mod", tmp_path)
    txt = (tmp_path / "fig_fc1.mod").read_text().replace("forecast(periods=8);", "forecast(periods=8);")
    (tmp_path / "fig_fc1.mod").write_text(txt.replace("stoch_simul(", "options_.graph_format='pdf';\nstoch_simul(", 1))
    pydyn.dynare(tmp_path / "fig_fc1.mod", nograph=False)
    assert (tmp_path / "fig_fc1" / "graphs" / "forcst1.pdf").is_file()
    assert not (tmp_path / "fig_fc1" / "graphs" / "forcst1.eps").exists()

"""Figures: the data drawn must be the validated simulation, laid out as Dynare's occbin_graph does."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pydyn  # noqa: E402
from pydyn import plotting  # noqa: E402

EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))


@pytest.fixture(scope="module")
def occ(tmp_path_factory):
    work = tmp_path_factory.mktemp("plot")
    shutil.copy(EXAMPLES / "occbin" / "rbc_occbin.mod", work)
    m = pydyn.Model(pydyn.run_preprocessor(work / "rbc_occbin.mod"))
    ob = pydyn.Occbin(m)
    return m, ob, ob.run()


def test_layout_rule_matches_pltorg():
    assert [plotting.layout(n) for n in range(1, 10)] == [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 2),
                                                          (3, 3), (3, 3), (3, 3)]


@pytest.mark.parametrize("style", ["pydyn", "dynare"])
def test_occbin_graph(occ, style, tmp_path):
    m, _, res = occ
    names = ["log_C", "epsilon", "Lambda", "log_K", "log_Invest", "A"]      # as in the example's occbin_graph
    figs = plotting.plot_occbin(res, m, names, style=style)
    assert len(figs) == 1
    axes = [ax for ax in figs[0].axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == ["log_C", "Lambda", "log_K", "log_Invest", "A", "epsilon"]
    i = m.endo_names.index("log_K")
    zero, piecewise, linear = axes[2].lines
    np.testing.assert_allclose(piecewise.get_ydata(), res.piecewise[:, i] - res.ys[i])
    np.testing.assert_allclose(linear.get_ydata(), res.linear[:, i] - res.ys[i])
    assert len(axes[5].lines) == 2                                           # shocks: no linear counterpart
    assert axes[0].get_legend() is not None and axes[1].get_legend() is None
    paths = plotting.save(figs, str(tmp_path / "fig"))
    assert Path(paths[0]).stat().st_size > 10_000


def test_more_than_nine_variables_spill_onto_a_second_figure(occ):
    m, _, res = occ
    figs = plotting.plot_occbin(res, m, m.endo_names + m.exo_names + [])     # 9 panels
    assert len(figs) == 1
    series = {f"v{i}": np.arange(5.0) for i in range(11)}
    assert len(plotting.plot_irf(series)) == 2


def test_irf_comparison(occ):
    m, ob, _ = occ
    pw, lin = ob.irf([-0.05], periods=20), ob.ref.irf([-0.05], periods=20)
    fig, = plotting.plot_irf(pw, m.endo_names, ["log_Invest"], compare=lin, labels=("Piecewise Linear", "Linear"))
    _, a, b = fig.axes[0].lines
    np.testing.assert_allclose(a.get_ydata(), pw[:, m.endo_names.index("log_Invest")])
    np.testing.assert_allclose(b.get_ydata(), lin[:, m.endo_names.index("log_Invest")])

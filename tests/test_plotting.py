"""Figures: the data drawn must be the validated simulation, laid out as Dynare's occbin_graph does;
shock-decomposition figures must draw the bars Dynare's graph_decomp draws (recorded with stub
figure functions by reference/r_shock_decomposition.m)."""
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


# ====================================================================== shock decomposition figures
from scipy.io import loadmat  # noqa: E402

from pydyn import smoother  # noqa: E402
from pydyn.estimation import Estimation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
sd_marks = [pytest.mark.filterwarnings("ignore:statements not handled"),
            pytest.mark.filterwarnings("ignore::pydyn.options.UnsupportedOptionWarning")]


def sd_ref(name):
    return loadmat(REF / f"{name}.mat", squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def sd(tmp_path_factory):
    """Decomposition at Dynare's posterior mode (sd_nk case 1)."""
    import warnings
    work = tmp_path_factory.mktemp("sdplot")
    for f in ("sd_nk.mod", "est_nk_data.m"):
        shutil.copy(REF / f, work)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(work / "sd_nk.mod", extra_args=("-DCASE=1",)))
        res = smoother.smooth(m, estimation=Estimation(m), theta=sd_ref("sd_nk_case1")["theta_mode"],
                              parameter_set="posterior_mode")
        return m, smoother.shock_decomposition(m, res), smoother.initial_condition_decomposition(m, res)


def drawn_bars(ax):
    """(components, periods) arrays of the bars in an axes: height, bottom, centre."""
    cs = ax.containers
    h = np.array([[p.get_height() for p in c] for c in cs])
    y = np.array([[p.get_y() for p in c] for c in cs])
    x = np.array([[p.get_x() + p.get_width() / 2 for p in c] for c in cs])
    return h, y, x


def dynare_bars(g, fig):
    """The fill() calls of graph_decomp for one figure: rows (component, x left, x right, y from, y to)."""
    b = getattr(g, f"f{fig}_p0_bars")
    ncomp = int(b[:, 0].max())
    T = b.shape[0] // ncomp
    b = b.reshape(T, ncomp, 5).transpose(1, 0, 2)
    assert np.all(b[:, :, 0] == np.arange(1, ncomp + 1)[:, None])
    return b[:, :, 4] - b[:, :, 3], b[:, :, 3], (b[:, :, 1] + b[:, :, 2]) / 2


def check_against_dynare(fig, g, k, labels, tol=dict(rtol=1e-9, atol=1e-12)):
    ax = fig.axes[0]
    h, y, x = drawn_bars(ax)
    dh, dy, dx = dynare_bars(g, k)
    np.testing.assert_allclose(h, dh, **tol)                  # contribution of every component and period
    np.testing.assert_allclose(y, dy, **tol)                  # where each bar starts: Dynare's stacking
    np.testing.assert_allclose(x, dx)                         # bars centred on the period
    line = ax.lines[-1] if len(ax.lines) == 2 else ax.lines[1]
    np.testing.assert_allclose(line.get_ydata(), getattr(g, f"f{k}_p0_line"), **tol)
    np.testing.assert_allclose(line.get_xdata(), getattr(g, f"f{k}_p0_line_x"))
    assert [c.get_label() for c in ax.containers] == labels
    # Dynare writes its legend bottom-up
    assert getattr(g, f"f{k}_p0_labels").split("|")[1:][::-1] == labels


@pytest.mark.parametrize("style", ["pydyn", "dynare"])
@pytest.mark.parametrize("which,options,labels", [
    ("grp", dict(use_shock_groups="grp"), ["supply", "demand and policy", "Initial values"]),
    ("plain", dict(), ["ea", "eu", "er", "Initial values"]),
    ("partial", dict(use_shock_groups="partial"), ["policy", "Others + Initial Values"]),
    ("ss", dict(steadystate=True, flip=True), ["ea", "eu", "er", "Initial values"]),
])
@pytest.mark.filterwarnings("ignore")
def test_shock_decomposition_figures_match_dynare(sd, style, which, options, labels):
    """The bars and the line of plot_shock_decomposition for periods 11-40 (plot_init_date,
    plot_end_date), one figure per variable, against the fill()/plot() calls of graph_decomp.m."""
    import matplotlib.pyplot as plt
    m, dec, _ = sd
    g = sd_ref("sd_nk_case1_drawn")["drawn_" + which]
    figs = plotting.plot_shock_decomposition(dec, m, ["pi", "robs"], style=style, statement=None,
                                             plot_init_date=11, plot_end_date=40, **options)
    assert len(figs) == int(g.nfig) == 2
    for k, fig in enumerate(figs, 1):
        check_against_dynare(fig, g, k, labels)
        assert fig.legends and len(fig.legends[0].get_texts()) == len(labels) + 1    # legend outside the axes
        assert fig.legends[0].get_texts()[0].get_text() == labels[-1]
    names = g.names.split("|")
    assert figs[0].axes[0].get_title().startswith(names[0].split(":")[0])            # e.g. "... group grp"
    assert "pi (Posterior Mode)" in figs[0].axes[0].get_title()
    plt.close("all")


@pytest.mark.filterwarnings("ignore")
def test_full_sample_figure_and_plot_data_are_the_same_numbers(sd, tmp_path):
    import matplotlib.pyplot as plt
    m, dec, _ = sd
    g = sd_ref("sd_nk_case1_drawn")["drawn_full"]
    data = smoother.decomposition_plot_data(dec, ["pi", "robs"], m)
    figs = plotting.plot_shock_decomposition(data)
    for k, (fig, panel) in enumerate(zip(figs, data.panels), 1):
        check_against_dynare(fig, g, k, ["ea", "eu", "er", "Initial values"])
        h, y, _ = drawn_bars(fig.axes[0])
        np.testing.assert_allclose(h, panel["contributions"], rtol=1e-13, atol=1e-15)   # a bar's height is top - bottom: one rounding
        np.testing.assert_array_equal(y, smoother.stack_bars(panel["contributions"])[0])
    paths = plotting.save(figs, str(tmp_path / "sd"))
    assert len(paths) == 2 and Path(paths[0]).stat().st_size > 10_000
    with pytest.raises(TypeError):
        plotting.plot_shock_decomposition(data, var_list=["pi"])
    plt.close("all")


@pytest.mark.parametrize("style", ["pydyn", "dynare"])
@pytest.mark.filterwarnings("ignore")
def test_detail_plot_matches_dynare(sd, style):
    """detail_plot with max_nrows=1: three panels per figure, each component against the rest
    (the bar() calls of graph_decomp_detail.m)."""
    import matplotlib.pyplot as plt
    m, dec, _ = sd
    g = sd_ref("sd_nk_case1_drawn")["drawn_detail"]
    figs = plotting.plot_shock_decomposition(dec, m, ["pi", "robs"], style=style, statement=None, detail_plot=True,
                                             max_nrows=1, plot_init_date=11, plot_end_date=40)
    assert len(figs) == int(g.nfig) == 4 and list(g.grid) == [1, 3]
    tol = dict(rtol=1e-9, atol=1e-12)
    for k, fig in enumerate(figs, 1):
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(axes) == (3 if k % 2 else 1)
        for j, ax in enumerate(axes, 1):
            bar = getattr(g, f"f{k}_p{j}_bar")               # rows: positive part (own, rest), negative part
            h, _, x = drawn_bars(ax)
            np.testing.assert_allclose(h[0], bar[0] + bar[2], **tol)
            np.testing.assert_allclose(h[1], bar[1] + bar[3], **tol)
            np.testing.assert_allclose(x[0], getattr(g, f"f{k}_p{j}_barx"))
            np.testing.assert_allclose(ax.lines[-1].get_ydata(), getattr(g, f"f{k}_p{j}_line"), **tol)
            assert ax.get_title() == getattr(g, f"f{k}_p{j}_title")
    plt.close("all")


@pytest.mark.filterwarnings("ignore")
def test_initial_condition_decomposition_figure(sd):
    import matplotlib.pyplot as plt
    m, _, icd = sd
    g = sd_ref("sd_nk_case1_drawn")["drawn_icd"]
    fig, = plotting.plot_initial_condition_decomposition(icd, m, ["robs"], plot_init_date=11, plot_end_date=40)
    h, y, x = drawn_bars(fig.axes[0])
    dh, dy, dx = dynare_bars(g, 1)
    np.testing.assert_allclose(h, dh, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(y, dy, rtol=1e-9, atol=1e-12)
    labels = [c.get_label() for c in fig.axes[0].containers]
    dyn = g.f1_p0_labels.split("|")[1:][::-1]
    assert labels[:-1] == dyn[:-1] == m.endo_names
    # the remainder is the effect of the shocks; Dynare's figure calls it "Initial values"
    assert labels[-1] == "Shocks" and dyn[-1] == "Initial values"
    assert fig.axes[0].get_title().startswith("Initial condition decomposition")
    with pytest.raises(TypeError):
        plotting.plot_initial_condition_decomposition(sd[1])
    plt.close("all")


@pytest.mark.parametrize("style", ["pydyn", "dynare"])
def test_screened_shocks_figure(style):
    """21 shocks with screen_shocks: 16 kept for each variable, 'Others', 'Initial values'
    (graph_decomp.m and graph_decomp_detail.m on an artificial decomposition)."""
    import matplotlib.pyplot as plt
    p = sd_ref("sd_nk_case3_plot")
    z = p["screen_z"]
    cols = [f"s{j}" for j in range(1, z.shape[1] - 1)] + ["initial_conditions", "smoothed"]
    dec = smoother.ShockDecomposition(z=z, endo_names=["v1", "v2"], columns=cols)
    figs = plotting.plot_shock_decomposition(dec, screen_shocks=True, style=style)
    g = p["drawn_screen"]
    assert len(figs) == 2
    for k, fig in enumerate(figs, 1):
        labels = getattr(g, f"f{k}_p0_labels").split("|")[1:][::-1]
        check_against_dynare(fig, g, k, labels)
        assert len(labels) == 18 and "SCREEN" in fig.axes[0].get_title()
    colors = plotting.decomposition_colors(labels)
    assert colors[-1] == (plotting.DECOMP_NEUTRAL["initial"], "") and colors[-2] == (plotting.DECOMP_NEUTRAL["others"], "")
    assert [c for c, _ in colors[:8]] == plotting.DECOMP_COLORS and colors[8] == (plotting.DECOMP_COLORS[0], "////")
    assert len(set(colors)) == 18                             # every component can be told apart
    g = p["drawn_screen_detail"]
    figs = plotting.plot_shock_decomposition(dec, screen_shocks=True, style=style, detail_plot=True)
    assert len(figs) == int(g.nfig) == 2 and list(g.grid) == [6, 3]
    for k, fig in enumerate(figs, 1):
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(axes) == 18 and fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (6, 3)
        for j, ax in enumerate(axes, 1):
            bar = getattr(g, f"f{k}_p{j}_bar")
            h, _, _ = drawn_bars(ax)
            np.testing.assert_allclose(h[0], bar[0] + bar[2], rtol=1e-9, atol=1e-12)
            assert ax.get_title() == getattr(g, f"f{k}_p{j}_title")
    plt.close("all")


@pytest.mark.parametrize("style", ["pydyn", "dynare"])
@pytest.mark.filterwarnings("ignore")
def test_decomposition_panels_and_statement_defaults(sd, style):
    """Several variables in Dynare's panel layout with one legend; options and variables default
    to the plot_shock_decomposition statement of the .mod file (use_shock_groups=grp)."""
    import matplotlib.pyplot as plt
    m, dec, _ = sd
    figs = plotting.plot_shock_decomposition(dec, m, ["y", "pi", "r", "robs", "a"], style=style, panels=True)
    assert len(figs) == 1
    axes = [ax for ax in figs[0].axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == ["y", "pi", "r", "robs", "a"]
    assert figs[0].axes[0].get_subplotspec().get_gridspec().get_geometry() == plotting.layout(5)
    assert [c.get_label() for c in axes[0].containers] == ["supply", "demand and policy", "Initial values"]
    grouped = smoother.group_shocks(dec, model=m, name="grp")
    h, _, _ = drawn_bars(axes[1])
    np.testing.assert_allclose(h, grouped.z[m.endo_names.index("pi"), :-1], rtol=1e-13, atol=1e-15)   # a bar's height is top - bottom: one rounding
    assert len(figs[0].legends) == 1 and all(ax.get_legend() is None for ax in axes)
    # statement=None ignores the file: ungrouped
    fig, = plotting.plot_shock_decomposition(dec, m, ["pi"], style=style, statement=None)
    assert len(fig.axes[0].containers) == 4
    # steadystate: tick labels are levels, bars stay deviations
    fig, = plotting.plot_shock_decomposition(dec, m, ["robs"], style=style, statement=None, steadystate=True)
    ax = fig.axes[0]
    fig.canvas.draw()
    ss = dec.steady_state[m.endo_names.index("robs")]
    ticks = [(t, lab.get_text()) for t, lab in zip(ax.get_yticks(), ax.get_yticklabels())]
    assert ticks and all(abs(float(txt.replace("−", "-")) - (t + ss)) < 1e-3 for t, txt in ticks)
    h, _, _ = drawn_bars(ax)
    np.testing.assert_allclose(h, dec.z[m.endo_names.index("robs"), :-1], rtol=1e-13, atol=1e-15)   # a bar's height is top - bottom: one rounding
    plt.close("all")


def test_example_figure(sd):
    """The example saved for the user (when the folder exists): groups, steady state on the axis."""
    import matplotlib.pyplot as plt
    m, dec, _ = sd
    fig, = plotting.plot_shock_decomposition(dec, m, ["robs"], statement=None, steadystate=True,
                                             plot_init_date=41, plot_end_date=120)
    fig.canvas.draw()
    box = fig.legends[0].get_window_extent()
    axbox = fig.axes[0].get_window_extent()
    assert box.x0 > axbox.x1 and box.x1 <= fig.bbox.x1         # legend to the right of the bars, inside the figure
    plt.close("all")

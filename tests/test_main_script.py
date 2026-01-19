import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class DummyAxes:
    """Minimal stand-in for Matplotlib Axes to capture plotting calls."""

    def __init__(self):
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.grid_args = None
        self.plots = []
        self.legend_called = False
        self.ylim = None

    def set_xlabel(self, value):
        self.xlabel = value

    def set_ylabel(self, value):
        self.ylabel = value

    def set_title(self, value):
        self.title = value

    def grid(self, *args, **kwargs):
        self.grid_args = (args, kwargs)

    def plot(self, x, y, lw=1, label=None):
        self.plots.append(SimpleNamespace(x=np.asarray(x), y=np.asarray(y), lw=lw, label=label))
        return self

    def legend(self):
        self.legend_called = True

    def set_ylim(self, a, b):
        self.ylim = (a, b)


def _run_main(monkeypatch):
    """Execute pkoffee/main.py with heavy dependencies stubbed for test isolation."""

    df = pd.DataFrame({"cups": [0, 1, 2, 3, 4], "productivity": [1.0, 2.0, 3.0, 3.5, 3.8]})
    monkeypatch.setattr("pandas.read_csv", lambda path: df)

    ax = DummyAxes()
    monkeypatch.setattr("matplotlib.pyplot.figure", lambda figsize=None: None)
    monkeypatch.setattr("matplotlib.pyplot.gca", lambda: ax)
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", lambda: None)
    save_calls = []
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda path, dpi=None: save_calls.append((path, dpi)))
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    violin_calls = []
    def _violinplot(*args, **kwargs):
        violin_calls.append(kwargs)
        return None
    monkeypatch.setattr("seaborn.violinplot", _violinplot)

    def _curve_fit(func, X, Y, p0, bounds, maxfev):
        # Return initial guesses to make r2 deterministic for tests.
        return np.asarray(p0), None
    monkeypatch.setattr("scipy.optimize.curve_fit", _curve_fit)

    # Execute the script by reading and executing it in a namespace
    main_path = Path(__file__).parent.parent / "pkoffee" / "main.py"
    namespace = {"__file__": str(main_path)}
    exec(main_path.read_text(), namespace)
    
    # Convert namespace to object for attribute access
    main = SimpleNamespace(**namespace)
    return main, ax, save_calls, violin_calls, df


def test_fits_are_sorted_and_complete(monkeypatch):
    main, ax, _, _, _ = _run_main(monkeypatch)

    assert len(main.fits) == len(main.MODELS)

    r2s = [fit["r2"] for fit in main.fits]
    expected_order = sorted(r2s, key=lambda v: v if np.isfinite(v) else -np.inf, reverse=True)
    assert r2s == expected_order
    assert all(np.isfinite(r) for r in r2s)

    assert ax.xlabel == "Cups of Coffee"
    assert ax.ylabel == "Productivity"
    assert ax.title == "Productivity vs Coffee"


def test_plotting_and_io(monkeypatch):
    main, ax, save_calls, violin_calls, df = _run_main(monkeypatch)

    assert len(save_calls) == 1
    saved_path, dpi = save_calls[0]
    assert str(saved_path).endswith("fit_plot.png")
    assert dpi == 150

    assert ax.legend_called is True
    assert ax.ylim == (-0.2, 8)
    assert len(ax.plots) == len(main.MODELS)

    assert len(violin_calls) == 1
    call_kwargs = violin_calls[0]
    assert call_kwargs["data"].equals(df)
    assert call_kwargs["x"] == "cups"
    assert call_kwargs["y"] == "pr=oductivity"

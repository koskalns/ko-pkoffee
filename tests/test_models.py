import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from main import MODELS, fits, x_smooth, y_smooth, out_path

# Test data loading
def test_data_loading():
    data = pd.read_csv("coffee_productivity.csv")
    assert not data.empty, "Data should not be empty"
    assert "cups" in data.columns, "Data should contain 'cups' column"
    assert "productivity" in data.columns, "Data should contain 'productivity' column"

# Test plot creation
def test_plot_creation():
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    assert fig is not None, "Figure should be created"
    assert ax is not None, "Axis should be created"

# Test violin plot
def test_violin_plot():
    data = pd.read_csv("coffee_productivity.csv")
    fig, ax = plt.subplots()
    sns.violinplot(data=data, x="cups", y="productivity", hue="cups", ax=ax, inner="quartile", cut=0, density_norm='width', palette="Greens", linewidth=0.8, legend=False)
    assert len(ax.collections) > 0, "Violin plot should be created"

# Test model definitions
def test_model_definitions():
    assert len(MODELS) == 5, "There should be 5 models defined"
    for model in MODELS:
        assert "name" in model, "Model should have a name"
        assert "p0" in model, "Model should have initial parameters"
        assert "bounds" in model, "Model should have bounds"
        assert "f" in model, "Model should have a function"

# Test model fitting
def test_model_fitting():
    data = pd.read_csv("coffee_productivity.csv")
    X = data["cups"].values
    Y = data["productivity"].values
    for m in MODELS:
        popt, _ = curve_fit(m["f"], X, Y, p0=m["p0"], bounds=m["bounds"], maxfev=20000)
        assert popt is not None, "Model fitting should return parameters"
        assert len(popt) == len(m["p0"]), "Number of fitted parameters should match initial parameters"

# Test R² calculation
def test_r2_calculation():
    data = pd.read_csv("coffee_productivity.csv")
    X = data["cups"].values
    Y = data["productivity"].values
    for res in fits:
        assert "r2" in res, "Fit result should contain R² value"
        assert np.isfinite(res["r2"]), "R² value should be finite"

# Test sorting of fits
def test_sorting_of_fits():
    fits.sort(key=lambda d: (d["r2"] if np.isfinite(d["r2"]) else -np.inf), reverse=True)
    assert fits[0]["r2"] >= fits[-1]["r2"], "Fits should be sorted in descending order of R²"

# Test plotting of fits
def test_plotting_of_fits():
    fig, ax = plt.subplots()
    for idx, res in enumerate(fits):
        y_s = res["func"](x_smooth, *res["params"])
        ax.plot(x_smooth, y_s, lw=2, label=f"{res['name']} (R²={res['r2']:.3f})")
    assert len(ax.lines) == len(fits), "All fits should be plotted"

# Test saving of plot
def test_saving_of_plot():
    plt.savefig(out_path, dpi=150)
    assert Path(out_path).exists(), "Plot should be saved as a file"
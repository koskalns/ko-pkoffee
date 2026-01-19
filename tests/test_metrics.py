"""Unit tests for model evaluation metrics."""

import numpy as np
import pytest

from pkoffee.metrics import compute_mae, compute_r2, compute_rmse


def test_compute_r2_perfect_fit() -> None:
    """Test R² for perfect prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    r2 = compute_r2(y_true, y_pred)

    assert r2 == pytest.approx(1.0)


def test_compute_r2_poor_fit() -> None:
    """Test R² for poor prediction (negative values possible)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    r2 = compute_r2(y_true, y_pred)

    assert r2 < 0  # Worse than predicting the mean


def test_compute_r2_mean_prediction() -> None:
    """Test R² when predicting the mean (should be 0)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, y_true.mean())

    r2 = compute_r2(y_true, y_pred)

    assert r2 == pytest.approx(0.0)


def test_compute_r2_length_mismatch() -> None:
    """Test that ValueError is raised for mismatched lengths."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same length"):
        compute_r2(y_true, y_pred)


def test_compute_rmse_perfect_fit() -> None:
    """Test RMSE for perfect prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    rmse = compute_rmse(y_true, y_pred)

    assert rmse == pytest.approx(0.0)


def test_compute_rmse_known_value() -> None:
    """Test RMSE with known expected value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])

    rmse = compute_rmse(y_true, y_pred)

    # RMSE = sqrt(mean((0.5, 0.5, 0.5)^2)) = sqrt(0.25) = 0.5
    assert rmse == pytest.approx(0.5)


def test_compute_mae_perfect_fit() -> None:
    """Test MAE for perfect prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mae = compute_mae(y_true, y_pred)

    assert mae == pytest.approx(0.0)


def test_compute_mae_known_value() -> None:
    """Test MAE with known expected value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])

    mae = compute_mae(y_true, y_pred)

    # MAE = mean(|0.5, 0.5, 1.0|) = 2.0 / 3 ≈ 0.6667
    assert mae == pytest.approx(2.0 / 3.0)

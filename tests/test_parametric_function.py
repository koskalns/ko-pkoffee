"""Unit test for parametric functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pkoffee.data import data_dtype
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    Peak2Model,
    PeakModel,
    Quadratic,
)


def test_quadratic_model() -> None:
    """Test quadratic model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    quadratic_function = Quadratic()
    result = quadratic_function(x, a0=data_dtype(1.0), a1=data_dtype(2.0), a2=data_dtype(0.5))

    expected = np.array([1.0, 3.5, 7.0, 11.5])
    assert_allclose(result, expected)


def test_michaelis_menten_model() -> None:
    """Test Michaelis-Menten model evaluation."""
    x = np.array([0.0, 1.0, 5.0, 10.0])
    michaelis_menten_function = MichaelisMentenSaturation()
    result = michaelis_menten_function(x, v_max=data_dtype(10.0), k=data_dtype(2.0), y0=data_dtype(1.0))

    # At x=0: 1.0 + 10*0/(2+0) = 1.0
    # At x=2: 1.0 + 10*2/(2+2) = 6.0
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0 + 10.0 * 1.0 / 3.0)


def test_logistic_model() -> None:
    """Test logistic model evaluation."""
    x = np.array([0.0, 5.0, 10.0])
    logistic_function = Logistic()
    result = logistic_function(x, L=data_dtype(10.0), k=data_dtype(1.0), x0=data_dtype(5.0), y0=data_dtype(0.0))

    # At x0 (midpoint), should be y0 + L/2
    assert result[1] == pytest.approx(5.0, abs=0.01)
    # Function should be monotonically increasing
    assert result[0] < result[1] < result[2]


def test_peak_model() -> None:
    """Test peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    peak_function = PeakModel()
    result = peak_function(x, a=data_dtype(5.0), b=data_dtype(2.0))

    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should increase then decrease (peak behavior)
    assert result[1] < result[2]
    assert result[2] > result[3]


def test_peak2_model() -> None:
    """Test quadratic peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    peak2_model = Peak2Model()
    result = peak2_model(x, a=data_dtype(1.0), b=data_dtype(3.0))

    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should have peak behavior
    assert result[1] < result[2]

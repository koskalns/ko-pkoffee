"""Unit tests for model fitting."""

import numpy as np
import pytest

from pkoffee.data import AnyShapeDataDtypeArray, neg_inf, pos_inf
from pkoffee.data import data_dtype as dt
from pkoffee.fit_model import (
    FunctionIdNotFoundInMappingError,
    FunctionNotFoundInMappingError,
    Model,
    ModelParsingError,
    fit_model,
)
from pkoffee.parametric_function import ParametersBounds


class Linear:
    """Linear function."""

    def __call__(self, x: AnyShapeDataDtypeArray, a: dt, b: dt) -> AnyShapeDataDtypeArray:
        """Evaluate y = a * x + b."""
        return a * x + b  # pyright: ignore[reportReturnType] return dtype is data_dtype alright.

    @staticmethod
    def param_guess() -> dict[str, dt]:
        """Guess parameters for tests: a: 2.0, b: 1.0."""
        return {"a": dt(2.0), "b": dt(1.0)}

    @staticmethod
    def param_bounds() -> ParametersBounds:
        """Parameter bounds in [-inf, +inf]."""
        return ParametersBounds(min={"a": neg_inf, "b": neg_inf}, max={"a": pos_inf, "b": pos_inf})


class LinearSQRT:
    """Passthrough model with square root of coefficient parameter."""

    def __call__(self, x: AnyShapeDataDtypeArray, a: dt) -> AnyShapeDataDtypeArray:
        """Evaluate y = sqrt(a) * x."""
        return np.sqrt(a) * x

    @staticmethod
    def param_guess() -> dict[str, dt]:
        """Guess parameter: a: 1.0."""
        return {"a": dt(1.0)}

    @staticmethod
    def param_bounds() -> ParametersBounds:
        """Parameter bounds: "a" should be positive."""
        return ParametersBounds(min={"a": dt(5e-7)}, max={"a": pos_inf})


class PassThrough:
    """PassThrough empty model for testing."""

    def __call__(self, x: AnyShapeDataDtypeArray) -> AnyShapeDataDtypeArray:
        """Passthrough."""
        return x

    @staticmethod
    def param_guess() -> dict[str, dt]:
        """No parameter to guess."""
        return {}

    @staticmethod
    def param_bounds() -> ParametersBounds:
        """No parameter bounds."""
        return ParametersBounds(min={}, max={})


def test_model_result_predict() -> None:
    """Test ModelResult prediction method."""
    model_result = Model(
        name="Linear",
        function=Linear(),
        params=Linear.param_guess(),
        bounds=Linear.param_bounds(),
        r_squared=dt(0.95),
    )

    x_new = np.array([0.0, 1.0, 2.0], dtype=dt)
    predictions = model_result.predict(x_new)

    expected = np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(predictions, expected)


def test_fit_model_success() -> None:
    """Test successful model fitting."""
    # Create simple linear data
    x = np.linspace(0, 10, 50)
    rng = np.random.default_rng(1337)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 50)

    lin_model = Model(
        name="Test Linear",
        function=Linear(),
        params={"a": dt(1.0), "b": dt(0.0)},
        bounds=ParametersBounds(min=dict.fromkeys(["a", "b"], neg_inf), max=dict.fromkeys(["a", "b"], pos_inf)),
    )

    result, _ = fit_model(x, y, lin_model)

    assert result is not None
    assert result.name == "Test Linear"
    assert result.r_squared > 0.9  # Should fit well  # noqa: PLR2004
    assert len(result.params) == 2  # noqa: PLR2004
    assert result.params["a"] == pytest.approx(2.0, abs=0.2)


def test_fit_model_failure() -> None:
    """Test that fit_model raises error on failure."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    # Create a config that will fail
    config = Model(
        name="Bad Model",
        function=LinearSQRT(),
        params={"a": dt(-10.0)},
        bounds=ParametersBounds(min={"a": dt(-100.0)}, max={"a": dt(-1.0)}),
    )

    # Verify value error is raised, ignore RuntimeWarning from sqrt(0)
    with pytest.raises(ValueError), pytest.warns(RuntimeWarning):  # noqa: PT011 error raised by scipy
        _ = fit_model(x, y, config)


def test_model_result_repr() -> None:
    """Test ModelResult string representation."""
    model = Model(
        name="TestModel",
        function=PassThrough(),
        params={},
        r_squared=dt(0.8765),
        bounds=ParametersBounds({}, {}),
    )

    repr_str = repr(model)

    assert repr_str == f"ModelFit(name='{model.name}', R²={model.r_squared:.3f})"


def test_model_sort() -> None:
    """Test Model.sort."""
    r_squared_min = dt(0.1)
    r_squared_max = dt(0.3)
    model_list = [
        Model(
            name="a",
            function=PassThrough(),
            params={},
            r_squared=r_squared_max,
            bounds=ParametersBounds({}, {}),
        ),
        Model(
            name="b",
            function=PassThrough(),
            params={},
            r_squared=r_squared_min,
            bounds=ParametersBounds({}, {}),
        ),
    ]
    Model.sort(model_list)
    assert model_list[0].r_squared == r_squared_max
    assert model_list[1].r_squared == r_squared_min


def test_model_to_dict() -> None:
    """Test model conversion to dictionary."""
    linear_model = Model(
        name="test_linear",
        function=Linear(),
        params={"a": dt(1.0), "b": dt(0.5)},
        bounds=ParametersBounds(min={"a": dt(-1.0), "b": dt(-5.0)}, max={"a": dt(1.0), "b": dt(5.0)}),
        r_squared=dt(0.2),
    )
    assert linear_model.to_dict({Linear: "Linear"}) == {
        "name": "test_linear",
        "function": "Linear",
        "params": {"a": 1.0, "b": 0.5},
        "bounds": {"min": {"a": -1.0, "b": -5.0}, "max": {"a": 1.0, "b": 5.0}},
        "r_squared": 0.20000000298023224,
    }


def test_model_to_dict_missing_mapping() -> None:
    """Test Error raising when function mapping is missing during dict conversion."""
    model = Model(
        name="Passthrough",
        function=PassThrough(),
        params={},
        bounds=ParametersBounds(min={}, max={}),
        r_squared=neg_inf,
    )
    with pytest.raises(FunctionNotFoundInMappingError):
        model.to_dict({})


def test_model_from_dict() -> None:
    """Test model creation from dictionary."""
    d = {
        "name": "test_linear",
        "function": "Linear",
        "params": {"a": 1.0, "b": 0.5},
        "bounds": {"min": {"a": -1.0, "b": -5.0}, "max": {"a": 1.0, "b": 5.0}},
        "r_squared": 0.20000000298023224,
    }
    linear_model = Model.from_dict(d, {"Linear": Linear})
    assert linear_model.name == d["name"]
    assert isinstance(linear_model.function, Linear)
    assert linear_model.params == {p: dt(v) for p, v in d["params"].items()}
    assert linear_model.bounds == ParametersBounds(
        min={p: dt(v) for p, v in d["bounds"]["min"].items()}, max={p: dt(v) for p, v in d["bounds"]["max"].items()}
    )
    assert linear_model.r_squared == d["r_squared"]


def test_model_from_dict_missing_mapping() -> None:
    """Test model creation from dictionary error in case of missing mapping."""
    d = {
        "name": "passthrough",
        "function": "PassThrough",
        "params": {},
        "bounds": {"min": {}, "max": {}},
        "r_squared": neg_inf,
    }
    with pytest.raises(FunctionIdNotFoundInMappingError):
        Model.from_dict(d, {})


def test_model_from_dict_bad_dict() -> None:
    """Test model creation error when the dictionary doesn't have the required content."""
    with pytest.raises(ModelParsingError):
        Model.from_dict({}, {})

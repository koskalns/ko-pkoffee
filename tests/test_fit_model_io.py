"""Test suite of `pkoffee.fit_model_io`."""

from dataclasses import astuple
from pathlib import Path

from pkoffee.data import data_dtype as dt
from pkoffee.fit_model import Model
from pkoffee.fit_model_io import (
    load_models,
    load_models_json,
    load_models_toml,
    pkoffee_function_id_mapping,
    save_models,
    save_models_json,
    save_models_toml,
)
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    ParametricFunction,
    Peak2Model,
    PeakModel,
    Quadratic,
)


def test_pkoffee_function_id_mapping() -> None:
    """Test the function to string identifier bijection mapping."""
    mapping = pkoffee_function_id_mapping()
    assert mapping["Quadratic"] == Quadratic
    assert mapping.inv[Quadratic] == "Quadratic"
    assert mapping["Logistic"] == Logistic
    assert mapping.inv[Logistic] == "Logistic"
    assert mapping["MichaelisMentenSaturation"] == MichaelisMentenSaturation
    assert mapping.inv[MichaelisMentenSaturation] == "MichaelisMentenSaturation"
    assert mapping["PeakModel"] == PeakModel
    assert mapping.inv[PeakModel] == "PeakModel"
    assert mapping["Peak2Model"] == Peak2Model
    assert mapping.inv[Peak2Model] == "Peak2Model"


def assert_model_equal(lhs: Model, rhs: Model) -> None:
    """Assert 2 models are equal."""
    # Assert both are models
    assert isinstance(lhs, Model)
    assert type(lhs) is type(rhs)
    # Assert models functions are the same
    assert type(lhs.function) is type(rhs.function)
    # assert equality of all other fields
    for lhs_field, rhs_field in zip(astuple(lhs), astuple(rhs), strict=True):
        if isinstance(lhs_field, ParametricFunction):
            continue
        assert lhs_field == rhs_field


def test_save_load_models_json(tmp_path: Path) -> None:
    """Test model saving to json."""
    ref_model = Model(
        name="TestModel",
        function=Quadratic(),
        params=Quadratic.param_guess(dt(0.0)),
        bounds=Quadratic.param_bounds(),
        r_squared=dt(0.8765),
    )
    model_json_file = tmp_path / "test_save_models_json.json"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models_json([ref_model.to_dict(fct_id_to_fct.inv)], model_json_file)
    read_model = load_models_json(model_json_file, fct_id_to_fct)[0]
    assert_model_equal(ref_model, read_model)


def test_save_models_toml(tmp_path: Path) -> None:
    """Test model saving to toml."""
    ref_model = Model(
        name="TestModel",
        function=Quadratic(),
        params=Quadratic.param_guess(dt(0.0)),
        bounds=Quadratic.param_bounds(),
        r_squared=dt(0.8765),
    )
    model_toml_file = tmp_path / "test_save_models_toml.toml"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models_toml([ref_model.to_dict(fct_id_to_fct.inv)], model_toml_file)
    read_model = load_models_toml(model_toml_file, fct_id_to_fct)[0]
    assert_model_equal(ref_model, read_model)


def test_save_models(tmp_path: Path) -> None:
    """Test saving several models."""
    models = [
        Model(
            name="QuadraticBest",
            function=Quadratic(),
            params=Quadratic.param_guess(dt(0.0)),
            bounds=Quadratic.param_bounds(),
            r_squared=dt(0.8765),
        ),
        Model(
            name="QuadraticWorse",
            function=Quadratic(),
            params=Quadratic.param_guess(dt(0.0)),
            bounds=Quadratic.param_bounds(),
            r_squared=dt(0.2244),
        ),
        Model(
            name="Logistic",
            function=Logistic(),
            params=Logistic.param_guess(dt(0), dt(5), dt(0), dt(10)),
            bounds=Logistic.param_bounds(),
            r_squared=dt(0.7542),
        ),
    ]
    model_toml_file = tmp_path / "test_save_models_toml.toml"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models(models, fct_id_to_fct.inv, model_toml_file)
    read_models = load_models(model_toml_file, fct_id_to_fct)
    for ref_md, read_md in zip(models, read_models, strict=True):  # pyright: ignore[reportArgumentType] model_dicts is iterable alright
        assert_model_equal(ref_md, read_md)

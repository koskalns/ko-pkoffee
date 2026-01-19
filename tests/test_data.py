"""Unit tests for data loading and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkoffee.data import (
    ColumnTypeError,
    CSVReadError,
    MissingColumnsError,
    RequiredColumn,
    curate,
    data_dtype,
    extract_arrays,
    load_csv,
    validate,
)


def test_validate() -> None:
    """Test validate with valide DataFrame."""
    assert validate(pd.DataFrame({"cups": [0], "productivity": [1.2]})) is None


def test_validate_wrong_type() -> None:
    """Test validate with incorrect required column type."""
    with pytest.raises(ColumnTypeError):
        validate(pd.DataFrame({"cups": [0], "productivity": ["a"]}))


def test_validate_missing_column() -> None:
    """Test validate with missing required column in DataFrame."""
    with pytest.raises(MissingColumnsError):
        validate(pd.DataFrame({f"{RequiredColumn.CUPS}": [1], "notproductivity": [1.2]}))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            pd.DataFrame({"cups": [1, np.nan, 2], "productivity": [1.2, 2.1, np.nan]}),
            pd.DataFrame({"cups": [1.0], "productivity": [1.2]}),
        ),
        (
            pd.DataFrame({"cups": [np.nan, np.nan, np.nan], "productivity": [1.2, 2.1, 3.4]}),
            pd.DataFrame({"cups": [], "productivity": []}),
        ),
        (
            pd.DataFrame({"cups": [1, 1, 4], "productivity": [1.2, 2.1, 0.5]}),
            pd.DataFrame({"cups": [1, 1, 4], "productivity": [1.2, 2.1, 0.5]}),
        ),
    ],
)
def test_currate(data: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Test curate with different DataFrames containing nans."""
    assert curate(data).equals(expected)


def test_load_csv_valid_file(tmp_path: Path) -> None:
    """Test loading valid CSV."""
    data_file = tmp_path / "valid.csv"
    cups = np.array([1, 2, 3], dtype=int)
    prod = np.array([2.3, 1.2, 4.8], dtype=data_dtype)
    np.savetxt(
        data_file,
        np.stack([cups, prod], axis=1),
        fmt=["%d", "%10.4f"],
        delimiter=",",
        header=f"{RequiredColumn.CUPS},{RequiredColumn.PRODUCTIVITY}",
        comments="",
    )

    data = load_csv(data_file)
    assert RequiredColumn.CUPS in data.columns
    assert RequiredColumn.PRODUCTIVITY in data.columns
    assert np.isclose(data[RequiredColumn.CUPS].to_numpy(), cups).all()
    assert np.isclose(data[RequiredColumn.PRODUCTIVITY].to_numpy(), prod).all()
    assert data.dtypes[RequiredColumn.CUPS] == np.int64
    assert data.dtypes[RequiredColumn.PRODUCTIVITY] == np.float64


def test_load_csv_missing_file() -> None:
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_csv(Path("nonexistent_file.csv"))


def test_load_csv_missing_columns(tmp_path: Path) -> None:
    """Test MissingColumnsError is raised for missing required columns."""
    wrong_col_file = tmp_path / "missing_columns.csv"
    data = np.stack([[1], [2.3]], axis=1)
    np.savetxt(
        wrong_col_file,
        data,
        fmt=["%d", "%10.4f"],
        delimiter=",",
        header=f"{RequiredColumn.CUPS},wrong_column",
        comments="",
    )
    with pytest.raises(MissingColumnsError, match="Missing required columns"):
        load_csv(wrong_col_file)


def test_load_data_with_nan_values(tmp_path: Path) -> None:
    """Test that rows with NaN values are dropped."""
    data_file = tmp_path / "valid_with_nan.csv"
    with data_file.open("w") as fh:
        fh.write(f"{RequiredColumn.CUPS},{RequiredColumn.PRODUCTIVITY}\n")
        fh.write("1,10.5\n")
        fh.write("2,\n")  # Missing productivity
        fh.write("3,18.2\n")

    data = load_csv(data_file)
    expected = pd.DataFrame({RequiredColumn.CUPS: [1, 3], RequiredColumn.PRODUCTIVITY: [10.5, 18.2]})
    assert data.equals(expected)


def test_load_data_with_extra_values(tmp_path: Path) -> None:
    """Test that rows with NaN values are dropped."""
    data_file = tmp_path / "valid_with_nan.csv"
    with data_file.open("w") as fh:
        fh.write(f"{RequiredColumn.CUPS},{RequiredColumn.PRODUCTIVITY}\n")
        fh.write("1,2.1\n")
        # try to read the file while it is open for write
        with pytest.raises(CSVReadError):
            load_csv(data_file)


def test_extract_arrays() -> None:
    """Test extracting numpy arrays from DataFrame."""
    cups_ref = np.array([1, 2, 3], dtype=int)
    productivity_ref = np.array([10.5, 15.3, 18.2], dtype=np.float64)
    data = pd.DataFrame({RequiredColumn.CUPS: cups_ref, RequiredColumn.PRODUCTIVITY: productivity_ref})

    cups, productivity = extract_arrays(data)

    assert np.allclose(cups_ref, cups)
    assert np.allclose(productivity_ref, productivity)

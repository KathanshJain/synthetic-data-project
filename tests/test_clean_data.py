"""Author: Kathansh (Data Analysis Track)

Unit Tests for Step 1: Data Cleaning Pipeline
==============================================
Tests verify that:
1. The cleaned data file exists and is populated.
2. 'row_id' exists as the unique identifier in the first column.
3. 'fnlwgt' and 'education' columns have been dropped.
4. No null / NaN values exist anywhere in the dataset.
5. Target values are strictly '<=50K' and '>50K'.
6. Missing values have been converted to the explicit category "Missing".
"""

from pathlib import Path
import pandas as pd
import pytest


PROCESSED_DATA_PATH = Path("data/processed/adult_cleaned.csv")


@pytest.fixture(scope="module")
def cleaned_df():
    """Load cleaned dataset fixture for testing."""
    assert PROCESSED_DATA_PATH.exists(), f"Cleaned dataset not found at {PROCESSED_DATA_PATH}"
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def test_file_exists_and_not_empty(cleaned_df):
    """Ensure the processed dataset has rows and columns."""
    assert len(cleaned_df) > 40000, f"Expected > 40,000 rows, got {len(cleaned_df)}"
    assert len(cleaned_df.columns) == 14, f"Expected 14 columns, got {len(cleaned_df.columns)}"


def test_row_id_column(cleaned_df):
    """Ensure 'row_id' is present, unique, and sequential."""
    assert "row_id" in cleaned_df.columns, "'row_id' column is missing!"
    assert cleaned_df.columns[0] == "row_id", "'row_id' should be the first column!"
    assert cleaned_df["row_id"].nunique() == len(cleaned_df), "'row_id' values must be unique!"
    assert cleaned_df["row_id"].iloc[0] == 0, "'row_id' should start at 0"


def test_dropped_columns(cleaned_df):
    """Ensure 'fnlwgt' and 'education' columns are absent."""
    assert "fnlwgt" not in cleaned_df.columns, "'fnlwgt' was not dropped!"
    assert "education" not in cleaned_df.columns, "'education' was not dropped!"
    # 'education-num' must be retained
    assert "education-num" in cleaned_df.columns, "'education-num' should be retained!"


def test_no_null_values(cleaned_df):
    """Verify that there are zero NaN / null values anywhere."""
    total_nulls = cleaned_df.isnull().sum().sum()
    assert total_nulls == 0, f"Found {total_nulls} null values in the cleaned data!"


def test_missing_category_exists(cleaned_df):
    """Check that 'Missing' category is present in columns that had missing values (e.g. workclass)."""
    assert "workclass" in cleaned_df.columns
    workclass_vals = set(cleaned_df["workclass"].unique())
    assert "Missing" in workclass_vals, "Expected 'Missing' category in 'workclass' column!"


def test_target_values(cleaned_df):
    """Verify target column has strictly the binary categories '<=50K' and '>50K'."""
    assert "income" in cleaned_df.columns, "Target column 'income' is missing!"
    unique_targets = set(cleaned_df["income"].unique())
    expected_targets = {"<=50K", ">50K"}
    assert unique_targets == expected_targets, f"Expected {expected_targets}, got {unique_targets}"

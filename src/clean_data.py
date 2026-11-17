"""Author: Kathansh (Data Analysis Track)

Step 1: Load and Clean Dataset
================================
What this file does:
1. Downloads or loads the raw Adult Census Income dataset from OpenML.
2. Removes exact duplicate rows BEFORE dropping any columns.
3. Drops the 'fnlwgt' and 'education' columns as configured in config.yaml.
4. Replaces missing values (like '?' or NaN) with an explicit category named "Missing".
5. Adds a unique 'row_id' column for tracking each record through the pipeline.
6. Saves the cleaned dataset into data/processed/adult_cleaned.csv.

Why this file exists:
In any machine learning pipeline, data cleaning is the foundation. For synthetic
tabular data, cleaning decisions are especially critical:
- Dropping duplicates BEFORE column removal avoids falsely deleting records that
  only look identical because we dropped distinguishing features.
- Imputing missing values with "Missing" treats missingness as a real signal,
  which prevents data loss and allows generative models to learn patterns in missingness.
- Creating a unique 'row_id' ensures strict traceability and anti-leakage audits.
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def load_config(config_path: str = "config.yaml") -> dict:
    """Read configuration settings from config.yaml.
    
    Why: All settings must come from config.yaml so that parameters are never
    hardcoded in the logic, making experiments repeatable and clean.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fetch_raw_data(dataset_name: str, dataset_version: int, raw_save_path: Path) -> pd.DataFrame:
    """Download the dataset from OpenML or load it from the local cache if already saved.
    
    Why: We cache the raw dataset locally so that future runs do not depend on
    an active internet connection or risk network timeouts.
    """
    if raw_save_path.exists():
        print(f"[INFO] Found cached raw dataset at {raw_save_path}. Loading from disk...")
        raw_df = pd.read_csv(raw_save_path)
        return raw_df

    print(f"[INFO] Fetching '{dataset_name}' (version {dataset_version}) from OpenML...")
    dataset = fetch_openml(
        name=dataset_name,
        version=dataset_version,
        as_frame=True,
        parser="auto",
    )
    raw_df = dataset.frame.copy()

    # Save a pristine copy of the raw download
    raw_save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_save_path, index=False)
    print(f"[INFO] Saved pristine raw data to {raw_save_path} ({len(raw_df)} rows).")

    return raw_df


def clean_adult_data(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean the raw DataFrame following all rules in the project brief.
    
    Step-by-step logic:
    1. Drop exact duplicate rows while all original columns are still present.
    2. Standardize target column name to 'income'.
    3. Drop 'fnlwgt' and 'education' columns.
    4. Replace missing values with the category "Missing".
    5. Clean any trailing whitespace or dots in text columns.
    6. Add a unique 'row_id' column at the front.
    """
    df = raw_df.copy()
    initial_row_count = len(df)
    print(f"[INFO] Initial raw row count: {initial_row_count}")

    # Rule 1: Drop exact duplicate rows BEFORE dropping columns.
    # Why: If we drop columns first, rows that were previously distinct might
    # appear to be duplicates and get deleted mistakenly. Doing it first ensures
    # only genuine duplicate real-world entries are removed.
    df = df.drop_duplicates()
    duplicates_removed = initial_row_count - len(df)
    print(f"[INFO] Dropped {duplicates_removed} exact duplicate rows before dropping columns.")
    print(f"[INFO] Row count after duplicate removal: {len(df)}")

    # Target column standardization
    target_name = config["dataset"]["target"]
    if "class" in df.columns and target_name not in df.columns:
        # OpenML v2 names the target column 'class'
        df = df.rename(columns={"class": target_name})

    # Rule 2: Drop configured columns ('fnlwgt' and 'education')
    # Why 'fnlwgt'? It is an administrative census weight created by the Census
    # Bureau, not a personal characteristic, and can cause spurious correlations.
    # Why 'education'? 'education-num' already represents educational attainment
    # as an ordered integer scale, making the text column redundant.
    drop_cols = config["dataset"]["drop_columns"]
    columns_to_drop_present = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=columns_to_drop_present)
    print(f"[INFO] Dropped redundant columns: {columns_to_drop_present}")

    # Rule 3: Strip whitespace and standardize strings
    # In census datasets, categorical values often contain leading spaces like " Private"
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Rule 4: Clean the target column values (remove any trailing dots like '>50K.')
    positive_label = config["dataset"]["positive_label"]
    df[target_name] = df[target_name].str.rstrip(".")

    # Rule 5: Impute missing values with explicit category "Missing"
    # Why: For categorical features (like workclass, occupation, native-country),
    # a missing answer is often informative (e.g. unemployed people may leave workclass blank).
    # Replacing missing with "Missing" treats this as a distinct state and avoids dropping rows.
    # We replace '?', 'nan', and empty strings with "Missing".
    missing_placeholders = ["?", "nan", "None", ""]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace(missing_placeholders, "Missing")
        df[col] = df[col].fillna("Missing")

    # For numerical columns, verify if any nulls exist
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled missing numerical values in '{col}' with median: {median_val}")

    # Rule 6: Add a unique 'row_id' column starting from 0 at the very front
    # Why: Having an immutable identifier per record allows us to trace sample lineage
    # and strictly verify anti-leakage properties in later testing steps.
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    # Verify class balance
    target_counts = df[target_name].value_counts()
    target_percentages = df[target_name].value_counts(normalize=True) * 100
    print("\n--- Cleaned Dataset Target Class Balance ---")
    for label, count in target_counts.items():
        pct = target_percentages[label]
        print(f"  Class '{label}': {count} rows ({pct:.2f}%)")

    return df


def run_data_cleaning_pipeline(config_path: str = "config.yaml"):
    """Main function to run the full Step 1 data cleaning pipeline."""
    print("=" * 60)
    print("STEP 1: LOAD AND CLEAN DATASET (Author: Kathansh)")
    print("=" * 60)

    # 1. Load settings
    config = load_config(config_path)
    dataset_name = config["dataset"]["name"]
    dataset_version = config["dataset"]["version"]

    # 2. Paths
    raw_path = Path("data/raw/adult_raw.csv")
    processed_path = Path("data/processed/adult_cleaned.csv")

    # 3. Fetch raw data
    raw_df = fetch_raw_data(dataset_name, dataset_version, raw_path)

    # 4. Clean data
    cleaned_df = clean_adult_data(raw_df, config)

    # 5. Save cleaned data
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    print(f"\n[SUCCESS] Cleaned dataset successfully saved to {processed_path}")
    print(f"  Total Cleaned Records: {len(cleaned_df)}")
    print(f"  Total Cleaned Columns: {len(cleaned_df.columns)}")
    print(f"  Columns List: {list(cleaned_df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    run_data_cleaning_pipeline()

# tests/test_preprocess.py

import pandas as pd
import pytest
from ml_ids_analyzer.preprocessing.preprocess import (
    load_and_merge_csvs,
    clean_and_label,
    LABEL_COL,
)


def write_csv(path, df):
    """Helper to write a DataFrame to CSV with no index."""
    df.to_csv(path, index=False)


def test_load_and_merge_csvs(tmp_path):
    # Create two small CSVs in a temporary directory
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5], "B": [6]})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(raw_dir / "one.csv", df1)
    write_csv(raw_dir / "two.csv", df2)

    # Point the function at our tmp dir
    merged = load_and_merge_csvs(str(raw_dir))
    # Expect 3 rows, 2 columns
    assert merged.shape == (3, 2)
    # Values should appear in order
    assert merged["A"].tolist() == [1, 2, 5]


def test_clean_and_label_drops_and_maps():
    # Construct a DataFrame with:
    # - column 'X' all NaN (should drop)
    # - constant column 'Y' (should drop)
    # - good column 'feat'
    # - label column with mixed case
    data = {
        "X": [None, None, None],
        "Y": [7, 7, 7],
        "feat": [1, None, 3],
        LABEL_COL: ["Benign", "attack", "BENIGN"],
    }
    df = pd.DataFrame(data)

    cleaned = clean_and_label(df)

    # 'X' and 'Y' gone, only 'feat' and LABEL_COL remain
    assert set(cleaned.columns) == {"feat", LABEL_COL}

    # All rows with any missing were dropped → only rows 0 and 2 remain
    assert cleaned.shape[0] == 2

    # Label mapping: 'Benign' and 'BENIGN' → 0; 'attack' → 1
    expected_labels = [0, 0]
    assert cleaned[LABEL_COL].tolist() == expected_labels


def test_clean_and_label_missing_label_column():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(KeyError):
        clean_and_label(df)

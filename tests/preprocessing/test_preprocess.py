import pandas as pd
import pytest

from ml_ids_analyzer.preprocessing.preprocess import (
    load_and_merge_csvs,
    clean_and_label,
    LABEL_COL,
    MISSING_THRESH,
)


def test_load_and_merge_csvs_success(tmp_path):
    """
    Create two small CSV files in a temporary directory.
    Ensure load_and_merge_csvs reads both, concatenates them,
    and returns a DataFrame of the expected shape and content.
    """
    # Arrange: create a temp directory with two CSVs
    dir_path = tmp_path / "raw_data"
    dir_path.mkdir()

    df1 = pd.DataFrame(
        {
            "A": [1, 2],
            "B": ["x", "y"],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [3],
            "B": ["z"],
        }
    )

    file1 = dir_path / "part1.csv"
    file2 = dir_path / "part2.csv"
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    # Act: merge
    merged = load_and_merge_csvs(str(dir_path))

    # Assert: 3 rows total, same columns, values match
    assert merged.shape == (3, 2)
    # Because sorted(["part1.csv", "part2.csv"]) => ["part1.csv", "part2.csv"]
    # The first two rows should be df1, then df2
    assert list(merged["A"]) == [1, 2, 3]
    assert list(merged["B"]) == ["x", "y", "z"]


def test_load_and_merge_csvs_directory_not_found(tmp_path):
    """
    If the directory does not exist, load_and_merge_csvs should raise FileNotFoundError.
    """
    nonexistent = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_and_merge_csvs(str(nonexistent))

    assert "Data directory not found" in str(excinfo.value)


def test_load_and_merge_csvs_no_csv_files(tmp_path):
    """
    If the directory exists but contains no .csv files, FileNotFoundError should be raised.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    # Create a non-CSV file
    (empty_dir / "readme.txt").write_text("not a CSV")

    with pytest.raises(FileNotFoundError) as excinfo:
        load_and_merge_csvs(str(empty_dir))

    assert "No CSV files found in raw data directory" in str(excinfo.value)


def test_clean_and_label_drops_high_missing_and_constant_columns():
    """
    Create a DataFrame of fixed length and:
      - Compute the actual `min_non_na` based on MISSING_THRESH.
      - Build a column with exactly `min_non_na - 1` non-NA entries → should be dropped.
      - Build a constant column → should be dropped.
      - Build a fully valid column → should remain.
      - Label column with mixed-case values and one missing → mapping and row-drop logic tested.
    """
    # Choose a small number of rows
    n_rows = 5

    # Compute the minimum required non-NA for any column to be kept:
    #   min_non_na = ceil((1 - MISSING_THRESH) * n_rows)
    min_non_na = int(__import__("math").ceil((1.0 - MISSING_THRESH) * n_rows))

    # Now build data such that:
    #   - 'too_many_missing' has only (min_non_na - 1) non-null entries → WILL be dropped
    #   - 'constant_col' has same value everywhere → WILL be dropped
    #   - 'mixed' has all valid entries → WILL be kept
    #   - LABEL_COL has exactly one None (to be dropped) and the rest mixed-case → gets mapped

    # Generate exactly (min_non_na - 1) non-null values:
    non_null_count = min_non_na - 1
    # Fill the first 'non_null_count' entries with 1, then the rest with None
    too_many_missing = [1] * non_null_count + [None] * (
        n_rows - non_null_count
    )

    data = {
        "too_many_missing": too_many_missing,
        "constant_col": [7] * n_rows,
        "mixed": [10, 20, 30, 40, 50],
        LABEL_COL: ["Benign", "BENIGN", "Malware", None, "malware"],
    }
    df = pd.DataFrame(data)

    # Sanity check: the actual number of non-null in 'too_many_missing'
    assert df["too_many_missing"].count() == non_null_count

    # Act: clean + label
    cleaned = clean_and_label(df)

    # Assert:
    # 1) 'too_many_missing' must be dropped because its non-NA count < min_non_na
    assert "too_many_missing" not in cleaned.columns

    # 2) 'constant_col' must be dropped (only one unique value)
    assert "constant_col" not in cleaned.columns

    # 3) 'mixed' must remain, and its values for rows without any NA should match
    assert "mixed" in cleaned.columns
    # We dropped only the row where LABEL_COL was None (index 3) → so 'mixed' values should be [10,20,30,50]
    assert list(cleaned["mixed"]) == [10, 20, 30, 50]

    # 4) After dropping rows with missing, we expect length = n_rows - 1 = 4
    #    (only the row with LABEL_COL=None is removed)
    assert len(cleaned) == n_rows - 1

    # 5) Label mapping: uppercase "BENIGN" → 0, anything else → 1
    #    Original labels (in order, excluding the None at index 3):
    #      ["Benign", "BENIGN", "Malware", "malware"] → uppercased: ["BENIGN", "BENIGN", "MALWARE", "MALWARE"]
    #      Mapped: [0, 0, 1, 1]
    mapped = list(cleaned[LABEL_COL])
    assert mapped == [0, 0, 1, 1]


def test_clean_and_label_raises_if_label_missing():
    """
    If the DataFrame does not contain LABEL_COL, clean_and_label should raise KeyError.
    """
    df_no_label = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    with pytest.raises(KeyError) as excinfo:
        clean_and_label(df_no_label)

    assert f"Expected a column named '{LABEL_COL}'" in str(excinfo.value)

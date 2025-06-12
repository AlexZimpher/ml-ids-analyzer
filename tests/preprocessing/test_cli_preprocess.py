# File: tests/preprocessing/test_cli_preprocess.py

import pandas as pd
from click.testing import CliRunner

from ml_ids_analyzer.preprocessing.preprocess import main as preprocess_cli


def test_preprocess_cli_success(tmp_path):
    """
    1) Create two small CSV files under tmp_path/"raw_data".
    2) Invoke the CLI: mlids-preprocess --data-dir <raw_data> --output-file <out.csv>.
    3) Read the output CSV and verify that:
       - The merged rows equal the total from both input files.
       - The cleaning logic (dropping high-missing, constant columns,
         dropping rows with missing labels, mapping labels) has been applied.
    """
    # Arrange: set up a directory with two CSVs
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()

    df1 = pd.DataFrame(
        {
            "A": [1, None, 3],
            "constant": [5, 5, 5],
            "Label": ["Benign", "Malware", "Benign"],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [4, 5],
            "constant": [5, 5],
            "Label": ["Malware", None],
        }
    )

    (raw_dir / "file1.csv").write_text(df1.to_csv(index=False))
    (raw_dir / "file2.csv").write_text(df2.to_csv(index=False))

    output_file = tmp_path / "cleaned.csv"

    runner = CliRunner()
    result = runner.invoke(
        preprocess_cli,
        [
            "--data-dir",
            str(raw_dir),
            "--output-file",
            str(output_file),
        ],
    )

    # Assert CLI exited without error
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    cleaned = pd.read_csv(output_file)

    # After merging 5 rows, drop “constant” (all-5), then drop rows with any missing:
    #   file1 row1 (A=None) → dropped
    #   file2 row1 (Label=None) → dropped
    # Kept rows: (1,Benign), (3,Benign), (4,Malware) → 3 rows
    assert list(cleaned.columns) == ["A", "Label"]
    assert cleaned.shape[0] == 3
    assert list(cleaned["A"]) == [1.0, 3.0, 4.0]
    assert list(cleaned["Label"]) == [0, 0, 1]


def test_preprocess_cli_failure_missing_dir(tmp_path):
    """
    If the data-dir does not exist, the CLI should exit with a nonzero code,
    and no output file should be created.
    """
    nonexist = tmp_path / "no_such_dir"
    output_file = tmp_path / "out.csv"

    runner = CliRunner()
    result = runner.invoke(
        preprocess_cli,
        [
            "--data-dir",
            str(nonexist),
            "--output-file",
            str(output_file),
        ],
    )

    # Exit code must be nonzero
    assert result.exit_code != 0

    # Output file should not exist
    assert not output_file.exists()


def test_preprocess_cli_failure_no_csv(tmp_path):
    """
    If data-dir exists but contains no CSVs, the CLI should exit with a nonzero code,
    and no output file should be created.
    """
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("not a CSV")

    output_file = tmp_path / "out.csv"
    runner = CliRunner()
    result = runner.invoke(
        preprocess_cli,
        [
            "--data-dir",
            str(empty_dir),
            "--output-file",
            str(output_file),
        ],
    )

    assert result.exit_code != 0
    assert not output_file.exists()

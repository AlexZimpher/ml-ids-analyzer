# File: tests/inference/test_predict_cli.py

import pandas as pd
import numpy as np
import pytest
import joblib
from click.testing import CliRunner
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import ml_ids_analyzer.config as config_module
from ml_ids_analyzer.inference.predict import main as predict_cli  # entry point


@pytest.fixture(autouse=True)
def set_features(monkeypatch):
    """
    Ensure cfg['features'] is set to ["f1", "f2"] so predict_alerts
    and the CLI know which columns to read from the input CSV.
    """
    monkeypatch.setitem(config_module.cfg, "features", ["f1", "f2"])
    yield


@pytest.fixture
def tmp_model_and_scaler(tmp_path):
    """
    Create and save a small RandomForestClassifier and StandardScaler.
    Returns (model_file, scaler_file).
    """
    # 1) Build a tiny dataset
    X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y_train = np.array([0, 1, 0, 1])

    # 2) Train RandomForest
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X_train, y_train)

    # 3) Fit StandardScaler on the same X_train
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 4) Save both to .joblib files
    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return str(model_path), str(scaler_path)


@pytest.fixture
def input_csv(tmp_path):
    """
    Create a simple CSV with columns f1 and f2, four rows.
    Returns path to that CSV file.
    """
    df = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "f2": [1, 0, 0, 1],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_predict_cli_success(input_csv, tmp_model_and_scaler, tmp_path):
    """
    - Given valid input CSV, model-file, and scaler-file,
      when invoking the CLI with --threshold 0.5 and --output-file <tmp>,
      it should exit code 0, write an output CSV, and the contents should
      match manually computed probabilities and labels.
    """
    model_file, scaler_file = tmp_model_and_scaler
    output_path = tmp_path / "predictions_out.csv"

    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        [
            "--input-file", input_csv,
            "--model-file", model_file,
            "--scaler-file", scaler_file,
            "--output-file", str(output_path),
            "--threshold", "0.5",
        ],
    )

    # 1) CLI should succeed
    assert result.exit_code == 0, f"CLI failed unexpectedly:\n{result.output}"

    # 2) Output CSV should exist
    assert output_path.exists() and output_path.is_file(), "Output CSV not created"

    # 3) Read the output and verify columns
    out_df = pd.read_csv(output_path)
    assert list(out_df.columns) == ["f1", "f2", "prob_attack", "pred_attack"]

    # 4) Manually compute expected probabilities and labels
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X)
    model = joblib.load(model_file)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Compare values
    assert np.allclose(out_df["prob_attack"].values, probs)
    assert np.array_equal(out_df["pred_attack"].values, preds)


@pytest.mark.parametrize("missing_arg", ["input", "model", "scaler"])
def test_predict_cli_missing_file(input_csv, tmp_model_and_scaler, tmp_path, missing_arg):
    """
    If any of --input-file, --model-file, or --scaler-file is missing,
    the CLI should exit with code 1.
    """
    model_file, scaler_file = tmp_model_and_scaler
    output_path = tmp_path / "out.csv"

    # Determine which argument to pass as a nonexistent path
    if missing_arg == "input":
        args = [
            "--input-file", str(tmp_path / "no_input.csv"),
            "--model-file", model_file,
            "--scaler-file", scaler_file,
            "--output-file", str(output_path),
            "--threshold", "0.5",
        ]
    elif missing_arg == "model":
        args = [
            "--input-file", input_csv,
            "--model-file", str(tmp_path / "no_model.joblib"),
            "--scaler-file", scaler_file,
            "--output-file", str(output_path),
            "--threshold", "0.5",
        ]
    else:  # missing_arg == "scaler"
        args = [
            "--input-file", input_csv,
            "--model-file", model_file,
            "--scaler-file", str(tmp_path / "no_scaler.joblib"),
            "--output-file", str(output_path),
            "--threshold", "0.5",
        ]

    runner = CliRunner()
    result = runner.invoke(predict_cli, args)
    assert result.exit_code == 1, f"Expected exit code 1 when '{missing_arg}' is missing"


def test_predict_cli_invalid_threshold_non_numeric(input_csv, tmp_model_and_scaler, tmp_path):
    """
    If threshold is not a valid float (e.g. "not_a_number"), Click should
    error out with a non-zero exit code.
    """
    model_file, scaler_file = tmp_model_and_scaler
    output_path = tmp_path / "out.csv"

    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        [
            "--input-file", input_csv,
            "--model-file", model_file,
            "--scaler-file", scaler_file,
            "--output-file", str(output_path),
            "--threshold", "not_a_number",
        ],
    )

    assert result.exit_code != 0, f"Expected non-zero exit code for non-numeric threshold"


@pytest.mark.parametrize("threshold", ["0", "1", "-0.1"])
def test_predict_cli_numeric_thresholds_accepted(input_csv, tmp_model_and_scaler, tmp_path, threshold):
    """
    Any string that can be cast to float (even "0", "1", or "-0.1") is accepted.
    The CLI should exit with code 0, though downstream behavior may vary.
    """
    model_file, scaler_file = tmp_model_and_scaler
    output_path = tmp_path / f"out_{threshold}.csv"

    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        [
            "--input-file", input_csv,
            "--model-file", model_file,
            "--scaler-file", scaler_file,
            "--output-file", str(output_path),
            "--threshold", threshold,
        ],
    )

    # CLI should still succeed on numeric thresholds
    assert result.exit_code == 0, f"CLI failed for numeric threshold '{threshold}': {result.output}"

    # Output file should exist
    assert output_path.exists() and output_path.is_file(), f"Output not created for threshold '{threshold}'"

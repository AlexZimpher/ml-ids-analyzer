# File: tests/modeling/test_train_cli.py

import pandas as pd
import numpy as np
import pytest
import joblib
from click.testing import CliRunner
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import ml_ids_analyzer.modeling.train as train_module
import ml_ids_analyzer.config as config_module


@pytest.fixture
def minimal_cleaned_csv(tmp_path, monkeypatch):
    """
    Create a tiny 'cleaned' CSV with exactly the columns expected by train_model:
    - Feature columns (cfg['features'])
    - Label column (cfg['label_column'])
    Then monkeypatch DATA_FILE so the pipeline reads from this CSV.
    """
    # Determine label column name and feature list from cfg
    label_col = config_module.cfg["label_column"]
    features = config_module.cfg.get("features", [])

    # If no features are configured, set a default list of 3 features
    if not features:
        features = ["f1", "f2", "f3"]
        monkeypatch.setitem(config_module.cfg, "features", features)

    # Build a small DataFrame: 20 rows, features, balanced binary label
    n = 20
    X = np.random.RandomState(0).randn(n, len(features))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    df = pd.DataFrame(X, columns=features)
    df[label_col] = y

    # Write to CSV under tmp_path
    csv_path = tmp_path / "cleaned.csv"
    df.to_csv(csv_path, index=False)

    # Monkeypatch DATA_FILE in train_module to point to our CSV
    monkeypatch.setitem(train_module.__dict__, "DATA_FILE", str(csv_path))

    return csv_path


@pytest.fixture
def override_output_paths(tmp_path, monkeypatch):
    """
    Monkeypatch all cfg['paths'] entries so that training artifacts
    go under tmp_path, and override the module‐level constants in train_module
    so that train_model writes into tmp_path.
    """
    # Ensure cfg["paths"] exists
    if "paths" not in config_module.cfg:
        config_module.cfg["paths"] = {}

    # Override cfg so that nothing crashes if any code re‐reads cfg again
    monkeypatch.setitem(config_module.cfg["paths"], "output_dir", str(tmp_path))
    monkeypatch.setitem(config_module.cfg["paths"], "predictions", str(tmp_path / "predictions.csv"))
    monkeypatch.setitem(config_module.cfg["paths"], "model_file", str(tmp_path / "random_forest_model.joblib"))
    monkeypatch.setitem(config_module.cfg["paths"], "scaler_file", str(tmp_path / "scaler.joblib"))

    # Now override the module‐level constants in train_module directly
    # (these were already bound at import time to whatever cfg had originally).
    monkeypatch.setattr(train_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(train_module, "PRED_CSV", str(tmp_path / "predictions.csv"))
    monkeypatch.setattr(train_module, "MODEL_FILE", str(tmp_path / "random_forest_model.joblib"))
    monkeypatch.setattr(train_module, "SCALER_FILE", str(tmp_path / "scaler.joblib"))

    return tmp_path


def test_train_cli_creates_artifacts(minimal_cleaned_csv, override_output_paths, monkeypatch):
    """
    1) Given a minimal cleaned CSV and overridden output paths under tmp_path,
       invoke the training CLI with no arguments (so it reads from DATA_FILE).
    2) Verify that after execution:
       - predictions.csv exists and has the expected columns.
       - model_file and scaler_file exist and load correctly.
       - Confusion‐matrix PNG and SHAP summary PNG exist under tmp_path.
    """
    runner = CliRunner()

    # Invoke the CLI command object: train_module.cli (not main)
    result = runner.invoke(train_module.cli, [])

    # Assert exit code == 0
    assert result.exit_code == 0, f"Training CLI failed:\n{result.output}"

    tmp = override_output_paths

    # 1) Check predictions CSV
    pred_csv = tmp / "predictions.csv"
    assert pred_csv.exists() and pred_csv.is_file(), "predictions.csv not created"
    df_pred = pd.read_csv(pred_csv)
    # Expect columns ["Actual", "Predicted"]
    assert set(df_pred.columns) == {"Actual", "Predicted"}
    # Should have at least one row
    assert len(df_pred) >= 1

    # 2) Check model file
    model_file = tmp / "random_forest_model.joblib"
    assert model_file.exists() and model_file.is_file(), "random_forest_model.joblib not created"
    loaded_model = joblib.load(model_file)
    assert isinstance(loaded_model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier"

    # 3) Check scaler file
    scaler_file = tmp / "scaler.joblib"
    assert scaler_file.exists() and scaler_file.is_file(), "scaler.joblib not created"
    loaded_scaler = joblib.load(scaler_file)
    assert isinstance(loaded_scaler, StandardScaler), "Loaded scaler is not a StandardScaler"

    # 4) Check confusion‐matrix PNG (any file containing "confusion_matrix.png")
    png_files = list(tmp.glob("*confusion_matrix.png"))
    assert png_files, "Confusion matrix PNG not created"
    for p in png_files:
        assert p.stat().st_size > 0, f"{p.name} is empty"

    # 5) Check SHAP summary PNG
    shap_png = tmp / "shap_summary.png"
    assert shap_png.exists() and shap_png.is_file(), "shap_summary.png not created"
    assert shap_png.stat().st_size > 0, "shap_summary.png is empty"

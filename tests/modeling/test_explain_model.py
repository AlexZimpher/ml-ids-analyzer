# File: tests/modeling/test_explain_model.py

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Import explain_model from its actual location under explainability
from ml_ids_analyzer.explainability.explain import explain_model
import ml_ids_analyzer.config as config_module


def test_explain_model_creates_shap_summary(tmp_path, monkeypatch):
    """
    1) Train a tiny RandomForestClassifier on synthetic data.
    2) Monkeypatch cfg['paths']['output_dir'] to point to tmp_path.
    3) Call explain_model(model, X_train).
    4) Verify that 'shap_summary.png' appears under tmp_path and is non-empty.
    """
    # Arrange: generate a small dataset and fit a RandomForest
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, y)

    # Ensure cfg["paths"] exists
    if "paths" not in config_module.cfg:
        config_module.cfg["paths"] = {}
    # Monkeypatch cfg["paths"]["output_dir"] so explain_model writes into tmp_path
    monkeypatch.setitem(
        config_module.cfg["paths"], "output_dir", str(tmp_path)
    )

    # Act: run explain_model
    explain_model(model, X)

    # Assert: shap_summary.png is created under tmp_path and is non-zero size
    output_file = tmp_path / "shap_summary.png"
    assert (
        output_file.exists() and output_file.is_file()
    ), f"Expected shap_summary.png in {tmp_path}, but not found."
    assert output_file.stat().st_size > 0, "Saved SHAP summary PNG is empty."

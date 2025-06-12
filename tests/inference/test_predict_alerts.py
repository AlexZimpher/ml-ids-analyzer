# File: tests/inference/test_predict_alerts.py

import numpy as np
import pandas as pd
import pytest
import ml_ids_analyzer.config as config_module
from ml_ids_analyzer.inference.predict import predict_alerts


class DummyScaler:
    """A scaler that returns the input array unchanged."""

    def transform(self, X):
        return X.values if isinstance(X, pd.DataFrame) else X


class DummyModel:
    """
    A “model” whose predict_proba returns a fixed probability for the positive class.
    We’ll store those probabilities in an array matching the number of rows.
    """

    def __init__(self, probs):
        # probs: 1D numpy array of length n_samples
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        # Return a 2D array [[1-p, p], ...] for each sample
        p = self._probs
        n = len(p)
        arr = np.zeros((n, 2), dtype=float)
        arr[:, 1] = p
        arr[:, 0] = 1 - p
        return arr


@pytest.fixture(autouse=True)
def set_cfg_features(monkeypatch):
    # Always set cfg['features'] to ["f1", "f2"] for all tests
    monkeypatch.setitem(config_module.cfg, "features", ["f1", "f2"])
    yield


def test_predict_alerts_all_below_threshold(tmp_path):
    """
    Given probabilities all below 0.5, predict_alerts should assign predicted_label = 0.
    """
    # Arrange: build a DataFrame with two feature columns
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 1.1, 1.2],
            "extra": ["a", "b", "c"],
        }
    )
    dummy_probs = [0.1, 0.2, 0.3]
    model = DummyModel(dummy_probs)
    scaler = DummyScaler()
    # Act: use threshold=0.5
    result = predict_alerts(model, scaler, df, threshold=0.5)
    # Assert: original columns remain
    assert list(result.columns) == [
        "f1",
        "f2",
        "extra",
        "prediction_prob",
        "predicted_label",
    ]
    # All probabilities < 0.5 → predicted_label should be 0
    assert np.allclose(
        np.asarray(result["prediction_prob"].values), np.asarray(dummy_probs)
    )
    assert np.array_equal(
        np.asarray(result["predicted_label"].values), np.zeros(3, dtype=int)
    )


def test_predict_alerts_mixed_probabilities():
    """
    Given a mix of probabilities above and below threshold, labels should follow.
    """
    df = pd.DataFrame(
        {
            "f1": [0, 1, 2, 3],
            "f2": [3, 2, 1, 0],
        }
    )
    dummy_probs = [0.0, 0.5, 0.75, 1.0]
    model = DummyModel(dummy_probs)
    scaler = DummyScaler()
    result = predict_alerts(model, scaler, df, threshold=0.75)
    assert np.allclose(
        np.asarray(result["prediction_prob"].values), np.asarray(dummy_probs)
    )
    expected_labels = np.array([0, 0, 1, 1], dtype=int)
    assert np.array_equal(
        np.asarray(result["predicted_label"].values), expected_labels
    )


def test_predict_alerts_without_scaler():
    """
    If scaler is None, predict_alerts should use X without scaling.
    """
    df = pd.DataFrame(
        {
            "f1": [10.0, 20.0],
            "f2": [30.0, 40.0],
            "note": ["x", "y"],
        }
    )
    dummy_probs = [0.6, 0.4]
    model = DummyModel(dummy_probs)
    scaler = None  # No scaling
    result = predict_alerts(model, scaler, df, threshold=0.5)
    assert list(result.columns) == [
        "f1",
        "f2",
        "note",
        "prediction_prob",
        "predicted_label",
    ]
    assert np.allclose(
        np.asarray(result["prediction_prob"].values), np.asarray(dummy_probs)
    )
    assert np.array_equal(
        np.asarray(result["predicted_label"].values),
        np.array([1, 0], dtype=int),
    )


def test_predict_alerts_preserves_input_order_and_dtype():
    """
    The returned DataFrame should preserve input order and dtypes for existing columns.
    """
    df = pd.DataFrame(
        {
            "f1": pd.Series([5, 6, 7], dtype=np.int64),
            "f2": pd.Series([8, 9, 10], dtype=np.int64),
        }
    )
    dummy_probs = [0.2, 0.8, 0.2]
    model = DummyModel(dummy_probs)
    scaler = DummyScaler()
    result = predict_alerts(model, scaler, df, threshold=0.5)
    assert pd.api.types.is_integer_dtype(result["f1"].dtype)
    assert pd.api.types.is_integer_dtype(result["f2"].dtype)
    assert result.loc[1, "prediction_prob"] == pytest.approx(0.8)
    assert result.loc[1, "predicted_label"] == 1

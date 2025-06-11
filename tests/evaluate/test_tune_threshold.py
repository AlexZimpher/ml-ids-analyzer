import numpy as np
import pytest

# Correct import path
from ml_ids_analyzer.modeling.evaluate import tune_threshold


class DummyModel:
    """
    A “model” whose predict_proba returns perfect probabilities:
    - For y=0 → [1.0, 0.0]
    - For y=1 → [0.0, 1.0]
    """

    def __init__(self, y_true):
        self._y_true = np.asarray(y_true)

    def predict_proba(self, X_val):
        probs = np.zeros((len(self._y_true), 2), dtype=float)
        for i, y in enumerate(self._y_true):
            if y == 1:
                probs[i, 1] = 1.0
            else:
                probs[i, 0] = 1.0
        return probs


def test_tune_threshold_perfect_separation():
    """
    With perfect separation (prob=1.0 for y=1, prob=0.0 for y=0),
    precision_recall_curve yields thresholds [0.0, 1.0], and the best F1
    occurs at threshold=1.0. We assert that tune_threshold returns exactly 1.0.
    """
    # Arrange
    y_val = [0, 1, 0, 1]
    X_val = np.zeros((len(y_val), 3))  # features are unused by DummyModel
    dummy = DummyModel(y_val)

    # Act
    best_thr = tune_threshold(dummy, X_val, y_val)

    # Assert: because F1 is maximized at threshold=1.0, tune_threshold should return 1.0
    assert best_thr == pytest.approx(1.0), f"Expected threshold 1.0, got {best_thr}"


def test_tune_threshold_all_zeros():
    """
    If y_val is all zeros, precision_recall_curve produces thresholds [0.0],
    and F1 is zero for all; tune_threshold will pick thresholds[0] → 0.0.
    """
    # Arrange
    y_all_zero = [0, 0, 0]
    X_zero = np.zeros((3, 2))
    dummy_zero = DummyModel(y_all_zero)

    # Act
    best_thr_zero = tune_threshold(dummy_zero, X_zero, y_all_zero)

    # Assert: should return 0.0
    assert best_thr_zero == pytest.approx(
        0.0
    ), f"Expected threshold 0.0, got {best_thr_zero}"


def test_tune_threshold_all_ones():
    """
    If y_val is all ones, precision_recall_curve produces thresholds [1.0],
    and F1 is 1.0 at threshold=1.0. tune_threshold should return 1.0.
    """
    # Arrange
    y_all_one = [1, 1, 1]
    X_one = np.zeros((3, 2))
    dummy_one = DummyModel(y_all_one)

    # Act
    best_thr_one = tune_threshold(dummy_one, X_one, y_all_one)

    # Assert: should return 1.0
    assert best_thr_one == pytest.approx(
        1.0
    ), f"Expected threshold 1.0, got {best_thr_one}"

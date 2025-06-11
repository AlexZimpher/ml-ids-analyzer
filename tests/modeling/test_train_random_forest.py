# File: tests/modeling/test_train_random_forest.py

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Import the functions and RF_CONFIG from the train module
import ml_ids_analyzer.modeling.train as train_module


def test_search_hyperparameters_with_minimal_config(monkeypatch):
    """
    Override RF_CONFIG to a minimal configuration so that search_hyperparameters
    can run a tiny RandomizedSearchCV without external files. Then assert:
      1) The returned model is a RandomForestClassifier.
      2) The model is fitted (i.e., has feature_importances_).
    """
    # Arrange: create a tiny synthetic dataset
    X_train, y_train = make_classification(
        n_samples=20,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )

    # Monkeypatch RF_CONFIG inside the train_module to a minimal dict
    minimal_config = {
        "n_estimators": 5,
        "random_state": 0,
        # Provide a tiny search space for max_depth
        "search_params": {"max_depth": [1, 2]},
    }
    monkeypatch.setattr(train_module, "RF_CONFIG", minimal_config)

    # Also ensure _get_base_rf_params returns the base parameters correctly
    def fake_get_base_rf_params():
        # Return everything except "search_params"
        return {
            "n_estimators": minimal_config["n_estimators"],
            "random_state": minimal_config["random_state"],
        }

    monkeypatch.setattr(train_module, "_get_base_rf_params", fake_get_base_rf_params)

    # Act: call search_hyperparameters
    model = train_module.search_hyperparameters(X_train, y_train)

    # Assert: returned model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"

    # The model must be fitted: check for feature_importances_
    assert hasattr(
        model, "feature_importances_"
    ), "Model should have feature_importances_"
    fi = model.feature_importances_
    assert isinstance(fi, np.ndarray), "feature_importances_ should be a numpy array"
    assert fi.shape == (4,), f"Expected 4 feature importances, got {fi.shape}"


def test_search_hyperparameters_without_search_params(monkeypatch):
    """
    If RF_CONFIG does NOT contain 'search_params', search_hyperparameters
    should still instantiate a RandomForestClassifier using _get_base_rf_params
    and return a fitted model. We assert:
      1) The returned model is a RandomForestClassifier.
      2) The model is fitted (has feature_importances_).
    """
    # Arrange: tiny dataset
    X_train, y_train = make_classification(
        n_samples=20,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=1,
    )

    # Monkeypatch RF_CONFIG to exclude "search_params"
    no_search_config = {
        "n_estimators": 4,
        "random_state": 1,
        # No "search_params" key
    }
    monkeypatch.setattr(train_module, "RF_CONFIG", no_search_config)

    # Monkeypatch _get_base_rf_params accordingly
    def fake_get_base_rf_params():
        return {
            "n_estimators": no_search_config["n_estimators"],
            "random_state": no_search_config["random_state"],
        }

    monkeypatch.setattr(train_module, "_get_base_rf_params", fake_get_base_rf_params)

    # Act: call search_hyperparameters
    model = train_module.search_hyperparameters(X_train, y_train)

    # Assert: returned model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"

    # The model must be fitted: check for feature_importances_
    assert hasattr(
        model, "feature_importances_"
    ), "Model should have feature_importances_"
    fi = model.feature_importances_
    assert isinstance(fi, np.ndarray), "feature_importances_ should be a numpy array"
    assert fi.shape == (3,), f"Expected 3 feature importances, got {fi.shape}"

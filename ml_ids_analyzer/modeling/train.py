#!/usr/bin/env python3
"""
Training entry-point for ML-IDS-Analyzer.

Loads cleaned CICIDS2017 data, optionally performs hyperparameter search
on a RandomForestClassifier, evaluates, and saves model + scaler + predictions.
"""
import os
import logging

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ml_ids_analyzer.config import cfg
from ml_ids_analyzer.modeling.evaluate import evaluate_model, explain_model

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# --- Constants from config ---
DATA_FILE   = cfg["data"]["clean_file"]
OUTPUT_DIR  = cfg["paths"]["output_dir"]
PRED_CSV    = cfg["paths"]["predictions"]
MODEL_FILE  = cfg["paths"]["model_file"]
SCALER_FILE = cfg["paths"]["scaler_file"]
FEATURES    = cfg["features"]
LABEL       = cfg["label_column"]
RF_CONFIG   = cfg["model"]["random_forest"]


def search_hyperparameters(X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
    """
    Perform RandomizedSearchCV using the 'search_params' in RF_CONFIG.
    Returns the best estimator.
    """
    # Extract base RF parameters (excluding search space)
    base_params = {k: v for k, v in RF_CONFIG.items() if k != "search_params"}
    rf = RandomForestClassifier(**base_params)

    # Define search space
    param_dist = RF_CONFIG.get("search_params", {})

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=RF_CONFIG["random_state"],
    )
    search.fit(X_train, y_train)
    logging.info("Best hyperparameters found: %s", search.best_params_)
    return search.best_estimator_


def train_model() -> None:
    """Load data, train (with optional hyperparameter search), evaluate, and persist artifacts."""
    # --- Load & Clean Data ---
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df.dropna(subset=FEATURES + [LABEL], inplace=True)

    X = df[FEATURES]
    y = df[LABEL]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RF_CONFIG["random_state"]
    )

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- Model Training ---
    if "search_params" in RF_CONFIG:
        model = search_hyperparameters(X_train_scaled, y_train)
    else:
        model = RandomForestClassifier(**RF_CONFIG)
        model.fit(X_train_scaled, y_train)

    # --- Evaluation & Explainability ---
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred, model_name="Random Forest")
    explain_model(model, X_train_scaled)

    # --- Save Outputs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"Actual": y_test, "Predicted": y_pred}) \
      .to_csv(PRED_CSV, index=False)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logging.info("Saved predictions   → %s", PRED_CSV)
    logging.info("Saved model         → %s", MODEL_FILE)
    logging.info("Saved scaler        → %s", SCALER_FILE)


if __name__ == "__main__":
    train_model()
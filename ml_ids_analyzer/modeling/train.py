#!/usr/bin/env python3
"""
Training entry-point for ML-IDS-Analyzer.

Loads cleaned CICIDS2017 data, optionally performs hyperparameter search,
threshold tuning, evaluation, and saves model + scaler + predictions.
"""
import os
import logging
import argparse

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ml_ids_analyzer.config import cfg
from ml_ids_analyzer.modeling.evaluate import (
    evaluate_model,
    explain_model,
    tune_threshold,
)

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


def _get_base_rf_params() -> dict:
    """Return RF_CONFIG without 'search_params' key."""
    return {k: v for k, v in RF_CONFIG.items() if k != "search_params"}


def search_hyperparameters(
    X_train: np.ndarray, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Perform RandomizedSearchCV using the 'search_params' in RF_CONFIG.
    Returns the best estimator.
    """
    base_params = _get_base_rf_params()
    rf = RandomForestClassifier(**base_params)
    param_dist = RF_CONFIG.get("search_params", {})

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=RF_CONFIG.get("random_state", None),
    )
    search.fit(X_train, y_train)
    logging.info("Best hyperparameters found: %s", search.best_params_)
    return search.best_estimator_


def train_model(no_search: bool = False) -> None:
    """
    Load data, train model (with optional hyperparameter search),
    tune probability threshold, evaluate, and save artifacts.
    """
    # --- Load & Clean Data ---
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df.dropna(subset=FEATURES + [LABEL], inplace=True)

    X = df[FEATURES]
    y = df[LABEL]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=RF_CONFIG.get("random_state", None),
    )

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- Model Training ---
    if not no_search and "search_params" in RF_CONFIG:
        model = search_hyperparameters(X_train_scaled, y_train)
    else:
        base_params = _get_base_rf_params()
        model = RandomForestClassifier(**base_params)
        model.fit(X_train_scaled, y_train)

    # --- Threshold Tuning ---
    X_val, X_final, y_val, y_final = train_test_split(
        X_test_scaled,
        y_test,
        test_size=0.5,
        stratify=y_test,
        random_state=RF_CONFIG.get("random_state", None),
    )
    best_thr = tune_threshold(model, X_val, y_val)
    logging.info("Using probability threshold: %.3f", best_thr)
    
    probs_final  = model.predict_proba(X_final)[:, 1]
    y_pred_final = (probs_final >= best_thr).astype(int)

    # --- Evaluation & Explainability ---
    evaluate_model(y_final, y_pred_final, model_name="Random Forest (tuned)")
    explain_model(model, X_train_scaled)

    # --- Save Outputs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"Actual": y_final, "Predicted": y_pred_final}) \
      .to_csv(PRED_CSV, index=False)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logging.info("Saved predictions   → %s", PRED_CSV)
    logging.info("Saved model         → %s", MODEL_FILE)
    logging.info("Saved scaler        → %s", SCALER_FILE)


def main() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(
        description="Train the ML-IDS-Analyzer model with optional hyperparameter search."
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip hyperparameter tuning and train with default RF_CONFIG"
    )
    args = parser.parse_args()
    train_model(no_search=args.no_search)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Training pipeline for ml_ids_analyzer
"""
This script loads cleaned CICIDS2017 data, trains a Random Forest model
(with optional hyperparameter tuning), tunes the decision threshold,
evaluates performance, generates SHAP plots, and saves model artifacts.
"""

import os
import logging
import click
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ml_ids_analyzer.config import cfg as base_cfg
from ml_ids_analyzer.modeling.evaluate import evaluate_model, tune_threshold
from ml_ids_analyzer.explainability.explain import explain_model

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)

# Load config and allow override via environment variable
cfg = base_cfg.copy()
output_override = os.getenv("MLIDS_OUTPUT_DIR")
if output_override:
    output_path = Path(output_override)
    cfg["paths"]["output_dir"] = output_path
    cfg["paths"]["predictions"] = output_path / "predictions.csv"
    cfg["paths"]["model_file"] = output_path / "random_forest_model.joblib"
    cfg["paths"]["scaler_file"] = output_path / "scaler.joblib"

# Config constants
DATA_FILE = cfg["data"]["clean_file"]
OUTPUT_DIR = cfg["paths"]["output_dir"]
PRED_CSV = cfg["paths"]["predictions"]
MODEL_FILE = cfg["paths"]["model_file"]
SCALER_FILE = cfg["paths"]["scaler_file"]
FEATURES = cfg["features"]
LABEL = cfg["label_column"]
RF_CONFIG = cfg["model"]["random_forest"]

# Helper to get base RF params


def _get_base_rf_params() -> dict:
    """Return RF_CONFIG without 'search_params' key."""
    return {k: v for k, v in RF_CONFIG.items() if k != "search_params"}


# Hyperparameter search for Random Forest
def search_hyperparameters(
    X_train: np.ndarray, y_train: pd.Series
) -> RandomForestClassifier:
    """Perform RandomizedSearchCV using parameters in RF_CONFIG."""
    rf = RandomForestClassifier(**_get_base_rf_params())
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
    return search.best_estimator_  # type: ignore


# Main training function
def train_model(no_search: bool = False) -> None:
    """Main training pipeline."""
    # Load and clean data
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.dropna(subset=FEATURES + [LABEL], inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=FEATURES + [LABEL], inplace=True)

    X = df[FEATURES]
    y = df[LABEL]

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RF_CONFIG.get("random_state", None),
    )

    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning or default training
    if not no_search and "search_params" in RF_CONFIG:
        model = search_hyperparameters(X_train_scaled, y_train)
    else:
        model = RandomForestClassifier(**_get_base_rf_params())
        model.fit(X_train_scaled, y_train)

    # Split test set for final evaluation
    X_val, X_final, y_val, y_final = train_test_split(
        X_test_scaled,
        y_test,
        test_size=0.5,
        stratify=y_test,
        random_state=RF_CONFIG.get("random_state", None),
    )

    # Threshold tuning and final prediction
    best_thr = tune_threshold(model, X_val, y_val)
    logging.info("Using probability threshold: %.3f", best_thr)

    probs_final = model.predict_proba(X_final)[:, 1]
    y_pred_final = (probs_final >= best_thr).astype(int)

    # Model evaluation
    evaluate_model(y_final, y_pred_final, model_name="Random Forest (tuned)")

    # SHAP explainability
    try:
        explain_model(model, X_train_scaled)
    except Exception as e:
        logging.warning("SHAP explainability failed: %s", e)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"Actual": y_final, "Predicted": y_pred_final}).to_csv(
        PRED_CSV, index=False
    )
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logging.info("Saved predictions   → %s", PRED_CSV)
    logging.info("Saved model         → %s", MODEL_FILE)
    logging.info("Saved scaler        → %s", SCALER_FILE)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--no-search",
    is_flag=True,
    help="Skip hyperparameter tuning and train with default RF_CONFIG.",
)
def cli(no_search: bool) -> None:
    """Train the ML-IDS-Analyzer model."""
    train_model(no_search=no_search)


def main(args: list[str] | None = None) -> None:
    """CLI entry point used by both scripts and tests."""
    cli.main(args=args, standalone_mode=False)


if __name__ == "__main__":
    main()

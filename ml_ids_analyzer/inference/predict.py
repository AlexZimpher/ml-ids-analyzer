#!/usr/bin/env python3
"""
Inference entry-point for ML-IDS-Analyzer.

Loads saved model + scaler, reads new CSV of alerts,
applies the tuned probability threshold, and writes out
predicted labels + confidence.
"""
import os
import logging

import numpy as np
import pandas as pd
import joblib

from ml_ids_analyzer.config import cfg

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# --- Configured paths & settings ---
MODEL_PATH     = cfg["paths"]["model_file"]
SCALER_PATH    = cfg["paths"]["scaler_file"]
INPUT_FILE     = cfg["data"].get("new_input_file", "data/new_data.csv")
OUTPUT_FILE    = cfg["paths"]["new_predictions"]
FEATURES       = cfg["features"]
THRESHOLD      = cfg["model"].get("threshold", 0.5)  # default 0.5 if missing


def predict_new_data():
    # 1) Load pipeline artifacts
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Loaded model   → %s", MODEL_PATH)
    logging.info("Loaded scaler  → %s", SCALER_PATH)
    logging.info("Using cutoff   → %.3f", THRESHOLD)

    # 2) Read & clean input
    logging.info("Reading input → %s", INPUT_FILE)
    df_new = pd.read_csv(INPUT_FILE, skipinitialspace=True)
    df_new.columns = df_new.columns.str.strip()

    # 2a) Replace infinities and NaNs
    df_new.replace([np.inf, -np.inf], pd.NA, inplace=True)
    if df_new.isna().any().any():
        logging.info("Filling NaNs with 0")
        df_new.fillna(0, inplace=True)

    # 3) Extract & scale features
    X_new      = df_new[FEATURES]
    X_scaled   = scaler.transform(X_new)

    # 4) Compute probabilities & apply threshold
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
        df_new["Confidence (Attack)"] = probs
        df_new["Predicted Label"] = (probs >= THRESHOLD).astype(int)
    else:
        # fallback if no predict_proba
        df_new["Predicted Label"] = model.predict(X_scaled)

    # 5) Persist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_new.to_csv(OUTPUT_FILE, index=False)
    logging.info("Wrote predictions → %s", OUTPUT_FILE)


if __name__ == "__main__":
    predict_new_data()
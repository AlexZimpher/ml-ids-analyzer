#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib
import logging
from ml_ids_analyzer.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# === Paths & Settings from config ===
MODEL_PATH = cfg["paths"]["model_file"]
SCALER_PATH = cfg["paths"]["scaler_file"]
INPUT_FILE = cfg["data"].get("new_input_file", "data/new_data.csv")
OUTPUT_FILE = cfg["paths"]["new_predictions"]
FEATURES = cfg["features"]
LABEL_COLUMN = cfg["label_column"]


def predict_new_data():
    # 1) Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2) Read & clean input
    logging.info(f"Reading input file → {INPUT_FILE}")
    df_new = pd.read_csv(INPUT_FILE, skipinitialspace=True)
    df_new.columns = df_new.columns.str.strip()

    # 2a) Replace infinities with NaN, then fill or drop
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_new.isna().any().any():
        logging.info("Found NaN or infinite values in input; filling with 0")
        df_new.fillna(0, inplace=True)

    # 3) Scale features
    X_new = df_new[FEATURES]
    X_scaled = scaler.transform(X_new)

    # 4) Predict
    df_new["Predicted Label"] = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        df_new["Confidence (Attack)"] = model.predict_proba(X_scaled)[:, 1]

    # 5) Write out
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_new.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Predictions saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    predict_new_data()


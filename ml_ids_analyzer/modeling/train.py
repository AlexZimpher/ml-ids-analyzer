#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from evaluate import evaluate_model, explain_model
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_FILE = cfg["data"]["clean_file"]
OUTPUT_DIR = cfg["paths"]["output_dir"]
PRED_CSV = cfg["paths"]["predictions"]
MODEL_FILE = cfg["paths"]["model_file"]
SCALER_FILE = cfg["paths"]["scaler_file"]
FEATURES = cfg["features"]
RF_PARAMS = cfg["model"]["random_forest"]


def train_model():
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df.dropna(subset=FEATURES + [cfg["label_column"]], inplace=True)

    X = df[FEATURES]
    y = df[cfg["label_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RF_PARAMS["random_state"]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred, model_name="Random Forest")
    explain_model(model, X_train)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(PRED_CSV, index=False)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logging.info(f"Saved predictions → {PRED_CSV}")
    logging.info(f"Saved model → {MODEL_FILE}")
    logging.info(f"Saved scaler → {SCALER_FILE}")


if __name__ == "__main__":
    train_model()

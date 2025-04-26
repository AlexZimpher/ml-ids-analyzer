#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import logging
from src.config import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

MODEL_PATH = cfg['paths']['model_file']
SCALER_PATH = cfg['paths']['scaler_file']
INPUT_FILE = cfg['data'].get('new_input_file', 'data/new_data.csv')
OUTPUT_FILE = cfg['paths']['new_predictions']
FEATURES = cfg['features']
LABEL_COL = cfg['label_column']


def predict_new_data():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_new = pd.read_csv(INPUT_FILE, skipinitialspace=True)
    df_new.columns = df_new.columns.str.strip()
    missing = set(FEATURES) - set(df_new.columns)
    if missing:
        raise KeyError(f"Missing features in input: {missing}")
    X_new = df_new[FEATURES]
    X_scaled = scaler.transform(X_new)
    df_new['Predicted Label'] = model.predict(X_scaled)
    if hasattr(model, 'predict_proba'):
        df_new['Confidence (Attack)'] = model.predict_proba(X_scaled)[:, 1]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_new.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    predict_new_data()

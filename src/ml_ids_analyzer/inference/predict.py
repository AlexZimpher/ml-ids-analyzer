#!/usr/bin/env python3
"""
Inference entry-point for ML-IDS-Analyzer.

Loads saved model + scaler, reads new CSV of alerts,
applies the tuned probability threshold (configurable),
and writes out predicted labels + confidence.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ml_ids_analyzer.config import cfg


# configure logging once at import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


def predict_file(
    model_path: Path,
    scaler_path: Path,
    features: list[str],
    threshold: float,
    input_csv: Path,
    output_csv: Path,
) -> None:
    """Run inference: read, scale, predict, and save results."""
    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("Loaded model   → %s", model_path)
    logging.info("Loaded scaler  → %s", scaler_path)
    logging.info("Using cutoff   → %.3f", threshold)

    # Read & clean input
    logging.info("Reading input  → %s", input_csv)
    df = pd.read_csv(input_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Replace infinities and NaNs
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    if df.isna().any().any():
        logging.info("Filling NaNs with 0")
        df.fillna(0, inplace=True)

    # Extract & scale features
    X = df[features]
    X_scaled = scaler.transform(X)

    # Predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
        df["Confidence (Attack)"] = probs
        df["Predicted Label"] = (probs >= threshold).astype(int)
    else:
        df["Predicted Label"] = model.predict(X_scaled)

    # Persist results
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info("Wrote predictions → %s", output_csv)


def main():
    parser = ArgumentParser(description="ML-IDS-Analyzer inference tool")
    parser.add_argument(
        "-i", "--input-csv",
        type=Path,
        default=Path(cfg["data"].get("new_input_file", "data/new_data.csv")),
        help="Path to input CSV (default from config)"
    )
    parser.add_argument(
        "-o", "--output-csv",
        type=Path,
        default=Path(cfg["paths"]["new_predictions"]),
        help="Where to write predictions (default from config)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=cfg["model"].get("threshold", 0.5),
        help="Probability cutoff for positive class"
    )
    args = parser.parse_args()

    predict_file(
        model_path=Path(cfg["paths"]["model_file"]),
        scaler_path=Path(cfg["paths"]["scaler_file"]),
        features=cfg["features"],
        threshold=args.threshold,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
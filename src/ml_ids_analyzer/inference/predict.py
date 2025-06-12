"""
Inference and prediction utilities for ml_ids_analyzer.
"""

import joblib
import click
import logging
import numpy as np
import os
import pandas as pd

from typing import Optional, Tuple

from ml_ids_analyzer.config import cfg

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
_logger = logging.getLogger(__name__)

# Config fallback values
model_cfg = cfg.get("model", {})
infer_cfg = cfg.get("inference", {})

# Default paths and threshold
DEFAULT_INPUT: str = infer_cfg.get(
    "input_csv",
    cfg.get("data", {}).get("clean_file", "data/cicids2017_clean.csv"),
)
DEFAULT_MODEL: str = cfg.get("paths", {}).get(
    "model_file", "outputs/random_forest_model.joblib"
)
DEFAULT_SCALER: Optional[str] = cfg.get("paths", {}).get(
    "scaler_file", "outputs/scaler.joblib"
)
DEFAULT_OUTPUT: str = infer_cfg.get("output_csv", "outputs/predictions.csv")
THRESHOLD: float = infer_cfg.get("threshold", 0.5)


def load_model_and_scaler() -> Tuple:
    """Load model and scaler from configured paths."""
    model = joblib.load(DEFAULT_MODEL)
    scaler = (
        joblib.load(DEFAULT_SCALER)
        if DEFAULT_SCALER and os.path.isfile(DEFAULT_SCALER)
        else None
    )
    return model, scaler


def predict_alerts(
    model, scaler, df: pd.DataFrame, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """
    Run predictions on a DataFrame using provided model and scaler.
    Adds 'predicted_label' and 'prediction_prob' columns to the result.
    Ensures feature alignment and type safety for API/CLI use.
    """
    features = cfg["features"]
    # Ensure all required features are present
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    # Align columns and convert to float
    X = df[features].copy().astype(float)
    X_scaled = scaler.transform(X) if scaler is not None else X

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["prediction_prob"] = probs
    result["predicted_label"] = preds
    return result


# CLI for running predictions from the command line
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-file",
    default=DEFAULT_INPUT,
    show_default=True,
    help="Path to CSV of features to score",
)
@click.option(
    "--model-file",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Path to saved model (.joblib)",
)
@click.option(
    "--scaler-file",
    default=DEFAULT_SCALER,
    show_default=True,
    help="Path to saved scaler (.joblib), if used",
)
@click.option(
    "--output-file",
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Where to write predictions CSV",
)
@click.option(
    "--threshold",
    default=THRESHOLD,
    show_default=True,
    help="Threshold for classifying as attack",
)
def main(
    input_file: str,
    model_file: str,
    scaler_file: Optional[str],
    output_file: str,
    threshold: float,
) -> None:
    """
    CLI entry point: load data, run predictions, and save results to CSV.
    """
    # Validate input files
    for path, desc in [(input_file, "input CSV"), (model_file, "model file")]:
        if not os.path.isfile(path):
            _logger.error("Missing %s: %s", desc, path)
            raise SystemExit(1)
    if scaler_file and not os.path.isfile(scaler_file):
        _logger.error("Missing scaler file: %s", scaler_file)
        raise SystemExit(1)

    # Load model and scaler
    _logger.info("Loading model from %s", model_file)
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if scaler_file else None

    # Load input data
    _logger.info("Reading input data from %s", input_file)
    df = pd.read_csv(input_file)

    # 1) Replace ±∞ with NaN, then drop any rows containing NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)

    # 2) Subset to exactly the feature columns (in the exact order)
    feature_cols = cfg["features"]
    X_df = df[feature_cols].copy()

    # 3) Convert all columns to float64 (raise if conversion fails)
    X_df = X_df.astype(np.float64, errors="raise")

    # 4) Extract numpy array and scale if applicable
    X = X_df.values
    if scaler is not None:
        X = scaler.transform(X)

    # 5) Run predictions
    _logger.info("Running predictions")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # 6) Save results (feature columns + prob_attack + pred_attack)
    out = X_df.copy()
    out["prob_attack"] = probs
    out["pred_attack"] = preds
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out.to_csv(output_file, index=False)
    _logger.info("Wrote predictions to %s", output_file)


if __name__ == "__main__":
    main()

# ml_ids_analyzer/inference/predict.py

import os
import logging
from typing import Optional

import click
import pandas as pd
import joblib

from ml_ids_analyzer.config import cfg

# module‐level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
_logger = logging.getLogger(__name__)

# Safely pull config with defaults
model_cfg = cfg.get("model", {})
infer_cfg = cfg.get("inference", {})

DEFAULT_INPUT: str = infer_cfg.get(
    "input_csv", cfg.get("data", {}).get("clean_file", "data/cicids2017_clean.csv")
)
DEFAULT_MODEL: str = cfg.get("paths", {}).get("model_file", "outputs/random_forest_model.joblib")
DEFAULT_SCALER: Optional[str] = cfg.get("paths", {}).get("scaler_file", "outputs/scaler.joblib")
DEFAULT_OUTPUT: str = infer_cfg.get("output_csv", "outputs/predictions.csv")
THRESHOLD: float = infer_cfg.get("threshold", 0.5)


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
    help="Probability threshold for classifying an attack",
)
def main(
    input_file: str,
    model_file: str,
    scaler_file: Optional[str],
    output_file: str,
    threshold: float,
) -> None:
    """
    Load model (and optional scaler), read INPUT_FILE, run predictions,
    and write results to OUTPUT_FILE.
    """
    # 1. Validate files
    for path, desc in [
        (input_file, "input CSV"),
        (model_file, "model file"),
    ]:
        if not os.path.isfile(path):
            _logger.error("Missing %s: %s", desc, path)
            raise SystemExit(1)
    if scaler_file and not os.path.isfile(scaler_file):
        _logger.error("Missing scaler file: %s", scaler_file)
        raise SystemExit(1)

    # 2. Load artifacts
    _logger.info("Loading model from %s", model_file)
    model = joblib.load(model_file)
    scaler = None
    if scaler_file:
        _logger.info("Loading scaler from %s", scaler_file)
        scaler = joblib.load(scaler_file)

    # 3. Read input
    _logger.info("Reading input data from %s", input_file)
    df = pd.read_csv(input_file)
    features = df.drop(columns=[cfg.get("label_column", "Label")], errors="ignore")

    # 4. Apply scaler
    X = features.values
    if scaler is not None:
        X = scaler.transform(X)

    # 5. Predict probabilities and labels
    _logger.info("Running predictions")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # 6. Assemble and write output
    out = features.copy()
    out["prob_attack"] = probs
    out["pred_attack"] = preds
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out.to_csv(output_file, index=False)
    _logger.info("Wrote predictions to %s", output_file)


if __name__ == "__main__":
    main()

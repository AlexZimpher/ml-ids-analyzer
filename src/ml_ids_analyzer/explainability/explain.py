import logging
import matplotlib.pyplot as plt
import shap
import numpy as np
from pathlib import Path

from ml_ids_analyzer.config import cfg

OUTPUT_DIR = Path(cfg["paths"]["output_dir"])  # Ensure Path object

def explain_model(model, X_train):
    """Generate SHAP summary plot using a sample of the training set."""
    try:
        sample_size = min(100, len(X_train))
        X_sample = X_train[:sample_size]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values[1], X_sample, show=False)
        plt.tight_layout()

        output_dir = Path(cfg["paths"]["output_dir"])  # Re-fetch at runtime
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shap_summary.png"

        plt.savefig(output_path)
        logging.info("Saved SHAP summary plot to %s", output_path)
    except Exception as e:
        logging.warning("SHAP explainability failed: %s", e)
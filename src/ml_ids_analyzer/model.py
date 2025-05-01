# ml_ids_analyzer/model.py
"""
Console-script entry point and API for ML-IDS-Analyzer training.
"""

import logging
from ml_ids_analyzer.modeling.train import train_model

__all__ = ["train_model", "main"]

_logger = logging.getLogger(__name__)

def main() -> None:
    """Entry point for the `mlids-train` console script."""
    try:
        train_model()
    except Exception:
        _logger.exception("Training failed")
        raise

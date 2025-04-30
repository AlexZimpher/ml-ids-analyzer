#!/usr/bin/env python3
"""
Evaluation and explainability utilities for ML-IDS-Analyzer.

Includes:
- evaluate_model: prints classification report, ROC AUC, and saves confusion matrix plot.
- explain_model: SHAP summary plot of training data.
- tune_threshold: computes Precision–Recall curve, saves it, and returns best threshold.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend to avoid blocking

# Ensure outputs directory exists
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(y_true, y_pred, model_name="Model") -> None:
    """Print classification report, ROC AUC, and save confusion matrix."""
    logging.info("=== Evaluation Report: %s ===", model_name)
    print(classification_report(y_true, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path)
    logging.info("Saved confusion matrix to %s", out_path)
    plt.close()


def explain_model(model, X_train) -> None:
    """Generate and save a SHAP summary plot if SHAP is installed."""
    try:
        import shap
    except ImportError:
        logging.warning("SHAP not installed; skipping explainability.")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Create summary plot without blocking
    shap.summary_plot(shap_values, X_train, show=False)
    out_path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(out_path)
    logging.info("Saved SHAP summary plot to %s", out_path)
    plt.close()


def tune_threshold(model, X_val, y_val) -> float:
    """
    Compute Precision–Recall curve, save it,
    and return threshold maximizing F1 score.
    """
    prob_pos = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, prob_pos)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores.argmax()
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.scatter(
        recall[best_idx],
        precision[best_idx],
        label=f"Best F1={best_f1:.2f} @thr={best_thr:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–
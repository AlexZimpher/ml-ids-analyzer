#!/usr/bin/env python3
"""
Evaluation and explainability utilities for ML-IDS-Analyzer.
Includes:
- evaluate_model: prints classification report, ROC AUC, and confusion matrix plot.
- explain_model: SHAP summary plot of training data.
- tune_threshold: computes Precision–Recall curve, plots it, and returns best threshold.
"""
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


def evaluate_model(y_true, y_pred, model_name="Model") -> None:
    """Print classification report, ROC AUC, and show confusion matrix."""
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
    plt.show()


def explain_model(model, X_train) -> None:
    """Generate a SHAP summary plot if SHAP is installed."""
    try:
        import shap
    except ImportError:
        logging.warning("SHAP not installed; skipping explainability.")
        return
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=True)


def tune_threshold(model, X_val, y_val) -> float:
    """
    Compute Precision–Recall curve, plot it, and return threshold maximizing F1 score.
    """
    # Positive-class probabilities
    prob_pos = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, prob_pos)

    # Compute F1 for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores.argmax()
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Plot Precision-Recall curve with best point highlighted
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.scatter(
        recall[best_idx], precision[best_idx],
        label=f"Best F1={best_f1:.2f} @thr={best_thr:.2f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info("Chosen threshold: %.3f with best F1: %.3f", best_thr, best_f1)
    return best_thr
# File: tests/evaluate/test_evaluate_model.py

import numpy as np
import pytest

# Import evaluate_model from its actual location
from ml_ids_analyzer.modeling.evaluate import evaluate_model


def test_evaluate_model_saves_confusion_matrix_and_prints_report(
    tmp_path, capsys
):
    """
    1) Use y_true and y_pred that are identical (perfect predictions):
       - classification_report should report accuracy = 1.00
       - ROC AUC Score should be 1.0000
    2) Provide tmp_path as output_dir so that a file
       "TestModel_confusion_matrix.png" is saved there.
    3) Capture stdout and verify the printed lines.
    """
    # Arrange: simple binary labels with perfect prediction
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    model_name = "TestModel"

    # Ensure tmp_path is empty
    assert not any(tmp_path.iterdir())

    # Act
    evaluate_model(y_true, y_pred, model_name=model_name, output_dir=tmp_path)

    # Capture stdout
    captured = capsys.readouterr().out.lower()

    # Assert classification report printed “accuracy”
    assert "accuracy" in captured, "Classification report missing 'accuracy'"

    # Assert ROC AUC Score line appears exactly
    assert (
        "roc auc score: 1.0000" in captured
    ), f"Expected ROC AUC = 1.0000, got:\n{captured}"

    # Assert confusion-matrix file was created under tmp_path
    expected_filename = f"{model_name}_confusion_matrix.png"
    output_file = tmp_path / expected_filename
    assert (
        output_file.exists() and output_file.is_file()
    ), f"Expected file {expected_filename} not found in {tmp_path}"
    assert output_file.stat().st_size > 0, "Saved PNG file is empty"


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (np.array([0, 0, 0]), np.array([0, 0, 0])),  # all zeros
        (np.array([1, 1, 1]), np.array([1, 1, 1])),  # all ones
    ],
)
def test_evaluate_model_edge_cases_no_positive_class(
    tmp_path, capsys, y_true, y_pred
):
    """
    If y_true contains only one class, roc_auc_score yields nan and
    evaluate_model prints “ROC AUC Score: nan” without raising.
    We assert that:
      1) No exception is raised.
      2) The printed ROC AUC line shows “nan”.
      3) The confusion-matrix PNG is still created.
    """
    model_name = "EdgeCase"

    # Act (should not raise)
    evaluate_model(y_true, y_pred, model_name=model_name, output_dir=tmp_path)

    # Capture stdout
    captured = capsys.readouterr().out.lower()
    # Assert “roc auc score: nan” appears
    assert (
        "roc auc score: nan" in captured
    ), f"Expected ROC AUC = nan, got:\n{captured}"

    # Assert confusion-matrix file was created
    expected_filename = f"{model_name}_confusion_matrix.png"
    output_file = tmp_path / expected_filename
    assert (
        output_file.exists() and output_file.is_file()
    ), f"Expected file {expected_filename} not found"
    assert output_file.stat().st_size > 0, "Saved PNG file is empty"

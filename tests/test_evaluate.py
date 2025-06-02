import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from ml_ids_analyzer.modeling.evaluate import evaluate_model
import tempfile
from pathlib import Path
import shutil

def test_evaluate_model_creates_outputs():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10).fit(X, y)
    y_pred = model.predict(X)

    temp_dir = Path(tempfile.mkdtemp())
    try:
        evaluate_model(y, y_pred, model_name="test_model", output_dir=temp_dir)
        assert (temp_dir / "test_model_confusion_matrix.png").exists()
    finally:
        shutil.rmtree(temp_dir)


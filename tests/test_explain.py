import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from ml_ids_analyzer.explainability.explain import explain_model
from ml_ids_analyzer.config import cfg

def test_explain_model_creates_shap_plot():
    # Setup: create dummy data and model
    X = pd.DataFrame(np.random.rand(10, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=10)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Temp override output dir
    temp_dir = Path(tempfile.mkdtemp())
    original_output_dir = cfg["paths"]["output_dir"]
    cfg["paths"]["output_dir"] = temp_dir

    try:
        explain_model(model, X)
        assert (temp_dir / "shap_summary.png").exists()
    finally:
        shutil.rmtree(temp_dir)
        cfg["paths"]["output_dir"] = original_output_dir

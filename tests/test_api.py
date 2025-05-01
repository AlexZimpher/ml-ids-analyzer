# tests/test_api.py

import pytest
import pandas as pd
import joblib
from fastapi.testclient import TestClient

from ml_ids_analyzer.api.app import app


@pytest.fixture
def api_env(tmp_path):
    # Prepare dummy data records
    df = pd.DataFrame({"feature1": [0, 1], "feature2": [1, 0]})
    data_records = df.to_dict(orient="records")

    # Train and dump dummy scaler and model
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    scaler = StandardScaler().fit(df.values)
    X_scaled = scaler.transform(df.values)
    model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
    model.fit(X_scaled, [0, 1])

    model_file = tmp_path / "model.joblib"
    scaler_file = tmp_path / "scaler.joblib"
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    return {
        "data": data_records,
        "model_file": str(model_file),
        "scaler_file": str(scaler_file),
    }


def test_predict_endpoint(api_env):
    client = TestClient(app)
    payload = {
        "data": api_env["data"],
        "model_file": api_env["model_file"],
        "scaler_file": api_env["scaler_file"],
        "threshold": 0.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

    results = response.json()["results"]
    # Check each record has probability and prediction
    assert all("prob_attack" in rec and "pred_attack" in rec for rec in results)
    # Predictions mirror training labels [0, 1]
    assert [rec["pred_attack"] for rec in results] == [0, 1]

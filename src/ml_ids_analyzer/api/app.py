from typing import Any, Dict, List, Optional
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_ids_analyzer.config import cfg

app = FastAPI(
    title="ML-IDS-Analyzer API",
    description="Predict intrusion alerts via REST endpoints",
    version=cfg.get("version", "0.1.0"),
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "ML-IDS-Analyzer API is up. Try POST /predict or GET /health"}


@app.get("/health", tags=["health"])
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of feature dicts")
    model_file: Optional[str] = Field(
        cfg.get("paths", {}).get("model_file", "outputs/model.joblib"),
        description="Path to trained model (.joblib)",
    )
    scaler_file: Optional[str] = Field(
        cfg.get("paths", {}).get("scaler_file"),
        description="Path to scaler artifact (.joblib)",
    )
    threshold: Optional[float] = Field(
        cfg.get("inference", {}).get("threshold", 0.5),
        description="Probability threshold to flag an attack",
    )


class PredictResponse(BaseModel):
    results: List[Dict[str, Any]]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Validate model file
    if not os.path.isfile(request.model_file):
        raise HTTPException(
            status_code=400, detail=f"Model not found: {request.model_file}"
        )

    # Validate scaler file (if provided)
    if request.scaler_file and not os.path.isfile(request.scaler_file):
        raise HTTPException(
            status_code=400, detail=f"Scaler not found: {request.scaler_file}"
        )

    # Load model
    try:
        model = joblib.load(request.model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Load scaler
    scaler = None
    if request.scaler_file:
        try:
            scaler = joblib.load(request.scaler_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load scaler: {e}")

    # Build DataFrame
    try:
        df = pd.DataFrame(request.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    X = df.values
    if scaler:
        X = scaler.transform(X)

    # Predictions
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    preds = (probs >= request.threshold).astype(int)

    # Assemble results
    results = []
    for idx, row in df.iterrows():
        rec = row.to_dict()
        rec["prob_attack"] = float(probs[idx])
        rec["pred_attack"] = int(preds[idx])
        results.append(rec)

    return PredictResponse(results=results)

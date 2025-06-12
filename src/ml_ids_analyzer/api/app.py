"""
FastAPI app for ML-IDS-Analyzer REST API.
"""

from typing import Optional
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from ml_ids_analyzer.config import cfg

# Create FastAPI app instance
app = FastAPI(
    title="ML-IDS-Analyzer API",
    description=(
        "Predict intrusion alerts via REST endpoints and interactive dashboard"
    ),
    version=cfg.get("version", "0.1.0"),
)


# Root endpoint for health/info
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "ML-IDS-Analyzer API is up. Try POST /predict or GET /health"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


# Prediction endpoint for CSV upload
@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with feature columns"),
    model_file: Optional[str] = Form(
        cfg.get("paths", {}).get("model_file", "outputs/model.joblib")
    ),
    scaler_file: Optional[str] = Form(cfg.get("paths", {}).get("scaler_file")),
    threshold: Optional[float] = Form(
        cfg.get("inference", {}).get("threshold", 0.5)
    ),
):
    """
    Upload a CSV file to get predictions. Returns summary statistics and a preview of results.
    """
    # Validate model file
    if not model_file or not os.path.isfile(model_file):
        raise HTTPException(
            status_code=400, detail=f"Model not found: {model_file}"
        )

    if scaler_file and not os.path.isfile(scaler_file):
        raise HTTPException(
            status_code=400, detail=f"Scaler not found: {scaler_file}"
        )

    # Load model and scaler
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if scaler_file else None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model/scaler: {e}"
        )

    # Read CSV into DataFrame
    try:
        contents = await file.read()
        import io

        df = pd.read_csv(io.BytesIO(contents))
        feature_cols = cfg["features"]
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing required features: {missing}"
            )
        # Limit to first 1000 rows for performance (adjust as needed)
        df = df.head(1000)
        X_df = df[feature_cols].copy().astype(float)
        # Clean data: replace inf/-inf with NaN, drop rows with NaN
        X_df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        X_df.dropna(axis=0, how="any", inplace=True)
        X_df = X_df.reset_index(drop=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV input: {e}")

    X = X_df.values
    if scaler:
        X = scaler.transform(X)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    preds = (probs >= threshold).astype(int)

    results = []
    for idx, row in X_df.iterrows():
        rec = row.to_dict()
        rec["prob_attack"] = float(probs[idx])
        rec["pred_attack"] = int(preds[idx])
        results.append(rec)

    # Summary stats
    total = len(results)
    attacks = sum(r["pred_attack"] for r in results)
    preview = results[:10]  # Show first 10 predictions as a preview

    return {
        "summary": {
            "total_records": total,
            "attacks_detected": attacks,
            "threshold": threshold,
        },
        "preview": preview,
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard_redirect():
    """
    Redirects to the Streamlit dashboard if running, or provides instructions.
    """
    return {
        "message": "To use the dashboard, run: streamlit run src/ml_ids_analyzer/api/dashboard.py"
    }

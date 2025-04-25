import pandas as pd
import joblib
import os

# === Config ===
MODEL_PATH = "outputs/random_forest_model.joblib"
SCALER_PATH = "outputs/scaler.joblib"
INPUT_FILE = "data/new_data.csv"  # Replace with your test input file
OUTPUT_FILE = "outputs/new_predictions.csv"

FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Flow IAT Mean',
    'Flow Bytes/s', 'Flow Packets/s'
]

def predict_new_data():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_new = pd.read_csv(INPUT_FILE)
    X_new = df_new[FEATURES]
    X_scaled = scaler.transform(X_new)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    df_new["Predicted Label"] = y_pred
    df_new["Confidence (Attack)"] = y_prob
    os.makedirs("outputs", exist_ok=True)
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    predict_new_data()

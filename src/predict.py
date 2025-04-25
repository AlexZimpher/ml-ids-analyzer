import os
import pandas as pd
import joblib

# === Paths ===
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH  = os.path.join(BASE_DIR, 'outputs', 'random_forest_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'outputs', 'scaler.joblib')
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'new_data.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'outputs', 'new_predictions.csv')

FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Flow IAT Mean',
    'Flow Bytes/s', 'Flow Packets/s'
]


def predict_new_data():
    # Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Read & clean input
    df_new = pd.read_csv(INPUT_FILE, skipinitialspace=True)
    df_new.columns = df_new.columns.str.strip()
    X_new = df_new[FEATURES]
    X_scaled = scaler.transform(X_new)

    # Predict
    df_new['Predicted Label'] = model.predict(X_scaled)
    df_new['Confidence (Attack)'] = model.predict_proba(X_scaled)[:, 1]

    # Write out
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    predict_new_data()

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.evaluate import evaluate_model

# === Paths ===
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE   = os.path.join(BASE_DIR, 'data', 'cicids2017_clean.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
PRED_CSV    = os.path.join(OUTPUT_DIR, 'predictions.csv')
MODEL_FILE  = os.path.join(OUTPUT_DIR, 'random_forest_model.joblib')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.joblib')

# === Features ===
FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Flow IAT Mean',
    'Flow Bytes/s', 'Flow Packets/s'
]

def train_model():
    # 1) Load
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # 2) Clean any non-finite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES + ['Label'], inplace=True)  # drop rows missing any feature or label

    # 3) Split features & target
    X = df[FEATURES]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 4) Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 5) Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) Evaluate
    evaluate_model(y_test, y_pred, model_name='Random Forest')

    # 7) Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) \
      .to_csv(PRED_CSV, index=False)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved predictions → {PRED_CSV}")
    print(f"Saved model → {MODEL_FILE}")
    print(f"Saved scaler → {SCALER_FILE}")

if __name__ == '__main__':
    train_model()

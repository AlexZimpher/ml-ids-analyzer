import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Configuration ===
DATA_FILE = "data/cicids2017_clean.csv"
OUTPUT_FILE = "outputs/predictions.csv"
FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Flow IAT Mean',
    'Flow Bytes/s', 'Flow Packets/s'
]

def train_model():
    # Load and split data
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate using external module
    evaluate_model(y_test, y_pred, model_name="Random Forest")

    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

# Save trained model and scaler
import joblib
joblib.dump(model, "outputs/random_forest_model.joblib")
joblib.dump(scaler, "outputs/scaler.joblib")
print("Model and scaler saved to outputs/")


if __name__ == "__main__":
    train_model()

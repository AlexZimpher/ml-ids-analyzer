# ml_ids_analyzer/dashboard.py

import streamlit as st
import pandas as pd
from ml_ids_analyzer.inference.predict import load_model_and_scaler, predict_alerts

def main():
    st.set_page_config(page_title="ML-IDS Analyzer Dashboard", layout="wide")
    st.title("📊 ML-IDS Analyzer Dashboard")

    st.sidebar.header("Upload Input Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", df.head())

        model, scaler = load_model_and_scaler()
        results = predict_alerts(model, scaler, df)

        st.write("### Predictions", results.head())
        st.download_button("Download Predictions CSV", results.to_csv(index=False), file_name="predictions.csv")

    else:
        st.info("Upload a CSV file in the sidebar to begin.")

# Required for CLI to find `main`
if __name__ == "__main__":
    main()


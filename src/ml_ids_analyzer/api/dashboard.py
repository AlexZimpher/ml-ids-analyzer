import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import requests
import io

API_URL = "http://localhost:8000/predict/csv"  # Update if deployed elsewhere


def main():
    st.set_page_config(page_title="ML-IDS Analyzer Dashboard", layout="wide")
    st.title("📊 ML-IDS Analyzer Dashboard")

    # Project description and instructions
    st.markdown(
        """
        <style>
        .big-font {font-size:18px !important;}
        .attack-row {background-color: #ffe6e6 !important;}
        .good-row {background-color: #e6ffe6 !important;}
        .summary-box {border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: #f9f9f9;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="big-font">
        Welcome to the <b>ML-IDS Analyzer</b>! This dashboard demonstrates a Machine Learning-based Intrusion Detection System.<br><br>
        <b>Instructions:</b> Upload a CSV file with network traffic features in the sidebar. The model will predict potential attacks and display results below.<br>
        <ul>
        <li>Download a sample CSV from the <a href="https://github.com/your-repo" target="_blank">project repository</a>.</li>
        <li>See <a href="https://github.com/your-repo/docs" target="_blank">documentation</a> for feature details.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Upload Input Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About this project:**")
    st.sidebar.info(
        "ML-IDS Analyzer is a showcase of ML-powered intrusion detection. Built for recruiters and engineers.\n\n"
        "[GitHub Repo](https://github.com/your-repo) | "
        "[Docs](https://github.com/your-repo/docs)"
    )

    if uploaded_file:
        st.write("### Uploaded Data", pd.read_csv(uploaded_file).head())
        uploaded_file.seek(0)
        with st.spinner("Running predictions on uploaded data..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                summary = result["summary"]
                preview = pd.DataFrame(result["preview"])
                st.markdown(
                    f"""
                <div class="summary-box" style="
                    border: 2px solid #222;
                    border-radius: 10px;
                    padding: 18px 24px;
                    background: linear-gradient(90deg, #f9f9f9 80%, #e0e7ff 100%);
                    margin-bottom: 18px;
                    font-size: 18px;
                    color: #222;
                    box-shadow: 0 2px 8px rgba(67,97,238,0.07);
                ">
                <b>Total records:</b> <span style="color:#4361ee;">{summary['total_records']}</span><br>
                <b>Attacks detected:</b> <span style='color:#e63946; font-weight:bold;'>{summary['attacks_detected']}</span><br>
                <b>Threshold:</b> <span style="color:#222; background:#e0e7ff; border-radius:4px; padding:2px 8px;">{summary['threshold']}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.write("### Prediction Preview (first 10 rows)")

                def highlight_attacks(row):
                    if row.get("pred_attack", 0) == 1:
                        return [
                            "background-color: #fff0f0; color: #222; font-weight: bold;"
                        ] * len(row)
                    else:
                        return ["background-color: #f8fafd; color: #222;"] * len(row)

                if not preview.empty:
                    st.dataframe(
                        preview.style.apply(highlight_attacks, axis=1),
                        use_container_width=True,
                    )
                st.info("Full predictions CSV available for download below.")
                # Download full predictions by re-running locally
                uploaded_file.seek(0)
                df_full = pd.read_csv(uploaded_file)
                from ml_ids_analyzer.inference.predict import (
                    load_model_and_scaler,
                    predict_alerts,
                )

                model, scaler = load_model_and_scaler()
                # Clean data: replace inf/-inf with NaN, drop rows with NaN
                df_full.replace(
                    [float("inf"), float("-inf")], float("nan"), inplace=True
                )
                df_full.dropna(axis=0, how="any", inplace=True)
                df_full = df_full.reset_index(drop=True)
                results_full = predict_alerts(model, scaler, df_full)
                # Defensive: check if 'pred_attack' exists before plotting
                if "pred_attack" in results_full.columns:
                    st.write("### Attack Distribution")
                    attack_counts = (
                        results_full["pred_attack"]
                        .value_counts()
                        .rename({0: "Benign", 1: "Attack"})
                    )
                    st.bar_chart(attack_counts)
                else:
                    st.warning(
                        "No predictions available for attack distribution plot (no 'pred_attack' column). Please check your input data."
                    )
                st.write("### Probability Histogram")
                import matplotlib.pyplot as plt

                if "prob_attack" in results_full.columns:
                    fig, ax = plt.subplots()
                    ax.hist(
                        results_full["prob_attack"],
                        bins=20,
                        color="#4361ee",
                        edgecolor="white",
                    )
                    ax.set_title("Probability of Attack (Histogram)")
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                else:
                    st.warning(
                        "No probability scores available for histogram plot (no 'prob_attack' column). Please check your input data."
                    )
            else:
                st.error(f"Prediction failed: {response.text}")
    else:
        st.info("Upload a CSV file in the sidebar to begin.")


# Required for CLI to find `main`
if __name__ == "__main__":
    main()

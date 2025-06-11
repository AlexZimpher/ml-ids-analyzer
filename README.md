# ML-IDS-Analyzer

[![Build Status](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Built with Poetry](https://img.shields.io/badge/Built%20with-Poetry-612C63.svg?logo=python&logoColor=white)
[![codecov](https://codecov.io/gh/AlexZimpher/ml-ids-analyzer/graph/badge.svg?token=DMYGFS3OEO)](https://codecov.io/gh/AlexZimpher/ml-ids-analyzer)

**ML-IDS-Analyzer** is a machine learning pipeline and REST API for analyzing intrusion detection system (IDS) alerts. It uses the CICIDS2017 dataset to build a classifier that distinguishes malicious from benign traffic. The system includes data preprocessing, model training, threshold tuning, feature importance analysis, and a real-time prediction API.

---

## 🚀 Features

- 📦 Modular ML pipeline (preprocessing, training, prediction)
- 🧠 Random Forest with threshold optimization
- 📊 SHAP-based explainability and evaluation plots
- ⚙️ FastAPI prediction server -TODO
- 🐳 Docker-only workflow - TODO
- 🛠️ Suricata integration - TODO

---

## 📦 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/AlexZimpher/ml-ids-analyzer.git
cd ml-ids-analyzer
```

## 🧪 CLI Usage (via Docker)

### 🧼 1. Preprocess CICIDS2017
```bash
  poetry run mlids-preprocess
```

### 🏋️‍♂️ 2. Train the Model
```bash
  poetry run mlids-train
```

### 📈 3. Predict on New Data
```bash
  poetry run mlids-predict 
```

---

## 🌐 Run the API & Dashboard

### 1. Start the FastAPI prediction server

```bash
poetry run uvicorn src.ml_ids_analyzer.api.app:app --reload --host 0.0.0.0 --port 8000
```

- Test the API at: [http://localhost:8000/docs](http://localhost:8000/docs)
- Main prediction endpoint: `POST /predict/csv` (upload a CSV file)

### 2. Start the Streamlit Dashboard

In a new terminal:

```bash
poetry run streamlit run src/ml_ids_analyzer/api/dashboard.py
```

- Access the dashboard at: [http://localhost:8501](http://localhost:8501)
- The dashboard lets you upload a CSV, view predictions, download results, and see visualizations.

---

## 📊 Visualizations

Generated automatically during training and saved to `/outputs`:

- **Confusion Matrix**
 
  ![Confusion Matrix](outputs/Random_Forest_tuned_confusion_matrix.png)
- **Precision-Recall Curve**

  ![PR Curve](outputs/precision_recall_curve.png)
- **SHAP Feature Importance**

  ![SHAP Summary](outputs/shap_summary.png)

---

## 📁 Project Structure

```
ml-ids-analyzer/
├── config/                  # YAML config file
├── data/                    # Raw & processed input/output
├── docker/                  # Dockerfile and entrypoint
├── outputs/                 # Model artifacts, plots, logs
├── src/ml_ids_analyzer/     # Source code package
│   ├── preprocessing/       # Feature extraction & cleaning
│   ├── modeling/            # Training & threshold tuning
│   ├── inference/           # Prediction logic
│   └── api/                 # FastAPI app & Streamlit dashboard
└── tests/                   # Unit tests
```

---

## 👥 Authors

- **Alexander Zimpher**
- **Spencer Hendren**
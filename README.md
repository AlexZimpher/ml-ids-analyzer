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
- ⚙️ FastAPI prediction server

---

## 🏁 Quickstart

### 1. Clone the Repo
```bash
# Clone and enter the project
git clone https://github.com/AlexZimpher/ml-ids-analyzer.git
cd ml-ids-analyzer
```

### 2. Install with Poetry
```bash
poetry install
poetry shell
```

---

## 🛠️ CLI Usage

### Preprocess Data
```bash
poetry run mlids-preprocess --data-dir data/cicids2017_raw --output-file outputs/cleaned.csv
```

### Train the Model
```bash
poetry run mlids-train --input-file outputs/cleaned.csv --output-dir outputs/
```

### Predict on New Data
```bash
poetry run mlids-predict --input-file <your_data.csv> --model-file outputs/random_forest_model.joblib --scaler-file outputs/scaler.joblib --output-file outputs/predictions.csv
```

### All-in-one CLI
```bash
poetry run mlids-analyzer --help
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

## 📁 Project Structure & Data Flow

A clear, professional overview of the repository for reviewers and recruiters:

```
ml-ids-analyzer/
├── config/                  # YAML configuration files (paths, features, labels, etc.)
├── data/                    # Raw and processed input data (e.g., CICIDS2017)
├── outputs/                 # Model artifacts, predictions, and generated plots
│   ├── random_forest_model.joblib      # Trained model (ready for use)
│   ├── scaler.joblib                  # Scaler for preprocessing
│   ├── Random_Forest_tuned_confusion_matrix.png  # Confusion matrix plot
│   ├── precision_recall_curve.png     # Precision-Recall curve
│   ├── shap_summary.png               # SHAP feature importance plot
│   └── ...                            # Other results
├── src/
│   └── ml_ids_analyzer/     # Main Python package
│       ├── preprocessing/   # Data cleaning, feature extraction, label mapping
│       │   └── preprocess.py
│       ├── modeling/        # Model training, evaluation, threshold tuning
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── model.py
│       ├── inference/       # Prediction logic for new data
│       │   └── predict.py
│       ├── explainability/  # SHAP explainability and feature importance
│       │   └── explain.py
│       ├── api/             # FastAPI app and Streamlit dashboard
│       │   ├── app.py       # REST API for real-time predictions
│       │   └── dashboard.py # Interactive dashboard for data exploration
│       └── cli/             # Unified CLI entry point
│           └── cli.py
├── tests/                   # Pytest-based unit and integration tests
├── pyproject.toml           # Poetry project configuration
├── README.md                # Project overview and instructions
└── ...                      # Other supporting files
```

### 🔄 Typical Workflow
1. **Preprocess**: Clean and merge raw data, outputting a ready-to-train CSV.
2. **Train**: Fit a Random Forest, tune threshold, and save model/scaler.
3. **Predict**: Use the trained model to predict on new data.
4. **Explain**: Generate SHAP plots and evaluation metrics.
5. **Serve**: Run the API for real-time predictions or launch the dashboard for interactive exploration.

---

## 📊 Visualizations (Auto-generated)

- **Confusion Matrix**: Shows model accuracy and error types.
  
  ![Confusion Matrix](outputs/Random_Forest_tuned_confusion_matrix.png)
  > *Interpretation: Diagonal = correct predictions; off-diagonal = errors.*

- **Precision-Recall Curve**: Evaluates model performance across thresholds.
  
  ![PR Curve](outputs/precision_recall_curve.png)
  > *Interpretation: Higher area = better precision/recall tradeoff.*

- **SHAP Feature Importance**: Explains which features drive model decisions.
  
  ![SHAP Summary](outputs/shap_summary.png)
  > *Interpretation: Top features have the most impact on predictions.*

---

## 🚦 Essential Commands

All commands assume you are in the project root and have run `poetry install`.

| Task                | Command                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Preprocess data     | `poetry run mlids-preprocess --data-dir data/cicids2017_raw --output-file outputs/cleaned.csv` |
| Train the model     | `poetry run mlids-train --input-file outputs/cleaned.csv --output-dir outputs/`                |
| Predict on new data | `poetry run mlids-predict --input-file <your_data.csv> --model-file outputs/random_forest_model.joblib --scaler-file outputs/scaler.joblib --output-file outputs/predictions.csv` |
| API server          | `poetry run uvicorn src.ml_ids_analyzer.api.app:app --reload --host 0.0.0.0 --port 8000`      |
| Dashboard           | `poetry run streamlit run src/ml_ids_analyzer/api/dashboard.py`                                 |
| Run all tests       | `poetry run pytest`                                                                             |

- Replace `<your_data.csv>` with your own file as needed.
- See [http://localhost:8000/docs](http://localhost:8000/docs) for API docs after starting the server.
- The dashboard is available at [http://localhost:8501](http://localhost:8501) after launch.

---

## 👥 Authors

- **Alexander Zimpher**
- **Spencer Hendren**
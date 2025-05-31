# ML-IDS-Analyzer

![Build](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9--3.10-blue.svg)
![Built with Poetry](https://img.shields.io/badge/Built%20with-Poetry-612C63.svg?logo=python&logoColor=white)

**ML-IDS-Analyzer** is a modular machine learning pipeline for analyzing intrusion detection system (IDS) alerts. It uses the CICIDS2017 dataset to build a binary classifier that predicts whether a network connection is an attack or benign traffic. The project includes preprocessing tools, model training and tuning, a prediction CLI, and a REST API using FastAPI.

---

## Features

* Preprocess and clean CICIDS2017 data
* Train a Random Forest model with threshold tuning for F1-score
* Predict attacks on new data via CLI or API
* Visualize model performance with confusion matrix, precision-recall curve, and SHAP feature importance
* Dockerized setup and configurable environment

---

## Installation

### Option 1: Local (Poetry)

```bash
git clone https://github.com/AlexZimpher/ml-ids-analyzer.git
cd ml-ids-analyzer
poetry install
```

Ensure `data/cicids2017` exists and contains the raw CSVs. Update the config paths as needed.

### Option 2: Docker (Recommended)

```bash
docker build -t ml-ids-analyzer .
docker run -it --rm -p 8000:8000 ml-ids-analyzer
```

---

## Usage

### 1. Preprocess Data

```bash
mlids-preprocess
```

This merges and cleans the raw CICIDS2017 CSVs. Output is `data/cicids2017_clean.csv`.

### 2. Train Model

```bash
mlids-train
```

Trains a Random Forest classifier, tunes the threshold, and saves the model/scaler.
Outputs include:

* `outputs/model.joblib`
* `outputs/scaler.joblib`
* Evaluation metrics
* Confusion matrix and PR curve
* SHAP summary plot

### 3. Predict New Data

```bash
mlids-predict --input-file path/to/file.csv --output-file path/to/results.csv
```

Adds `prob_attack` and `pred_attack` columns to the output file.

---

## API

### Start the API

```bash
uvicorn ml_ids_analyzer.api.app:app --reload
```

### Endpoints

* `GET /health` – health check
* `POST /predict` – submit JSON data with a list of feature dicts:

```json
{
  "data": [
    {"feature1": 0.1, "feature2": 3.5, ...},
    {"feature1": 0.2, "feature2": 4.0, ...}
  ]
}
```

Returns:

```json
{
  "results": [
    {"prob_attack": 0.91, "pred_attack": 1},
    {"prob_attack": 0.04, "pred_attack": 0}
  ]
}
```
### 4. Extract Features from Suricata Alerts

You can convert raw Suricata JSON logs into model-ready feature vectors:
```bash
mlids-suricata-features -i data/suricata -o data/suricata_features.csv

---

## Project Structure

```
ml-ids-analyzer/
├── src/
│   └── ml_ids_analyzer/
│       ├── preprocessing/
│       ├── modeling/
│       ├── inference/
│       ├── api/
│       └── config/
├── tests/
├── data/
├── outputs/
└── docker/
```

---

## Visualizations

**Confusion Matrix**  
![Confusion Matrix](outputs/confusion_matrix.png)

**Precision-Recall Curve**  
![Precision-Recall Curve](outputs/pr_curve.png)

**SHAP Summary Plot**  
![SHAP Summary](outputs/shap_summary.png)

---

## Configuration

YAML config files define paths and hyperparameters:

* `config/base.yaml`
* `config/dev.yaml`

Set the `ENV` variable to control which config is loaded:

```bash
ENV=dev mlids-train
```

---

## License

MIT License

---

## Authors

Alexander Zimpher
Spencer Hendren

---

## Future Improvements

* Add authentication and input validation to API
* Deploy API/Streamlit app for live demo
* Improve UI and interactivity
* Write full Colab demo notebook

---

## Dataset

[CICIDS2017 - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
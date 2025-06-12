# ML-IDS-Analyzer

[![Build Status](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Built with Poetry](https://img.shields.io/badge/Built%20with-Poetry-612C63.svg?logo=python&logoColor=white)
[![codecov](https://codecov.io/gh/AlexZimpher/ml-ids-analyzer/graph/badge.svg?token=DMYGFS3OEO)](https://codecov.io/gh/AlexZimpher/ml-ids-analyzer)

**ML-IDS-Analyzer** is a machine learning pipeline and REST API for analyzing intrusion detection system (IDS) alerts. It uses the CICIDS2017 dataset to build a classifier that distinguishes malicious from benign traffic. The system includes data preprocessing, model training, threshold tuning, feature importance analysis, and a real-time prediction API.

---

## 🚀 Features

- Modular ML pipeline (preprocessing, training, prediction)
- Random Forest with threshold optimization
- SHAP-based explainability and evaluation plots
- FastAPI prediction server (coming soon)
- Docker workflow (coming soon)
- Suricata integration (coming soon)

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

## 🧪 Testing
```bash
poetry run pytest --cov=src/ml_ids_analyzer tests/
```

---

## 🐳 Docker (WIP)
- See `docker/` for Dockerfile and compose setup.

---

## 📁 Project Structure
```
ml-ids-analyzer/
├── src/ml_ids_analyzer/
│   ├── cli/cli.py
│   ├── modeling/model.py
│   ├── ...
├── tests/
│   ├── ...
├── pyproject.toml
├── README.md
```

---

## 🤝 Contributing
PRs and issues welcome!

---

## 📜 License
MIT
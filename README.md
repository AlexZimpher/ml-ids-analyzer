# ML-IDS-Analyzer

![Build](https://github.com/AlexZimpher/ml-ids-analyzer/actions/workflows/test.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Built with Poetry](https://img.shields.io/badge/Built%20with-Poetry-612C63.svg?logo=python&logoColor=white)

**ML-IDS-Analyzer** is a machine learning pipeline and REST API for analyzing intrusion detection system (IDS) alerts. It uses the CICIDS2017 dataset to build a classifier that distinguishes malicious from benign traffic. The system includes data preprocessing, model training, threshold tuning, feature importance analysis, and a real-time prediction API.

---

## 🚀 Features

- 📦 Modular ML pipeline (preprocessing, training, prediction)
- 🧠 Random Forest with threshold optimization
- 📊 SHAP-based explainability and evaluation plots
- ⚙️ FastAPI prediction server
- 🐳 Docker-only workflow
- 🛠️ Command-line tools for batch processing and feature extraction from Suricata alerts

---

## 📦 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/AlexZimpher/ml-ids-analyzer.git
cd ml-ids-analyzer
```

### 2. Build the Docker Image
```bash
docker build -f docker/Dockerfile.dev -t mlids-analyzer .
```

---

## 🧪 CLI Usage (via Docker)

All functionality is exposed through Dockerized CLI tools. No local Python or Poetry install is required.

> Replace paths as needed based on your local filesystem.

### 🧼 1. Preprocess CICIDS2017
```bash
docker run -it --rm -v ${PWD}:/app -w /app mlids-analyzer \
  poetry run mlids-preprocess
```

### 🏋️‍♂️ 2. Train the Model
```bash
docker run -it --rm -v ${PWD}:/app -w /app mlids-analyzer \
  poetry run mlids-train
```

### 📈 3. Predict on New Data
```bash
docker run -it --rm -v ${PWD}:/app -w /app mlids-analyzer \
  poetry run mlids-predict \
  --input-file data/sample_input.csv \
  --output-file data/sample_output.csv
```

### 🛡️ 4. Extract Features from Suricata Alerts
```bash
docker run -it --rm -v ${PWD}:/app -w /app mlids-analyzer \
  poetry run mlids-suricata-features \
  -i data/suricata \
  -o data/suricata_features.csv
```

---

## 🌐 Run the API (FastAPI)

Launch the prediction API server:

```bash
docker run -it --rm -v ${PWD}:/app -w /app -p 8000:8000 mlids-analyzer \
  poetry run uvicorn ml_ids_analyzer.api.app:app --host 0.0.0.0 --port 8000
```

Test it at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Visualizations

Generated automatically during training and saved to `/outputs`:

- **Confusion Matrix**

  ![Confusion Matrix](outputs/Random_Forest_tuned_confusion_matrix)

- **Precision-Recall Curve**

  ![PR Curve](outputs/precision_recall_curve.png)

- **SHAP Feature Importance**

  ![SHAP Summary](outputs/shap_summary.png)

---

## 📁 Project Structure

```
ml-ids-analyzer/
├── config/                  # YAML config files (base/dev/prod)
├── data/                    # Raw & processed input/output
├── docker/                  # Dockerfile and entrypoint
├── outputs/                 # Model artifacts, plots, logs
├── src/ml_ids_analyzer/    # Source code package
│   ├── preprocessing/       # Feature extraction & cleaning
│   ├── modeling/            # Training & threshold tuning
│   ├── inference/           # Prediction logic
│   └── api/                 # FastAPI app
└── tests/                   # Unit tests
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Authors

- **Alexander Gregory Zimpher**
- **Spencer Hendren**
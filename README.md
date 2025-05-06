README.md (public, what any user/reader sees):

# ML-IDS-Analyzer

**Authors:** Alexander Zimpher & Spencer Hendren

## 🚀 Overview

ML-IDS-Analyzer is a modular, containerized machine-learning pipeline for turning raw IDS alerts (e.g., Suricata EVE logs) into actionable predictions (true vs. false positives).

Key features:
- **End-to-end ML pipeline**: ingest → preprocess → train → evaluate → infer
- **Threshold tuning** to maximize production F1
- **Explainability** via SHAP
- **CLI tools** and **FastAPI** REST interface
- **Dockerized** for both dev & prod workflows

## 🗂 Repository Layout

ml-ids-analyzer/
├── config/ # base/dev/prod YAMLs + .env.example + entrypoint.sh
├── docker/ # Dockerfiles & helper scripts
├── data/ # raw & processed datasets
├── docs/ # docs & slide decks
├── notebooks/ # exploratory & demo notebooks
├── src/
│ └── ml_ids_analyzer/ # application package
│ ├── api/ # FastAPI app
│ ├── config/ # in-package loader
│ ├── preprocessing/
│ ├── modeling/
│ └── inference/
├── tests/ # unit & integration tests
├── pyproject.toml # Poetry settings
├── .gitignore
└── README.md # this file


## ⚙️ Configuration

All runtime settings live in `config/`:

- `base.yaml` → defaults  
- `dev.yaml` → local overrides  
- `prod.yaml` → production overrides  
- `.env.example` → copy to `config/.env` and fill in secrets  
- `entrypoint.sh` → loads `config/.env` at container start  

The package merges these plus any real `$ENV` into `ml_ids_analyzer.config.cfg`.

## 🛠️ Quickstart

### 1. Native (Poetry + Uvicorn)

```bash
# install deps
poetry install

# copy & edit your secrets
cp config/.env.example config/.env

# run FastAPI
poetry run uvicorn ml_ids_analyzer.api.app:app --reload --port 8000

# healthcheck
curl http://localhost:8000/health

2. Docker (Development)

# build dev image
docker build -f docker/Dockerfile.dev -t ml-ids-dev .

# run (mount your .env)
docker run --rm -p 8000:8000 \
  -v "$(pwd)/config/.env:/app/config/.env" \
  ml-ids-dev

# verify
curl http://localhost:8000/

3. Docker (Production)

# build prod image
docker build -f docker/Dockerfile.prod -t ml-ids-analyzer:latest .

# run
docker run --rm -p 8000:8000 \
  -v "$(pwd)/config/.env:/app/config/.env" \
  ml-ids-analyzer:latest

curl http://localhost:8000/health

🔧 CLI Tools

After poetry install, you get:

    mlids-preprocess – run preprocessing

    mlids-train – train & tune model

    mlids-predict – batch inference

🧪 Testing

poetry run pytest --maxfail=1 --disable-warnings -q

🎯 Tech Stack
Layer	Tools
Language	Python 3.9
ML & Modeling	scikit-learn, XGBoost
Data & EDA	pandas, matplotlib
Explainability	SHAP
API	FastAPI, Uvicorn
Packaging & CLI	Poetry
Deployment	Docker
🔮 Roadmap

    Real-time streaming of Suricata EVE JSON

    Web dashboard for threshold tweaking

    Automated retraining on data drift

    Monitoring & alerting integration

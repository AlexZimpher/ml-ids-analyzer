# ML-IDS-Analyzer

**Authors:** Alexander Zimpher (@AlexZimpher) & Spencer Hendren (@SnakAttack)

---

## 🚀 Overview

ML-IDS-Analyzer is a modular machine-learning pipeline for turning raw IDS alerts (e.g. Suricata EVE logs) into actionable predictions (true vs. false positives). Key features:

- **End-to-end ML pipeline** (ingest, preprocess, train, evaluate, infer)
- **Threshold tuning** to maximize F1 in production
- **Explainability** via SHAP
- **CLI tools** and **FastAPI** REST interface
- **Containerized** for both dev & prod workflows

---

## 🗂️ Repository Layout

ml-ids-analyzer/
├── config/ # Base, dev & prod YAMLs + .env.example + entrypoint
├── docker/ # Dockerfiles & healthcheck scripts
├── data/ # Raw & processed datasets
├── docs/ # Documentation & slide decks
├── src/
│ └── ml_ids_analyzer/ # Your application package
│ ├── api/ # FastAPI application
│ ├── config/ # In-package loader merging config/*
│ ├── preprocessing/ # Data cleaning & feature engineering
│ ├── modeling/ # Train & evaluate code
│ ├── inference/ # Batch & streaming prediction scripts
│ └── model.py # CLI entrypoint for training
├── tests/ # Unit & integration tests
├── pyproject.toml # Poetry project & dependency management
├── README.md # This file
└── .gitignore


---

## ⚙️ Configuration

All run-time settings live in `config/`:

- `base.yaml` → common defaults  
- `dev.yaml` → overrides for local development  
- `prod.yaml` → overrides for production deployment  
- `.env.example` → sample secrets file (copy to `config/.env`)  
- `entrypoint.sh` → loads `.env` at container start  

Your package code reads and merges these plus any real `$ENV`:

```bash
ENV=dev|prod
# then ml_ids_analyzer.config.cfg contains {**base, **override, **os.environ}

🛠️ Quickstart
1. Local (Poetry + Uvicorn)

# 1. Install dependencies
poetry install

# 2. Copy your secrets
cp config/.env.example config/.env
# (fill in any DB_PASSWORD, API_KEY, etc.)

# 3. Run the FastAPI server
poetry run uvicorn ml_ids_analyzer.api.app:app --reload --port 8000

# 4. Healthcheck
curl http://localhost:8000/health

2. Docker (Development)

# 1. Build dev image (mounts your source & config)
docker build -f docker/Dockerfile.dev -t ml-ids-dev .

# 2. Run (mount config/.env so secrets load)
docker run --rm -p 8000:8000 \
  -v "$(pwd)/config/.env:/app/config/.env" \
  ml-ids-dev

# 3. Verify
curl http://localhost:8000/

3. Docker (Production)

# 1. Build prod image (slimmer, with venv baked in)
docker build -f docker/Dockerfile.prod -t ml-ids-analyzer:latest .

# 2. Run with your .env mounted
docker run --rm -p 8000:8000 \
  -v "$(pwd)/config/.env:/app/config/.env" \
  ml-ids-analyzer:latest

🔧 CLI Tools

Once installed via Poetry, you have:

mlids-preprocess    # runs preprocessing pipeline
mlids-train         # trains & tunes model
mlids-predict       # batch inference script

🧪 Testing

poetry run pytest --maxfail=1 --disable-warnings -q

📊 Demo Notebook

See notebooks/03_demo.ipynb for an end-to-end walkthrough of loading data, training, threshold tuning, SHAP explainability, and inference.
🎯 Tech Stack
Layer	Tools
Language	Python 3.9
ML & Modeling	scikit-learn, XGBoost
Data & EDA	pandas, matplotlib
Explainability	SHAP
API	FastAPI, Uvicorn
Packaging & CLI	Poetry
Deployment	Docker
🔮 Next Steps

    Real-time streaming of Suricata EVE JSON

    Web dashboard for threshold tweaking

    Automated retraining on data drift

    Monitoring & alerting integration

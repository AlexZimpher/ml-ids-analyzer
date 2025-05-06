# ML-IDS-Analyzer

ML-IDS-Analyzer is a machine learning pipeline for classifying IDS (Intrusion Detection System) alerts—specifically those from Suricata—as either valid or invalid. It is designed to assist small-scale SOC environments in reducing alert fatigue and improving response prioritization.

## Features

- End-to-end ML pipeline: preprocessing, training, prediction
- Supports Suricata alert JSONs
- FastAPI-based REST API
- Configurable YAML-based settings
- Dockerized development and production environments
- SHAP model explainability

## Setup (Dev)

```bash
# Build
docker build -f docker/Dockerfile.dev -t ml-ids-dev .

# Run with mounted .env
docker run --rm -p 8000:8000 -v "${PWD}/config/.env:/app/config/.env" ml-ids-dev
```

## Setup (Prod)

```bash
# Build
docker build -f docker/Dockerfile.prod -t ml-ids-prod .

# Run
docker run --rm -p 8000:8000 -v "${PWD}/config/.env:/app/config/.env" ml-ids-prod
```

## API

- `GET /health` – API health check
- `GET /` – Welcome message
- `POST /predict` – Submit alert JSONs for classification

## CLI Scripts

```bash
# Preprocess Suricata alert data
poetry run mlids-preprocess

# Train model
poetry run mlids-train

# Predict from file
poetry run mlids-predict
```

## License

MIT
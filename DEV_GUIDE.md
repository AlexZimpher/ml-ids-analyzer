# ML-IDS-Analyzer Developer Guide

## Overview

This private developer guide is intended for Alexander and Spencer. It documents setup, development, deployment, and usage procedures for internal use and showcases project competency for job-seeking purposes.

---

## 🛠 Setup

### Prerequisites

- Docker Desktop
- Poetry (automatically installed in containers)
- Python 3.9 (if running locally, outside Docker)

### Initial Setup

```bash
git clone https://github.com/AlexZimpher/ml-ids-analyzer
cd ml-ids-analyzer
cp config/.env.example config/.env
```

---

## 🔧 Development

### Build Dev Image

```bash
docker build -f docker/Dockerfile.dev -t ml-ids-dev .
```

### Run Dev Server

```bash
docker run --rm -p 8000:8000 -v "${PWD}/config/.env:/app/config/.env" ml-ids-dev
```

### Poetry Shell (if needed)

```bash
poetry shell
```

---

## 🚀 Production

### Build Prod Image

```bash
docker build -f docker/Dockerfile.prod -t ml-ids-prod .
```

### Run Prod Server

```bash
docker run --rm -p 8000:8000 -v "${PWD}/config/.env:/app/config/.env" ml-ids-prod
```

---

## 🧪 Testing

```bash
pytest
```

---

## 🔄 Updating Dependencies

```bash
poetry add <package>
poetry lock
```

---

## 🧠 Notes / TODO

- Implement proper validation + schema enforcement in FastAPI
- Add logging middleware to API
- Improve model packaging (e.g. model registry or versioning)
- Add basic auth or API key support
- Create full end-to-end test harness
- Optional: Integrate with frontend dashboard or SIEM

This file is for internal use and **must be deleted before public release**.
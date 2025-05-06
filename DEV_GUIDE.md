
---

**`DEV_GUIDE.md`** (private, for you & Spencer—drop this before publishing your portfolio):

```markdown
# ML-IDS-Analyzer Developer Guide (🔒 Private)

## 1. Environment Setup

1. **Python 3.9.22**  
   - Install from https://www.python.org/downloads/release/python-3922/  
   - On Windows, add to PATH or use the `py` launcher:  
     ```ps1
     py -3.9 --version
     ```
2. **Poetry**  
   ```bash
   pip install poetry
   poetry self update

    Activate Poetry env

poetry env use C:/path/to/python3.9/python.exe
poetry install

Copy & configure secrets

    cp config/.env.example config/.env
    # Edit config/.env: set ENV=dev or prod, fill DB_PASSWORD, API_KEY, etc.

2. Local Development
API

poetry run uvicorn ml_ids_analyzer.api.app:app --reload --port 8000
# Browse:
curl http://localhost:8000/
curl http://localhost:8000/health

CLI

mlids-preprocess     # cleans & featurizes data
mlids-train          # trains model & saves artifacts
mlids-predict        # batch inference script

3. Docker Workflows
Dockerfile.dev

    Build (skips packaging, mounts source & config):

docker build -f docker/Dockerfile.dev -t ml-ids-dev .

Run:

    docker run --rm -p 8000:8000 \
      -v "$(pwd)/config/.env:/app/config/.env" \
      ml-ids-dev

Dockerfile.prod

    Build (multistage, bakes venv):

docker build -f docker/Dockerfile.prod -t ml-ids-analyzer:latest .

Run:

    docker run --rm -p 8000:8000 \
      -v "$(pwd)/config/.env:/app/config/.env" \
      ml-ids-analyzer:latest

Troubleshooting

    CRLF vs LF in entrypoint.sh:

    sed -i 's/\r$//' config/entrypoint.sh
    chmod +x config/entrypoint.sh

    Ensure config/ is copied after src so base.yaml & friends live at /app/config.

4. Testing & QA

poetry run pytest --maxfail=1 --disable-warnings -q

CI/CD can mirror this (pytest + docker build).
5. Git & GitHub

git add .
git commit -m "feat: final Docker + config setup"
git push origin main

    PR reviews: focus on Docker caching & test coverage.

    Remove DEV_GUIDE.md before public release.

6. Pending Work

    Streaming ingestion for live Suricata feeds

    Dashboard for threshold tuning (React + Tailwind)

    Auto-retrain on data drift + model versioning

    Monitoring (e.g. Prometheus + Grafana)

    CI with GitHub Actions: lint, format, test, build images

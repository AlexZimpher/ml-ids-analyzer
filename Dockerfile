# Stage 1: build
FROM python:3.13-slim-bullseye AS builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    python -m pip install --upgrade pip setuptools wheel

COPY setup.py setup.cfg /app/
COPY src /app/src
RUN pip install --no-cache-dir -e .

# Stage 2: runtime
FROM python:3.13-slim-bullseye
WORKDIR /app

# Copy installed packages and code
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/src /app/src

RUN python -m pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl --fail http://localhost:8000/docs || exit 1

CMD ["python", "-m", "uvicorn", "ml_ids_analyzer.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

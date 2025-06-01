# ---- Base Python image ----
FROM python:3.11-slim AS base

# ---- System setup ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Install Poetry ----
RUN pip install --no-cache-dir poetry

# ---- Copy only dependency files for caching ----
COPY pyproject.toml poetry.lock ./

# ---- Install runtime dependencies (including SHAP) ----
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-root --no-interaction

# ---- Copy application code ----
COPY src/ml_ids_analyzer ./ml_ids_analyzer
COPY config ./config
COPY entrypoint.sh ./
COPY config/default.yaml ./config/default.yaml

# ---- Add CLI to path ----
ENV PATH="/app/.local/bin:$PATH"

# ---- Default: Run FastAPI app ----
EXPOSE 8000
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["uvicorn", "ml_ids_analyzer.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
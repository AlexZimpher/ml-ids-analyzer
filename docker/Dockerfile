# Use slim Python base image
FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /app

# Copy metadata files first for caching
COPY pyproject.toml poetry.lock README.md /app/

# Install Poetry and dependencies
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the application code and configuration
COPY src /app/src
COPY config /app/config
COPY docker/entrypoint.sh /entrypoint.sh

# Make the entrypoint executable
RUN chmod +x /entrypoint.sh

# Set entrypoint to use CLI dispatcher
ENTRYPOINT ["/entrypoint.sh"]
#!/usr/bin/env sh
set -e

# Load root .env if present
if [ -f /app/.env ]; then
  export $(grep -v '^#' /app/.env | xargs)
fi

# Load config/.env if present
if [ -f /app/config/.env ]; then
  export $(grep -v '^#' /app/config/.env | xargs)
fi

# Exec the CMD from Dockerfile
exec "$@"
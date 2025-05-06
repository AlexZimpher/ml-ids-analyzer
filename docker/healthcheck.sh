#!/usr/bin/env sh
# docker/healthcheck.sh
if curl -sf http://localhost:8000/health; then
  exit 0
else
  exit 1
fi

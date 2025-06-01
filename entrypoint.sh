#!/bin/bash
set -e

# Load environment if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

exec "$@"

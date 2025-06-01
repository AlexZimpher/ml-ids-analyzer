#!/bin/bash
set -e

case "$1" in
  train)
    poetry run mlids-train
    ;;
  preprocess)
    poetry run mlids-preprocess
    ;;
  predict)
    poetry run mlids-predict --input-file "$2" --output-file "$3"
    ;;
  suricata)
    poetry run mlids-suricata-features -i "$2" -o "$3"
    ;;
  api)
    poetry run uvicorn ml_ids_analyzer.api.app:app --host 0.0.0.0 --port 8000
    ;;
  bash)
    exec bash
    ;;
  *)
    echo "ML-IDS Analyzer Docker CLI"
    echo ""
    echo "Usage:"
    echo "  train                             - Train the model"
    echo "  preprocess                        - Preprocess CICIDS2017 data"
    echo "  predict <in.csv> <out.csv>       - Predict using trained model"
    echo "  suricata <dir> <out.csv>         - Extract features from Suricata logs"
    echo "  api                               - Launch FastAPI server"
    echo "  bash                              - Open an interactive shell"
    exit 1
    ;;
esac
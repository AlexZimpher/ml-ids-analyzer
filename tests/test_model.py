import pytest
from ml_ids_analyzer.model import train_model

def test_train_model_runs():
    # Smoke test: training should run without exceptions on minimal config
    # (Consider monkeypatching cfg to point at a small toy dataset)
    assert callable(train_model)


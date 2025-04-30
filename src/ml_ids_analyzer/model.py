# ml_ids_analyzer/model.py
"""
Console‐script entry point for ML‐IDS‐Analyzer training.
"""
from .modeling.train import train_model

def main():
    """Entry point for `mlids-train`."""
    train_model()
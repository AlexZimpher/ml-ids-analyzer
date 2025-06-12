# src/ml_ids_analyzer/config/__init__.py
"""
Configuration loader for ml_ids_analyzer.
"""

import os
from pathlib import Path
import yaml

# project root is three levels above this file:
#    src/ml_ids_analyzer/config/__init__.py  <-- 0
#    src/ml_ids_analyzer/config/            <-- 1
#    src/ml_ids_analyzer/                   <-- 2
#    src/                                   <-- 3  <-- ROOT
ROOT = Path(__file__).parents[3]

# load only default.yaml (no environment‐specific overrides)
default_path = ROOT / "config" / "default.yaml"
base = yaml.safe_load(default_path.read_text(encoding="utf-8"))

# merge default settings with any environment variables
cfg = {**base, **os.environ}

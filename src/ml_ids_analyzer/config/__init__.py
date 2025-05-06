# src/ml_ids_analyzer/config/__init__.py

import os
from pathlib import Path
import yaml

# project root is three levels above this file:
#    src/ml_ids_analyzer/config/__init__.py  <-- 0
#    src/ml_ids_analyzer/config/            <-- 1
#    src/ml_ids_analyzer/                   <-- 2
#    src/                                   <-- 3  <-- ROOT
ROOT = Path(__file__).parents[3]

# which environment? default to "dev"
env = os.getenv("ENV", "dev")

# load the base + override YAMLs
base_path     = ROOT / "config" / "base.yaml"
override_path = ROOT / "config" / f"{env}.yaml"

base     = yaml.safe_load(base_path.read_text(encoding="utf-8"))
override = yaml.safe_load(override_path.read_text(encoding="utf-8"))

# merge: base ← override ← os.environ
cfg = {**base, **override, **os.environ}

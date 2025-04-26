import os
import yaml

# Load config from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

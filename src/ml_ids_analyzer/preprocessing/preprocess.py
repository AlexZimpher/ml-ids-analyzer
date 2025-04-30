# ml_ids_analyzer/preprocessing/preprocess.py

import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

from ml_ids_analyzer.config import cfg

# configure module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
_logger = logging.getLogger(__name__)

# pull settings from config.yaml
DATA_DIR = cfg["data"]["raw_dir"]
OUTPUT_CLEAN = cfg["data"]["clean_file"]
LABEL_COL = cfg["label_column"]
MISSING_THRESH = cfg["cleaning"]["missing_threshold"]


def load_and_merge_csvs(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Read all .csv files from `data_dir`, strip column whitespace,
    and concatenate into a single DataFrame.
    """
    data_dir = data_dir or DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir!r}")

    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        _logger.warning("No CSV files found in %r", data_dir)

    frames = []
    for fname in sorted(csv_files):
        path = os.path.join(data_dir, fname)
        _logger.info("Loading %s …", fname)
        df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _logger.info("Merged DataFrame shape: %s", combined.shape)
    return combined


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and label the input DataFrame:
    - Strip column names
    - Drop columns with > (1 - MISSING_THRESH) missing values
    - Drop constant or binary columns
    - Drop rows with missing values
    - Map LABEL_COL: 'BENIGN' → 0, others → 1
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    thresh = int(MISSING_THRESH * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    nun = df.nunique(dropna=False)
    to_drop = nun[nun <= 1].index.tolist()
    if to_drop:
        _logger.info("Dropping constant columns: %s", to_drop)
        df = df.drop(columns=to_drop)

    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]
    if before != after:
        _logger.info("Dropped %d rows with missing values", before - after)

    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected a column named '{LABEL_COL}'")
    df[LABEL_COL] = df[LABEL_COL].apply(
        lambda v: 0 if str(v).upper() == "BENIGN" else 1
    )

    _logger.info("After cleaning & labeling: %s", df.shape)
    return df


def main() -> None:
    """
    Pipeline entrypoint: merge, clean, label, and write out the cleaned CSV.
    """
    df_raw = load_and_merge_csvs(DATA_DIR)
    df_clean = clean_and_label(df_raw)

    out_dir = os.path.dirname(OUTPUT_CLEAN) or "."
    os.makedirs(out_dir, exist_ok=True)
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    _logger.info("Saved cleaned dataset to %s", OUTPUT_CLEAN)


if __name__ == "__main__":
    main()
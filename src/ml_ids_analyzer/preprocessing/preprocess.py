"""
Preprocessing utilities for ml_ids_analyzer.
"""

import os
import logging
from math import ceil

import click
import pandas as pd

from ml_ids_analyzer.config import cfg

# configure module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
_logger = logging.getLogger(__name__)

# defaults from config.yaml
DEFAULT_DATA_DIR: str = cfg["data"]["raw_dir"]
DEFAULT_OUTPUT_CLEAN: str = cfg["data"]["clean_file"]
LABEL_COL: str = cfg["label_column"]
MISSING_THRESH: float = cfg["cleaning"]["missing_threshold"]


def load_and_merge_csvs(data_dir: str) -> pd.DataFrame:
    """
    Load and merge all CSV files in the given directory into a single DataFrame.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir!r}")

    csvs = sorted(
        f for f in os.listdir(data_dir) if f.lower().endswith(".csv")
    )
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in raw data directory: {data_dir!r}"
        )

    dfs = []
    for fn in csvs:
        path = os.path.join(data_dir, fn)
        _logger.info("Loading %s ", fn)
        df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
        df.rename(
            columns=lambda x: x.strip() if isinstance(x, str) else x,
            inplace=True,
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    _logger.info("Merged DataFrame shape: %s", combined.shape)
    return combined


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping columns/rows with too many missing values,
    dropping constant columns, and mapping labels to 0/1.
    """
    df = df.copy()
    df.rename(
        columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True
    )

    # drop cols with too many missing
    min_non_na = ceil((1.0 - MISSING_THRESH) * len(df))
    df = df.dropna(axis=1, thresh=min_non_na)

    # drop constant cols
    nun = df.nunique(dropna=False)
    const = nun[nun == 1].index.tolist()
    if const:
        _logger.info("Dropping constant columns: %s", const)
        df = df.drop(columns=const)

    # drop rows with missing
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        _logger.info("Dropped %d rows with missing values", dropped)

    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected a column named '{LABEL_COL}'")

    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.upper()
        .map(lambda v: 0 if v == "BENIGN" else 1)
    )

    _logger.info("After cleaning & labeling: %s", df.shape)
    return df


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-dir",
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Directory containing raw CSV files",
)
@click.option(
    "--output-file",
    default=DEFAULT_OUTPUT_CLEAN,
    show_default=True,
    help="Path to write cleaned CSV",
)
def main(data_dir: str, output_file: str) -> None:
    """
    CLI entry point: merge, clean, and label raw CSVs, then save cleaned data.
    """
    try:
        df_raw = load_and_merge_csvs(data_dir)
        df_clean = clean_and_label(df_raw)
        df_clean.to_csv(output_file, index=False)
        _logger.info(f"Cleaned data saved to {output_file}")
    except Exception as e:
        _logger.error(f"Preprocessing failed: {e}")
        raise

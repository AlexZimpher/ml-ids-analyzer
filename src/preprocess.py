#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import logging
from config import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DATA_DIR = cfg['data']['raw_dir']
OUTPUT_CLEAN = cfg['data']['clean_file']
LABEL_COL = cfg['label_column']
MISSING_THRESH = cfg['cleaning']['missing_threshold']


def load_and_merge_csvs(data_dir):
    csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    frames = []
    for fn in csvs:
        path = os.path.join(data_dir, fn)
        logging.info(f"Loading {fn}…")
        df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logging.info(f"Merged DataFrame shape: {combined.shape}")
    return combined


def clean_and_label(df):
    df.columns = df.columns.str.strip()
    thresh = int(MISSING_THRESH * len(df))
    df = df.dropna(axis=1, thresh=thresh)
    nun = df.nunique()
    df = df.drop(columns=nun[nun <= 1].index)
    df = df.dropna()
    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected a column named '{LABEL_COL}'")
    df[LABEL_COL] = df[LABEL_COL].apply(lambda v: 0 if v == "BENIGN" else 1)
    logging.info(f"After cleaning & labeling: {df.shape}")
    return df


def main():
    df_raw = load_and_merge_csvs(DATA_DIR)
    df_clean = clean_and_label(df_raw)
    os.makedirs(os.path.dirname(OUTPUT_CLEAN), exist_ok=True)
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    logging.info(f"Saved cleaned dataset to {OUTPUT_CLEAN}")


if __name__ == "__main__":
    main()

import os
import pandas as pd

# === Configuration ===
# We assume you invoke this from the project root (ml-ids-analyzer)
BASE_DIR      = os.getcwd()
DATA_DIR      = os.path.join(BASE_DIR, 'data', 'cicids2017')
OUTPUT_CLEAN  = os.path.join(BASE_DIR, 'data', 'cicids2017_clean.csv')
LABEL_COL     = "Label"
POSITIVE_CLASSES = [
    'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
    'Bot', 'Web Attack – Brute Force', 'Web Attack – XSS',
    'Infiltration', 'Web Attack – Sql Injection', 'Heartbleed'
]

def load_and_merge_csvs(data_dir):
    csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    frames = []
    for fn in csvs:
        path = os.path.join(data_dir, fn)
        print(f"Loading {fn}…")
        df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
        # strip whitespace off every column
        df.columns = df.columns.str.strip()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Merged DataFrame shape: {combined.shape}")
    return combined

def clean_and_label(df):
    # strip any lingering whitespace in column names
    df.columns = df.columns.str.strip()

    # drop columns with >10% missing
    thresh = int(0.9 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    # drop any columns that are constant
    nun = df.nunique()
    df = df.drop(columns=nun[nun <= 1].index)

    # drop rows with any missing values
    df = df.dropna()

    # ensure the label column exists
    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected a column named '{LABEL_COL}' but got {df.columns.tolist()}")

    # encode: BENIGN=0, attack=1
    df[LABEL_COL] = df[LABEL_COL].apply(lambda v: 0 if v == "BENIGN" else 1)

    print(f"After cleaning & labeling: {df.shape}")
    return df

def main():
    df_raw   = load_and_merge_csvs(DATA_DIR)
    df_clean = clean_and_label(df_raw)

    # make sure target directory exists
    os.makedirs(os.path.dirname(OUTPUT_CLEAN), exist_ok=True)
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CLEAN}")

if __name__ == "__main__":
    main()

import os
import pandas as pd

# === Config ===
DATA_DIR = "data/cicids2017"
OUTPUT_FILE = "data/cicids2017_clean.csv"
LABEL_COL = "Label"
POSITIVE_CLASSES = [  # These are all attacks
    'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
    'Bot', 'Web Attack – Brute Force', 'Web Attack – XSS',
    'Infiltration', 'Web Attack – Sql Injection', 'Heartbleed'
]

def load_and_merge_csvs(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    df_list = []

    for file in csv_files:
        path = os.path.join(data_dir, file)
        print(f"Loading {file}...")
        df = pd.read_csv(path, low_memory=False)
        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True)
    print(f"Merged shape: {combined.shape}")
    return combined

def clean_and_label(df):
    # Drop columns with only 1 value or mostly NaNs
    df = df.dropna(axis=1, thresh=int(0.9 * len(df)))  # Drop if >10% NaNs
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique <= 1].index)

    # Drop rows with NaNs
    df = df.dropna()

    # Encode binary label: 0 = BENIGN, 1 = ATTACK
    df[LABEL_COL] = df[LABEL_COL].apply(lambda x: 0 if x == "BENIGN" else 1)

    print(f"Final shape after cleaning: {df.shape}")
    return df

def main():
    df = load_and_merge_csvs(DATA_DIR)
    df_clean = clean_and_label(df)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

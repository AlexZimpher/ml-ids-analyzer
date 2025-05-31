# src/ml_ids_analyzer/preprocessing/suricata_to_features.py

import json
import argparse
from pathlib import Path

import pandas as pd

def parse_single_alert(alert_json: dict) -> dict:
    """
    Given one Suricata alert (as a dict), extract the model's features
    and return a dictionary mapping each feature name to its value.
    """
    features = {}

    # Example: flow start & end times
    flow_info = alert_json.get("flow", {})
    start = pd.to_datetime(flow_info.get("start"))
    end   = pd.to_datetime(flow_info.get("end"))
    features["flow_duration"] = (end - start).total_seconds()

    # Example: total forward/backward packets & bytes
    features["tot_fwd_pkts"] = flow_info.get("pkts_toserver", 0)
    features["tot_bwd_pkts"] = flow_info.get("pkts_toclient", 0)
    features["totlen_fwd_pkts"] = flow_info.get("bytes_toserver", 0)
    features["totlen_bwd_pkts"] = flow_info.get("bytes_toclient", 0)

    # TODO: Continue mapping all required features here...
    # For instance, if your cleaned CSV had a column named "fwd_psh_flags",
    # you might do something like:
    # tcp_flags = alert_json.get("tcp", {}).get("tcp_flags", "")
    # features["fwd_psh_flags"] = int("0x020" in tcp_flags)  # example

    # Placeholder: fill other features with default/zero until implemented
    # e.g., features["some_other_feature"] = 0

    return features

def parse_suricata_file(json_path: Path) -> pd.DataFrame:
    """
    Read a Suricata JSON file—which may be either:
      1) a single JSON object (possibly multi-line),
      2) a JSON array of objects,
      3) or many JSON objects, one per line.
    For each alert (dict), call parse_single_alert and collect feature dicts.
    Return a DataFrame of feature vectors.
    """
    records = []

    # First, try to load the entire file as one JSON document
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            content = json.load(f)

        # If content is a list, assume each element is one alert dict
        if isinstance(content, list):
            for alert in content:
                rec = parse_single_alert(alert)
                records.append(rec)
        # If content is a dict, treat it as a single alert
        elif isinstance(content, dict):
            rec = parse_single_alert(content)
            records.append(rec)
        else:
            # Unexpected JSON structure (e.g., a raw string or number)
            print(f"Warning: {json_path} did not yield a dict or list. Skipping.")
    except json.JSONDecodeError:
        # Fallback: file is probably line-delimited JSON (one object per line)
        with open(json_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Strip any leftover BOM on each line
                clean_line = line.lstrip("\ufeff")
                try:
                    alert = json.loads(clean_line)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {json_path}: {e}")
                    continue
                rec = parse_single_alert(alert)
                records.append(rec)

    if not records:
        return pd.DataFrame()  # no valid alerts found

    df = pd.DataFrame.from_records(records)
    return df

def main():
    parser = argparse.ArgumentParser(prog="mlids-suricata-features")
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing Suricata JSON alert files"
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        required=True,
        help="Path to save the resulting feature‐vector CSV"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_dfs = []

    # Iterate over all files ending in .json (or .log) in that directory
    for json_file in input_dir.glob("*.json"):
        df = parse_suricata_file(json_file)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        result_df = pd.concat(all_dfs, ignore_index=True)
    else:
        print(f"No alerts found in {input_dir}. Exiting.")
        return

    # Reorder columns (if necessary) to match your model’s training order
    # e.g., result_df = result_df[["flow_duration", "tot_fwd_pkts", ..., "some_last_feature"]]

    result_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result_df)} records to {args.output_csv}")

if __name__ == "__main__":
    main()

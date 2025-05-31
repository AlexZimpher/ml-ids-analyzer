import json
import argparse
from pathlib import Path

import pandas as pd


def parse_single_alert(alert_json: dict) -> dict:
    """
    Extract relevant model features from a single Suricata alert JSON object.
    """
    features = {}

    # Flow timing
    flow_info = alert_json.get("flow", {})
    try:
        start = pd.to_datetime(flow_info.get("start"))
        end = pd.to_datetime(flow_info.get("end"))
        features["flow_duration"] = (end - start).total_seconds()
    except Exception:
        features["flow_duration"] = 0.0

    # Basic packet/byte stats
    features["tot_fwd_pkts"] = flow_info.get("pkts_toserver", 0)
    features["tot_bwd_pkts"] = flow_info.get("pkts_toclient", 0)
    features["totlen_fwd_pkts"] = flow_info.get("bytes_toserver", 0)
    features["totlen_bwd_pkts"] = flow_info.get("bytes_toclient", 0)

    # Additional Suricata fields (optional, if you mapped them during training)
    # These will need to match your trained model's expectations:
    features["fwd_pkt_len_max"] = alert_json.get("flow", {}).get("fwd_pkt_len_max", 0)
    features["fwd_pkt_len_mean"] = alert_json.get("flow", {}).get("fwd_pkt_len_mean", 0)
    features["bwd_pkt_len_max"] = alert_json.get("flow", {}).get("bwd_pkt_len_max", 0)
    features["bwd_pkt_len_mean"] = alert_json.get("flow", {}).get("bwd_pkt_len_mean", 0)

    # Placeholder: Add default 0s for any features not yet parsed
    # Add as needed based on model training columns

    return features


def parse_suricata_file(json_path: Path) -> pd.DataFrame:
    """
    Load a Suricata JSON file and extract alerts into a feature DataFrame.
    Handles:
    - single JSON object
    - JSON array
    - line-delimited JSON (one alert per line)
    """
    records = []

    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            content = json.load(f)

        if isinstance(content, list):
            for alert in content:
                records.append(parse_single_alert(alert))
        elif isinstance(content, dict):
            records.append(parse_single_alert(content))
        else:
            print(f"Warning: {json_path} is not a valid dict or list. Skipping.")
    except json.JSONDecodeError:
        # Likely newline-delimited JSON
        with open(json_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    alert = json.loads(line.lstrip("\ufeff"))
                    records.append(parse_single_alert(alert))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {json_path}: {e}")
                    continue

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(prog="mlids-suricata-features")
    parser.add_argument(
        "--input-dir", "-i", required=True, help="Directory with Suricata JSON files"
    )
    parser.add_argument(
        "--output-csv", "-o", required=True, help="Path to save resulting CSV"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_dfs = []

    for json_file in input_dir.glob("*.json"):
        df = parse_suricata_file(json_file)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print(f"No usable alerts found in {input_dir}. Exiting.")
        return

    result_df = pd.concat(all_dfs, ignore_index=True)

    # If training expects fixed column order, enforce here:
    expected_columns = [
        "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts",
        "totlen_fwd_pkts", "totlen_bwd_pkts",
        "fwd_pkt_len_max", "fwd_pkt_len_mean",
        "bwd_pkt_len_max", "bwd_pkt_len_mean",
    ]
    for col in expected_columns:
        if col not in result_df:
            result_df[col] = 0
    result_df = result_df[expected_columns]

    result_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result_df)} records to {args.output_csv}")


if __name__ == "__main__":
    main()

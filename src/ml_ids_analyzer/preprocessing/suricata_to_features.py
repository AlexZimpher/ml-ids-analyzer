#!/usr/bin/env python3
"""
Parse Suricata alert logs into a structured feature CSV for ML-IDS-Analyzer.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import click

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def parse_single_alert(alert_json: dict) -> dict:
    """Extract relevant model features from a single Suricata alert JSON object."""
    features = {}

    flow_info = alert_json.get("flow", {})
    try:
        start = pd.to_datetime(flow_info.get("start"))
        end = pd.to_datetime(flow_info.get("end"))
        features["flow_duration"] = (end - start).total_seconds()
    except Exception:
        features["flow_duration"] = 0.0

    features["tot_fwd_pkts"] = flow_info.get("pkts_toserver", 0)
    features["tot_bwd_pkts"] = flow_info.get("pkts_toclient", 0)
    features["totlen_fwd_pkts"] = flow_info.get("bytes_toserver", 0)
    features["totlen_bwd_pkts"] = flow_info.get("bytes_toclient", 0)
    features["fwd_pkt_len_max"] = flow_info.get("fwd_pkt_len_max", 0)
    features["fwd_pkt_len_mean"] = flow_info.get("fwd_pkt_len_mean", 0)
    features["bwd_pkt_len_max"] = flow_info.get("bwd_pkt_len_max", 0)
    features["bwd_pkt_len_mean"] = flow_info.get("bwd_pkt_len_mean", 0)

    return features


def parse_suricata_file(json_path: Path) -> pd.DataFrame:
    """Load a Suricata JSON file and extract alerts into a feature DataFrame."""
    records = []

    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            content = json.load(f)

        if isinstance(content, list):
            records.extend(parse_single_alert(alert) for alert in content)
        elif isinstance(content, dict):
            records.append(parse_single_alert(content))
        else:
            logging.warning(f"{json_path} is not a valid dict or list. Skipping.")
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
                    logging.warning(f"Skipping malformed line in {json_path}: {e}")
                    continue

    return pd.DataFrame.from_records(records)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False),
    help="Directory containing Suricata JSON files."
)
@click.option(
    "-o", "--output-csv", required=True, type=click.Path(writable=True),
    help="Path to save the resulting feature CSV."
)
def main(input_dir: str, output_csv: str):
    """Extract Suricata JSON alerts into structured CSV features."""
    input_path = Path(input_dir)
    all_dfs = []

    for json_file in input_path.glob("*.json"):
        df = parse_suricata_file(json_file)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logging.error(f"No usable alerts found in {input_dir}. Exiting.")
        return

    result_df = pd.concat(all_dfs, ignore_index=True)

    # Derived Features
    result_df["flow_bytes_s"] = (
        (result_df["totlen_fwd_pkts"] + result_df["totlen_bwd_pkts"]) /
        result_df["flow_duration"].replace({0: pd.NA})
    ).fillna(0.0)

    result_df["flow_pkts_s"] = (
        (result_df["tot_fwd_pkts"] + result_df["tot_bwd_pkts"]) /
        result_df["flow_duration"].replace({0: pd.NA})
    ).fillna(0.0)

    # Column renaming to match model
    column_map = {
        "flow_duration": "Flow Duration",
        "tot_fwd_pkts": "Total Fwd Packets",
        "tot_bwd_pkts": "Total Backward Packets",
        "totlen_fwd_pkts": "Total Length of Fwd Packets",
        "totlen_bwd_pkts": "Total Length of Bwd Packets",
        "fwd_pkt_len_max": "Fwd Packet Length Max",
        "fwd_pkt_len_mean": "Fwd Packet Length Mean",
        "bwd_pkt_len_max": "Bwd Packet Length Max",
        "bwd_pkt_len_mean": "Bwd Packet Length Mean",
        "flow_bytes_s": "Flow Bytes/s",
        "flow_pkts_s": "Flow Packets/s"
    }
    result_df.rename(columns=column_map, inplace=True)

    # Enforce expected column order
    expected_columns = list(column_map.values())
    for col in expected_columns:
        if col not in result_df:
            result_df[col] = 0
    result_df = result_df[expected_columns]

    result_df.to_csv(output_csv, index=False)
    logging.info("Wrote %d records to %s", len(result_df), output_csv)


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import pandas as pd

def parse_suricata_alerts(log_file_path: str) -> pd.DataFrame:
    """
    Parse a Suricata EVE JSON alert log into a DataFrame.
    Only extracts fields needed for ML processing.
    """
    alerts = []
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("event_type") != "alert":
                    continue

                flow = entry.get("flow", {})
                alert = entry.get("alert", {})

                alerts.append({
                    "src_ip": entry.get("src_ip"),
                    "dest_ip": entry.get("dest_ip"),
                    "proto": entry.get("proto"),
                    "alert_signature": alert.get("signature"),
                    "alert_category": alert.get("category"),
                    "alert_severity": alert.get("severity"),
                    "flow_start": flow.get("start"),
                    "flow_end": flow.get("end"),
                    "bytes_toserver": flow.get("bytes_toserver"),
                    "bytes_toclient": flow.get("bytes_toclient"),
                    "pkts_toserver": flow.get("pkts_toserver"),
                    "pkts_toclient": flow.get("pkts_toclient"),
                })
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(alerts)

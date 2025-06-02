import json
import tempfile
import pandas as pd
from ml_ids_analyzer.preprocessing.suricata_to_features import parse_single_alert, parse_suricata_file

def test_parse_single_alert_basic():
    alert = {
        "flow": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T00:00:01Z",
            "bytes_toserver": 500,
            "bytes_toclient": 1000,
            "pkts_toserver": 5,
            "pkts_toclient": 10,
        },
        "alert": {"severity": 2}
    }
    row = parse_single_alert(alert)
    print("\n\nReturned keys:", list(row.keys()))
    assert row["flow_duration"] == 1.0


def test_parse_suricata_file_json_array():
    data = [{
        "flow": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T00:00:02Z",
            "bytes_toserver": 200,
            "bytes_toclient": 300,
            "pkts_toserver": 2,
            "pkts_toclient": 3,
        },
        "alert": {"severity": 3}
    }]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.seek(0)
        df = parse_suricata_file(f.name)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "flow_duration" in df.columns
    assert df.iloc[0]["flow_duration"] == 2.0

import subprocess
import tempfile
from pathlib import Path
from ml_ids_analyzer.config import cfg
import os

def test_mlids_train_cli_runs_no_search():
    # Create temporary output dir
    temp_dir = Path(tempfile.mkdtemp())
    original_output_dir = cfg["paths"]["output_dir"]
    cfg["paths"]["output_dir"] = temp_dir

    try:
        env = os.environ.copy()
        env["MLIDS_OUTPUT_DIR"] = str(temp_dir)

        result = subprocess.run(
            ["poetry", "run", "mlids-train", "--no-search"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        assert (temp_dir / "random_forest_model.joblib").exists()
    finally:
        cfg["paths"]["output_dir"] = original_output_dir

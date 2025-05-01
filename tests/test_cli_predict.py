import pandas as pd
import pytest
import joblib
from click.testing import CliRunner

from ml_ids_analyzer.inference.predict import main as predict_main


@pytest.fixture
def dummy_env(tmp_path):
    # 1. Tiny input CSV
    df = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
    inp = tmp_path / "in.csv"
    df.to_csv(inp, index=False)

    # 2. Fit & dump dummy scaler + model
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    X = df.values
    y = [0, 1]
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
    model.fit(scaler.transform(X), y)

    mfile = tmp_path / "model.joblib"
    sfile = tmp_path / "scaler.joblib"
    joblib.dump(model, mfile)
    joblib.dump(scaler, sfile)

    out = tmp_path / "out.csv"
    return {"inp": str(inp), "model": str(mfile), "scaler": str(sfile), "out": str(out)}


def test_predict_cli(dummy_env):
    runner = CliRunner()
    opts = dummy_env
    result = runner.invoke(
        predict_main,
        [
            "--input-file",
            opts["inp"],
            "--model-file",
            opts["model"],
            "--scaler-file",
            opts["scaler"],
            "--output-file",
            opts["out"],
            "--threshold",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output

    df_out = pd.read_csv(opts["out"])
    # Expect original features + two new cols
    assert set(df_out.columns) == {"f1", "f2", "prob_attack", "pred_attack"}
    # Predictions should mirror y = [0,1]
    assert df_out["pred_attack"].tolist() == [0, 1]

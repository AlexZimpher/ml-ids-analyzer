# Test importability and syntax of the dashboard script
from pathlib import Path


def test_dashboard_imports():
    """Test that the dashboard script is importable and has no syntax errors."""
    import importlib.util
    # Get dashboard.py path
    dashboard_path = str(
        Path(__file__).parent.parent.parent
        / "src"
        / "ml_ids_analyzer"
        / "api"
        / "dashboard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dashboard", dashboard_path
    )
    assert spec is not None, (
        f"Could not create import spec for {dashboard_path}"
    )
    dashboard = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"No loader for spec {spec}"
    spec.loader.exec_module(dashboard)

import click
from ml_ids_analyzer.modeling.train import main as train_main
from ml_ids_analyzer.dashboard import main as dashboard_main

@click.group()
def cli():
    pass

cli.add_command(train_main, name="train")

@cli.command("dashboard")
def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run(["streamlit", "run", "src/ml_ids_analyzer/dashboard.py"])

if __name__ == "__main__":
    cli()

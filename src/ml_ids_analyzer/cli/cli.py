"""
CLI entry point for ml_ids_analyzer.
"""

import click
from ml_ids_analyzer.modeling.train import train_model


# Define the main CLI group
@click.group()
def cli():
    """Main CLI group for ml_ids_analyzer commands."""
    pass


# Add the 'train' command
@cli.command()
def train():
    """Train the model."""
    train_model()


# Add the 'dashboard' command
@cli.command("dashboard")
def launch_dashboard():
    """Launch the Streamlit dashboard (safe subprocess call)."""
    import subprocess
    import sys
    import os

    dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api", "dashboard.py"))
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path
    ], check=True)


# Run the CLI if this file is executed directly
if __name__ == "__main__":
    cli()

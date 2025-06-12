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
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run(["streamlit", "run", "src/ml_ids_analyzer/dashboard.py"])

# Run the CLI if this file is executed directly
if __name__ == "__main__":
    cli()

# tests/test_cli.py

import sys
from unittest.mock import MagicMock

import ml_ids_analyzer.model as model_mod
import ml_ids_analyzer.modeling.train as train_mod


def test_model_main_invokes_train_model(monkeypatch):
    """
    Ensure that calling ml_ids_analyzer.model.main()
    simply calls through to train_model().
    """
    mock_train = MagicMock()
    # Replace train_model in the model module
    monkeypatch.setattr(model_mod, "train_model", mock_train)

    # Call the CLI entry point
    model_mod.main()

    # Verify it was invoked exactly once, with no arguments
    mock_train.assert_called_once_with()


def test_train_main_parses_no_search_flag(monkeypatch):
    """
    Ensure that modeling/train.py’s main()
    parses the --no-search flag and passes it correctly.
    """
    mock_train = MagicMock()
    # Replace train_model in the train module
    monkeypatch.setattr(train_mod, "train_model", mock_train)

    # Simulate CLI args: script name + --no-search
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-search"])

    # Invoke the argparse-driven main()
    train_mod.main()

    # Should have been called with no_search=True
    mock_train.assert_called_once_with(no_search=True)

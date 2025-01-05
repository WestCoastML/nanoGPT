import os
import json
import pytest
import tempfile
import subprocess
import torch

from unittest.mock import patch, MagicMock
from pathlib import Path
import datetime

############################################
# Utility: create a mock run directory
############################################

def create_mock_run_dir(base_dir, run_id, config=None):
    """
    Creates a mock run directory with:
    - config.json
    - checkpoints/latest.pt (with 'timestamp' to avoid KeyError)
    """
    run_dir = base_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Default config if none provided
    if config is None:
        config = {
            "max_iters": 1000,
            "learning_rate": 0.001,
            "model_architecture": "original",
            "n_layer": 12,
            "layer_dims": [768, 768, 768],
            # Potentially add 'shape_variant' too if your code references it
            "shape_variant": "none",
        }

    # Write config.json
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create a checkpoint with all required keys, including top-level 'timestamp'
    # If you need best_val_loss, set it here: 'best_val_loss': 2.5 or similar
    checkpoint = {
        "state_dict": {
            "model": {"test_weight": torch.randn(2, 2)},
            "optimizer": {"test_optim_weight": torch.randn(2, 2)},
        },
        "metadata": {
            "iteration": 500,
            "config": config,
            # "best_val_loss": 2.5,  # If you need best_val_loss in the checkpoint
        },
        # Must be top-level
        "timestamp": datetime.datetime.now().isoformat(),
    }

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoints_dir / "latest.pt")

    return run_dir, config

############################################
# Tests
############################################

def test_output_directory_structure():
    """
    Option Two fix:
    Pass the 'wandb_run' argument into setup_output_dir so that 
    we do get a path ending with 'run_{mock_run.id}', e.g. run_test123.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        from train import setup_output_dir

        # Mock wandb.run
        with patch("wandb.run") as mock_run:
            mock_run.id = "test123"
            mock_run.resumed = False

            config = {"out_dir": tmpdir}
            # By passing wandb_run=mock_run, we ensure we create 'run_test123'
            out_dir = setup_output_dir(config, master_process=True, wandb_run=mock_run)

            # Check directory structure
            assert os.path.isdir(out_dir)
            assert os.path.isfile(os.path.join(out_dir, "config.json"))
            # Should indeed end with run_test123
            assert str(out_dir).endswith("run_test123")


def test_run_continuation():
    """Test that run continuation loads config correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_id = "test123"

        # Create mock run directory with config + checkpoint
        run_dir, orig_config = create_mock_run_dir(tmp_path, run_id)

        # We'll import load_training_state + CheckpointManager from training_utils
        from utils.training_utils import load_training_state, CheckpointManager

        model = MagicMock()
        optimizer = MagicMock()

        ckp_path = run_dir / "checkpoints" / "latest.pt"
        # Provide a real manager so we don't get 'NoneType' error
        manager = CheckpointManager(run_dir / "checkpoints")

        # Fix: Unpack all 4 return values
        loaded_model, loaded_optimizer, iter_num, best_val = load_training_state(
            checkpoint_manager=manager,
            model=model,
            optimizer=optimizer,
            device="cpu",
            config={},  # If we need config matching, set it here
            ddp=False
        )

        assert iter_num == 500, f"Expected iteration=500, got {iter_num}"


@pytest.fixture
def mock_sweep_env():
    """
    Pytest fixture that sets up a typical environment with:
      - base_dir (temp directory)
      - run_dir (with run_{mock_run_456})
      - config
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sweep_id = "mock_sweep_123"
        run_id = "mock_run_456"

        base_dir = tmp_path / "runs" / "sweeps" / sweep_id
        base_dir.mkdir(parents=True, exist_ok=True)

        run_dir, config = create_mock_run_dir(base_dir, run_id)
        yield {
            "base_dir": base_dir,
            "sweep_id": sweep_id,
            "run_id": run_id,
            "run_dir": run_dir,
            "config": config,
        }


def test_sweep_continuation(mock_sweep_env):
    """
    Test continuing a sweep with new max_iters using train.py's module-level globals patching.
    """
    import os
    import sys
    import datetime
    import json
    import torch
    import train  # Make sure train.py is loaded
    from unittest.mock import patch, MagicMock
    from utils.continue_run import main
    from utils.training_utils import CheckpointManager
    from model import GPT, GPTConfig
    from config.defaults import get_default_config

    # Start with default config to ensure all necessary keys exist
    base_config = get_default_config()
    
    # Complete model arguments setup
    model_args = {
        "block_size": 1024,
        "vocab_size": 50304,
        "n_layer": 12,
        "layer_dims": [768] * 12,
        "n_heads": [12] * 12,
        "dropout": 0.0,
        "bias": True,
        "model_architecture": "original",
        "use_unet": False
    }

    # DDP configuration
    ddp_config = {
        "backend": "nccl",
        "init_method": "env://",
        "world_size": 1,
        "rank": 0
    }

    base_config.update({
        "model_args": model_args,
        "vocab_size": 50304,
        "ddp": ddp_config,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float32",
        "compile": False,
        "gradient_accumulation_steps": 1,
        "max_iters": 1000,
        "lr_decay_iters": 1000,
        "learning_rate": 0.001,
        "init_from": "scratch",
        "wandb_log": False
    })

    # Update mock_sweep_env config with complete configuration
    mock_sweep_env["config"].update(base_config)

    run_dir = mock_sweep_env["run_dir"]
    run_id = mock_sweep_env["run_id"]

    # Create an actual model instance to get a valid state dict
    model = GPT(GPTConfig(**model_args))
    model.to(mock_sweep_env["config"]["device"])

    # Create complete optimizer state
    optimizer_state = {
        "state": {},
        "param_groups": [{
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "params": list(range(len(list(model.parameters()))))
        }]
    }

    # Create checkpoint directory
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create complete checkpoint with actual model state dict and metadata
    mock_checkpoint = {
        "state_dict": {
            "model": model.state_dict(),
            "optimizer": optimizer_state
        },
        "metadata": {
            "iteration": 500,
            "config": mock_sweep_env["config"],
            "model_args": model_args,
            "best_val_loss": 2.5,
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "latest.pt"
    torch.save(mock_checkpoint, checkpoint_path)

    # Verify checkpoint was created
    assert checkpoint_path.exists(), f"Checkpoint not created at {checkpoint_path}"

    # Test checkpoint can be loaded
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    test_load = checkpoint_manager.load_checkpoint(map_location=mock_sweep_env["config"]["device"])
    assert test_load is not None, "Failed to load test checkpoint"

    # Create continuation config with complete parameters
    new_config = base_config.copy()
    new_config.update({
        "max_iters": 2000,
        "lr_decay_iters": 2000,
        "learning_rate": 0.0005,
    })

    # Create config file in the run directory
    temp_config_path = os.path.join(run_dir, "continue_config.json")
    with open(temp_config_path, "w") as f:
        json.dump(new_config, f)

    # Mock wandb API and init
    with patch("wandb.Api") as mock_api:
        mock_run = MagicMock()
        mock_run.id = run_id
        mock_run.config = mock_sweep_env["config"]
        mock_run.sweep = MagicMock()
        mock_run.sweep.id = mock_sweep_env["sweep_id"]
        mock_api_instance = MagicMock()
        mock_api_instance.run.return_value = mock_run
        mock_api.return_value = mock_api_instance

        with patch("wandb.init") as mock_init:
            mock_init.return_value = mock_run

            # Set up test arguments
            test_argv = [
                "utils/continue_run.py",
                "--run_id", run_id,
                "--config", str(temp_config_path),
                "--gpus", "0",
                "--out_dir", str(mock_sweep_env["base_dir"].parents[1])
            ]

            # Patch both sys.argv and train.py's module-level globals
            with patch.object(sys, "argv", test_argv):
                # Patch train.py's module-level globals so configurator sees them
                with patch.dict(sys.modules["train"].__dict__, {
                    "max_iters": 9999,
                    "lr_decay_iters": 9999,
                    "learning_rate": 0.0005,
                }, clear=False):
                    exit_code = main()
                    assert exit_code == 0

            # Verify wandb logging was attempted
            mock_init.assert_called_once()


def test_checkpoint_loading(mock_sweep_env):
    """
    Test checkpoint loading and validation logic (including top-level 'timestamp').
    Also unify the usage to 'latest.pt' if we rely on that naming.
    """
    from utils.training_utils import CheckpointManager

    checkpoint_dir = mock_sweep_env["run_dir"] / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)

    ckp = manager.load_checkpoint()
    assert ckp["metadata"]["iteration"] == 500
    assert "timestamp" in ckp, "Checkpoint must contain a top-level 'timestamp' key"


def test_wandb_continuation(mock_sweep_env):
    """Test wandb logging for an updated run config"""
    from utils.wandb_logger import WandBLogger

    config = dict(mock_sweep_env["config"])
    config["shape_variant"] = config.get("shape_variant", "none")

    with patch("wandb.init") as mock_init:
        # Setup mock run with proper context manager behavior
        mock_run = MagicMock()
        mock_run.id = mock_sweep_env["run_id"]
        mock_run.__enter__ = lambda x: mock_run
        mock_run.__exit__ = lambda x, y, z, w: None
        mock_init.return_value = mock_run

        # Initialize logger
        logger = WandBLogger(
            config=config,
            project="test_project",
            entity="test_entity",
            name=f"continued_{mock_sweep_env['run_id']}",
        )

        # Test logging
        logger.log_training_step(501, {"loss": 2.4, "lr": 0.0005, "mfu": 0.5})

        # Verify finish was called
        logger.finish()
        assert mock_run.finish.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

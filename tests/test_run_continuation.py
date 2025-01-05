import os
import pytest
import tempfile
import json
import torch
from unittest.mock import patch, MagicMock
from utils.continue_run import create_ddp_command, create_single_gpu_command

def test_create_ddp_command():
    """Test DDP command creation"""
    args = MagicMock(
        run_id="test123",
        gpus="0,1,2,3",
        max_iters=2000
    )
    
    config = {
        'max_iters': 1000,
        'learning_rate': 0.001
    }
    
    cmd, env = create_ddp_command(args, config, "test_dir")
    
    # Verify command structure
    assert "torchrun" in cmd
    assert "--standalone" in cmd
    assert "--nproc_per_node" in cmd
    assert "4" in cmd  # Should have 4 GPUs
    assert env['WANDB_RUN_ID'] == "test123"
    assert env['RESUME_RUN'] == '1'

def test_create_single_gpu_command():
    """Test single GPU command creation"""
    args = MagicMock(
        run_id="test123",
        gpus="0",
        max_iters=2000
    )
    
    config = {
        'max_iters': 1000,
        'learning_rate': 0.001
    }
    
    cmd, env = create_single_gpu_command(args, config, "test_dir")
    
    # Verify command structure
    assert "python" in cmd
    assert "train.py" in cmd
    assert env['WANDB_RUN_ID'] == "test123"
    assert env['CUDA_VISIBLE_DEVICES'] == "0"

def test_config_updates():
    """Test configuration update mechanism"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create original config
        config = {
            'max_iters': 1000,
            'learning_rate': 0.001,
            'model_args': {
                'n_layer': 12,
                'n_head': 12
            }
        }
        
        # Create updates
        updates = {
            'max_iters': 2000,
            'learning_rate': 0.0005
        }
        
        update_file = os.path.join(tmpdir, 'updates.json')
        with open(update_file, 'w') as f:
            json.dump(updates, f)
        
        # Mock args
        args = MagicMock(
            config=update_file,
            max_iters=None,
            learning_rate=None
        )
        
        # Test update logic
        from utils.continue_run import main
        with patch('utils.continue_run.subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock(
                stdout=iter([b"test output"]),
                wait=lambda: 0
            )
            
            # This should not raise any errors
            main(args)
            
            # Verify config was updated
            assert config['max_iters'] == 2000
            assert config['learning_rate'] == 0.0005
            # Original model args should be preserved
            assert config['model_args']['n_layer'] == 12
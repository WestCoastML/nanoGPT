import os
import pytest
import tempfile
import torch

def create_mock_checkpoint(path, iter_num=500):
    """Create a mock checkpoint for testing"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'iter_num': iter_num,
        'model_state': {'test': torch.randn(10, 10)},
        'optimizer_state': {'test': torch.randn(5, 5)},
        'model_args': {
            'n_layer': 12,
            'n_head': 12
        }
    }
    torch.save(checkpoint, path)
    return checkpoint

def create_mock_config(path, config=None):
    """Create a mock config file for testing"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if config is None:
        config = {
            'max_iters': 1000,
            'learning_rate': 0.001,
            'model_args': {
                'n_layer': 12,
                'n_head': 12
            }
        }
    with open(path, 'w') as f:
        json.dump(config, f)
    return config

def create_mock_run_dir(base_dir, run_id="test123", iter_num=500):
    """Create a complete mock run directory for testing"""
    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create config
    config = create_mock_config(os.path.join(run_dir, 'config.json'))
    
    # Create checkpoint
    checkpoint = create_mock_checkpoint(os.path.join(run_dir, 'ckpt.pt'), iter_num)
    
    return run_dir, config, checkpoint
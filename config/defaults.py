"""Default configuration values for training."""

def get_default_config():
    """Get default configuration values."""
    return {
        # Output and logging
        'out_dir': 'out',
        'eval_interval': 2000,
        'log_interval': 1,
        'eval_iters': 200,
        'eval_only': False,
        'always_save_checkpoint': True,
        
        # Model initialization
        'init_from': 'scratch',  # 'scratch', 'resume', or 'gpt2'
        
        # Wandb settings
        'wandb_log': False,
        'wandb_project': 'owt',
        'wandb_entity': "",
        'wandb_group': None,
        'wandb_run_name': 'gpt2',
        
        # Dataset configuration
        'datasets': {"openwebtext":0.4, "wikipedia":0.3, "books":0.2, "tinystories":0.1},  # Multi-dataset weights
        'dataset': 'openwebtext',  # Single dataset fallback
        
        # Training hyperparameters
        'gradient_accumulation_steps': 40,
        'batch_size': 12,
        'block_size': 1024,
        'n_layer': 12,
        'dropout': 0.0,
        'bias': False,
        'learning_rate': 6e-4,
        'max_iters': 600000,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'decay_lr': True,
        'warmup_iters': 2000,
        'lr_decay_iters': 600000,
        'min_lr': 6e-5,
        
        # System settings
        'backend': 'nccl',
        'device': 'cuda',
        'dtype': 'bfloat16',  # Will fall back to float16 if bfloat16 not supported
        'compile': True,
        
        # Architecture settings
        'model_architecture': 'original',
        'use_unet': False,
        'base_dim': 384,
        'max_dim': 1024,
        'head_dim': 64,
        'shape_variant': 'symmetry',
        'n_dims': None  # Overrides architecture-based dims if provided
    }
"""
Utility for continuing training runs with updated parameters.
Supports both single-GPU and DDP modes.
"""

import os
import json
import wandb
import torch
import logging
import argparse
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .training_utils import CheckpointManager, validate_checkpoint_compatibility
from model import GPT, GPTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Continue a wandb run with modified parameters')
    parser.add_argument('--run_id', required=True, help='Wandb run ID to continue')
    parser.add_argument('--max-iters', type=int, help='New max iterations')
    parser.add_argument('--learning-rate', type=float, help='New learning rate')
    parser.add_argument('--config', type=str, help='JSON file with multiple parameter updates')
    parser.add_argument('--gpus', type=str, default='0', 
                       help='Comma-separated GPU IDs for DDP, or single GPU ID')
    parser.add_argument('--ddp', action='store_true', help='Use DDP mode')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to load')
    parser.add_argument('--out_dir', type=str, help='Output directory for runs')
    return parser.parse_args()

def load_and_validate_config(original_config: Dict[str, Any], 
                           updates: Dict[str, Any]) -> Dict[str, Any]:
    """Load and validate configuration updates"""
    # Import at function level to avoid circular imports
    from config.defaults import get_default_config
    from utils.training_utils import validate_config_compatibility
    
    # Deep copy original config
    config = json.loads(json.dumps(original_config))
    
    # Apply updates to create new config
    new_config = json.loads(json.dumps(config))
    for key, value in updates.items():
        new_config[key] = value
    
    # Validate compatibility between old and new configs
    if not validate_config_compatibility(config, new_config):
        raise ValueError("New configuration is not compatible with original checkpoint configuration")
        
    # Handle datasets configuration
    if 'datasets' in new_config:
        if isinstance(new_config['datasets'], str):
            try:
                new_config['datasets'] = json.loads(new_config['datasets'])
            except json.JSONDecodeError:
                logger.warning("Invalid datasets JSON string in new config, keeping original datasets")
                new_config['datasets'] = config.get('datasets', {"openwebtext": 1.0})
    
    # Get default config for additional validation
    default_config = get_default_config()
    
    # Validate all keys against default config
    for key in new_config:
        if key not in default_config and key not in ['model_args', 'run_id', 'sweep_id']:
            logger.warning(f"Adding new config parameter: {key}")
    
    return new_config

def setup_environment(run_id: str, 
                     gpus: str, 
                     use_ddp: bool = False) -> Dict[str, str]:
    """Setup runtime environment variables"""
    env = os.environ.copy()
    
    # Basic setup
    env.update({
        'WANDB_RESUME': 'allow',
        'WANDB_RUN_ID': run_id,
        'RESUME_RUN': '1',
        'CUDA_VISIBLE_DEVICES': gpus,
    })
    
    # DDP-specific setup
    if use_ddp:
        env.update({
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': str(29500 + hash(run_id) % 1000),  # Unique port based on run_id
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_TIMEOUT': '23',
            'NCCL_SOCKET_TIMEOUT': '120',
            'USE_DDP': '1',
            'NUM_GPUS': str(len(gpus.split(',')))
        })
        
    return env

def create_command(config: Dict[str, Any],
                  run_dir: str,
                  use_ddp: bool = False,
                  gpus: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """Create command for running training"""
    env = setup_environment(config['run_id'], gpus, use_ddp)
    
    # Base command
    if use_ddp:
        num_gpus = len(gpus.split(','))
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node", str(num_gpus),
            "train.py"
        ]
    else:
        cmd = ["python", "train.py"]
        
    # Add config parameters
    for k, v in config.items():
        if isinstance(v, (list, dict)):
            cmd.append(f"--{k}='{json.dumps(v)}'")
            if k == 'datasets':
                logger.info(f"Using datasets configuration: {v}")
        else:
            cmd.append(f("--{k}={v}")
            
    return cmd, env

def run_training(cmd: List[str], 
                env: Dict[str, str], 
                run_dir: str) -> int:
    """Run training process with proper handling"""
    
    def handle_signal(signum, frame):
        if process:
            process.send_signal(signum)
    
    # Setup signal handling
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, handle_signal)
        
    # Start process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        start_new_session=True  # New process group
    )
    
    try:
        # Stream output with timestamps
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {line}", end='', flush=True)
            
        return process.wait()
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
        
    finally:
        # Restore signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, signal.SIG_DFL)

def load_and_validate_model(run_dir: str, config: Dict[str, Any], device: str) -> Tuple[GPT, int, float]:
    """Load model from checkpoint with validation"""
    checkpoint_manager = CheckpointManager(Path(run_dir) / "checkpoints")
    
    try:
        # Initialize model with config
        model_args = config.get('model_args', {})
        model = GPT(GPTConfig(**model_args))
        model.to(device)
        
        # Load checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(map_location=device)
        
        # Validate checkpoint compatibility
        if not validate_checkpoint_compatibility(checkpoint, model, config):
            raise ValueError("Checkpoint not compatible with current configuration")
        
        # Load state
        model.load_state_dict(checkpoint['state_dict']['model'])
        iter_num = checkpoint['metadata'].get('iteration', 0)
        best_val_loss = checkpoint['metadata'].get('best_val_loss', float('inf'))
        
        return model, iter_num, best_val_loss
        
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def determine_run_directory(config: Dict[str, Any], out_dir: Optional[str] = None) -> Path:
    """Determine the run directory based on config and out_dir."""
    if out_dir:
        # If out_dir is specified, use it as base
        base_dir = Path(out_dir)
    else:
        # Default to 'runs' directory
        base_dir = Path('runs')
    
    # If we have a sweep_id, include it in the path
    if 'sweep_id' in config:
        run_dir = base_dir / 'sweeps' / config['sweep_id'] / f"run_{config['run_id']}"
    else:
        run_dir = base_dir / 'single' / f"run_{config['run_id']}"
    
    return run_dir

def main():
    args = parse_args()
    
    # Initialize wandb API
    api = wandb.Api()
    try:
        original_run = api.run(f"wcml/VSLM/{args.run_id}")
        original_config = original_run.config
    except wandb.CommError as e:
        logger.error(f"Failed to fetch run {args.run_id}: {e}")
        raise

    # Load and validate config
    config_updates = {}
    if args.max_iters:
        config_updates['max_iters'] = args.max_iters
        config_updates['lr_decay_iters'] = args.max_iters
    if args.learning_rate:
        config_updates['learning_rate'] = args.learning_rate
    if args.config:
        with open(args.config) as f:
            file_updates = json.load(f)
            config_updates.update(file_updates)
            
    config = load_and_validate_config(original_config, config_updates)
    
    # Add sweep info to config if available
    if original_run.sweep:
        config['sweep_id'] = original_run.sweep.id
    config['run_id'] = args.run_id
    
    # Determine run directory using out_dir if specified
    run_dir = determine_run_directory(config, args.out_dir)
    
    # Load model and state
    device = f"cuda:{args.gpus.split(',')[0]}" if torch.cuda.is_available() else "cpu"
    model, iter_num, best_val_loss = load_and_validate_model(run_dir, config, device)
    
    # Update config with current state
    config.update({
        'init_from': 'resume',
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'resume_run_id': args.run_id
    })
    
    # Potential fix: if you want the run to see ddp or not, we can do:
    if args.ddp:
        config["ddp"] = True
    else:
        config["ddp"] = False
    # This ensures train.py can see cfg.ddp

    # Create command and environment
    cmd, env = create_command(config, str(run_dir), args.ddp, args.gpus)
    
    # Log command
    logger.info("Executing command: %s", " ".join(cmd))
    
    # Run training
    try:
        exit_code = run_training(cmd, env, str(run_dir))
        # If the training process returns 0, consider that success:
        if exit_code == 0:
            return 0
        else:
            # Otherwise propagate or convert nonzero to 1
            return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
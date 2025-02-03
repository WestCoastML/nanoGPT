"""
Utility for continuing training runs with updated parameters.
Supports both single-GPU and DDP modes.

 USAGE NOTE:
   - If you want to run this script directly (e.g. `python utils/continue_run.py ...`),
     you must be in the project root (`nanoGPT/`) and set `PYTHONPATH=.`
     so Python recognizes `utils/` as a top-level package:

         cd /path/to/nanoGPT
         PYTHONPATH=. python utils/continue_run.py --run_id <RUN_ID> ...

   - Alternatively, run it as a module:

         cd /path/to/nanoGPT
         python -m utils.continue_run --run_id <RUN_ID> ...
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

from .training_utils import (
    # Optionally import if you need these:
    CheckpointManager,
    validate_checkpoint_compatibility,
)
# If model.py is one directory above `utils/`, you can do either:
#   from ..model import GPT, GPTConfig
# or keep an absolute import if your top-level folder is on PYTHONPATH
from model import GPT, GPTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Continue a wandb run with modified parameters')
    parser.add_argument('--run_id', required=True, help='Wandb run ID to continue')
    parser.add_argument('--max-iters', type=int, help='New max iterations')
    parser.add_argument('--learning-rate', type=float, help='New learning rate')
    parser.add_argument('--config', type=str, help='JSON string or file with multiple parameter updates')
    parser.add_argument('--gpus', type=str, default='0', 
                       help='Comma-separated GPU IDs for DDP, or single GPU ID')
    parser.add_argument('--ddp', action='store_true', help='Use DDP mode')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="""
Path to the checkpoint from which to continue (mandatory).
If this is a directory, we will attempt to load <directory>/latest.pt
If it is a file, we will load exactly that file.
""")
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
            # cmd.append(f("--{k}={v}")
            cmd.append(f"--{k}={v}")
            
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

def resolve_checkpoint_path(checkpoint_arg: str) -> str:
    """
    If user supplies a directory path for --checkpoint, automatically
    look for 'latest.pt' in that directory. If user supplies a file path,
    load that file directly.
    """
    import os
    if os.path.isdir(checkpoint_arg):
        candidate = os.path.join(checkpoint_arg, 'latest.pt')
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_arg}' does not contain 'latest.pt'."
            )
        return candidate
    else:
        # If not a directory, assume it's a file
        if not os.path.isfile(checkpoint_arg):
            raise FileNotFoundError(
                f"Checkpoint file '{checkpoint_arg}' not found."
            )
        return checkpoint_arg

def load_and_validate_model(run_dir: str,
                            config: Dict[str, Any],
                            device: str,
                            checkpoint_path: str) -> Tuple[GPT, int, float]:
    """
    Load the model from the user-specified checkpoint path (or auto-resolved).
    Bypass the normal fallback to latest/best in the CheckpointManager, so there's no confusion.
    """
    import torch
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find checkpoint file: {checkpoint_path}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")

    metadata      = checkpoint_data.get('metadata', {})
    old_config    = metadata.get('config', {})
    old_model_args = old_config.get('model_args', {})

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} with "
        f"iteration={metadata.get('iteration', 0)} "
        f"best_val_loss={metadata.get('best_val_loss', float('inf'))}"
    )

    # 2) Build new_config by merging original checkpoint config with CLI updates
    new_config = dict(old_config)  # shallow copy from checkpoint
    for key, val in config.items():
        if key != "model_args":
            new_config[key] = val

    # 3) Merge model_args carefully
    new_model_args = dict(old_model_args)
    cli_model_args = config.get('model_args', {})
    for k, v in cli_model_args.items():
        new_model_args[k] = v

    # If user didn't override layer_dims/n_heads, keep the old
    if "layer_dims" not in new_model_args and "layer_dims" in old_model_args:
        new_model_args["layer_dims"] = old_model_args["layer_dims"]
    if "n_heads" not in new_model_args and "n_heads" in old_model_args:
        new_model_args["n_heads"] = old_model_args["n_heads"]

    new_config["model_args"] = new_model_args

    # ---------------------------------------------------------------------
    # Force the new config to use the old block_size, vocab_size, bias, etc.
    # if we see them in the old config. This ensures shapes match exactly.
    # You can comment out whichever you do NOT want forced.
    # (Of course, if you intentionally want to use a different shape, skip this!)
    # 
    # For example, if old_config had block_size=256, vocab_size=50257, bias=false:
    old_block_size = old_config.get('block_size', None)
    if old_block_size is not None:
        logger.info(f"Forcing new model block_size to {old_block_size} to match old checkpoint.")
        new_config["block_size"] = old_block_size
        new_model_args["block_size"] = old_block_size

    old_bias = old_config.get('bias', None)
    if old_bias is not None:
        logger.info(f"Forcing new model bias={old_bias} to match old checkpoint.")
        new_config["bias"] = old_bias
        new_model_args["bias"] = old_bias

    old_vocab_size = old_model_args.get('vocab_size', None)
    if old_vocab_size is not None:
        logger.info(f"Forcing new model vocab_size to {old_vocab_size} to match old checkpoint.")
        new_model_args["vocab_size"] = old_vocab_size

    # For shaping the attention bias, also check if old_config had a certain 'n_layer' or something else
    # but typically the main mismatch is block_size, bias, vocab_size.
    # ---------------------------------------------------------------------

    # 4) Create the model from the final merged config
    model = GPT(GPTConfig(**new_config["model_args"]))
    model.to(device)

    # 5) Load iteration/best_val_loss from checkpoint metadata
    iter_num = metadata.get("iteration", 0)
    best_val_loss = metadata.get("best_val_loss", float('inf'))

    # 6) Actually load model state dict
    orig_sd = checkpoint_data["state_dict"]["model"]

    # --- NEW LOGIC: Remove any `_orig_mod.` or `module.` prefix from checkpoint keys --- #
    # Torch 2.0's `torch.compile` can store model weights under "_orig_mod."
    # Also if you used DDP, you might have "module." prefix.
    # We'll remove both if present:
    renamed_sd = {}
    for k, v in orig_sd.items():
        new_k = k
        # If there's a "_orig_mod." prefix, remove it
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        # If there's a "module." prefix, remove it
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        renamed_sd[new_k] = v

    # Now load the renamed keys into our model
    model.load_state_dict(renamed_sd)
    # --- End new logic --- #

    return model, iter_num, best_val_loss

def determine_run_directory(config: Dict[str, Any], out_dir: Optional[str] = None) -> Path:
    """
    Decide on a new run directory for continuing. E.g. create 'checkpoints_continued'.
    """
    if out_dir:
        base_dir = Path(out_dir)
    else:
        base_dir = Path('runs')

    if 'sweep_id' in config:
        run_dir = base_dir / 'sweeps' / config['sweep_id'] / f"run_{config['run_id']}"
    else:
        run_dir = base_dir / 'single' / f"run_{config['run_id']}"

    # Example: append a 'checkpoints_continued' subdir so we don't overwrite old run's 'checkpoints'
    continued_dir = run_dir / "checkpoints_continued"
    return continued_dir

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
        try:
            # Try to parse the config argument as JSON string
            file_updates = json.loads(args.config)
        except json.JSONDecodeError:
            # If it fails, assume it's a file path
            with open(args.config) as f:
                file_updates = json.load(f)
        config_updates.update(file_updates)
            
    config = load_and_validate_config(original_config, config_updates)
    
    # Add sweep info to config if available
    if original_run.sweep:
        config['sweep_id'] = original_run.sweep.id
    config['run_id'] = args.run_id
    
    # 1) Determine new run_dir => place we want to store new logs, new checkpoints, etc.
    new_run_dir = determine_run_directory(config, args.out_dir)

    # 2) Resolve checkpoint path from user input (file vs directory => latest.pt)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    # 3) Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and "," in args.gpus:
        device = f"cuda:{args.gpus.split(',')[0]}"

    # 4) Load model & iteration from that checkpoint
    model, iter_num, best_val_loss = load_and_validate_model(
        run_dir=new_run_dir.parent,  # If you need parent folder for references
        config=config,
        device=device,
        checkpoint_path=checkpoint_path
    )

    # 5) Update config with state
    config.update({
        'init_from': 'resume',
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'resume_run_id': args.run_id
    })

    # 6) Possibly set ddp => True if user wants distributed
    if args.ddp:
        config["ddp"] = True
    else:
        config["ddp"] = False

    # 7) Build command & environment => we store new artifacts in new_run_dir
    cmd, env = create_command(config, str(new_run_dir), args.ddp, args.gpus)

    logger.info("Executing command: %s", " ".join(cmd))

    # 8) Run training
    exit_code = run_training(cmd, env, str(new_run_dir))
    if exit_code == 0:
        logger.info("Training ended successfully.")
    else:
        logger.error(f"Training process exited with code {exit_code}.")
    return exit_code

if __name__ == '__main__':
    exit(main())
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
It has been extended to handle different architectures: original, diamond, unetxformer.
Also included is the integrated logic for 'shape_variant' when using
diamond or unetxformer architectures.

This training script can handle:
- original architecture
- diamond architecture
- unetxformer architecture
- custom layer dimensions via n_dims
- Logging to wandb with additional axes (tokens, compute, time, etc.)
- Long running experiments for scaling law studies
- Multi-dataset training with proper sampling
- Distributed training coordination
- Scaling experiments with comprehensive logging
- Resource management and monitoring
- Improved process management, error handling, and resource utilization

If n_dims is provided, it will override the architecture-based dimension calculation.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
"""
By setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True early,
we reduce fragmentation in certain GPU memory usage patterns.
We use .setdefault() so that if user already has it set externally,
that takes precedence. This call occurs before any big GPU calls,
increasing the chance that it is respected.
"""

# ------------------------------------------------------------------
# 1) NumExpr max threads environment variable if not set:
import os
if "NUMEXPR_MAX_THREADS" not in os.environ:
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
    # Note: setting this does not meaningfully affect GPU memory usage,
    # but prevents the warning about "defaulting to 8 threads" when you have many CPU cores.
import numexpr
# ------------------------------------------------------------------

import time
import math
import signal
import pickle
import psutil
from contextlib import nullcontext
import numpy as np
import torch
import wandb
import json
from utils.wandb_logger import WandBLogger, ScalingExperimentConfig
from utils.dataset_handler import ScalingDatasetHandler
from utils.training_utils import (
    CheckpointManager, setup_output_dir, 
    load_training_state, save_training_state
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
import gc
import sys
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from utils.diamond_dim_utils import calculate_diamond_dims
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from config.defaults import get_default_config
# Start with defaults
config = get_default_config()

def validate_config(config):
    """Validate all required config parameters are present and have valid values"""
    required_keys = {
        # Output and logging
        'out_dir', 'eval_interval', 'log_interval', 'eval_iters',
        'eval_only', 'always_save_checkpoint',
        
        # Model initialization
        'init_from',
        
        # Training hyperparameters
        'gradient_accumulation_steps', 'batch_size', 'block_size',
        'n_layer', 'dropout', 'bias', 'learning_rate', 'max_iters',
        'weight_decay', 'beta1', 'beta2', 'grad_clip', 'decay_lr',
        'warmup_iters', 'lr_decay_iters', 'min_lr',
        
        # System settings
        'backend', 'device', 'dtype', 'compile',
        
        # Architecture settings
        'model_architecture', 'base_dim', 'max_dim', 'head_dim'
    }
    
    # Check for missing keys
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # Validate specific parameter constraints
    if config['max_iters'] <= 0:
        raise ValueError("max_iters must be positive")
        
    if config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
        
    if config['model_architecture'] not in ['original', 'diamond', 'unetxformer']:
        raise ValueError(f"Invalid model_architecture: {config['model_architecture']}")
        
    if config['block_size'] <= 0:
        raise ValueError("block_size must be positive")
        
    if config['n_layer'] <= 0:
        raise ValueError("n_layer must be positive")
        
    if config['gradient_accumulation_steps'] <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
        
    if config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
        
    if config['dropout'] < 0 or config['dropout'] > 1:
        raise ValueError("dropout must be between 0 and 1")
        
    if config['device'] not in ['cpu', 'cuda'] and not config['device'].startswith('cuda:'):
        raise ValueError(f"Invalid device: {config['device']}")
        
    if config['dtype'] not in ['float32', 'bfloat16', 'float16']:
        raise ValueError(f"Invalid dtype: {config['dtype']}")
    
    # Validate diamond/unetxformer specific parameters
    if config['model_architecture'] in ['diamond', 'unetxformer']:
        if config['base_dim'] <= 0:
            raise ValueError("base_dim must be positive")
        if config['max_dim'] < config['base_dim']:
            raise ValueError("max_dim must be >= base_dim")
        if config['head_dim'] <= 0:
            raise ValueError("head_dim must be positive")
        if 'shape_variant' in config and config['shape_variant'] not in ['symmetry', 'stay_wide', 'kite']:
            raise ValueError(f"Invalid shape_variant: {config['shape_variant']}")
            
    return True

def process_datasets_config(datasets_config):
    """Process and validate datasets configuration."""
    if isinstance(datasets_config, str):
        try:
            ds = json.loads(datasets_config)
        except json.JSONDecodeError:
            logger.warning("Invalid datasets JSON string. Falling back to single dataset.")
            return {}
        
        # Validate weights sum to approximately 1
        total_weight = sum(ds.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Dataset weights sum to {total_weight}, normalizing...")
            ds = {k: v / total_weight for k, v in ds.items()}
        
        return ds
    
    return datasets_config if isinstance(datasets_config, dict) else {}

def calculate_compute_metrics(iter_num: int, num_params: int, tokens_per_iter: int, training_start_time: float) -> Dict[str, float]:
    """Calculate compute-related metrics for scaling law analysis"""
    tokens_so_far = iter_num * tokens_per_iter
    # 6 * N flops per token for attention+MLP
    compute = tokens_so_far * 6 * num_params
    time_elapsed = time.time() - training_start_time
    return {
        'tokens': tokens_so_far,
        'compute': compute,
        'compute_efficiency': compute / time_elapsed if time_elapsed > 0 else 0,
        'tokens_per_second': tokens_so_far / time_elapsed if time_elapsed > 0 else 0
    }

def setup_model_dimensions(config: Dict[str, Any]) -> Tuple[list, list, bool]:
    """Setup model dimensions based on architecture type"""
    if config['n_dims'] is not None:
        logger.debug("Using explicitly provided n_dims")
        layer_dims = config['n_dims']
        assert len(layer_dims) == config['n_layer'], \
            f"n_dims has {len(layer_dims)} entries, but n_layer={config['n_layer']}"
        n_heads = [d // config['head_dim'] for d in layer_dims]
        use_unet = False
    else:
        logger.debug("Calculating dimensions based on model_architecture")
        if config['model_architecture'] == 'original':
            n_embd = config['base_dim']
            n_head = n_embd // config['head_dim']
            layer_dims = [n_embd] * config['n_layer']
            n_heads = [n_head] * config['n_layer']
            use_unet = False
        elif config['model_architecture'] in ['diamond', 'unetxformer']:
            shape_variant = config.get('shape_variant', 'symmetry')
            layer_dims = calculate_diamond_dims(
                config['n_layer'],
                config['base_dim'],
                config['max_dim'],
                config['head_dim'],
                shape_variant=shape_variant
            )
            n_heads = [dim // config['head_dim'] for dim in layer_dims]
            use_unet = (config['model_architecture'] == 'unetxformer')
        else:
            raise ValueError(f"Invalid model_architecture: {config['model_architecture']}")
    
    return layer_dims, n_heads, use_unet

class TrainingManager:
    """Manages the training process with proper resource handling"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exit_flag = False
        self.checkpoint_manager = None
        self.wandb_logger = None
        self.dataset_handler = None
        self.training_start_time = time.time()
        self.device = None
        # A safety measure to track how often we halved batch_size due to OOM
        self.oom_retries = 0
        # We'll limit the number of times we can halve batch_size
        self.oom_retry_limit = 3
        
        # Process dataset configuration
        datasets = cfg.get('datasets', {})
        if not isinstance(datasets, dict):
            logger.warning("`cfg.datasets` is not a dictionary. Falling back to single dataset mode.")
            datasets = {"openwebtext": 1.0}

        self.dataset_handler = ScalingDatasetHandler(
            datasets=datasets,
            data_dir='data',
            block_size=cfg.block_size,
            seed=1337
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, initiating cleanup...")
        self.exit_flag = True

    def verify_ddp_consistency(self, model: torch.nn.Module, ddp_world_size: int):
        """Verify model structure consistency across DDP processes"""
        if torch.distributed.is_initialized():
            for name, param in model.named_parameters():
                if param.device.type != 'cuda':
                    param.data = param.data.cuda()
                shapes = [torch.zeros_like(param.data) for _ in range(ddp_world_size)]
                try:
                    torch.distributed.all_gather(shapes, param.data)
                except RuntimeError as e:
                    logger.error(f"DDP inconsistency for param {name}: {e}")
                    raise

    def setup_distributed(self) -> Tuple[bool, str, bool, int, int, int, int]:
        """Setup distributed training environment with enhanced error handling"""
        try:
            # Set NCCL environment variables with timeouts for robustness
            os.environ.update({
                'NCCL_DEBUG': 'INFO',
                'NCCL_IB_TIMEOUT': '23',
                'NCCL_SOCKET_TIMEOUT': '120',
                'NCCL_IB_RETRY_CNT': '7'  # Add retry count for robustness
            })

            ddp = int(os.environ.get('RANK', -1)) != -1

            # Example: if user sets cfg.ddp to true, do distributed init here
            if getattr(self.cfg, "ddp", False):
                # This is a placeholder. You could integrate real logic such as:
                # init_process_group(backend=self.cfg.backend, ...)
                # local_rank = ...
                # device = f"cuda:{local_rank}"
                # torch.cuda.set_device(device)
                # self.cfg.gradient_accumulation_steps //= world_size
                logger.info("cfg.ddp=True: user is requesting DDP mode (placeholder example)")
                # The real logic is up to you. This is just a demonstration.

            if ddp:
                logger.info("Initializing distributed training via RANK, LOCAL_RANK, WORLD_SIZE env vars")
                
                # Validate required environment variables
                required_env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']
                missing_vars = [var for var in required_env_vars if var not in os.environ]
                if missing_vars:
                    raise ValueError(f"Missing required environment variables: {missing_vars}")
                
                ddp_rank = int(os.environ['RANK'])
                ddp_local_rank = int(os.environ['LOCAL_RANK'])
                ddp_world_size = int(os.environ['WORLD_SIZE'])
                
                # Initialize process group with timeout and error handling
                try:
                    init_process_group(
                        backend=self.cfg.backend,
                        timeout=datetime.timedelta(minutes=30)
                    )
                    logger.info(f"Successfully initialized process group for rank {ddp_rank}")
                except Exception as e:
                    logger.error(f"Failed to initialize process group: {str(e)}")
                    raise RuntimeError(f"DDP initialization failed: {str(e)}") from e
                
                # Set up device
                device = f'cuda:{ddp_local_rank}'
                try:
                    torch.cuda.set_device(device)
                except Exception as e:
                    logger.error(f"Failed to set CUDA device {device}: {str(e)}")
                    raise RuntimeError(f"Device initialization failed: {str(e)}") from e
                
                master_process = ddp_rank == 0
                seed_offset = ddp_rank
                
                # Validate gradient accumulation steps
                if self.cfg.gradient_accumulation_steps % ddp_world_size != 0:
                    raise ValueError(
                        f"gradient_accumulation_steps ({self.cfg.gradient_accumulation_steps}) "
                        f"must be divisible by world_size ({ddp_world_size})"
                    )
                self.cfg.gradient_accumulation_steps //= ddp_world_size
                
                # Set logging level for non-master processes
                if not master_process:
                    logging.getLogger().setLevel(logging.WARNING)
                
                logger.info(
                    f"DDP setup complete: rank={ddp_rank}, local_rank={ddp_local_rank}, "
                    f"world_size={ddp_world_size}, device={device}"
                )
            else:
                # Single-GPU or CPU setup
                ddp_rank = 0
                ddp_local_rank = 0
                ddp_world_size = 1
                master_process = True
                seed_offset = 0
                device = self.cfg.device
                
                # Validate device configuration
                if 'cuda' in device and not torch.cuda.is_available():
                    logger.warning("CUDA device requested but not available. Falling back to CPU.")
                    device = 'cpu'
                    self.cfg.device = 'cpu'
            
            # Store device in class attribute
            self.device = device
            
            # Check CUDA availability and memory
            if 'cuda' in device:
                try:
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    logger.info(f"GPU {device} total memory: {total_memory / 1e9:.2f} GB")
                except Exception as e:
                    logger.error(f"Failed to query CUDA device properties: {str(e)}")
                    raise RuntimeError(f"CUDA device query failed: {str(e)}") from e
            
            return ddp, device, master_process, seed_offset, ddp_world_size, ddp_rank, ddp_local_rank
            
        except Exception as e:
            logger.error(f"Failed to setup distributed environment: {str(e)}")
            if ddp and torch.distributed.is_initialized():
                try:
                    destroy_process_group()
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup process group: {str(cleanup_error)}")
            raise RuntimeError("Distributed setup failed") from e

    def setup_training(self, 
                      device: str,
                      master_process: bool,
                      seed_offset: int) -> Tuple[Path, Path, float]:
        """Setup training environment, directories and loggers"""
        
        # Verify critical config values
        if self.cfg.init_from not in ['scratch', 'resume'] and not self.cfg.init_from.startswith('gpt2'):
            raise ValueError(f"Invalid init_from: {self.cfg.init_from}")
            
        # Setup directories
        run_dir = setup_output_dir(self.cfg, self.master_process)
        self.checkpoint_manager = CheckpointManager(
            run_dir / "checkpoints",
            max_checkpoints=5
        )

        # Process dataset configuration
        datasets = self.cfg.get('datasets', {})
        # Force it to be a dictionary if Hydra loaded it differently
        if not isinstance(datasets, dict):
            logger.info("Converting cfg.datasets to a dictionary to avoid fallback mode.")
            if hasattr(self.cfg, 'dataset') and isinstance(self.cfg.dataset, str):
                datasets = {self.cfg.dataset: 1.0}
            else:
                # fallback to single dataset named "openwebtext", or adapt as needed:
                datasets = {"openwebtext": 1.0}
            self.cfg.datasets = datasets

        self.dataset_handler = ScalingDatasetHandler(
            datasets=datasets,
            data_dir='data',
            block_size=self.cfg.block_size,
            seed=1337 + seed_offset
        )

        data_dir = Path('data')
        self.tokens_per_iter = (
            self.cfg.gradient_accumulation_steps *
            self.cfg.batch_size *
            self.cfg.block_size *
            self.world_size
        )

        return run_dir, data_dir, float('inf')  # Initial best val loss

    def setup_model(self, 
                device: str,
                ddp: bool,
                ddp_local_rank: int,
                ddp_world_size: int) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Setup model and optimizer with proper parameter count logging"""
        
        # Setup model dimensions
        layer_dims, n_heads, use_unet = setup_model_dimensions(self.cfg)
        # Hydra config is often in "struct" mode; to add or modify keys, use open_dict
        with open_dict(self.cfg):
            self.cfg.layer_dims = layer_dims
            self.cfg.n_heads = n_heads
            self.cfg.use_unet = use_unet
        
        model_args = {
            'layer_dims': layer_dims,
            'n_heads': n_heads,
            'block_size': self.cfg.block_size,
            'bias': self.cfg.bias,
            'vocab_size': self.dataset_handler.vocab_size if self.dataset_handler else 50304,
            'dropout': self.cfg.dropout,
            'n_layer': self.cfg.n_layer,
            'model_architecture': self.cfg.model_architecture,
            'use_unet': use_unet
        }
        
        # Initialize model
        if self.cfg.init_from.startswith('gpt2'):
            logger.info(f"Initializing from OpenAI GPT-2 weights: {self.cfg.init_from}")
            override_args = dict(dropout=self.cfg.dropout)
            model = GPT.from_pretrained(self.cfg.init_from, override_args)
        else:
            model = GPT(GPTConfig(**model_args))
            
        # Calculate and log parameter count
        num_params = sum(p.numel() for p in model.parameters())
        if self.cfg.get('print_params_only', False):
            logger.info(f"Number of parameters: {num_params/1e6:.2fM}")
            # Log to wandb before exiting if wandb is enabled
            if self.wandb_logger and self.master_process:
                wandb.run.summary["number_of_parameters"] = num_params
            sys.exit(0)
        
        # Log parameter count for normal training runs
        if not ddp or ddp_local_rank == 0:
            logger.info(f"Number of parameters: {num_params/1e6:.2f}M")
            if self.wandb_logger:
                wandb.run.summary["number_of_parameters"] = num_params
                
        model.to(device)
        
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
            # Verify DDP consistency
            self.verify_ddp_consistency(model, ddp_world_size)
            
        optimizer = model.configure_optimizers(
            weight_decay=self.cfg.weight_decay,
            learning_rate=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            device_type='cuda' if 'cuda' in device else 'cpu'
        )

        if self.cfg.compile and torch.cuda.is_available():
            model = torch.compile(model)
            
        run_id = os.environ.get("WANDB_RUN_ID", "noID")
        base_dim = self.cfg.get('base_dim', 0)
        head_dim = self.cfg.get('head_dim', 0)
        # We'll initially set default_name = None; then define it below once we gather the real sweep_id, etc.
        default_name = None

        # Safely check for a user-specified wandb_run_name
        wandb_run_name = self.cfg.get('wandb_run_name', None)
        if wandb_run_name:
            # If user wrote something like "run_${sweep_id}"
            if "${sweep_id}" in wandb_run_name:
                actual_sweep_id = os.environ.get("WANDB_SWEEP_ID", "")
                if actual_sweep_id:
                    wandb_run_name = wandb_run_name.replace("${sweep_id}", actual_sweep_id)
                else:
                    wandb_run_name = wandb_run_name.replace("${sweep_id}", "NOSWEEPID")
            # Put it back into cfg
            with open_dict(self.cfg):
                self.cfg.wandb_run_name = wandb_run_name

        # Now create a final default name that includes the actual sweep_id from environment:
        run_id    = os.environ.get("WANDB_RUN_ID", "noID")
        sweep_id  = os.environ.get("WANDB_SWEEP_ID", "mysweep")  # fallback if not set
        short_run_id = run_id[:6] if len(run_id) >= 6 else run_id
        default_name = f"{sweep_id}-base{base_dim}-head{head_dim}-run{short_run_id}"

        if self.cfg.wandb_log and self.master_process:
            if self.cfg.datasets:
                from utils.wandb_logger import ScalingExperimentConfig

        # Create a more descriptive run name if using wandb
        # For example: "mysweep-base384-head64-runASDF12"
        # Using random short run ID from wandb if available
        run_id    = os.environ.get("WANDB_RUN_ID", "noID")
        sweep_id  = os.environ.get("WANDB_SWEEP_ID", "mysweep")  # default to "mysweep" if not found
        base_dim  = self.cfg.get('base_dim', 0)
        head_dim  = self.cfg.get('head_dim', 0)

        # Instead of "mysweep", use the actual sweep_id as prefix:
        # e.g. cthcyydc-base384-head64-run8tpq09
        short_run_id = run_id[:6] if len(run_id) >= 6 else run_id
        default_name = f"{sweep_id}-base{base_dim}-head{head_dim}-run{short_run_id}"

        if self.cfg.wandb_log and self.master_process:
            # If user didn't specify 'wandb_run_name', fallback to default_name
            if self.cfg.datasets:
                from utils.wandb_logger import ScalingExperimentConfig
                experiment_config = ScalingExperimentConfig(
                    n_layer=self.cfg.n_layer,
                    n_heads=self.cfg.n_heads,
                    layer_dims=self.cfg.layer_dims,
                    max_tokens=self.cfg.max_iters * self.tokens_per_iter,
                    batch_size=self.cfg.batch_size,
                    learning_rate=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    warmup_tokens=self.cfg.warmup_iters * self.tokens_per_iter,
                    final_tokens=self.cfg.max_iters * self.tokens_per_iter,
                    datasets=self.cfg.datasets,
                    eval_datasets=list(self.cfg.datasets.keys()),
                    device=self.cfg.device,
                    dtype=self.cfg.dtype
                )
                self.wandb_logger = WandBLogger(
                    config=experiment_config,
                    project=self.cfg.get('wandb_project', 'VSLM'),
                    entity=self.cfg.get('wandb_entity', 'wcml'),
                    name=self.cfg.get('wandb_run_name', default_name),
                    group=self.cfg.get('wandb_group', None)
                )

        return model, optimizer

    def train_step(self, 
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  X: torch.Tensor,
                  Y: torch.Tensor,
                  ddp: bool,
                  scaler: torch.cuda.amp.GradScaler,
                  micro_step: int) -> torch.Tensor:
        """
        Single training micro-step with gradient accumulation.
        Wrapped in an OOM retry so we can attempt smaller batch size if we fail.
        """
        attempt = 0
        while True:
            try:
                return self._train_step_once(
                    model, optimizer, X, Y, ddp, scaler, micro_step
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.cfg.batch_size > 1 and self.oom_retries < self.oom_retry_limit:
                        self.cfg.batch_size = max(1, self.cfg.batch_size // 2)
                        self.oom_retries += 1
                        # Update tokens_per_iter after batch size change
                        self.tokens_per_iter = (
                            self.cfg.gradient_accumulation_steps * 
                            self.cfg.batch_size * 
                            self.cfg.block_size * 
                            self.world_size
                        )
                        logger.warning(
                            f"OOM detected, halving batch_size to {self.cfg.batch_size} "
                            f"(retry {self.oom_retries}/{self.oom_retry_limit})"
                        )
                        # Force freeing cache, then re-fetch batch
                        self.cleanup_memory()
                        X, Y = self.get_batch('train')
                        continue
                # If no more fallback is possible, re-raise
                raise e

    def _train_step_once(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         X: torch.Tensor,
                         Y: torch.Tensor,
                         ddp: bool,
                         scaler: torch.cuda.amp.GradScaler,
                         micro_step: int) -> torch.Tensor:
        """Single training step with gradient accumulation"""
        
        # DDP gradient sync control
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == self.cfg.gradient_accumulation_steps - 1
            )
        
        # Forward pass
        with self.ctx:
            logits, loss = model(X, Y)
            loss = loss / self.cfg.gradient_accumulation_steps
            
        # Backward pass
        scaler.scale(loss).backward()
        
        return loss

    def get_batch(self, split: str):
        """Get batch from dataset handler or memory map"""
        if self.dataset_handler:
            return self.dataset_handler.get_batch(
                self.cfg.batch_size,
                split,
                self.device
            )
        else:
            data = np.memmap(
                os.path.join(self.data_dir, f'{split}.bin'),
                dtype=np.uint16,
                mode='r'
            )
            ix = torch.randint(
                len(data) - self.cfg.block_size,
                (self.cfg.batch_size,)
            )
            x = torch.stack([
                torch.from_numpy((data[i:i+self.cfg.block_size]).astype(np.int64))
                for i in ix
            ])
            y = torch.stack([
                torch.from_numpy((data[i+1:i+1+self.cfg.block_size]).astype(np.int64))
                for i in ix
            ])
            
            if 'cuda' in self.device:
                x = x.pin_memory().to(self.device, non_blocking=True)
                y = y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            return x, y
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, device: str) -> Dict[str, float]:
        """Evaluation loop with proper resource management"""
        model.eval()
        losses = {}
        
        try:
            for split in ['train', 'val']:
                batch_losses = torch.zeros(self.cfg.eval_iters, device=device)
                for k in range(self.cfg.eval_iters):
                    X, Y = self.get_batch(split)
                    with self.ctx:
                        _, loss = model(X, Y)
                    batch_losses[k] = loss.item()
                losses[split] = batch_losses.mean().item()
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
            
        finally:
            model.train()
            self.cleanup_memory()
            
        return losses

    def train(self):
        """Main training loop with proper error handling"""
        try:
            # Setup distributed environment first
            (
                self.ddp, self.device, self.master_process,
                self.seed_offset, self.world_size,
                self.ddp_rank, self.ddp_local_rank
            ) = self.setup_distributed()
            
            # Setup device context
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            ptdtype = {
                'float32': torch.float32,
                'bfloat16': torch.bfloat16,
                'float16': torch.float16
            }[self.cfg.dtype]
            self.ctx = nullcontext() if device_type == 'cpu' else \
                      torch.amp.autocast(device_type=device_type, dtype=ptdtype)
            
            # Setup training environment
            self.run_dir, self.data_dir, self.best_val_loss = self.setup_training(
                self.device,
                self.master_process,
                self.seed_offset
            )
            
            # Create model and optimizer
            self.model, self.optimizer = self.setup_model(
                self.device,
                self.ddp,
                self.ddp_local_rank,
                self.world_size
            )
            
            # Load state if resuming
            if self.cfg.init_from == 'resume':
                (
                    self.model,
                    self.optimizer,
                    self.iter_num,
                    self.best_val_loss
                ) = load_training_state(
                    checkpoint_manager=self.checkpoint_manager,
                    model=self.model,
                    optimizer=self.optimizer,
                    device=self.device,
                    config=self.cfg,
                    ddp=self.ddp
                )
            else:
                self.iter_num = 0
                
            # Initialize training state
            self.scaler = torch.cuda.amp.GradScaler(
                # Allowed in PyTorch 2.5.0
                enabled=(self.cfg.dtype == 'float16'),
                growth_interval=2000,
                init_scale=65536.0
            )
            
            self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model
            
            # Get initial batch
            self.X, self.Y = self.get_batch('train')
            
            # Training loop state
            t0 = time.time()
            local_iter_num = 0
            running_mfu = -1.0
            
            while not self.exit_flag:
                # Learning rate update
                lr = self.get_lr(self.iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                # Evaluation
                if self.iter_num % self.cfg.eval_interval == 0 and self.master_process:
                    losses = self.evaluate(self.model, self.device)
                    logger.info(
                        f"step {self.iter_num}: train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )
                    
                    if self.wandb_logger:
                        compute_metrics = calculate_compute_metrics(
                            self.iter_num,
                            self.raw_model.get_num_params(),
                            self.tokens_per_iter,
                            self.training_start_time
                        )
                        
                        self.wandb_logger.log_evaluation(
                            self.iter_num,
                            losses['train'],
                            losses['val'],
                            compute_metrics['compute']
                        )
                        
                        # Log additional metrics
                        self.wandb_logger.log_system_stats()
                        self.wandb_logger.log_memory_per_gpu()
                        
                    # Save checkpoint
                    is_best = losses['val'] < self.best_val_loss
                    if is_best or self.cfg.always_save_checkpoint:
                        save_training_state(
                            self.checkpoint_manager,
                            self.model,
                            self.optimizer,
                            self.cfg,
                            self.iter_num,
                            losses['val'],
                            self.best_val_loss,
                            is_best
                        )
                        if is_best:
                            self.best_val_loss = losses['val']
                            
                # Gradient accumulation training loop
                for micro_step in range(self.cfg.gradient_accumulation_steps):
                    # <-- Now calls the new wrapper that can catch OOM
                    step_loss = self.train_step(
                        self.model,
                        self.optimizer,
                        self.X,
                        self.Y,
                        self.ddp,
                        self.scaler,
                        micro_step
                    )
                    # step_loss is only the micro-batch portion of total loss
                    loss = step_loss

                    # Re-fetch next batch in case batch_size changed from OOM handling
                    self.X, self.Y = self.get_batch('train')

                # Gradient clipping and optimization step
                if self.cfg.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                if self.iter_num % self.cfg.log_interval == 0 and self.master_process:
                    # Calculate loss and MFU
                    lossf = loss.item() * self.cfg.gradient_accumulation_steps
                    if local_iter_num >= 5:  # Let training stabilize
                        mfu = self.raw_model.estimate_mfu(
                            self.cfg.batch_size * self.cfg.gradient_accumulation_steps,
                            dt
                        )
                        running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
                    
                    logger.info(
                        f"iter {self.iter_num}: loss {lossf:.4f}, "
                        f"time {dt*1000:.2f}ms, "
                        f"mfu {running_mfu*100:.2f}%"
                    )
                    
                    if self.wandb_logger:
                        self.wandb_logger.log_training_step(
                            self.iter_num,
                            {
                                'loss': lossf,
                                'lr': lr,
                                'mfu': running_mfu,
                                'iter_time_ms': dt * 1000,
                            }
                        )
                        
                        # Log scaling metrics
                        compute_metrics = calculate_compute_metrics(
                            self.iter_num,
                            self.raw_model.get_num_params(),
                            self.tokens_per_iter,
                            self.training_start_time
                        )
                        time_elapsed = time.time() - self.training_start_time
                        self.wandb_logger.log_scaling_metrics(
                            compute_metrics['tokens'],
                            compute_metrics['compute'],
                            time_elapsed
                        )
                        
                self.iter_num += 1
                local_iter_num += 1
                
                # Exit conditions
                if self.iter_num > self.cfg.max_iters:
                    break
                    
                # Periodic cleanup
                if self.iter_num % 1000 == 0:
                    self.cleanup_memory()
                    
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
            
        finally:
            self.cleanup()

    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def cleanup(self):
        """Final cleanup"""
        if self.dataset_handler:
            self.dataset_handler.cleanup()
            
        if torch.distributed.is_initialized():
            destroy_process_group()
            
        if self.wandb_logger:
            self.wandb_logger.finish()
            
    def get_lr(self, it: int) -> float:
        """Get learning rate for current iteration"""
        if not self.cfg.decay_lr:
            return self.cfg.learning_rate
            
        # Learning rate decay logic
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * it / self.cfg.warmup_iters
            
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr
            
        # Cosine learning rate decay
        decay_ratio = (it - self.cfg.warmup_iters) / (
            self.cfg.lr_decay_iters - self.cfg.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.cfg.min_lr + coeff * (
            self.cfg.learning_rate - self.cfg.min_lr
        )

@hydra.main(
    version_base=None,  # or "1.3" if Hydra 1.3
    config_path="config/hydra",
    config_name="train.yaml"
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Final Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    def maybe_replace_sweeprun_placeholder(config_dict):
        placeholder = "@@SWEEPID@@"

        def replace_placeholder_in_string(s: str) -> str:
            if placeholder not in s:
                return s
            wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID", "")
            local_sweep_id = os.environ.get("LOCAL_SWEEP_ID", "")
            if wandb_sweep_id:
                return s.replace(placeholder, wandb_sweep_id)
            elif local_sweep_id:
                return s.replace(placeholder, local_sweep_id)
            else:
                logger.warning(
                    f"No WANDB_SWEEP_ID or LOCAL_SWEEP_ID found in environment; placeholder remains in {s}"
                )
                return s

        # Recurse into nested structures if needed
        for key, val in list(config_dict.items()):
            if isinstance(val, str):
                config_dict[key] = replace_placeholder_in_string(val)
            elif isinstance(val, dict):
                maybe_replace_sweeprun_placeholder(val)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, str):
                        val[i] = replace_placeholder_in_string(item)

    # Do that replacement before the training manager is constructed
    maybe_replace_sweeprun_placeholder(cfg)

    trainer = TrainingManager(cfg)
    trainer.train()
    logger.info("Training ended OK.")

if __name__ == "__main__":
    main()
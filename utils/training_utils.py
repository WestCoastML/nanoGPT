"""
Enhanced training utilities with improved checkpoint management and directory structure.
Includes proper error handling and atomic operations.
Used by both train.py and continue_run.py
"""

import os
import json
import shutil
import logging
import tempfile
import hashlib
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints with rotation and validation"""
    
    def __init__(self, 
                 checkpoint_dir: Union[str, Path],
                 max_checkpoints: int = 5,
                 keep_best: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self,
                       state_dict: Dict[str, Any],
                       metadata: Dict[str, Any],
                       is_best: bool = False) -> str:
        """Atomic checkpoint save with rotation"""
        
        # Build an extended checkpoint dict with a guaranteed top-level "timestamp"
        checkpoint_full = {
            'state_dict': state_dict,
            'metadata': metadata,
        }

        # Insert top-level timestamp outside of 'metadata'
        checkpoint_full['timestamp'] = datetime.now().isoformat()

        # For clarity, final reference is 'checkpoint'
        checkpoint = checkpoint_full

        # Generate unique name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Atomic save
        with tempfile.NamedTemporaryFile(dir=str(self.checkpoint_dir), delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            tmp.flush()
            os.fsync(tmp.fileno())
            
        os.rename(tmp.name, checkpoint_path)
        
        # Update symlinks
        latest_link = self.checkpoint_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        os.symlink(checkpoint_name, latest_link)
        
        if is_best:
            best_link = self.checkpoint_dir / "best.pt"
            if best_link.exists():
                best_link.unlink()
            os.symlink(checkpoint_name, best_link)
        
        # Backup before rotating
        self.backup_existing_checkpoint(checkpoint_path)
            
        # Rotate old checkpoints
        self._rotate_checkpoints()
        
        return str(checkpoint_path)
        
    def load_checkpoint(self, 
                       map_location: Optional[str] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """Load latest checkpoint with fallback to best"""
        
        paths_to_try = [
            self.checkpoint_dir / "latest.pt",
            self.checkpoint_dir / "best.pt"
        ]
        
        last_error = None
        for path in paths_to_try:
            try:
                if path.exists():
                    checkpoint = torch.load(str(path), map_location=map_location)
                    self._validate_checkpoint(checkpoint, strict)
                    logger.info(f"Loaded checkpoint from {path}")
                    return checkpoint
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load checkpoint from {path}: {e}")
                
        if last_error:
            raise RuntimeError(f"Failed to load any checkpoint: {last_error}")
        else:
            raise FileNotFoundError("No checkpoints found")
            
    def backup_existing_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Create backup before any modifications"""
        backup_dir = self.checkpoint_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            backup_name = f"backup_{datetime.now():%Y%m%d_%H%M%S}.pt"
            shutil.copy2(checkpoint_path, backup_dir / backup_name)
            
    def _validate_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        """Validate checkpoint structure"""
        required_keys = ['state_dict', 'metadata', 'timestamp']
        if strict and not all(k in checkpoint for k in required_keys):
            raise ValueError(f"Invalid checkpoint structure. Missing keys: {set(required_keys) - set(checkpoint.keys())}")
            
    def _rotate_checkpoints(self):
        """Remove old checkpoints while keeping important ones"""
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_*.pt")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep max_checkpoints newest files and best if specified
        keep_files = set(checkpoints[:self.max_checkpoints])
        best_path = self.checkpoint_dir / "best.pt"
        if self.keep_best and best_path.exists():
            keep_files.add(best_path.resolve())
            
        # Remove others
        for checkpoint in checkpoints:
            if checkpoint not in keep_files:
                try:
                    checkpoint.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove old checkpoint {checkpoint}: {e}")

def setup_output_dir(config: Dict[str, Any], 
                    master_process: bool,
                    wandb_run: Optional[Any] = None) -> Path:
    """Setup training output directory structure"""
    
    if wandb_run:
        sweep_id = wandb_run.sweep_id if wandb_run.sweep else None
        run_id = wandb_run.id
    else:
        sweep_id = None
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Determine directory structure
    base_dir = Path(config.get('out_dir', 'runs'))
    if sweep_id:
        run_dir = base_dir / "sweeps" / sweep_id / f"run_{run_id}"
    else:
        run_dir = base_dir / "single" / f"run_{run_id}"
        
    if master_process:
        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        
        # Save config
        save_config(config, run_dir)
        
    return run_dir

def save_config(config: Dict[str, Any], run_dir: Path):
    """Save config with backup"""
    config_path = run_dir / "config.json"
    if config_path.exists():
        # Backup existing config
        backup_dir = run_dir / "config_backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(config_path, backup_dir / f"config_{timestamp}.json")
        
    # Save new config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def resume_from_checkpoint(checkpoint_manager: CheckpointManager, 
                         run_id: str,
                         config: Dict[str, Any],
                         device: str) -> Tuple[Dict[str, Any], int, float]:
    """Resume training from checkpoint with validation"""
    try:
        checkpoint = checkpoint_manager.load_checkpoint(map_location=device)
        
        # Validate run_id if present
        if 'run_id' in checkpoint['metadata'] and checkpoint['metadata']['run_id'] != run_id:
            raise ValueError("Checkpoint run_id mismatch")
            
        # Validate config compatibility
        if not validate_config_compatibility(checkpoint['metadata'].get('config', {}), config):
            raise ValueError("Checkpoint configuration not compatible with current config")
            
        # Get training state
        iter_num = checkpoint['metadata'].get('iteration', 0)
        best_val_loss = checkpoint['metadata'].get('best_val_loss', float('inf'))
        
        return checkpoint, iter_num, best_val_loss
        
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        raise

def load_training_state(checkpoint_manager: CheckpointManager,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: str,
                       config: Dict[str, Any],
                       ddp: bool = False) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """Load complete training state with config validation"""
    
    try:
        checkpoint = checkpoint_manager.load_checkpoint(map_location=device)
        
        # Validate config compatibility
        if not validate_config_compatibility(
            checkpoint['metadata'].get('config', {}),
            config
        ):
            raise ValueError("Checkpoint config not compatible with current config")
        
        # Load model state
        model_state = checkpoint['state_dict']['model']
        if ddp:
            # Handle DDP state dict
            if not isinstance(model, DDP) and all(k.startswith('module.') for k in model_state.keys()):
                model_state = {k[7:]: v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['state_dict']['optimizer'])
        
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        # Get metadata
        metadata = checkpoint['metadata']
        iteration = metadata.get('iteration', 0)
        best_val_loss = metadata.get('best_val_loss', float('inf'))
        
        return model, optimizer, iteration, best_val_loss
        
    except (FileNotFoundError, RuntimeError) as e:
        logger.warning(f"No valid checkpoint found ({str(e)}), starting from scratch")
        return model, optimizer, 0, float('inf')

def save_training_state(checkpoint_manager: CheckpointManager,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       config: Dict[str, Any],
                       iteration: int,
                       val_loss: float,
                       best_val_loss: float,
                       is_best: bool = False):
    """Save complete training state with config hash"""
    
    state_dict = {
        'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    metadata = {
        'iteration': iteration,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'config_hash': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    }
    
    checkpoint_manager.save_checkpoint(state_dict, metadata, is_best)

def validate_resumption(config: Dict[str, Any], checkpoint_metadata: Dict[str, Any]) -> bool:
    """Validate whether resumption is safe given checkpoint metadata"""
    critical_keys = ['model_architecture', 'n_layer', 'vocab_size', 'layer_dims', 'n_heads']
    for key in critical_keys:
        if key in checkpoint_metadata and key in config:
            if checkpoint_metadata[key] != config[key]:
                logger.error(f"Config mismatch for {key}: {checkpoint_metadata[key]} != {config[key]}")
                return False
    return True

def validate_checkpoint_compatibility(checkpoint: Dict[str, Any], 
                                   model: torch.nn.Module,
                                   config: Dict[str, Any]) -> bool:
    """Validate if checkpoint is compatible with current model and config"""
    try:
        # Check critical parameters
        checkpoint_meta = checkpoint.get('metadata', {})
        if not validate_resumption(config, checkpoint_meta):
            return False
        
        # Check model state dict compatibility
        checkpoint_state = checkpoint['state_dict']['model']
        model_state = model.state_dict()
        
        if set(checkpoint_state.keys()) != set(model_state.keys()):
            logger.error("Checkpoint state dict structure mismatch")
            return False
            
        for key in checkpoint_state:
            if checkpoint_state[key].shape != model_state[key].shape:
                logger.error(f"Tensor shape mismatch for {key}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False

def validate_config_compatibility(old_config: Dict[str, Any], 
                                new_config: Dict[str, Any]) -> bool:
    """Validate if two configs are compatible for resuming training."""
    
    # First check model_args if they exist
    if 'model_args' in old_config and 'model_args' in new_config:
        model_critical_keys = [
            'block_size', 'vocab_size', 'n_layer', 'layer_dims',
            'n_heads', 'model_architecture'
        ]
        for key in model_critical_keys:
            if key in old_config['model_args'] and key in new_config['model_args']:
                if old_config['model_args'][key] != new_config['model_args'][key]:
                    logger.error(f"Model argument mismatch for {key}: "
                               f"old={old_config['model_args'][key]}, "
                               f"new={new_config['model_args'][key]}")
                    return False

    # Then check top-level config keys that must match
    config_critical_keys = [
        'model_architecture', 
        'n_layer', 
        'vocab_size',
        'block_size'
    ]
    
    for key in config_critical_keys:
        if key in old_config and key in new_config:
            if old_config[key] != new_config[key]:
                logger.error(f"Config mismatch for {key}: "
                           f"old={old_config[key]}, new={new_config[key]}")
                return False
            
    return True

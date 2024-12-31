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

If n_dims is provided, it will override the architecture-based dimension calculation.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import wandb
import json
from utils.wandb_logger import WandBLogger, ScalingExperimentConfig
from utils.dataset_handler import ScalingDatasetHandler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
import psutil
import gc
import sys
import datetime
import logging

# Basic initial logging config - will be enhanced after DDP setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# default config values
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_entity = ""
wandb_group = None
wandb_run_name = 'gpt2'

# For multi-dataset usage, this can be empty or set to e.g. {"openwebtext":0.4,"wikipedia":0.3,"books":0.2,"tinystories":0.1}
datasets = {}

# If only one dataset is used (fallback if `datasets` is empty)
dataset = 'openwebtext'

gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024
n_layer = 12
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
model_architecture = 'original'
use_unet = False
print_params_only = False

# Additional parameters for diamond/unetxformer
base_dim = 384
max_dim = 1024
head_dim = 64
shape_variant = 'symmetry'

# If provided, n_dims overrides architecture-based dims
n_dims = None

def process_datasets_config(datasets_config):
    """Process and validate datasets configuration"""
    if isinstance(datasets_config, str):
        try:
            ds = json.loads(datasets_config)
        except json.JSONDecodeError:
            logger.warning("Invalid datasets JSON string. Falling back to single dataset.")
            return {}
        
        total_weight = sum(ds.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Dataset weights sum to {total_weight}, normalizing...")
            ds = {k: v / total_weight for k, v in ds.items()}
        
        return ds
    return datasets_config if isinstance(datasets_config, dict) else {}

def setup_wandb_logging(config, master_process):
    """Setup WandB logging with proper configuration"""
    if not config.get('wandb_log', False) or not master_process:
        return None
        
    if config['datasets']:
        experiment_config = ScalingExperimentConfig(
            n_layer=config['n_layer'],
            n_heads=config['n_heads'],
            layer_dims=config['layer_dims'],
            max_tokens=config['max_iters'] * config['tokens_per_iter'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_tokens=config['warmup_iters'] * config['tokens_per_iter'],
            final_tokens=config['max_iters'] * config['tokens_per_iter'],
            datasets=config['datasets'],
            eval_datasets=list(config['datasets'].keys()),
            device=config['device'],
            dtype=config['dtype']
        )
    else:
        experiment_config = config

    return WandBLogger(
        config=experiment_config,
        project=config.get('wandb_project', 'VSLM'),
        entity=config.get('wandb_entity', 'wcml'),
        name=config.get('wandb_run_name'),
        group=config.get('wandb_group')
    )

def setup_distributed(config):
    """Setup distributed training environment with improved error handling"""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        logger.info("Setting up distributed training...")
        
        # Set NCCL parameters for better stability
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_IB_TIMEOUT'] = '23'
        os.environ['NCCL_SOCKET_TIMEOUT'] = '120'
        
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group with increased timeout
        init_process_group(
            backend=config['backend'],
            timeout=datetime.timedelta(minutes=30)
        )
        
        # Set device and ensure GPU cache is empty
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        
        if config['gradient_accumulation_steps'] % ddp_world_size != 0:
            raise ValueError(
                f"gradient_accumulation_steps ({config['gradient_accumulation_steps']}) "
                f"must be divisible by world_size ({ddp_world_size})"
            )
        
        config['gradient_accumulation_steps'] //= ddp_world_size
        
        # Adjust logging level for non-master processes
        if not master_process:
            logging.getLogger().setLevel(logging.WARNING)
        
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = config['device']
    
    return ddp, device, master_process, seed_offset, ddp_world_size, ddp_rank, ddp_local_rank

def setup_model_dimensions(config):
    """Setup model dimensions based on architecture"""
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
            from utils.diamond_dim_utils import calculate_diamond_dims
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

def setup_dataset_handler(config, device, seed_offset):
    """Setup dataset handler or fallback to single dataset"""
    data_dir = None
    if config['datasets']:
        # Multi-dataset approach
        handler = ScalingDatasetHandler(
            datasets=config['datasets'],
            data_dir='data',
            block_size=config['block_size'],
            seed=1337 + seed_offset
        )
        vocab_size = handler.vocab_size
        data_dir = 'data'  # We'll store data in the common folder
    else:
        # Fallback to single dataset
        handler = None
        data_dir = os.path.join('data', config['dataset'])
        meta_path = os.path.join(data_dir, 'meta.pkl')
        vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                vocab_size = meta.get('vocab_size', None)
    
    return handler, vocab_size, data_dir

def get_batch(split, config, dataset_handler, data_dir, device):
    """Get a batch of data from either dataset handler or single dataset"""
    if dataset_handler is not None:
        return dataset_handler.get_batch(
            batch_size=config['batch_size'],
            split=split,
            device=device
        )
    else:
        # Single-dataset fallback
        data = np.memmap(
            os.path.join(data_dir, f'{split}.bin'),
            dtype=np.uint16, mode='r'
        )
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([
            torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64))
            for i in ix
        ])

        device_type = 'cuda' if 'cuda' in device else 'cpu'
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

def get_lr(it, config):
    """Get learning rate for current iteration"""
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # Cosine decay from warmup_iters to lr_decay_iters
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

@torch.no_grad()
def estimate_loss(model, config, dataset_handler, data_dir, device, ctx):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split, config, dataset_handler, data_dir, device)
            with ctx:
                _, loss_val = model(X, Y)
            losses[k] = loss_val.item()
        out[split] = losses.mean()
    model.train()
    return out

def cleanup_memory():
    """Cleanup memory and CUDA allocations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_memory_stats(wandb_logger, iter_num):
    """Log memory statistics via wandb.log if wandb_logger is active"""
    if wandb_logger is None:
        return
    wandb_logger.log_memory_per_gpu()

def cumulative_compute(iter_num, num_params, tokens_per_iter):
    """
    Roughly calculates a measure of cumulative compute used:
    6 * N flops per token for attention+MLP,
    multiplied by number of tokens processed so far.
    """
    N = num_params
    return (iter_num * tokens_per_iter * 6 * N)

def setup_ddp_logging():
    """Setup logging for DDP processes"""
    if int(os.environ.get('RANK', -1)) != -1:
        # This is a DDP process
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Set up logging format to include rank
        logging.basicConfig(
            format=f'[Rank {rank}] %(message)s',
            level=logging.INFO if rank == 0 else logging.WARNING
        )
        
        local_logger = logging.getLogger(__name__)
        local_logger.info(f"Initializing DDP process: rank={rank}, "
                          f"local_rank={local_rank}, world_size={world_size}")

def main():
    # Read and process config
    config_path = ""
    config_keys = [k for k, v in globals().items()
                   if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, dict))]
    # ensure these in config
    for needed_key in ["config_path", "n_dims", "datasets"]:
        if needed_key not in config_keys:
            config_keys.append(needed_key)

    from ast import literal_eval
    context = globals()
    context.update({
        'literal_eval': literal_eval,
        'json': json
    })
    # Use f-string or similar to avoid the UnboundLocalError
    logger.info(f"Command line arguments: {sys.argv}")
    exec(open('configurator.py').read(), context)

    config = {k: globals()[k] for k in config_keys}

    # Process multi-dataset config
    config['datasets'] = process_datasets_config(config.get('datasets', {}))

    # Setup distributed training
    ddp, device_str, master_process, seed_offset, ddp_world_size, ddp_rank, ddp_local_rank = setup_distributed(config)

    # setup_ddp_logging() after DDP is configured
    setup_ddp_logging()
    
    if ddp:
        # Synchronize random seeds
        torch.manual_seed(1337 + seed_offset)
        
        # Ensure all processes have same config
        for k, v in config.items():
            if torch.distributed.is_initialized():
                # Create tensor on the correct device
                if torch.distributed.get_rank() == 0:
                    if isinstance(v, (int, float)):
                        tensor = torch.tensor([float(v)], device=device_str)
                    else:
                        tensor = torch.tensor([0.0], device=device_str)
                else:
                    tensor = torch.tensor([0.0], device=device_str)
                
                # Broadcast and update config
                torch.distributed.broadcast(tensor, 0)
                if torch.distributed.get_rank() != 0:
                    if isinstance(v, (int, float)):
                        config[k] = int(tensor.item()) if isinstance(v, int) else tensor.item()

    # Print final config if master_process (helps avoid confusion)
    if master_process:
        logger.info("\n------ Effective Config (after overrides) ------")
        for k, v in config.items():
            logger.info(f"{k} = {v}")
        if config['datasets']:
            logger.info(f"Note: The single 'dataset' ({config['dataset']}) is overridden by multi-datasets {config['datasets']}")

    # Setup model dimensions
    layer_dims, n_heads, use_unet_local = setup_model_dimensions(config)
    config['layer_dims'] = layer_dims
    config['n_heads'] = n_heads
    config['use_unet'] = use_unet_local

    # Calculate tokens per iteration
    tokens_per_iter = (config['gradient_accumulation_steps'] *
                       ddp_world_size *
                       config['batch_size'] *
                       config['block_size'])
    config['tokens_per_iter'] = tokens_per_iter
    logger.info(f"\ntokens per iteration: {tokens_per_iter:,}")

    # Setup directories and random seed
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup dtype and autocast context
    device_type = 'cuda' if 'cuda' in device_str else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Setup dataset handler (multi or single dataset)
    dataset_handler, vocab_size, data_dir = setup_dataset_handler(config, device_str, seed_offset)

    # Setup wandb logging (renamed to avoid scoping collisions)
    wandb_logger = setup_wandb_logging(config, master_process)

    # Initialize model
    model_args = dict(
        layer_dims=config['layer_dims'],
        n_heads=config['n_heads'],
        block_size=config['block_size'],
        bias=config['bias'],
        vocab_size=vocab_size if vocab_size is not None else 50304,
        dropout=config['dropout'],
        n_layer=config['n_layer'],
        model_architecture=config['model_architecture'],
        use_unet=config['use_unet'],
    )

    # Model init from scratch/resume/gpt2
    if config['init_from'] == 'scratch':
        logger.info("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
    elif config['init_from'] == 'resume':
        logger.info(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device_str)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif config['init_from'].startswith('gpt2'):
        logger.info(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        iter_num = 0
        best_val_loss = 1e9
    else:
        raise ValueError(f"Unknown init_from: {config['init_from']}")

    # # Find this section in train.py around line 515
    # if ddp:
    #     # Verify all processes have same model structure
    #     for name, param in model.named_parameters():
    #         if torch.distributed.is_initialized():
    #             # Make sure param.data is on GPU and create tensor_list on same device
    #             param_device = param.data.device
    #             shapes = [torch.zeros_like(param.data, device=param_device) for _ in range(ddp_world_size)]
    #             torch.distributed.all_gather(shapes, param.data)
    #             if torch.distributed.get_rank() == 0:
    #                 for i, shape in enumerate(shapes):
    #                     if not torch.equal(shape, param.data):
    #                         raise ValueError(f"Parameter {name} shape mismatch between processes")

    # Move model to device
    model.to(device_str)

    if ddp:
        # Verify all processes have same model structure
        for name, param in model.named_parameters():
            if torch.distributed.is_initialized():
                # Ensure param is on GPU and create tensor_list on same device
                if param.device.type != 'cuda':
                    param.data = param.data.cuda()
                shapes = [torch.zeros_like(param.data, device=param.device) 
                        for _ in range(ddp_world_size)]
                try:
                    torch.distributed.all_gather(shapes, param.data)
                except RuntimeError as e:
                    print(f"Error during all_gather for param {name} on "
                        f"device {param.device}: {str(e)}")
                    raise

    # Optionally compile (PyTorch 2.0+)
    if config['compile']:
        logger.info("Compiling the model... (PyTorch 2.0+)")
        model = torch.compile(model)

    # Setup DDP if needed
    if ddp:
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model, device_ids=[ddp_local_rank])

    # Setup optimizer
    scaler = torch.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(
        config['weight_decay'],
        config['learning_rate'],
        (config['beta1'], config['beta2']),
        device_type
    )

    # Load optimizer state if resuming
    if config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint  # free memory

    # Log number of parameters
    num_params = raw_model.get_num_params()
    if master_process:
        logger.info(f"Number of parameters: {num_params/1e6:.2f}M")
        if config.get('wandb_log', False) and wandb.run is not None:
            wandb.run.summary["number_of_parameters"] = num_params

    # Training loop setup
    training_start_time = time.time()
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    # Fetch initial batch
    X, Y = get_batch('train', config, dataset_handler, data_dir, device_str)

    try:
        while True:
            # Decide current learning rate
            lr = (get_lr(iter_num, config) if config['decay_lr']
                  else config['learning_rate'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate every eval_interval steps
            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss(model, config, dataset_handler, data_dir, device_str, ctx)
                logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if wandb_logger:  # Log evaluation
                    time_elapsed = time.time() - training_start_time
                    tokens_so_far = iter_num * tokens_per_iter
                    comp = cumulative_compute(iter_num, num_params, tokens_per_iter)

                    wandb_logger.log_evaluation(iter_num, losses['train'], losses['val'], comp)
                    wandb_logger.log_scaling_metrics(tokens_so_far, comp, time_elapsed)
                    wandb_logger.log_memory_per_gpu()

                # Save checkpoint if val loss improved or always_save_checkpoint
                if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint_to_save = {
                            'model': (model.module.state_dict() if ddp
                                      else model.state_dict()),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        logger.info(f"saving checkpoint to {config['out_dir']}")
                        torch.save(checkpoint_to_save, os.path.join(config['out_dir'], 'ckpt.pt'))

            # Gradient accumulation steps
            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    # sync gradients only on last micro step
                    model.require_backward_grad_sync = (
                        micro_step == config['gradient_accumulation_steps'] - 1
                    )

                with ctx:
                    _, loss_val = model(X, Y)
                    # scale the loss to account for grad accumulation
                    loss_val = loss_val / config['gradient_accumulation_steps']

                # async prefetch next batch
                X, Y = get_batch('train', config, dataset_handler, data_dir, device_str)

                # backward pass
                scaler.scale(loss_val).backward()

            # Clip gradient
            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Timing & logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % config['log_interval'] == 0 and master_process:
                lossf = loss_val.item() * config['gradient_accumulation_steps']
                if local_iter_num >= 5:
                    if ddp:
                        mfu = raw_model.estimate_mfu(
                            config['batch_size'] * config['gradient_accumulation_steps'],
                            dt
                        )
                    else:
                        mfu = model.estimate_mfu(
                            config['batch_size'] * config['gradient_accumulation_steps'],
                            dt
                        )
                    running_mfu = (mfu if running_mfu < 0
                                   else 0.9 * running_mfu + 0.1 * mfu)
                logger.info(f"iter {iter_num}: loss {lossf:.4f}, "
                            f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

                if wandb_logger:
                    time_elapsed = time.time() - training_start_time
                    tokens_so_far = iter_num * tokens_per_iter
                    comp = cumulative_compute(iter_num, num_params, tokens_per_iter)

                    wandb_logger.log_training_step(iter_num, {
                        'loss': lossf,
                        'lr': lr,
                        'mfu': running_mfu,
                        'iter_time_ms': dt * 1000,
                    })
                    
                    wandb_logger.log_scaling_metrics(tokens_so_far, comp, time_elapsed)

            iter_num += 1
            local_iter_num += 1

            # End condition
            if iter_num > config['max_iters']:
                break

            # Periodic cleanup
            if iter_num % 1000 == 0:
                cleanup_memory()

    except Exception as e:
        if ddp:
            logger.error(f"[Rank {ddp_rank}] Error occurred: {str(e)}")
            # Ensure clean process group shutdown
            destroy_process_group()
        raise e

    finally:
        # Cleanup
        if dataset_handler is not None:
            logger.info("Cleaning up dataset handler...")
            dataset_handler.cleanup()
        if ddp:
            destroy_process_group()
        if wandb_logger:
            logger.info("Finishing wandb logging...")
            wandb_logger.finish()

if __name__ == '__main__':
    main()

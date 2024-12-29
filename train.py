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
            print("WARNING: Invalid datasets JSON string. Falling back to single dataset.")
            return {}
        
        total_weight = sum(ds.values())
        if not (0.99 <= total_weight <= 1.01):
            print(f"WARNING: Dataset weights sum to {total_weight}, normalizing...")
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
    """Setup distributed training environment"""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        assert config['gradient_accumulation_steps'] % ddp_world_size == 0, \
            "gradient_accumulation_steps must be divisible by world size"
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = config['device']
    
    return ddp, device, master_process, seed_offset, ddp_world_size

def setup_model_dimensions(config):
    """Setup model dimensions based on architecture"""
    if config['n_dims'] is not None:
        print("\nDEBUG: Using explicitly provided n_dims")
        layer_dims = config['n_dims']
        assert len(layer_dims) == config['n_layer'], \
            f"n_dims has {len(layer_dims)} entries, but n_layer={config['n_layer']}"
        n_heads = [d // config['head_dim'] for d in layer_dims]
        use_unet = False
    else:
        print("\nDEBUG: Calculating dimensions based on model_architecture")
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

def log_memory_stats(logger, iter_num):
    """Log memory statistics via wandb.log if logger is active"""
    if logger is None:
        return
    logger.log_memory_per_gpu()

def cumulative_compute(iter_num, num_params, tokens_per_iter):
    """
    Roughly calculates a measure of cumulative compute used:
    6 * N flops per token for attention+MLP,
    multiplied by number of tokens processed so far.
    """
    N = num_params
    return (iter_num * tokens_per_iter * 6 * N)

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
    print("Command line arguments:", sys.argv)
    exec(open('configurator.py').read(), context)

    config = {k: globals()[k] for k in config_keys}

    # Process multi-dataset config
    config['datasets'] = process_datasets_config(config.get('datasets', {}))

    # Setup distributed training
    ddp, device, master_process, seed_offset, ddp_world_size = setup_distributed(config)

    # Print final config if master_process (helps avoid confusion)
    if master_process:
        print("\n------ Effective Config (after overrides) ------")
        for k, v in config.items():
            print(f"{k} = {v}")
        if config['datasets']:
            print(f"Note: The single 'dataset' ({config['dataset']}) is overridden by multi-datasets {config['datasets']}")

    # Setup model dimensions
    layer_dims, n_heads, use_unet = setup_model_dimensions(config)
    config['layer_dims'] = layer_dims
    config['n_heads'] = n_heads
    config['use_unet'] = use_unet

    # Calculate tokens per iteration
    tokens_per_iter = (config['gradient_accumulation_steps'] *
                       ddp_world_size *
                       config['batch_size'] *
                       config['block_size'])
    config['tokens_per_iter'] = tokens_per_iter
    print(f"\ntokens per iteration: {tokens_per_iter:,}")

    # Setup directories and random seed
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup dtype and autocast context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Setup dataset handler (multi or single dataset)
    dataset_handler, vocab_size, data_dir = setup_dataset_handler(config, device, seed_offset)

    # Setup wandb logging
    logger = setup_wandb_logging(config, master_process)

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
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
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
        print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        iter_num = 0
        best_val_loss = 1e9
    else:
        raise ValueError(f"Unknown init_from: {config['init_from']}")

    # Move model to device
    model.to(device)

    # Optionally compile (PyTorch 2.0+)
    if config['compile']:
        print("Compiling the model... (PyTorch 2.0+)")
        model = torch.compile(model)

    # Setup DDP if needed
    if ddp:
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model, device_ids=[ddp_local_rank])

    # Setup optimizer
    scaler = torch.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    optimizer = model.configure_optimizers(
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
    num_params = model.module.get_num_params() if ddp else model.get_num_params()
    if master_process:
        print(f"Number of parameters: {num_params/1e6:.2f}M")
        if config.get('wandb_log', False) and wandb.run is not None:
            wandb.run.summary["number_of_parameters"] = num_params

    # Training loop setup
    training_start_time = time.time()
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    # Fetch initial batch
    X, Y = get_batch('train', config, dataset_handler, data_dir, device)

    try:
        while True:
            # Decide current learning rate
            lr = (get_lr(iter_num, config) if config['decay_lr']
                  else config['learning_rate'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate every eval_interval steps
            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss(model, config, dataset_handler, data_dir, device, ctx)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if logger:  # Log evaluation
                    time_elapsed = time.time() - training_start_time
                    tokens_so_far = iter_num * tokens_per_iter
                    comp = cumulative_compute(iter_num, num_params, tokens_per_iter)

                    logger.log_evaluation(iter_num, losses['train'], losses['val'], comp)
                    logger.log_scaling_metrics(tokens_so_far, comp, time_elapsed)
                    logger.log_memory_per_gpu()

                # Save checkpoint if val loss improved or always_save_checkpoint
                if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': (model.module.state_dict() if ddp
                                      else model.state_dict()),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {config['out_dir']}")
                        torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))

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
                X, Y = get_batch('train', config, dataset_handler, data_dir, device)

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
                    mfu = model.estimate_mfu(
                        config['batch_size'] * config['gradient_accumulation_steps'],
                        dt
                    )
                    running_mfu = (mfu if running_mfu < 0
                                   else 0.9 * running_mfu + 0.1 * mfu)
                print(f"iter {iter_num}: loss {lossf:.4f}, "
                      f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

                if logger:
                    time_elapsed = time.time() - training_start_time
                    tokens_so_far = iter_num * tokens_per_iter
                    comp = cumulative_compute(iter_num, num_params, tokens_per_iter)

                    logger.log_training_step(iter_num, {
                        'loss': lossf,
                        'lr': lr,
                        'mfu': running_mfu,
                        'iter_time_ms': dt * 1000,
                    })
                    
                    logger.log_scaling_metrics(tokens_so_far, comp, time_elapsed)


            iter_num += 1
            local_iter_num += 1

            # End condition
            if iter_num > config['max_iters']:
                break

            # Periodic cleanup
            if iter_num % 1000 == 0:
                cleanup_memory()

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, stopping training...")

    finally:
        # Cleanup
        if dataset_handler is not None:
            print("Cleaning up dataset handler...")
            dataset_handler.cleanup()
        if ddp:
            destroy_process_group()
        if logger:
            logger.finish()

if __name__ == '__main__':
    main()


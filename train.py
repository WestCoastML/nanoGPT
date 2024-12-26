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
from utils.wandb_logger import WandBLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

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
wandb_group = None  # For grouping related runs
wandb_run_name = 'gpt2'
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
model_architecture = 'original'  # 'original', 'diamond', 'unetxformer'
use_unet = False
print_params_only = False

# Additional parameters for diamond/unetxformer
base_dim = 384
max_dim = 1024
head_dim = 64
shape_variant = 'symmetry'  # default

# If provided, n_dims overrides architecture-based dims
n_dims = None  # e.g. [768,768,...], or keep it None to rely on model_architecture logic

# read overrides
config_path = ""
config_keys = [k for k,v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]

# explicitly allow config_path
if "config_path" not in config_keys:
    config_keys.append("config_path")

# explicitly ensure n_dims is included, even if None
if "n_dims" not in config_keys:
    config_keys.append("n_dims")

# First add debug prints to see what's coming in
print("\nDEBUG: Configuration before processing:")
print(f"base_dim (global): {globals().get('base_dim')}")
print(f"n_layer (global): {globals().get('n_layer')}")
print(f"head_dim (global): {globals().get('head_dim')}")

exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

print("\nDEBUG: Configuration after initial processing:")
print(f"base_dim (config): {config.get('base_dim')}")
print(f"n_layer (config): {config.get('n_layer')}")
print(f"head_dim (config): {config.get('head_dim')}")

# Ensure critical parameters have valid values
config['base_dim'] = int(config.get('base_dim', 384))  # Ensure integer
config['n_layer'] = int(config.get('n_layer', 12))     # Ensure integer
config['head_dim'] = int(config.get('head_dim', 64))   # Ensure integer
config['model_architecture'] = config.get('model_architecture', 'original')
config['n_dims'] = config.get('n_dims', None)

print("\nDEBUG: Configuration after validation:")
print(f"model_architecture: {config['model_architecture']}")
print(f"base_dim: {config['base_dim']}")
print(f"n_layer: {config['n_layer']}")
print(f"head_dim: {config['head_dim']}")

# Decide layer_dims and n_heads
if config['n_dims'] is not None:
    print("\nDEBUG: Using explicitly provided n_dims")
    layer_dims = config['n_dims']
    assert len(layer_dims) == config['n_layer'], f"length of n_dims ({len(layer_dims)}) must match n_layer ({config['n_layer']})"
    n_heads = [d // config['head_dim'] for d in layer_dims]
    use_unet = False  # no unet if directly specifying dims
else:
    print("\nDEBUG: Calculating dimensions based on model_architecture")
    if config['model_architecture'] == 'original':
        n_embd = config['base_dim']
        n_head = n_embd // config['head_dim']
        layer_dims = [n_embd] * config['n_layer']
        n_heads = [n_head] * config['n_layer']
        use_unet = False
        print(f"Original architecture: using {n_embd} dimensions across {config['n_layer']} layers")
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
        print(f"Diamond/Unet architecture: calculated varying dimensions across {config['n_layer']} layers")
    else:
        raise ValueError(f"model_architecture must be one of: original, diamond, unetxformer, got {config['model_architecture']}")

print("\nDEBUG: Final configuration:")
print(f"layer_dims: {layer_dims}")
print(f"n_heads: {n_heads}")

config['layer_dims'] = layer_dims
config['n_heads'] = n_heads
config['use_unet'] = use_unet

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert config['gradient_accumulation_steps'] % ddp_world_size == 0
    config['gradient_accumulation_steps'] //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    config['gradient_accumulation_steps']
    * ddp_world_size
    * config['batch_size']
    * config['block_size']
)
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config['out_dir'], exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[config['dtype']]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print("DEBUG: final dataset =", config['dataset'])
data_dir = os.path.join('data', config['dataset'])

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([
        torch.from_numpy((data[i:i + config['block_size']]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + config['block_size']]).astype(np.int64))
        for i in ix
    ])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}")

model_args = dict(
    layer_dims=config['layer_dims'],
    n_heads=config['n_heads'],
    block_size=config['block_size'],
    bias=config['bias'],
    vocab_size=None,
    dropout=config['dropout'],
    n_layer=config['n_layer'],
    model_architecture=config['model_architecture'],
    use_unet=config['use_unet'],
)

if config['init_from'] == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif config['init_from'] == 'resume':
    print(f"Resuming training from {config['out_dir']}")
    ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['layer_dims', 'n_heads', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif config['init_from'].startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
    override_args = dict(dropout=config['dropout'])
    model = GPT.from_pretrained(config['init_from'], override_args)
    gptconf = model.config
    model_args = dict(
        layer_dims=gptconf.layer_dims,
        n_heads=gptconf.n_heads,
        block_size=gptconf.block_size,
        bias=gptconf.bias,
        vocab_size=gptconf.vocab_size,
        dropout=gptconf.dropout,
        n_layer=gptconf.n_layer,
        model_architecture='original',
        use_unet=False
    )
else:
    raise ValueError("Invalid init_from specified.")

model.to(device)

if config['print_params_only']:
    print("Exiting now since print_params_only is True.")
    exit(0)

scaler = torch.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
optimizer = model.configure_optimizers(
    config['weight_decay'],
    config['learning_rate'],
    (config['beta1'], config['beta2']),
    device
)

if config['init_from'] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

if config['compile']:
    print("compiling the model... (PyTorch 2.0+)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

num_params = model.module.get_num_params() if ddp else model.get_num_params()

if config.get('wandb_log', False) and master_process and wandb.run is not None:
    wandb.run.summary["number_of_parameters"] = num_params

if master_process:
    with open(os.path.join(config['out_dir'], "config_used.txt"), "a") as f:
        f.write(f"number_of_parameters: {num_params}\n")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    if it > config['lr_decay_iters']:
        return config['min_lr']
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

if config.get('model_architecture', None) and config.get('dataset', None):
    run_name = config.get('wandb_run_name', None)
    if not run_name:
        arch = config['model_architecture']
        n_l = config['n_layer']
        last_dim = model_args['layer_dims'][-1] if 'layer_dims' in model_args else 768
        ds = config['dataset']
        run_name = f"{arch}_{n_l}L_{last_dim}D_{ds}"
else:
    run_name = config.get('wandb_run_name', 'run')

training_start_time = time.time()

if config.get('wandb_log', False) and master_process:
    logger = WandBLogger(config)
    if wandb.run is not None:
        wandb.run.summary.update({
            "number_of_parameters": num_params,
            "tokens_per_iter": tokens_per_iter,
            "max_expected_tokens": tokens_per_iter * config['max_iters'],
        })

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

def cumulative_compute(iter_i):
    N = num_params
    return (iter_i * tokens_per_iter * 6 * N)

while True:
    lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % config['eval_interval'] == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        time_elapsed = time.time() - training_start_time
        tokens_so_far = iter_num * tokens_per_iter
        comp = cumulative_compute(iter_num)

        if config.get('wandb_log', False) and master_process:
            logger.log_evaluation(iter_num, losses['train'], losses['val'], comp)
            metrics = {
                'lr': lr,
                'mfu': running_mfu,
                'tokens': tokens_so_far,
                'compute': comp,
                'time_elapsed': time_elapsed,
                'tokens_per_second': tokens_so_far / time_elapsed if time_elapsed > 0 else 0,
                'compute_per_second': comp / time_elapsed if time_elapsed > 0 else 0,
            }
            logger.log_training_step(iter_num, metrics)

        if losses['val'] < best_val_loss or config.get('always_save_checkpoint', False):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {config['out_dir']}")
                torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))

    if iter_num == 0 and config.get('eval_only', False):
        break

    for micro_step in range(config['gradient_accumulation_steps']):
        if ddp:
            # only sync gradients on last micro step
            model.require_backward_grad_sync = (
                micro_step == config['gradient_accumulation_steps'] - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config['gradient_accumulation_steps']
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if config['grad_clip'] != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config['log_interval'] == 0 and master_process:
        lossf = loss.item() * config['gradient_accumulation_steps']
        if local_iter_num >= 5:
            calc_mfu = (
                model.module.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                if ddp else
                model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
            )
            running_mfu = calc_mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*calc_mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        if config.get('wandb_log', False) and master_process:
            time_elapsed = time.time() - training_start_time
            tokens_so_far = iter_num * tokens_per_iter
            comp = cumulative_compute(iter_num)
            metrics = {
                'loss': lossf,
                'lr': lr,
                'mfu': running_mfu,
                'tokens': tokens_so_far,
                'compute': comp,
                'time_elapsed': time_elapsed,
                'tokens_per_second': tokens_so_far / time_elapsed if time_elapsed > 0 else 0,
                'compute_per_second': comp / time_elapsed if time_elapsed > 0 else 0,
            }
            logger.log_training_step(iter_num, metrics)

    iter_num += 1
    local_iter_num += 1
    if iter_num > config['max_iters']:
        break

if ddp:
    destroy_process_group()

if config.get('wandb_log', False) and master_process:
    logger.finish()

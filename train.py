"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
It has been extended to handle different architectures: original, diamond, unetxformer.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import wandb

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils.diamond_dim_utils import calculate_diamond_dims

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
model_architecture = 'original'  # can be 'original', 'diamond', 'unetxformer'
use_unet = False

# New config to only print parameters and exit
print_params_only = False

# read overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# Compute layer_dims and n_heads if diamond or unetxformer
if config['model_architecture'] == 'original':
    # For original: uniform dimensions based on a single embedding size (e.g., n_embd)
    n_embd = 768
    n_head = 12
    layer_dims = [n_embd]*config['n_layer']
    n_heads = [n_head]*config['n_layer']
    use_unet = False
elif config['model_architecture'] in ['diamond', 'unetxformer']:
    # Ensure base_dim, max_dim, head_dim are set
    base_dim = config.get('base_dim', 384)
    max_dim = config.get('max_dim', 1024)
    head_dim = config.get('head_dim', 64)
    layer_dims = calculate_diamond_dims(config['n_layer'], base_dim, max_dim, head_dim)
    n_heads = [dim // head_dim for dim in layer_dims]
    use_unet = (config['model_architecture'] == 'unetxformer')
else:
    raise ValueError("model_architecture must be one of: original, diamond, unetxformer")

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

tokens_per_iter = (config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size'])
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config['out_dir'], exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', config['dataset'])
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i + config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config['block_size']]).astype(np.int64)) for i in ix])
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
    # Overwrite model_args with what the model got from GPT2
    model_args = dict(layer_dims=gptconf.layer_dims,
                      n_heads=gptconf.n_heads,
                      block_size=gptconf.block_size,
                      bias=gptconf.bias,
                      vocab_size=gptconf.vocab_size,
                      dropout=gptconf.dropout,
                      n_layer=gptconf.n_layer,
                      model_architecture='original',
                      use_unet=False)

model.to(device)

# If we only want to print parameters and exit
if config['print_params_only']:
    print("Exiting now since print_params_only is True.")
    exit(0)

scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)

if config['init_from'] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

if config['compile']:
    print("compiling the model... (PyTorch 2.0+)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

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

# Prepare a descriptive run name if not provided
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

if config.get('wandb_log', False) and master_process:
    wandb.init(
        project="VSLM",
        entity="wcml",
        name=run_name,
        config=config,
        tags=[config.get('model_architecture', 'original'), config['dataset']]
    )
    wandb.watch(model, log="all")

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
while True:
    lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % config['eval_interval'] == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config.get('wandb_log', False):
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
                "tokens": iter_num * tokens_per_iter
            })
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
            model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
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
            mfu = model.module.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt) if ddp else model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu+0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        if config.get('wandb_log', False):
            wandb.log({
                "iter": iter_num,
                "train/loss_step": lossf,
                "lr": lr,
                "mfu": running_mfu*100,
                "tokens": iter_num * tokens_per_iter
            })

    iter_num += 1
    local_iter_num += 1
    if iter_num > config['max_iters']:
        break

if ddp:
    destroy_process_group()

if config.get('wandb_log', False) and master_process:
    wandb.finish()

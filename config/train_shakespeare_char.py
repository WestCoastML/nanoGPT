# config/train_shakespeare_char.py
# This config file sets defaults for training a miniature character-level shakespeare model.
# good for debugging and playing on macbooks and such
# It supports switching between 'original', 'diamond', and 'unetxformer' architectures.
# The actual computation of layer_dims and n_heads based on model_architecture
# will be done in train.py after reading these configs.

# Basic training parameters
out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_architecture = 'diamond'  # can override with --model_architecture=original or --model_architecture=unetxformer

dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Parameters for diamond/unetxformer architectures
base_dim = 384
max_dim = 1024
head_dim = 64
n_layer = 12

# For 'original' architecture, train.py will use uniform dimensions.
# For 'diamond' and 'unetxformer', train.py will compute diamond-shaped layer_dims and n_heads.

# Note: No layer_dims, n_heads, or use_unet variables are computed here.
# They will be computed inside train.py based on model_architecture.

# if model_architecture == 'original':
#     # For original, no dimension changes, just use a fixed dimension (e.g., n_embd)
#     n_embd = 768
#     n_head = 12
#     layer_dims = [n_embd]*n_layer
#     n_heads = [n_head]*n_layer
#     use_unet = False

# elif model_architecture == 'diamond':
#     from utils.diamond_dim_utils import calculate_diamond_dims
#     layer_dims = calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim)
#     n_heads = [dim // head_dim for dim in layer_dims]
#     use_unet = False

# elif model_architecture == 'unetxformer':
#     from utils.diamond_dim_utils import calculate_diamond_dims
#     layer_dims = calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim)
#     n_heads = [dim // head_dim for dim in layer_dims]
#     # In unetxformer, we will have skip connections enabled inside the model
#     use_unet = True

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# Optional: Add more hyperparameters or configurations to log
# config = {
#     "out_dir": out_dir,
#     "eval_interval": eval_interval,
#     "eval_iters": eval_iters,
#     "log_interval": log_interval,
#     "always_save_checkpoint": always_save_checkpoint,
#     "wandb_log": wandb_log,
#     "wandb_project": wandb_project,
#     "wandb_run_name": wandb_run_name,
#     "dataset": dataset,
#     "gradient_accumulation_steps": gradient_accumulation_steps,
#     "batch_size": batch_size,
#     "block_size": block_size,
#     "n_layer": n_layer,
#     "base_dim": base_dim,
#     "max_dim": max_dim,
#     "head_dim": head_dim,
#     "n_heads": n_heads,
#     "dropout": dropout,
#     "learning_rate": learning_rate,
#     "max_iters": max_iters,
#     "lr_decay_iters": lr_decay_iters,
#     "min_lr": min_lr,
#     "beta2": beta2,
#     "warmup_iters": warmup_iters,
#     # Add any additional parameters you want to log
# }
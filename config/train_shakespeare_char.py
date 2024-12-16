# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

from utils.diamond_dim_utils import calculate_diamond_dims

# Diamond-shaped GPT model
n_layer = 12
base_dim = 384
max_dim = 1024
head_dim = 64  # Dimension per head (constant across all layers)

layer_dims = calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim)
n_heads = [dim // head_dim for dim in layer_dims]

dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

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
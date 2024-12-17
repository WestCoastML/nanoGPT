# config/train_tinystories.py
out_dir = 'out-tinystories'
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = True
wandb_project = 'VSLM'
wandb_run_name = 'tinystories_experiment'

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

model_architecture = 'original'
dropout = 0.1
learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Define these so they exist in globals even if you're not using them yet
base_dim = 384
max_dim = 1024
head_dim = 64
n_layer = 12

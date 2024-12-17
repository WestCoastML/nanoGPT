# config/train_shakespeare.py
# Train a GPT model on the Shakespeare dataset (word-level or BPE-based).
# Adjust these settings as needed.

out_dir = 'out-shakespeare'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'shakespeare_experiment'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# Choose your model architecture: 'original', 'diamond', or 'unetxformer'
model_architecture = 'original'
dropout = 0.1
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# If using diamond/unetxformer, specify these:
# base_dim = 384
# max_dim = 1024
# head_dim = 64
# n_layer = 12

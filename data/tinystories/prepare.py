import os
import numpy as np
import tiktoken
from datasets import load_dataset
import pickle

# Load the dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories")

# TinyStories has 'train' and 'validation' splits already
train_data = dataset['train']['text']
val_data = dataset['validation']['text']

# Initialize the tokenizer (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")

def encode_and_save(split_data, filename):
    # Encode all text into a single array of tokens
    ids = []
    for text in split_data:
        # Encode the text and add an end-of-text token
        tokens = enc.encode_ordinary(text)
        tokens.append(enc.eot_token)
        ids.extend(tokens)
    arr = np.array(ids, dtype=np.uint16)  # GPT-2 vocab fits in 16 bits
    arr.tofile(filename)
    return len(arr)

data_dir = os.path.dirname(__file__)

# Encode and save train set
train_bin_path = os.path.join(data_dir, 'train.bin')
train_total_tokens = encode_and_save(train_data, train_bin_path)

# Encode and save val set
val_bin_path = os.path.join(data_dir, 'val.bin')
val_total_tokens = encode_and_save(val_data, val_bin_path)

print("Preparation of TinyStories complete!")
print(f"train.bin has {train_total_tokens:,} tokens")
print(f"val.bin has {val_total_tokens:,} tokens")

# Create meta.pkl with relevant info
meta = {
    'vocab_size': enc.max_token_value + 1,      # e.g. 50257 for GPT-2
    'train_tokens': train_total_tokens,
    'val_tokens': val_total_tokens,
}

meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved meta information to {meta_path}")

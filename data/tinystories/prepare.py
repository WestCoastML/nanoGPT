import os
import numpy as np
import tiktoken
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories")

# Split into train and validation
# TinyStories should have 'train' and 'validation' splits already
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

data_dir = os.path.dirname(__file__)
encode_and_save(train_data, os.path.join(data_dir, 'train.bin'))
encode_and_save(val_data, os.path.join(data_dir, 'val.bin'))

print("Preparation of TinyStories complete!")

"""
Prepares the OpenWebText dataset for language model training:
- Downloads from Hugging Face (requires ~54 GB in .cache)
- Splits into train/val
- Tokenizes with GPT-2 BPE
- Writes out train.bin, val.bin, and meta.pkl
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

import pickle
import multiprocessing

# number of workers in .map() call
num_proc = 8
# num_proc = multiprocessing.cpu_count()
# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading OpenWebText dataset. This may be large (~8M documents)...")
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # By default, openwebtext has only a 'train' split.
    # We'll create a tiny validation split (0.05%)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename test split to val

    print("Defining tokenization function...")
    def process(example):
        # GPT-2 BPE
        ids = enc.encode_ordinary(example['text'])
        # Add end-of-text token
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    print("Tokenizing splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # We'll now concatenate all tokens for each split into a single binary file
    data_dir = os.path.dirname(__file__)

    # For counting final # of tokens
    train_token_count = np.sum(tokenized['train']['len'], dtype=np.uint64)
    val_token_count = np.sum(tokenized['val']['len'], dtype=np.uint64)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16  # GPT-2 tokens fit in uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        # We'll write in total_batches shards
        total_batches = 1024
        idx = 0
        print(f"\nWriting {split}.bin...")
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Each shard is contiguous: we use .shard(...) in the HF dataset
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {arr_len:,} tokens to {filename}")

    # Summaries
    print(f"\ntrain.bin has {train_token_count:,} tokens")
    print(f"val.bin has   {val_token_count:,} tokens")

    # Save meta information
    meta = {
        'vocab_size': enc.max_token_value + 1,       # e.g. 50257
        'train_tokens': int(train_token_count),
        'val_tokens': int(val_token_count),
    }
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nWrote meta.pkl with vocab_size={meta['vocab_size']}, "
          f"train_tokens={meta['train_tokens']:,}, val_tokens={meta['val_tokens']:,}.")
    print("OpenWebText dataset preparation complete!")

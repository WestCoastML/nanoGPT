# saves the bookcorpus dataset to a binary file for training
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading BookCorpus dataset...")
    dataset = load_dataset("bookcorpus", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # Create train/val split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename test split to val

    print(f"Train set size: {len(split_dataset['train'])} books")
    print(f"Val set size: {len(split_dataset['val'])} books")

    # Define tokenization function
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores special tokens
        ids.append(enc.eot_token)  # add end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize both splits
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc
    )

    # Write tokenized data to binary files
    for split, dset in tokenized.items():
        # Calculate total array length
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # GPT2 tokens fit in uint16
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        print(f"\nWriting {split}.bin...")
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Process dataset in batches
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            
            # Write batch to mmap
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
        print(f"Saved {arr_len:,} tokens to {filename}")

    # Save meta information
    meta = {
        'vocab_size': enc.max_token_value + 1,
        'train_tokens': tokenized['train'].num_rows,
        'val_tokens': tokenized['val'].num_rows,
    }
    import pickle
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset preparation complete!")
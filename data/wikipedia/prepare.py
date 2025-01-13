# saves the wikipedia dataset to a binary file for training
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
    print("Loading Wikipedia dataset...")
    # Load latest English Wikipedia dump
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # Filter out short articles and redirect pages
    def is_valid_article(example):
        return len(example['text']) >= 100 and not example.get('redirect', False)

    print("Filtering dataset...")
    filtered_dataset = dataset["train"].filter(is_valid_article)

    # Create train/val split
    split_dataset = filtered_dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename test split to val

    print(f"Train set size: {len(split_dataset['train'])} articles")
    print(f"Val set size: {len(split_dataset['val'])} articles")

    # Define tokenization function
    def process(example):
        # Combine title and text with separator
        full_text = example['title'] + "\n\n" + example['text']
        ids = enc.encode_ordinary(full_text)  # encode_ordinary ignores special tokens
        ids.append(enc.eot_token)  # add end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize both splits
    tokenized = split_dataset.map(
        process,
        remove_columns=['id', 'url', 'title', 'text'],
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
        'dataset_version': "20220301.en"
    }
    import pickle
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset preparation complete!")

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Train tokens: {meta['train_tokens']:,}")
    print(f"Val tokens: {meta['val_tokens']:,}")
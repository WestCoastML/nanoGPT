import os
import sys
import json
import numpy as np
import torch
import subprocess
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import tiktoken
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingDatasetHandler:
    """Handler for managing multiple datasets for scaling law experiments"""
    
    SUPPORTED_DATASETS = {
        "openwebtext": {"prepare_script": "data/openwebtext/prepare.py"},
        "wikipedia": {"prepare_script": "data/wikipedia/prepare.py"},
        "books": {"prepare_script": "data/books/prepare.py"},
        "tinystories": {"prepare_script": "data/tinystories/prepare.py"}
    }
    
    def __init__(self,
                 datasets: Dict[str, float],  # dataset name -> sampling weight
                 data_dir: str = 'data',
                 tokenizer_name: str = 'gpt2',
                 block_size: int = 1024,
                 seed: int = 1337):
        """
        Initialize dataset handler
        
        Args:
            datasets: Dict mapping dataset names to sampling weights
            data_dir: Base directory for dataset storage
            tokenizer_name: Name of tokenizer to use (default gpt2 for consistency)
            block_size: Context length for training
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.base_data_dir = data_dir
        self.block_size = block_size
        self.seed = seed
        
        # Validate datasets
        for dataset_name in datasets:
            if dataset_name not in self.SUPPORTED_DATASETS:
                raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}")
        
        # Initialize tokenizer - use gpt2 for consistency across datasets
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Prepare each dataset and load memory maps
        self.dataset_bins = {}
        self.dataset_splits = {}
        self.dataset_meta = {}
        
        for dataset_name in datasets:
            train_path, val_path, meta = self._ensure_dataset_prepared(dataset_name)
            
            # Load memory maps
            train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
            
            self.dataset_bins[dataset_name] = {
                'train': train_data,
                'val': val_data
            }
            
            self.dataset_splits[dataset_name] = {
                'train': len(train_data),
                'val': len(val_data)
            }
            
            self.dataset_meta[dataset_name] = meta
            
        # Calculate total sizes
        self.total_tokens = {
            'train': sum(splits['train'] for splits in self.dataset_splits.values()),
            'val': sum(splits['val'] for splits in self.dataset_splits.values())
        }
        
        # Calculate sampling probabilities
        total_weight = sum(datasets.values())
        self.sample_probs = {
            name: weight/total_weight 
            for name, weight in datasets.items()
        }
        
        logger.info(f"Initialized combined dataset with {self.total_tokens['train']} training tokens and {self.total_tokens['val']} validation tokens")
        for name, prob in self.sample_probs.items():
            logger.info(f"Dataset {name}: {prob*100:.1f}% sampling probability")

    def _ensure_dataset_prepared(self, dataset_name: str) -> Tuple[str, str, Dict]:
        """
        Ensure dataset is prepared, running prepare script if needed.
        
        Args:
            dataset_name: Name of dataset to prepare
            
        Returns:
            Tuple of (train_path, val_path, meta_dict) for the prepared dataset
        """
        dataset_dir = os.path.join(self.base_data_dir, dataset_name)
        train_path = os.path.join(dataset_dir, 'train.bin')
        val_path = os.path.join(dataset_dir, 'val.bin')
        meta_path = os.path.join(dataset_dir, 'meta.pkl')
        
        # Check if already prepared
        if os.path.exists(train_path) and os.path.exists(val_path):
            logger.info(f"Found prepared dataset at {dataset_dir}")
            
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    import pickle
                    meta = pickle.load(f)
            
            return train_path, val_path, meta
            
        logger.info(f"Dataset {dataset_name} not found. Running prepare script...")
        
        # Get prepare script path
        prepare_script = self.SUPPORTED_DATASETS[dataset_name]["prepare_script"]
        prepare_script = os.path.join(os.getcwd(), prepare_script)
        
        if not os.path.exists(prepare_script):
            raise FileNotFoundError(f"Prepare script not found at {prepare_script}")
        
        # Create dataset directory
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Run prepare script
        try:
            completed = subprocess.run(
                [sys.executable, prepare_script],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Prepare script output:\n{completed.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Prepare script failed with error:\n{e.stderr}")
            raise RuntimeError(f"Failed to prepare dataset {dataset_name}")
            
        # Verify files were created
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise RuntimeError(f"Prepare script completed but files not found at {dataset_dir}")
            
        # Load meta if available
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                import pickle
                meta = pickle.load(f)
                
        return train_path, val_path, meta

    def get_batch(self, 
                 batch_size: int,
                 split: str = 'train',
                 device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data, sampling from datasets according to their weights
        
        Args:
            batch_size: Number of sequences per batch
            split: Either 'train' or 'val'
            device: Optional device to put tensors on
            
        Returns:
            Tuple of input and target tensors of shape (batch_size, block_size)
        """
        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {split}")
            
        x = torch.zeros((batch_size, self.block_size), dtype=torch.long)
        y = torch.zeros((batch_size, self.block_size), dtype=torch.long)
        
        # Sample which dataset to use for each sequence in batch
        dataset_choices = np.random.choice(
            list(self.datasets.keys()),
            size=batch_size,
            p=list(self.sample_probs.values())
        )
        
        for i, dataset_name in enumerate(dataset_choices):
            # Get data for this dataset
            data = self.dataset_bins[dataset_name][split]
            data_len = self.dataset_splits[dataset_name][split]
            
            # Sample random chunk
            start_idx = np.random.randint(0, data_len - self.block_size - 1)
            chunk = data[start_idx:start_idx + self.block_size + 1]
            
            # Create input/target pair
            x[i] = torch.from_numpy((chunk[:-1]).astype(np.int64))
            y[i] = torch.from_numpy((chunk[1:]).astype(np.int64))
        
        # Move to device if specified
        if device is not None:
            x = x.to(device)
            y = y.to(device)
            
        return x, y
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the combined dataset"""
        stats = {
            'total_tokens': self.total_tokens,
            'datasets': {}
        }
        
        for name in self.datasets:
            stats['datasets'][name] = {
                'train_tokens': self.dataset_splits[name]['train'],
                'val_tokens': self.dataset_splits[name]['val'],
                'sampling_prob': self.sample_probs[name],
                'meta': self.dataset_meta[name]
            }
            
        return stats

    def calculate_stats(self, data, vocab_size):
        """Helper function to calculate stats for a single dataset."""
        unique_tokens = len(np.unique(data))
        total_tokens = len(data)
        
        unique_token_ratio = unique_tokens / total_tokens
        unique_token_ratio_normalized = unique_tokens / vocab_size
        repeat_token_ratio = 1 - unique_token_ratio
        
        return {
            'unique_token_ratio': unique_token_ratio,
            'unique_token_ratio_normalized': unique_token_ratio_normalized,
            'repeat_token_ratio': repeat_token_ratio
        }

    def get_memorization_stats(self) -> Dict:
        """Calculate dataset memorization statistics in parallel."""
        stats = {}
        vocab_size = 50257  # Fixed vocabulary size (50K)
        
        # Prepare data for parallel processing
        dataset_items = [(self.dataset_bins[name]['train'], vocab_size) for name in self.datasets]
        
        # Use all available CPUs
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self.calculate_stats, dataset_items)
        
        for name, result in zip(self.datasets, results):
            stats[name] = result
            
        return stats
        
    def cleanup(self):
        """Clean up memory maps"""
        for dataset_bins in self.dataset_bins.values():
            for data in dataset_bins.values():
                if hasattr(data, '_mmap') and data._mmap is not None:
                    data._mmap.close()
                    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size of tokenizer"""
        return self.tokenizer.max_token_value + 1

# Example usage
if __name__ == "__main__":
    # Example configuration for scaling law experiments
    datasets = {
        "openwebtext": 0.4,    # 40% sampling probability
        "wikipedia": 0.3,      # 30% sampling probability
        "books": 0.2,         # 20% sampling probability
        "tinystories": 0.1    # 10% sampling probability
    }
    
    handler = ScalingDatasetHandler(datasets)
    
    # Get some batches
    x, y = handler.get_batch(batch_size=4, split='train')
    print(f"Batch shapes: {x.shape}, {y.shape}")
    
    # Print dataset stats
    stats = handler.get_dataset_stats()
    print("\nDataset stats:")
    print(json.dumps(stats, indent=2))
    
    # Print memorization stats
    mem_stats = handler.get_memorization_stats()
    print("\nMemorization stats:")
    print(json.dumps(mem_stats, indent=2))
    
    handler.cleanup()
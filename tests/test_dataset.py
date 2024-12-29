import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset_handler import ScalingDatasetHandler
import torch
import json

def test_handler():
    # Test configuration
    datasets = {
        "openwebtext": 0.7,
        "wikipedia": 0.2,
        "books": 0.1
    }
    
    print("Initializing dataset handler...")
    handler = ScalingDatasetHandler(
        datasets=datasets,
        data_dir='data',
        block_size=1024,
        seed=1337
    )
    
    print("\nTesting batch generation...")
    x, y = handler.get_batch(
        batch_size=4, 
        split='train',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
    
    print("\nGetting dataset statistics...")
    stats = handler.get_dataset_stats()
    print(json.dumps(stats, indent=2))
    
    print("\nGetting memorization statistics...")
    mem_stats = handler.get_memorization_stats()
    print(json.dumps(mem_stats, indent=2))
    
    print("\nCleaning up...")
    handler.cleanup()
    print("Test complete!")

if __name__ == "__main__":
    test_handler()
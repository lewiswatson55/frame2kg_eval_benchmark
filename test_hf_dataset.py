#!/usr/bin/env python3
"""Test HuggingFace dataset loading and provide alternatives."""

import sys
from pathlib import Path

def test_huggingface_loading():
    """Test different methods of loading the HuggingFace dataset."""
    
    print("Testing HuggingFace dataset loading methods...")
    print("=" * 60)
    
    # Method 1: Direct load with trust_remote_code
    print("\n1. Trying direct load with trust_remote_code...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("lewiswatson/Frame2KG-YC2", split="validation_dev", trust_remote_code=True)
        print(f"   ✓ Success! Loaded {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Method 2: Load without split specification
    print("\n2. Trying to load all splits first...")
    try:
        from datasets import load_dataset
        all_datasets = load_dataset("lewiswatson/Frame2KG-YC2")
        print(f"   Available splits: {list(all_datasets.keys())}")
        if "validation_dev" in all_datasets:
            dataset = all_datasets["validation_dev"]
            print(f"   ✓ Success! Loaded {len(dataset)} examples")
            return dataset
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Method 3: Force redownload
    print("\n3. Trying with force_redownload...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "lewiswatson/Frame2KG-YC2", 
            split="validation_dev",
            download_mode="force_redownload"
        )
        print(f"   ✓ Success! Loaded {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Method 4: Clear cache and retry
    print("\n4. Trying after clearing cache...")
    try:
        import shutil
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/lewiswatson___frame2-kg-yc2")
        if os.path.exists(cache_dir):
            print(f"   Removing cache: {cache_dir}")
            shutil.rmtree(cache_dir)
        
        from datasets import load_dataset
        dataset = load_dataset("lewiswatson/Frame2KG-YC2", split="validation_dev")
        print(f"   ✓ Success! Loaded {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("All methods failed. Possible solutions:")
    print("1. Update datasets library: pip install -U datasets")
    print("2. Clear all HuggingFace cache: rm -rf ~/.cache/huggingface")
    print("3. Use local JSON files instead of HuggingFace dataset")
    
    return None


def create_local_fallback():
    """Create instructions for using local data as fallback."""
    
    print("\n" + "=" * 60)
    print("FALLBACK: Using Local JSON Files")
    print("=" * 60)
    print("""
If HuggingFace dataset loading continues to fail, you can:

1. Download the dataset manually:
   - Go to: https://huggingface.co/datasets/lewiswatson/Frame2KG-YC2
   - Download the validation_dev split data
   - Extract to a local directory

2. Convert to JSON format:
   Create a Python script to export the data:
   
   ```python
   # If you can load the dataset on another machine:
   from datasets import load_dataset
   import json
   from pathlib import Path
   
   dataset = load_dataset("lewiswatson/Frame2KG-YC2", split="validation_dev")
   
   output_dir = Path("ground_truth_local")
   output_dir.mkdir(exist_ok=True)
   
   for example in dataset:
       video_id = example["video_id"]
       frame_no = example["frame_number"]
       graph = example["graph"]
       
       filename = output_dir / f"{video_id}.{frame_no:03d}.json"
       with open(filename, 'w') as f:
           json.dump(graph if isinstance(graph, dict) else json.loads(graph), f)
   ```

3. Use local ground truth:
   frame2kg-eval --pred-dir ./predictions --gt ./ground_truth_local --out results.csv
""")


if __name__ == "__main__":
    dataset = test_huggingface_loading()
    
    if dataset:
        print("\n✓ Dataset loading successful!")
        print(f"  Dataset has {len(dataset)} examples")
        
        # Show a sample
        example = dataset[0]
        print(f"\nSample example keys: {list(example.keys())}")
        if "video_id" in example:
            print(f"  video_id: {example['video_id']}")
        if "frame_number" in example:
            print(f"  frame_number: {example['frame_number']}")
    else:
        create_local_fallback()

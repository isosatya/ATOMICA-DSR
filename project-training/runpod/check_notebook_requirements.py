#!/usr/bin/env python3
"""
Script to check if all required files for the predicting_affinity.ipynb notebook are present.
"""

import os
import json
import pickle
import torch
from pathlib import Path

def check_file(path, description, required=True):
    """Check if a file exists and is accessible."""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✅ {description}: {path} ({size:,} bytes)")
        return True
    else:
        status = "❌" if required else "⚠️"
        print(f"{status} {description}: {path} (MISSING)")
        return False

def check_json_file(path, description):
    """Check if a JSON file exists and is valid."""
    if check_file(path, description):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"   JSON structure: {list(data.keys())}")
            return True
        except json.JSONDecodeError:
            print(f"   ❌ Invalid JSON file")
            return False
    return False

def check_pickle_file(path, description):
    """Check if a pickle file exists and is loadable."""
    if check_file(path, description):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"   Pickle type: {type(data)}")
            if isinstance(data, list):
                print(f"   List length: {len(data)}")
            return True
        except Exception as e:
            print(f"   ❌ Error loading pickle: {e}")
            return False
    return False

def check_torch_checkpoint(path, description):
    """Check if a PyTorch checkpoint file exists and is loadable."""
    if check_file(path, description):
        try:
            # Try to load with weights_only=False to handle the error in your notebook
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            print(f"   Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            return True
        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
            return False
    return False

def main():
    print("Checking notebook requirements...")
    print("=" * 50)
    
    # Define base paths
    base_path = Path("/workspace")
    project_path = base_path / "project-training"
    
    # Required files from your notebook
    required_files = [
        (project_path / "original-model-config" / "pretrain_model_config.json", 
         "Pretrain model config", check_json_file),
        (project_path / "model-outputs" / "model-outputs" / "version_0" / "args.json", 
         "Training args config", check_json_file),
        (project_path / "model-outputs" / "model-outputs" / "version_0" / "checkpoint" / "epoch33_step3060.ckpt", 
         "Model checkpoint", check_torch_checkpoint),
        (project_path / "data" / "test_items.pkl", 
         "Test items data", check_pickle_file),
    ]
    
    # Optional files
    optional_files = [
        (project_path / "data" / "train_items.pkl", 
         "Train items data", check_pickle_file),
        (project_path / "data" / "val_items.pkl", 
         "Validation items data", check_pickle_file),
        (project_path / "predicting_affinity.ipynb", 
         "Main notebook", check_file),
    ]
    
    print("=== Required Files ===")
    missing_required = []
    for file_path, description, check_func in required_files:
        if not check_func(file_path, description):
            missing_required.append(str(file_path))
        print()
    
    print("=== Optional Files ===")
    for file_path, description, check_func in optional_files:
        check_func(file_path, description, required=False)
        print()
    
    # Check environment
    print("=== Environment Check ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Check disk space
    print("=== Storage Check ===")
    try:
        statvfs = os.statvfs(base_path)
        free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        total_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
        print(f"Available space: {free_gb:.1f} GB / {total_gb:.1f} GB")
    except Exception as e:
        print(f"Could not check disk space: {e}")
    print()
    
    # Summary
    print("=== SUMMARY ===")
    if missing_required:
        print("❌ MISSING REQUIRED FILES:")
        for file_path in missing_required:
            print(f"   - {file_path}")
        print("\nYou need to add these files to your storage before running the notebook.")
    else:
        print("✅ All required files are present!")
        print("You should be able to run the notebook successfully.")
    
    print("\nTo fix missing files:")
    print("1. Copy the missing files to your storage volume")
    print("2. Make sure the file paths match exactly")
    print("3. Re-run this check script")

if __name__ == "__main__":
    main() 
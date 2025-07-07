#!/usr/bin/env python3

import os
import sys

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Add parent directory to Python path
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
print(f"Added to Python path: {parent_dir}")

# Test imports
try:
    import models
    print("✓ Successfully imported models")
    
    from models import AffinityPredictor
    print("✓ Successfully imported AffinityPredictor")
    
    from data.dataset import PDBBindBenchmark
    print("✓ Successfully imported PDBBindBenchmark")
    
    from trainers.abs_trainer import Trainer
    print("✓ Successfully imported Trainer")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Python path: {sys.path}") 
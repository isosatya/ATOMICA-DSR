#!/usr/bin/env python3
"""
Script to fix PyTorch loading issues in the notebook.
This addresses the weights_only=True issue in PyTorch 2.6+.
"""

import torch
import torch.serialization
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def fix_pytorch_loading():
    """
    Fix PyTorch loading issues by adding safe globals.
    """
    # Add safe globals for the models that need to be loaded
    from models.affinity_predictor import AffinityPredictor
    from models.prediction_model import PredictionModel
    from models.pretrain_model import DenoisePretrainModel
    
    # Add these classes to safe globals
    torch.serialization.add_safe_globals([
        AffinityPredictor,
        PredictionModel, 
        DenoisePretrainModel
    ])
    
    print("PyTorch loading fixes applied!")
    print("Safe globals added for:")
    print("- AffinityPredictor")
    print("- PredictionModel")
    print("- DenoisePretrainModel")

def load_model_safely(checkpoint_path, **kwargs):
    """
    Safely load a model checkpoint with proper error handling.
    """
    try:
        # First try with weights_only=False (old behavior)
        print("Attempting to load model with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Loading with weights_only=False failed: {e}")
        
        try:
            # Try with weights_only=True and safe globals
            print("Attempting to load model with weights_only=True...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return checkpoint
        except Exception as e2:
            print(f"Loading with weights_only=True failed: {e2}")
            raise Exception(f"Failed to load checkpoint: {e2}")

if __name__ == "__main__":
    fix_pytorch_loading()
    print("Run this script before loading your model to fix PyTorch loading issues.") 
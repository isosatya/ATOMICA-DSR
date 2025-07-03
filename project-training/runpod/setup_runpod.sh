#!/bin/bash

# RunPod setup script for ATOMICA
echo "Setting up ATOMICA environment on RunPod..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1-mesa-glx \
    screen

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
echo "Installing PyTorch Geometric dependencies..."
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies
echo "Installing additional dependencies..."
pip install huggingface_hub

# Create necessary directories
echo "Creating directories..."
mkdir -p project-training/model_checkpoints
mkdir -p project-training/datasets
mkdir -p project-training/outputs
mkdir -p project-training/logs
mkdir -p project-training/data
mkdir -p project-training/original-model-config

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/ATOMICA-DSR"

echo "ATOMICA environment setup complete!"
echo "You can now run training commands." 
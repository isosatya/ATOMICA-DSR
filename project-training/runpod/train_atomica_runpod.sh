#!/bin/bash

# ATOMICA Training Script for RunPod
# This script trains the ATOMICA model with the specified parameters in RunPod environment

echo "Starting ATOMICA training on RunPod (atomicadsr environment)..."

# Set environment variables for RunPod
export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p /workspace/project-training/model-outputs
mkdir -p /workspace/project-training/data
mkdir -p /workspace/project-training/original-model-config

# Training parameters for RunPod
python train.py \
  --train_set /workspace/project-training/data/train_items.pkl \
  --valid_set /workspace/project-training/data/val_items.pkl \
  --task PDBBind \
  --num_workers 4 \
  --gpus 0 \
  --lr 1e-3 \
  --max_epoch 20 \
  --patience 10 \
  --atom_hidden_size 16 \
  --block_hidden_size 16 \
  --n_layers 1 \
  --edge_size 8 \
  --k_neighbors 9 \
  --max_n_vertex_per_gpu 512 \
  --max_n_vertex_per_item 256 \
  --global_message_passing \
  --save_dir /workspace/project-training/model-outputs \
  --pretrain_weights /workspace/project-training/original-model-config/pretrain_model_weights.pt \
  --pretrain_config /workspace/project-training/original-model-config/pretrain_model_config.json \
  --use_wandb \
  --run_name "training_increased_radius_20_epochs" \
  --grad_clip 1 \
  --seed 42 \
  --shuffle

echo "Training completed on RunPod!" 
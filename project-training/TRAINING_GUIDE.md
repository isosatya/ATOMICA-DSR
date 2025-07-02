# ATOMICA Training Guide

This guide explains how to run ATOMICA training in both local and RunPod environments.

## Prerequisites

### Environment Setup
The project uses the `atomicadsr` conda environment with Python 3.9.

### Required Files Structure

Before running training, ensure you have the following file structure:

```
ATOMICA-DSR/
├── project-training/
│   ├── data/
│   │   ├── train_items.pkl          # Training dataset
│   │   └── val_items.pkl            # Validation dataset
│   ├── original-model-config/
│   │   ├── pretrain_model_weights.pt    # Pre-trained model weights
│   │   └── pretrain_model_config.json   # Pre-trained model configuration
│   ├── model-outputs/               # Output directory (will be created)
│   ├── runpod/                      # RunPod deployment files
│   │   ├── runpod_startup.sh        # RunPod startup script
│   │   ├── train_atomica_runpod.sh  # RunPod training script
│   │   ├── Dockerfile               # Container configuration
│   │   ├── requirements.txt         # Python dependencies
│   │   ├── .dockerignore            # Docker build optimization
│   │   └── RUNPOD_README.md         # RunPod deployment guide
│   ├── train_atomica.sh             # Local training script
│   └── TRAINING_GUIDE.md            # This guide
├── data/                            # Data processing scripts
├── models/                          # Model architectures
├── trainers/                        # Training utilities
└── utils/                           # Utility functions
```

### Data Preparation

1. **Training/Validation Data**: Place your processed data files in `project-training/data/`
2. **Pre-trained Model**: Download from [Hugging Face](https://huggingface.co/ada-f/ATOMICA) and place in `project-training/original-model-config/`

## Local Training

### 1. Environment Setup

```bash
# Create and activate the conda environment
conda create -n atomicadsr python=3.9 -y
conda activate atomicadsr

# Install dependencies
bash setup_env.sh
```

### 2. Run Training

```bash
# Make script executable (if not already)
chmod +x project-training/train_atomica.sh

# Run training
./project-training/train_atomica.sh
```

### 3. Local Training Parameters

The local script uses these key parameters:
- **GPU**: `--gpus -1` (CPU mode)
- **Workers**: `--num_workers 0` (single process)
- **Paths**: Relative to current directory

## RunPod Training

### 1. Data Upload

Before starting training on RunPod, upload your data:

```bash
# Upload training data
scp -r project-training/data/ user@runpod-ip:/workspace/project-training/

# Upload pre-trained model
scp -r project-training/original-model-config/ user@runpod-ip:/workspace/project-training/
```

### 2. Run Training

```bash
# Make script executable
chmod +x project-training/runpod/train_atomica_runpod.sh

# Run training
./project-training/runpod/train_atomica_runpod.sh
```

### 3. RunPod Training Parameters

The RunPod script uses these optimized parameters:
- **GPU**: `--gpus 0` (GPU mode)
- **Workers**: `--num_workers 4` (multi-process)
- **Paths**: Absolute paths in `/workspace/`

## Key Differences Between Environments

| Parameter | Local | RunPod |
|-----------|-------|--------|
| GPU Usage | `--gpus -1` (CPU) | `--gpus 0` (GPU) |
| Workers | `--num_workers 0` | `--num_workers 4` |
| Paths | Relative | Absolute `/workspace/` |
| Memory | Limited by system | Optimized for GPU |

## Training Parameters Explained

### Model Architecture
- `--atom_hidden_size 16`: Hidden dimension for atom representations
- `--block_hidden_size 16`: Hidden dimension for block representations
- `--n_layers 1`: Number of message passing layers
- `--edge_size 8`: Dimension of edge embeddings
- `--k_neighbors 9`: Number of neighbors in KNN graph

### Training Configuration
- `--task PDBBind`: Training task type
- `--lr 1e-3`: Learning rate
- `--max_epoch 20`: Maximum training epochs
- `--patience 10`: Early stopping patience
- `--grad_clip 1`: Gradient clipping threshold

### Memory Management
- `--max_n_vertex_per_gpu 512`: Maximum vertices per GPU batch
- `--max_n_vertex_per_item 256`: Maximum vertices per item
- `--global_message_passing`: Enable global message passing

### Pre-training
- `--pretrain_weights`: Path to pre-trained weights
- `--pretrain_config`: Path to pre-trained configuration

## Monitoring Training

### WandB Integration
The training script includes WandB logging:
- `--use_wandb`: Enable WandB logging
- `--run_name "training_increased_radius_20_epochs"`: Experiment name

### Local Monitoring
```bash
# Check training logs
tail -f project-training/model-outputs/training.log

# Monitor GPU usage (RunPod)
nvidia-smi
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--max_n_vertex_per_gpu`
   - Reduce `--batch_size`
   - Use CPU mode locally

2. **File Not Found**
   - Check file paths are correct
   - Ensure data files are uploaded to RunPod

3. **Import Errors**
   - Verify environment setup
   - Check `PYTHONPATH` is set correctly

### Performance Optimization

1. **Local**: Use CPU mode for small datasets
2. **RunPod**: Increase `--num_workers` for faster data loading
3. **Both**: Adjust `--max_n_vertex_per_gpu` based on available memory

## Output Files

Training will create:
- Model checkpoints in `project-training/model-outputs/`
- Training logs
- WandB experiment tracking (if enabled)

## Next Steps

After training:
1. Evaluate model performance
2. Fine-tune hyperparameters if needed
3. Use trained model for inference with `get_embeddings.py` 
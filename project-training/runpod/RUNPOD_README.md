# ATOMICA on RunPod

This guide explains how to deploy and use the ATOMICA molecular interaction modeling framework on RunPod.

## Quick Start

### 1. Deploy on RunPod

1. **Create a new pod** on RunPod with:
   - **GPU**: RTX 4090, RTX 3090, or A100 (recommended)
   - **Container**: Use the provided Dockerfile
   - **Port**: 8888 (for Jupyter notebook access)

2. **Build the container** using the provided Dockerfile:
   ```bash
   docker build -t atomica-runpod .
   ```

3. **Start the container** with the startup script:
   ```bash
   ./project-training/runpod/runpod_startup.sh
   ```

### 2. Access Jupyter Notebook

Once deployed, access Jupyter notebook at:
- **URL**: `http://your-runpod-ip:8888`
- **No authentication required** (configured for RunPod)

## Usage Examples

### Data Processing

Process PDB files for embedding:
```bash
python data/process_pdbs.py \
    --data_index_file data/example/example_inputs.csv \
    --out_path project-training/outputs/processed_data.pkl \
    --interface_dist_th 8.0 \
    --fragmentation_method PS_300
```

### Model Training

Train ATOMICA model:
```bash
python train.py \
    --task pretrain_torsion \
    --train_set project-training/data/train.pkl \
    --valid_set project-training/data/valid.pkl \
    --gpus 0 \
    --save_dir project-training/model_checkpoints \
    --max_epoch 100 \
    --batch_size 16
```

### Generate Embeddings

Get embeddings from processed data:
```bash
python get_embeddings.py \
    --model_config project-training/original-model-config/config.json \
    --model_weights project-training/original-model-config/weights.pt \
    --data_path project-training/outputs/processed_data.pkl \
    --output_path project-training/outputs/embeddings.pkl \
    --batch_size 4
```

## File Structure

```
/workspace/
├── data/                                    # Data processing scripts
├── models/                                  # Model architectures
├── trainers/                                # Training utilities
├── utils/                                   # Utility functions
├── case_studies/                           # Example notebooks
├── project-training/                       # Project-specific files
│   ├── model_checkpoints/                  # Saved models
│   ├── datasets/                           # Training/validation data
│   ├── outputs/                            # Generated embeddings
│   ├── logs/                               # Log files
│   ├── data/                               # Training data
│   ├── original-model-config/              # Pre-trained model files
│   ├── runpod/                             # RunPod deployment files
│   │   ├── runpod_startup.sh               # RunPod startup script
│   │   ├── train_atomica_runpod.sh         # RunPod training script
│   │   ├── Dockerfile                      # Container configuration
│   │   ├── requirements.txt                # Python dependencies
│   │   ├── .dockerignore                   # Docker build optimization
│   │   └── RUNPOD_README.md                # This file
│   ├── train_atomica.sh                    # Local training script
│   └── TRAINING_GUIDE.md                   # Training guide
└── project-training/runpod/requirements.txt # Python dependencies
```

## Key Dependencies

- **PyTorch 2.1.1** with CUDA 11.8 support
- **e3nn 0.5.1** for geometric neural networks
- **RDKit** for molecular processing
- **Biotite** for structural biology
- **Atom3D** for molecular datasets

## Troubleshooting

### CUDA Issues
- Ensure GPU is properly detected: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size in training scripts
- Use `--max_n_vertex_per_gpu` parameter to limit memory usage

### Dependency Issues
- All dependencies are pre-installed in the Docker container
- If issues persist, check the requirements.txt file

## Model Weights

Download pre-trained models from [Hugging Face](https://huggingface.co/ada-f/ATOMICA):
- ATOMICA base model
- ATOMICA-Interface models
- ATOMICA-Ligand models for various ligands

## Support

For issues specific to this RunPod deployment, check the logs in `/workspace/project-training/logs/`.
For general ATOMICA questions, refer to the main README.md or create an issue on the original repository.

## Environment Setup

The project uses the `atomicadsr` conda environment. To set up the environment locally:

```bash
# Create and activate the environment
conda create -n atomicadsr python=3.9 -y
conda activate atomicadsr

# Install dependencies
bash setup_env.sh
``` 
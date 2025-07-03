# ATOMICA on RunPod

This guide explains how to deploy and use the ATOMICA molecular interaction modeling framework on RunPod using either **Docker (Recommended)** or **Manual Setup**.

## 🐳 Docker Approach (Recommended)

### Prerequisites
- Docker installed on your local machine
- RunPod account with GPU access
- GitHub Personal Access Token (for cloning private repositories)

### Step 1: Build the Docker Image Locally

**Note**: Docker is not available on RunPod containers, so you must build locally.

```bash
# Clone the repository (requires authentication for private repos)
git clone https://github.com/your-username/ATOMICA-DSR.git
cd ATOMICA-DSR

# Build the Docker image
docker build -t atomica-runpod -f project-training/runpod/Dockerfile .

# Tag for Docker Hub
docker tag atomica-runpod your-username/atomica-runpod:latest

# Push to Docker Hub
docker push your-username/atomica-runpod:latest
```

### Step 2: Deploy on RunPod
1. **Create a new pod** on RunPod with:
   - **GPU**: RTX 4090, RTX 3090, or A100 (recommended)
   - **Container**: Select "Custom Image"
   - **Custom Image**: `your-username/atomica-runpod:latest`
   - **Port**: 8888 (for Jupyter notebook access)

### Step 3: Download Model Weights
```bash
# Connect to your RunPod instance
ssh your-runpod-connection

# Download pre-trained model weights
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt', local_dir='project-training/original-model-config')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_config.json', local_dir='project-training/original-model-config')"
```

### Step 4: Run Training
```bash
# Start training with screen (for background execution)
screen -S training

# Run training command
python train.py \
  --train_set project-training/data/train_items.pkl \
  --valid_set project-training/data/val_items.pkl \
  --task PDBBind \
  --num_workers 4 \
  --gpus 0 \
  --lr 1e-3 \
  --max_epoch 1000 \
  --patience 10 \
  --atom_hidden_size 32 \
  --block_hidden_size 32 \
  --n_layers 4 \
  --edge_size 32 \
  --k_neighbors 8 \
  --max_n_vertex_per_gpu 512 \
  --max_n_vertex_per_item 256 \
  --global_message_passing \
  --save_dir project-training/model_checkpoints/pdbind_training \
  --pretrain_weights project-training/original-model-config/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --pretrain_config project-training/original-model-config/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --grad_clip 1 \
  --seed 42 \
  --shuffle \
  --batch_size 32

# Detach from screen: Ctrl+A, D
# Reattach: screen -r training
```

## 🔧 Manual Setup Approach

### Step 1: Deploy Basic RunPod
1. Create a new pod with any base image (Ubuntu recommended)
2. Choose GPU: RTX 4090, RTX 3090, or A100

### Step 2: Clone Repository (Authentication Required)

**GitHub Authentication Options:**

#### Option A: Personal Access Token (Recommended)
```bash
# Create a GitHub Personal Access Token:
# 1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
# 2. Generate new token with 'repo' scope
# 3. Use the token in the clone command:

git clone https://YOUR_TOKEN@github.com/your-username/ATOMICA-DSR.git
```

#### Option B: SSH Keys
```bash
# Copy your SSH key to RunPod (from local machine):
cat ~/.ssh/id_rsa | ssh your-runpod-connection "mkdir -p ~/.ssh && cat > ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa"

# Then clone using SSH:
git clone git@github.com:your-username/ATOMICA-DSR.git
```

#### Option C: Download as ZIP
```bash
# Download repository as ZIP file:
wget https://github.com/your-username/ATOMICA-DSR/archive/refs/heads/main.zip
unzip main.zip
mv ATOMICA-DSR-main ATOMICA-DSR
cd ATOMICA-DSR
```

### Step 3: Run Setup Script
```bash
# Make setup script executable
chmod +x project-training/runpod/setup_runpod.sh

# Run setup script
./project-training/runpod/setup_runpod.sh
```

### Step 4: Download Model Weights
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt', local_dir='project-training/original-model-config')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_config.json', local_dir='project-training/original-model-config')"
```

### Step 5: Run Training
Same as Docker approach Step 4.

## 📊 Monitoring Training

### GPU Usage
```bash
nvidia-smi -l 1
```

### Training Logs
```bash
# If using nohup
tail -f training.log

# If using screen
screen -r training
```

### TensorBoard (if enabled)
```bash
tensorboard --logdir project-training/model_checkpoints/pdbind_training
```

## 🔄 Model Architecture Options

### Pre-trained Model Architecture (Recommended for fine-tuning)
```bash
--atom_hidden_size 32 \
--block_hidden_size 32 \
--n_layers 4 \
--edge_size 32 \
--k_neighbors 8
```

### Custom Architecture (Train from scratch)
```bash
--atom_hidden_size 128 \
--block_hidden_size 128 \
--block_embedding_size 256 \
--pred_hidden_size 256 \
--pred_dropout 0 \
--n_layers 6 \
--edge_size 64 \
--k_neighbors 9
```

## 💾 Persistence Options

### Option 1: GitHub
- Commit training data and model weights to GitHub
- Pull on new pod deployment

### Option 2: RunPod Persistent Storage
- Mount persistent volume to `/workspace` or `/data`
- Survives pod restarts

### Option 3: External Storage
- Use S3, Google Drive, or similar
- Download on pod startup

## 🚀 Quick Start Commands

### For Docker approach:
```bash
# Build and deploy
docker build -t atomica-runpod -f project-training/runpod/Dockerfile .
docker push your-username/atomica-runpod:latest

# On RunPod
# Use custom image: your-username/atomica-runpod:latest
# Download weights and run training
```

### For Manual approach:
```bash
# On RunPod
git clone https://YOUR_TOKEN@github.com/your-username/ATOMICA-DSR.git
cd ATOMICA-DSR
./project-training/runpod/setup_runpod.sh
# Download weights and run training
```

## 🐛 Troubleshooting

### GitHub Authentication Issues
- **"Repository not found"**: Check repository URL and token permissions
- **"Authentication failed"**: Verify Personal Access Token is correct
- **"Permission denied"**: Ensure token has 'repo' scope for private repositories

### CUDA Issues
- Ensure GPU is detected: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Reduce vertex limits: `--max_n_vertex_per_gpu 256`

### Dependency Issues
- All dependencies are pre-installed in Docker image
- For manual setup, run setup script again

### Training Issues
- Use pre-trained architecture for fine-tuning
- Remove pre-trained weights for custom architecture

## 🔒 Security Notes

- **Personal Access Tokens**: Store securely and never commit to version control
- **SSH Keys**: Use passphrase-protected keys when possible
- **Docker Images**: Consider using private Docker registries for sensitive projects

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
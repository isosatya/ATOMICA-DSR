#!/bin/bash

# RunPod startup script for ATOMICA
echo "Starting ATOMICA on RunPod..."

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories if they don't exist
mkdir -p /workspace/project-training/model_checkpoints
mkdir -p /workspace/project-training/datasets
mkdir -p /workspace/project-training/outputs
mkdir -p /workspace/project-training/logs
mkdir -p /workspace/project-training/data
mkdir -p /workspace/project-training/original-model-config

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Check if key dependencies are installed
echo "Checking dependencies..."
python -c "import torch, e3nn, rdkit, biotite, atom3d; print('All key dependencies imported successfully')"

# Display environment information
echo "Environment: atomicadsr (RunPod)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Set up Jupyter if needed (for notebook access)
pip install jupyter notebook

# Start Jupyter notebook in background
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > /workspace/project-training/logs/jupyter.log 2>&1 &

echo "ATOMICA is ready!"
echo "Jupyter notebook is running on port 8888"
echo ""
echo "Available commands:"
echo "  python train.py --help                    # Show training options"
echo "  python get_embeddings.py --help          # Show embedding options"
echo "  python data/process_pdbs.py --help       # Show data processing options"
echo ""
echo "Example usage:"
echo "  # Process PDB files:"
echo "  python data/process_pdbs.py --data_index_file data/example/example_inputs.csv --out_path project-training/outputs/processed_data.pkl"
echo ""
echo "  # Get embeddings:"
echo "  python get_embeddings.py --model_config project-training/original-model-config/config.json --model_weights project-training/original-model-config/weights.pt --data_path project-training/outputs/processed_data.pkl --output_path project-training/outputs/embeddings.pkl"
echo ""
echo "  # Train model:"
echo "  python train.py --task pretrain_torsion --train_set project-training/data/train.pkl --valid_set project-training/data/valid.pkl --gpus 0 --save_dir project-training/model_checkpoints"

# Keep the container running
tail -f /dev/null 
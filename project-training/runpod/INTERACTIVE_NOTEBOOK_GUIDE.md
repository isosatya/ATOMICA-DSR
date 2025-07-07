# Interactive Notebook Guide for RunPod

This guide explains how to run your Jupyter notebook interactively in a RunPod environment when you only have terminal access.

## Method 1: Jupyter Lab with Browser Access (Recommended)

### Step 1: Setup Jupyter Lab
```bash
# Make the setup script executable and run it
chmod +x setup_jupyter.sh
./setup_jupyter.sh
```

### Step 2: Start Jupyter Lab
```bash
# Option A: Use the convenience script
./start_jupyter.sh

# Option B: Run directly
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 3: Access via Browser
1. Get your pod's public IP from RunPod dashboard
2. Open your browser and go to: `http://YOUR_POD_IP:8888`
3. You may need to configure port forwarding in RunPod to access port 8888

## Method 2: Background Session with Screen

### Step 1: Setup and Start
```bash
# Make the script executable and run it
chmod +x run_notebook_background.sh
./run_notebook_background.sh
```

### Step 2: Attach to Session
```bash
# Attach to see the Jupyter output
screen -r jupyter_session

# To detach: Press Ctrl+A, then D
# To list sessions: screen -ls
# To kill session: screen -S jupyter_session -X quit
```

## Method 3: Direct Notebook Execution

If you prefer to run the notebook directly without Jupyter interface:

### Step 1: Convert Notebook to Python
```bash
# Install jupyter if not already installed
pip install jupyter

# Convert notebook to Python script
jupyter nbconvert --to python predicting_affinity.ipynb
```

### Step 2: Run the Python Script
```bash
# Run the converted script
python predicting_affinity.py
```

## Method 4: Interactive Python Session

### Step 1: Start Interactive Python
```bash
# Navigate to your notebook directory
cd /workspace/project-training

# Start Python with your notebook code
python -i
```

### Step 2: Copy and Paste Code
Copy the code cells from your notebook and paste them into the Python session.

## Troubleshooting

### Port Forwarding Issues
If you can't access Jupyter Lab via browser:
1. Check if port 8888 is open in RunPod
2. Try using a different port: `jupyter lab --port=8889`
3. Use Method 2 (screen) to run in background

### GPU Issues
If you encounter GPU-related errors:
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
If you run out of memory:
```bash
# Monitor memory usage
watch -n 1 nvidia-smi

# Clear GPU cache in your notebook
torch.cuda.empty_cache()
```

## Quick Start Commands

```bash
# 1. Setup Jupyter Lab
chmod +x setup_jupyter.sh && ./setup_jupyter.sh

# 2. Start Jupyter Lab in background
./run_notebook_background.sh

# 3. Attach to session to see output
screen -r jupyter_session

# 4. Access in browser at http://YOUR_POD_IP:8888
```

## Environment Variables

Make sure these are set in your RunPod environment:
```bash
export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=0  # or appropriate GPU number
```

## File Paths

Your notebook expects these files to exist:
- `/workspace/project-training/original-model-config/pretrain_model_config.json`
- `/workspace/project-training/model-outputs/model-outputs/version_0/args.json`
- `/workspace/project-training/model-outputs/model-outputs/version_0/checkpoint/epoch33_step3060.ckpt`
- `/workspace/project-training/data/test_items.pkl`

Make sure these files are present in your RunPod environment. 
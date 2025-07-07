#!/bin/bash

# Script to check if mounted storage has all necessary project files
echo "Checking mounted storage for required project files..."
echo "=================================================="

# Define the base paths
BASE_PATH="/workspace"
PROJECT_PATH="$BASE_PATH/project-training"

# Function to check if file/directory exists
check_path() {
    local path="$1"
    local description="$2"
    
    if [ -e "$path" ]; then
        echo "✅ $description: $path"
        if [ -f "$path" ]; then
            echo "   Size: $(du -h "$path" | cut -f1)"
        elif [ -d "$path" ]; then
            echo "   Type: Directory"
            echo "   Contents: $(ls "$path" | wc -l) items"
        fi
    else
        echo "❌ $description: $path (MISSING)"
    fi
    echo ""
}

# Check main project structure
echo "=== Project Structure ==="
check_path "$PROJECT_PATH" "Project directory"
check_path "$PROJECT_PATH/data" "Data directory"
check_path "$PROJECT_PATH/original-model-config" "Original model config directory"
check_path "$PROJECT_PATH/model-outputs" "Model outputs directory"

# Check specific required files
echo "=== Required Configuration Files ==="
check_path "$PROJECT_PATH/original-model-config/pretrain_model_config.json" "Pretrain model config"
check_path "$PROJECT_PATH/model-outputs/model-outputs/version_0/args.json" "Training args config"
check_path "$PROJECT_PATH/model-outputs/model-outputs/version_0/checkpoint/epoch33_step3060.ckpt" "Model checkpoint"

# Check data files
echo "=== Required Data Files ==="
check_path "$PROJECT_PATH/data/test_items.pkl" "Test items data"
check_path "$PROJECT_PATH/data/train_items.pkl" "Train items data"
check_path "$PROJECT_PATH/data/val_items.pkl" "Validation items data"

# Check model outputs structure
echo "=== Model Outputs Structure ==="
if [ -d "$PROJECT_PATH/model-outputs" ]; then
    echo "Model outputs directory structure:"
    find "$PROJECT_PATH/model-outputs" -type f -name "*.ckpt" | head -10
    echo ""
fi

# Check available checkpoints
echo "=== Available Checkpoints ==="
if [ -d "$PROJECT_PATH/model-outputs/model-outputs/version_0/checkpoint" ]; then
    echo "Available checkpoints:"
    ls -la "$PROJECT_PATH/model-outputs/model-outputs/version_0/checkpoint/"*.ckpt 2>/dev/null || echo "No .ckpt files found"
    echo ""
fi

# Check disk space
echo "=== Storage Information ==="
echo "Available disk space:"
df -h "$BASE_PATH"
echo ""

# Check if we're in the right environment
echo "=== Environment Check ==="
echo "Current working directory: $(pwd)"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "PyTorch not installed")"
echo ""

# Check for the notebook
echo "=== Notebook Check ==="
check_path "$PROJECT_PATH/predicting_affinity.ipynb" "Main notebook"

# Summary
echo "=== SUMMARY ==="
echo "If you see ❌ marks above, those files/directories are missing and need to be added to your storage."
echo "Make sure to mount the correct storage volume that contains your project files."
echo ""
echo "To mount storage in RunPod:"
echo "1. Go to your pod settings"
echo "2. Add a volume mount"
echo "3. Point it to the directory containing your project files"
echo "4. Mount it at /workspace" 
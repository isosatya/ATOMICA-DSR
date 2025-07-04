#!/bin/bash

# Real-time Training Monitoring Script for RunPod
# This script monitors training progress and detects overfitting patterns

echo "Starting real-time training monitoring..."

# Configuration
SAVE_DIR="/workspace/project-training/model-outputs"
WANDB_PROJECT="InteractNN-PDBBind"
WANDB_RUN_NAME="training_1000_epochs_overfitting_monitoring"
CHECK_INTERVAL=300  # Check every 5 minutes

# Function to check if training is still running
check_training_status() {
    if pgrep -f "python train.py" > /dev/null; then
        return 0  # Training is running
    else
        return 1  # Training has stopped
    fi
}

# Function to get latest checkpoint epoch
get_latest_epoch() {
    if [ -d "$SAVE_DIR/checkpoint" ]; then
        latest_epoch=$(ls "$SAVE_DIR/checkpoint"/epoch*.ckpt 2>/dev/null | \
                      sed 's/.*epoch\([0-9]*\)_.*/\1/' | \
                      sort -n | tail -1)
        echo "${latest_epoch:-0}"
    else
        echo "0"
    fi
}

# Function to check GPU usage
check_gpu_usage() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=, read -r gpu_util mem_used mem_total; do
        echo "GPU: ${gpu_util}% utilization, ${mem_used}MB/${mem_total}MB memory"
    done
}

# Function to check disk space
check_disk_space() {
    df -h /workspace | tail -1 | awk '{print "Disk: " $4 " available of " $2 " total"}'
}

# Function to run overfitting analysis
run_overfitting_analysis() {
    echo "Running overfitting analysis..."
    python /workspace/project-training/monitor_overfitting.py \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$WANDB_RUN_NAME" \
        --report
}

# Main monitoring loop
echo "Monitoring training progress..."
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo "Save directory: $SAVE_DIR"
echo "WandB project: $WANDB_PROJECT"
echo "WandB run: $WANDB_RUN_NAME"
echo ""

while true; do
    echo "=========================================="
    echo "$(date): Training Status Check"
    echo "=========================================="
    
    # Check if training is running
    if check_training_status; then
        echo "✓ Training is running"
        
        # Get current epoch
        current_epoch=$(get_latest_epoch)
        echo "✓ Current epoch: $current_epoch"
        
        # Check system resources
        echo ""
        echo "System Resources:"
        check_gpu_usage
        check_disk_space
        
        # Check for recent checkpoints
        if [ -d "$SAVE_DIR/checkpoint" ]; then
            recent_checkpoints=$(find "$SAVE_DIR/checkpoint" -name "epoch*.ckpt" -mtime -1 | wc -l)
            echo "✓ Recent checkpoints (last 24h): $recent_checkpoints"
        fi
        
        # Run overfitting analysis every 10 checks (50 minutes)
        if [ $((SECONDS % (CHECK_INTERVAL * 10))) -eq 0 ]; then
            echo ""
            run_overfitting_analysis
        fi
        
    else
        echo "✗ Training has stopped"
        echo ""
        echo "Final Analysis:"
        run_overfitting_analysis
        echo ""
        echo "Training completed. Exiting monitor."
        break
    fi
    
    echo ""
    echo "Next check in ${CHECK_INTERVAL} seconds..."
    echo ""
    
    sleep $CHECK_INTERVAL
done 
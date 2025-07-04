# Overfitting Monitoring Guide for ATOMICA Training

This guide explains how to monitor and prevent overfitting when training the ATOMICA model for many epochs (e.g., 1000) on RunPod cloud.

## Built-in Overfitting Prevention

### 1. Early Stopping with Patience
The ATOMICA training framework includes automatic early stopping:

```bash
# Configure patience for 1000 epochs
--patience 50  # Stop if validation doesn't improve for 50 epochs
--warmup_epochs 10  # Don't start early stopping until epoch 10
```

### 2. Comprehensive WandB Logging
The training automatically logs detailed metrics:

**Training Metrics:**
- `train_loss` - Per-batch training loss
- `train_epoch_loss` - Average training loss per epoch
- `train_last500_loss` - Rolling average of last 500 batches
- `lr` - Learning rate
- `param_grad_norm` - Gradient norm

**Validation Metrics:**
- `val_loss` - Validation loss
- `val_RMSELoss` - Root mean square error
- `val_pearson` - Pearson correlation
- `val_spearman` - Spearman correlation

## How to Monitor Training

### 1. Real-time Monitoring on RunPod

Start the monitoring script in a separate terminal:

```bash
# Make the script executable
chmod +x /workspace/project-training/runpod/monitor_training.sh

# Start monitoring
/workspace/project-training/runpod/monitor_training.sh
```

This script will:
- Check training status every 5 minutes
- Monitor GPU usage and disk space
- Track current epoch progress
- Run overfitting analysis every 50 minutes

### 2. WandB Dashboard Monitoring

Access your WandB dashboard to monitor:

1. **Training vs Validation Loss Curves**
   - Look for divergence between training and validation loss
   - Validation loss should not increase while training loss decreases

2. **Loss Difference Plot**
   - Plot: `val_loss - train_loss`
   - If this difference grows consistently, overfitting is occurring

3. **Learning Rate Schedule**
   - Monitor if learning rate is decreasing appropriately
   - Sudden drops might indicate convergence issues

### 3. Manual Overfitting Analysis

Run the overfitting analysis script:

```bash
# Generate comprehensive report
python /workspace/project-training/monitor_overfitting.py \
    --save_dir /workspace/project-training/model-outputs \
    --wandb_project "InteractNN-PDBBind" \
    --wandb_run_name "training_1000_epochs_overfitting_monitoring" \
    --report \
    --plot
```

## Signs of Overfitting

### 1. Validation Loss Divergence
- Validation loss increases while training loss continues to decrease
- Gap between training and validation loss grows larger

### 2. Performance Metrics Degradation
- Validation RMSE increases
- Pearson/Spearman correlations decrease
- Model performance on validation set worsens

### 3. Early Stopping Triggers
- Patience counter reaches 0
- Training stops automatically when validation doesn't improve

## Prevention Strategies

### 1. Adjust Training Parameters

```bash
# Increase patience for longer training
--patience 50

# Add warmup period
--warmup_epochs 10

# Reduce learning rate
--lr 5e-4  # Instead of 1e-3

# Add gradient clipping
--grad_clip 1
```

### 2. Model Regularization

The model already includes:
- Weight decay in optimizers (1e-3)
- Gradient clipping
- Dropout in some layers

### 3. Data Augmentation

Consider:
- Increasing data diversity
- Using larger validation sets
- Cross-validation for more robust evaluation

## Recommended Training Configuration for 1000 Epochs

```bash
python train.py \
  --train_set /workspace/project-training/data/train_items.pkl \
  --valid_set /workspace/project-training/data/val_items.pkl \
  --task PDBBind \
  --num_workers 4 \
  --gpus 0 \
  --lr 5e-4 \
  --max_epoch 1000 \
  --patience 50 \
  --warmup_epochs 10 \
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
  --run_name "training_1000_epochs_optimized" \
  --grad_clip 1 \
  --seed 42 \
  --shuffle \
  --save_topk 5
```

## Monitoring Checklist

### Before Training
- [ ] Set appropriate patience (50 for 1000 epochs)
- [ ] Configure WandB logging
- [ ] Set up monitoring script
- [ ] Check available disk space

### During Training
- [ ] Monitor WandB dashboard regularly
- [ ] Check training logs for errors
- [ ] Verify checkpoint saving
- [ ] Monitor GPU usage and memory

### After Training
- [ ] Run final overfitting analysis
- [ ] Compare best validation performance
- [ ] Save best model checkpoint
- [ ] Generate training report

## Troubleshooting

### Training Stops Too Early
- Increase patience value
- Check if validation set is too small
- Verify data quality

### Training Never Stops
- Check if patience is set too high
- Verify validation metrics are being computed
- Look for errors in validation loop

### Overfitting Detected
- Reduce learning rate
- Increase regularization
- Add more training data
- Reduce model complexity

## Best Practices

1. **Start with Conservative Settings**
   - Lower learning rate (5e-4 instead of 1e-3)
   - Higher patience (50 instead of 10)
   - Enable all monitoring

2. **Monitor Continuously**
   - Use WandB dashboard
   - Run monitoring script
   - Check logs regularly

3. **Save Multiple Checkpoints**
   - Use `--save_topk 5` to keep best 5 models
   - Compare performance across checkpoints

4. **Document Your Runs**
   - Use descriptive run names
   - Note any parameter changes
   - Record final performance metrics

## Example Monitoring Commands

```bash
# Start training
./project-training/runpod/train_atomica_runpod.sh

# In another terminal, start monitoring
./project-training/runpod/monitor_training.sh

# Check training status manually
ps aux | grep "python train.py"
ls -la /workspace/project-training/model-outputs/checkpoint/

# Run overfitting analysis
python project-training/monitor_overfitting.py \
    --save_dir project-training/model-outputs \
    --wandb_project "InteractNN-PDBBind" \
    --wandb_run_name "training_1000_epochs_overfitting_monitoring" \
    --report \
    --plot
```

This comprehensive monitoring setup will help you ensure your 1000-epoch training run doesn't overfit and stops at the optimal point. 
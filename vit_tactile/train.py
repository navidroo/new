"""
Training script for the ViT-Tiny tactile encoder with Performer attention and early-exit heads.

This script handles:
1. Training with multi-exit loss computation (weighted combination of losses at each exit point)
2. Evaluation on validation set
3. Logging of training metrics
4. Saving checkpoints

For best results, train with:
    python -m vit_tactile.train --batch_size 64 --epochs 100 --lr 1e-4 --tau 0.8
"""

import os
import argparse
import time
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from vit_tactile.model import PerformerViTTactile

# Try to import wandb, but don't fail if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DummyTactileDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    def __init__(self, size=1000, img_size=224, num_classes=10):
        self.size = size
        self.img_size = img_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image and label
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


def train_one_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device,
    epoch,
    args,
    logger=None
):
    """Train for one epoch with multi-exit loss computation."""
    model.train()
    
    # Exit weights for loss computation
    w4, w8, w12 = 0.3, 0.3, 1.0
    
    total_loss = 0
    total_correct = [0, 0, 0]  # Correct predictions at each exit
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass (returns logits from all exit heads)
        logits_list = model(data, enable_early_exit=False)
        
        # Multi-exit loss computation
        loss = 0
        for i, (logits, weight) in enumerate(zip(logits_list, [w4, w8, w12])):
            batch_loss = criterion(logits, target)
            loss += weight * batch_loss
            
            # Calculate accuracy for each exit
            pred = logits.argmax(dim=1)
            total_correct[i] += (pred == target).sum().item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        
        # Print progress
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
                  
            # Log to TensorBoard or W&B
            if logger:
                step = epoch * len(dataloader) + batch_idx
                logger.add_scalar('train/loss', loss.item(), step)
                
                # Log exit probabilities distribution
                if batch_idx % (args.log_interval * 5) == 0:
                    exit_probs = [F.softmax(logits, dim=-1).max(dim=-1)[0].detach().cpu().numpy() for logits in logits_list]
                    
                    if WANDB_AVAILABLE and args.use_wandb:
                        for i, probs in enumerate(exit_probs):
                            exit_idx = [4, 8, 12][i]
                            wandb.log({f"exit{exit_idx}/prob_hist": wandb.Histogram(probs)}, step=step)
    
    # Calculate average statistics
    avg_loss = total_loss / total_samples
    avg_acc = [correct / total_samples for correct in total_correct]
    
    end_time = time.time()
    
    # Print epoch summary
    print(f'Epoch {epoch} summary:')
    print(f'  Time: {end_time - start_time:.2f}s')
    print(f'  Loss: {avg_loss:.4f}')
    print(f'  Acc (Exit 4): {avg_acc[0]:.4f}')
    print(f'  Acc (Exit 8): {avg_acc[1]:.4f}')
    print(f'  Acc (Exit 12): {avg_acc[2]:.4f}')
    
    # Log to TensorBoard or W&B
    if logger:
        logger.add_scalar('train/epoch_loss', avg_loss, epoch)
        for i, acc in enumerate(avg_acc):
            exit_idx = [4, 8, 12][i]
            logger.add_scalar(f'train/acc_exit{exit_idx}', acc, epoch)
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    'train/epoch_loss': avg_loss,
                    f'train/acc_exit{exit_idx}': acc,
                    'epoch': epoch
                })
    
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device, epoch, args, logger=None):
    """Evaluate model on validation data."""
    model.eval()
    
    val_loss = 0
    correct = [0, 0, 0]  # Correct predictions at each exit
    early_exit_counts = [0, 0, 0]  # Count of samples that exit at each stage
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass with early exit
            outputs, exit_idx = model(data, enable_early_exit=True)
            
            # Track which exit was taken
            early_exit_counts[exit_idx] += data.size(0)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, target)
            pred = outputs.argmax(dim=1, keepdim=True)
            
            # Update statistics (only for the used exit)
            val_loss += loss.item() * data.size(0)
            correct[exit_idx] += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    # Calculate average statistics
    avg_loss = val_loss / total
    avg_acc = [correct[i] / total if early_exit_counts[i] > 0 else 0 for i in range(3)]
    exit_rates = [count / total for count in early_exit_counts]
    
    # Print validation summary
    print(f'Validation:')
    print(f'  Loss: {avg_loss:.4f}')
    print(f'  Acc (Exit 4): {avg_acc[0]:.4f}, Exit rate: {exit_rates[0]:.4f}')
    print(f'  Acc (Exit 8): {avg_acc[1]:.4f}, Exit rate: {exit_rates[1]:.4f}')
    print(f'  Acc (Exit 12): {avg_acc[2]:.4f}, Exit rate: {exit_rates[2]:.4f}')
    
    # Log to TensorBoard or W&B
    if logger:
        logger.add_scalar('val/loss', avg_loss, epoch)
        for i in range(3):
            exit_idx = [4, 8, 12][i]
            logger.add_scalar(f'val/acc_exit{exit_idx}', avg_acc[i], epoch)
            logger.add_scalar(f'val/exit_rate{exit_idx}', exit_rates[i], epoch)
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    'val/loss': avg_loss,
                    f'val/acc_exit{exit_idx}': avg_acc[i],
                    f'val/exit_rate{exit_idx}': exit_rates[i],
                    'epoch': epoch
                })
    
    return avg_loss, avg_acc, exit_rates


def main():
    parser = argparse.ArgumentParser(description='Train ViT-Tiny tactile encoder')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=64, help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--tau', type=float, default=0.8, help='confidence threshold for early exit')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    
    # Model parameters
    parser.add_argument('--feature-map-dim', type=int, default=256, help='dimension of Performer feature map')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--use-wandb', action='store_true', help='use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='vit-tactile', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PerformerViTTactile(
        num_classes=args.num_classes,
        feature_map_dim=args.feature_map_dim,
        tau=args.tau
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy datasets (replace with your actual datasets)
    train_dataset = DummyTactileDataset(size=5000, num_classes=args.num_classes)
    val_dataset = DummyTactileDataset(size=1000, num_classes=args.num_classes)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)
    
    # Setup logging
    logger = SummaryWriter(log_dir=f'runs/vit_tactile_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    # Initialize W&B if available and requested
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f'vit_tactile_tau{args.tau}_lr{args.lr}'
        )
        wandb.watch(model)
    
    # Train model
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args, logger)
        
        # Validate
        val_loss, val_acc, exit_rates = validate(model, val_loader, criterion, device, epoch, args, logger)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.save_dir) / f'vit_tactile_best_tau{args.tau}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'exit_rates': exit_rates,
                'args': vars(args)
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        
        # Also save a checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(args.save_dir) / f'vit_tactile_epoch{epoch}_tau{args.tau}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'exit_rates': exit_rates,
                'args': vars(args)
            }, checkpoint_path)
    
    # Close logger
    logger.close()
    
    # Close W&B if used
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 
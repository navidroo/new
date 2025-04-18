import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import requests
import zipfile
import tarfile
import tqdm
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
from model import VisionTransformer, ViViT, load_pretrained_weights

def download_file(url, destination, desc=None, max_retries=3):
    """
    Download a file from a URL to a destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        desc: Description for the progress bar
        max_retries: Maximum number of retry attempts
    """
    if desc is None:
        desc = f"Downloading {os.path.basename(destination)}"
    
    # Create destination directory
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            progress_bar = tqdm.tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"{desc} (Attempt {attempt+1}/{max_retries})"
            )
            
            with open(destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            
            progress_bar.close()
            
            # If download completed successfully, return
            if total_size > 0 and progress_bar.n != total_size:
                print(f"Downloaded size doesn't match expected size. Retrying ({attempt+1}/{max_retries})...")
                time.sleep(1)  # Wait before retrying
                continue
            return True
            
        except (requests.exceptions.RequestException, IOError) as e:
            progress_bar.close() if 'progress_bar' in locals() else None
            print(f"Download failed: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(2)  # Wait before retrying
    
    # If all retries failed
    raise Exception(f"Failed to download {url} after {max_retries} attempts.")

def extract_archive(archive_path, extract_path):
    """
    Extract a compressed archive.
    
    Args:
        archive_path: Path to the archive file
        extract_path: Path to extract the contents to
    """
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Extracting {os.path.basename(archive_path)} to {extract_path}...")
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_path)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        print(f"Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

def get_file_extension_from_url(url):
    """Extract file extension from URL, handling various URL formats."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # First check for .tar.gz
    if path.endswith('.tar.gz'):
        return '.tar.gz'
    
    # Then check for regular extensions
    _, ext = os.path.splitext(path)
    
    # If no extension found, default to .zip
    if not ext:
        print(f"Warning: Could not determine file extension from URL {url}. Defaulting to .zip")
        return '.zip'
    
    return ext

def check_and_download_dataset(data_dir, dataset_url=None, data_config=None):
    """
    Check if the dataset exists and download it if not.
    
    Args:
        data_dir: Directory where the dataset should be stored
        dataset_url: URL to download the dataset from
        data_config: Configuration for the dataset
    
    Returns:
        True if the dataset exists or was downloaded successfully
    """
    if data_config is None:
        data_config = {
            'required_subdirs': ['train', 'val'],
            'min_files_per_subdir': 1
        }
    
    # Convert to Path for easier manipulation
    data_dir = Path(data_dir)
    
    # Check if the dataset exists
    exists = True
    if not data_dir.exists():
        exists = False
    else:
        # Check if required subdirectories exist and have files
        for subdir in data_config['required_subdirs']:
            subdir_path = data_dir / subdir
            if not subdir_path.exists() or len(list(subdir_path.glob('*'))) < data_config['min_files_per_subdir']:
                exists = False
                break
    
    if exists:
        print(f"Dataset found at {data_dir}")
        return True
    
    # Dataset doesn't exist or is incomplete
    if dataset_url is None:
        raise ValueError(
            f"Dataset not found at {data_dir} and no download URL provided.\n"
            f"Please provide a dataset_url or manually download the dataset."
        )
    
    print(f"Dataset not found at {data_dir}. Downloading...")
    os.makedirs(data_dir.parent, exist_ok=True)
    
    # Download and extract the dataset
    file_ext = get_file_extension_from_url(dataset_url)
    archive_path = data_dir.parent / f"dataset_archive{file_ext}"
    
    try:
        # Download the dataset
        download_file(dataset_url, archive_path, "Downloading dataset")
        
        # Extract the archive
        extraction_success = extract_archive(archive_path, data_dir.parent)
        if not extraction_success:
            raise ValueError("Failed to extract the dataset archive.")
        
        # Remove the archive file after extraction
        if os.path.exists(archive_path):
            os.remove(archive_path)
        
        # Check if the dataset exists after download
        dataset_exists = True
        for subdir in data_config['required_subdirs']:
            subdir_path = data_dir / subdir
            if not subdir_path.exists():
                # In some cases, the archive might extract to a subdirectory
                # Try to find and move the correct directory structure
                potential_dirs = list(data_dir.parent.glob('**/train'))
                if potential_dirs:
                    # Found a potential dataset directory structure
                    src_dir = potential_dirs[0].parent
                    if src_dir != data_dir:
                        print(f"Found dataset at unexpected location {src_dir}, moving to {data_dir}")
                        # Move extracted contents to the target directory
                        if data_dir.exists():
                            shutil.rmtree(data_dir)
                        shutil.move(src_dir, data_dir)
                        break
                
                dataset_exists = False
                break
        
        if not dataset_exists:
            raise ValueError(
                f"Dataset download completed, but required directory structure not found.\n"
                f"The dataset archive may have a different structure than expected."
            )
        
        print(f"Dataset successfully downloaded and extracted to {data_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        # Clean up partial downloads
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise

def check_and_download_weights(weights_path, weights_url=None):
    """
    Check if the model weights exist and download them if not.
    
    Args:
        weights_path: Path where the weights should be stored
        weights_url: URL to download the weights from
    
    Returns:
        The path to the weights file
    """
    weights_path = Path(weights_path)
    
    # Check if the weights file exists
    if weights_path.exists():
        print(f"Pretrained weights found at {weights_path}")
        return str(weights_path)
    
    # Weights don't exist
    if weights_url is None:
        raise ValueError(
            f"Pretrained weights not found at {weights_path} and no download URL provided.\n"
            f"Please provide a weights_url or manually download the weights."
        )
    
    print(f"Pretrained weights not found at {weights_path}. Downloading...")
    os.makedirs(weights_path.parent, exist_ok=True)
    
    try:
        # Download the weights
        download_file(weights_url, weights_path, "Downloading pretrained weights")
        
        # Verify the file was downloaded successfully
        if not weights_path.exists() or weights_path.stat().st_size == 0:
            raise ValueError(f"Downloaded weights file is empty or does not exist.")
        
        print(f"Pretrained weights successfully downloaded to {weights_path}")
        return str(weights_path)
        
    except Exception as e:
        print(f"Error downloading weights: {e}")
        # Clean up partial downloads
        if weights_path.exists():
            os.remove(weights_path)
        raise

class TVLDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Custom dataset for tactile data
        Args:
            data_dir: Directory containing the data
            transform: Optional transforms to apply to the data
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # TODO: Modify this to load your specific data format
        # This is a placeholder implementation
        self.samples = []
        self.labels = []
        
        # Load data based on split
        split_dir = os.path.join(data_dir, split)
        
        # Implement your data loading logic here
        # Example:
        # for class_dir in os.listdir(split_dir):
        #     class_path = os.path.join(split_dir, class_dir)
        #     if os.path.isdir(class_path):
        #         class_idx = int(class_dir) or class_to_idx mapping
        #         for file in os.listdir(class_path):
        #             if file.endswith('.npy'):  # or your data format
        #                 self.samples.append(os.path.join(class_path, file))
        #                 self.labels.append(class_idx)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # TODO: Modify this to load your specific data format
        # This is a placeholder implementation
        sample_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and preprocess your tactile data
        # Example: 
        # data = np.load(sample_path)
        # if self.transform:
        #     data = self.transform(data)
        
        # Placeholder for now - replace with actual data loading
        data = torch.randn(3, 224, 224)  # Example for image data
        if self.transform:
            data = self.transform(data)
            
        return data, label

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with multiple exit heads
        if args.task_type == 'classification':
            outputs = model(inputs)
            
            # Handle multiple exit heads if enabled
            if isinstance(outputs, list):
                # Calculate loss for each exit head with increasing weights
                loss = 0
                weight_sum = 0
                for j, out in enumerate(outputs):
                    # Apply increasing weights to deeper exit heads
                    weight = (j + 1) / len(outputs)
                    weight_sum += weight
                    loss += weight * criterion(out, targets)
                
                # Normalize by weight sum
                loss = loss / weight_sum
                
                # Use the last (deepest) head for accuracy calculation
                _, predicted = outputs[-1].max(1)
            else:
                # Single head output
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
        else:
            # For regression tasks
            outputs = model(inputs)
            if isinstance(outputs, list):
                # Similar weighting for regression
                loss = 0
                weight_sum = 0
                for j, out in enumerate(outputs):
                    weight = (j + 1) / len(outputs)
                    weight_sum += weight
                    loss += weight * criterion(out, targets)
                loss = loss / weight_sum
                predicted = outputs[-1]
            else:
                loss = criterion(outputs, targets)
                predicted = outputs
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        if args.task_type == 'classification':
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (i + 1) % args.print_freq == 0:
                print(f'Epoch: {epoch} | Batch: {i+1}/{len(dataloader)} | Loss: {running_loss/(i+1):.4f} | Acc: {100.*correct/total:.2f}%')
        else:
            if (i + 1) % args.print_freq == 0:
                print(f'Epoch: {epoch} | Batch: {i+1}/{len(dataloader)} | Loss: {running_loss/(i+1):.4f}')
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, args):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if args.task_type == 'classification':
                outputs = model(inputs)
                
                # Handle multiple exit heads
                if isinstance(outputs, list):
                    # Use only the last (deepest) head for validation
                    outputs = outputs[-1]
                
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                # For regression tasks
                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    
    if args.task_type == 'classification':
        accuracy = 100. * correct / total
        print(f'Validation Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
        return avg_loss, accuracy
    else:
        print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train tactile model with pretrained weights')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--dataset_url', type=str, default=None, help='URL to download dataset if not found locally')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--weights_url', type=str, default=None, help='URL to download weights if not found locally')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base model layers')
    parser.add_argument('--freeze_n_layers', type=int, default=0, help='Number of transformer layers to freeze')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input size')
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'], help='Task type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--no_download', action='store_true', help='Disable automatic downloading of dataset and weights')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if dataset exists and download if necessary
    if not args.no_download:
        try:
            check_and_download_dataset(
                args.data_dir, 
                dataset_url=args.dataset_url,
                data_config={
                    'required_subdirs': ['train', 'val'],
                    'min_files_per_subdir': 1
                }
            )
        except Exception as e:
            print(f"Warning: Failed to check/download dataset: {e}")
            print("Continuing with training assuming dataset is available...")
    
    # Check if pretrained weights exist and download if necessary
    pretrained_weights_path = None
    if args.pretrained_weights and not args.no_download:
        try:
            pretrained_weights_path = check_and_download_weights(
                args.pretrained_weights,
                weights_url=args.weights_url
            )
        except Exception as e:
            print(f"Warning: Failed to check/download weights: {e}")
            print("Continuing with training without pretrained weights...")
            pretrained_weights_path = args.pretrained_weights if os.path.exists(args.pretrained_weights) else None
    else:
        pretrained_weights_path = args.pretrained_weights
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    # TODO: Adjust these transforms to match your data format
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = TVLDataset(args.data_dir, transform=transform_train, split='train')
    val_dataset = TVLDataset(args.data_dir, transform=transform_val, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    # Adjust model parameters as needed for your specific model configuration
    model = VisionTransformer(
        img_size=args.input_size,
        patch_size=16,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        dynamic_early_exit=True,
        exit_layer_nums=[3, 6, 9, 12],
        performer_attention=True,
        feature_map_dim=256,
        tau=1.0
    )
    
    # Load pretrained weights if provided
    if pretrained_weights_path:
        print(f"Loading pre-trained weights from {pretrained_weights_path}")
        try:
            model = load_pretrained_weights(model, pretrained_weights_path)
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Training with randomly initialized weights")
    
    # Freeze layers if specified
    if args.freeze_base:
        print("Freezing base model layers")
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    
    elif args.freeze_n_layers > 0:
        print(f"Freezing first {args.freeze_n_layers} transformer layers")
        # Freeze only specific transformer layers
        for name, param in model.named_parameters():
            if 'blocks' in name:
                layer_num = int(name.split('.')[1])
                if layer_num < args.freeze_n_layers:
                    param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    
    model = model.to(device)
    
    # Define loss function
    if args.task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_loss = float('inf')
    best_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        train_losses.append(train_loss)
        
        # Validate
        if args.task_type == 'classification':
            val_loss, val_acc = validate(model, val_loader, criterion, device, args)
            val_losses.append(val_loss)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
        else:
            val_loss = validate(model, val_loader, criterion, device, args)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model with loss: {best_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'))
    
    print("Training complete!")
    
    # Final model save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_dir, 'final_model.pth'))

if __name__ == "__main__":
    main() 
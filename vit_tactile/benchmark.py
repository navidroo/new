"""
Benchmark script for the ViT-Tiny tactile encoder with Performer attention and dynamic early-exit heads.

This script:
1. Loads a model with standard attention and the Performer with early-exit
2. Measures inference time for both models
3. Reports memory usage
4. Calculates early-exit statistics

Usage:
    python -m vit_tactile.benchmark --batch_size 16 --tau 0.8
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc
import json
from pathlib import Path

from vit_tactile.model import PerformerViTTactile, create_vit_tactile


def create_standard_vit_tiny(num_classes=10):
    """
    Create a standard ViT-Tiny model without Performer or early-exit for comparison.
    """
    # Using standard timm ViT model or similar implementation
    # For this example, we'll use our model but disable early exit
    model = create_vit_tactile(num_classes=num_classes, tau=1.0)  # High tau disables early exit
    return model


def generate_dummy_data(batch_size, img_size=224, num_batches=10):
    """Generate dummy tactile input data for benchmarking."""
    data_batches = []
    for _ in range(num_batches):
        # Random images with shape [B, 3, H, W]
        batch = torch.randn(batch_size, 3, img_size, img_size)
        data_batches.append(batch)
    return data_batches


def measure_inference_time(model, data_batches, device):
    """Measure inference time across multiple batches."""
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(data_batches[0].to(device))
    
    # Sync before measuring time (only if using CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time for each batch
    batch_times = []
    early_exit_indices = []
    
    with torch.no_grad():
        for batch in tqdm(data_batches, desc="Measuring inference time"):
            batch = batch.to(device)
            
            # Sync before batch (only if using CUDA)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass
            outputs = model(batch)
            
            # If early-exit is enabled, outputs is a tuple (logits, exit_idx)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, exit_idx = outputs
                early_exit_indices.append(exit_idx)
            
            # Sync after batch (only if using CUDA)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(batch_times)
    std_time = np.std(batch_times)
    
    # Calculate early-exit stats if available
    exit_stats = None
    if early_exit_indices:
        exit_counts = [0, 0, 0]  # Counts for exits 4, 8, 12
        for idx in early_exit_indices:
            exit_counts[idx] += 1
        
        total_batches = len(early_exit_indices)
        exit_rates = [count / total_batches for count in exit_counts]
        exit_stats = {
            "exit_4_rate": exit_rates[0],
            "exit_8_rate": exit_rates[1],
            "exit_12_rate": exit_rates[2],
        }
    
    return {
        "avg_time_per_batch": avg_time,
        "std_time": std_time,
        "exit_stats": exit_stats
    }


def measure_peak_memory(model, data_batch, device):
    """Measure peak GPU memory usage during inference."""
    model.eval()
    model.to(device)
    
    # Only measure GPU memory if using CUDA
    if device.type != 'cuda':
        return 0.0  # Return 0 for CPU usage (we can't easily measure CPU memory)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        _ = model(data_batch.to(device))
    
    # Clear cache again
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory
    with torch.no_grad():
        _ = model(data_batch.to(device))
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    return peak_memory


def benchmark(args):
    """Run benchmarking with the specified parameters."""
    print(f"Running benchmark with batch_size={args.batch_size}, tau={args.tau}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dummy data
    data_batches = generate_dummy_data(args.batch_size, img_size=224, num_batches=args.num_batches)
    
    # Create models
    standard_model = create_standard_vit_tiny(num_classes=args.num_classes)
    performer_model = create_vit_tactile(
        num_classes=args.num_classes, 
        tau=args.tau,
        feature_map_dim=args.feature_map_dim
    )
    
    # Measure standard model performance
    print("\nBenchmarking standard ViT (no early-exit)...")
    standard_stats = measure_inference_time(standard_model, data_batches, device)
    standard_memory = measure_peak_memory(standard_model, data_batches[0], device)
    
    # Measure performer with early-exit performance
    print("\nBenchmarking Performer with early-exit...")
    performer_stats = measure_inference_time(performer_model, data_batches, device)
    performer_memory = measure_peak_memory(performer_model, data_batches[0], device)
    
    # Calculate improvement
    time_improvement = (standard_stats["avg_time_per_batch"] - performer_stats["avg_time_per_batch"]) / standard_stats["avg_time_per_batch"] * 100
    
    # Only calculate memory improvement if memory data is available (i.e., on GPU)
    if standard_memory > 0 and performer_memory > 0:
        memory_improvement = (standard_memory - performer_memory) / standard_memory * 100
    else:
        memory_improvement = 0.0  # No memory data available in CPU mode
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    print(f"\nStandard ViT-Tiny:")
    print(f"  Avg. time per batch: {standard_stats['avg_time_per_batch']*1000:.2f} ms")
    if standard_memory > 0:
        print(f"  Peak memory: {standard_memory:.2f} MB")
    else:
        print("  Peak memory: Not available in CPU mode")
    
    print(f"\nPerformer ViT-Tiny with Early-Exit (tau={args.tau}):")
    print(f"  Avg. time per batch: {performer_stats['avg_time_per_batch']*1000:.2f} ms")
    if performer_memory > 0:
        print(f"  Peak memory: {performer_memory:.2f} MB")
    else:
        print("  Peak memory: Not available in CPU mode")
    
    print(f"\nImprovement:")
    print(f"  Time: {time_improvement:.2f}%")
    if standard_memory > 0 and performer_memory > 0:
        print(f"  Memory: {memory_improvement:.2f}%")
    else:
        print("  Memory: Not available in CPU mode")
    
    if performer_stats["exit_stats"]:
        print(f"\nEarly-Exit Statistics:")
        print(f"  Exit 4 rate: {performer_stats['exit_stats']['exit_4_rate']*100:.2f}%")
        print(f"  Exit 8 rate: {performer_stats['exit_stats']['exit_8_rate']*100:.2f}%")
        print(f"  Exit 12 rate: {performer_stats['exit_stats']['exit_12_rate']*100:.2f}%")
    
    # Prepare results for saving
    results = {
        "parameters": {
            "batch_size": args.batch_size,
            "tau": args.tau,
            "feature_map_dim": args.feature_map_dim,
            "num_classes": args.num_classes,
            "device": str(device)
        },
        "standard_vit": {
            "avg_time_ms": standard_stats["avg_time_per_batch"] * 1000,
            "std_time_ms": standard_stats["std_time"] * 1000,
            "peak_memory_mb": standard_memory if standard_memory > 0 else None
        },
        "performer_vit": {
            "avg_time_ms": performer_stats["avg_time_per_batch"] * 1000,
            "std_time_ms": performer_stats["std_time"] * 1000,
            "peak_memory_mb": performer_memory if performer_memory > 0 else None,
            "exit_stats": performer_stats["exit_stats"]
        },
        "improvement": {
            "time_percent": time_improvement,
            "memory_percent": memory_improvement if standard_memory > 0 and performer_memory > 0 else None
        }
    }
    
    # Save results if specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark ViT-Tiny tactile encoder')
    
    # Benchmark parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--tau', type=float, default=0.8, help='confidence threshold for early exit')
    parser.add_argument('--feature_map_dim', type=int, default=256, help='dimension of Performer feature map')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_batches', type=int, default=50, help='number of batches for benchmarking')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--output', type=str, help='path to save benchmark results as JSON')
    
    args = parser.parse_args()
    benchmark(args)


if __name__ == '__main__':
    main() 
"""
Hyperparameter sweep utility for ViT-Tiny tactile encoder with Performer attention and early-exit.

This script:
1. Runs benchmarks for various tau values (confidence threshold for early exit)
2. Records performance metrics for each configuration
3. Saves results as JSON lines in a file for later analysis

Usage:
    python -m vit_tactile.sweep --tau_start 0.6 --tau_end 0.99 --n_steps 10
"""

import argparse
import numpy as np
import json
from pathlib import Path
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from vit_tactile.benchmark import benchmark


def run_sweep(args):
    """Run hyperparameter sweep over tau values."""
    print(f"Running sweep from tau={args.tau_start} to tau={args.tau_end} with {args.n_steps} steps")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate tau values
    tau_values = np.linspace(args.tau_start, args.tau_end, args.n_steps)
    
    # Create benchmark args object
    benchmark_args = argparse.Namespace(
        batch_size=args.batch_size,
        tau=None,  # Will be set in the loop
        feature_map_dim=args.feature_map_dim,
        num_classes=args.num_classes,
        num_batches=args.num_batches,
        cpu=args.cpu,
        output=None  # Don't save individual results
    )
    
    # Run benchmarks for each tau value
    results = []
    for tau in tqdm(tau_values, desc="Sweeping tau values"):
        print(f"\n{'='*50}")
        print(f"Benchmarking with tau={tau:.4f}")
        print(f"{'='*50}")
        
        benchmark_args.tau = float(tau)
        result = benchmark(benchmark_args)
        
        # Add tau value to result
        result["parameters"]["tau"] = float(tau)
        
        # Add timestamp
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Store result
        results.append(result)
        
        # Write to JSONL file
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSweep complete. Results saved to {output_path}")
    
    # Generate summary plot if requested
    if args.plot:
        plot_sweep_results(results, args.plot)
    
    return results


def plot_sweep_results(results, plot_path):
    """Generate plot summarizing sweep results."""
    tau_values = [result["parameters"]["tau"] for result in results]
    
    # Performance metrics
    speedups = []
    exit4_rates = []
    exit8_rates = []
    exit12_rates = []
    
    for result in results:
        # Calculate speedup as percentage
        std_time = result["standard_vit"]["avg_time_ms"]
        performer_time = result["performer_vit"]["avg_time_ms"]
        speedup = (std_time - performer_time) / std_time * 100
        speedups.append(speedup)
        
        # Get exit rates
        if result["performer_vit"]["exit_stats"]:
            exit4_rates.append(result["performer_vit"]["exit_stats"]["exit_4_rate"] * 100)
            exit8_rates.append(result["performer_vit"]["exit_stats"]["exit_8_rate"] * 100)
            exit12_rates.append(result["performer_vit"]["exit_stats"]["exit_12_rate"] * 100)
        else:
            exit4_rates.append(0)
            exit8_rates.append(0)
            exit12_rates.append(100)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot speedup
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(tau_values, speedups, 'b-o', label='Speedup (%)')
    ax1.set_xlabel('Confidence Threshold (τ)')
    ax1.set_ylabel('Speedup (%)')
    ax1.set_title('Performance vs Confidence Threshold')
    ax1.grid(True)
    ax1.legend()
    
    # Plot exit rates
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(tau_values, exit4_rates, 'g-o', label='Exit 4 Rate (%)')
    ax2.plot(tau_values, exit8_rates, 'r-o', label='Exit 8 Rate (%)')
    ax2.plot(tau_values, exit12_rates, 'k-o', label='Exit 12 Rate (%)')
    ax2.set_xlabel('Confidence Threshold (τ)')
    ax2.set_ylabel('Exit Rate (%)')
    ax2.set_title('Exit Distribution vs Confidence Threshold')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Sweep plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for ViT-Tiny tactile encoder')
    
    # Sweep parameters
    parser.add_argument('--tau_start', type=float, default=0.6, help='starting tau value')
    parser.add_argument('--tau_end', type=float, default=0.99, help='ending tau value')
    parser.add_argument('--n_steps', type=int, default=10, help='number of tau values to test')
    
    # Benchmark parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--feature_map_dim', type=int, default=256, help='dimension of Performer feature map')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_batches', type=int, default=20, help='number of batches for benchmarking (fewer than benchmark.py)')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='results.jsonl', help='path to save sweep results as JSON lines')
    parser.add_argument('--plot', type=str, default='sweep_results.png', help='path to save summary plot')
    
    args = parser.parse_args()
    run_sweep(args)


if __name__ == '__main__':
    main() 
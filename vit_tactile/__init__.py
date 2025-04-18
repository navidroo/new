"""
ViT-Tactile: A Vision Transformer implementation for tactile data with Performer attention and early-exit.

This package provides:
1. A ViT-Tiny tactile encoder with Performer attention for improved efficiency
2. Dynamic early-exit mechanism to accelerate inference for "easy" samples
3. Training, benchmarking, and hyperparameter tuning utilities
"""

from vit_tactile.model import PerformerViTTactile, create_vit_tactile

__version__ = "0.1.0"
__all__ = ["PerformerViTTactile", "create_vit_tactile"] 
"""
ViT-Tiny tactile encoder with Performer attention and dynamic early-exit heads.

This file implements:
1. Performer Self-Attention: A more efficient attention mechanism that approximates standard
   attention using random feature maps, reducing the computational complexity from O(N²) to O(N).
   This allows faster processing of longer sequences with lower memory requirements.

2. Dynamic Early-Exit Heads: Allows the model to exit early for "easy" samples, by attaching
   classification heads at intermediate layers. If a prediction has high confidence (max_prob >= tau),
   the model returns that prediction without computing the rest of the network, saving computation.

Usage:
    from vit_tactile.model import PerformerViTTactile
    
    # Create model with default settings
    model = PerformerViTTactile(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        feature_map_dim=256,
        tau=0.8
    )
    
    # Forward pass with early-exit during inference
    logits, exit_idx = model(x)
    
    # Forward pass during training (computes all exit heads)
    logits4, logits8, logits12 = model(x, enable_early_exit=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import AttentionBuilder
from fast_transformers.feature_maps import Favor
from einops import rearrange
from einops.layers.torch import Rearrange


class ExitHead(nn.Module):
    """Classification head for early-exit points."""
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: [B, N, D]  (N patches)
        z = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, D]
        return self.fc(z)  # [B, C]


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # B, C, H, W -> B, D, H//P, W//P -> B, D, N -> B, N, D
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PerformerViTTactile(nn.Module):
    """
    ViT-Tiny with Performer attention and dynamic early-exit heads.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,  # ViT-Tiny hidden dimension
        depth: int = 12,
        num_heads: int = 3,    # ViT-Tiny number of heads
        mlp_ratio: float = 4.,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        feature_map_dim: int = 256,  # Performer feature map dimension
        tau: float = 0.8,  # Early-exit confidence threshold
        exit_idxs: List[int] = [4, 8, 12],  # Blocks to attach exit heads to
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Early-exit related
        self.exit_idxs = exit_idxs
        self.tau = tau
        self.exit_heads = nn.ModuleList([
            ExitHead(embed_dim, num_classes) for _ in range(len(exit_idxs))
        ])
        
        # Initialize Performer transformer blocks
        self.blocks = nn.ModuleList()
        
        # Create Performer attention builder
        attention_builder = AttentionBuilder.from_kwargs(
            attention_dropout=attn_drop_rate,
            attention_type="linear",
            feature_map=Favor(n_dims=feature_map_dim)
        )
        
        # Create builder for each transformer block
        builder = TransformerEncoderBuilder.from_kwargs(
            attention_builder=attention_builder,
            n_layers=depth,
            n_heads=num_heads,
            feed_forward_dimensions=int(embed_dim * mlp_ratio),
            query_dimensions=embed_dim // num_heads,
            value_dimensions=embed_dim // num_heads,
            dropout=drop_rate,
        )
        
        # We need to handle each block individually for early-exit
        for i in range(depth):
            # Create a single-layer transformer
            single_layer_builder = TransformerEncoderBuilder.from_kwargs(
                attention_builder=attention_builder,
                n_layers=1,
                n_heads=num_heads,
                feed_forward_dimensions=int(embed_dim * mlp_ratio),
                query_dimensions=embed_dim // num_heads,
                value_dimensions=embed_dim // num_heads,
                dropout=drop_rate,
            )
            self.blocks.append(single_layer_builder.get())
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x, enable_early_exit=True):
        """Extract features with potential early exit."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks with optional early exit
        exit_outputs = []
        exit_indices = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Check if we need to compute early exit at this block
            if (i + 1) in self.exit_idxs:
                exit_idx = self.exit_idxs.index(i + 1)
                exit_logits = self.exit_heads[exit_idx](x)
                exit_outputs.append(exit_logits)
                
                # During inference with early exit enabled, we check confidence
                if enable_early_exit and self.training is False:
                    # Check if prediction confidence exceeds threshold
                    probs = F.softmax(exit_logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    
                    # If all samples in batch exceed threshold, we can exit
                    if torch.all(max_probs >= self.tau):
                        return exit_outputs, exit_idx
        
        # Apply final normalization
        x = self.norm(x)
        
        # If no early exit was taken or during training, return all exit outputs
        return exit_outputs, len(self.exit_idxs) - 1
    
    def forward(self, x, enable_early_exit=True):
        """Forward pass with early exit logic."""
        exit_outputs, exit_idx = self.forward_features(x, enable_early_exit)
        
        if self.training or not enable_early_exit:
            # During training, return logits from all exit heads
            return exit_outputs
        else:
            # During inference, return the logits from the chosen exit
            return exit_outputs[exit_idx], exit_idx


def create_vit_tactile(num_classes=1000, tau=0.8, feature_map_dim=256):
    """Helper function to create a ViT-Tiny tactile encoder with default settings."""
    return PerformerViTTactile(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,  # ViT-Tiny hidden dim
        depth=12,
        num_heads=3,    # ViT-Tiny num heads
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        feature_map_dim=feature_map_dim,
        tau=tau
    ) 
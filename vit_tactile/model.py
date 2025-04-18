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
import math
from einops import rearrange


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


# Orthogonal random features for Performer attention
def orthogonal_matrix_chunk(cols, device=None):
    """Creates a random orthogonal matrix chunk."""
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block)
    return q


def gaussian_orthogonal_random_matrix(n_rows, n_cols, scaling=0, device=None):
    """Creates a random orthogonal matrix with Gaussian values."""
    nb_full_blocks = int(n_rows / n_cols)
    block_list = []
    
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(n_cols, device=device)
        block_list.append(q)
    
    remaining_rows = n_rows - nb_full_blocks * n_cols
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(n_cols, device=device)
        block_list.append(q[:remaining_rows])
    
    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        multiplier = torch.randn((n_rows, 1), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(n_cols))) * torch.ones((n_rows,), device=device)
    else:
        raise ValueError(f"Invalid scaling {scaling}")
    
    return torch.diag(multiplier) @ final_matrix


class PerformerAttention(nn.Module):
    """Performer attention mechanism using random projections."""
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.,
        feature_dim=256,
        kernel_type="exp"
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.feature_dim = feature_dim
        self.kernel_type = kernel_type
        self.head_dim = head_dim
        
        # Initialize projection matrix
        # We'll create it on the first forward pass to ensure it's on the right device
        self.projection_matrix = None
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        
        # Create or move projection matrix to the right device if needed
        if self.projection_matrix is None or self.projection_matrix.device != x.device:
            self.projection_matrix = gaussian_orthogonal_random_matrix(
                n_rows=self.feature_dim,
                n_cols=self.head_dim,
                scaling=0,
                device=x.device
            )
        
        # Apply random projection to queries and keys
        proj_q = self.feature_map(q)  # (B, H, N, F)
        proj_k = self.feature_map(k)  # (B, H, N, F)
        
        # Linear attention
        # (B, H, N, F) @ (B, H, F) -> (B, H, N)
        k_cumsum = torch.sum(proj_k, dim=2)  # (B, H, F)
        D_inv = 1.0 / torch.einsum('bhnd,bhd->bhn', proj_q, k_cumsum)  # (B, H, N)
        
        # Compute attention
        # (B, H, N, F) @ (B, H, M, F).transpose(-1, -2) -> (B, H, N, M)
        attn = torch.einsum('bhnd,bhmd->bhnm', proj_q, proj_k)
        
        # Apply attention to values
        # (B, H, N, M) @ (B, H, M, D) -> (B, H, N, D)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        
        # Scale by normalization factor
        out = out * D_inv.unsqueeze(-1)  # (B, H, N, D)
        
        # Reshape to original dimensions
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        
        # Project to output dimension
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
    def feature_map(self, x):
        """Apply feature map to input based on the kernel type."""
        if self.kernel_type == "exp":
            # Explicit feature map for e^{qk^T/sqrt(d)}
            # Project x from (B, H, N, D) to (B, H, N, F)
            projection = torch.matmul(x, self.projection_matrix.t())
            
            # Normalize and apply exponential
            norm_x = torch.sum(x**2, dim=-1, keepdim=True) / 2.0
            return torch.exp(projection - norm_x) * (self.head_dim ** -0.25)
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented")


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with Performer Attention."""
    
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        feature_dim=256,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PerformerAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            feature_dim=feature_dim,
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Work with different rank tensors: B, ..., *dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Build transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                feature_dim=feature_map_dim,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize pos_embed and cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize all other weights
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
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
from torch import nn, einsum
import numpy as np
from einops import rearrange
from typing import Union, List

from .utils.helpers import CyclicShift3D


class Residual3D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return self.fn(x, **kwargs)

class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.net(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.drop(x)
        return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: Union[int, List[int]]):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted

        if self.shifted:
            displacement = np.array([s // 2 for s in window_size])
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # Rearrange x to have feature dimension (C) as the last axis
        b, c, d, h, w = x.shape  # [batch_size, channels, depth, height, width]
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # Shape: [B, D*H*W, C]
        if self.shifted:
            x = self.cyclic_shift(x)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into query, key, value

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # [B, heads, tokens, head_dim]
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Compute attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B, heads, tokens, tokens]
        attn = self.softmax(dots)  # Softmax over tokens
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, heads, tokens, head_dim]

        # Merge heads and project back
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, tokens, inner_dim]
        out = self.to_out(out)  # Project to original feature size
        # Rearrange back to original spatial shape
        out = rearrange(out, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)  # Restore to [B, C, D, H, W]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

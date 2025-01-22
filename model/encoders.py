import torch.nn as nn
from typing import Union, List
from .swin_blocks import Residual3D, WindowAttention3D, FeedForward3D, PreNorm3D

class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]], dropout: float = 0.0):
        super().__init__()
        self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size)))
        self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, dim, layers, downscale_factor, heads, head_dim, window_size, dropout=0.0):
        super().__init__()
        self.patch_merge = nn.Conv3d(in_channels, dim, kernel_size=(1, downscale_factor, downscale_factor), stride=(1, downscale_factor, downscale_factor))
        self.layers = nn.ModuleList([
            SwinBlock3D(dim, heads, head_dim, mlp_dim=dim * 4, shifted=(i % 2 == 1), window_size=window_size, dropout=dropout) for i in range(layers)
        ])

    def forward(self, x):
        x = self.patch_merge(x)
        for layer in self.layers:
            x = layer(x)
        return x

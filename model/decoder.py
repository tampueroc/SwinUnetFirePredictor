import torch.nn as nn
from .swin_blocks import SwinBlock3D

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, layers, up_scale_factor, heads, head_dim, window_size, dropout=0.0):
        super().__init__()
        self.patch_expand = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=up_scale_factor, stride=up_scale_factor)
        self.layers = nn.ModuleList([
            SwinBlock3D(out_channels, heads, head_dim, mlp_dim=out_channels * 4, shifted=(i % 2 == 1), window_size=window_size, dropout=dropout) for i in range(layers)
        ])

    def forward(self, x):
        x = self.patch_expand(x)
        for layer in self.layers:
            x = layer(x)
        return x

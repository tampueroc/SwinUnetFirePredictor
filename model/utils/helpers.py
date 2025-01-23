import torch
import torch.nn as nn

class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = tuple(displacement)

    def forward(self, x):
        return torch.roll(x, shifts=self.displacement, dims=(1, 2, 3))


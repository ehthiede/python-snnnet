"""
Basic layers that don't depend on the order of the S_n representation.
"""

import torch


class BiasLayer(torch.nn.Module):
    """Simple bias layer in channels first format.

    This layer expects the first two dimension of the input tensor
    to be the batch and channel dimensions.
    """
    def __init__(self, channels):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.transpose(torch.transpose(x, 1, -1) + self.bias, 1, -1)

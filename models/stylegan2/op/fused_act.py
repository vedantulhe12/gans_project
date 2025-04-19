# This is a CPU-safe dummy fallback for FusedLeakyReLU

import torch
from torch import nn
import torch.nn.functional as F

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, scale=1.0):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.scale = scale

    def forward(self, input):
        if self.bias is not None:
            input = input + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(input, negative_slope=0.2) * self.scale


def fused_leaky_relu(input, bias):
    input = input + bias.view(1, -1, 1, 1)
    return F.leaky_relu(input, negative_slope=0.2)

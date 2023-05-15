import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from proxy import proxy


"""
Progressive GAN paper introduced so called "equalized learning rate" (section 4.1).
Progressive GAN, StyleGAN, StyleGAN2, and StyleGAN2-ADA all use equalized learning rate.
The idea of equalized learning rate is explained in that section 4.1, but what we need
to know is that, instead of working with weights of individual layers as usual, we scale
them with c, He's initialization constant.

The reason why c is calculated the way it is way because of publicly available source code.
For example, from the official implementation for StyleGAN2-ADA:
    1. FC layer: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L102
    2. Conv2d layer: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L144

Essentially, the modules do exactly the same things as their "un-equalized" counterparts,
we just need to rescale the weights a bit.
"""


class EqualizedLinear(nn.Module):
    # Module is very similar as nn.Linear
    def __init__(self, in_features, out_features, bias_init=0):
        super().__init__()

        self.c = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.ones(out_features) * bias_init)  # bias will be initialized to 0 or 1 typically

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor of shape [batch_size, in_features] (usually, x = w, the intermediate latent variable
        coming from the mapping network, so in_features = dim_latent)
        """
        return F.linear(x, weight=self.weight * self.c, bias=self.bias)
    
    __call__ = proxy(forward)


class EqualizedConv2d(nn.Module):
    # Module is very similar as nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.padding = padding
        self.c = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        return F.conv2d(x, weight=self.weight * self.c, bias=self.bias, padding=self.padding)

    __call__ = proxy(forward)
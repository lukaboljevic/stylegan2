import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from general_utils.equalized import EqualizedLinear
from general_utils.proxy import proxy


class ToRGB(nn.Module):
    def __init__(self, dim_latent, in_channels, out_channels=3, kernel_size=1):
        """
        Although not visible in any of the papers (Progressive GAN, StyleGANs) that mention toRGB,
        the ToRGB layer has got very similar structure as the GeneratorConvBlock layer.
        The differences are:
            - ToRGB doesn't do weight demodulation, only modulation
            - ToRGB uses linear activation instead of leaky ReLU

        The way we know the structure of ToRGB is eg. StyleGAN2-ADA official code:
            https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L310
        or other publicly available source codes.
        """
        super().__init__()

        # The affine transformation layer that uses equalized learning rate
        # StyleGAN2 initialized affine transformation layer bias to 1
        self.affine = EqualizedLinear(dim_latent, in_channels, bias_init=1.0)

        # The (grouped!) convolution operation also uses equalized learning rate
        self.c = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # activation function is linear ("passthrough")

    def forward(self, x, w):
        """
        Parameters
        ----------
        `x` : Input tensor of shape [batch_size, in_channels, height, width]
        `w` : Intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]

        Returns
        -------
        `out` : Tensor of shape [batch_size, out_channels, height, width]
        """

        batch_size, in_channels, height, width = x.shape

        # Get style and reshape so it can be multiplied with weights
        style = self.affine(w).view(batch_size, 1, in_channels, 1, 1)

        # Modulate the rescaled weights (equalized learning rate trick)
        # Shape is [batch_size, out_channels, in_channels, kernel_size, kernel_size]
        modulated_weight = (self.weight * self.c) * style
        _, out_channels, _, kernel_size, kernel_size = modulated_weight.shape

        # ToRGB doesn't perform weight demodulation

        # Perform reshaping for group convolution
        modulated_weight = modulated_weight.view(batch_size * out_channels, in_channels, kernel_size, kernel_size)
        x = x.reshape(1, -1, height, width)  # equivalent for -1 would have been batch_size * in_channels

        # Perform group convolution on the modulated weights
        # out shape is [1, batch_size * out_channels, height, width]
        out = F.conv2d(x, weight=modulated_weight, groups=batch_size)  # padding=0 since kernel_size=1

        # Reshape back to the right size ([batch_size, out_channels, height, width]) and add to bias
        # "Reshape" bias to 1, out_channels, 1, 1 so they can be added
        out = out.reshape(-1, out_channels, height, width)
        out = out + self.bias[None, :, None, None]

        # linear activation i.e. passthrough

        return out

    __call__ = proxy(forward)

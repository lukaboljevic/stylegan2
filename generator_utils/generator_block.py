import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from general_utils.equalized import EqualizedLinear
from generator_utils.to_rgb import ToRGB


class GeneratorBlock(nn.Module):
    """
    One generator block consists of the two convolution layers with weight modulation and
    demodulation, and the ToRGB layer
    """
    def __init__(self, dim_latent, in_channels, out_channels):
        super().__init__()

        # TODO Add noise in forward
        self.conv_block1 = GeneratorConvBlock(dim_latent, in_channels, out_channels)
        self.conv_block2 = GeneratorConvBlock(dim_latent, out_channels, out_channels)
        self.to_rgb = ToRGB(dim_latent, out_channels)  # out_channels -> 3 channels

    def forward(self, x, w):
        # TODO Add noise
        out = self.conv_block1(x, w)
        out = self.conv_block2(out, w)
        rgb = self.to_rgb(out, w)

        return out, rgb



class GeneratorConvBlock(nn.Module):
    """
    One of the convolution blocks that performs group convolution using modulation and
    demodulation of rescaled weights (equalized learning rate trick).
    """
    def __init__(self, dim_latent, in_channels, out_channels, kernel_size=3, eps=1e-8):
        super().__init__()

        # The affine transformation layer that uses equalized learning rate
        self.affine = EqualizedLinear(dim_latent, in_channels, bias_init=1.0)  # StyleGAN2 initialized affine transformation layer bias to 1

        # The (grouped!) convolution operation also uses equalized learning rate
        self.c = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.eps = eps  # for numerical stability when demodulating

        # TODO Add noise, noise scaling parameter
        

    def forward(self, x, w):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, in_channels, height, width]
        w : intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]

        Returns
        -------
        out : tensor of shape [batch_size, out_channels, height, width]
        """

        batch_size, in_channels, height, width = x.shape

        # Get style and reshape so it can be multiplied with weights
        style = self.affine(w).view(batch_size, 1, in_channels, 1, 1)
        
        # Modulate the rescaled weights (equalized learning rate trick)
        # Shape is [batch_size, out_channels, in_channels, kernel_size, kernel_size]
        modulated_weight = (self.weight * self.c) * style

        # Demodulate the modulated weights
        # Shape of sigma_inverse is [batch_size, out_channels, 1, 1, 1]
        sigma_inverse = torch.rsqrt(
            (modulated_weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + self.eps
        )

        # Shape is [batch_size, out_channels, in_channels, kernel_size, kernel_size]
        demodulated_weight = modulated_weight * sigma_inverse
        _, out_channels, _, kernel_size, _ = demodulated_weight.shape

        # Perform reshaping for group convolution
        demodulated_weight = demodulated_weight.view(batch_size * out_channels, in_channels, kernel_size, kernel_size)
        x = x.reshape(1, -1, height, width)  # equivalent for -1 would have been batch_size * in_channels

        # Perform group convolution on the demodulated weights
        # out shape is [1, batch_size * out_channels, height, width]
        out = F.conv2d(x, weight=demodulated_weight, padding=1, groups=batch_size)  # padding=1 since kernel_size=3
        
        # Reshape back to the right size ([batch_size, out_channels, height, width]) and add to bias
        # "Reshape" bias to 1, out_channels, 1, 1 so they can be added
        out = out.reshape(-1, out_channels, height, width)
        out = self.activation(out + self.bias[None, :, None, None])

        # TODO noise!

        return out


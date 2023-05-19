import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generator_utils.to_rgb import ToRGB
from general_utils.equalized import EqualizedLinear
from general_utils.proxy import proxy


class GeneratorBlock(nn.Module):
    def __init__(self, dim_latent, in_channels, out_channels):
        """
        One generator block consists of the two convolution layers with weight modulation and
        demodulation, and the ToRGB layer
        """
        super().__init__()

        self.conv_block1 = GeneratorConvBlock(dim_latent, in_channels, out_channels)
        self.conv_block2 = GeneratorConvBlock(dim_latent, out_channels, out_channels)
        self.to_rgb = ToRGB(dim_latent, out_channels)  # converts out_channels to 3 channels by default

    def forward(self, x, w, noise):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, in_channels, height, width]
        w : intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]
        noise : a tuple containing two random noise tensors, one for each convolution
            block. Each noise tensor is of shape [batch_size, 1, block_resolution, block_resolution]
            where block_resolution is the resolution (image size) of this block.

        Returns
        -------
        out : tensor of shape [batch_size, out_channels, height, width]
        """
        out = self.conv_block1(x, w, noise[0])
        out = self.conv_block2(out, w, noise[1])
        rgb = self.to_rgb(out, w)

        return out, rgb

    __call__ = proxy(forward)


class GeneratorConvBlock(nn.Module):
    def __init__(self, dim_latent, in_channels, out_channels, kernel_size=3, eps=1e-8):
        """
        One of the convolution blocks that performs group convolution using modulation and
        demodulation of rescaled weights (equalized learning rate trick).
        """
        super().__init__()

        # The affine transformation layer that uses equalized learning rate
        # StyleGAN2 initialized affine transformation layer bias to 1
        self.affine = EqualizedLinear(dim_latent, in_channels, bias_init=1.0)

        # The (grouped!) convolution operation also uses equalized learning rate
        self.c = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = nn.LeakyReLU(0.2, inplace=True)  # negative_slope = 0.2 is used ever since Progressive GAN
        self.eps = eps  # for numerical stability when demodulating

        # Noise scaling parameter
        # This (as far as I'm aware) corresponds to the green "B" operation from the architectures
        self.noise_scaling_parameter = nn.Parameter(torch.zeros([]))

    def forward(self, x, w, noise):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, in_channels, height, width]
        w : intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]
        noise : random noise tensor of shape [batch_size, 1, block_resolution, block_resolution]
            where block_resolution is the resolution (image size) of this block.

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

        # Reshape back to the right size ([batch_size, out_channels, height, width])
        out = out.reshape(-1, out_channels, height, width)

        # Add the scaled noise to the current output
        out = out + self.noise_scaling_parameter * noise

        # "Reshape" bias to 1, out_channels, 1, 1 so they can be added
        out = self.activation(out + self.bias[None, :, None, None])

        return out

    __call__ = proxy(forward)

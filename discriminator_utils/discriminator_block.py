import math
import torch.nn as nn

from general_utils.equalized import EqualizedConv2d
from general_utils.upsample import Downsample
from general_utils.proxy import proxy


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        The core structure of the discriminator block used in StyleGAN2 has not been changed
        since Progressive GAN. In the paper for Progressive GAN, the structure of discriminator
        (and generator) is very nicely shown in Table 2.

        In StyleGAN2, they decided to make a slight change and use a "residual" discriminator (Figure 7c).
        """
        super().__init__()

        self.convolutions = nn.Sequential(
            EqualizedConv2d(in_channels, in_channels, kernel_size=kernel_size),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedConv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Convolution used for the residual part of the block
        self.residual_conv = EqualizedConv2d(in_channels, out_channels, kernel_size=1)

        # Downsampling used before the residual 1x1 convolution, or after the two 3x3 ones
        self.downsample = Downsample()

        # From StyleGAN2, section 4.1: "In residual network architectures, the addition of two
        # paths leads to a doubling of signal variance, which we cancel by multiplying with 1/√2.
        # This is crucial for our networks, whereas in classification resnets the issue is
        # typically hidden by batch normalization."
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        """
        Parameters
        ----------
        `x` : Input tensor of shape [batch_size, in_channels, height, width]

        Returns
        -------
        `out` : Output tensor of shape [batch_size, out_channels, height // 2, width // 2]
        """
        # Get the residual first
        residual = self.residual_conv(self.downsample(x))

        # Apply standard 3x3 convolutions and downsample
        out = self.convolutions(x)
        out = self.downsample(out)

        # Add and scale
        return (out + residual) * self.scale

    __call__ = proxy(forward)

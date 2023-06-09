import torch
import torch.nn as nn

from generator_utils.to_rgb import ToRGB
from generator_utils.generator_block import GeneratorBlock, GeneratorConvBlock
from general_utils.upsample import Upsample
from general_utils.proxy import proxy


class Generator(nn.Module):
    def __init__(self, dim_latent=512):
        """
        This module essentially corresponds to the synthesis network from the paper(s).
        """
        super().__init__()

        # Learnable constant input
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Mapping network will not be a part of the generator itself, but the Generator
        # will take the output of the mapping network as its input

        # The first block input is 4x4 with 512 channels - we consider it as separate as
        # it's the only one that operates with 4x4 size
        self.conv_block1 = GeneratorConvBlock(dim_latent, 512, 512)
        self.to_rgb1 = ToRGB(dim_latent, 512)

        # Other blocks i.e. parts of the generator
        self.blocks = nn.ModuleList(
            [
                GeneratorBlock(dim_latent, 512, 512),  # input is 8x8, 512 channels, output is 8x8, 512
                GeneratorBlock(dim_latent, 512, 512),  # input is 16x16, 512 channels, output is 16x16, 512
                GeneratorBlock(dim_latent, 512, 512),  # input is 32x32, 512 channels, output is 32x32, 512
                GeneratorBlock(dim_latent, 512, 256),  # input is 64x64, 512 channels, output is 64x64, 256
            ]
        )

        # Upsampling operation is applied after each generator block
        self.upsample = Upsample()

    def forward(self, w, noise):
        """
        Parameters
        ----------
        `w` : intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]
        `noise` : a list of tuples, one tuple for each generator block, with each tuple
            containing the random noise input for the given generator block. The first
            generator block receives only one noise tensor, while the rest receive two.
            Each random noise tensor is of shape [batch_size, 1, block_resolution, block_resolution],
            where block_resolution is the resolution (image size) at a particular block.
            For example, in the first block, the resolution is 4x4. The second block's
            resolution is 8x8, third block's resolution is 16x16 and so on.

        Returns
        -------
        `rgb_out` : batch_size RGB images i.e. tensor of shape [batch_size, 3, image_height, image_width].
            For the time being, image_height = image_width = 64
        """
        # TODO for now I'm not going to do style mixing regularization (multiple ws)

        batch_size, _ = w.shape

        # Make batch_size copies of the constant input
        inp = self.constant.expand(batch_size, -1, -1, -1)

        # First generator block
        # Output of conv_block1 is shape [batch_size, out_channels, height, width] (out_channels = 512)
        # rgb_out is shape [batch_size, 3, height, width]
        x = self.conv_block1(inp, w, noise[0][0])  # first block receives only one noise tensor
        rgb_out = self.to_rgb1(x, w)

        for i, block in enumerate(self.blocks, start=1):
            # Upsample the output from previous generator block first by 2x
            x = self.upsample(x)

            # Obtain the output of current generator block - feature map and RGB image at given scale
            # x is shape [batch_size, out_channels, height, width] (out_channels, height, width refer to the current block)
            # rgb_out is shape [batch_size, 3, height, width]
            x, rgb_new = block(x, w, noise[i])

            # Upsample the previous RGB and add it to the current
            rgb_out = self.upsample(rgb_out) + rgb_new

        # Generator output
        return rgb_out

    __call__ = proxy(forward)

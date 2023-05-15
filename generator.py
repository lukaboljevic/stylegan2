import torch
import torch.nn as nn

from mapping_network import MappingNetwork
from general_utils.upsample import Upsample
from generator_utils.to_rgb import ToRGB
from generator_utils.generator_block import GeneratorBlock, GeneratorConvBlock


class Generator(nn.Module):
    def __init__(self, dim_latent=512):
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
        self.blocks = nn.ModuleList([
            GeneratorBlock(dim_latent, 512, 512), # input is 8x8, 512 channels, output is 8x8, 512
            GeneratorBlock(dim_latent, 512, 512), # input is 16x16, 512 channels, output is 16x16, 512
            GeneratorBlock(dim_latent, 512, 512), # input is 32x32, 512 channels, output is 32x32, 512
            GeneratorBlock(dim_latent, 512, 256), # input is 64x64, 512 channels, output is 64x64, 256
        ])

        # Upsampling operation is applied after each generator block
        self.upsample = Upsample()

        # TODO add noise


    def forward(self, w):
        """
        Parameters
        ----------
        w : intermediate latent variable coming from the mapping network of shape
            [batch_size, dim_latent]

        Returns
        -------
        rgb_out : "RGB image" i.e. tensor of shape [batch_size, 3, image_height, image_width].
            For the time being, image_height = image_width = 64
        """
        # TODO for now I'm not going to do style mixing regularization (multiple ws)
        # TODO add noise

        batch_size, _ = w.shape

        # Make batch_size copies of the constant input
        inp = self.constant.expand(batch_size, -1, -1, -1)

        # First generator block
        # Output of conv_block1 is shape [batch_size, out_channels, height, width] (out_channels = 512)
        # rgb_out is shape [batch_size, 3, height, width]
        x = self.conv_block1(inp, w)
        rgb_out = self.to_rgb1(x, w)

        for block in self.blocks:
            # Upsample the output from previous generator block first by 2x
            x = self.upsample(x)

            # Obtain the output of current generator block - feature map and RGB image at given scale
            # x is shape [batch_size, out_channels, height, width] (out_channels, height, width refer to the current block)
            # rgb_out is shape [batch_size, 3, height, width]
            x, rgb_new = block(x, w)

            # Upsample the previous RGB and add it to the current
            rgb_out = self.upsample(rgb_out) + rgb_new

        # Generator outputs
        return rgb_out


# Test current implementation of generator

dim_latent = 512
image_size = 64
batch_size = 7  # intentionally a 'weird' number so it can be easily distinguished

device = "cuda"
mapping = MappingNetwork(dim_latent).to(device)

z = torch.randn(batch_size, dim_latent).to(device)
w = mapping(z)

generator = Generator(dim_latent).to(device)
rgb = generator(w)
print("-" * 80)
print("Yay!")
print(rgb.shape)
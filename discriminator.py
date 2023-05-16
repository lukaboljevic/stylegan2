import torch.nn as nn

from general_utils.proxy import proxy
from general_utils.equalized import EqualizedConv2d, EqualizedLinear
from discriminator_utils.from_rgb import FromRGB
from discriminator_utils.discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    """
    Discriminator structure has not changed much since Progressive GAN, where the structure
    is described in Table 2. In StyleGAN2, they opt for a "residual" discriminator (Figure 7)
    """
    def __init__(self):
        super().__init__()

        # Since we're for now limited to 64x64 image size, converts [batch_size, 3, 64, 64] into [batch_size, 256, 64, 64] 
        self.from_rgb = FromRGB(256)

        self.blocks = nn.Sequential(*[
            DiscriminatorBlock(256, 512),  # from [batch_size, 256, 64, 64] to [batch_size, 512, 32, 32]
            DiscriminatorBlock(512, 512),  # from [batch_size, 512, 32, 32] to [batch_size, 512, 16, 16]
            DiscriminatorBlock(512, 512),  # from [batch_size, 512, 16, 16] to [batch_size, 512, 8, 8]
            DiscriminatorBlock(512, 512),  # from [batch_size, 512, 8, 8] to [batch_size, 512, 4, 4]
        ])

        # TODO MiniBatchStdDev not implemented for now

        # Final 3x3 convolution, will be [batch_size, 512, 4, 4]
        self.conv = EqualizedConv2d(512, 512, kernel_size=3)
        self.conv_activation = nn.LeakyReLU(0.2, inplace=True)

        # Last two FC layers: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L639-L640
        # `linear` input shape is [batch_size, 512*4*4], output is [batch_size, 512]
        self.linear = EqualizedLinear(512 * 4 * 4, 512)
        self.linear_activation = nn.LeakyReLU(0.2, inplace=True)

        # The final linear layer classifies the batch_size images as real or fake
        # Output is of shape [batch_size, 1]
        self.out_linear = EqualizedLinear(512, 1)
        # activation for out_linear is linear i.e. passthrough


    def forward(self, x):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, 3, 64, 64] i.e. batch_size RGB images (real or fake)

        Returns
        -------
        cls : output of shape [batch_size, 1], indicating the class (real/fake) of all batch_size images
        """
        # Convert from RGB to feature vectors
        x = self.from_rgb(x)

        # Pass through discriminator blocks
        x = self.blocks(x)

        # Perform the 3x3 convolution
        x = self.conv_activation(self.conv(x))

        # Reshape before applying linear layers
        batch_size, *_ = x.shape
        x = x.view(batch_size, -1)

        # Apply final two (linear) layers
        x = self.linear_activation(self.linear(x))
        return self.out_linear(x)
    
    __call__ = proxy(forward)

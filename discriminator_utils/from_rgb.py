import torch.nn as nn

from general_utils.equalized import EqualizedConv2d
from general_utils.proxy import proxy


class FromRGB(nn.Module):
    """
    The FromRGB layer in the discriminator is much simpler than the ToRGB layer in the
    generator - it is just a Conv2d layer using scaled weights (equalized learning rate).
    """

    def __init__(self, out_channels, in_channels=3, kernel_size=1):
        super().__init__()

        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size)
        self.activation = nn.LeakyReLU(0.2, inplace=True)  # negative_slope = 0.2 is used ever since Progressive GAN

    def forward(self, x):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, 3, height, width] (i.e. batch_size RGB images). At the
            moment, height = width = 64.

        Returns
        -------
        out : feature vectors of shape [batch_size, out_channels, height, width]. At the moment,
            out_channels = 256 since height = width = 64.
        """
        x = self.conv(x)
        return self.activation(x)

    __call__ = proxy(forward)

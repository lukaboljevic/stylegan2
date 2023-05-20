import torch
import torch.nn as nn

from general_utils.equalized import EqualizedLinear
from general_utils.proxy import proxy


class MappingNetwork(nn.Module):
    def __init__(self, dim_latent=512, w_avg_beta=0.995):
        """
        Implements the mapping network first introduced in StyleGAN.

        Parameters
        ----------
        `dim_latent` : Dimension of latent variables `z` and `w`, i.e. the input and output
            of the mapping network respectively
        `w_avg_beta` : Related to the truncation trick StyleGAN(2) uses:
            - https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L185
            - https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L212
        """
        super().__init__()

        layers = []
        for _ in range(8):
            layers.append(EqualizedLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # negative_slope = 0.2 is used ever since Progressive GAN

        self.net = nn.Sequential(*layers)

        # Truncation trick (from StyleGAN onwards)
        self.w_avg_beta = w_avg_beta
        self.w_avg = nn.Parameter(torch.zeros([dim_latent]), requires_grad=False)
        # self.register_buffer("w_avg", torch.zeros([dim_latent]))

    def forward(self, z, truncation_psi=1):
        """
        Parameters
        ----------
        `z` : Random vector of shape [batch_size, dim_latent]

        Returns
        -------
        `w` : Intermediate latent variable of shape [batch_size, dim_latent]
        """
        z = nn.functional.normalize(z)  # dim=1 is default
        out = self.net(z)

        # Update moving average of w
        self.w_avg.copy_(out.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Truncation trick
        # TODO? (right?) During training, truncation_psi = 1, but during GENERATION, we might
        # want to set it to 0.7 or 0.5 or something like that.
        if truncation_psi != 1:
            out = self.w_avg.lerp(out, truncation_psi)

        return out

    __call__ = proxy(forward)

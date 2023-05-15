import torch.nn as nn

from general_utils.equalized import EqualizedLinear


class MappingNetwork(nn.Module):
    def __init__(self, dim_latent=512):  # dimension of z/w from latent space Z/W
        super().__init__()

        layers = []
        for _ in range(8):
            layers.append(EqualizedLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # negative_slope = 0.2 is used ever since Progressive GAN

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Parameters
        ----------
        z : random vector of shape [batch_size, dim_latent]

        Returns
        -------
        w : intermediate latent variable of shape [batch_size, dim_latent]
        """
        z = nn.functional.normalize(z)  # dim=1 is default
        return self.net(z)

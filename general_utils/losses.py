import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proxy import proxy


"""
DiscriminatorLoss and GeneratorLoss copied from:
    https://github.com/ppeetteerrs/stylegan2-torch/blob/main/stylegan2_torch/loss.py
as the original StyleGAN2-ADA implementation also used losses from original GAN:
    https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py

GradientPenalty and PathLengthRegularization (originally named PathLengthPenalty) copied from
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/gan/stylegan/__init__.py

This is the reason I didn't comment on those functions much
"""


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss from the original GAN paper (utilized from StyleGAN onwards)
    """

    def forward(self, real, fake):
        real_loss = F.softplus(-real)  # log(D(x)) from original GAN
        fake_loss = F.softplus(fake)  # log(1 - D(G(z)))
        return real_loss.mean() + fake_loss.mean()

    __call__ = proxy(forward)


class GeneratorLoss(nn.Module):
    """
    Non-saturating generator loss from the original GAN paper (stated just above Figure 1)
    """

    def forward(self, fake):
        fake_loss = F.softplus(-fake)
        return fake_loss.mean()

    __call__ = proxy(forward)


class GradientPenalty(nn.Module):
    """
    The R1 regularization used alongside discriminator loss (utilized from StyleGAN onwards)
    """

    def forward(self, x, d: torch.Tensor):
        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(
            outputs=d,
            inputs=x,
            grad_outputs=d.new_ones(d.shape),
            create_graph=True
        )

        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)

    __call__ = proxy(forward)


class PathLengthRegularization(nn.Module):
    """
    Path length regularization used alongside generator loss, introduced in StyleGAN2
    """

    def __init__(self, beta):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, w):
        """
        TODO input size
        """
        print(f"{__class__.__name__} x shape:", x.shape)
        print(f"{__class__.__name__} w shape:", w.shape)
        # x, w should be [batch_size, 3, height, width] and [batch_size, dim_latent]

        device = x.device

        image_size = x.shape[2] * x.shape[3]

        y = torch.randn(x.shape, device=device)
        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True,
        )

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.0)

        return loss

    __call__ = proxy(forward)

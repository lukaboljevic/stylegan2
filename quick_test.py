import torch
from torchinfo import summary

from general_utils.generator_noise import generate_noise
from model.mapping_network import MappingNetwork
from model.generator import Generator
from model.discriminator import Discriminator


"""
Sanity check to make sure all the shapes match up and the batch goes through the entire GAN
"""

dim_latent = 512
image_size = 64
batch_size = 7  # intentionally a 'weird' number so it can be easily distinguished
device = "cuda"

# StyleGAN2
mapping = MappingNetwork(dim_latent).to(device)
generator = Generator(dim_latent).to(device)
discriminator = Discriminator().to(device)

# Intermediate latent variable w
z = torch.randn(batch_size, dim_latent).to(device)
w = mapping(z)

# Generator input noise
generator_noise = generate_noise(batch_size, device)

# Generator output
rgb = generator(w, generator_noise)

# Put it through the discriminator
cls = discriminator(rgb)

print("-" * 80)
print("All good!")
print(cls.shape)

# print(summary(generator))
# print(summary(discriminator))

import torch


def generate_noise(batch_size, device):
    """
    Generate random noise input for the generator. The size of the generated
    noise depends on the resolution (image size) used at a particular block.
    For example, the resolution of the first block is 4x4, of the second is
    8x8, and so on. The noise is standard normal, as per StyleGAN paper
    (Figure 1).

    For each generator block except the first, we should generate two noise
    tensors of the same shape, [batch_size, 1, resolution, resolution]. Since
    the first generator block performs only one 3x3 convolution, we need to
    supply it with only one noise tensor.

    At the time being, we go only up to 64x64 size.
    """
    i = 0
    resolution = 4
    noise_inputs = []

    while True:
        # Generate two random noise tensors, or only one for the first generator block
        noise_1 = torch.randn(batch_size, 1, resolution, resolution).to(device)
        noise_2 = None
        if i > 0:
            noise_2 = torch.randn(batch_size, 1, resolution, resolution).to(device)

        noise_inputs.append((noise_1, noise_2))
        i += 1

        # At the moment, we only go up to 64x64 size
        if resolution == 64:
            break

        # Resolution of next generator block is doubled
        resolution *= 2

    return noise_inputs

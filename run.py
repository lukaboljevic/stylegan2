import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import Image

from model.model import StyleGan2


"""
Test if training would work
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 3
batch_size = 4
num_training_images = 12
save_every_num_epoch = -1
use_loss_regularization = False

model = StyleGan2(
    root="./",
    num_epochs=num_epochs,
    batch_size=batch_size,
    num_training_images=num_training_images,
    save_every_num_epoch=save_every_num_epoch,
    use_loss_regularization=use_loss_regularization
)

model.train()
# model.save()

# TODO there's a bug with generation atm, it's related to the generated `z` and `noise` tensors
with torch.no_grad():
    # Generate as many output images as you want
    num_images = 4

    images, _ = model.generate_images(num_images)
    image_grid = make_grid(images, nrow=2, padding=0).permute(1, 2, 0).cpu().numpy()

image_grid = Image.fromarray(np.uint8(image_grid*255)).convert("RGB")
image_grid.show()

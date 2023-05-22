import os
from tqdm import tqdm

from model.model import StyleGan2


"""
Generate images using a pretrained model
"""

root = "./"  # Directory where `celeba` is located
model = StyleGan2(root=root)

# Pretrained model
model.load_model(path_to_model="./stylegan2-3idx-20000steps.pth")

# Destination
dest_dir = "./generated_3idx_20000steps"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

# Generate 19962 images. On an RTX 2070 Mobile, this took right around 3 minutes.
num_images = len(os.listdir("./celeba_test_64x64"))
for idx in tqdm(range(num_images)):
    model.generate_output(1, 1, base_path=dest_dir, truncation_psi=1, output_name=str(idx), log=False)


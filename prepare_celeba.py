import os
import torchvision as tv
from PIL import Image
from tqdm import tqdm


"""
Resize the test images from CelebA to 64x64 for calculating FID
"""

dest_dir = "celeba_test_64x64"
data = tv.datasets.CelebA(root="./", split="test")
images = data.filename
# print(len(images))  # 19962 for test split

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for num, imagename in tqdm(enumerate(images)):
    if num > 20000:
        break

    imagenum = int(imagename[:-4])
    img = Image.open(f"./celeba/img_align_celeba/{imagename}")
    img = img.resize((64, 64))
    img.save(f"./{dest_dir}/{imagenum}.jpg")

# Deep learning project - StyleGAN2

This is the source code for a project at course Deep Learning, regarding the re-implementation and testing of [StyleGAN2 paper](https://arxiv.org/abs/1912.04958), "Analyzing and Improving the Image Quality of StyleGAN". The implementation is done using PyTorch.

The goal is to first implement, train and evaluate StyleGAN2, using the popular [FID metric](https://github.com/mseitzer/pytorch-fid). Then, the idea (hope) is to implement the improvement of StyleGAN2 - [StyleGAN2-ADA](https://arxiv.org/pdf/2006.06676.pdf).

## Note

If you intend to run anything from this repository, make sure the path to the directory where you cloned it is added to `PYTHONPATH`.



# Repository structure

A brief overview of the structure of the repository:

```
./
├── celeba/                         # Dataset. Download instructions below
│   └── img_align_celeba
│       ├── 000001.jpg
│       ├── 000002.jpg
│       ├── 000003.jpg
│       └── ...
│
├── discriminator_utils/            # Discriminator layers
│   ├── discriminator_block.py      # Residual discriminator block (2 3x3 conv layers, residual 1x1 conv)
│   └── from_rgb.py                 # FromRGB layer
│
├── general_utils/                  # General utility functions
│   ├── equalized.py                # Linear and convolution layers with equalized learning rate
│   ├── generator_noise.py          # Function that generates the noise input for each generator block
│   ├── logger.py                   # Function for setting up a logger
│   ├── losses.py                   # Non-saturating logistic loss (original GAN), R1 and path length regularizations
│   ├── proxy.py                    # Proxy function that shows hints when invoking the forward function of a nn.Module
│   └── upsample.py                 # Upsampling and downsampling operations using FIR filter smoothing
│
├── generator_utils/                # Generator layers
│   ├── generator_block.py          # Skip generator block (2 3x3 convs with weight demodulation, ToRGB)
│   └── to_rgb.py                   # ToRGB layer
│
├── model/
│   ├── dataset.py                  # Function to return train (and optional test) DataLoader
│   ├── discriminator.py            # Discriminator put together
│   ├── generator.py                # Generator put together
│   ├── mapping_network.py          # Mapping network + truncation trick
│   └── model.py                    # Put entire model together, hyperparameters, train loop, checkpoints, ...
│
└── run_training.py                 # Instantiate model and train
```



# CelebA dataset

The dataset used is CelebA, which is [available in PyTorch](https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html). However, if you try to download it by setting `download=True`, you may (and most probably will) get a RuntimeError `The daily quota of the file img_align_celeba.zip is exceeded and it can't be downloaded. This is a limitation of Google Drive and can only be overcome by trying again later.`

To avoid having to wait, I suggest going to the [official Google Drive for CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), and downloading the following 6 files:
- From **Img**
  - img_align_celeba.zip
- From **Eval**
  - list_eval_partition.txt
- From **Anno**
  - list_landmarks_align_celeba.txt
  - list_bbox_celeba.txt
  - list_attr_celeba.txt
  - identity_CelebA.txt

Place all of them inside the folder `celeba`, following the repository structure. Then, simply let PyTorch and `model/dataset.py` do the rest.

import os
import torchvision as tv
from torch.utils.data import DataLoader, Subset


def setup_data_loaders(root, batch_size, image_size=64, train_subset_size=-1, return_test_loader=False):
    """
    Setup the train and optionally test DataLoader for CelebA dataset.

    Perform the following torchvision.transforms:
    - Resize (to image_size)
    - CenterCrop
    - ToTensor
    - Normalize with mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5)

    Parameters
    ----------
    `root` : Path to the directory containing the `celeba` folder, either absolute or relative to the 
        script this function is being called in.
    `batch_size` : Batch size to be forwarded to DataLoader
    `image_size` : What size to resize images to (this will be the size StyleGAN2 will generate images at)    
    `train_subset_size` : Whether to use the entire training set (train_subset_size=-1), or a subset of
        it of given size.
    `return_test_loader` : Whether to return the DataLoader for test dataset as well

    Returns
    -------
    `train_loader` : DataLoader for training dataset
    `test_loader` : None if return_test_loader=False, else DataLoader for test dataset
    """

    # Understanding transforms.Normalize():
    # https://discuss.pytorch.org/t/understanding-transform-normalize/21730/27

    # I used the below transforms based on the following link:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#data
    transform = tv.transforms.Compose([
        tv.transforms.Resize(image_size),
        tv.transforms.CenterCrop(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Extract the ZIP with CelebA images if not already
    if not os.path.exists(os.path.join(root, "celeba", "img_align_celeba")):
        tv.datasets.CelebA(root=root, split="all", download=True)

    # PyTorch automatically looks for a "celeba" folder in the provided `root` parameter
    train_data = tv.datasets.CelebA(root=root, split="train", transform=transform)
    if train_subset_size != -1:
        # Select first train_subset_size images (TODO add parameter to choose first or random?)
        train_data = Subset(train_data, range(train_subset_size))

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    test_loader = None
    if return_test_loader:
        test_data = tv.datasets.CelebA(root=root, split="test", transform=transform)
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

    return train_loader, test_loader


def cycle_data_loader(data_loader):
    """
    Makes the DataLoader cyclic - if its end is reached, it will return batches from the start again.
    """
    while True:
        for idx, batch in enumerate(data_loader):
            yield idx, batch

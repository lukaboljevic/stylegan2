import torchvision as tv
from torch.utils.data import DataLoader
import PIL

transform = tv.transforms.Compose([
    # more ... ?
    tv.transforms.ToTensor()
])

train_data = tv.datasets.CelebA(root="./", split="train", transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=0)

valid_data = tv.datasets.CelebA(root="./", split="valid", transform=transform)
valid_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False, num_workers=0)

test_data = tv.datasets.CelebA(root="./", split="test", transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=0)

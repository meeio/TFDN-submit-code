import torchvision.transforms as transforms
import torch
import numpy as np

ROOT = "./DATASET/"
ROOT_MNIST = "./DATASET/MNIST"


def for_dataset(name, split='train', transfrom=None, with_targets=False):
    data_set = None
    data_info = dict()
    if name.upper() == 'MNIST':
        from torchvision.datasets import MNIST
        dataset = MNIST(
            root=ROOT + name.upper(),
            train=(split == 'train'),
            transform=transfrom
        )
        data_info['labels'] = dataset.targets
    elif name.upper() == "USPS":
        from mdata.usps import USPS
        dataset = USPS(
            root=ROOT + name.upper(),
            train=(split == 'train'),
            transform=transfrom,
            download= True,
        )
        data_info['labels'] = torch.tensor(dataset.targets)
        # print(type(dataset.targets))
        # assert False
    elif name.upper() == "SVHN":
        assert False
        from torchvision.datasets import SVHN
        dataset = SVHN(
            root=ROOT + name.upper(),
            split = split,
            transform=transfrom,
            download=True,
        )
        data_info['labels'] = torch.from_numpy(dataset.labels) 
    return dataset, data_info

def for_digital_transforms(is_rgb=True):
    channel = 3 if is_rgb else 1
    trans = [
        transforms.Resize(28),
        transforms.Grayscale(channel),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,) * channel,
            std=(0.5,) * channel
        )
    ]
    return transforms.Compose(trans)



import torch
from torchvision import datasets, transforms
from typing import Tuple

def load_cifar10_data() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the CIFAR-10 dataset and returns the training and test sets.
    @author: Stephen Krol

    :return: the training and test sets
    :rtype: tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    return train_set, test_set
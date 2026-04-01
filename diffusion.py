
from typing import Tuple

import yaml
import random
import torch
from torchvision import datasets, transforms

from utils import plot_images, LABEL_TO_CLASS

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

def plot_random_images(dataset: torch.utils.data.Dataset, num_images: int = 10) -> None:
    """
    Plots a random selection of images from the given dataset.
    @author: Stephen Krol

    :param dataset: the dataset to plot images from
    :type dataset: torch.utils.data.Dataset
    :param num_images: the number of images to plot
    :type num_images: int

    :return: None
    :rtype: None
    """

    random_indices = random.sample(range(len(dataset)), num_images)
    images = [dataset[i][0].permute(1, 2, 0) for i in random_indices]
    images = [(img + 1) / 2 for img in images] # convert images from [-1, 1] to [0, 1]

    titles = [LABEL_TO_CLASS[dataset[i][1]] for i in random_indices]
    plot_images(images, titles)

def plot_batch_images(batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """
    Plots a batch of images with their corresponding labels.
    """

    images = [batch[0][i].permute(1, 2, 0) for i in range(len(batch[0]))]
    images = [(img + 1) / 2 for img in images] # convert images from [-1, 1] to [0, 1]

    titles = [LABEL_TO_CLASS[batch[1][i].item()] for i in range(len(batch[1]))]
    plot_images(images, titles)

def plot_image_noisy_pairs(clean_images: torch.Tensor, noisy_images: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Plots pairs of clean and noisy images with their corresponding labels.
    """

    images = []
    titles = []
    for i in range(len(clean_images)):
        clean_img = clean_images[i].permute(1, 2, 0)
        noisy_img = noisy_images[i].permute(1, 2, 0)

        clean_img = (clean_img + 1) / 2 # convert images from [-1, 1] to [0, 1]
        noisy_img = (noisy_img + 1) / 2

        images.extend([clean_img, noisy_img])
        titles.extend([f"Clean - {LABEL_TO_CLASS[labels[i].item()]}", f"Noisy - {LABEL_TO_CLASS[labels[i].item()]}"])

    plot_images(images, titles, cols=2)


def calculate_noise_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """
    Returns the noise schedule for the diffusion process.
    @author: Stephen Krol

    :param T: the number of timesteps
    :type T: int
    :param beta_start: the starting value of beta
    :type beta_start: float
    :param beta_end: the ending value of beta
    :type beta_end: float
    
    :return: the noise schedule
    :rtype: torch.Tensor
    """
    return torch.linspace(beta_start, beta_end, T)

def calculate_alpha_bar(noise_schedule: torch.Tensor) -> torch.Tensor:
    """
    Returns the alpha_bar value for a given timestep.
    @author: Stephen Krol

    :param noise_schedule: the noise schedule
    :type noise_schedule: torch.Tensor

    :type T: int
    :param noise_schedule: the noise schedule
    :type noise_schedule: torch.Tensor

    :return: the alpha_bar value
    :rtype: torch.Tensor
    """
    return torch.cumprod(1 - noise_schedule, dim=0)


def forward_diffusion_sample(x_0: torch.tensor,
                             t: int, 
                             noise_schedule: torch.Tensor,
                             device="cpu") -> torch.Tensor:
    """
    Takes an image and a timestep as input and returns the noisy version of it
    at the given timestep.
    @author: Stephen Krol

    :param x_0: the original image
    :type x_0: torch.Tensor
    :param t: the timestep
    :type t: int

    :param device: the device to run the computation on
    :type device: str

    :return: the noisy version of the image at the given timestep
    :rtype: torch.Tensor
    """

    alpha_bar = calculate_alpha_bar(noise_schedule)
    noise = torch.randn_like(x_0).to(device)

    return torch.sqrt(alpha_bar[t])*x_0 + torch.sqrt(1 - alpha_bar[t])*noise
    
    


if __name__ == "__main__":

    with open("CIFAR10.yaml", "r") as f:
        config = yaml.safe_load(f)

    noise_schedule = calculate_noise_schedule(config["params"]["T"], config["params"]["beta_start"], config["params"]["beta_end"])  


    train_set, test_set = load_cifar10_data()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=False)
    batch = next(iter(train_dataloader))
    
    x_t = forward_diffusion_sample(batch[0], t=500, noise_schedule=noise_schedule)

    plot_image_noisy_pairs(batch[0], x_t, batch[1])


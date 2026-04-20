
from typing import Tuple
import random
import torch
import matplotlib.pyplot as plt


LABEL_TO_CLASS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def plot_images(images: torch.Tensor, titles=None, cols:int =5, figsize=(15, 10)) -> None:
    """
    Plot a list of images with optional titles.
    @author: Stephen Krol

    :param images: a list of images to plot
    :type images: torch.Tensor
    :param titles: a list of titles for the images
    :type titles: list[str], optional
    :param cols: the number of columns in the plot
    :type cols: int, optional
    :param figsize: the size of the figure
    :type figsize: tuple, optional
    
    :return: None
    :rtype: None
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().clamp(0, 1)
        plt.imshow(img)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()


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

    try:
        titles = [LABEL_TO_CLASS[dataset[i][1]] for i in random_indices]
    except KeyError:
        titles = [LABEL_TO_CLASS[dataset[i][2]] for i in random_indices]
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

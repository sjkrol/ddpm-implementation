
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
        plt.imshow(img)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

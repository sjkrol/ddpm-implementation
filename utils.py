
import os
import torch
import random
import datetime

import matplotlib.pyplot as plt

from typing import Tuple, Optional, Dict, Any
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


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


# DDPM HELPER FUNCTIONS

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


# HELPER FUNCTIONS FOR DISTRIBUTED TRAINING

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Returns the underlying module when wrapped by DistributedDataParallel.
    @author: Stephen Krol

    :param model: the model to unwrap
    :type model: torch.nn.Module
    
    :return: the unwrapped model
    :rtype: torch.nn.Module
    """

    return model.module if isinstance(model, DistributedDataParallel) else model


def setup_distributed_training(distributed_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Initializes distributed state from config and torchrun environment variables.
    @author: Stephen Krol

    :param distributed_config: the distributed training configuration
    :type distributed_config: dict

    :return: the runtime context for distributed training
    :rtype: dict
    """

    distributed_config = distributed_config or {}
    enabled = bool(distributed_config.get("enabled", False))

    if not enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main_process": True,
            "device": device,
        }

    backend = distributed_config.get("backend") or ("nccl" if torch.cuda.is_available() else "gloo")
    init_method = distributed_config.get("init_method", "env://")

    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as exc:
        raise RuntimeError(
            "Distributed training requires torchrun-style launch environment variables: "
            "RANK, LOCAL_RANK, and WORLD_SIZE."
        ) from exc

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank) if torch.cuda.is_available() else None,
    )

    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "device": device,
    }

def cleanup_distributed_training(runtime_context: Dict[str, Any]) -> None:
    """
    Cleans up the active process group when distributed training is enabled.
    @author: Stephen Krol

    :param runtime_context: the runtime context for distributed training
    :type runtime_context: dict
    """

    if runtime_context.get("enabled") and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_checkpoint_dir(
    project_name: str,
    runtime_context: Dict[str, Any],
    base_dir: str = "checkpoints") -> str:
    """
    Creates a shared checkpoint directory name, only materialized by rank 0.
    @author: Stephen Krol

    :param project_name: the name of the project
    :type project_name: str
    :param runtime_context: the runtime context for distributed training
    :type runtime_context: dict
    :param base_dir: the base directory for checkpoints
    :type base_dir: str

    :return: the path to the checkpoint directory
    :rtype: str
    """

    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if runtime_context["is_main_process"] else None

    if runtime_context["enabled"]:
        run_name_payload = [run_name]
        dist.broadcast_object_list(run_name_payload, src=0)
        run_name = run_name_payload[0]

    save_dir = os.path.join(base_dir, project_name, run_name)

    if runtime_context["is_main_process"]:
        os.makedirs(save_dir, exist_ok=True)
    if runtime_context["enabled"]:
        dist.barrier()

    return save_dir


# TENSOR CORE HELPER FUNCTIONS

def configure_tensor_core_backend(enabled: bool = True) -> None:
    """
    Configures CUDA backend flags for Tensor Core acceleration.

    :param enabled: whether to enable Tensor Core-friendly backend settings
    :type enabled: bool
    """

    if not torch.cuda.is_available():
        return

    if enabled:
        # Allow TF32 kernels on Ampere+ for faster matmul/conv while keeping fp32 interfaces.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        # Force full-fp32 path for users who want maximum numeric fidelity.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
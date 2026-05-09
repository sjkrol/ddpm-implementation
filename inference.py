"""
Inference script for generating samples and calculating evaluation metrics from a trained diffusion model.
"""


import os
import PIL
import yaml
import torch
import argparse

from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader

from Unet import UNet
from diffusion import calculate_noise_schedule, calculate_alpha_bar, load_cifar10_data

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

try:
    from torchmetrics.image.inception import InceptionScore
except ImportError:
    InceptionScore = None

def ddpm_sample(model: torch.nn.Module,
           noise_schedule: torch.Tensor,
           num_samples: int, 
           device: torch.device,
           resolution: Tuple[int, int] = (32, 32)) -> torch.Tensor:
    """
    Function to generate samples from the trained model using DDPM sampling.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :param device: The device to run the sampling on (e.g., 'cuda' or 'cpu').
    :type device: torch.device

    :return: A tensor containing the generated samples.
    :rtype: torch.Tensor
    """

    noise_schedule = noise_schedule.to(device)
    alpha_bar = calculate_alpha_bar(noise_schedule).to(device)
    x = torch.randn(num_samples, 3, resolution[0], resolution[1], device=device)  # Start with random noise
    for t in tqdm(reversed(range(noise_schedule.shape[0]))):
        with torch.no_grad():
            
            x = ddpm_update_helper(x, model, noise_schedule, alpha_bar, t)

    return x

def ddpm_update_helper(x: torch.Tensor,
                       model: torch.nn.Module,
                       noise_schedule: torch.Tensor,
                       alpha_bar: torch.Tensor,
                       t: int) -> torch.Tensor:
    """
    Helper function to update the sample during the denoising process.
    @author: Stephen Krol

    :param x: The current sample tensor.
    :type x: torch.Tensor
    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param alpha_bar: The cumulative product of (1 - beta) values from the noise schedule.
    :type alpha_bar: torch.Tensor
    :param t: The current timestep.
    :type t: int

    :return: The updated sample tensor after one denoising step.
    :rtype: torch.Tensor
    """

    T = torch.full((x.size(0),), t, device=device, dtype=torch.long) # Create a tensor for the current timestep
    noise_pred =  model(x, T)  # Predict the noise at the current timestep

    a_t = 1 - noise_schedule[t]

    if t > 0:
        z = torch.randn_like(x)
    else:
        z = torch.zeros_like(x)


    return 1 / a_t.sqrt() * (x - (1 - a_t) / (1 - alpha_bar[t]).sqrt() * noise_pred) + noise_schedule[t].sqrt() * z    

def ddim_sample(model: torch.nn.Module,
                noise_schedule: torch.Tensor,
                num_samples: int,
                device: torch.device,
                resolution: Tuple[int, int] = (32, 32),
                eta: float = 0.0,
                sampling_steps: int = 200) -> torch.Tensor:
    """
    Function to generate samples from the trained model using DDIM sampling.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :param device: The device to run the sampling on (e.g., 'cuda', 'cpu').
    :type device: torch.device
    :param eta: The noise scale for DDIM sampling (default 0.0 for deterministic sampling).
    :type eta: float
    :param sampling_steps: The number of sampling steps to use (default 200).
    :type sampling_steps: int

    :return: A tensor containing the generated samples.
    :rtype: torch.Tensor
    """

    noise_schedule = noise_schedule.to(device)
    alpha_bar = calculate_alpha_bar(noise_schedule).to(device)
    x = torch.randn(num_samples, 3, resolution[0], resolution[1], device=device)  # Start with random noise

    # Create a list of timesteps to sample from, spaced evenly across the noise schedule
    total_steps = noise_schedule.shape[0]
    step_size = max(1, total_steps // sampling_steps)
    t_s = list(reversed(range(0, total_steps, step_size))) # sampling timesteps in reverse order (from T-1 down to 0)

    for i, t in tqdm(enumerate(t_s), desc="DDIM Sampling"):
        with torch.no_grad():

            t_prev = t_s[i + 1] if i < len(t_s) - 1 else 0 # because t_s is reversed, the "previous" timestep is actually the next one in the list
            x = ddim_update_helper(x, model, noise_schedule, alpha_bar, t, t_prev, eta)

    return x

def ddim_update_helper(x: torch.Tensor,
                       model: torch.nn.Module,
                       noise_schedule: torch.Tensor,
                       alpha_bar: torch.Tensor,
                       t: int,
                       t_prev: int,
                       eta: float) -> torch.Tensor:
    """
    Helper function to update the sample during the DDIM denoising process.
    @author: Stephen Krol

    :param x: The current sample tensor.
    :type x: torch.Tensor
    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param alpha_bar: The cumulative product of (1 - beta) values from the noise schedule.
    :type alpha_bar: torch.Tensor
    :param t: The current timestep.
    :type t: int
    :param t_prev: The previous timestep in the sampling process.
    :type t_prev: int
    :param eta: The noise scale for DDIM sampling.
    :type eta: float

    :return: The updated sample tensor after one DDIM denoising step.
    :rtype: torch.Tensor
    """

    noise_pred = model(x, torch.full((x.size(0),), t, device=device, dtype=torch.long))
    sigma_t = eta * ((1 - alpha_bar[t_prev]) / (1 - alpha_bar[t])).sqrt() * (1 - alpha_bar[t] / alpha_bar[t_prev]).sqrt() if t > 0 else 0.0

    if t > 0:
        z = torch.randn_like(x)
    else:
        z = torch.zeros_like(x)
    
    x = alpha_bar[t_prev].sqrt() * ((x - (1 - alpha_bar[t]).sqrt() * noise_pred) / alpha_bar[t].sqrt()) + (1 - alpha_bar[t_prev] - sigma_t**2).sqrt() * noise_pred + sigma_t * z

    return x

def load_model(checkpoint_path: str, 
               config: dict,
               device: torch.device) -> torch.nn.Module:
    """
    Function to load a trained model from a checkpoint.
    :param checkpoint_path: The path to the model checkpoint.
    :type checkpoint_path: str
    :param device: The device to load the model on (e.g., 'cuda' or 'cpu').
    :type device: torch.device

    :return: The loaded model.
    :rtype: torch.nn.Module
    """

    model = UNet(original_channels=3, 
                 base_channels=config["model"]["base_channels"], 
                 channel_multipliers=config["model"]["channel_multipliers"],
                 num_res_blocks=config["model"]["num_res_blocks"],
                 in_resolution=config["model"]["in_resolution"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()


    return model


def _denormalize_from_model_space(images: torch.Tensor) -> torch.Tensor:
    """
    Converts images from training space [-1, 1] to [0, 1].
    @author: Stephen Krol

    :param images: A tensor of images in the range [-1, 1].
    :type images: torch.Tensor

    :return: A tensor of images in the range [0, 1].
    :rtype: torch.Tensor
    """
    return ((images.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def _to_fid_uint8(images: torch.Tensor) -> torch.Tensor:
    """
    Formats image tensors to uint8 [0, 255] for torchmetrics FID.
    @author: Stephen Krol

    :param images: A tensor of images in the range [-1, 1].
    :type images: torch.Tensor

    :return: A tensor of images in the range [0, 255] as uint8.
    :rtype: torch.Tensor
    """
    images = _denormalize_from_model_space(images)
    return (images * 255.0).round().to(torch.uint8)


def calculate_fid(model: torch.nn.Module,
                  noise_schedule: torch.Tensor,
                  num_real: int,
                  num_fake: int,
                  batch_size: int,
                  device: torch.device,
                  fake_batches: list = None) -> float:
    """
    Calculates FID against the CIFAR10 training split using generated samples.
    This matches the evaluation setup reported in the DDPM paper.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_real: The number of real samples to use for FID calculation.
    :type num_real: int
    :param num_fake: The number of fake samples to generate for FID calculation.
    :type num_fake: int
    :param batch_size: The batch size to use during FID calculation.
    :type batch_size: int
    :param device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :type device: torch.device
    :param fake_batches: Optional pre-generated list of uint8 fake image tensors. If
        provided, sample generation is skipped and these batches are used directly.
    :type fake_batches: list, optional

    :return: The calculated FID score.
    :rtype: float
    """

    if FrechetInceptionDistance is None:
        raise ImportError(
            "torchmetrics with image FID support is required. "
            "Install with: pip install torchmetrics torch-fidelity"
        )

    train_set, _ = load_cifar10_data()
    dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    seen_real = 0
    for real_batch, _ in tqdm(dataloader, desc="FID real", leave=False):
        if seen_real >= num_real:
            break

        remaining = num_real - seen_real
        real_batch = real_batch[:remaining].to(device)
        fid.update(_to_fid_uint8(real_batch), real=True)
        seen_real += real_batch.size(0)

    if fake_batches is not None:
        for batch in fake_batches:
            fid.update(batch.to(device), real=False)
    else:
        seen_fake = 0
        while seen_fake < num_fake:
            current_batch = min(batch_size, num_fake - seen_fake)
            fake_batch = sample(model, noise_schedule, current_batch, device)
            fid.update(_to_fid_uint8(fake_batch), real=False)
            seen_fake += current_batch

    return float(fid.compute().item())


def calculate_inception_score(
        model: torch.nn.Module,
        noise_schedule: torch.Tensor,
        num_fake: int,
        batch_size: int,
        device: torch.device,
        splits: int = 10,
        fake_batches: list = None) -> Tuple[float, float]:
    """
    Calculates the Inception Score (IS) for generated samples.

    IS = exp(E_x[ KL( p(y|x) || p(y) ) ])

    A higher IS indicates more diverse and visually meaningful samples.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_fake: The number of fake samples to generate.
    :type num_fake: int
    :param batch_size: The batch size to use when generating samples.
    :type batch_size: int
    :param device: The device to run the evaluation on.
    :type device: torch.device
    :param splits: Number of splits for IS estimation (default 10).
    :type splits: int
    :param fake_batches: Optional pre-generated list of uint8 fake image tensors. If
        provided, sample generation is skipped and these batches are used directly.
    :type fake_batches: list, optional

    :return: A tuple of (IS mean, IS std).
    :rtype: Tuple[float, float]
    """

    if InceptionScore is None:
        raise ImportError(
            "torchmetrics with image IS support is required. "
            "Install with: pip install torchmetrics[image]"
        )

    is_metric = InceptionScore(feature="logits_unbiased", splits=splits, normalize=False).to(device)

    if fake_batches is not None:
        for batch in fake_batches:
            is_metric.update(batch.to(device))
    else:
        seen_fake = 0
        while seen_fake < num_fake:
            current_batch = min(batch_size, num_fake - seen_fake)
            fake_batch = sample(model, noise_schedule, current_batch, device)
            is_metric.update(_to_fid_uint8(fake_batch))
            seen_fake += current_batch

    mean, std = is_metric.compute()
    return float(mean.item()), float(std.item())


def calculate_metrics(model: torch.nn.Module,
                      noise_schedule: torch.Tensor,
                      num_real: int,
                      num_fake: int,
                      batch_size: int,
                      device: torch.device) -> dict:
    """
    Function to calculate evaluation metrics for generated samples.
    Samples are generated once and shared between FID and IS to avoid redundant computation.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_real: The number of real samples to use for FID calculation.
    :type num_real: int
    :param num_fake: The number of fake samples to generate for FID and IS calculation.
    :type num_fake: int
    :param batch_size: The batch size to use during evaluation.
    :type batch_size: int
    :param device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :type device: torch.device

    
    :return: A dictionary containing the calculated metrics.
    :rtype: dict
    """

    # Generate fake samples once; store as uint8 on CPU to save GPU memory.
    fake_batches = []
    seen_fake = 0
    while seen_fake < num_fake:
        current_batch = min(batch_size, num_fake - seen_fake)
        fake_batch = ddpm_sample(model, noise_schedule, current_batch, device)
        fake_batches.append(_to_fid_uint8(fake_batch).cpu())
        seen_fake += current_batch

    fid_score = calculate_fid(
        model=model,
        noise_schedule=noise_schedule,
        num_real=num_real,
        num_fake=num_fake,
        batch_size=batch_size,
        device=device,
        fake_batches=fake_batches,
    )

    is_mean, is_std = calculate_inception_score(
        model=model,
        noise_schedule=noise_schedule,
        num_fake=num_fake,
        batch_size=batch_size,
        device=device,
        fake_batches=fake_batches,
    )

    metrics = {
        "FID": fid_score,
        "IS": {"mean": is_mean, "std": is_std},
    }
    
    return metrics

def save_samples(samples: torch.Tensor, save_path: str):
    """
    Function to save generated samples to disk.
    @author: Stephen Krol

    :param samples: The tensor containing the generated samples.
    :type samples: torch.Tensor
    :param save_path: The path to save the generated samples.
    :type save_path: str
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Convert samples to CPU and save as images
    samples = samples.cpu()
    for i in range(samples.shape[0]):
        sample = samples[i]
        sample = _to_fid_uint8(sample)  # Convert to uint8 format
        sample = sample.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)

        PIL.Image.fromarray(sample.numpy()).save(f"{save_path}/sample_{i}.png")

    
def plot_denoising_process(model: torch.nn.Module,
                           noise_schedule: torch.Tensor,
                           num_samples: int,
                           device: torch.device,
                           step_interval: int = 5,
                           spacing_strength: float = 1.5,
                           resolution: Tuple[int, int] = (32, 32),
                           save_path: str = "denoising_process.png") -> None:
    """
    Function to visualize the denoising process of the diffusion model.
    @author: Stephen Krol
                        step_size = max(1, noise_schedule.shape[0] // step_interval)
    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param num_samples: The number of samples to generate for visualization.
    :type num_samples: int
    :param device: The device to run the sampling on (e.g., 'cuda' or 'cpu').
    :type device: torch.device
    :param spacing_strength: Controls bias strength toward later denoising steps (default is 2.0). Higher values = stronger clustering at clean end.
    :type spacing_strength: float
    :param resolution: The resolution of the generated samples (default is (32, 32)).
    :type resolution: Tuple[int, int]
    :param save_path: The path to save the visualization image (default is "denoising_process.png").
    :type save_path: str

    :return: None
    :rtype: None
    """

    noise_schedule = noise_schedule.to(device)
    alpha_bar = calculate_alpha_bar(noise_schedule).to(device)

    # Generate unequal step indices that favor later (cleaner) denoising steps
    # Using power-law spacing: t_i = round(((i / (N-1)) ^ spacing_strength) * (T-1))
    import numpy as np
    total_steps = noise_schedule.shape[0]
    normalized = np.linspace(0, 1, step_interval) ** spacing_strength
    step_indices = np.round(normalized * (total_steps - 1)).astype(int)
    step_indices = np.unique(step_indices)
    step_indices_set = set(step_indices.tolist())

    x = torch.randn(num_samples, 3, resolution[0], resolution[1], device=device)  # Start with random noise

    images = []
    for t in tqdm(reversed(range(noise_schedule.shape[0])), desc="Denoising process"):
        with torch.no_grad():
            x = ddpm_update_helper(x, model, noise_schedule, alpha_bar, t)

        if t in step_indices_set or t == 0:
            images.append(x.cpu())
    
    # Create a grid of images for visualization
    # Build a single image where each row is one sample across denoising steps.
    # frames: List[(B, C, H, W)] -> (B, S, C, H, W)
    frames = torch.stack([_to_fid_uint8(img) for img in images], dim=1)

    # Reorder and tile to a 2D canvas for PIL: (B*H, S*W, C)
    # Rows = samples, columns = denoising checkpoints.
    tiled = frames.permute(0, 3, 1, 4, 2).reshape(
        num_samples * resolution[0],
        frames.shape[1] * resolution[1],
        3,
    )

    PIL.Image.fromarray(tiled.numpy()).save(save_path)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Inference script for DDPM model")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = args.model_path
    if model_path is None:
        raise ValueError("Model path must be provided for inference.")

    model = load_model(model_path, config, device)
    noise_schedule = calculate_noise_schedule(
        config["model"]["T"],
        config["model"]["beta_start"],
        config["model"]["beta_end"],
    ).to(device)

    # fid_num_real = config["evaluation"]["fid_samples"]
    # fid_num_fake = config["evaluation"]["fid_samples"]
    # fid_batch_size = config["evaluation"]["fid_batch_size"]

    # metrics = calculate_metrics(
    #     model=model,
    #     noise_schedule=noise_schedule,
    #     num_real=fid_num_real,
    #     num_fake=fid_num_fake,
    #     batch_size=fid_batch_size,
    #     device=device,
    # )
    # print(f"FID ({fid_num_fake} fake / {fid_num_real} real): {metrics['FID']:.4f}")

    samples = ddim_sample(model, noise_schedule, 10, device, eta=0.2)
    save_samples(samples, "ddim_test_samples")

    # plot_denoising_process(
    #     model=model,
    #     noise_schedule=noise_schedule,
    #     num_samples=5,
    #     device=device,
    #     step_interval=10,
    #     spacing_strength=2.0,
    #     resolution=(32, 32),
    #     save_path="denoising_process.png",
    # )
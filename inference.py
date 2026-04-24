
import os
import PIL
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Unet import UNet
from diffusion import calculate_noise_schedule, calculate_alpha_bar, load_cifar10_data

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

def sample(model: torch.nn.Module,
           noise_schedule: torch.Tensor,
           alpha_bar: torch.Tensor, 
           num_samples: int, device: 
           torch.device) -> torch.Tensor:
    """
    Function to generate samples from the trained model.
    @author: Stephen Krol

    :param model: The trained diffusion model.
    :type model: torch.nn.Module
    :param noise_schedule: The noise schedule used during training.
    :type noise_schedule: torch.Tensor
    :param alpha_bar: The alpha_bar values for the noise schedule.
    :type alpha_bar: torch.Tensor
    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :param device: The device to run the sampling on (e.g., 'cuda' or 'cpu').
    :type device: torch.device

    :return: A tensor containing the generated samples.
    :rtype: torch.Tensor
    """

    noise_schedule = noise_schedule.to(device)
    alpha_bar = alpha_bar.to(device)
    x = torch.randn(num_samples, 3, 32, 32, device=device)  # Start with random noise

    for t in tqdm(reversed(range(noise_schedule.shape[0]))):
        with torch.no_grad():
            
            T = torch.full((x.size(0),), t, device=device, dtype=torch.long)  # Create a tensor for the current timestep   
            
            noise_pred =  model(x, T)  # Predict the noise at the current timestep
            a_t = 1 - noise_schedule[t]

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = 1 / a_t.sqrt() * (x - (1 - a_t) / (1 - alpha_bar[t]).sqrt() * noise_pred) + noise_schedule[t].sqrt() * z  # Update x using the predicted noise

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

    model = UNet()  # Initialize your model architecture
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()


    return model


def _denormalize_from_model_space(images: torch.Tensor) -> torch.Tensor:
    """Converts images from training space [-1, 1] to [0, 1]."""
    return ((images.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def _to_fid_uint8(images: torch.Tensor) -> torch.Tensor:
    """Formats image tensors to uint8 [0, 255] for torchmetrics FID."""
    images = _denormalize_from_model_space(images)
    return (images * 255.0).round().to(torch.uint8)


def calculate_fid(model: torch.nn.Module,
                  noise_schedule: torch.Tensor,
                  alpha_bar: torch.Tensor,
                  num_real: int,
                  num_fake: int,
                  batch_size: int,
                  device: torch.device) -> float:
    """
    Calculates FID against the CIFAR10 training split using generated samples.
    This matches the evaluation setup reported in the DDPM paper.
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

    seen_fake = 0
    while seen_fake < num_fake:
        current_batch = min(batch_size, num_fake - seen_fake)
        fake_batch = sample(model, noise_schedule, alpha_bar, current_batch, device)
        fid.update(_to_fid_uint8(fake_batch), real=False)
        seen_fake += current_batch

    return float(fid.compute().item())


def calculate_metrics(model: torch.nn.Module,
                      noise_schedule: torch.Tensor,
                      alpha_bar: torch.Tensor,
                      num_real: int,
                      num_fake: int,
                      batch_size: int,
                      device: torch.device) -> dict:
    """
    Function to calculate evaluation metrics for generated samples.

    :return: A dictionary containing the calculated metrics.
    :rtype: dict
    """

    fid_score = calculate_fid(
        model=model,
        noise_schedule=noise_schedule,
        alpha_bar=alpha_bar,
        num_real=num_real,
        num_fake=num_fake,
        batch_size=batch_size,
        device=device,
    )

    metrics = {
        "FID": fid_score,
        "IS": None,
    }
    
    return metrics

def save_samples(samples: torch.Tensor, save_path: str):
    """
    Function to save generated samples to disk.
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
        sample = (sample - sample.min()) / (sample.max() - sample.min())  # Normalize to [0, 1]
        sample = (sample * 255).byte()  # Scale to [0, 255] and convert to uint8
        sample = sample.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)

        PIL.Image.fromarray(sample.numpy()).save(f"{save_path}/sample_{i}.png")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = "checkpoints/2026-04-23_10-57-54/ema_model.pth"

    model = load_model(model_path, config, device)
    noise_schedule = calculate_noise_schedule(
        config["cifar10"]["T"],
        config["cifar10"]["beta_start"],
        config["cifar10"]["beta_end"],
    ).to(device)
    alpha_bar = calculate_alpha_bar(noise_schedule).to(device)

    fid_num_real = 10000
    fid_num_fake = 10000
    fid_batch_size = 128

    metrics = calculate_metrics(
        model=model,
        noise_schedule=noise_schedule,
        alpha_bar=alpha_bar,
        num_real=fid_num_real,
        num_fake=fid_num_fake,
        batch_size=fid_batch_size,
        device=device,
    )
    print(f"FID ({fid_num_fake} fake / {fid_num_real} real): {metrics['FID']:.4f}")

    samples = sample(model, noise_schedule, alpha_bar, 10, device)
    save_samples(samples, "test_samples")
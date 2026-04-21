
from typing import Tuple, Optional, Dict, Any

import os
import datetime
import yaml
import random
import torch
import wandb
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from Unet import UNet

from utils import plot_images, LABEL_TO_CLASS, plot_image_noisy_pairs, plot_random_images

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
                             device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
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

    :return: the noisy version of the image at the given timestep and the noise
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    alpha_bar = calculate_alpha_bar(noise_schedule)
    noise = torch.randn_like(x_0).to(device)

    return torch.sqrt(alpha_bar[t])*x_0 + torch.sqrt(1 - alpha_bar[t])*noise, noise


class DiffusionDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the diffusion process.
    @author: Stephen Krol
    """

    def __init__(self, dataset: torch.utils.data.Dataset, noise_schedule: torch.Tensor, device="cpu"):
        """
        Initializes the dataset with the given dataset and noise schedule.
        @author: Stephen Krol

        :param dataset: the original dataset
        :type dataset: torch.utils.data.Dataset
        :param noise_schedule: the noise schedule for the diffusion process
        :type noise_schedule: torch.Tensor
        :param device: the device to run the computation on
        :type device: str
        """
        self.dataset = dataset
        self.noise_schedule = noise_schedule
        self.device = device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x_0, label = self.dataset[idx]
        t = random.randint(0, len(self.noise_schedule) - 1)
        x_t, noise = forward_diffusion_sample(x_0, t, self.noise_schedule, self.device)
        return x_t, t, label, noise
    
class Trainer:
    """
    A trainer class for the diffusion model.
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataset: torch.utils.data.Dataset, 
                 val_dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 lr: float,
                 lr_scheduler: bool = True,
                 wandb_config: Optional[Dict[str, Any]] = None):
        
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=10, gamma=0.1)
        else:
            self.scheduler = None

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.loss_fn = torch.nn.MSELoss()
        self.global_step = 0

        self.wandb_enabled = bool(wandb_config and wandb_config.get("enabled", False))
        self.wandb_log_every_steps = 10
        if self.wandb_enabled:
            self.wandb_log_every_steps = int(wandb_config.get("log_every_steps", 10))
            run_name = wandb_config.get("run_name")

            wandb.init(
                project=wandb_config.get("project", "ddpm-cifar10"),
                entity=wandb_config.get("entity"),
                name=run_name,
                tags=wandb_config.get("tags"),
                notes=wandb_config.get("notes"),
                mode=wandb_config.get("mode", "online"),
                config={
                    "batch_size": batch_size,
                    "lr": lr,
                    "lr_scheduler": bool(lr_scheduler),
                    "model": type(model).__name__,
                },
            )
            wandb.watch(self.model, log="all", log_freq=self.wandb_log_every_steps)

        base_dir = "checkpoints"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.save_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.save_dir, exist_ok=True)

    
    def train(self, num_epochs: int) -> None:
        """
        Trains the model for the given number of epochs.
        """

        for epoch in range(num_epochs):

            train_loss_total = 0.0
            train_batches = 0
            train_pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
                leave=False,
            )
            
            for batch in train_pbar:
                x_t, t, labels, eps = batch
                x_t, t, labels, eps = x_t.to(self.device), t.to(self.device), labels.to(self.device), eps.to(self.device)

                self.optimiser.zero_grad()
                eps_hat = self.model(x_t, t)

                loss = self.loss_fn(eps_hat, eps)
                loss.backward()
                self.optimiser.step()
                self.global_step += 1
                train_loss_total += loss.item()
                train_batches += 1

                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

                if self.wandb_enabled and self.global_step % self.wandb_log_every_steps == 0:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/lr": self.optimiser.param_groups[0]["lr"],
                        "global_step": self.global_step,
                        "epoch": epoch + 1,
                    }, step=self.global_step)
            

            if self.scheduler is not None:
                self.scheduler.step()

            # calculate validation loss at the end of each epoch
            val_loss = 0.0
            with torch.no_grad():
                val_pbar = tqdm(
                    self.val_dataloader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
                    leave=False,
                )
                for batch in val_pbar:
                    x_t, t, labels, eps = batch
                    x_t, t, labels, eps = x_t.to(self.device), t.to(self.device), labels.to(self.device), eps.to(self.device)

                    eps_hat = self.model(x_t, t)
                    loss = self.loss_fn(eps_hat, eps)
                    val_loss += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= len(self.val_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            train_loss = train_loss_total / max(train_batches, 1)
            if self.wandb_enabled:
                wandb.log({
                    "train/loss_epoch": train_loss,
                    "val/loss_epoch": val_loss,
                    "train/lr_epoch": self.optimiser.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }, step=self.global_step)

        if self.wandb_enabled:
            wandb.finish()


if __name__ == "__main__":

    with open("CIFAR10.yaml", "r") as f:
        config = yaml.safe_load(f)

    noise_schedule = calculate_noise_schedule(config["params"]["T"], config["params"]["beta_start"], config["params"]["beta_end"])  

    train_set, test_set = load_cifar10_data()

    train_dataset = DiffusionDataset(train_set, noise_schedule)
    val_dataset = DiffusionDataset(test_set, noise_schedule)

    yaml_path = "config.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model = UNet(original_channels=3, 
                 base_channels=config["cifar10"]["base_channels"], 
                 channel_multipliers=config["cifar10"]["channel_multipliers"],
                 num_res_blocks=config["cifar10"]["num_res_blocks"],
                 in_resolution=config["cifar10"]["in_resolution"])
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        config["training"]["batch_size"],
        float(config["training"]["lr"]),
        config["training"]["lr_scheduler"],
        config.get("wandb", {"enabled": False}),
    )
    trainer.train(config["training"]["num_epochs"])


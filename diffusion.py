
from typing import Tuple, Optional, Dict, Any

import os
import datetime
import copy
import yaml
import torch
import wandb
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from Unet import UNet

from utils import plot_images, LABEL_TO_CLASS, plot_image_noisy_pairs, plot_random_images

EMA_DECAY = 0.9999

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
                             alpha_bar: torch.Tensor,
                             device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes an image and a timestep as input and returns the noisy version of it
    at the given timestep.
    @author: Stephen Krol

    :param x_0: the original image
    :type x_0: torch.Tensor
    :param t: the timestep
    :type t: int
    :param alpha_bar: the alpha_bar values for the noise schedule
    :type alpha_bar: torch.Tensor
    :param device: the device to run the computation on
    :type device: str

    :return: the noisy version of the image at the given timestep and the noise
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    noise = torch.randn_like(x_0).to(device)

    return torch.sqrt(alpha_bar[t])*x_0 + torch.sqrt(1 - alpha_bar[t])*noise, noise


class DiffusionDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the diffusion process.
    @author: Stephen Krol
    """

    def __init__(self, dataset: torch.utils.data.Dataset, 
                 noise_schedule: torch.Tensor, 
                 train: bool=True,
                 device="cpu"):
        """
        Initializes the dataset with the given dataset and noise schedule.
        @author: Stephen Krol

        :param dataset: the original dataset
        :type dataset: torch.utils.data.Dataset
        :param noise_schedule: the noise schedule for the diffusion process
        :type noise_schedule: torch.Tensor
        :param train: whether the dataset is for training (applies data augmentation)
        :type train: bool
        :param device: the device to run the computation on
        :type device: str
        """
        self.dataset = dataset
        self.noise_schedule = noise_schedule
        self.alpha_bar = calculate_alpha_bar(noise_schedule)
        self.train = train

        # horizontal flip augmentation for training dataset
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x_0, label = self.dataset[idx]

        if self.train:
            x_0 = self.transform(x_0)

        return x_0, label
    
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
                 warmup_steps: int = 0,
                 lr_scheduler: bool = True,
                 wandb_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the trainer with the given model, datasets, and training parameters.
        @author: Stephen Krol

        :param model: the diffusion model to train
        :type model: torch.nn.Module
        :param train_dataset: the training dataset
        :type train_dataset: torch.utils.data.Dataset
        :param val_dataset: the validation dataset
        :type val_dataset: torch.utils.data.Dataset
        :param batch_size: the batch size for training
        :type batch_size: int
        :param lr: the learning rate for training
        :type lr: float
        :param warmup_steps: number of optimizer steps used for linear learning-rate warmup
        :type warmup_steps: int
        :param lr_scheduler: whether to use a learning rate scheduler
        :type lr_scheduler: bool
        :param wandb_config: the configuration for Weights & Biases logging
        :type wandb_config: dict, optional
        """
        
        # Initialize model and device
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Exponential moving average (EMA) copy used for evaluation/checkpointing.
        self.ema_decay = EMA_DECAY
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Keep diffusion coefficients on device to sample noised batches efficiently.
        self.noise_schedule = train_dataset.noise_schedule.to(self.device)
        self.alpha_bar = train_dataset.alpha_bar.to(self.device)

        # Initialize optimizer and learning rate scheduler
        self.base_lr = float(lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        # Start from lr=0 when warmup is enabled, then ramp linearly to base_lr.
        if self.warmup_steps > 0:
            for param_group in self.optimiser.param_groups:
                param_group["lr"] = 0.0

        # Set up learning rate scheduler if enabled
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=10, gamma=0.1)
        else:
            self.scheduler = None

        # Set up data loaders with worker and memory settings for better throughput.
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = self.device == "cuda"

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

        # Set up loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Initialize training state and Weights & Biases logging
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

        # Set up directory for saving checkpoints
        base_dir = "checkpoints"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.save_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.ema_checkpoint_path = os.path.join(self.save_dir, "ema_model.pth")

    def _set_warmup_lr(self, step: int) -> None:
        """Sets the learning rate for the given global optimizer step during warmup."""

        if self.warmup_steps <= 0:
            return

        if step <= self.warmup_steps:
            warmup_progress = float(step) / float(self.warmup_steps)
            lr = self.base_lr * warmup_progress
            for param_group in self.optimiser.param_groups:
                param_group["lr"] = lr

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Updates EMA model parameters from the current trainable model."""

        model_state = self.model.state_dict()
        ema_state = self.ema_model.state_dict()
        for key, model_val in model_state.items():
            ema_val = ema_state[key]
            if torch.is_floating_point(ema_val):
                ema_val.mul_(self.ema_decay).add_(model_val.detach(), alpha=1.0 - self.ema_decay)
            else:
                ema_val.copy_(model_val)

    def _batch_forward_diffusion_sample(self, x_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies forward diffusion to a whole batch on-device.

        :param x_0: clean input images with shape [B, C, H, W]
        :type x_0: torch.Tensor

        :return: tuple of (x_t, t, noise)
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        batch_size = x_0.size(0)
        t = torch.randint(0, self.alpha_bar.shape[0], (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, t, noise

    
    def train(self, num_epochs: int) -> None:
        """
        Trains the model for the given number of epochs.
        @author: Stephen Krol

        :param num_epochs: the number of epochs to train for
        :type num_epochs: int
        """

        best_val_loss = float("inf")

        # iterate over epochs and batches, calculating training and validation loss, and logging to Weights & Biases
        for epoch in range(num_epochs):

            self.model.train()
            train_loss_total = 0.0
            train_batches = 0
            train_pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
                leave=False,
            )

            for batch in train_pbar:
                x_0, labels = batch
                x_0 = x_0.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                x_t, t, eps = self._batch_forward_diffusion_sample(x_0)

                # Apply linear LR warmup per optimizer step.
                self._set_warmup_lr(self.global_step + 1)

                self.optimiser.zero_grad()
                eps_hat = self.model(x_t, t)

                loss = self.loss_fn(eps_hat, eps)
                loss.backward()
                self.optimiser.step()
                
                self._update_ema()

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
            

            if self.scheduler is not None and self.global_step >= self.warmup_steps:
                self.scheduler.step()

            # calculate validation loss at the end of each epoch
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                val_pbar = tqdm(
                    self.val_dataloader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
                    leave=False,
                )
                for batch in val_pbar:
                    x_0, labels = batch
                    x_0 = x_0.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    x_t, t, eps = self._batch_forward_diffusion_sample(x_0)

                    eps_hat = self.model(x_t, t)
                    loss = self.loss_fn(eps_hat, eps)
                    val_loss += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= len(self.val_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # save model checkpoint if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))

            train_loss = train_loss_total / max(train_batches, 1)

            # Save EMA weights every 100 epochs to a fixed file, overwriting prior EMA checkpoint.
            if (epoch + 1) % 100 == 0:
                torch.save(self.ema_model.state_dict(), self.ema_checkpoint_path)

            if self.wandb_enabled:
                wandb.log({
                    "train/loss_epoch": train_loss,
                    "val/loss_epoch": val_loss,
                    "train/lr_epoch": self.optimiser.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }, step=self.global_step)

        if self.wandb_enabled:
            wandb.finish()
        
        torch.save(self.ema_model.state_dict(), self.ema_checkpoint_path)


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    noise_schedule = calculate_noise_schedule(config["cifar10"]["T"], config["cifar10"]["beta_start"], config["cifar10"]["beta_end"])  

    train_set, test_set = load_cifar10_data()

    train_dataset = DiffusionDataset(train_set, noise_schedule)
    val_dataset = DiffusionDataset(test_set, noise_schedule, train=False)

    model = UNet(original_channels=3, 
                 base_channels=config["cifar10"]["base_channels"], 
                 channel_multipliers=config["cifar10"]["channel_multipliers"],
                 num_res_blocks=config["cifar10"]["num_res_blocks"],
                 in_resolution=config["cifar10"]["in_resolution"])
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["training"]["batch_size"],
        lr=float(config["training"]["lr"]),
        warmup_steps=int(config["training"].get("warmup_steps", 0)),
        lr_scheduler=False,
        wandb_config=config.get("wandb", {"enabled": False}),
    )

    trainer.train(config["training"]["num_epochs"])


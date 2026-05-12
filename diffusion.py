
from typing import Tuple, Optional, Dict, Any

import os
import datetime
import copy
import yaml
import torch
import torch.distributed as dist
import wandb
import argparse
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from Unet import UNet

from utils import unwrap_model, setup_distributed_training, cleanup_distributed_training, create_checkpoint_dir, configure_tensor_core_backend
from datasets.cifar10 import load_cifar10_data
from datasets.celeba_hq256 import load_celeba_hq256_data

# TODO: move to config
EMA_DECAY = 0.9999


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

    def __init__(self, 
                 dataset: torch.utils.data.Dataset, 
                 noise_schedule: torch.Tensor, 
                 train: bool=True):
        """
        Initializes the dataset with the given dataset and noise schedule.
        @author: Stephen Krol

        :param dataset: the original dataset
        :type dataset: torch.utils.data.Dataset
        :param noise_schedule: the noise schedule for the diffusion process
        :type noise_schedule: torch.Tensor
        :param train: whether the dataset is for training (applies data augmentation)
        :type train: bool
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
                 use_mixed_precision: bool = True,
                 warmup_steps: int = 0,
                 lr_scheduler: bool = True,
                 runtime_context: Optional[Dict[str, Any]] = None,
                 distributed_config: Optional[Dict[str, Any]] = None,
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
        :param use_mixed_precision: whether to run training in mixed precision on CUDA
        :type use_mixed_precision: bool
        :param warmup_steps: number of optimizer steps used for linear learning-rate warmup
        :type warmup_steps: int
        :param lr_scheduler: whether to use a learning rate scheduler
        :type lr_scheduler: bool
        :param runtime_context: distributed runtime state such as rank and device
        :type runtime_context: dict, optional
        :param distributed_config: configuration for distributed training
        :type distributed_config: dict, optional
        :param wandb_config: the configuration for Weights & Biases logging
        :type wandb_config: dict, optional
        """

        # set context and configuration defaults, then initialize distributed training if enabled, and log the effective training precision
        runtime_context = runtime_context or {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main_process": True,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        distributed_config = distributed_config or {}
        wandb_config = wandb_config or {"enabled": False}

        # parameters for distributed training and mixed precision are set as attributes on the trainer for use in the training loop
        self.distributed = bool(runtime_context["enabled"])
        self.rank = int(runtime_context["rank"])
        self.world_size = int(runtime_context["world_size"])
        self.is_main_process = bool(runtime_context["is_main_process"])
        self.device = runtime_context["device"]
        self.find_unused_parameters = bool(distributed_config.get("find_unused_parameters", False))

        # initialise model and set distributed data parallel if enabled, keeping an unwrapped reference for checkpointing and EMA updates
        self.model = model.to(self.device)

        if self.distributed:
            ddp_kwargs: Dict[str, Any] = {
                "find_unused_parameters": self.find_unused_parameters,
            }

            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            self.model = DistributedDataParallel(self.model, **ddp_kwargs)

        self.model_for_saving = unwrap_model(self.model)

        # Enable mixed precision on CUDA to unlock Tensor Core kernels.
        self.amp_enabled = (self.device.type == "cuda" and bool(use_mixed_precision))

        # set precision for autocasting based on config and CUDA capabilities, and log the effective training precision
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float16

        # Use a GradScaler for mixed precision training to prevent underflow of small gradients when float16.
        self.scaler = torch.amp.GradScaler(
            enabled=self.amp_enabled and self.autocast_dtype == torch.float16
        )

        amp_mode = str(self.autocast_dtype).replace("torch.", "") if self.amp_enabled else "fp32"
        self._log(
            f"[precision] device={self.device} amp={'on' if self.amp_enabled else 'off'} dtype={amp_mode}"
        )

        # Exponential moving average (EMA) copy used for evaluation/checkpointing.
        self.ema_decay = EMA_DECAY
        self.ema_model = copy.deepcopy(self.model_for_saving).to(self.device)
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
        num_workers = min(12, os.cpu_count() or 1)
        pin_memory = self.device.type == "cuda"

        # setup data sampler for distributed training, and ensure validation dataloader 
        # is only created on the main process to avoid redundant work and contention 
        # on disk access in distributed training
        self.train_sampler = None
        if self.distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        self.val_dataloader = None
        if not self.distributed or self.is_main_process:
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
        self.wandb_enabled = bool(wandb_config.get("enabled", False) and self.is_main_process)
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
                    "model": type(self.model_for_saving).__name__,
                    "distributed": self.distributed,
                    "world_size": self.world_size,
                },
            )
            wandb.watch(self.model_for_saving, log="all", log_freq=self.wandb_log_every_steps)

        # Set up directory for saving checkpoints
        project_name = wandb_config.get("project", "ddpm-training")
        self.save_dir = create_checkpoint_dir(project_name, runtime_context)
        self.ema_checkpoint_path = os.path.join(self.save_dir, "ema_model.pth")
        self.interrupt_checkpoint_path = os.path.join(self.save_dir, "interrupt_model.pth")

    def train(self, num_epochs: int) -> None:
        """
        Trains the model for the given number of epochs.
        @author: Stephen Krol

        :param num_epochs: the number of epochs to train for
        :type num_epochs: int
        """

        best_val_loss = float("inf")
        interrupted = False

        try:
            # iterate over epochs and batches, calculating training and validation loss, and logging to Weights & Biases
            for epoch in range(num_epochs):
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                self.model.train()
                train_loss_total = 0.0
                train_samples = 0
                train_iterator = self.train_dataloader
                if self.is_main_process:
                    train_iterator = tqdm(
                        self.train_dataloader,
                        desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
                        leave=False,
                    )

                for batch in train_iterator:
                    x_0, _ = batch
                    x_0 = x_0.to(self.device, non_blocking=True)
                    x_t, t, eps = self._batch_forward_diffusion_sample(x_0)

                    # Apply linear LR warmup per optimizer step.
                    self._set_warmup_lr(self.global_step + 1)

                    self.optimiser.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.amp_enabled):
                        eps_hat = self.model(x_t, t)
                        loss = self.loss_fn(eps_hat, eps)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimiser)
                    self.scaler.update()

                    self._update_ema()

                    self.global_step += 1
                    batch_size_actual = x_0.size(0)
                    train_loss_total += loss.item() * batch_size_actual
                    train_samples += batch_size_actual

                    if self.is_main_process:
                        train_iterator.set_postfix(loss=f"{loss.item():.4f}")

                    if self.wandb_enabled and self.global_step % self.wandb_log_every_steps == 0:
                        wandb.log({
                            "train/loss_step": loss.item(),
                            "train/lr": self.optimiser.param_groups[0]["lr"],
                            "global_step": self.global_step,
                            "epoch": epoch + 1,
                        }, step=self.global_step)

                if self.scheduler is not None and self.global_step >= self.warmup_steps:
                    self.scheduler.step()

                train_loss_total, train_samples = self._reduce_totals(train_loss_total, train_samples)
                train_loss = train_loss_total / max(train_samples, 1)

                # calculate validation loss at the end of each epoch
                val_loss = None
                if self.val_dataloader is not None:
                    val_loss_total = 0.0
                    val_samples = 0
                    self.model.eval()
                    with torch.no_grad():
                        val_iterator = self.val_dataloader
                        if self.is_main_process:
                            val_iterator = tqdm(
                                self.val_dataloader,
                                desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
                                leave=False,
                            )
                        for batch in val_iterator:
                            x_0, _ = batch
                            x_0 = x_0.to(self.device, non_blocking=True)
                            x_t, t, eps = self._batch_forward_diffusion_sample(x_0)

                            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.amp_enabled):
                                eps_hat = self.model(x_t, t)
                                loss = self.loss_fn(eps_hat, eps)

                            batch_size_actual = x_0.size(0)
                            val_loss_total += loss.item() * batch_size_actual
                            val_samples += batch_size_actual
                            if self.is_main_process:
                                val_iterator.set_postfix(loss=f"{loss.item():.4f}")

                    val_loss = val_loss_total / max(val_samples, 1)
                    self._log(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

                # save model checkpoint if validation loss has improved
                if self.is_main_process and val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model_for_saving.state_dict(), os.path.join(self.save_dir, "best_model.pth"))

                # Save EMA weights every 100 epochs to a fixed file, overwriting prior EMA checkpoint.
                if self.is_main_process and (epoch + 1) % 100 == 0:
                    torch.save(self.ema_model.state_dict(), self.ema_checkpoint_path)

                if self.wandb_enabled:
                    log_payload = {
                        "train/loss_epoch": train_loss,
                        "train/lr_epoch": self.optimiser.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    }
                    if val_loss is not None:
                        log_payload["val/loss_epoch"] = val_loss
                    wandb.log(log_payload, step=self.global_step)

                if self.distributed:
                    dist.barrier()

        except KeyboardInterrupt:
            interrupted = True
            self._log("Training interrupted by user. Shutting down cleanly.")
        finally:
            if self.wandb_enabled:
                wandb.finish()

            if self.is_main_process:
                if interrupted:
                    torch.save(self.model_for_saving.state_dict(), self.interrupt_checkpoint_path)
                    torch.save(self.ema_model.state_dict(), self.ema_checkpoint_path)
                else:
                    torch.save(self.ema_model.state_dict(), self.ema_checkpoint_path)

        return not interrupted

    def _log(self, message: str) -> None:
        """
        Prints only from the main process when running distributed training.
        @author: Stephen Krol

        :param message: the message to print
        :type message: str
        """

        if self.is_main_process:
            print(message)

    def _reduce_totals(self, loss_total: float, sample_total: int) -> Tuple[float, int]:
        """
        All-reduces accumulated loss and sample counts across ranks.
        @author: Stephen Krol

        :param loss_total: the total loss to reduce
        :type loss_total: float
        :param sample_total: the total number of samples to reduce
        :type sample_total: int
        :return: tuple of (reduced_loss_total, reduced_sample_total)
        :rtype: tuple[float, int]
        """

        if not self.distributed:
            return loss_total, sample_total

        totals = torch.tensor([loss_total, sample_total], device=self.device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return float(totals[0].item()), int(totals[1].item())

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

        model_state = self.model_for_saving.state_dict()
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


if __name__ == "__main__":

    # initialise parser for cli arguments
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--dataset", help="Dataset to use (cifar10 or celeba_hq256)")
    args = parser.parse_args()

    # read config file and set up distributed training
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_config = config.get("training", {})
    distributed_config = training_config.get("distributed", {})
    runtime_context = setup_distributed_training(distributed_config)


    try:

        # configure precision and Tensor Core settings based on config, and log the effective training precision
        precision_mode = str(config.get("precision", {}).get("mode", "mixed")).lower()

        if precision_mode not in {"mixed", "fp32"}:
            raise ValueError(f"Unsupported precision.mode: {precision_mode}. Use 'mixed' or 'fp32'.")
        
        use_mixed_precision = (precision_mode == "mixed")
        configure_tensor_core_backend(enabled=use_mixed_precision)

        # calculate noise schedule and load datasets based on config, 
        # then initialize model and trainer
        noise_schedule = calculate_noise_schedule(config["model"]["T"], config["model"]["beta_start"], config["model"]["beta_end"])  

        if args.dataset == "cifar10":
            train_set, test_set = load_cifar10_data()
        elif args.dataset == "celeba_hq256":
            train_set, test_set = load_celeba_hq256_data()
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        train_dataset = DiffusionDataset(train_set, noise_schedule)
        val_dataset = DiffusionDataset(test_set, noise_schedule, train=False)

        model = UNet(original_channels=3, 
                     base_channels=config["model"]["base_channels"], 
                     channel_multipliers=config["model"]["channel_multipliers"],
                     num_res_blocks=config["model"]["num_res_blocks"],
                     in_resolution=config["model"]["in_resolution"])

        if runtime_context["is_main_process"]:
            print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=training_config["batch_size"],
            lr=float(training_config["lr"]),
            use_mixed_precision=use_mixed_precision,
            warmup_steps=int(training_config.get("warmup_steps", 0)),
            lr_scheduler=False,
            runtime_context=runtime_context,
            distributed_config=distributed_config,
            wandb_config=config.get("wandb", {"enabled": False}),
        )

        # run training loop, and save an interrupt checkpoint if training is stopped early by the user
        completed = trainer.train(training_config["num_epochs"])
        if not completed and runtime_context["is_main_process"]:
            print(f"Saved interrupt checkpoint to {trainer.interrupt_checkpoint_path}")

    except KeyboardInterrupt:
        if runtime_context["is_main_process"]:
            print("Training interrupted by user.")

    finally:
        cleanup_distributed_training(runtime_context)


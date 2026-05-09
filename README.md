# DDPM Implementation

This repository contains a from-scratch PyTorch implementation of denoising diffusion probabilistic models (DDPM), plus DDIM sampling utilities.

The current training pipeline is configured for CelebA-HQ 256x256 images, with CIFAR-10 utilities still available for evaluation experiments.

## What is implemented

- Linear noise schedule with configurable T, beta_start, and beta_end.
- Forward diffusion and random-timestep training objective (noise prediction).
- DDPM U-Net backbone with residual blocks, timestep embeddings, and attention.
- Exponential moving average (EMA) model tracking during training.
- Optional Weights and Biases logging.
- DDPM and DDIM sampling in inference.
- Optional FID and Inception Score utilities in inference.

## Repository layout

```text
.
├── diffusion.py                 # training entrypoint and trainer
├── inference.py                 # sampling and optional evaluation metrics
├── Unet.py                      # DDPM U-Net
├── utils.py                     # plotting/helpers
├── CELEBA-HQ_config.yaml        # CelebA-HQ 256 training config
├── CIFAR10_config.yaml          # CIFAR-10 config template/experiment config
├── datasets/
│   ├── celeba_hq256.py          # CelebA-HQ disk loader
│   └── cifar10.py               # CIFAR-10 loader
├── checkpoints/                 # saved run directories
├── data/                        # local datasets
└── DEPENDENCIES.md              # dependency and environment details
```

## Environment setup

Use the existing conda environment:

```bash
conda activate aesthetic-evolution
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

Install optional evaluation dependencies (for FID/IS in inference):

```bash
pip install -r requirements-eval.txt
```

For exact environment reproduction, see environment.yml and DEPENDENCIES.md.

## Dataset setup

### CelebA-HQ 256

Place image files in:

```text
data/celeba_hq_256/
```

Supported file extensions are .jpg, .jpeg, .png, .webp, and .bmp.

The current training entrypoint in diffusion.py uses the CelebA-HQ loader.

### CIFAR-10

CIFAR-10 is downloaded automatically to data/ when using torchvision loaders.

## Configuration files

Both YAML config files follow the same structure:

- model:
	- channel_multipliers
	- base_channels
	- num_res_blocks
	- in_resolution
	- T
	- beta_start
	- beta_end
- training:
	- batch_size
	- lr
	- warmup_steps
	- lr_scheduler (defined in YAML, currently disabled in the training entrypoint)
	- num_epochs
- evaluation:
	- fid_samples
	- fid_batch_size
- wandb:
	- enabled
	- project
	- entity
	- run_name
	- mode
	- log_every_steps
	- tags
	- notes

## Train

Run training with a config path:

```bash
python diffusion.py --config CELEBA-HQ_config.yaml
```

What the trainer does:

- Uses GPU if available.
- Applies random horizontal flips on training images.
- Uses Adam and linear warmup for learning rate.
- Tracks an EMA copy of model weights.
- Evaluates on a validation split each epoch.
- Saves checkpoints to:

```text
checkpoints/<wandb.project>/<timestamp>/
```

Checkpoint files:

- best_model.pth: best validation-loss checkpoint.
- ema_model.pth: EMA weights (saved periodically and again at end of training).

## Inference and sampling

Run inference with:

```bash
python inference.py --model_path checkpoints/<project>/<run_timestamp>/best_model.pth --config_path CELEBA-HQ_config.yaml
```

Current default behavior in inference.py:

- Loads the model checkpoint and noise schedule from config.
- Generates 10 samples with DDIM (eta=0.2).
- Saves outputs to:

```text
ddim_test_samples/
```

FID and Inception Score helper functions are implemented and can be enabled in inference.py when needed.

## Notes

- This codebase is an implementation-focused project and evolves with ongoing experiments.
- CelebA-HQ is the primary training target at the moment.
- CIFAR-10 config and loaders remain useful for metric/evaluation workflows and smaller-scale experiments.

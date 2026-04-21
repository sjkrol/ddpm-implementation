# DDPM From Scratch

> This repository contains **my own implementation of DDPM** (Denoising Diffusion Probabilistic Models), written by me with only **minimal AI input**. The goal of this project is to understand the model by building the core pieces myself in PyTorch.

## Overview

This project is a personal diffusion-model implementation focused on CIFAR-10. It includes the main ingredients needed to train a DDPM-style noise prediction model:

- a linear noise schedule
- the forward diffusion process
- a custom dataset wrapper that samples random timesteps
- a U-Net backbone with residual blocks, time embeddings, and self-attention
- a training loop with validation
- plotting utilities for visualising data and noisy samples

This is primarily a **learning and implementation project**, rather than a polished production training framework.

## Project Structure

```text
.
├── diffusion.py      # dataset loading, diffusion utilities, training loop
├── Unet.py           # U-Net architecture used for noise prediction
├── utils.py          # plotting and visualisation helpers
├── config.yaml       # model and training hyperparameters
├── CIFAR10.yaml      # diffusion schedule parameters
├── checkpoints/      # saved training runs
└── data/             # CIFAR-10 dataset
```

## Implemented Components

### Diffusion process
The implementation includes:

- linear $\beta$ schedule from `beta_start` to `beta_end`
- computation of $\bar{\alpha}_t$
- forward noising of clean images into noisy samples

### Model
The model is a DDPM-style U-Net with:

- residual convolutional blocks
- sinusoidal timestep embeddings
- GroupNorm + SiLU activations
- dropout
- self-attention at selected resolutions
- skip connections between encoder and decoder

### Training
The training code:

- loads CIFAR-10 using `torchvision`
- creates noisy examples on the fly
- trains the model to predict the injected Gaussian noise
- reports training and validation loss
- creates timestamped directories inside `checkpoints/`

## Configuration

There are two config files:

### `CIFAR10.yaml`
Controls the diffusion schedule:

- `T`: number of diffusion steps
- `beta_start`: starting noise value
- `beta_end`: ending noise value

### `config.yaml`
Controls the model and optimisation setup:

- `channel_multipliers`
- `base_channels`
- `num_res_blocks`
- `in_resolution`
- `batch_size`
- `lr`
- `num_epochs`

## Installation

Create a Python environment and install the main dependencies:

```bash
pip install torch torchvision matplotlib pyyaml wandb
```

## Running Training

From the repository root, run:

```bash
python diffusion.py
```

To enable Weights & Biases logging, set `wandb.enabled: true` in `config.yaml`.
You can also set `wandb.project`, `wandb.entity`, `wandb.run_name`, and `wandb.log_every_steps`.

The dataset will download automatically into the `data/` directory if needed.

## Notes

- This repo is intended to show my understanding of DDPMs through implementation.
- The code is still a work in progress and may be refined as I continue experimenting.
- The training script currently uses a local absolute path for `config.yaml`, so if you move the repo to a different location you may need to update that path.

## Motivation

I built this project to better understand how diffusion models work internally instead of only using prebuilt libraries. Writing the components myself helped me learn the mechanics of the forward process, timestep conditioning, and U-Net-based noise prediction.

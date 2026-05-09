# Dependency Documentation

This project is intended to run in the conda environment named `aesthetic-evolution`.

## Environment

Use the existing environment:

```bash
conda activate aesthetic-evolution
```

If it does not exist yet on another machine, create it and activate it:

```bash
conda create -n aesthetic-evolution python=3.10 -y
conda activate aesthetic-evolution
```

## Install Dependencies

For exact reproduction of the full `aesthetic-evolution` environment on another machine, use the exported lock file:

```bash
conda env create -f environment.yml
conda activate aesthetic-evolution
```

If you only want the project's direct dependencies (also pinned to exact versions), use the requirements files below.

Upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

Install optional evaluation dependencies (FID and Inception Score):

```bash
pip install -r requirements-eval.txt
```

## Dependency List

The requirements files are now pinned to the exact versions currently installed in the `aesthetic-evolution` environment.

Core packages:

- torch==2.7.0+cu128
- torchvision==0.22.0+cu128
- PyYAML==6.0.3
- tqdm==4.67.1
- matplotlib==3.10.8
- Pillow==12.0.0
- wandb==0.26.0
- numpy==2.3.5

Optional evaluation packages:

- torchmetrics==1.9.0
- torch-fidelity==0.4.0

## Verify Installation

Exact environment verification:

```bash
conda env list | grep aesthetic-evolution
python -c "import torch; print(torch.__version__)"
```

Core verification:

```bash
python -c "import torch, torchvision, yaml, tqdm, matplotlib, PIL, wandb, numpy; print('core dependencies OK')"
```

Optional evaluation verification:

```bash
python -c "from torchmetrics.image.fid import FrechetInceptionDistance; from torchmetrics.image.inception import InceptionScore; print('evaluation dependencies OK')"
```

## Run Commands

Train:

```bash
python diffusion.py --config CELEBA-HQ_config.yaml
```

Inference:

```bash
python inference.py --model_path checkpoints/<run_dir>/best_model.pth --config_path CELEBA-HQ_config.yaml
```

## Dataset Notes

- CIFAR-10 downloads automatically to `./data` via torchvision.
- CelebA-HQ expects images in `./data/celeba_hq_256/`.

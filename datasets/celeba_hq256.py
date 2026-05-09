
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List


class CelebAHQ256DiskDataset(torch.utils.data.Dataset):
    """Loads CelebA-HQ images from a flat directory on disk."""

    def __init__(self, root: str, transform: transforms.Compose):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = self._collect_image_paths()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in '{self.root}'.")

    def _collect_image_paths(self) -> List[Path]:
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return sorted(
            path for path in self.root.iterdir() if path.is_file() and path.suffix.lower() in image_extensions
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        image = self.transform(image)

        # Keep the same (image, label) contract expected by the training pipeline.
        return image, 0

def load_celeba_hq256_data() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the CelebA-HQ 256 dataset (not from torch) and returns the training and test sets.
    @author: Stephen Krol

    :return: the training and test sets
    :rtype: tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
    """

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebAHQ256DiskDataset(root="./data/celeba_hq_256/", transform=transform)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000])

    return train_set, test_set

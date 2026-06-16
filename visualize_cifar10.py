"""Visualise CIFAR-10 real images for side-by-side comparison with generated samples.

Generates an 8×8 grid of real CIFAR-10 test images (matching the 64-sample
grids produced during training) so you can compare real vs generated quality.

Usage
-----
  uv run python visualize_cifar10.py                  # show interactively
  uv run python visualize_cifar10.py --save cifar10_real.png   # save to file
  uv run python visualize_cifar10.py --nrow 10 --count 100     # custom grid size
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid

# CIFAR-10 class names
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_cifar10(count: int = 64) -> tuple[torch.Tensor, list[int]]:
    """Load `count` images from the CIFAR-10 test set (without data augmentation).

    Images are in [-1, 1] range (matching the training pipeline normalisation).
    Returns (images, labels).
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    indices = torch.randperm(len(dataset))[:count]
    images = torch.stack([dataset[i][0] for i in indices])
    labels = [dataset[i][1] for i in indices]
    return images, labels


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Reverse training normalisation: [-1, 1] → [0, 1]."""
    return (images + 1.0) * 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Visualise CIFAR-10 real images for comparison with generated samples"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path instead of showing interactively",
    )
    parser.add_argument(
        "--count", type=int, default=64,
        help="Number of images to show (default: 64, i.e. 8×8 grid)",
    )
    parser.add_argument(
        "--nrow", type=int, default=None,
        help="Number of images per row (default: sqrt(count))",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    nrow = args.nrow or int(args.count ** 0.5)

    # ── Load real CIFAR-10 images ──
    torch.manual_seed(args.seed)
    print(f"Loading {args.count} CIFAR-10 test images...")
    images, labels = load_cifar10(args.count)

    # Denormalise to [0, 1] for display
    images = denormalize(images).clamp(0.0, 1.0)

    # Make grid
    grid = make_grid(images, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()  # CHW → HWC

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np)
    ax.set_title(
        f"CIFAR-10 Real Images ({args.count} samples, {nrow}×{nrow} grid)",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    # Add tiny class labels in the corner of each cell
    cell_w = grid_np.shape[1] / nrow
    cell_h = grid_np.shape[0] / nrow
    for i in range(args.count):
        row, col = divmod(i, nrow)
        x = col * cell_w + 4
        y = row * cell_h + 10
        ax.text(
            x, y, CLASSES[labels[i]],
            fontsize=5, color="white",
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "black", "alpha": 0.5},
        )

    fig.tight_layout(pad=1.5)

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

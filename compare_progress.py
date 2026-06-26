"""Compare generated samples across training progress with real CIFAR-10 images.

Creates a side-by-side figure showing how image quality improves through
training (Epoch 1 → 50 → 150 → 300), alongside real CIFAR-10 images for
reference — even though the MSE loss plateaus after ~100 epochs.

Usage
-----
  uv run python compare_progress.py                          # show interactively
  uv run python compare_progress.py --save progress_demo.png  # save to file
  uv run python compare_progress.py --run 20260615_234944     # specify run dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import make_grid

# CIFAR-10 class names
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_real_cifar10(count: int = 64, seed: int = 42) -> np.ndarray:
    """Load and denormalise `count` real CIFAR-10 test images → HWC numpy [0,1]."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform,
    )
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:count]
    images = torch.stack([dataset[i][0] for i in indices])
    images = (images + 1.0) * 0.5  # denormalise
    images = images.clamp(0.0, 1.0)
    grid = make_grid(images, nrow=8, padding=2)
    return grid.permute(1, 2, 0).numpy()


def load_sample_png(path: Path) -> np.ndarray:
    """Load a saved sample PNG → HWC numpy [0,1]."""
    img = Image.open(path)
    return np.array(img) / 255.0


def main():
    parser = argparse.ArgumentParser(
        description="Compare DDPM sample quality across training epochs"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path instead of showing interactively",
    )
    parser.add_argument(
        "--run", type=str, default="20260615_234944",
        help="Training run ID under outputs/ (default: 20260615_234944)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output DPI (default: 200)",
    )
    args = parser.parse_args()

    samples_dir = Path(f"outputs/{args.run}/samples")

    # ── Define the 4 key snapshots ──
    snapshots = [
        ("Epoch 1\n(step 1)\nLoss ≈ 0.175", "sample_0000001.png"),
        ("Epoch 50\n(step 10,000)\nLoss ≈ 0.058", "sample_0010000.png"),
        ("Epoch 150\n(step 29,500)\nLoss ≈ 0.056", "sample_0029500.png"),
        ("Epoch 300\n(step 58,500)\nLoss ≈ 0.055", "sample_0058500.png"),
    ]

    # ── Load generated samples ──
    gen_images = []
    for label, filename in snapshots:
        path = samples_dir / filename
        if not path.exists():
            print(f"⚠  Missing: {path} — trying to find closest match...")
            # Find closest by step number
            stem = filename.replace(".png", "")
            step_num = int(stem.split("_")[1])
            # Search for nearby files
            existing = sorted(samples_dir.glob("sample_*.png"))
            if existing:
                closest = min(existing, key=lambda p: abs(
                    int(p.stem.split("_")[1]) - step_num
                ))
                path = closest
                print(f"  → using {path.name}")
        if path.exists():
            gen_images.append(load_sample_png(path))
        else:
            gen_images.append(np.zeros((32*8 + 2*7, 32*8 + 2*7, 3)))  # placeholder

    # ── Load real CIFAR-10 ──
    print("Loading real CIFAR-10 images...")
    real_img = load_real_cifar10(64)

    # ── Composite figure ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))

    # Real CIFAR-10 on the left
    axes[0].imshow(real_img)
    axes[0].set_title("Real CIFAR-10\n(64 test images)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Generated samples at each epoch
    for ax, (label, _), img in zip(axes[1:], snapshots, gen_images):
        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

    # ── Overall title ──
    fig.suptitle(
        "DDPM Sample Quality Progression — Loss Plateaus but Visual Quality Keeps Improving",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Footer annotation ──
    fig.text(
        0.5, -0.02,
        "Epoch 1: random noise  →  Epoch 50: rough shapes emerge  →  "
        "Epoch 150: textures sharpen  →  Epoch 300: structures refine further\n"
        "Meanwhile, loss drops from 0.175 → 0.058 → 0.056 → 0.055 (nearly flat after epoch ~100)",
        ha="center", fontsize=10, style="italic", color="dimgray",
    )

    fig.tight_layout(pad=1.0)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

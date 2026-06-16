"""Plot training loss curve from training_history.txt.

Demonstrates that DDPM loss plateaus early while sample quality continues to
improve — the key evidence that loss is not a reliable quality metric for
diffusion models.

Usage
-----
  uv run python plot_loss.py                           # show interactively
  uv run python plot_loss.py --save loss_curve.png      # save to file
  uv run python plot_loss.py --smooth 0.85 --annotate    # with smooth curve + epoch annotations
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_training_history(path: str | Path = "training_history.txt") -> dict[int, float]:
    """Parse training_history.txt and return {epoch: avg_loss}.

    Handles entries like:  Epoch   1/300 | avg loss: 0.175051
    """
    history: dict[int, float] = {}
    pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|\s+avg loss:\s+([\d.]+)")

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(3))
                history[epoch] = loss

    return history


def exponential_smooth(values: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Apply exponential smoothing to a 1-D array.

    alpha: smoothing factor — higher = more smoothing, closer to raw data.
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Plot DDPM training loss curve from training_history.txt"
    )
    parser.add_argument(
        "--history", type=str, default="training_history.txt",
        help="Path to training_history.txt",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path instead of showing interactively",
    )
    parser.add_argument(
        "--smooth", type=float, default=None,
        help="Exponential smoothing factor (e.g. 0.85). Higher = closer to raw data.",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Mark epochs 50, 150, 300 to highlight loss plateau vs quality improvement",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output DPI (default: 200)",
    )
    args = parser.parse_args()

    # ── Parse ──
    history = parse_training_history(args.history)
    epochs = np.array(list(history.keys()))
    losses = np.array([history[e] for e in epochs])

    print(f"Loaded {len(history)} epochs from {args.history}")
    print(f"  Loss range: [{losses.min():.6f}, {losses.max():.6f}]")
    print(f"  Loss drop (first → last): {losses[0]:.6f} → {losses[-1]:.6f} "
          f"({(losses[0] - losses[-1]) / losses[0] * 100:.1f}% reduction)")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # ---- Main loss curve ----
    ax1.plot(epochs, losses, linewidth=0.8, alpha=0.6, color="steelblue", label="Per-epoch loss")

    if args.smooth is not None:
        smoothed = exponential_smooth(losses, alpha=args.smooth)
        ax1.plot(
            epochs, smoothed,
            linewidth=2.0, color="crimson",
            label=f"Smoothed (α={args.smooth})",
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Average MSE Loss", fontsize=12)
    ax1.set_title("DDPM Training Loss on CIFAR-10", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, epochs[-1] + 5)

    # Annotate key epochs
    if args.annotate:
        markers = [
            (50, "Epoch 50\nloss still dropping\nquickly"),
            (150, "Epoch 150\nloss nearly flat\n"),
            (300, "Epoch 300\nloss plateaued\nbut samples still improve!"),
        ]
        for ep, text in markers:
            if ep in history:
                y = history[ep]
                ax1.annotate(
                    text, xy=(ep, y),
                    xytext=(ep + 25, y + 0.005),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                    fontsize=9, color="darkred",
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.85},
                )

    # ---- Loss delta (change per epoch) ----
    loss_deltas = np.diff(losses)
    ax2.bar(epochs[1:], loss_deltas, width=0.8, color="steelblue", alpha=0.7)
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Δ Loss (per epoch)", fontsize=12)
    ax2.set_title("Loss Change per Epoch (negative = improvement)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, epochs[-1] + 5)

    # Add a horizontal line at 0 for reference
    ax2.axhline(y=0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

    # ---- Summary text box ----
    summary = (
        f"Total epochs: {len(history)}\n"
        f"Initial loss: {losses[0]:.5f}\n"
        f"Final loss:   {losses[-1]:.5f}\n"
        f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%\n"
        f"Min loss:     {losses.min():.5f} (epoch {epochs[losses.argmin()]})\n\n"
        f"Key insight: Loss plateaus after ~100 epochs,\n"
        f"but visual sample quality continues to improve\n"
        f"throughout training. DDPM loss is not a reliable\n"
        f"proxy for perceptual quality."
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.97, 0.97, summary,
        transform=ax1.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=props, family="monospace",
    )

    fig.tight_layout(pad=2.0)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

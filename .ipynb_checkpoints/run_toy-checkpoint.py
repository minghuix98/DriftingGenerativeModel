#!/usr/bin/env python3
"""Quick script to run toy 2D examples for Drifting Models.

This demonstrates the core drifting algorithm on simple 2D distributions.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import ToyDataset2D
from drifting import SimpleDriftingLoss, compute_pairwise_distances


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ToyGenerator(nn.Module):
    """Simple MLP generator for 2D data."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        input_dim: int = 2,
        output_dim: int = 2,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialize to near-identity (helpful for training)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_simple_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute simple drifting field for 2D data.

    This is the core of the Drifting algorithm:
    V = V_attraction - V_repulsion

    where samples are attracted to data and repelled from other generated samples.
    """
    # Compute distances to positive samples
    dist_pos = compute_pairwise_distances(generated, positive)

    # Attraction weights (softmax over positive samples)
    weights_pos = F.softmax(-dist_pos / temperature, dim=1)

    # Attraction: weighted sum of directions to positive samples
    diff_pos = positive.unsqueeze(0) - generated.unsqueeze(1)  # (N, M, 2)
    attraction = (weights_pos.unsqueeze(-1) * diff_pos).sum(dim=1)

    # Compute distances among generated samples
    dist_neg = compute_pairwise_distances(generated, generated)

    # Mask self-pairs
    mask = torch.eye(generated.shape[0], device=generated.device, dtype=torch.bool)
    dist_neg = dist_neg.masked_fill(mask, float("inf"))

    # Repulsion weights (softmax over other generated samples)
    weights_neg = F.softmax(-dist_neg / temperature, dim=1)

    # Repulsion: weighted sum of directions to other generated samples
    diff_neg = generated.unsqueeze(0) - generated.unsqueeze(1)
    repulsion = (weights_neg.unsqueeze(-1) * diff_neg).sum(dim=1)

    # Combined drifting field
    drift = attraction - repulsion

    return drift


def train_toy(
    dataset_name: str = "swiss_roll",
    num_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.1,
    hidden_dim: int = 256,
    num_layers: int = 4,
    device: str = "cuda",
    save_dir: str = "./toy_outputs",
    log_every: int = 100,
    visualize: bool = True,
):
    """Train a Drifting Model on 2D toy data.

    Args:
        dataset_name: Name of toy dataset (swiss_roll, checkerboard, etc.)
        num_steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        temperature: Kernel temperature for drifting field
        hidden_dim: Generator hidden dimension
        num_layers: Number of generator layers
        device: Device to use
        save_dir: Directory to save outputs
        log_every: Logging frequency
        visualize: Whether to create visualizations
    """
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    logger.info(f"Creating {dataset_name} dataset...")
    dataset = ToyDataset2D(name=dataset_name, n_samples=10000)

    # Create generator
    generator = ToyGenerator(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    # Training loop
    losses = {"total": [], "drift_norm": []}
    generated_history = []
    history_steps = []

    logger.info("Starting training...")

    for step in tqdm(range(num_steps)):
        generator.train()

        # Sample positive data
        idx = torch.randint(0, len(dataset), (batch_size,))
        positive = dataset.data[idx].to(device)

        # Sample noise and generate
        noise = torch.randn(batch_size, 2, device=device)
        generated = generator(noise)

        # Compute drifting field
        with torch.no_grad():
            drift = compute_simple_drift(generated, positive, temperature)

        # Target is generated + drift (with stop gradient on drift)
        target = generated + drift

        # Loss: minimize distance to target
        loss = F.mse_loss(generated, target.detach())

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record metrics
        losses["total"].append(loss.item())
        losses["drift_norm"].append(drift.norm(dim=-1).mean().item())

        # Logging
        if step % log_every == 0:
            logger.info(
                f"Step {step} | Loss: {loss.item():.4e} | "
                f"Drift Norm: {drift.norm(dim=-1).mean().item():.4f}"
            )

        # Save snapshots for visualization
        if step in [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]:
            with torch.no_grad():
                generator.eval()
                noise = torch.randn(1000, 2, device=device)
                samples = generator(noise).cpu()
                generated_history.append(samples)
                history_steps.append(step)

    # Final samples
    logger.info("Generating final samples...")
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1000, 2, device=device)
        final_samples = generator(noise).cpu()

    # Save results
    logger.info(f"Saving results to {save_dir}...")
    torch.save(generator.state_dict(), save_dir / "generator.pt")
    np.save(save_dir / "final_samples.npy", final_samples.numpy())
    np.save(save_dir / "losses.npy", {k: np.array(v) for k, v in losses.items()})

    # Visualization
    if visualize:
        try:
            from visualize import (
                plot_2d_samples,
                plot_training_progress,
                plot_loss_curves,
                plot_drifting_field,
            )

            logger.info("Creating visualizations...")

            # Final samples comparison
            plot_2d_samples(
                dataset.data,
                final_samples,
                title=f"Drifting Model on {dataset_name}",
                save_path=save_dir / "samples.png",
            )

            # Training progress
            plot_training_progress(
                dataset.data,
                generated_history,
                history_steps,
                save_path=save_dir / "progress.png",
            )

            # Loss curves
            plot_loss_curves(
                losses,
                save_path=save_dir / "losses.png",
            )

            # Drifting field visualization
            with torch.no_grad():
                noise = torch.randn(100, 2, device=device)
                samples = generator(noise)
                positive = dataset.data[:100].to(device)
                drift = compute_simple_drift(samples, positive, temperature)

            plot_drifting_field(
                samples.cpu(),
                drift.cpu(),
                real_data=dataset.data[:500],
                save_path=save_dir / "drift_field.png",
                scale=2.0,
            )

            logger.info(f"Visualizations saved to {save_dir}")

        except ImportError as e:
            logger.warning(f"Could not create visualizations: {e}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Final loss: {losses['total'][-1]:.4e}")
    logger.info(f"Final drift norm: {losses['drift_norm'][-1]:.4f}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 50)

    return generator, final_samples


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Model on 2D toy data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="swiss_roll",
        choices=["swiss_roll", "checkerboard", "circles", "moons", "gaussian_mixture"],
        help="Toy dataset name",
    )
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Kernel temperature")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Generator hidden dim")
    parser.add_argument("--num_layers", type=int, default=4, help="Generator layers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="./toy_outputs", help="Output directory")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    train_toy(
        dataset_name=args.dataset,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
        save_dir=args.save_dir,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()

"""Drifting Field Computation for Generative Modeling.

Implements the core drifting field computation from the paper:
"Generative Modeling via Drifting" (arXiv:2602.04770)

The drifting field V_{p,q}(x) is computed as:
    V_{p,q}(x) = V^+_p(x) - V^-_q(x)

where:
    V^+_p(x) = (1/Z_p) * E_p[k(x,y^+) * (y^+ - x)]  (attraction to data)
    V^-_q(x) = (1/Z_q) * E_q[k(x,y^-) * (y^- - x)]  (repulsion from generated)

The kernel is: k(x,y) = exp(-||x-y|| / tau)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """Compute pairwise L2 distances between two sets of vectors.

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
        squared: If True, return squared distances

    Returns:
        Distances of shape (N, M)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    x_norm = (x**2).sum(dim=-1, keepdim=True)  # (N, 1)
    y_norm = (y**2).sum(dim=-1, keepdim=True)  # (M, 1)
    dist_sq = x_norm + y_norm.T - 2 * x @ y.T  # (N, M)
    dist_sq = dist_sq.clamp(min=0)  # Numerical stability

    if squared:
        return dist_sq
    return dist_sq.sqrt()


def compute_kernel(
    distances: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute the kernel values from distances.

    k(x, y) = exp(-||x - y|| / tau)

    Args:
        distances: Pairwise distances (N, M)
        temperature: Temperature parameter tau

    Returns:
        Kernel values (N, M)
    """
    return torch.exp(-distances / temperature)


class DriftingField(nn.Module):
    """Compute the drifting field for generated samples.

    The drifting field governs the movement of samples during training,
    attracting them toward the data distribution and repelling them from
    the current generated distribution.
    """

    def __init__(
        self,
        temperatures: List[float] = [0.02, 0.05, 0.2],
        normalize_drift: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.temperatures = temperatures
        self.normalize_drift = normalize_drift
        self.eps = eps

        # Running statistics for drift normalization (Eq. 25)
        # lambda_j = sqrt(E[(1/C_j) * ||V_j||^2])
        self.register_buffer(
            "drift_scales", torch.ones(len(temperatures))
        )
        self.register_buffer("num_updates", torch.tensor(0))

    def compute_attraction(
        self,
        gen_features: torch.Tensor,
        pos_features: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute attraction toward positive (data) samples.

        V^+_p(x) = (1/Z_p) * E_p[k(x, y^+) * (y^+ - x)]

        Args:
            gen_features: Generated sample features (N_gen, D)
            pos_features: Positive sample features (N_pos, D)
            temperature: Kernel temperature

        Returns:
            Attraction vectors (N_gen, D)
        """
        # Compute distances
        distances = compute_pairwise_distances(gen_features, pos_features)

        # Compute kernel weights
        kernel = compute_kernel(distances, temperature)

        # Normalize (softmax over positive samples)
        weights = F.softmax(-distances / temperature, dim=1)  # (N_gen, N_pos)

        # Compute weighted difference
        # V^+ = sum_j w_j * (y^+_j - x)
        diff = pos_features.unsqueeze(0) - gen_features.unsqueeze(1)  # (N_gen, N_pos, D)
        attraction = (weights.unsqueeze(-1) * diff).sum(dim=1)  # (N_gen, D)

        return attraction

    def compute_repulsion(
        self,
        gen_features: torch.Tensor,
        temperature: float,
        exclude_self: bool = True,
    ) -> torch.Tensor:
        """Compute repulsion from other generated samples.

        V^-_q(x) = (1/Z_q) * E_q[k(x, y^-) * (y^- - x)]

        Args:
            gen_features: Generated sample features (N_gen, D)
            temperature: Kernel temperature
            exclude_self: Whether to exclude self-pairs

        Returns:
            Repulsion vectors (N_gen, D)
        """
        N = gen_features.shape[0]

        # Compute self-distances
        distances = compute_pairwise_distances(gen_features, gen_features)

        # Exclude self-pairs by setting diagonal to large value
        if exclude_self:
            mask = torch.eye(N, device=distances.device, dtype=torch.bool)
            distances = distances.masked_fill(mask, float("inf"))

        # Normalize (softmax over negative samples)
        weights = F.softmax(-distances / temperature, dim=1)  # (N, N)

        # Compute weighted difference
        diff = gen_features.unsqueeze(0) - gen_features.unsqueeze(1)  # (N, N, D)
        repulsion = (weights.unsqueeze(-1) * diff).sum(dim=1)  # (N, D)

        return repulsion

    def forward(
        self,
        gen_features: torch.Tensor,
        pos_features: torch.Tensor,
        neg_features: Optional[torch.Tensor] = None,
        temperature_idx: Optional[int] = None,
        update_stats: bool = False,
    ) -> torch.Tensor:
        """Compute the drifting field.

        V_{p,q}(x) = V^+_p(x) - V^-_q(x)

        Args:
            gen_features: Generated sample features (N_gen, D)
            pos_features: Positive sample features (N_pos, D)
            neg_features: Negative sample features for repulsion, if None uses gen_features
            temperature_idx: If provided, use single temperature, else average all
            update_stats: Whether to update normalization statistics

        Returns:
            Drifting field vectors (N_gen, D)
        """
        if neg_features is None:
            neg_features = gen_features

        if temperature_idx is not None:
            temperatures = [self.temperatures[temperature_idx]]
            drift_scale_indices = [temperature_idx]
        else:
            temperatures = self.temperatures
            drift_scale_indices = list(range(len(temperatures)))

        # Compute drift for each temperature
        drifts = []
        for i, (temp, scale_idx) in enumerate(zip(temperatures, drift_scale_indices)):
            # Attraction to positive samples
            attraction = self.compute_attraction(gen_features, pos_features, temp)

            # Repulsion from negative samples
            repulsion = self.compute_repulsion(neg_features, temp)

            # Combined drift
            drift = attraction - repulsion

            # Normalize drift magnitude (Eq. 25)
            if self.normalize_drift:
                if update_stats and self.training:
                    with torch.no_grad():
                        D = gen_features.shape[-1]
                        drift_norm = (drift.norm(dim=-1).mean() / (D**0.5))
                        momentum = 0.01
                        self.drift_scales[scale_idx] = (
                            1 - momentum
                        ) * self.drift_scales[scale_idx] + momentum * drift_norm
                        self.num_updates += 1

                scale = self.drift_scales[scale_idx].clamp(min=self.eps)
                drift = drift / scale

            drifts.append(drift)

        # Average over temperatures
        drift = torch.stack(drifts, dim=0).mean(dim=0)

        return drift


class MultiScaleDriftingField(nn.Module):
    """Compute drifting field in multi-scale feature space.

    This computes the drifting field at multiple feature scales and
    combines them for a more robust gradient signal.
    """

    def __init__(
        self,
        feature_names: List[str],
        temperatures: List[float] = [0.02, 0.05, 0.2],
        normalize_drift: bool = True,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.temperatures = temperatures

        # Create drifting field for each feature scale
        self.drift_fields = nn.ModuleDict({
            name: DriftingField(temperatures, normalize_drift)
            for name in feature_names
        })

        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(len(feature_names)))

    def forward(
        self,
        gen_features: Dict[str, torch.Tensor],
        pos_features: Dict[str, torch.Tensor],
        neg_features: Optional[Dict[str, torch.Tensor]] = None,
        update_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute drifting field at each feature scale.

        Args:
            gen_features: Dict of generated sample features
            pos_features: Dict of positive sample features
            neg_features: Dict of negative sample features (optional)
            update_stats: Whether to update normalization statistics

        Returns:
            Dict of drifting field vectors at each scale
        """
        if neg_features is None:
            neg_features = gen_features

        drifts = {}
        weights = F.softmax(self.scale_weights, dim=0)

        for i, name in enumerate(self.feature_names):
            if name not in gen_features:
                continue

            gen_feat = gen_features[name]
            pos_feat = pos_features[name]
            neg_feat = neg_features.get(name, gen_feat)

            # Flatten spatial dimensions if needed
            if gen_feat.dim() == 3:
                B, C, N = gen_feat.shape
                gen_feat = gen_feat.permute(0, 2, 1).reshape(-1, C)  # (B*N, C)
                pos_feat = pos_feat.permute(0, 2, 1).reshape(-1, C)
                neg_feat = neg_feat.permute(0, 2, 1).reshape(-1, C)

            drift = self.drift_fields[name](
                gen_feat, pos_feat, neg_feat, update_stats=update_stats
            )

            # Reshape back if needed
            if gen_features[name].dim() == 3:
                B = gen_features[name].shape[0]
                drift = drift.reshape(B, N, C).permute(0, 2, 1)

            drifts[name] = drift * weights[i]

        return drifts


class DriftingLoss(nn.Module):
    """Compute the drifting loss for training.

    L = E[||f_theta(epsilon) - stopgrad(f_theta(epsilon) + V)||^2]

    The loss minimizes the squared drift norm by moving generated samples
    toward targets that incorporate the drifting field.
    """

    def __init__(
        self,
        feature_encoder: nn.Module,
        temperatures: List[float] = [0.02, 0.05, 0.2],
        normalize_drift: bool = True,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.temperatures = temperatures
        self.loss_type = loss_type

        # Get feature names from encoder
        feature_names = self.feature_encoder.get_feature_names()

        # Multi-scale drifting field
        self.drift_field = MultiScaleDriftingField(
            feature_names, temperatures, normalize_drift
        )

    def forward(
        self,
        generated: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        update_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the drifting loss.

        Args:
            generated: Generated samples (B, C, H, W)
            positive: Positive (real) samples (B_pos, C, H, W)
            negative: Negative samples for repulsion (optional)
            update_stats: Whether to update normalization statistics

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract features
        gen_features = self.feature_encoder(generated, update_stats=update_stats)
        pos_features = self.feature_encoder(positive, update_stats=False)

        if negative is not None:
            neg_features = self.feature_encoder(negative, update_stats=False)
        else:
            neg_features = gen_features

        # Compute drifting field at each scale
        drifts = self.drift_field(
            gen_features, pos_features, neg_features, update_stats=update_stats
        )

        # Compute loss at each scale
        total_loss = 0.0
        metrics = {}

        for name in drifts:
            gen_feat = gen_features[name]
            drift = drifts[name]

            # Target is generated feature + drift (with stop gradient)
            target = (gen_feat + drift).detach()

            # MSE loss
            if self.loss_type == "mse":
                loss = F.mse_loss(gen_feat, target)
            elif self.loss_type == "l1":
                loss = F.l1_loss(gen_feat, target)
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(gen_feat, target)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss = total_loss + loss
            metrics[f"loss_{name}"] = loss.item()

            # Also track drift magnitude
            metrics[f"drift_norm_{name}"] = drift.norm(dim=-1).mean().item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


class SimpleDriftingLoss(nn.Module):
    """Simplified drifting loss operating directly in pixel/latent space.

    This is useful for toy examples or when feature extraction is not needed.
    """

    def __init__(
        self,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        generated: torch.Tensor,
        positive: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute simple drifting loss.

        Args:
            generated: Generated samples (B, D) or (B, C, H, W)
            positive: Positive samples (B_pos, D) or (B_pos, C, H, W)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Flatten if needed
        gen_flat = generated.flatten(1)  # (B, D)
        pos_flat = positive.flatten(1)  # (B_pos, D)

        # Compute attraction
        dist_pos = compute_pairwise_distances(gen_flat, pos_flat)
        weights_pos = F.softmax(-dist_pos / self.temperature, dim=1)
        diff_pos = pos_flat.unsqueeze(0) - gen_flat.unsqueeze(1)
        attraction = (weights_pos.unsqueeze(-1) * diff_pos).sum(dim=1)

        # Compute repulsion
        dist_neg = compute_pairwise_distances(gen_flat, gen_flat)
        mask = torch.eye(gen_flat.shape[0], device=dist_neg.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(mask, float("inf"))
        weights_neg = F.softmax(-dist_neg / self.temperature, dim=1)
        diff_neg = gen_flat.unsqueeze(0) - gen_flat.unsqueeze(1)
        repulsion = (weights_neg.unsqueeze(-1) * diff_neg).sum(dim=1)

        # Drifting field
        drift = attraction - repulsion

        # Target and loss
        target = (gen_flat + drift).detach()
        loss = F.mse_loss(gen_flat, target)

        metrics = {
            "loss": loss.item(),
            "drift_norm": drift.norm(dim=-1).mean().item(),
            "attraction_norm": attraction.norm(dim=-1).mean().item(),
            "repulsion_norm": repulsion.norm(dim=-1).mean().item(),
        }

        return loss, metrics

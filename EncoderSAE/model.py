"""SAE Model Architecture: Encoder + Decoder + TopK Activation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderSAE(nn.Module):
    """
    Sparse Autoencoder for Encoder-only models.

    Architecture:
    - Encoder: Linear layer that expands the input dimension
    - TopK Activation: Sparsity constraint via top-k selection
    - Decoder: Linear layer that reconstructs the original dimension
    """

    def __init__(self, input_dim: int, expansion_factor: int = 32, sparsity: int = 64):
        """
        Initialize the SAE.

        Args:
            input_dim: Dimension of input activations (e.g., 1024 for E5-large)
            expansion_factor: Expansion factor for the encoder (dict_size = input_dim * expansion_factor)
            sparsity: Number of top features to keep (k-top sparsity)
        """
        super().__init__()
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.sparsity = sparsity
        self.dict_size = input_dim * expansion_factor

        # Encoder: projects input to expanded dictionary space
        self.encoder = nn.Linear(input_dim, self.dict_size, bias=False)

        # Decoder: reconstructs input from sparse features
        self.decoder = nn.Linear(self.dict_size, input_dim, bias=False)

        # Initialize decoder weights to be transpose of encoder (tied weights initialization)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations of shape (batch_size, input_dim)

        Returns:
            tuple of (reconstructed, features, l0_norm, topk_threshold, raw_features):
            - reconstructed: Reconstructed input (batch_size, input_dim)
            - features: Sparse feature activations (batch_size, dict_size)
            - l0_norm: Average number of active features per sample
            - topk_threshold: Per-sample k-th largest activation value (batch_size,).
              Used to build a differentiable approximation of "in top-k" without
              materializing a huge boolean mask.
            - raw_features: Raw feature activations before top-k (batch_size, dict_size)
        """
        # Encode: project to dictionary space
        features = self.encoder(x)  # (batch_size, dict_size)

        # Apply ReLU activation
        raw_features = F.relu(features)  # Store before top-k for auxiliary loss
        features = raw_features

        # TopK sparsity: keep only top-k features per sample
        topk_threshold = None
        if self.sparsity > 0 and self.sparsity < self.dict_size:
            topk_values, topk_indices = torch.topk(features, k=self.sparsity, dim=1)
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(1, topk_indices, topk_values)

            features = sparse_features
            # k-th largest value per sample (detach: treat selection boundary as constant)
            topk_threshold = topk_values[:, -1].detach()
        else:
            # If no top-k, use 0 as boundary (approx "active if >0")
            topk_threshold = torch.zeros(
                features.shape[0], device=features.device, dtype=features.dtype
            )

        # Decode: reconstruct input
        reconstructed = self.decoder(features)

        # Compute L0 norm (number of active features)
        l0_norm = (features > 0).float().sum(dim=1).mean()

        return reconstructed, features, l0_norm, topk_threshold, raw_features

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        topk_threshold: torch.Tensor,
        raw_features: torch.Tensor,
        l1_coeff: float = 0.0,
        aux_loss_coeff: float = 0.0,
        aux_loss_target: float = 0.01,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute reconstruction loss and auxiliary metrics.

        Args:
            x: Original input (batch_size, input_dim)
            reconstructed: Reconstructed input (batch_size, input_dim)
            features: Sparse features (batch_size, dict_size)
            topk_threshold: Per-sample k-th largest activation value (batch_size,)
            raw_features: Raw feature activations before top-k (batch_size, dict_size)
            l1_coeff: L1 regularization coefficient (default: 0.0, not used in top-k)
            aux_loss_coeff: Coefficient for auxiliary loss that encourages feature usage (default: 0.0)
            aux_loss_target: Target fraction of samples where each feature should activate (default: 0.01)
                This is a fraction between 0 and 1 (e.g., 0.01 = 1% of samples)

        Returns:
            tuple of (loss, metrics_dict):
            - loss: Total loss (MSE + L1 + auxiliary loss)
            - metrics_dict: Dictionary with fvu, dead_features, etc.
        """
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstructed, x, reduction="mean")

        # L1 regularization (optional, typically not used with top-k)
        l1_loss = features.abs().mean() * l1_coeff

        # Auxiliary loss: encourage features to activate at least occasionally
        # Use raw_features (before top-k) so gradients flow to all features
        # Always compute aux_loss from the computation graph (even if coeff is 0) to ensure consistency
        # Compute fraction of samples where each feature is selected in top-k
        # raw_features shape: (batch_size, dict_size)
        # Use sigmoid as differentiable approximation of step function
        # High temperature makes it approximate (raw_features > topk_threshold) closely
        # Since activations are small (std ~ 0.03), we need a high temperature.
        temperature = 10000.0

        # Memory-efficient chunked computation of feature_activation_fraction
        # to avoid OOM with large batches and dict_sizes (e.g., 131k * 32k)
        batch_size = int(raw_features.shape[0])
        dict_size = int(raw_features.shape[1])
        # Choose chunk size to keep temporary tensors manageable.
        # We approximate memory as 2 * chunk * dict_size * element_size
        # (one for the scaled tensor and one for the sigmoid output).
        target_bytes = 1_000_000_000  # ~1GB working set
        elem_size = raw_features.element_size()
        chunk_size = max(
            1, min(batch_size, int(target_bytes / max(1, 2 * dict_size * elem_size)))
        )

        if batch_size <= chunk_size:
            thr = topk_threshold.float().unsqueeze(1)
            # Use a slightly more robust calculation: sigmoid( (raw/thr - 1) * large_temp )
            # or just high temperature on the difference.
            activation_indicator = torch.sigmoid(
                (raw_features.float() - thr) * temperature
            )
            feature_activation_fraction = activation_indicator.mean(dim=0)
        else:
            # IMPORTANT: do NOT use in-place accumulation into a non-grad tensor here,
            # otherwise gradients won't flow to raw_features and aux loss becomes a no-op.
            indicator_sum = None
            total = 0
            for i in range(0, batch_size, chunk_size):
                chunk = raw_features[i : i + chunk_size].float()
                thr = topk_threshold[i : i + chunk_size].float().unsqueeze(1)
                chunk_sum = torch.sigmoid((chunk - thr) * temperature).sum(dim=0)
                indicator_sum = (
                    chunk_sum if indicator_sum is None else (indicator_sum + chunk_sum)
                )
                total += int(chunk.shape[0])
            feature_activation_fraction = indicator_sum / total

        # Penalize features that activate in fewer than target fraction of samples
        # aux_loss_target is the target fraction (e.g., 0.01 = 1% of samples)
        activation_deficit = torch.clamp(
            aux_loss_target - feature_activation_fraction, min=0.0
        )
        aux_loss = (activation_deficit.pow(2).mean()) * aux_loss_coeff

        # DEBUG: log actual usage to see if sigmoid is still blurry
        # (Only if coeff > 0 to avoid noise in baseline runs)
        if (
            aux_loss_coeff > 0
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        ):
            if torch.rand(1).item() < 0.01:  # 1% of batches
                print(
                    f"DEBUG usage: target={aux_loss_target}, avg_usage={feature_activation_fraction.mean().item():.4f}, deficit={activation_deficit.mean().item():.4f}, loss={aux_loss.item():.8f}"
                )

        total_loss = mse_loss + l1_loss + aux_loss

        # Compute metrics
        # FVU: Fraction of Variance Unexplained
        residual = x - reconstructed
        total_variance = (x - x.mean(dim=0, keepdim=True)).pow(2).mean()
        explained_variance = total_variance - residual.pow(2).mean()
        fvu = 1.0 - (explained_variance / (total_variance + 1e-8))

        # Dead feature fraction (features that never fired in this batch), in [0, 1]
        dead_features = (features.abs().max(dim=0)[0] == 0).float().mean()

        metrics = {
            # Log the true optimized objective as "loss" (includes aux)
            "loss": total_loss.item(),
            # Also log pure reconstruction separately for easy comparison across aux settings
            "mse_loss": mse_loss.item(),
            "fvu": fvu.item(),
            "dead_features": dead_features.item(),
            "l1_loss": l1_loss.item() if l1_coeff > 0 else 0.0,
            "aux_loss": aux_loss.item() if aux_loss_coeff > 0 else 0.0,
        }

        return total_loss, metrics

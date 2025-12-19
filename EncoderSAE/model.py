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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations of shape (batch_size, input_dim)

        Returns:
            tuple of (reconstructed, features, l0_norm):
            - reconstructed: Reconstructed input (batch_size, input_dim)
            - features: Sparse feature activations (batch_size, dict_size)
            - l0_norm: Average number of active features per sample
        """
        # Encode: project to dictionary space
        features = self.encoder(x)  # (batch_size, dict_size)

        # Apply ReLU activation
        features = F.relu(features)

        # TopK sparsity: keep only top-k features per sample
        if self.sparsity > 0 and self.sparsity < self.dict_size:
            topk_values, topk_indices = torch.topk(features, k=self.sparsity, dim=1)
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(1, topk_indices, topk_values)
            features = sparse_features

        # Decode: reconstruct input
        reconstructed = self.decoder(features)

        # Compute L0 norm (number of active features)
        l0_norm = (features > 0).float().sum(dim=1).mean()

        return reconstructed, features, l0_norm

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        l1_coeff: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute reconstruction loss and auxiliary metrics.

        Args:
            x: Original input (batch_size, input_dim)
            reconstructed: Reconstructed input (batch_size, input_dim)
            features: Sparse features (batch_size, dict_size)
            l1_coeff: L1 regularization coefficient (default: 0.0, not used in top-k)

        Returns:
            tuple of (loss, metrics_dict):
            - loss: Total loss (MSE reconstruction loss)
            - metrics_dict: Dictionary with fvu, dead_feature_pct, etc.
        """
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstructed, x, reduction="mean")

        # L1 regularization (optional, typically not used with top-k)
        l1_loss = features.abs().mean() * l1_coeff

        total_loss = mse_loss + l1_loss

        # Compute metrics
        # FVU: Fraction of Variance Unexplained
        residual = x - reconstructed
        total_variance = (x - x.mean(dim=0, keepdim=True)).pow(2).mean()
        explained_variance = total_variance - residual.pow(2).mean()
        fvu = 1.0 - (explained_variance / (total_variance + 1e-8))

        # Dead feature percentage (features that never fired in this batch)
        dead_features = (features.abs().max(dim=0)[0] == 0).float().mean()
        dead_feature_pct = dead_features.item() * 100.0

        metrics = {
            "loss": mse_loss.item(),
            "fvu": fvu.item(),
            "dead_feature_pct": dead_feature_pct,
            "l1_loss": l1_loss.item() if l1_coeff > 0 else 0.0,
        }

        return total_loss, metrics

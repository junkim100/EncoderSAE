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
            tuple of (reconstructed, features, l0_norm, topk_mask, raw_features):
            - reconstructed: Reconstructed input (batch_size, input_dim)
            - features: Sparse feature activations (batch_size, dict_size)
            - l0_norm: Average number of active features per sample
            - topk_mask: Binary mask (batch_size, dict_size) indicating which features were in top-k
            - raw_features: Raw feature activations before top-k (batch_size, dict_size)
        """
        # Encode: project to dictionary space
        features = self.encoder(x)  # (batch_size, dict_size)

        # Apply ReLU activation
        raw_features = F.relu(features)  # Store before top-k for auxiliary loss
        features = raw_features

        # TopK sparsity: keep only top-k features per sample
        topk_mask = None
        if self.sparsity > 0 and self.sparsity < self.dict_size:
            topk_values, topk_indices = torch.topk(features, k=self.sparsity, dim=1)
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(1, topk_indices, topk_values)

            # Create binary mask indicating which features were selected in top-k
            topk_mask = torch.zeros_like(features, dtype=torch.bool)
            topk_mask.scatter_(
                1, topk_indices, torch.ones_like(topk_indices, dtype=torch.bool)
            )

            features = sparse_features
        else:
            # If no top-k, all non-zero features are "selected"
            topk_mask = features > 0

        # Decode: reconstruct input
        reconstructed = self.decoder(features)

        # Compute L0 norm (number of active features)
        l0_norm = (features > 0).float().sum(dim=1).mean()

        return reconstructed, features, l0_norm, topk_mask, raw_features

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        topk_mask: torch.Tensor,
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
            topk_mask: Binary mask (batch_size, dict_size) indicating which features were in top-k
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
        if aux_loss_coeff > 0:
            # Compute fraction of samples where each feature activates (activation > 0)
            # raw_features shape: (batch_size, dict_size)
            # Use sigmoid as differentiable approximation of step function
            # High temperature (100) makes it approximate (raw_features > 0) closely
            # This ensures gradients flow through the computation
            temperature = 100.0
            activation_indicator = torch.sigmoid(
                raw_features * temperature
            )  # Differentiable approximation
            feature_activation_fraction = activation_indicator.mean(
                dim=0
            )  # (dict_size,)

            # Penalize features that activate in fewer than target fraction of samples
            # aux_loss_target is the target fraction (e.g., 0.01 = 1% of samples)
            activation_deficit = torch.clamp(
                aux_loss_target - feature_activation_fraction, min=0.0
            )
            aux_loss = (activation_deficit.pow(2).mean()) * aux_loss_coeff
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

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
            "loss": mse_loss.item(),
            "fvu": fvu.item(),
            "dead_features": dead_features.item(),
            "l1_loss": l1_loss.item() if l1_coeff > 0 else 0.0,
            "aux_loss": aux_loss.item() if aux_loss_coeff > 0 else 0.0,
        }

        return total_loss, metrics

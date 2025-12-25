"""SAE Runtime for loading and running SAE models."""

import torch
import json
import re
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EncoderSAE.model import EncoderSAE


class SAERuntime:
    """Runtime for loading and running SAE models."""

    @staticmethod
    def _parse_hparams_from_ckpt_name(name: str) -> dict:
        """
        Best-effort parse of SAE hyperparams from checkpoint folder naming convention.

        Expected pattern used by this repo's sweeps:
          exp{EXPANSION}_k{SPARSITY}_lr...
        Example:
          exp32_k1024_lr3e-04_aux1e+00_tgt2e-02
        """
        out: dict = {}
        m_exp = re.search(r"(?:^|_)exp(?P<exp>\d+)(?:_|$)", name)
        m_k = re.search(r"(?:^|_)k(?P<k>\d+)(?:_|$)", name)
        if m_exp:
            out["expansion_factor_from_name"] = int(m_exp.group("exp"))
        if m_k:
            out["sparsity_from_name"] = int(m_k.group("k"))
        return out

    def __init__(self, ckpt_dir: str):
        """
        Initialize SAE Runtime.

        Args:
            ckpt_dir: Path to checkpoint directory containing final_model.pt
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metadata = None
        self._load_model()

    def _load_model(self):
        """Load SAE model from checkpoint."""
        # Find checkpoint file
        potential_files = list(self.ckpt_dir.glob("*.pt")) + list(
            self.ckpt_dir.glob("*.bin")
        )
        if not potential_files:
            raise FileNotFoundError(f"No .pt or .bin file found in {self.ckpt_dir}")

        # Prefer final_model.pt
        sae_path = None
        for p in potential_files:
            if "final_model" in p.name or "sae" in p.name:
                sae_path = p
                break
        if sae_path is None:
            sae_path = potential_files[0]

        print(f"Loading SAE from {sae_path}")
        checkpoint = torch.load(sae_path, map_location=self.device)

        # Extract state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "encoder.weight" in checkpoint:
            state_dict = checkpoint
        else:
            raise ValueError(f"Could not parse checkpoint format from {sae_path}")

        # Infer model dimensions from checkpoint
        encoder_weight = state_dict["encoder.weight"]
        input_dim = encoder_weight.shape[1]
        dict_size = encoder_weight.shape[0]
        expansion_factor = dict_size // input_dim

        # -----------------------------
        # Infer sparsity (top-k) correctly
        # -----------------------------
        # CRITICAL: In this repo, `final_model.pt` is saved as a *plain state_dict*
        # (see EncoderSAE/main.py). That means it does NOT contain a "sparsity" key,
        # so defaulting to 64 here silently constructs the wrong SAE at eval time.
        sparsity = None
        if isinstance(checkpoint, dict):
            # Support both "full checkpoint" and "state_dict-only" formats.
            for key in ("sparsity", "k", "top_k", "topk"):
                if key in checkpoint:
                    try:
                        sparsity = int(checkpoint[key])
                        break
                    except Exception:
                        pass

        # Fallback: parse from checkpoint folder name (exp{E}_k{K}_...)
        if sparsity is None:
            parsed = self._parse_hparams_from_ckpt_name(self.ckpt_dir.name)
            sparsity = parsed.get("sparsity_from_name")
            # Cross-check expansion factor if present in name (for sanity only)
            exp_from_name = parsed.get("expansion_factor_from_name")
            if exp_from_name is not None and exp_from_name != expansion_factor:
                print(
                    f"Warning: expansion_factor inferred from weights ({expansion_factor}) "
                    f"differs from folder name ({exp_from_name}) for {self.ckpt_dir.name}"
                )

        if sparsity is None:
            # Last resort
            sparsity = 64
            print(
                f"WARNING: Could not infer SAE sparsity (top-k) from checkpoint. "
                f"Defaulting to {sparsity}. This can severely degrade eval results. "
                f"Checkpoint dir: {self.ckpt_dir}"
            )

        # Clamp to valid range (avoid accidental 'no-topk' mode when k >= dict_size)
        if sparsity <= 0:
            print(
                f"WARNING: Parsed sparsity={sparsity} is not positive; defaulting to 64."
            )
            sparsity = 64
        if sparsity >= dict_size:
            print(
                f"WARNING: Parsed sparsity={sparsity} >= dict_size={dict_size}. "
                f"Clamping to dict_size-1 to preserve top-k behavior."
            )
            sparsity = dict_size - 1

        # Create and load model
        self.model = EncoderSAE(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            sparsity=sparsity,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Store metadata
        self.metadata = {
            "input_dim": input_dim,
            "dict_size": dict_size,
            "expansion_factor": expansion_factor,
            "sparsity": sparsity,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        disable_mask: bool = False,
    ):
        """
        Forward pass through SAE with optional masking.

        Args:
            x: Input embeddings [N, D]
            mask: Boolean mask of shape [dict_size] where True means disable (zero out) feature.
                Features where mask[i] = True fired in >= threshold% of samples for a language
                and will be zeroed out to create language-agnostic embeddings.
                Example: If mask_threshold=0.99, features that fire in >=99% of samples are marked True.
            disable_mask: If True, ignore mask even if provided

        Returns:
            x_hat_orig: Original reconstruction without masking [N, D]
            z: Sparse features [N, dict_size] (unmasked, original activations)
            x_hat_masked: Reconstruction after masking language-specific features [N, D]
        """
        x = x.to(self.device)

        with torch.no_grad():
            # EncoderSAE.forward returns: (x_hat, z, l0, topk_thr, raw_features)
            x_hat_orig, z, _, _, _ = self.model(x)

            if mask is not None and not disable_mask:
                mask = mask.to(self.device)
                # Apply mask: zero out features where mask[i] = True
                # This disables language-specific features (neurons) that fired in >= threshold% of samples
                # mask[i] = True means feature i should be disabled (zeroed out)
                # Use explicit indexing to ensure correct column selection
                z_masked = z.clone()
                # Zero out features where mask is True: multiply by inverse of mask
                # This correctly zeros out columns (features) for all samples
                z_masked = z_masked * (~mask).float().unsqueeze(0)  # [N, dict_size] * [1, dict_size]
                # Decode masked features to get language-agnostic embeddings
                x_hat_masked = self.model.decoder(z_masked)
            else:
                x_hat_masked = x_hat_orig

        return x_hat_orig, z, x_hat_masked

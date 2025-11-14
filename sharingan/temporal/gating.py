"""Cross-Frame Gating Network for temporal influence modeling."""

import numpy as np
import torch
import torch.nn as nn


class CrossFrameGatingNetwork(nn.Module):
    """Lightweight MLP-based temporal gating."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        """
        Initialize Cross-Frame Gating Network.

        Args:
            feature_dim: Feature dimension
            hidden_dim: Hidden layer dimension (default: 128)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Compact MLP for gating (< 1M parameters)
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # Concatenate current and previous
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Sigmoid()  # Gate values between 0 and 1
        )

        # Temporal influence weights
        self.influence_network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # Influence values between -1 and 1
        )

        self._verify_parameter_count()

    def _verify_parameter_count(self) -> None:
        """Verify parameter count is below 1M."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1_000_000:
            print(f"Warning: Parameter count ({total_params:,}) exceeds 1M limit")

    def forward(self, x_t: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute gated combination of current and previous frame.

        Args:
            x_t: Current frame features of shape (B, D) or (D,)
            x_prev: Previous frame features of shape (B, D) or (D,)

        Returns:
            Gated features of shape (B, D) or (D,)
        """
        # Handle single sample case
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Concatenate current and previous features
        combined = torch.cat([x_t, x_prev], dim=-1)

        # Compute gate values
        gate = self.gate_network(combined)  # (B, D)

        # Compute temporal influence
        influence = self.influence_network(combined)  # (B, D)

        # Apply gating: blend current frame with influenced previous frame
        output = gate * x_t + (1 - gate) * (x_prev + influence)

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process entire sequence with gating.

        Args:
            x: Input sequence of shape (T, D) or (B, T, D)

        Returns:
            Gated sequence of same shape
        """
        # Handle both (T, D) and (B, T, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = x.shape
        output = torch.zeros_like(x)

        # First frame passes through unchanged
        output[:, 0] = x[:, 0]

        # Process remaining frames with gating
        for t in range(1, T):
            output[:, t] = self.forward(x[:, t], output[:, t - 1])

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_numpy(self, x_t: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """
        Apply gating to numpy arrays.

        Args:
            x_t: Current frame features
            x_prev: Previous frame features

        Returns:
            Gated features as numpy array
        """
        # Convert to tensors
        x_t_tensor = torch.from_numpy(x_t).float()
        x_prev_tensor = torch.from_numpy(x_prev).float()

        # Apply forward pass
        with torch.no_grad():
            output_tensor = self.forward(x_t_tensor, x_prev_tensor)

        return output_tensor.numpy()

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """String representation."""
        param_count = self.get_parameter_count()
        return (f"CrossFrameGatingNetwork(feature_dim={self.feature_dim}, "
                f"hidden_dim={self.hidden_dim}, params={param_count:,})")

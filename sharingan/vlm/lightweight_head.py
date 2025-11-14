"""Lightweight projection head for VLM features."""

import numpy as np
import torch
import torch.nn as nn


class LightweightVLMHead(nn.Module):
    """Lightweight projection head for VLM features."""

    def __init__(self, input_dim: int, output_dim: int = 256):
        """
        Initialize projection head.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension (default: 256)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Simple linear projection with optional bottleneck
        if input_dim > output_dim * 2:
            # Use bottleneck architecture for large dimension reduction
            hidden_dim = (input_dim + output_dim) // 2
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # Direct projection for smaller reductions
            self.projection = nn.Linear(input_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to lower dimension.

        Args:
            embeddings: Input embeddings of shape (*, input_dim)

        Returns:
            Projected embeddings of shape (*, output_dim)
        """
        return self.projection(embeddings)

    def forward_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project numpy embeddings.

        Args:
            embeddings: Input embeddings as numpy array

        Returns:
            Projected embeddings as numpy array
        """
        # Convert to tensor
        input_tensor = torch.from_numpy(embeddings).float()
        original_shape = input_tensor.shape

        # Flatten if needed
        if len(original_shape) > 2:
            input_tensor = input_tensor.reshape(-1, self.input_dim)

        # Project
        with torch.no_grad():
            output_tensor = self.forward(input_tensor)

        # Reshape back
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.output_dim]
            output_tensor = output_tensor.reshape(new_shape)

        return output_tensor.numpy()

    def __repr__(self) -> str:
        """String representation."""
        return f"LightweightVLMHead(input_dim={self.input_dim}, output_dim={self.output_dim})"

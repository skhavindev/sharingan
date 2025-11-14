"""Temporal Dilated Attention (TDA) module."""

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDilatedAttention(nn.Module):
    """Multi-scale temporal attention with dilated intervals."""

    def __init__(self, feature_dim: int, dilations: List[int] = None):
        """
        Initialize Temporal Dilated Attention.

        Args:
            feature_dim: Feature dimension
            dilations: List of dilation intervals (default: [1, 4, 8, 16])
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.dilations = dilations if dilations is not None else [1, 4, 8, 16]
        self.num_dilations = len(self.dilations)

        # Query, Key, Value projections for each dilation level
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # Multi-head attention parameters
        self.num_heads = 4
        self.head_dim = feature_dim // self.num_heads

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # Dilation-specific weights
        self.dilation_weights = nn.Parameter(torch.ones(self.num_dilations) / self.num_dilations)

        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor, history: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply dilated attention over temporal history.

        Args:
            x: Current frame features of shape (B, D) or (D,)
            history: List of historical frame features (most recent first)

        Returns:
            Temporally attended features
        """
        # Handle single sample case
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, D = x.shape

        # If no history, return input
        if history is None or len(history) == 0:
            return x.squeeze(0) if squeeze_output else x

        # Ensure history tensors have batch dimension
        history = [h.unsqueeze(0) if h.dim() == 1 else h for h in history]

        # Compute query from current frame
        query = self.query_proj(x)  # (B, D)
        query = query.reshape(B, self.num_heads, self.head_dim)  # (B, H, D/H)

        # Collect attended features from each dilation level
        attended_features = []

        for dilation_idx, dilation in enumerate(self.dilations):
            # Collect frames at this dilation interval
            dilated_frames = []
            for i in range(0, len(history), dilation):
                if i < len(history):
                    dilated_frames.append(history[i])

            if len(dilated_frames) == 0:
                continue

            # Stack dilated frames
            dilated_stack = torch.cat(dilated_frames, dim=0)  # (T', D)

            # Compute keys and values
            keys = self.key_proj(dilated_stack)  # (T', D)
            values = self.value_proj(dilated_stack)  # (T', D)

            # Reshape for multi-head attention
            T_prime = keys.shape[0]
            keys = keys.reshape(T_prime, self.num_heads, self.head_dim)  # (T', H, D/H)
            values = values.reshape(T_prime, self.num_heads, self.head_dim)  # (T', H, D/H)

            # Compute attention scores
            # query: (B, H, D/H), keys: (T', H, D/H)
            scores = torch.einsum('bhd,thd->bht', query, keys) / np.sqrt(self.head_dim)  # (B, H, T')
            attn_weights = F.softmax(scores, dim=-1)  # (B, H, T')

            # Apply attention to values
            # attn_weights: (B, H, T'), values: (T', H, D/H)
            attended = torch.einsum('bht,thd->bhd', attn_weights, values)  # (B, H, D/H)
            attended = attended.reshape(B, D)  # (B, D)

            attended_features.append(attended)

        # Combine features from different dilations
        if len(attended_features) > 0:
            # Normalize dilation weights
            dilation_weights = F.softmax(self.dilation_weights[:len(attended_features)], dim=0)

            # Weighted combination
            combined = sum(w * feat for w, feat in zip(dilation_weights, attended_features))

            # Residual connection and layer norm
            output = self.layer_norm(x + self.output_proj(combined))
        else:
            output = x

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_sequence(self, x: torch.Tensor, max_history: int = 32) -> torch.Tensor:
        """
        Process entire sequence with dilated attention.

        Args:
            x: Input sequence of shape (T, D) or (B, T, D)
            max_history: Maximum history length to maintain

        Returns:
            Attended sequence of same shape
        """
        # Handle both (T, D) and (B, T, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = x.shape
        output = torch.zeros_like(x)

        # Process each frame with its history
        for t in range(T):
            # Collect history (frames before current)
            history_start = max(0, t - max_history)
            history = [x[:, i] for i in range(t - 1, history_start - 1, -1)]

            # Apply dilated attention
            output[:, t] = self.forward(x[:, t], history)

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_numpy(self, x: np.ndarray, history: List[np.ndarray] = None) -> np.ndarray:
        """
        Apply TDA to numpy arrays.

        Args:
            x: Current frame features
            history: List of historical frame features

        Returns:
            Attended features as numpy array
        """
        # Convert to tensors
        x_tensor = torch.from_numpy(x).float()
        history_tensors = [torch.from_numpy(h).float() for h in history] if history else None

        # Apply forward pass
        with torch.no_grad():
            output_tensor = self.forward(x_tensor, history_tensors)

        return output_tensor.numpy()

    def __repr__(self) -> str:
        """String representation."""
        return (f"TemporalDilatedAttention(feature_dim={self.feature_dim}, "
                f"dilations={self.dilations}, num_heads={self.num_heads})")

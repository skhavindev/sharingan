"""Temporal Memory Tokens for streaming video understanding."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMemoryTokens(nn.Module):
    """Streaming temporal memory using learned tokens."""

    def __init__(self, num_tokens: int = 8, token_dim: int = 256):
        """
        Initialize Temporal Memory Tokens.

        Args:
            num_tokens: Number of memory tokens (4-16)
            token_dim: Dimension of each token
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Initialize learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)

        # Update mechanism: cross-attention between tokens and new frame
        self.query_proj = nn.Linear(token_dim, token_dim)
        self.key_proj = nn.Linear(token_dim, token_dim)
        self.value_proj = nn.Linear(token_dim, token_dim)

        # Multi-head attention
        self.num_heads = 4
        self.head_dim = token_dim // self.num_heads

        # Output projection
        self.output_proj = nn.Linear(token_dim, token_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(token_dim)

        # Gating mechanism for update
        self.update_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid()
        )

        # Frame counter for tracking history length
        self.frame_count = 0

    def update(self, frame_embedding: torch.Tensor) -> None:
        """
        Update memory tokens with new frame.

        Args:
            frame_embedding: Frame embedding of shape (D,) or (1, D)
        """
        # Ensure batch dimension
        if frame_embedding.dim() == 1:
            frame_embedding = frame_embedding.unsqueeze(0)  # (1, D)

        # Expand to match token dimension if needed
        if frame_embedding.shape[-1] != self.token_dim:
            # Simple projection (in practice, use a learned projection)
            frame_embedding = F.adaptive_avg_pool1d(
                frame_embedding.unsqueeze(1),
                self.token_dim
            ).squeeze(1)

        # Cross-attention: tokens attend to new frame
        queries = self.query_proj(self.memory_tokens)  # (N, D)
        keys = self.key_proj(frame_embedding)  # (1, D)
        values = self.value_proj(frame_embedding)  # (1, D)

        # Reshape for multi-head attention
        queries = queries.reshape(self.num_tokens, self.num_heads, self.head_dim)  # (N, H, D/H)
        keys = keys.reshape(1, self.num_heads, self.head_dim)  # (1, H, D/H)
        values = values.reshape(1, self.num_heads, self.head_dim)  # (1, H, D/H)

        # Compute attention scores
        scores = torch.einsum('nhd,mhd->nhm', queries, keys) / np.sqrt(self.head_dim)  # (N, H, 1)
        attn_weights = F.softmax(scores, dim=-1)  # (N, H, 1)

        # Apply attention to values
        attended = torch.einsum('nhm,mhd->nhd', attn_weights, values)  # (N, H, D/H)
        attended = attended.reshape(self.num_tokens, self.token_dim)  # (N, D)

        # Project attended features
        updates = self.output_proj(attended)  # (N, D)

        # Gating mechanism: decide how much to update each token
        concat = torch.cat([self.memory_tokens, updates], dim=-1)  # (N, 2D)
        gates = self.update_gate(concat)  # (N, D)

        # Update tokens with gating
        new_tokens = gates * updates + (1 - gates) * self.memory_tokens

        # Apply layer norm
        new_tokens = self.layer_norm(new_tokens)

        # Update memory tokens (in-place)
        self.memory_tokens.data = new_tokens.data

        # Increment frame counter
        self.frame_count += 1

    def get_context(self) -> torch.Tensor:
        """
        Retrieve current temporal context.

        Returns:
            Memory tokens of shape (N, D)
        """
        return self.memory_tokens.clone()

    def get_pooled_context(self) -> torch.Tensor:
        """
        Get pooled context as single vector.

        Returns:
            Pooled context of shape (D,)
        """
        # Average pooling over tokens
        return torch.mean(self.memory_tokens, dim=0)

    def reset(self) -> None:
        """Reset memory for new video."""
        # Reinitialize tokens
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        self.frame_count = 0

    def update_numpy(self, frame_embedding: np.ndarray) -> None:
        """
        Update with numpy array.

        Args:
            frame_embedding: Frame embedding as numpy array
        """
        frame_tensor = torch.from_numpy(frame_embedding).float()
        self.update(frame_tensor)

    def get_context_numpy(self) -> np.ndarray:
        """
        Get context as numpy array.

        Returns:
            Memory tokens as numpy array
        """
        return self.get_context().detach().numpy()

    def get_history_length(self) -> int:
        """
        Get number of frames processed.

        Returns:
            Frame count
        """
        return self.frame_count

    def __repr__(self) -> str:
        """String representation."""
        return (f"TemporalMemoryTokens(num_tokens={self.num_tokens}, "
                f"token_dim={self.token_dim}, frames_processed={self.frame_count})")

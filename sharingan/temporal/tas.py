"""Temporal Attention Shift (TAS) module."""

import numpy as np
import torch
import torch.nn as nn


class TemporalAttentionShift(nn.Module):
    """Learnable attention-driven temporal shift mechanism."""

    def __init__(self, channels: int, shift_ratio: float = 0.125):
        """
        Initialize Temporal Attention Shift.

        Args:
            channels: Number of feature channels
            shift_ratio: Fraction of channels to shift (default: 0.125)
        """
        super().__init__()
        self.channels = channels
        self.shift_ratio = shift_ratio
        self.shift_channels = int(channels * shift_ratio)

        # Learnable attention weights for adaptive shifting
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # Channel split: forward shift, backward shift, no shift
        self.forward_channels = self.shift_channels
        self.backward_channels = self.shift_channels
        self.static_channels = channels - 2 * self.shift_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention shift.

        Args:
            x: Input tensor of shape (T, C, H, W) or (B, T, C, H, W)

        Returns:
            Shifted features of same shape
        """
        # Handle both (T, C, H, W) and (B, T, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, C, H, W = x.shape

        # Compute attention weights
        # Reshape to (B*T, C, H, W) for attention computation
        x_reshaped = x.reshape(B * T, C, H, W)
        attn_weights = self.attention(x_reshaped)  # (B*T, C, 1, 1)
        attn_weights = attn_weights.reshape(B, T, C, 1, 1)

        # Apply attention-weighted shifting
        output = torch.zeros_like(x)

        for t in range(T):
            # Get current frame
            current = x[:, t]  # (B, C, H, W)
            current_attn = attn_weights[:, t]  # (B, C, 1, 1)

            # Split channels
            forward_part = current[:, :self.forward_channels]
            backward_part = current[:, self.forward_channels:self.forward_channels + self.backward_channels]
            static_part = current[:, self.forward_channels + self.backward_channels:]

            # Forward shift (from t-1)
            if t > 0:
                prev = x[:, t - 1, :self.forward_channels]
                prev_attn = attn_weights[:, t - 1, :self.forward_channels]
                forward_part = forward_part * current_attn[:, :self.forward_channels] + \
                              prev * prev_attn * (1 - current_attn[:, :self.forward_channels])

            # Backward shift (from t+1)
            if t < T - 1:
                next_frame = x[:, t + 1, self.forward_channels:self.forward_channels + self.backward_channels]
                next_attn = attn_weights[:, t + 1, self.forward_channels:self.forward_channels + self.backward_channels]
                backward_part = backward_part * current_attn[:, self.forward_channels:self.forward_channels + self.backward_channels] + \
                               next_frame * next_attn * (1 - current_attn[:, self.forward_channels:self.forward_channels + self.backward_channels])

            # Concatenate parts
            output[:, t] = torch.cat([forward_part, backward_part, static_part], dim=1)

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Apply TAS to numpy array.

        Args:
            x: Input array of shape (T, C, H, W)

        Returns:
            Shifted features as numpy array
        """
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()

        # Apply forward pass
        with torch.no_grad():
            output_tensor = self.forward(x_tensor)

        return output_tensor.numpy()

    def __repr__(self) -> str:
        """String representation."""
        return f"TemporalAttentionShift(channels={self.channels}, shift_ratio={self.shift_ratio})"

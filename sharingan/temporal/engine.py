"""Unified Temporal Engine for coordinating temporal reasoning modules."""

from typing import List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn


class TemporalEngine(nn.Module):
    """Unified interface for temporal reasoning modules."""

    def __init__(self, modules: List[nn.Module] = None, config: Dict = None):
        """
        Initialize Temporal Engine.

        Args:
            modules: List of temporal modules to compose
            config: Configuration for module coordination

        Example:
            >>> from sharingan.temporal import *
            >>> engine = TemporalEngine([
            ...     TemporalAttentionShift(channels=512),
            ...     CrossFrameGatingNetwork(feature_dim=512),
            ...     TemporalDilatedAttention(feature_dim=512),
            ...     MotionAwareAdaptivePooling(),
            ...     TemporalMemoryTokens(num_tokens=8, token_dim=512)
            ... ])
        """
        super().__init__()
        self.modules_list = nn.ModuleList(modules) if modules else nn.ModuleList()
        self.config = config if config is not None else self._get_default_config()

        # Categorize modules by type
        self._categorize_modules()

        # Streaming state
        self._streaming_mode = False
        self._frame_history = []
        self._embedding_history = []
        self._max_history = self.config.get("max_history", 32)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "max_history": 32,
            "enable_all": True,
            "streaming_mode": False,
        }

    def _categorize_modules(self) -> None:
        """Categorize modules by type for efficient processing."""
        self.tas_modules = []
        self.gating_modules = []
        self.tda_modules = []
        self.pooling_modules = []
        self.memory_modules = []

        for module in self.modules_list:
            module_name = module.__class__.__name__
            if "TemporalAttentionShift" in module_name:
                self.tas_modules.append(module)
            elif "CrossFrameGating" in module_name:
                self.gating_modules.append(module)
            elif "TemporalDilatedAttention" in module_name:
                self.tda_modules.append(module)
            elif "MotionAwareAdaptivePooling" in module_name:
                self.pooling_modules.append(module)
            elif "TemporalMemoryTokens" in module_name:
                self.memory_modules.append(module)

    def process_sequence(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through all temporal modules.

        Args:
            embeddings: Input embeddings of shape (T, D) or (B, T, D)

        Returns:
            Processed embeddings of same shape
        """
        # Ensure batch dimension
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = embeddings.shape
        output = embeddings

        # Apply Cross-Frame Gating
        for gating_module in self.gating_modules:
            if hasattr(gating_module, 'forward_sequence'):
                output = gating_module.forward_sequence(output)

        # Apply Temporal Dilated Attention
        for tda_module in self.tda_modules:
            if hasattr(tda_module, 'forward_sequence'):
                output = tda_module.forward_sequence(output, max_history=self._max_history)

        # Update Memory Tokens with each frame
        for memory_module in self.memory_modules:
            for t in range(T):
                memory_module.update(output[:, t])

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def process_streaming(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Process single frame in streaming mode.

        Args:
            embedding: Frame embedding of shape (D,) or (1, D)

        Returns:
            Processed embedding
        """
        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, D)

        # Add to history
        self._embedding_history.append(embedding)
        if len(self._embedding_history) > self._max_history:
            self._embedding_history.pop(0)

        output = embedding

        # Apply Cross-Frame Gating with previous frame
        if len(self._embedding_history) > 1:
            for gating_module in self.gating_modules:
                output = gating_module.forward(output, self._embedding_history[-2])

        # Apply Temporal Dilated Attention with history
        if len(self._embedding_history) > 1:
            history = list(reversed(self._embedding_history[:-1]))  # Most recent first
            for tda_module in self.tda_modules:
                output = tda_module.forward(output, history)

        # Update Memory Tokens
        for memory_module in self.memory_modules:
            memory_module.update(output)

        return output.squeeze(0)

    def enable_streaming(self) -> None:
        """Enable streaming mode."""
        self._streaming_mode = True
        self.reset_streaming_state()

    def disable_streaming(self) -> None:
        """Disable streaming mode."""
        self._streaming_mode = False

    def reset_streaming_state(self) -> None:
        """Reset streaming state."""
        self._frame_history = []
        self._embedding_history = []
        for memory_module in self.memory_modules:
            memory_module.reset()

    def get_temporal_context(self) -> Optional[torch.Tensor]:
        """
        Get current temporal context from memory tokens.

        Returns:
            Temporal context tensor or None if no memory modules
        """
        if len(self.memory_modules) > 0:
            # Return context from first memory module
            return self.memory_modules[0].get_context()
        return None

    def add_module_by_name(self, module_name: str, **kwargs) -> None:
        """
        Add a temporal module by name.

        Args:
            module_name: Name of module ("TAS", "Gating", "TDA", "Pooling", "Memory")
            **kwargs: Module-specific parameters
        """
        from sharingan.temporal import (
            TemporalAttentionShift,
            CrossFrameGatingNetwork,
            TemporalDilatedAttention,
            MotionAwareAdaptivePooling,
            TemporalMemoryTokens
        )

        module_map = {
            "TAS": TemporalAttentionShift,
            "Gating": CrossFrameGatingNetwork,
            "TDA": TemporalDilatedAttention,
            "Pooling": MotionAwareAdaptivePooling,
            "Memory": TemporalMemoryTokens,
        }

        if module_name not in module_map:
            raise ValueError(f"Unknown module: {module_name}")

        module = module_map[module_name](**kwargs)
        self.modules_list.append(module)
        self._categorize_modules()

    def get_module_count(self) -> Dict[str, int]:
        """Get count of each module type."""
        return {
            "TAS": len(self.tas_modules),
            "Gating": len(self.gating_modules),
            "TDA": len(self.tda_modules),
            "Pooling": len(self.pooling_modules),
            "Memory": len(self.memory_modules),
        }

    def __repr__(self) -> str:
        """String representation."""
        counts = self.get_module_count()
        return (f"TemporalEngine(modules={sum(counts.values())}, "
                f"TAS={counts['TAS']}, Gating={counts['Gating']}, "
                f"TDA={counts['TDA']}, Pooling={counts['Pooling']}, "
                f"Memory={counts['Memory']})")

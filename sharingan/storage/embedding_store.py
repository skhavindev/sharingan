"""Efficient embedding storage with quantization.

Instead of storing raw frames (300MB for 5min video),
we store quantized embeddings (~2.3MB for 5min video).
"""

from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np
import json
from pathlib import Path


class QuantizationType(Enum):
    """Quantization types for embeddings."""
    FLOAT32 = "float32"  # No quantization (4 bytes per value)
    FLOAT16 = "float16"  # Half precision (2 bytes per value)
    INT8 = "int8"        # 8-bit quantization (1 byte per value)


class EmbeddingStore:
    """
    Efficient storage for video embeddings.
    
    Storage breakdown for 5-minute video (9,000 frames):
    - Raw frames: 1920x1080x3 bytes × 9,000 = ~56GB
    - JPEG frames: ~300MB
    - Float32 embeddings (512-dim): 512×4 bytes × 9,000 = ~18MB
    - Float16 embeddings: 512×2 bytes × 9,000 = ~9MB
    - Int8 embeddings: 512×1 bytes × 9,000 = ~4.5MB
    - With 256-dim: 256×1 bytes × 9,000 = ~2.3MB ✨
    """
    
    def __init__(self, quantization: QuantizationType = QuantizationType.INT8):
        """
        Initialize embedding store.
        
        Args:
            quantization: Quantization type for storage
        """
        self.quantization = quantization
        self.embeddings = []
        self.metadata = []
        self.scale_factors = []  # For int8 quantization
        self.zero_points = []    # For int8 quantization
        
    def add_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float,
        frame_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add embedding with metadata.
        
        Args:
            embedding: Frame embedding (float32)
            timestamp: Frame timestamp in seconds
            frame_index: Frame index
            metadata: Optional metadata (motion score, etc.)
        """
        # Quantize embedding
        quantized, scale, zero_point = self._quantize(embedding)
        
        self.embeddings.append(quantized)
        self.scale_factors.append(scale)
        self.zero_points.append(zero_point)
        
        # Store metadata
        meta = {
            "timestamp": timestamp,
            "frame_index": frame_index,
            **(metadata or {})
        }
        self.metadata.append(meta)
    
    def _quantize(self, embedding: np.ndarray) -> tuple:
        """
        Quantize embedding based on quantization type.
        
        Args:
            embedding: Float32 embedding
            
        Returns:
            (quantized_embedding, scale_factor, zero_point)
        """
        if self.quantization == QuantizationType.FLOAT32:
            return embedding.astype(np.float32), 1.0, 0.0
        
        elif self.quantization == QuantizationType.FLOAT16:
            return embedding.astype(np.float16), 1.0, 0.0
        
        elif self.quantization == QuantizationType.INT8:
            # Symmetric quantization: map [-max, max] to [-127, 127]
            max_val = np.max(np.abs(embedding))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            quantized = np.round(embedding / scale).astype(np.int8)
            return quantized, scale, 0.0
        
        return embedding, 1.0, 0.0
    
    def get_embedding(self, index: int) -> np.ndarray:
        """
        Get dequantized embedding at index.
        
        Args:
            index: Embedding index
            
        Returns:
            Float32 embedding
        """
        quantized = self.embeddings[index]
        scale = self.scale_factors[index]
        
        if self.quantization == QuantizationType.INT8:
            return quantized.astype(np.float32) * scale
        elif self.quantization == QuantizationType.FLOAT16:
            return quantized.astype(np.float32)
        else:
            return quantized
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all dequantized embeddings.
        
        Returns:
            Array of shape (N, D) with float32 embeddings
        """
        embeddings = []
        for i in range(len(self.embeddings)):
            embeddings.append(self.get_embedding(i))
        return np.array(embeddings)
    
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Get metadata at index."""
        return self.metadata[index]
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all metadata."""
        return self.metadata
    
    def save(self, path: str) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_array = np.array(self.embeddings)
        np.save(path / "embeddings.npy", embeddings_array)
        
        # Save scale factors and zero points
        np.save(path / "scales.npy", np.array(self.scale_factors))
        np.save(path / "zeros.npy", np.array(self.zero_points))
        
        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "quantization": self.quantization.value,
                "count": len(self.embeddings),
                "dimension": self.embeddings[0].shape[0] if self.embeddings else 0,
                "metadata": self.metadata
            }, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load embeddings and metadata from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        # Load embeddings directly as array (faster)
        self.embeddings = list(np.load(path / "embeddings.npy", allow_pickle=False))
        self.scale_factors = list(np.load(path / "scales.npy", allow_pickle=False))
        self.zero_points = list(np.load(path / "zeros.npy", allow_pickle=False))
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            data = json.load(f)
            self.quantization = QuantizationType(data["quantization"])
            self.metadata = data["metadata"]
    
    def get_storage_size(self) -> Dict[str, float]:
        """
        Get storage size information.
        
        Returns:
            Dict with size in bytes and MB
        """
        if not self.embeddings:
            return {"bytes": 0, "mb": 0, "per_frame_bytes": 0}
        
        # Calculate embedding storage
        embedding_bytes = sum(emb.nbytes for emb in self.embeddings)
        
        # Calculate metadata storage (approximate)
        metadata_bytes = len(json.dumps(self.metadata).encode())
        
        total_bytes = embedding_bytes + metadata_bytes
        
        return {
            "bytes": total_bytes,
            "mb": total_bytes / (1024 * 1024),
            "per_frame_bytes": total_bytes / len(self.embeddings) if self.embeddings else 0,
            "embedding_bytes": embedding_bytes,
            "metadata_bytes": metadata_bytes,
            "count": len(self.embeddings)
        }
    
    def __len__(self) -> int:
        """Get number of stored embeddings."""
        return len(self.embeddings)
    
    def __repr__(self) -> str:
        """String representation."""
        size_info = self.get_storage_size()
        return (f"EmbeddingStore(count={len(self)}, "
                f"quantization={self.quantization.value}, "
                f"size={size_info['mb']:.2f}MB)")

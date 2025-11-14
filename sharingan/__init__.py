"""
Sharingan: Semantic Video Understanding with Temporal Reasoning

A Python package for semantic video intelligence combining
vision-language models with advanced temporal reasoning.
"""

__version__ = "0.2.0"
__author__ = "Sharingan Contributors"

# Main API - Use this for most tasks
from sharingan.processor import VideoProcessor

# Core components for advanced usage
from sharingan.video import Video, VideoLoader, FrameSampler
from sharingan.vlm import FrameEncoder, SmolVLMEncoder
from sharingan.temporal import TemporalEngine
from sharingan.storage import EmbeddingStore, QuantizationType
from sharingan.events import Event, Timeline, EventDetector
from sharingan.chat import VideoLLM

__all__ = [
    # Main API
    "VideoProcessor",
    
    # Video processing
    "Video",
    "VideoLoader",
    "FrameSampler",
    
    # Vision models
    "FrameEncoder",
    "SmolVLMEncoder",
    
    # Temporal reasoning
    "TemporalEngine",
    
    # Storage
    "EmbeddingStore",
    "QuantizationType",
    
    # Events
    "Event",
    "Timeline",
    "EventDetector",
    
    # AI Chat
    "VideoLLM",
]

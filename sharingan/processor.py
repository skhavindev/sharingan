"""
High-level VideoProcessor with all features.

This is the main entry point for using Sharingan.
"""

from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path


class VideoProcessor:
    """
    Complete video processing pipeline with all features.
    
    Example:
        >>> processor = VideoProcessor(vlm_model='clip', device='auto')
        >>> results = processor.process('video.mp4')
        >>> matches = processor.query('person speaking')
        >>> response = processor.chat('What happens in this video?', use_llm=True)
    """
    
    def __init__(
        self,
        vlm_model: str = 'clip',
        device: str = 'auto',
        target_fps: float = 5.0,
        enable_temporal: bool = True,
        enable_tracking: bool = False,
        batch_size: int = 32,
        cache_dir: str = 'cache'
    ):
        """
        Initialize video processor.
        
        Args:
            vlm_model: Vision model ('clip' or 'smolvlm')
            device: Device to use ('cpu', 'cuda', or 'auto')
            target_fps: Frames per second to process
            enable_temporal: Enable temporal reasoning
            enable_tracking: Enable entity tracking
            batch_size: Batch size for processing
            cache_dir: Directory for caching embeddings
        """
        self.vlm_model = vlm_model
        self.device = device
        self.target_fps = target_fps
        self.enable_temporal = enable_temporal
        self.enable_tracking = enable_tracking
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # State
        self.embeddings = None
        self.timestamps = None
        self.frame_indices = None
        self.video_info = None
        self.events = None
        
        # Models (lazy loaded)
        self._encoder = None
        self._smolvlm = None
        self._llm = None
    
    def process(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with results:
                - video_info: Video metadata
                - events: Detected events
                - embeddings: Frame embeddings
                - timestamps: Frame timestamps
                - frame_indices: Frame indices
        """
        from sharingan.video import VideoLoader, FrameSampler
        from sharingan.vlm import FrameEncoder, SmolVLMEncoder
        from sharingan.temporal import TemporalEngine, CrossFrameGatingNetwork, TemporalMemoryTokens
        from sharingan.events import EventDetector
        from sharingan.storage import EmbeddingStore, QuantizationType
        import torch
        import hashlib
        import os
        
        print(f"🎬 Processing video: {video_path}")
        
        # Check cache
        file_stat = os.stat(video_path)
        cache_key = f"{os.path.basename(video_path)}_{file_stat.st_size}_{int(file_stat.st_mtime)}"
        video_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_path = self.cache_dir / f"video_{video_hash}"
        
        if cache_path.exists():
            print(f"💾 Loading from cache...")
            store = EmbeddingStore()
            store.load(str(cache_path))
            self.embeddings = store.get_all_embeddings()
            metadata = store.get_all_metadata()
            self.timestamps = [m['timestamp'] for m in metadata]
            self.frame_indices = [m['frame_index'] for m in metadata]
            print(f"✓ Loaded {len(self.embeddings)} cached embeddings")
        else:
            # Load video
            print(f"📹 Loading video...")
            loader = VideoLoader(video_path, backend='opencv')
            sampler = FrameSampler(strategy='adaptive', target_fps=self.target_fps)
            
            # Initialize encoder
            print(f"🧠 Initializing {self.vlm_model.upper()} encoder...")
            if self.vlm_model == 'smolvlm':
                if not self._smolvlm:
                    self._smolvlm = SmolVLMEncoder(device=self.device)
                encoder = self._smolvlm
            else:
                if not self._encoder:
                    self._encoder = FrameEncoder(model_name='clip-vit-b32', device=self.device)
                encoder = self._encoder
            
            # Process frames
            print(f"⚙️  Processing frames...")
            frames = []
            self.timestamps = []
            self.frame_indices = []
            
            for frame_idx, frame in sampler.sample(loader, source_fps=loader.fps):
                frames.append(frame)
                self.timestamps.append(frame_idx / loader.fps)
                self.frame_indices.append(frame_idx)
                
                # Process in batches
                if len(frames) >= self.batch_size:
                    if self.vlm_model == 'smolvlm':
                        # Generate descriptions and embed them
                        descriptions = self._smolvlm.describe_batch(frames)
                        temp_encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                        batch_embs = [temp_encoder.encode_text(desc) for desc in descriptions]
                    else:
                        batch_embs = encoder.encode_batch(frames)
                    
                    if self.embeddings is None:
                        self.embeddings = batch_embs
                    else:
                        self.embeddings = np.vstack([self.embeddings, batch_embs])
                    
                    frames = []
            
            # Process remaining
            if frames:
                if self.vlm_model == 'smolvlm':
                    descriptions = self._smolvlm.describe_batch(frames)
                    temp_encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                    batch_embs = [temp_encoder.encode_text(desc) for desc in descriptions]
                else:
                    batch_embs = encoder.encode_batch(frames)
                
                if self.embeddings is None:
                    self.embeddings = batch_embs
                else:
                    self.embeddings = np.vstack([self.embeddings, batch_embs])
            
            print(f"✓ Processed {len(self.embeddings)} frames")
            
            # Cache embeddings
            print(f"💾 Caching embeddings...")
            store = EmbeddingStore(quantization=QuantizationType.INT8)
            for i, emb in enumerate(self.embeddings):
                store.add_embedding(emb, self.timestamps[i], self.frame_indices[i])
            store.save(str(cache_path))
            print(f"✓ Cached to {cache_path}")
            
            self.video_info = {
                'fps': loader.fps,
                'total_frames': loader.total_frames,
                'duration': self.timestamps[-1] if self.timestamps else 0,
                'processed_frames': len(self.frame_indices)
            }
        
        # Temporal reasoning
        if self.enable_temporal:
            print(f"🔄 Applying temporal reasoning...")
            engine = TemporalEngine([
                CrossFrameGatingNetwork(feature_dim=512),
                TemporalMemoryTokens(num_tokens=8, token_dim=512)
            ])
            embeddings_tensor = torch.from_numpy(np.stack(self.embeddings)).float()
            with torch.no_grad():
                processed = engine.process_sequence(embeddings_tensor)
            self.embeddings = processed.numpy()
            print(f"✓ Temporal reasoning applied")
        
        # Event detection
        print(f"🔍 Detecting events...")
        detector = EventDetector(sensitivity=0.5)
        detected_events = detector.detect_events(
            np.array(self.embeddings),
            self.timestamps,
            self.frame_indices
        )
        
        self.events = []
        for event in detected_events:
            self.events.append({
                'id': event.event_id,
                'type': event.event_type,
                'timestamp': event.start_time,
                'frame': event.start_frame,
                'confidence': event.confidence,
                'description': event.description
            })
        
        print(f"✓ Detected {len(self.events)} events")
        print(f"✅ Processing complete!")
        
        return {
            'video_info': self.video_info,
            'events': self.events,
            'embeddings': self.embeddings,
            'timestamps': self.timestamps,
            'frame_indices': self.frame_indices
        }
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query video with natural language.
        
        Args:
            text: Query text
            top_k: Number of results to return
            
        Returns:
            List of matches with timestamps and confidence scores
        """
        if self.embeddings is None:
            raise ValueError("Process a video first using .process()")
        
        from sharingan.vlm import FrameEncoder
        
        print(f"🔍 Query: '{text}'")
        
        # Encode query
        if not self._encoder:
            self._encoder = FrameEncoder(model_name='clip-vit-b32', device=self.device)
        
        query_embedding = self._encoder.encode_text(text)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'timestamp': self.timestamps[idx],
                'frame': self.frame_indices[idx],
                'confidence': float(similarities[idx]),
                'description': f"Relevant content found"
            })
        
        print(f"✓ Found {len(results)} results")
        return results
    
    def chat(self, question: str, use_llm: bool = True) -> str:
        """
        Chat about the video using AI.
        
        Args:
            question: Question about the video
            use_llm: Use Qwen2.5 for conversational response
            
        Returns:
            Response text
        """
        if self.embeddings is None:
            raise ValueError("Process a video first using .process()")
        
        # Get relevant segments
        segments = self.query(question, top_k=5)
        
        if not use_llm:
            # Simple response
            return f"Found {len(segments)} relevant moments at: " + \
                   ", ".join([f"{s['timestamp']:.1f}s" for s in segments])
        
        # Use LLM
        from sharingan.chat import VideoLLM
        
        if not self._llm:
            print(f"🤖 Initializing Qwen2.5-0.5B...")
            self._llm = VideoLLM(device=self.device)
        
        response = self._llm.chat(question, segments)
        return response
    
    def reset_chat(self):
        """Reset chat history."""
        if self._llm:
            self._llm.reset_history()

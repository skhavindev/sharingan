"""High-level Video API for semantic video understanding."""

from typing import Union, Dict, List, Any, Optional
from sharingan.video.loader import VideoLoader
from sharingan.video.sampler import FrameSampler
from sharingan.storage import EmbeddingStore, QuantizationType


class Video:
    """Main user-facing video interface."""

    def __init__(self, source: Union[str, int], **config):
        """
        Initialize video with source and configuration.

        Args:
            source: File path, URL, or camera index
            **config: Configuration options
                - backend: Video backend ("opencv", "decord", "pyav")
                - sampling_strategy: Frame sampling strategy
                - target_fps: Target sampling rate
                - device: Processing device ("cpu", "cuda", "auto")
                - model_name: VLM model to use

        Example:
            >>> video = Video("path/to/video.mp4")
            >>> events = video.detect_events()
        """
        self.source = source
        self.config = self._get_default_config()
        self.config.update(config)

        # Initialize video loader
        self.loader = VideoLoader(
            source=source,
            backend=self.config.get("backend", "opencv")
        )

        # Initialize frame sampler
        self.sampler = FrameSampler(
            strategy=self.config.get("sampling_strategy", "uniform"),
            target_fps=self.config.get("target_fps", 5.0)
        )

        # Components will be initialized lazily
        self._encoder = None
        self._temporal_engine = None
        self._event_detector = None
        self._entity_tracker = None
        self._query_engine = None
        self._timeline = None
        
        # Embedding storage (efficient!)
        self._embedding_store = EmbeddingStore(
            quantization=QuantizationType.INT8  # ~2.3MB for 5min video
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "backend": "opencv",
            "sampling_strategy": "uniform",
            "target_fps": 5.0,
            "device": "auto",
            "model_name": "clip-vit-b32",
            "enable_temporal": True,
            "enable_tracking": True,
        }

    def describe(self) -> str:
        """
        Generate natural language description of video content.

        Returns:
            Natural language description of the video

        Example:
            >>> video = Video("demo.mp4")
            >>> print(video.describe())
            "A person walking in a park with trees in the background..."
        """
        from sharingan.vlm import FrameEncoder
        import torch
        import numpy as np
        
        # Initialize encoder if needed
        if self._encoder is None:
            self._encoder = FrameEncoder(
                model_name=self.config.get("model_name", "clip-vit-b32"),
                device=self.config.get("device", "auto")
            )
        
        # Sample key frames
        frames_to_sample = 5
        sampled_frames = []
        frame_count = 0
        
        for frame_idx, frame in self.sampler.sample(self.loader, source_fps=self.loader.fps):
            sampled_frames.append(frame)
            frame_count += 1
            if frame_count >= frames_to_sample:
                break
        
        if len(sampled_frames) == 0:
            return "Unable to process video"
        
        # Encode frames
        embeddings = self._encoder.encode_batch(sampled_frames)
        
        # Generate descriptions using CLIP text similarity
        candidate_descriptions = [
            "a video showing people in an indoor setting",
            "a video showing outdoor scenery",
            "a video with text and graphics",
            "a video showing a presentation or lecture",
            "a video showing sports or physical activity",
            "a video showing animals or nature",
            "a video showing urban environment",
            "a video showing educational content",
            "a video showing entertainment or performance",
            "a video showing technology or computers"
        ]
        
        # Encode candidate descriptions
        text_embeddings = []
        for desc in candidate_descriptions:
            text_emb = self._encoder.encode_text(desc)
            text_embeddings.append(text_emb)
        
        text_embeddings = np.array(text_embeddings)
        
        # Compute similarity between video and descriptions
        avg_video_embedding = np.mean(embeddings, axis=0)
        similarities = np.dot(text_embeddings, avg_video_embedding)
        
        # Get top 3 descriptions
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_descriptions = [candidate_descriptions[i] for i in top_indices]
        
        # Build description
        description = f"This is {top_descriptions[0]}. "
        description += f"The video appears to contain {top_descriptions[1].replace('a video showing', '')}. "
        description += f"Duration: {self.loader.total_frames / self.loader.fps:.1f}s, "
        description += f"FPS: {self.loader.fps:.1f}"
        
        return description

    def detect_events(self) -> List[Any]:
        """
        Detect significant events in video.

        Returns:
            List of Event objects

        Example:
            >>> video = Video("demo.mp4")
            >>> events = video.detect_events()
            >>> for event in events:
            ...     print(f"{event.event_type} at {event.start_time}s")
        """
        from sharingan.vlm import FrameEncoder
        from sharingan.events import EventDetector
        from sharingan.temporal import TemporalEngine, CrossFrameGatingNetwork, TemporalMemoryTokens
        import torch
        import numpy as np
        
        # Initialize components
        if self._encoder is None:
            self._encoder = FrameEncoder(
                model_name=self.config.get("model_name", "clip-vit-b32"),
                device=self.config.get("device", "auto")
            )
        
        if self._event_detector is None:
            self._event_detector = EventDetector(sensitivity=0.5)
        
        # Process video and store embeddings efficiently
        print(f"Processing video: {self.source}")
        print(f"FPS: {self.loader.fps}, Total frames: {self.loader.total_frames}")
        
        for frame_idx, frame in self.sampler.sample(self.loader, source_fps=self.loader.fps):
            timestamp = frame_idx / self.loader.fps
            
            # Encode frame
            embedding = self._encoder.encode_frame(frame)
            
            # Store embedding (quantized for efficiency)
            self._embedding_store.add_embedding(
                embedding=embedding,
                timestamp=timestamp,
                frame_index=frame_idx
            )
            
            if len(self._embedding_store) % 10 == 0:
                print(f"Processed {len(self._embedding_store)} frames...")
        
        print(f"Total processed: {len(self._embedding_store)} frames")
        
        # Show storage efficiency
        storage_info = self._embedding_store.get_storage_size()
        print(f"Storage: {storage_info['mb']:.2f}MB ({storage_info['per_frame_bytes']:.0f} bytes/frame)")
        
        # Get embeddings for processing
        embeddings = self._embedding_store.get_all_embeddings()
        metadata = self._embedding_store.get_all_metadata()
        timestamps = [m['timestamp'] for m in metadata]
        frame_indices = [m['frame_index'] for m in metadata]
        
        # Apply temporal reasoning if enabled
        if self.config.get("enable_temporal", True) and len(embeddings) > 1:
            print("Applying temporal reasoning...")
            if self._temporal_engine is None:
                self._temporal_engine = TemporalEngine([
                    CrossFrameGatingNetwork(feature_dim=self._encoder.embedding_dim),
                    TemporalMemoryTokens(num_tokens=8, token_dim=self._encoder.embedding_dim)
                ])
            
            embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float()
            with torch.no_grad():
                processed_embeddings = self._temporal_engine.process_sequence(embeddings_tensor)
            embeddings = processed_embeddings.numpy()
        
        # Detect events
        print("Detecting events...")
        events = self._event_detector.detect_events(
            np.array(embeddings),
            timestamps,
            frame_indices
        )
        
        print(f"Detected {len(events)} events")
        return events

    def query(self, question: str) -> List[Any]:
        """
        Natural language query interface.

        Args:
            question: Natural language question about the video

        Returns:
            List of QueryResult objects

        Example:
            >>> video = Video("demo.mp4")
            >>> results = video.query("Where is the person in red going?")
            >>> for result in results:
            ...     print(f"Found at {result.timestamp}s: {result.description}")
        """
        from sharingan.vlm import FrameEncoder
        from sharingan.query import QueryResult
        import numpy as np
        
        # Initialize encoder if needed
        if self._encoder is None:
            self._encoder = FrameEncoder(
                model_name=self.config.get("model_name", "clip-vit-b32"),
                device=self.config.get("device", "auto")
            )
        
        # Encode query
        query_embedding = self._encoder.encode_text(question)
        
        # Process video and collect embeddings (reuse if already processed)
        if not hasattr(self, '_cached_embeddings') or self._cached_embeddings is None:
            embeddings = []
            timestamps = []
            frame_indices = []
            
            print(f"Processing video for query: '{question}'")
            
            # Reset video to beginning
            self.loader = VideoLoader(self.source, backend=self.config.get("backend", "opencv"))
            
            for frame_idx, frame in self.sampler.sample(self.loader, source_fps=self.loader.fps):
                timestamp = frame_idx / self.loader.fps
                embedding = self._encoder.encode_frame(frame)
                
                embeddings.append(embedding)
                timestamps.append(timestamp)
                frame_indices.append(frame_idx)
            
            # Cache for future queries
            self._cached_embeddings = embeddings
            self._cached_timestamps = timestamps
            self._cached_frame_indices = frame_indices
        else:
            embeddings = self._cached_embeddings
            timestamps = self._cached_timestamps
            frame_indices = self._cached_frame_indices
            print(f"Using cached embeddings for query: '{question}'")
        
        # Compute similarities
        embeddings_array = np.array(embeddings)
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Get top 5 results
        top_k = min(5, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Create results
        results = []
        for idx in top_indices:
            result = QueryResult(
                timestamp=timestamps[idx],
                frame_index=frame_indices[idx],
                description=f"Relevant content found",
                confidence=float(similarities[idx]),
                bounding_box=None,
                related_entities=[],
                related_events=[]
            )
            results.append(result)
        
        print(f"Found {len(results)} relevant moments")
        return results

    def get_timeline(self) -> Any:
        """
        Get structured semantic timeline.

        Returns:
            Timeline object with events, entities, and scene boundaries

        Example:
            >>> video = Video("demo.mp4")
            >>> timeline = video.get_timeline()
            >>> print(f"Duration: {timeline.duration}s")
            >>> print(f"Events: {len(timeline.events)}")
        """
        # This will be implemented once we have timeline builder
        raise NotImplementedError(
            "get_timeline() will be implemented after timeline builder is ready"
        )

    def track_entities(self) -> Dict[str, Any]:
        """
        Track all entities across frames.

        Returns:
            Dictionary mapping entity IDs to Track objects

        Example:
            >>> video = Video("demo.mp4")
            >>> tracks = video.track_entities()
            >>> for track_id, track in tracks.items():
            ...     print(f"Entity {track_id}: {len(track.frame_indices)} frames")
        """
        # This will be implemented once we have entity tracker
        raise NotImplementedError(
            "track_entities() will be implemented after entity tracker is ready"
        )

    def process(self) -> None:
        """
        Process video and build internal representations.

        This method processes the entire video, extracting features,
        detecting events, and building the semantic timeline.
        """
        # This will orchestrate the full pipeline
        raise NotImplementedError(
            "process() will be implemented after all components are ready"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"Video(source={self.source}, fps={self.loader.fps})"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources
        pass

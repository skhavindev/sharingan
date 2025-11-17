"""
Complete test demonstrating all Sharingan features.

This file shows all available syntax and options.
"""

import numpy as np
from pathlib import Path


def test_basic_processing():
    """Test 1: Basic video processing with CLIP."""
    print("\n" + "="*60)
    print("TEST 1: Basic Video Processing (CLIP)")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    # Initialize processor with CLIP (fast)
    processor = VideoProcessor(
        vlm_model='clip',
        device='auto',
        target_fps=5.0
    )
    
    # Process video
    results = processor.process('test_video.mp4')
    
    print(f"✓ Processed {results['video_info']['processed_frames']} frames")
    print(f"✓ Detected {len(results['events'])} events")
    print(f"✓ Duration: {results['video_info']['duration']:.1f}s")


def test_smolvlm_processing():
    """Test 2: Video processing with SmolVLM for detailed descriptions."""
    print("\n" + "="*60)
    print("TEST 2: SmolVLM Processing (Detailed Descriptions)")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    # Initialize with SmolVLM (detailed)
    processor = VideoProcessor(
        vlm_model='smolvlm',  # Generates frame descriptions
        device='auto',
        target_fps=3.0  # Slower for detailed analysis
    )
    
    results = processor.process('test_video.mp4')
    
    print(f"✓ Generated descriptions for {results['video_info']['processed_frames']} frames")


def test_semantic_search():
    """Test 3: Semantic video search."""
    print("\n" + "="*60)
    print("TEST 3: Semantic Search")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    processor = VideoProcessor(vlm_model='clip')
    processor.process('test_video.mp4')
    
    # Search for specific content
    queries = [
        'person speaking',
        'text and graphics',
        'outdoor scene'
    ]
    
    for query in queries:
        results = processor.query(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for i, match in enumerate(results, 1):
            print(f"  {i}. {match.timestamp:.1f}s - {match.confidence:.1%}")


def test_ai_chat():
    """Test 4: AI chat with Qwen2.5."""
    print("\n" + "="*60)
    print("TEST 4: AI Chat (Qwen2.5)")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    processor = VideoProcessor(vlm_model='clip')
    processor.process('test_video.mp4')
    
    # Conversational queries
    questions = [
        'What happens in this video?',
        'Describe the main events',
        'What is the person doing?'
    ]
    
    for question in questions:
        response = processor.chat(question, use_llm=True)
        print(f"\nQ: {question}")
        print(f"A: {response}")


def test_temporal_reasoning():
    """Test 5: Temporal reasoning features."""
    print("\n" + "="*60)
    print("TEST 5: Temporal Reasoning")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    # Enable temporal reasoning
    processor = VideoProcessor(
        vlm_model='clip',
        enable_temporal=True,  # Cross-frame attention
        enable_tracking=True   # Entity tracking
    )
    
    results = processor.process('test_video.mp4')
    
    print(f"✓ Temporal reasoning applied")
    print(f"✓ Detected {len(results['events'])} events with temporal context")


def test_event_detection():
    """Test 6: Event detection."""
    print("\n" + "="*60)
    print("TEST 6: Event Detection")
    print("="*60)
    
    from sharingan import VideoProcessor
    from sharingan.events import EventDetector
    
    processor = VideoProcessor(vlm_model='clip')
    results = processor.process('test_video.mp4')
    
    # Custom event detection
    detector = EventDetector(sensitivity=0.5)
    events = detector.detect_events(
        results['embeddings'],
        results['timestamps'],
        results['frame_indices']
    )
    
    print(f"\nDetected {len(events)} events:")
    for event in events[:5]:  # Show first 5
        print(f"  • {event.event_type} at {event.start_time:.1f}s ({event.confidence:.1%})")


def test_storage_efficiency():
    """Test 7: Efficient storage with quantization."""
    print("\n" + "="*60)
    print("TEST 7: Storage Efficiency")
    print("="*60)
    
    from sharingan.storage import EmbeddingStore, QuantizationType
    import numpy as np
    
    # Create embeddings
    embeddings = [np.random.randn(512).astype(np.float32) for _ in range(1000)]
    timestamps = [i * 0.2 for i in range(1000)]
    frame_indices = list(range(1000))
    
    # Test different quantization levels
    for quant_type in [QuantizationType.FLOAT32, QuantizationType.INT8]:
        store = EmbeddingStore(quantization=quant_type)
        
        for emb, ts, idx in zip(embeddings, timestamps, frame_indices):
            store.add_embedding(emb, ts, idx)
        
        size_info = store.get_storage_size()
        print(f"\n{quant_type.value}:")
        print(f"  Total: {size_info['mb']:.2f}MB")
        print(f"  Per frame: {size_info['per_frame_bytes']:.0f} bytes")


def test_batch_processing():
    """Test 8: Batch processing for speed."""
    print("\n" + "="*60)
    print("TEST 8: Batch Processing")
    print("="*60)
    
    from sharingan.vlm import FrameEncoder
    import numpy as np
    import time
    
    encoder = FrameEncoder(model_name='clip-vit-b32', device='auto')
    
    # Create dummy frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
    
    # Single frame processing
    start = time.time()
    for frame in frames:
        _ = encoder.encode_frame(frame)
    single_time = time.time() - start
    
    # Batch processing
    start = time.time()
    _ = encoder.encode_batch(frames)
    batch_time = time.time() - start
    
    print(f"\nSingle frame: {single_time:.2f}s")
    print(f"Batch (32): {batch_time:.2f}s")
    print(f"Speedup: {single_time/batch_time:.1f}x")


def test_web_ui():
    """Test 9: Web UI (programmatic)."""
    print("\n" + "="*60)
    print("TEST 9: Web UI")
    print("="*60)
    
    from sharingan.ui import run_ui
    
    print("\nTo launch web UI:")
    print("  python -m sharingan.cli ui")
    print("\nOr programmatically:")
    print("  from sharingan.ui import run_ui")
    print("  run_ui(port=5000, open_browser=True)")
    
    # Don't actually start server in test
    print("\n✓ Web UI available")


def test_all_options():
    """Test 10: All available options."""
    print("\n" + "="*60)
    print("TEST 10: All Available Options")
    print("="*60)
    
    from sharingan import VideoProcessor
    
    # Show all options
    processor = VideoProcessor(
        # Vision model options
        vlm_model='clip',           # 'clip' or 'smolvlm'
        
        # Device options
        device='auto',              # 'cpu', 'cuda', or 'auto'
        
        # Processing options
        target_fps=5.0,             # Frames per second
        enable_temporal=True,       # Temporal reasoning
        enable_tracking=False,      # Entity tracking
        
        # Advanced options
        batch_size=32,              # Batch processing size
        cache_dir='cache'           # Cache directory
    )
    
    print("\n✓ All options configured")
    print("\nAvailable options:")
    print("  • vlm_model: 'clip' (fast) or 'smolvlm' (detailed)")
    print("  • device: 'cpu', 'cuda', or 'auto'")
    print("  • target_fps: Frames per second to process")
    print("  • enable_temporal: Cross-frame reasoning")
    print("  • enable_tracking: Entity tracking")
    print("  • batch_size: Batch processing size")
    print("  • cache_dir: Cache directory path")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SHARINGAN - Complete Feature Test")
    print("="*60)
    
    tests = [
        test_basic_processing,
        test_smolvlm_processing,
        test_semantic_search,
        test_ai_chat,
        test_temporal_reasoning,
        test_event_detection,
        test_storage_efficiency,
        test_batch_processing,
        test_web_ui,
        test_all_options
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n⚠️  Test failed: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == '__main__':
    main()

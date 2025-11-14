#!/usr/bin/env python3
"""
Comprehensive Sharingan Demo
Shows all features: describe, detect_events, and query
"""

from sharingan import Video
import sys


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    # Check for video path
    if len(sys.argv) < 2:
        print("Usage: python full_demo.py <video_path>")
        print("\nExample:")
        print("  python full_demo.py video.mp4")
        print("  python full_demo.py '/mnt/c/Users/Name/Downloads/video.mp4'")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print_header("🎬 Sharingan - Semantic Video Understanding")
    print(f"📹 Video: {video_path}\n")
    
    # Initialize video
    print("Loading video...")
    video = Video(
        video_path,
        target_fps=3.0,  # Sample 3 frames per second
        enable_temporal=True  # Enable temporal reasoning
    )
    
    print(f"✓ Loaded: {video.loader.fps:.1f} FPS, {video.loader.total_frames} frames")
    print(f"  Duration: {video.loader.total_frames / video.loader.fps:.1f} seconds\n")
    
    # Feature 1: Video Description
    print_header("📝 Feature 1: Video Description")
    print("Generating semantic description of video content...\n")
    
    try:
        description = video.describe()
        print(f"Description:\n{description}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Feature 2: Event Detection
    print_header("🔍 Feature 2: Event Detection")
    print("Detecting significant events and scene changes...\n")
    
    try:
        events = video.detect_events()
        
        if len(events) == 0:
            print("No significant events detected.\n")
        else:
            print(f"✓ Detected {len(events)} events:\n")
            
            # Show first 10 events
            for i, event in enumerate(events[:10], 1):
                mins = int(event.start_time // 60)
                secs = int(event.start_time % 60)
                
                print(f"{i:2d}. [{mins}:{secs:02d}] {event.event_type.upper()}")
                print(f"    {event.description}")
                print(f"    Confidence: {event.confidence*100:.1f}%")
                print(f"    Frames: {event.start_frame} → {event.end_frame}")
                print()
            
            if len(events) > 10:
                print(f"    ... and {len(events) - 10} more events\n")
    
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Feature 3: Natural Language Queries
    print_header("💬 Feature 3: Natural Language Queries")
    print("Searching video content using natural language...\n")
    
    # Example queries
    queries = [
        "educational content or presentation",
        "text and graphics on screen",
        "person speaking or talking"
    ]
    
    for query_text in queries:
        print(f"Query: \"{query_text}\"")
        print("-" * 60)
        
        try:
            results = video.query(query_text)
            
            if len(results) == 0:
                print("  No relevant moments found.\n")
            else:
                print(f"  Found {len(results)} relevant moments:\n")
                
                for i, result in enumerate(results[:3], 1):
                    mins = int(result.timestamp // 60)
                    secs = int(result.timestamp % 60)
                    
                    print(f"  {i}. [{mins}:{secs:02d}] - Relevance: {result.confidence*100:.1f}%")
                    print(f"     Frame: {result.frame_index}")
                
                print()
        
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    # Summary
    print_header("✅ Analysis Complete!")
    print("Sharingan successfully analyzed your video using:")
    print("  • Lightweight VLM (CLIP) for semantic understanding")
    print("  • Temporal reasoning for context-aware analysis")
    print("  • Scene change detection for event identification")
    print("  • Natural language queries for content search")
    print("\nTry the web UI for a visual interface:")
    print("  sharingan run")
    print()


if __name__ == "__main__":
    main()

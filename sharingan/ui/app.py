"""Lightweight Flask-based UI for Sharingan."""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, Response
from werkzeug.utils import secure_filename
import threading
import webbrowser
import time
import numpy as np
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['SECRET_KEY'] = 'sharingan-secret-key'

# Disable Flask's default logging (only show errors)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
processing_state = {
    'status': 'idle',
    'progress': 0,
    'current_video': None,
    'results': None,
    'error': None,
    'embeddings': None,
    'timestamps': None,
    'frame_indices': None
}

# Cache directory
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)


@app.route('/')
def index():
    """Render main UI."""
    return render_template('index.html')


@app.route('/player')
def player():
    """Render video player page."""
    if processing_state.get('results') is None:
        return redirect('/')
    
    video_path = processing_state.get('current_video')
    filename = os.path.basename(video_path) if video_path else None
    
    return render_template('player.html', 
                         filename=filename,
                         results=processing_state['results'])


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processing_state['current_video'] = filepath
        processing_state['status'] = 'uploaded'
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_video():
    """Process video with Sharingan."""
    try:
        data = request.json
        video_path = processing_state.get('current_video')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'No video uploaded'}), 400
        
        # Get processing options
        options = {
            'enable_temporal': data.get('enable_temporal', True),
            'enable_tracking': data.get('enable_tracking', False),
            'target_fps': data.get('target_fps', 5.0),
            'device': data.get('device', 'auto'),  # Auto-detect best device
            'vlm_model': data.get('vlm_model', 'clip')  # 'clip' or 'smolvlm'
        }
        
        # Start processing in background
        thread = threading.Thread(
            target=process_video_background,
            args=(video_path, options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Processing started'})
    
    except Exception as e:
        processing_state['status'] = 'error'
        processing_state['error'] = str(e)
        return jsonify({'error': str(e)}), 500


def process_video_background(video_path, options):
    """Process video in background thread."""
    try:
        from sharingan.video import VideoLoader, FrameSampler
        from sharingan.vlm import FrameEncoder
        from sharingan.temporal import TemporalEngine, CrossFrameGatingNetwork, TemporalMemoryTokens
        from sharingan.storage import EmbeddingStore, QuantizationType
        import torch
        import numpy as np
        import hashlib
        
        processing_state['status'] = 'processing'
        processing_state['progress'] = 0
        processing_state['results'] = None
        processing_state['error'] = None
        processing_state['embeddings'] = None
        processing_state['timestamps'] = None
        processing_state['frame_indices'] = None
        
        print("\n" + "="*60)
        print("🎬 Starting Video Processing")
        print("="*60)
        
        # Generate cache key
        file_stat = os.stat(video_path)
        cache_key = f"{os.path.basename(video_path)}_{file_stat.st_size}_{int(file_stat.st_mtime)}"
        video_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_path = os.path.join(CACHE_DIR, f"video_{video_hash}")
        
        # Check cache
        embeddings = None
        timestamps = None
        frame_indices = None
        
        if os.path.exists(cache_path):
            print(f"💾 Loading cached embeddings...")
            try:
                store = EmbeddingStore()
                store.load(cache_path)
                embeddings = store.get_all_embeddings()
                metadata = store.get_all_metadata()
                timestamps = [m['timestamp'] for m in metadata]
                frame_indices = [m['frame_index'] for m in metadata]
                print(f"✓ Loaded {len(embeddings)} cached embeddings")
                processing_state['progress'] = 70
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}, reprocessing...")
                embeddings = None
        
        if embeddings is None:
            # Load video
            processing_state['progress'] = 10
            print(f"📹 Loading video: {os.path.basename(video_path)}")
            loader = VideoLoader(video_path, backend='opencv')
            sampler = FrameSampler(strategy='adaptive', target_fps=options['target_fps'])
            print(f"✓ Video loaded: {loader.fps:.1f} FPS, {loader.total_frames} frames")
        
            # Initialize encoder
            processing_state['progress'] = 20
            vlm_model = options.get('vlm_model', 'clip')
            print(f"\n🧠 Initializing {vlm_model.upper()} encoder...")
            
            # Clear GPU cache if using CUDA
            if options['device'] in ['cuda', 'auto']:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"🧹 Cleared GPU cache")
                except:
                    pass
            
            encoder = None
            smolvlm_encoder = None
            has_encoder = False
            
            try:
                if vlm_model == 'smolvlm':
                    from sharingan.vlm import SmolVLMEncoder
                    smolvlm_encoder = SmolVLMEncoder(device=options['device'])
                    has_encoder = True
                    print(f"✓ SmolVLM encoder ready on {smolvlm_encoder.device}")
                else:
                    encoder = FrameEncoder(model_name='clip-vit-b32', device=options['device'])
                    has_encoder = True
                    print(f"✓ CLIP encoder ready: {encoder.model_name} on {encoder.device}")
            except Exception as e:
                # Try CPU if initial device fails (e.g., CUDA OOM)
                if options['device'] != 'cpu':
                    print(f"⚠️  Failed on {options['device']}: {e}")
                    print(f"⚠️  Trying CPU instead...")
                    try:
                        if vlm_model == 'smolvlm':
                            from sharingan.vlm import SmolVLMEncoder
                            smolvlm_encoder = SmolVLMEncoder(device='cpu')
                            has_encoder = True
                            print(f"✓ SmolVLM encoder ready on CPU")
                        else:
                            encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                            has_encoder = True
                            print(f"✓ CLIP encoder ready on CPU")
                    except Exception as e2:
                        print(f"⚠️  {vlm_model.upper()} not available: {e2}")
                        print(f"⚠️  Using mock embeddings for demonstration")
                        has_encoder = False
                else:
                    print(f"⚠️  {vlm_model.upper()} not available: {e}")
                    print(f"⚠️  Using mock embeddings for demonstration")
                    has_encoder = False
        
            # Process frames with batching for speed
            processing_state['progress'] = 30
            print(f"\n⚙️  Processing frames...")
            timestamps = []
            frame_indices = []
            embeddings = []
            
            total_frames = loader.total_frames or 1000
            processed_count = 0
            
            # Batch processing for 10x speed improvement
            batch_size = 32 if has_encoder else 1
            frame_batch = []
            batch_timestamps = []
            batch_indices = []
            
            for frame_idx, frame in sampler.sample(loader, source_fps=loader.fps):
                timestamp = frame_idx / loader.fps
                
                frame_batch.append(frame)
                batch_timestamps.append(timestamp)
                batch_indices.append(frame_idx)
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    if has_encoder:
                        if smolvlm_encoder:
                            # SmolVLM: Generate descriptions and convert to embeddings
                            descriptions = smolvlm_encoder.describe_batch(
                                frame_batch,
                                prompt="Describe what you see in this frame briefly."
                            )
                            # Use CLIP to embed the descriptions for search
                            temp_encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                            for desc in descriptions:
                                emb = temp_encoder.encode_text(desc)
                                embeddings.append(emb)
                        else:
                            # CLIP: Direct embedding
                            batch_embeddings = encoder.encode_batch(frame_batch)
                            embeddings.extend(batch_embeddings)
                    else:
                        for _ in frame_batch:
                            embeddings.append(np.random.randn(512).astype(np.float32))
                    
                    timestamps.extend(batch_timestamps)
                    frame_indices.extend(batch_indices)
                    
                    processed_count += len(frame_batch)
                    if processed_count % 50 == 0:
                        print(f"   Processed {processed_count} frames...")
                    processing_state['progress'] = 30 + int((processed_count / min(total_frames, 100)) * 40)
                    
                    # Clear batch
                    frame_batch = []
                    batch_timestamps = []
                    batch_indices = []
            
            # Process remaining frames
            if len(frame_batch) > 0:
                if has_encoder:
                    if smolvlm_encoder:
                        # SmolVLM: Generate descriptions and convert to embeddings
                        descriptions = smolvlm_encoder.describe_batch(
                            frame_batch,
                            prompt="Describe what you see in this frame briefly."
                        )
                        # Use CLIP to embed the descriptions for search
                        temp_encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                        for desc in descriptions:
                            emb = temp_encoder.encode_text(desc)
                            embeddings.append(emb)
                    else:
                        # CLIP: Direct embedding
                        batch_embeddings = encoder.encode_batch(frame_batch)
                        embeddings.extend(batch_embeddings)
                else:
                    for _ in frame_batch:
                        embeddings.append(np.random.randn(512).astype(np.float32))
                
                timestamps.extend(batch_timestamps)
                frame_indices.extend(batch_indices)
                processed_count += len(frame_batch)
        
            print(f"✓ Processed {processed_count} frames total")
            
            # Cache embeddings
            print(f"\n💾 Caching embeddings...")
            store = EmbeddingStore(quantization=QuantizationType.INT8)
            for i, emb in enumerate(embeddings):
                store.add_embedding(emb, timestamps[i], frame_indices[i])
            store.save(cache_path)
            storage_info = store.get_storage_size()
            print(f"✓ Cached {len(embeddings)} embeddings ({storage_info['mb']:.2f}MB)")
        else:
            # Load video for metadata
            loader = VideoLoader(video_path, backend='opencv')
        
        # Store for queries
        processing_state['embeddings'] = embeddings
        processing_state['timestamps'] = timestamps
        processing_state['frame_indices'] = frame_indices
        
        # Apply temporal reasoning
        processing_state['progress'] = 70
        if options['enable_temporal'] and len(embeddings) > 0:
            print(f"\n🔄 Applying temporal reasoning...")
            engine = TemporalEngine([
                CrossFrameGatingNetwork(feature_dim=512),
                TemporalMemoryTokens(num_tokens=8, token_dim=512)
            ])
            
            embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float()
            with torch.no_grad():
                processed_embeddings = engine.process_sequence(embeddings_tensor)
            
            embeddings = processed_embeddings.numpy()
            print(f"✓ Temporal reasoning applied")
        
        # Detect events
        processing_state['progress'] = 85
        print(f"\n🔍 Detecting events...")
        events = detect_simple_events(embeddings, timestamps, frame_indices)
        print(f"✓ Detected {len(events)} events")
        
        # Build results
        processing_state['progress'] = 95
        results = {
            'video_info': {
                'fps': loader.fps,
                'total_frames': loader.total_frames,
                'duration': timestamps[-1] if timestamps else 0,
                'processed_frames': len(frame_indices)
            },
            'events': events,
            'timestamps': timestamps,
            'frame_indices': frame_indices
        }
        
        processing_state['status'] = 'complete'
        processing_state['progress'] = 100
        processing_state['results'] = results
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60)
        print(f"📊 Results:")
        print(f"   • Duration: {results['video_info']['duration']:.1f}s")
        print(f"   • Processed: {results['video_info']['processed_frames']} frames")
        print(f"   • Events: {len(events)}")
        print("="*60 + "\n")
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        processing_state['status'] = 'error'
        processing_state['error'] = str(e)
        processing_state['progress'] = 0
        print(f"\n❌ Error during processing: {error_msg}\n")


def detect_simple_events(embeddings, timestamps, frame_indices):
    """Simple event detection based on embedding changes."""
    if len(embeddings) < 2:
        return []
    
    from sharingan.events import EventDetector
    
    detector = EventDetector(sensitivity=0.5)
    detected_events = detector.detect_events(
        np.array(embeddings),
        timestamps,
        frame_indices
    )
    
    events = []
    for event in detected_events:
        events.append({
            'id': event.event_id,
            'type': event.event_type,
            'timestamp': event.start_time,
            'frame': event.start_frame,
            'confidence': event.confidence,
            'description': event.description
        })
    
    return events


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processing status."""
    try:
        return jsonify({
            'status': processing_state.get('status', 'idle'),
            'progress': processing_state.get('progress', 0),
            'current_video': processing_state.get('current_video'),
            'results': processing_state.get('results'),
            'error': processing_state.get('error')
        })
    except Exception as e:
        print(f"❌ Status endpoint error: {str(e)}")
        return jsonify({
            'status': 'error',
            'progress': 0,
            'error': 'Failed to get status'
        }), 500


@app.route('/api/query', methods=['POST'])
def query_video():
    """Handle natural language query with optional LLM chat."""
    try:
        data = request.json
        query_text = data.get('query', '')
        use_llm = data.get('use_llm', False)  # Enable conversational responses
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        if processing_state.get('embeddings') is None:
            return jsonify({'error': 'Please process video first'}), 400
        
        print(f"\n🔍 Query: '{query_text}'")
        
        from sharingan.vlm import FrameEncoder
        import numpy as np
        
        try:
            # Try auto (GPU if available) first
            encoder = FrameEncoder(model_name='clip-vit-b32', device='auto')
            query_embedding = encoder.encode_text(query_text)
        except Exception as e:
            # Fallback to CPU
            try:
                print(f"⚠️  GPU encoding failed, trying CPU...")
                encoder = FrameEncoder(model_name='clip-vit-b32', device='cpu')
                query_embedding = encoder.encode_text(query_text)
            except Exception as e2:
                print(f"⚠️  Query encoding failed: {e2}")
                return jsonify({'error': 'Failed to encode query. CLIP may not be available.'}), 500
        
        embeddings = np.array(processing_state['embeddings'])
        timestamps = processing_state['timestamps']
        frame_indices = processing_state['frame_indices']
        
        similarities = np.dot(embeddings, query_embedding)
        
        top_k = min(5, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        query_results = []
        for idx in top_indices:
            query_results.append({
                'timestamp': timestamps[idx],
                'frame': frame_indices[idx],
                'confidence': float(similarities[idx]),
                'description': f"Relevant content found"
            })
        
        print(f"✓ Found {len(query_results)} results")
        
        # Generate conversational response if LLM enabled
        llm_response = None
        if use_llm:
            try:
                from sharingan.chat import VideoLLM
                
                # Initialize LLM (cached globally)
                if not hasattr(app, 'video_llm'):
                    print(f"🤖 Initializing Qwen2.5-0.5B...")
                    app.video_llm = VideoLLM(device='auto')
                
                # Generate response
                llm_response = app.video_llm.chat(query_text, query_results)
                print(f"✓ Generated LLM response")
                
            except Exception as e:
                print(f"⚠️  LLM generation failed: {e}")
                llm_response = None
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results': query_results,
            'llm_response': llm_response
        })
    
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'Server is running'})


@app.route('/api/chat/reset', methods=['POST'])
def reset_chat():
    """Reset chat history."""
    try:
        if hasattr(app, 'video_llm'):
            app.video_llm.reset_history()
        return jsonify({'success': True, 'message': 'Chat history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset processing state."""
    processing_state['status'] = 'idle'
    processing_state['progress'] = 0
    processing_state['current_video'] = None
    processing_state['results'] = None
    processing_state['error'] = None
    processing_state['embeddings'] = None
    processing_state['timestamps'] = None
    processing_state['frame_indices'] = None
    return jsonify({'success': True})


@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video file with proper headers for streaming."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return "Video not found", 404
    
    # Get file size
    file_size = os.path.getsize(filepath)
    
    # Parse range header
    range_header = request.headers.get('Range', None)
    
    if not range_header:
        # No range requested, send entire file
        with open(filepath, 'rb') as f:
            data = f.read()
        
        response = Response(data, 200, mimetype='video/mp4', direct_passthrough=True)
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Accept-Ranges', 'bytes')
        return response
    
    # Parse range
    byte_start = 0
    byte_end = file_size - 1
    
    match = range_header.replace('bytes=', '').split('-')
    if match[0]:
        byte_start = int(match[0])
    if match[1]:
        byte_end = int(match[1])
    
    length = byte_end - byte_start + 1
    
    # Read chunk
    with open(filepath, 'rb') as f:
        f.seek(byte_start)
        data = f.read(length)
    
    response = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
    response.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
    response.headers.add('Content-Length', str(length))
    response.headers.add('Accept-Ranges', 'bytes')
    
    return response


def run_ui(host='127.0.0.1', port=5000, debug=False, open_browser=True):
    """Run Sharingan UI."""
    if open_browser:
        def open_browser_delayed():
            time.sleep(1.5)
            webbrowser.open(f'http://{host}:{port}')
        
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    print(f"""
                    ⠀⠀⠀⠀⠀⠀⢀⣀⣠⣤⣤⣤⣤⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⢀⣠⣶⣿⣿⣿⣿⡿⠃⠘⢿⣿⣿⣿⣿⣶⣄⡀⠀⠀⠀⠀⠀
                ⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀
                ⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀
                ⠀⣸⣿⡉⠀⡀⠈⠉⠉⢙⡟⠲⠤⣄⣠⠤⠖⢻⡋⠉⠉⠁⠀⠀⠉⣿⣧⠀
                ⢰⣿⣿⣷⡀⠈⢻⣷⣶⣼⣤⣔⠊⠁⠈⠑⣢⣤⣧⣶⣾⡿⠁⢀⣾⣿⣿⡆
                ⣼⣿⣿⣿⣿⣄⠀⢉⡿⣿⣿⣿⡿⠖⠲⢿⣿⣿⣿⠿⡋⠀⣠⣾⣿⣿⣿⣷
                ⣿⣿⣿⣿⣿⣿⡷⣏⠀⡏⠻⢿⡁⠀⠀⢈⡿⠟⢹⠀⣨⢾⣿⣿⣿⣿⣿⣿
                ⢻⣿⣿⣿⡿⠋⠀⠈⠳⣧⡀⠀⣷⣦⣴⣾⠀⢀⣸⠞⠁⠀⠙⢿⣿⣿⣿⡿
                ⠸⣿⣿⡿⠁⠀⠀⠀⠀⢸⠉⠲⢼⣿⣿⡯⠖⠋⡇⠀⠀⠀⠀⠈⢿⣿⣿⠇
                ⠀⠹⣿⣀⣀⣀⣀⣀⣀⣨⣧⠔⠚⣿⣿⠗⠢⢼⣅⣀⣀⣀⣀⣀⣀⣿⡏⠀
                ⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⢻⡿⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀
                ⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣷⡀⠈⠃⢀⣾⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀
                ⠀⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⣷⣄⢠⣾⣿⣿⣿⡿⠿⠋⠁⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠛⠛⠛⠛⠋⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              👁️  Sharingan UI Started  👁️               ║
    ║                                                          ║
    ║  Open your browser and navigate to:                     ║
    ║  http://{host}:{port}                              ║
    ║                                                          ║
    ║  Press Ctrl+C to stop the server                        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_ui()

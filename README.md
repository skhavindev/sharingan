# 👁️ Sharingan - Semantic Video Understanding

**Sharingan** is a lightweight Python library for semantic video understanding with temporal reasoning. It combines vision-language models (CLIP, SmolVLM) with temporal analysis to understand video content at a deep semantic level.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎬 **Semantic Video Processing** - Understand video content beyond pixels
- 🔍 **Natural Language Queries** - Search videos using text descriptions
- 🤖 **AI Chat** - Conversational interface with Qwen2.5-0.5B
- ⚡ **Temporal Reasoning** - Cross-frame attention and memory tokens
- 🎯 **Event Detection** - Automatically identify key moments
- 💾 **Efficient Storage** - 130x compression with Int8 quantizations
- 🌐 **Web UI** - Beautiful Flask-based interface
- 🚀 **Fast Processing** - Batch processing and GPU acceleration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Sharingan Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Video Input                                                │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────┐                                           │
│  │ Frame Sampler│  (Adaptive FPS)                           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────┐                       │
│  │   Vision-Language Models         │                       │
│  │  ┌────────┐      ┌─────────────┐ │                       │
│  │  │  CLIP  │  or  │ SmolVLM-500M│ │                       │
│  │  └────────┘      └─────────────┘ │                       │
│  │   (Fast)         (Detailed)      │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────────────────┐                       │
│  │   Temporal Reasoning Engine      │                       │
│  │  • Cross-Frame Gating            │                       │
│  │  • Memory Tokens                 │                       │
│  │  • Temporal Attention            │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────────────────┐                       │
│  │   Embedding Storage (Int8)       │                       │
│  │   ~2.3MB for 5-min video         │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│             ├──────────┬──────────────┐                     │
│             ▼          ▼              ▼                     │
│      ┌──────────┐ ┌─────────┐  ┌──────────┐                 │
│      │  Events  │ │ Queries │  │ AI Chat  │                 │
│      │ Detector │ │  (CLIP) │  │ (Qwen2.5)│                 │
│      └──────────┘ └─────────┘  └──────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install sharingan-core

# Optional: For GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: For AI chat
pip install transformers bitsandbytes accelerate
```

### Basic Usage

```python
from sharingan import VideoProcessor

# Process a video
processor = VideoProcessor(
    vlm_model='clip',  # or 'smolvlm' for detailed descriptions
    device='auto'
)

results = processor.process('video.mp4')

# Query the video
matches = processor.query('person speaking')
for match in matches:
    print(f"Found at {match.timestamp}s - {match.confidence:.2%}")

# AI Chat (optional)
response = processor.chat('What happens in this video?')
print(response)
```

### Web UI

```bash
# Launch the web interface
python -m sharingan.cli ui

# Or programmatically
from sharingan.ui import run_ui
run_ui(port=5000, open_browser=True)
```

## 📖 Documentation

### Vision Models

**CLIP (Default)**
- Fast semantic embeddings
- Best for: Quick search, real-time processing
- Memory: ~400MB

**SmolVLM-500M**
- Detailed frame descriptions
- Best for: Rich understanding, detailed analysis
- Memory: ~538MB (8-bit quantized)

### Processing Options

```python
processor = VideoProcessor(
    vlm_model='clip',           # 'clip' or 'smolvlm'
    device='auto',              # 'cpu', 'cuda', or 'auto'
    target_fps=5.0,             # Frames per second to process
    enable_temporal=True,       # Temporal reasoning
    enable_tracking=False       # Entity tracking
)
```

### Query Options

```python
# Semantic search
results = processor.query(
    'person speaking',
    top_k=5
)

# AI chat (requires Qwen2.5)
response = processor.chat(
    'Describe the main events',
    use_llm=True
)
```

## 🎯 Use Cases

- **Video Search** - Find specific moments using natural language
- **Content Moderation** - Detect inappropriate content
- **Video Summarization** - Generate automatic summaries
- **Accessibility** - Create descriptions for visually impaired
- **Research** - Analyze video datasets at scale

## 🔧 Advanced Features

### Temporal Reasoning

Sharingan uses advanced temporal reasoning:
- **Cross-Frame Gating** - Learns which frames are important
- **Memory Tokens** - Maintains context across the video
- **Temporal Attention** - Understands relationships between frames

### Efficient Storage

Videos are compressed 130x using Int8 quantization:
- 5-minute video: ~2.3MB (vs 300MB raw)
- Fast cache loading
- No quality loss for search

### Event Detection

Automatically detects:
- Scene changes
- Motion patterns
- Content transitions

## 📊 Performance

| Model | Speed | Memory | Quality |
|-------|-------|--------|---------|
| CLIP | ⚡⚡⚡ | 400MB | Good |
| SmolVLM | ⚡⚡ | 538MB | Excellent |

*Tested on NVIDIA RTX 3060 (4GB VRAM)*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

Made with ☕
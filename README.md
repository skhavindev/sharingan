# 👁️ Sharingan - Semantic Video Understanding

<p align="center">
  <img src="https://media1.tenor.com/m/YeM3fMlamBoAAAAd/naruto.gif" alt="Sharingan GIF" style="width:100%; height:auto;"/>
</p>


**Sharingan** is a lightweight Python library for semantic video understanding with temporal reasoning. It combines vision-language models (CLIP, SmolVLM) with temporal analysis to understand video content at a deep semantic level.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

* 🎬 **Semantic Video Processing** – Understand video content beyond pixels
* 🔍 **Natural Language Queries** – Search videos using text descriptions
* 🤖 **AI Chat** – Conversational interface with Qwen2.5-0.5B
* ⚡ **Temporal Reasoning** – Cross-frame attention and memory tokens
* 🎯 **Event Detection** – Automatically identify key moments
* 💾 **Efficient Storage** – 130x compression with Int8 quantization
* 🌐 **Web UI** – Beautiful Flask-based interface
* 🚀 **Fast Processing** – Batch processing and GPU acceleration

---
You can read the [Author Note](https://github.com/skhavindev/sharingan/blob/master/author_note.md), check out the [Architecture](https://github.com/skhavindev/sharingan/blob/master/architecture.md), and see the [Contributing Guidelines](https://github.com/skhavindev/sharingan/blob/master/contributing.md) on GitHub.

---

## 🚀 Quick Start

### Installation

```bash
pip install sharingan-core

# Optional: GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: AI chat
pip install transformers bitsandbytes accelerate
```

### Basic Usage

```python
from sharingan import VideoProcessor

processor = VideoProcessor(
    vlm_model='clip',  # or 'smolvlm'
    device='auto'
)

results = processor.process('video.mp4')

matches = processor.query('person speaking')
for match in matches:
    print(f"Found at {match.timestamp}s - {match.confidence:.2%}")

response = processor.chat('What happens in this video?')
print(response)
```

### Web UI

```bash
python -m sharingan.cli ui
```

Or programmatically:

```python
from sharingan.ui import run_ui
run_ui(port=5000, open_browser=True)
```

---

## 📖 Documentation

**Vision Models**

* **CLIP** – Fast semantic embeddings; memory ~400MB
* **SmolVLM-500M** – Detailed frame descriptions; memory ~538MB (8-bit quantized)

**Processing Options**

```python
processor = VideoProcessor(
    vlm_model='clip',
    device='auto',
    target_fps=5.0,
    enable_temporal=True,
    enable_tracking=False
)
```

**Query Options**

```python
results = processor.query('person speaking', top_k=5)
response = processor.chat('Describe main events', use_llm=True)
```

---

## 🎯 Use Cases

* Video Search – Find moments using natural language
* Content Moderation – Detect inappropriate content
* Video Summarization – Auto summaries
* Accessibility – Descriptions for visually impaired
* Research – Analyze video datasets at scale

---

## 🔧 Advanced Features

**Temporal Reasoning**

* Cross-Frame Gating – Learns important frames
* Memory Tokens – Maintains context across video
* Temporal Attention – Understand relationships between frames

**Efficient Storage**

* 5-min video: ~2.3MB (vs 300MB raw)
* Fast cache loading
* Minimal quality loss for search

**Event Detection**

* Scene changes
* Motion patterns
* Content transitions

---

## 📊 Performance

| Model   | Speed | Memory | Quality   |
| ------- | ----- | ------ | --------- |
| CLIP    | ⚡⚡⚡   | 400MB  | Good      |
| SmolVLM | ⚡⚡    | 538MB  | Excellent |

*Tested on NVIDIA RTX 3060 (4GB VRAM)*

---

## 🤝 Contributing

Contributions welcome! Please submit a PR.

## 📄 License

MIT License – see LICENSE file.

## 🙏 Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
* [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 📧 Contact

Open an issue on GitHub for support.

---

Made with ☕ & ❤️


# Contributing to Sharingan

Thank you for your interest in contributing to Sharingan!

## Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sharingan.git
cd sharingan
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e ".[dev,all]"
```

4. **Install CLIP**
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Running Tests

```bash
python test_complete.py
```

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting

```bash
black sharingan/
flake8 sharingan/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Areas for Contribution

- 🎯 New vision models
- 🔄 Temporal reasoning modules
- 🎨 UI improvements
- 📚 Documentation
- 🐛 Bug fixes
- ⚡ Performance optimizations

## Questions?

Open an issue on GitHub!

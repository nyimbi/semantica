# Installation

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Basic Installation

Install Semantica from PyPI:

```bash
pip install semantica
```

## Verify Installation

Check that Semantica is installed correctly:

```bash
python -c "import semantica; print(semantica.__version__)"
```

You should see: `0.0.1`

## Development Installation

To install Semantica in development mode:

```bash
git clone https://github.com/Hawksight-AI/semantica.git
cd semantica
pip install -e .
```

## Optional Dependencies

Install optional features:

```bash
# GPU support
pip install semantica[gpu]

# Visualization
pip install semantica[viz]

# All LLM providers
pip install semantica[llm-all]

# Cloud integrations
pip install semantica[cloud]
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'semantica'`
- **Solution**: Make sure you've activated the correct Python environment

**Issue**: Installation fails
- **Solution**: Upgrade pip: `pip install --upgrade pip`

**Issue**: GPU dependencies fail
- **Solution**: Install CPU-only version first, then add GPU support

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration](configuration.md)
- [Examples](examples.md)


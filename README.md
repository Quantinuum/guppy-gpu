# guppy-gpu

Realtime GPU decoding with [Guppy](https://guppylang.org/) on Quantinuum hardware.

## Usage

See [`src/guppy_gpu/cudaq_qec.py`](./src/guppy_gpu/cudaq_qec.py).

## Installation

Install using `pip` in your Guppy project environment:

```sh
pip install git+https://github.com/quantinuum/guppy-gpu.git
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
# Create virtual environment and install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uvx ruff format
```

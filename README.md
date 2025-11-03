# guppy-gpu

GPU decoding with [Guppy](https://guppylang.org/).

## Usage

See `src/guppy_gpu/api.py`.

## Build and Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv build
```

Then `pip install` the `guppy_gpu-0.1.0-py3-none-any.whl` in your Guppy project
environment.

<!--
## Sample

`sample.py` includes a sample Guppy GPU program. You may generate compiled hugr
output using:

```sh
uv run sample.py > sample.hugr
```
-->

## Development

```sh
# Create virtual environment and install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uvx ruff format
```

# Gemini Code Assistant Context

This document provides a summary of the `ad-metrics` project to be used as instructional context for the Gemini Code Assistant.

## Project Overview

The `ad-metrics` project is a comprehensive Python library for evaluating autonomous driving perception and planning systems. It provides a wide range of metrics across several categories, including detection, tracking, trajectory prediction, localization, occupancy, planning, vector maps, and simulation quality.

The library is designed to be a standard tool for researchers and engineers working on autonomous driving systems, providing implementations of common metrics used in popular benchmarks like KITTI, nuScenes, Waymo, and nuPlan.

The project is a Python library built with `setuptools`. The main source code is located in the `admetrics` directory, with each sub-directory corresponding to a specific category of metrics. The library has the following dependencies:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib` (optional, for visualization)
- `open3d` (optional, for 3D visualization)

## Building and Running

### Installation

The project can be installed in editable mode using `pip`:

```bash
pip install -e .
```

To install the optional dependencies for visualization and development, you can use the following command:

```bash
pip install -e .[viz,dev]
```

### Testing

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
pytest
```

To run the tests with coverage, use the following command:

```bash
pytest --cov=admetrics
```

## Development Conventions

The project follows standard Python development conventions.

- **Formatting:** The code is formatted with `black`.
- **Linting:** The code is linted with `flake8`.
- **Typing:** The code is type-checked with `mypy`.
- **Documentation:** The code is well-documented with docstrings that follow the Google Python Style Guide. The docstrings include examples of how to use the functions.
- **Testing:** The project has a comprehensive test suite in the `tests` directory. Each metric category has its own test file.

## Key Files

- `README.md`: Provides a comprehensive overview of the project, including its features, installation instructions, and quick-start examples.
- `pyproject.toml`: Contains the project's metadata, dependencies, and build system configuration.
- `setup.py`: The `setuptools` build script for the project.
- `admetrics/`: The main package directory, containing the source code for all the metrics.
  - `admetrics/detection/`: Metrics for object detection, such as AP, mAP, and IoU.
  - `admetrics/tracking/`: Metrics for multi-object tracking, such as MOTA, HOTA, and IDF1.
  - `admetrics/planning/`: Metrics for end-to-end planning, such as L2 distance, collision rate, and comfort metrics.
- `tests/`: Contains the test suite for the project.
- `examples/`: Contains example scripts that demonstrate how to use the library.
- `docs/`: Contains the documentation for the project.

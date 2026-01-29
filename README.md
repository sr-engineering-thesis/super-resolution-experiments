# Super Resolution Experiments
This repository contains the code used for conducting experiments for Q2RTX domain. This includes training, inference, visualization and benchmarking.

## Project Structure

```
.
├── src/
│   ├── benchmark_plotter.py  # Plotting and visualization for benchmarks
│   ├── dataset.py             # Dataset loading and preprocessing
│   ├── logger.py              # Experiment logging utilities
│   ├── losses.py              # Loss functions for training
│   ├── metrics.py             # Evaluation metrics (PSNR, SSIM, etc.)
│   ├── trainer.py             # Training loop implementation
│   └── visualizer.py          # Image visualization tools
├── models/                    # Model architectures
├── configs/                   # Configuration files for experiments
├── experiments/               # Experiment scripts
├── checkpoints/               # Saved model checkpoints
├── benchmark.py               # Benchmark execution script
├── log_images.py              # Image logging script
├── constants.py               # Project constants
└── utils.py                   # General utility functions for dataset
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

## Configuration
The project uses Hydra for configuration management. Configuration files are located in the `configs/` directory. Parameters can be overriden via command line or by creating custom configuration files. 
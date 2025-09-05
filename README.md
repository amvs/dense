# Key Components
- `dense/` → Subpackage for PyTorch model definitions
- `training/` → Subpackage for training
- `scripts/` → Command-line entry point
- `experiments/` → Auto-created folder storing logs, saved model and configs for reproducibility
- `configs/` → Subpackage for configuration of hyperparameter

# Get Start

## Package Requirement

Ensure pytorch environment is working. And addtionally,
`conda install jupyter scikit-learn matplotlib`

## Install the packages to your environment

`pip install -e .`

## Run training

`python scripts/train.py --config configs/mnist.yaml`
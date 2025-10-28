## 🧑‍💻 Collaborative Workflow

For any task, follow these steps to ensure smooth collaboration and code quality:

1.  **Create an Issue:** Start by creating an **issue** for the task you'll be working on and assign it to yourself. This helps track progress and discussions.
2.  **Create a Branch:** When starting work, create a new **branch** from `main`. Name it descriptively (e.g., `feature/add-new-model` or `fix/bug-in-loader`).
    * *Tip:* Commit your changes frequently and push them to this branch.
3.  **Create a Pull Request (PR):** Once your work is complete, open a **Pull Request** (PR) targeting the `main` branch.
4.  **Review and Merge:** The programmer who *did not* write the code is responsible for **reviewing** the PR. The code should be reviewed and approved before it can be **merged** into `main`.
    * Bonus points if the person who writes the code includes some tests that the reviewer can run (Copilot can be very helpful for writing tests quickly).


# Key Components
- `dense/` → Subpackage for PyTorch model definitions of scattering transform
- `training/` → Subpackage for training
- `scripts/` → Command-line entry point
- `experiments/` → Auto-created folder storing logs, saved model and configs for reproducibility
- `configs/` → Subpackage for configuration of hyperparameter

# Support dataset
- MNIST
- curet

# Get Start

## Package Requirement

Ensure pytorch environment is working. And addtionally,
`conda install scikit-learn matplotlib pandas kagglehub`

## Install the packages to your environment

`pip install -e .`

## Run training

`python scripts/train.py --config configs/mnist.yaml`

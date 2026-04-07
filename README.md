# BlockTower

BlockTower is a neural physics simulation project for 3D block-tower dynamics.
This repository contains training pipelines for quaternion-based and Euler-angle-based models,
dataset documentation for raw trajectories.

## Repository Overview

- `models/`: Core neural simulators.
- `utils/`: Dataset loaders and shared utilities.
- `datasets/`: Raw dataset format documentation.
- `optuna/`: Hyperparameter search scripts.

## Key Entry Scripts

### Training

- `1scene_posnormed_train.py`: Main training entry for quaternion representation.
- `euler_1scene_posnormed_train.py`: Main training entry for Euler-angle representation.

### Model Implementations

- `models/neural_simulator.py`: Quaternion-based neural simulator.
- `models/euler_neural_simulator.py`: Euler-angle neural simulator.

### Hyperparameter Search

- `optuna/optuna_full_search_1scene.py`: Full search for 1-scene setting.
- `optuna/optuna_euler_1scene.py`: Euler 1-scene search.

### Visualization and Debug

- `visualize_animations.py`: Render and compare predicted vs. ground-truth trajectories.

## Data Documentation

- Raw simulation dataset specification:
	- `datasets/README_datasets.md`

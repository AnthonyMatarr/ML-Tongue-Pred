# ML-Tongue-Pred

## Description
This project implements Logistic Regression, Support Vector Classifier, LightGBM, Neural Network, and Stacked Generalization models to predict post-operative complications (Mortality, Bleeding, Aspiration, Surgical) in patients.

## Project layout
### Included directories
- notebooks/: end-to-end machine learning workflows (data cleaning, EDA, preprocessing, tuning, evaluation, and figure/table generation)

- src/: reusable functions shared across notebooks

### Omitted directories
- ```data/```
  - ```raw/```: Raw data files
  - ```processed/```:
    - Cleaned raw data files
    - Extracted outcome (target variable) data
- ```models/```: trained models for each outcome
- ```cal_models/```: calibrated models corresponding to those in ```models/```
- ```logs/```: logged data produced during model tuning
- ```results/```
  - ```tables/```:
    - summary & analysis
    - SHAP 
    - class report
    - p-value tables
  - ```figures/```:
    - ROC 
    - CM 
    - calibration 
    - DCA 
    - p-value heatmap 

## Installation

### Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

### Steps
1. Clone this repository with HTTPS or SSH:

- Using HTTPS (**recommemnded for simplicity**):
```
git clone https://github.com/AnthonyMatarr/ML-Tongue-Pred.git
```
- Using SSH:
```
git clone git@github.com:AnthonyMatarr/ML-Tongue-Pred.git
```
2. Navigate to project directory
```
cd ML-Tongue-Pred
```
3. Ensure data/object integrity and consistency
```
git fsck --full
```
5. Sync environment from pyproject.toml and uv.lock files:
```
uv sync --locked
```

## Usage
1. Adjust BASE_PATH in src/config.py to the absolute path to the root directory of ML-Tongue-Pred, for example:
```
BASE_PATH = Path("/Users/<user_name>/Downloads/ML-Tongue-Pred")
```
2. Run notebooks
  - Notebooks are numbered by stage, but assuming necessary data is available, can be run on their own
  - **NOTE**: Due to OS/architecture differences and solver choices, despite a consistent random state/seed used throughout the project, minor numerical deviations from the manuscript may occur in model tuning/training/evaluation and SHAP values

    
## Custom Modifications
- [MLstakit](https://github.com/Brritany/MLstatkit) was forked and slightly altered, with some code appended to `MLstatkit/metrics.py` to add functionality for ICI and Brier scores.
- This change should be consistent once the repo is cloned and ```uv sync --locked``` is run, however to view these changes or ensure their consistency, the forked repo can be found [here](https://github.com/AnthonyMatarr/MLstatkit), or in the project directory at:
```
/.venv/lib/python3.12/site-packages/MLstatkit/metrics.py
```
## License: MIT
- Code licensed under MIT
- No patient data are included

# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications After Glossectomy for Tongue Cancer


## Description
This project implements Logistic Regression, Support Vector Classifier, LightGBM, Neural Network, and Stacked Generalization models to predict post-operative complications (Mortality, Bleeding, Aspiration, Surgical) in glossectomy patients.

## Associated Risk Calculator
A web application was developed to deploy the calibrated Stacked Generalization models for Bleeding, Aspiration, and Surgical complications. Mortality was left out due to poor performance. This interface can be found [here](https://pro-tongue.streamlit.app/).
### Features
- Choose between any of the three available outcomes
- Input patient values into approriate fields (all fields used as input to the Stack model)
- Calculate results
  - Raw model output (risk probability)
  - Risk bin allocation
  - Percentile of model output relative to all patients included in the study (train + val + test)   

## Project layout
### Included directories
- ```notebooks/```: end-to-end machine learning workflows (data cleaning, EDA, preprocessing, tuning, evaluation, and figure/table generation)

- ```src/```: reusable functions shared accross the project
- `app/`: all code related to interface
  - `app.py`: source code for calculator interface
  - `.streamlit/`: Interface styling
  - `all_preds/`: directory containing predictions and truth values of all patients in cohort (train + val + test) for each outcome
  - `bin_thresholds/`: directory containing pre-defined risk-bin thresholds for each outcome
  - `deployment_prep.ipynb`: helper file to copy relevant models into `app/`

### Omitted directories
- ```data/```
  - ```raw/```: Raw data files
  - ```processed/```:
    - Cleaned raw data files
    - Extracted outcome (target variable) data
- ```models/```: trained models for each outcome (uncalibrated)
- ```cal_models/```: final calibrated models
- ```logs/```: logged data produced during model tuning
- ```results/```: figures and tables

## Installation

### Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

### Steps (run all in command prompt/terminal)
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
6. Paste your base path into the `BASE_PATH` variable in `src/config.py`. To get the path to your current working directory, run:
- Unix-based systems (Including MacOS)
```
pwd
```
- Windows OS
```
cd
```
## Troubleshooting

### macOS: LightGBM Import Error (libomp.dylib)
If you encounter an error when importing LightGBM on macOS:
```
OSError: Library not loaded: @rpath/libomp.dylib
```
**Solution**: Install the OpenMP library using Homebrew:
```
brew install libomp
```
Then restart your Python kernel/notebook. This issue may occur after macOS or Homebrew updates, even if LightGBM previously worked on your system.

## Usage

  - Notebooks are numbered by stage, but assuming necessary data is available, can be run on their own
  - **NOTE**: Due to OS/architecture differences and solver choices, despite a consistent random state/seed used throughout the project, minor numerical deviations from the manuscript may occur in model tuning/training/evaluation and SHAP values.
    - Values may also differ in imputation due to nature of Scikit-Learn's Iterative Imputer   

    
## Custom Modifications
- [MLstakit](https://github.com/Brritany/MLstatkit) was forked and slightly altered, with some code appended to `MLstatkit/metrics.py` `MLstatkit/ci.py`and  to add binning, ICI, and Brier functionality.
- This change should be consistent once the repo is cloned and `uv sync --locked` is run, however to view these changes or ensure their consistency, the forked repo can be found [here](https://github.com/AnthonyMatarr/MLstatkit), or in the project directory at:
```
/.venv/lib/python3.12/site-packages/MLstatkit/
```
## License: MIT
- Code licensed under MIT
- No patient data are included

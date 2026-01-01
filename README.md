# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications After Glossectomy for Tongue Cancer


## Description
This project implements Logistic Regression, Support Vector Classifier, LightGBM, Neural Network, and Stacked Generalization models to predict post-operative complications (Bleeding, Aspiration, Surgical, Mortality) in glossectomy patients found in the National Surgical Quality Improvement Program (NSQIP) dataset spanning 2008-2024.

## Associated Risk Calculator
A web application was developed to deploy calibrated SVC models for Surgical Wound Complications and Unplanned Re-operation, LR for Aspiration-related Complications, and LightGBM for Bleeding. Mortality was left out due to poor binning and discriminatory performance. The interface can be found [here](https://pro-tongue.streamlit.app/).
The app may also be run locally with command `uv run -m streamlit run app/base_app.py`, once all [Installation Steps](#installation) are completed.
### Features
- Select any available outcome from the sidebar
- Input patient values into appropriate fields (all features used as model inputs)
- View results:
  - Risk bin allocation
  - Feature contribution via regularized SHAP explanation values
  - Calibrated risk probability
  - Percentile of model output relative to all test cohort patients
  - View imputed numerical values as appropriate

## Project layout
### Included directories
- ```notebooks/```: end-to-end machine learning workflows 
  - data cleaning, preprocessing, tuning, training, evaluation, feature importance, and figure/table generation
  - assumes access to files in `data/` and `models/`

- ```src/```: reusable functions shared across the project
- `app/`: all code related to interface
  - `base_app.py`: source code for calculator interface
  - `.streamlit/`: Interface styling
  - `deployment_prep.ipynb`: helper file to copy relevant data into `app/`
  - `display_functions.py`: helper file containing modules displayed on interface

### Omitted directories
- ```data/```: raw + processed data, preprocessing pipelines
- ```models/```: raw trained and calibrated models for each outcome
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
4. Sync environment from pyproject.toml and uv.lock files:
```
uv sync --locked
```
5. Paste your base path into the `BASE_PATH` variable in `src/config.py`. To get the path to your current working directory, run:
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

    
## Custom Modifications
- [MLstakit](https://github.com/Brritany/MLstatkit) was forked and slightly altered, with some code appended to `MLstatkit/metrics.py` and `MLstatkit/ci.py` to add bin event rate, ICI, and Brier functionality.
- This change should be consistent once the repo is cloned and `uv sync --locked` is run, however to view these changes or ensure their consistency, the forked repo can be found [here](https://github.com/AnthonyMatarr/MLstatkit), or in the project directory at:
```
/.venv/lib/python3.12/site-packages/MLstatkit/
```
## License: MIT
- Code licensed under MIT
- No patient data are included
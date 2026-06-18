# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications After Glossectomy for Tongue Cancer

[![DOI](https://zenodo.org/badge/1126361921.svg)](https://doi.org/10.5281/zenodo.20750588)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
This project implements Logistic Regression, Support Vector Classifier, LightGBM, XGBoost, Neural Network, and Stacked Generalization models to predict post-operative complications (Bleeding, Pneumonia, Unplanned Reoperation, Surgical Site Infection, Serious, and Any complications) in glossectomy patients found in the American College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) dataset. Models are developed with data spanning 2008-2023 and validated on a held-out 2024 cohort.

## Associated Risk Calculator
A web application was developed deploying **LightGBM** models (Unplanned Reoperation, Surgical Site Infection, Pneumonia) and **XGBoost** models (Bleeding, Serious, Any) to stratify an input patient into one of **Very Low, Low, Moderate, or Very High** risk bins based on calibrated probability output. The interface can be found [here](https://pro-tongue.streamlit.app/).

Once all [Installation Steps](#installation) are completed, the app may also be run locally with command `uv run -m streamlit run app/base_app.py`.

### Features
- Select any available outcome from the sidebar
- Enter patient values into appropriate fields
- Results include:
  - **Risk stratification** into one of Very Low, Low, Moderate, or Very High risk
  - **Feature contribution** via regularized SHAP explanation values
  - **Calibrated risk** probability
  - **Percentile ranking** of model output relative to 2024 cohort
  - **Imputed numerical values** displayed as appropriate

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
  - `utils.py`: contains helper functions for data handling
  - `shap_utils.py`: contains helper functions for SHAP analysis

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

- Using HTTPS (**recommended for simplicity**):
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
5. Set the `BASE_PATH` variable in `src/config.py` to your working directory path. To find it, run:
- **maxOS/Linux:** `pwd`
- **Windows:** `cd`

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
- This change should be consistent once this repo is cloned and `uv sync --locked` is run, however to view these changes or ensure their consistency, the forked repo can be found [here](https://github.com/AnthonyMatarr/MLstatkit), or in the project directory at:
```
/.venv/lib/python3.12/site-packages/MLstatkit/
```
## License: MIT
- Code licensed under MIT
- No patient data are included

## Citation
If you use this code, please cite the associated manuscript and software:

**Manuscript:** Matar DY, Matar AY, Nimbalkar A, et al. Artificial Intelligence–Based Risk Prediction Models for Complications After Tongue Cancer Surgery. *JAMA Otolaryngol Head Neck Surg.* Published online June 18, 2026. doi:10.1001/jamaoto.2026.1453

**Software:** Matar AY (2026). ML-Tongue-Pred (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.20750589

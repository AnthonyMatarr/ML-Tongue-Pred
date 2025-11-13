from src.config import SEED, BASE_PATH

from operator import xor
import warnings
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from MLstatkit import Bootstrapping
import pandas as pd


########################################## Helper Functions ##########################################
def get_percentile_range_thresholds(y_proba, n_bins=4, lower=1, upper=99):
    """
    Returns thresholds using the [lower, upper] percentiles of y_proba, dividing the interval into (as close as possible) equal-width bins.
    """
    # Compute robust min/max using percentiles
    lo = np.percentile(y_proba, lower)
    hi = np.percentile(y_proba, upper)
    edges = np.linspace(lo, hi, n_bins + 1)
    # Internal edges only (drop low/high)
    thresholds = edges[1:-1]
    return thresholds  # 1D array, length n_bins-1


def get_logspace_thresholds(y_proba, n_bins=4, lower=1e-5, upper=None):
    """
    Returns thresholds spaced evenly on a log scale within [lower, upper], gracefully handling zero predictions.
    """
    if upper is None:
        upper = np.percentile(y_proba, 99)  # or use max(y_proba)
    # Avoid log(0) by setting very small lower bound
    lo = max(lower, np.min(y_proba[y_proba > 0]))
    hi = upper
    # Make log-spaced edges
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    thresholds = edges[1:-1]
    return thresholds


def plot_risk_bar_dot(y_true, y_proba, thresholds, ax=None):
    """
    Allocates predictions to bins, calculates outcome rate & mean prediction for each bin,
    and makes a bar graph with risk curve overlay.
    """
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    n_bins = len(thresholds) + 1
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1

    event_rates = []
    mean_preds = []
    counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        n = mask.sum()
        counts.append(n)
        if n == 0:
            event_rates.append(np.nan)
            mean_preds.append(np.nan)
        else:
            event_rates.append(y_true[mask].mean())
            mean_preds.append(y_proba[mask].mean())

    # Label bins as "Bin 0", "Bin 1", ...
    bins_labels = [f"Bin {i}" for i in range(n_bins)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_bins), event_rates, color="C0", alpha=0.7, label="Event Rate")
    ax.plot(range(n_bins), mean_preds, "o-", color="C1", label="Avg. Predicted Risk")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_ylabel("Fraction With Outcome / Mean Prediction")
    ax.set_xlabel("Risk Bin")
    ax.legend()

    # n=XXXX at bottom of bar
    for i, n in enumerate(counts):
        ax.text(
            i,
            0.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="k",
        )
    plt.tight_layout()
    return ax


def get_cm(
    model_name,
    outcome_name,
    data_type,
    y_true,
    y_pred,
    show_output=False,
    results_path=None,
):
    """
    Generate confusion matrix and (optionally) export it

    Parameters
    ---------
    model_name: str
        Specify name of model used to generate y_pred
    outcome_name: str
        Specify name of outcome that is being predicted
    data_type: str
        Specify subset of df
        Usually one of [train, val, test]
    y_true: numpy.ndarray
        Array containing true binary outcome/target labels
    y_pred: numpy.ndarray
        Array containing predicted binary outcome/target labels
    show_output: Optional boolean; defaults to False
        Boolean flag specifying whether to display generated confusion matrix
    results_path: Optional pathlib.Path; defaults to None
        Path to directory where results are stored
        If left None, will not write to memory
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{model_name}: {data_type}")
    if results_path:
        cm_path = (
            results_path
            / "figures"
            / "CM"
            / outcome_name
            / f"{model_name}_{data_type}_CM.pdf"
        )
        if cm_path.exists():
            warnings.warn(f"Over-writing confusion matrix at path: {cm_path}")
            cm_path.unlink()
        cm_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(cm_path, bbox_inches="tight")
    if show_output:
        plt.show()
    else:
        plt.close()


def get_discrimination_str(
    *_, y_true, y_proba, metric_str, threshold, n_bootstraps=5000, random_state=SEED
):
    """
    Calculate a value and 95% CI for a given metric using MLStakit.Bootstrapping

    Parameters
    ----------
    y_true: numpy.ndarray
        True binary class labels
    y_proba: numpy.ndarray
        Continues predicted probabilities
    metric_str: str
        Specify the metric type to get values for
    threshold: float
        Threshold value to use for converting probabilities into hard labels
    n_bootstraps: Optional int; defaults to 5000
        Number of iterations to run bootstrap method for
    random_state: Optional int; defaults to SEED from src.config
        Controls determinism

    Returns
    -------
    final_str: String of format
        '<metric_val> (<ci_lower>, <ci_upper>)'
    Raises
    ------
    ValueError:
        -If positional arguments are passed
        -If an unaccepted str type is passed. Must be one of:
            'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'average_precision', 'pr_auc', 'ici', 'brier'
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    metric_val, ci_lower, ci_upper = Bootstrapping(
        y_true,
        y_proba,
        metric_str=metric_str,
        n_bootstraps=n_bootstraps,
        confidence_level=0.95,
        threshold=threshold,
        random_state=random_state,
    )
    final_str = f"{metric_val:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
    return final_str


def plot_ROC(y_true, y_proba, data_type, n_bootstraps=5000, seed=SEED):
    """
    Plot ROC curve, get AUROC w/ CIs, determine threshold for hard predictions
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="roc_auc",
        n_bootstraps=n_bootstraps,
    )
    auc_string = f"{auc:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    model_score = f"AUROC = {auc_string}"

    ## Youden's J to determine threshold ##
    pr_dif = tpr - fpr
    optimal_idx = np.argmax(pr_dif)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr, tpr, lw=4, label=f"{data_type} {model_score}")
    return auc_string, optimal_threshold


########################################## Main function ##########################################
def evaluate_models(
    *_,
    model_dict,
    outcome_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
    n_bins=4,
    results_path=None,
    threshold_str="val",
    n_bootstraps=5000,
    show_cm=False,
    show_roc=False,
    show_cal=False,
):
    """
    Given an outcome, for each model:
        generate probabilities, plot ROC, get AUROC, determine prediction threshold,
        get discrimination metrics (f1, accuracy, recall, precision, brier, ici),
        plot calibration curves, and get calibration metrics

    Parameters
    ----------
    model_dict: dict
        Dictionary containing mapping model names to trained models
        Format:
            {
                <model_name> str: model sklearn.calibration.CalibratedClassifierCV
            }
    outcome_name: str
        String specifying outcome whose models are being evaluated
    X_train: pandas Dataframe
        Dataframe containing tabular training data (excluding target variable)
    y_train: numpy.ndarray
        Array containing training binary outcome/target variable
    X_val: pd.Dataframe
        Dataframe containing tabular validation data (excluding target variable)
    y_val: numpy.ndarray
        Array containing validation binary outcome/target variable
    X_test: Optional pd.Dataframe; defaults to None
        Dataframe containing tabular testing data (excluding target variable)
        If left None, will not evaluate test data
    y_test: Optional numpy.ndarray; defaults to None
        Array containing testing binary outcome/target variable
        If left None, will not evaluate test data
    results_path: Optional pathlib.Path; defaults to None
        Path to where results are saved
        If left None will not save results
    threshold_str: Optional str; defaults to "val"
        Determines which set to use to determine hard prediction threshold
        One of ['train', 'val']
    n_bootstraps: Optional int; defaults to 5000
        Number of bootstraps for metric CI calculation
    show_cm: Optional boolean; defaults to False
        Boolean flag indicating whether to output confusion matrices
    show_roc: Optional boolean; defaults to False
        Boolean flag indicating whether to output ROC plots
    show_cal: Optional boolean; defaults to False
        Boolean flag indicating whether to output calibration curve plots

    Returns
    -------
    class_report_dict: dict
        Dictionary containing metrics for each of train, val, and test sets
        Format:
            {
                'train': {
                    <model_name> str: {
                        <metric_name> str: <metric_val> str or float
                    }
                },
                'val': {
                    <model_name> str: {
                        <metric_name> str: <metric_val> str or float
                    }
                }
                'test': {
                    <model_name> str: {
                        <metric_name> str: <metric_val> str or float
                    }
                }
            }
    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if xor(X_test is None, y_test is None):
        raise ValueError(
            "One of X_test or y_test is None while the other is not. These arguments much match!"
        )
    CLASS_REPORT_DICT = {"train": {}, "val": {}, "test": {}}
    for model_name, model in model_dict.items():
        print(f'{"-"*20} Model: {model_name} {"-"*20}')
        ########## Get probabilities  ###########
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]
        if X_test is not None:
            y_proba_test = model.predict_proba(X_test)[:, 1]
        #################################################################################################################
        ############################## ROC Plot + AUROC and get threshold for predictions ###############################
        #################################################################################################################
        plt.figure(figsize=(12, 8))
        plt.plot(
            [0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier"
        )
        train_roc_str, train_estimated_threshold = plot_ROC(
            y_train, y_proba_train, "Training", n_bootstraps=n_bootstraps
        )
        val_roc_str, val_estimated_threshold = plot_ROC(
            y_val, y_proba_val, "Validation", n_bootstraps=n_bootstraps
        )
        if X_test is not None:
            test_roc_str, _ = plot_ROC(
                y_test, y_proba_test, "Testing", n_bootstraps=n_bootstraps
            )
        ###### Determine threshold ######
        if threshold_str == "val":
            print(
                f"Using threshold determined by validation set: ~{val_estimated_threshold:.3f}"
            )
            threshold = val_estimated_threshold
        elif threshold_str == "train":
            print(
                f"Using threshold determined by train set: ~{train_estimated_threshold}"
            )
            threshold = train_estimated_threshold
        else:
            warnings.warn(
                f'Invalid input to "<threshold_str>" provided. Needs to be one of {"val", "train"}, got {threshold_str} instead. Using 0.5 as a threshold.'
            )
            threshold = 0.5
        #### Plot ####
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=21, fontweight=550)
        plt.ylabel("True Positive Rate", fontsize=21, fontweight=550)
        plt.tick_params(axis="both", which="major", labelsize=15)
        plt.title(f"{model_name} ROC", fontweight="semibold", fontsize=25)
        plt.legend(loc="lower right", prop={"size": 19, "weight": 550})
        if results_path:
            roc_path = (
                Path(results_path)
                / "figures"
                / "ROC"
                / outcome_name
                / f"{model_name}_ROC.pdf"
            )
            if roc_path.exists():
                warnings.warn(f"Over-writing roc-curve at path {roc_path}")
                roc_path.unlink()
            roc_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(roc_path, bbox_inches="tight")
        if show_roc:
            plt.show()
        else:
            plt.close()
        #################################################################################################################
        ############################################## Risk Bins ########################################################
        #################################################################################################################
        # ================== GET THRESHOLDS ===================
        train_val_probs = np.concatenate([y_proba_train, y_proba_val])
        # thresholds = get_percentile_range_thresholds(
        #    train_val_probs, n_bins=n_bins
        # )  # linear-scale
        thresholds = get_logspace_thresholds(
            train_val_probs, n_bins=n_bins
        )  # log-scale
        if results_path:
            bins_path = BASE_PATH / "app" / "bin_thresholds" / f"{outcome_name}.npz"
            if bins_path.exists():
                print(f"Over-writing bin data at path {bins_path}")
                bins_path.unlink()
            bins_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                bins_path,
                thresholds=thresholds,
            )
        # ================== PLOT RISK BARS ===================
        if X_test is not None:
            ax = plot_risk_bar_dot(y_test, y_proba_test, thresholds)
            plt.title(f"{model_name} Test Risk Stratification: {outcome_name}")
            if results_path:
                bin_plot_path = (
                    results_path
                    / "figures"
                    / "risk_bins"
                    / outcome_name
                    / f"{model_name}.pdf"
                )
                if bin_plot_path.exists():
                    print(f"Over-writing bin plot at path {bin_plot_path}")
                bin_plot_path.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(bin_plot_path, bbox_inches="tight")
            if show_cal:
                plt.show()
            else:
                plt.close()
        #################################################################################################################
        ########################################### All predictions #####################################################
        #################################################################################################################
        # file path
        all_pred_path = BASE_PATH / "app" / "all_preds" / f"{outcome_name}.parquet"
        if all_pred_path.exists():
            print(f"Over-writing all preds at path {all_pred_path}")
        all_pred_path.parent.mkdir(exist_ok=True, parents=True)
        # get preds
        all_probs = np.concatenate([y_proba_train, y_proba_val, y_proba_test])
        all_labels = np.concatenate([y_train, y_val, y_test])  # type: ignore
        all_predictions = pd.DataFrame({"prob": all_probs, "label": all_labels})
        all_predictions.to_parquet(all_pred_path)
        #################################################################################################################
        ########################################### Get discrimination metrics ##########################################
        #################################################################################################################
        metrics_strs = ["f1", "accuracy", "recall", "precision", "brier", "ici"]
        ########## Get predictions  ###########
        y_pred_train = (y_proba_train >= threshold).astype(int)
        y_pred_val = (y_proba_val >= threshold).astype(int)
        if X_test is not None:
            y_pred_test = (y_proba_test >= threshold).astype(int)  # type: ignore
        ##Train
        get_cm(
            model_name,
            outcome_name,
            "Train",
            y_train,
            y_pred_train,
            show_cm,
            results_path=results_path,
        )

        CLASS_REPORT_DICT["train"][model_name] = {
            "AUROC (95% CI)": train_roc_str,
            "Threshold": round(threshold, 3),
        }
        for metric_str in metrics_strs:
            CLASS_REPORT_DICT["train"][model_name][metric_str] = get_discrimination_str(
                y_true=y_train,
                y_proba=y_proba_train,
                metric_str=metric_str,
                threshold=threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
            )
        ##Val
        get_cm(
            model_name,
            outcome_name,
            "Validation",
            y_val,
            y_pred_val,
            show_cm,
            results_path=results_path,
        )
        CLASS_REPORT_DICT["val"][model_name] = {
            "AUROC (95% CI)": val_roc_str,
            "Threshold": round(threshold, 3),
        }
        for metric_str in metrics_strs:
            CLASS_REPORT_DICT["val"][model_name][metric_str] = get_discrimination_str(
                y_true=y_val,
                y_proba=y_proba_val,
                metric_str=metric_str,
                threshold=threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
            )
        ##Test
        if X_test is not None:
            get_cm(
                model_name,
                outcome_name,
                "Test",
                y_test,
                y_pred_test,
                show_cm,
                results_path=results_path,
            )
            CLASS_REPORT_DICT["test"][model_name] = {
                "AUROC (95% CI)": test_roc_str,
                "Threshold": round(threshold, 3),
            }
            for metric_str in metrics_strs:
                CLASS_REPORT_DICT["test"][model_name][metric_str] = (
                    get_discrimination_str(
                        y_true=y_test,
                        y_proba=y_proba_test,
                        metric_str=metric_str,
                        threshold=threshold,
                        n_bootstraps=n_bootstraps,
                        random_state=SEED,
                    )
                )
    return CLASS_REPORT_DICT

from src.config import SEED, DEVICE
from src.data_utils import get_data, get_models
from src.nn_models import load_nn_clf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from MLstatkit import Bootstrapping
import argparse
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import json

BIN_NAME_DICT = {
    3: ["Low", "Medium", "High"],
    4: ["Very Low", "Low", "Medium", "High"],
    5: ["Very Low", "Low", "Medium", "High", "Very High"],
}

## y_max (depends on outcome)
BIN_MAX_DICT = {
    "ssi": 0.55,
    "serious": 0.6,
    "any": 0.6,
    "pneumo": 1,  # 0.25
    "bleed": 0.8,
    "reoper": 0.4,
}


class NumpyEncoder(json.JSONEncoder):
    """
    Helper for bin dict export
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_results(export_path, data_to_export):
    """
    Save different types of results.

    Params
    -----
    export_path: Pathlib.Path
        File path where results will be saved
        NOTE: If ending in `.npz`, assumes saving bin_thresholds
        NOTE: If ending in `.pdf`, assumes saving figure/plot
        NOTE: If ending in `.parquet`, assumes saving pandas DF
    """
    if export_path.exists():
        print(f"Over-writing at path {export_path}")
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    ## Dynamic saving
    suffix = export_path.suffix
    if suffix == ".npz":  # bin thresholds
        np.savez(
            export_path,
            thresholds=data_to_export,
        )
    elif suffix == ".pdf":  # plot/image
        # data_to_export is `ax` here
        data_to_export.figure.savefig(export_path, bbox_inches="tight")
        plt.close(data_to_export.figure)
    elif suffix == ".parquet":  # pandas DF
        data_to_export.to_parquet(export_path)
    elif suffix == ".json":  # dict--> json
        # only bin dict needs NumpyEncoder, but doesnt hurt to apply to class_report_dict
        with open(export_path, "w") as f:
            json.dump(data_to_export, f, indent=4, cls=NumpyEncoder)
    else:
        raise ValueError(f"Unrecognized suffix in export_path: {export_path}")


########################################################################
############################## BIN HELPERS #############################
########################################################################
def get_logspace_thresholds(y_proba, n_bins, lower=1e-5, upper=None):
    """
    Returns thresholds spaced evenly on a log scale within [lower, upper].
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


def get_bin_metrics(y_true, y_proba, thresholds, n_bootstraps, n_bins):
    """ """
    bin_report_dict = {}
    bin_names = BIN_NAME_DICT[n_bins]
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1
    tot_n = len(y_true)
    tot_n_pos = np.sum(y_true)
    tot_event_rate = tot_n_pos / tot_n
    for b in range(n_bins):  # 4 bins
        ## Get labels + probs of allocated to this bin
        mask = bin_indices == b
        n = mask.sum()
        in_bin_labels = y_true[mask]
        in_bin_probs = y_proba[mask]
        # ================= Populate bin dict =================
        bin_name = bin_names[b]
        bin_report_dict[bin_name] = {}
        ## Total patients in bin (% of test cohort)
        bin_report_dict[bin_name]["n_perc"] = {"n": n, "perc": n / tot_n}
        ## % of all pos patients in this bin
        in_bin_n_pos = np.sum(in_bin_labels)
        # n_perc_all_pos
        n_perc_pos = in_bin_n_pos / tot_n_pos if tot_n_pos > 0 else np.nan
        bin_report_dict[bin_name]["perc_all_pos"] = {
            "n": in_bin_n_pos,
            "perc": n_perc_pos,
        }
        ## Event rate w/ CIs
        # Check if bin has both classes (required for bootstrap)
        n_unique_classes = len(np.unique(in_bin_labels))

        if n > 0 and n_unique_classes > 1:
            try:
                event_rate_boot, ci_lower, ci_upper = Bootstrapping(
                    in_bin_labels,
                    in_bin_probs,  # this not used but need to pass
                    metric_str="event_rate",
                    n_bootstraps=n_bootstraps,
                    random_state=SEED,
                    show_progress=False,
                )
            except RuntimeError:
                # Fallback if bootstrap fails
                event_rate_boot = in_bin_labels.mean()
                ci_lower = np.nan
                ci_upper = np.nan
        else:
            # Not enough data or only one class
            event_rate_boot = in_bin_labels.mean() if n > 0 else np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        event_rate = in_bin_labels.mean()
        bin_report_dict[bin_name]["event_rate_w_CIs"] = {
            "event_rate": event_rate,
            "lower_CI": ci_lower,
            "upper_CI": ci_upper,
            "event_rate_boot": event_rate_boot,
        }
        ## Lift
        bin_report_dict[bin_name]["lift"] = event_rate / tot_event_rate
        ## thresholds
        if b == 0:
            threshold_str = f"[0%, {thresholds[0]:.2%})"
        elif b == n_bins - 1:
            threshold_str = f"[{thresholds[-1]:.2%}, 100%]"
        else:
            threshold_str = f"[{thresholds[b-1]:.2%}, {thresholds[b]:.2%})"
        bin_report_dict[bin_name]["thresholds"] = threshold_str
        ## mean model output
        bin_report_dict[bin_name]["mean_model_output"] = in_bin_probs.mean()
    return bin_report_dict


def plot_risk_bar_dot(bin_report_dict, n_bins, ax=None, y_max=1.0):
    """
    Create risk stratification plot with bar graph showing observed event rates and overlaid mean predictions per bin.
    """
    ## Label bins w/ thresholds
    event_rates = []
    bins_labels = []
    mean_preds = []
    counts = []
    bin_names = BIN_NAME_DICT[n_bins]
    for i in range(n_bins):
        bin_name = bin_names[i]
        cur_dict = bin_report_dict[bin_name]
        threshold_str = cur_dict["thresholds"]
        bins_labels.append(f"{bin_name}\n{threshold_str}")
        event_rates.append(cur_dict["event_rate_w_CIs"]["event_rate"])
        mean_preds.append(cur_dict["mean_model_output"])
        counts.append(cur_dict["n_perc"]["n"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_bins), event_rates, color="C0", alpha=0.7, label="Event Rate")
    ax.plot(range(n_bins), mean_preds, "o-", color="C1", label="Avg. Predicted Risk")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max + (y_max / 10), 5))
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
    ax.figure.tight_layout()
    return ax


########################################################################
######################## DISCRIMINATION HELPERS ########################
########################################################################


def get_roc_stats(y_true, y_proba, n_bootstraps=5000, seed=SEED, show_progress=False):
    """
    Get ROC metrics + elements for plotting
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    auc, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="roc_auc",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )

    ## Youden's J to determine threshold for binary classes ##
    pr_dif = tpr - fpr
    optimal_idx = np.argmax(pr_dif)
    optimal_threshold = thresholds[optimal_idx]

    auc_string = f"{auc:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "auc_string": auc_string,
        "optimal_threshold": optimal_threshold,
    }


def get_pr_stats(y_true, y_score, n_bootstraps=5000, seed=SEED, show_progress=False):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    ap, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_score,
        random_state=seed,
        metric_str="average_precision",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )

    ap_string = f"{ap:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    return {
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "ap_string": ap_string,
    }


def get_cm(
    model_name,
    outcome_name,
    data_type,
    y_true,
    y_pred,
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
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(f"{model_name}: {data_type}\n{outcome_name}")
    return ax


def get_discrimination_str(
    *_,
    y_true,
    y_proba,
    metric_str,
    threshold,
    n_bootstraps=5000,
    random_state=SEED,
    bin_thresholds_for_ici=None,
    show_progress=False,
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
    if metric_str == "ici":
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            bin_thresholds=bin_thresholds_for_ici,
            show_progress=show_progress,
        )
    else:
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            show_progress=show_progress,
        )
    final_str = f"{metric_val:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
    return final_str


########################################################################
############################ MAIN FUNCTION ############################
########################################################################


def eval_trained_models(
    *_,
    outcome_name,
    data_imp_dir,
    model_abrv_list,
    model_imp_dir,
    app_result_dir,
    n_bins,
    result_dir,
    n_bootstraps,
    metrics_str_list,
    threshold_str="val",
    eval_calibrated=True,
    show_progress=False,
):
    """
    outcome_name: str
        Name of current outcome whose models are being evaluated
    data_imp_dir: pathlib.Path
        Directory of data containing train, val, and test data
    model_abrv_list: list[str]
        List of model abbreviations to be imported for evaluated
    model_imp_dir: pathlib.Path
        Directory of models to be evaluated
    app_result_dir: pathlib.Path
        Where to export results soley used for app
    n_bins: int
        Number of bins to generate for risk.
    result_dir: pathlib.Path
        General directory for all results
    n_bootstraps: int
        Number of bootstraps to run for metric CIs
    metrics_str_list: list[str]
        List of metrics to get (must work with MLStatKit + ICI/Brier)
    threshold_str: Optional str; default: "val"
        Dataset used to determine binary cutoff probability
        One of ['train', 'val']
    show_progress: Optional bool; default: False
        If true, show bootstrapping progress
    eval_calibrated: Optional bool; default: False
        If true, evaluate calibrated models, otherwise raw
        NOTE: If raw, NN needs to be imported seperately
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    BIN_REPORT_DICT = {}
    CLASS_REPORT_DICT = {"train": {}, "val": {}, "test": {}}
    # ============================> Prep data/models
    ## Get data
    data_dict = get_data(
        outcome_folder=f"outcome_{outcome_name}", file_dir=data_imp_dir
    )
    X_train, y_train = data_dict["X_train"], data_dict["y_train"].values.ravel()
    X_val, y_val = data_dict["X_val"], data_dict["y_val"].values.ravel()
    X_test, y_test = data_dict["X_test"], data_dict["y_test"].values.ravel()
    ## Get models
    if (not eval_calibrated) and ("nn" in model_abrv_list):
        ## Import NN seperately if not calibrated (pytorch file)
        temp_model_list = [abrv for abrv in model_abrv_list if abrv != "nn"]
        model_dict = get_models(temp_model_list, outcome_name, file_dir=model_imp_dir)
        nn_in_dim = X_train.shape[1]
        nn_dir = model_imp_dir / outcome_name / "nn.pt"
        model_dict["nn"] = load_nn_clf(
            data_path=nn_dir, in_dim=nn_in_dim, device=DEVICE
        )
        print("Loaded NN seperately!")
    else:
        # Otherwise import joblib files for all models normally
        model_dict = get_models(model_abrv_list, outcome_name, file_dir=model_imp_dir)
        print("Loading all models from joblib!!")
    # ============================> FOR EACH MODEL
    for model_name, model in model_dict.items():
        print(f"Working on model: {model_name}...")
        ## Get preds
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        ################################################################################################
        ##################################### Risk Bins ################################################
        ################################################################################################
        # ============================> 1) Risk Bin Calculation
        # Concat probs used to fit risk-bin thresholds
        train_val_probs = np.concatenate([y_proba_train, y_proba_val])
        bin_thresholds = get_logspace_thresholds(
            y_proba=train_val_probs,
            n_bins=n_bins,
        )
        # Export
        bin_path = (
            app_result_dir / "bin_thresholds" / outcome_name / f"{model_name}.npz"
        )
        export_results(bin_path, bin_thresholds)
        # ============================> 2) Risk Bin Metrics
        BIN_REPORT_DICT[model_name] = get_bin_metrics(
            y_true=y_test,
            y_proba=y_proba_test,
            thresholds=bin_thresholds,
            n_bootstraps=n_bootstraps,
            n_bins=n_bins,
        )
        # ============================> 3) Plot Risk Bins
        y_max = BIN_MAX_DICT[outcome_name]
        ax = plot_risk_bar_dot(BIN_REPORT_DICT[model_name], n_bins=n_bins, y_max=y_max)
        ax.set_title(f"{model_name} Test Risk Stratification: {outcome_name}")
        # Export
        bin_plot_path = (
            result_dir / "figures" / "risk_bins" / outcome_name / f"{model_name}.pdf"
        )
        export_results(export_path=bin_plot_path, data_to_export=ax)
        # ============================> 4) Export test preds (for app)
        all_pred_path = (
            app_result_dir / "all_preds" / outcome_name / f"{model_name}.parquet"
        )
        all_predictions = pd.DataFrame({"prob": y_proba_test, "label": y_test})
        export_results(all_pred_path, all_predictions)
        print("\t Risk bins done!")
        ################################################################################################
        #################################### Trad Metrics ##############################################
        ################################################################################################
        ## Add train, val, & test curves on same plot
        split_data = {
            "train": {"y_true": y_train, "y_proba": y_proba_train},
            "val": {"y_true": y_val, "y_proba": y_proba_val},
            "test": {"y_true": y_test, "y_proba": y_proba_test},
        }
        # ============================> 1) ROC metrics + figure
        roc_stats = {
            split: get_roc_stats(
                vals["y_true"],
                vals["y_proba"],
                n_bootstraps=n_bootstraps,
                show_progress=show_progress,
            )
            for split, vals in split_data.items()
        }
        ## Add train, val, & test curves on same plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
        for split, stats in roc_stats.items():
            ax.plot(
                stats["fpr"],
                stats["tpr"],
                lw=4,
                label=f"{split.capitalize()} AUROC = {stats['auc_string']}",
            )
        ## Add meta
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=21, fontweight=550)
        ax.set_ylabel("True Positive Rate", fontsize=21, fontweight=550)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_title(f"{model_name} ROC", fontweight="semibold", fontsize=25)
        ax.legend(loc="lower right", prop={"size": 19, "weight": 550})
        ax.figure.tight_layout()
        ### Export plot
        roc_path = (
            result_dir / "figures" / "ROC" / outcome_name / f"{model_name}_ROC.pdf"
        )
        export_results(export_path=roc_path, data_to_export=ax)
        # ============================> 2) Get threshold
        if threshold_str == "val":
            print(f"\t Threshold determined by validation set")
            binary_threshold = roc_stats["val"]["optimal_threshold"]
        elif threshold_str == "train":
            print(f"\t Threshold determined by train set")
            binary_threshold = roc_stats["train"]["optimal_threshold"]
        else:
            print(
                f'Invalid input to "<threshold_str>" provided. Needs to be one of {"val", "train"}, got {threshold_str} instead. Using 0.5 as a threshold.'
            )
            binary_threshold = 0.5
        print("\t ROC done!")
        # ============================> 3) PRC Train/Val metrics + plot
        ## Add train, & val curves on same plot; test plot seperate
        pr_stats = {
            split: get_pr_stats(
                vals["y_true"],
                vals["y_proba"],
                n_bootstraps=n_bootstraps,
                show_progress=show_progress,
            )
            for split, vals in split_data.items()
        }
        #### TRAIN/VAL ####
        fig, ax = plt.subplots(figsize=(12, 8))
        # Get baseline
        dev_pos_rate = np.mean(y_train)
        ax.hlines(
            dev_pos_rate,
            0,
            1,
            color="gray",
            linestyle="--",
            label="Random Classifier",
        )
        for split in ["train", "val"]:
            stats = pr_stats[split]
            ax.plot(
                stats["recall"],
                stats["precision"],
                lw=4,
                label=f"{split.capitalize()} AUPRC = {stats['ap_string']}",
            )
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
        ## Add meta
        ax.set_title(
            f"{model_name} Train/Validation PR Curve",
            fontweight="semibold",
            fontsize=25,
        )
        ax.set_xlabel("Recall", fontsize=21, fontweight=550)
        ax.set_ylabel("Precision", fontsize=21, fontweight=550)
        ax.legend(loc="upper right", prop={"size": 19, "weight": 550})
        ax.figure.tight_layout()
        ## Export
        prc_path_dev = (
            result_dir
            / "figures"
            / "PRC"
            / "dev"
            / outcome_name
            / f"{model_name}_PRC.pdf"
        )
        export_results(export_path=prc_path_dev, data_to_export=ax)
        #### TEST ####
        fig, ax = plt.subplots(figsize=(12, 8))
        ## Get baseline
        test_pos_rate = np.mean(y_test)
        ax.hlines(
            test_pos_rate, 0, 1, color="gray", linestyle="--", label="Random Classifier"
        )
        ## plot
        ax.plot(
            pr_stats["test"]["recall"],
            pr_stats["test"]["precision"],
            lw=4,
            label=f"Test AUPRC = {pr_stats['test']['ap_string']}",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ## Add meta
        ax.set_title(f"{model_name} Test PR Curve", fontweight="semibold", fontsize=25)
        ax.set_xlabel("Recall", fontsize=21, fontweight=550)
        ax.set_ylabel("Precision", fontsize=21, fontweight=550)
        ax.legend(loc="upper right", prop={"size": 19, "weight": 550})
        ax.figure.tight_layout()
        ## Export
        prc_path = (
            result_dir
            / "figures"
            / "PRC"
            / "test"
            / outcome_name
            / f"{model_name}_PRC.pdf"
        )
        export_results(export_path=prc_path, data_to_export=ax)
        print("\t PRC done!")
        # ============================> 4) Add to class report
        for split, vals in split_data.items():
            y_true = vals["y_true"]
            y_proba = vals["y_proba"]
            y_pred_binary = (vals["y_proba"] >= binary_threshold).astype(int)
            ## Confusion Matrices
            ax = get_cm(
                model_name=model_name,
                outcome_name=outcome_name,
                data_type=split,
                y_true=y_true,
                y_pred=y_pred_binary,
            )
            cm_path = (
                result_dir
                / "figures"
                / "CM"
                / outcome_name
                / f"{model_name}_{split}_CM.pdf"
            )
            export_results(export_path=cm_path, data_to_export=ax)
            ## Class report
            CLASS_REPORT_DICT[split][model_name] = {
                "AUROC (95% CI)": roc_stats[split]["auc_string"],
                "AUPRC (95% CI)": pr_stats[split]["ap_string"],
                "Threshold": round(binary_threshold, 3),
            }
            ## Discrimination report
            for metric_str in metrics_str_list:
                CLASS_REPORT_DICT[split][model_name][metric_str] = (
                    get_discrimination_str(
                        y_true=y_true,
                        y_proba=y_proba,
                        metric_str=metric_str,
                        threshold=binary_threshold,
                        n_bootstraps=n_bootstraps,
                        random_state=SEED,
                        bin_thresholds_for_ici=bin_thresholds,  # only used for ICI
                        show_progress=show_progress,
                    )
                )
        print("\t ALL DONE with this model!")
    return CLASS_REPORT_DICT, BIN_REPORT_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome_name", required=True)
    parser.add_argument("--data_imp_dir", required=True)
    parser.add_argument("--model_imp_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--app_result_dir", required=True)
    parser.add_argument("--model_abrv_list", required=True, nargs="+")
    parser.add_argument("--n_bins", required=True, type=int, choices=[3, 4, 5])
    parser.add_argument("--n_bootstraps", required=True, type=int)
    parser.add_argument("--metrics_str_list", required=True, nargs="+")
    parser.add_argument(
        "--threshold_str", required=False, choices=["train", "val"], default="val"
    )
    parser.add_argument(
        "--eval_calibrated", required=False, choices=["True", "False"], default="True"
    )
    parser.add_argument(
        "--show_progress", required=False, choices=["True", "False"], default="False"
    )
    args = parser.parse_args()

    data_imp_dir_path = Path(args.data_imp_dir)
    model_imp_dir_path = Path(args.model_imp_dir)
    result_dir_path = Path(args.result_dir)
    app_result_dir_path = Path(args.app_result_dir)
    eval_calibrated_bool = True if args.eval_calibrated == "True" else False
    show_progress_bool = True if args.show_progress == "True" else False

    class_report_dict, bin_report_dict = eval_trained_models(
        outcome_name=args.outcome_name,
        data_imp_dir=data_imp_dir_path,
        model_abrv_list=args.model_abrv_list,
        model_imp_dir=model_imp_dir_path,
        app_result_dir=app_result_dir_path,
        n_bins=args.n_bins,
        result_dir=result_dir_path,
        n_bootstraps=args.n_bootstraps,
        metrics_str_list=args.metrics_str_list,
        threshold_str=args.threshold_str,
        eval_calibrated=eval_calibrated_bool,
        show_progress=show_progress_bool,
    )
    export_results(
        export_path=result_dir_path / "class_report" / f"{args.outcome_name}.json",
        data_to_export=class_report_dict,
    )
    export_results(
        export_path=result_dir_path / "bin_report" / f"{args.outcome_name}.json",
        data_to_export=bin_report_dict,
    )


if __name__ == "__main__":
    main()

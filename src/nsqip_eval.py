from src.evaluate import get_discrimination_str, export_results
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
import numpy as np
from MLstatkit import Delong_test


def avg_prec_metric(y_true, y_proba):
    return average_precision_score(y_true, y_proba)


def brier_metric(y_true, y_proba):
    return brier_score_loss(y_true, y_proba)


def construct_metric_dict(proba_df, model_name, ci_dict, rand_state):
    """
    Creates metric dict skeleton and populates w/
        Val + CIs for each of AUROC, AUPRC, Brier
        AUROC significance results
    Params
    ------
    proba_df: pd.DataFrame
        Contains y_true, y_proba_nsqip, and y_proba for each model
    model_name: str
        Name of model
    ci_dict: dict{}
        Keys: metric names used in Bootstrapping class
        Values: metric(lower_ci, upper_ci)
    """
    roc_dict = delong_res(
        proba_df=proba_df,
        model_name=model_name,
        rand_state=rand_state,
    )
    metric_dict = {
        "auroc": {
            "metric_fn": None,
            "perm_list": None,
            "nsqip_val": roc_dict["auc_nsqip"],
            "us_val": roc_dict["auc_us"],
            "p_val": roc_dict["roc_p"],
            "ci_str": ci_dict["roc_auc"],
        },
        "avg_prec": {
            "metric_fn": average_precision_score,
            "perm_list": [],
            "nsqip_val": -9999.0,
            "us_val": -9999.0,
            "p_val": None,
            "ci_str": ci_dict["average_precision"],
        },
        "brier": {
            "metric_fn": brier_metric,
            "perm_list": [],
            "nsqip_val": -9999.0,
            "us_val": -9999.0,
            "p_val": None,
            "ci_str": ci_dict["brier"],
        },
    }
    return metric_dict


def delong_res(proba_df, model_name, rand_state):
    """
    DeLongs AUROC p-vals
    """
    y_true = proba_df["y_true"]
    y_proba_nsqip = proba_df["y_proba_nsqip"]
    y_proba_us = proba_df[f"y_proba_{model_name}"]

    # returns (z, p_value, ci_A, ci_B, auc_A, auc_B, info)
    roc_skl_nsqip = roc_auc_score(y_true, y_proba_nsqip)
    roc_skl_us = roc_auc_score(y_true, y_proba_us)
    roc_results = Delong_test(
        true=y_true,
        prob_A=y_proba_nsqip,
        prob_B=y_proba_us,
        alpha=0.95,
        return_auc=True,
        return_ci=True,
        n_boot=1,
        random_state=rand_state,
        verbose=0,
    )

    roc_p = roc_results[1]
    auc_nsqip = roc_results[4]
    auc_us = roc_results[5]
    assert np.isclose(auc_nsqip, roc_skl_nsqip, atol=0.0001)
    assert np.isclose(auc_us, roc_skl_us, atol=0.0001)
    result_dict = {
        "roc_p": round(roc_p, 4),
        "auc_nsqip": round(auc_nsqip, 4),
        "auc_us": round(auc_us, 4),
    }
    return result_dict


def add_perm_deltas(proba_df, model_res_dict, n_perm, rand_state):
    """
    - Calculates difference in each *metric* over n_perm permutations for each *model*
    - Convention will be our_model - nsqip_model.
        - Rather arbitrary here as long as consistent.

    *metric and models provided in model_res_dict{}*

    Params
    ------
    proba_df: pd.DataFrame
        Contains y_true, NSQIP probability output, and probability output from each of our models for a given outcome
    model_res_dict: dict{}
        Keys: model name (lr, xgb, etc)
        Values: dict{}
            Keys: metric_name
            Values: dict{}
                Keys:
                'metric_fn': metric function, accepting y_true and y_proba,
                "perm_list": list of permutation delta metrics, appended at each iteration
                'nsqip_val': to be filled out later (outside this function),
                'us_val': to be filled out later (outside this function),
                'p_val': to be filled out later (outside this function
    n_perm: int
        Number of bootstraps
    rand_state: int
        Random seed for bootstrapping

    Returns
    ------
    model_res_dict{} w/ perm_list filled out for each model/metric pair
    """
    rng = np.random.default_rng(rand_state)
    y_true = proba_df["y_true"]
    y_proba_nsqip = proba_df["y_proba_nsqip"]
    n = len(y_true)
    for p in range(n_perm):
        swap_mask = rng.random(n) < 0.5
        # For each of our base models (svc, nn, lgbm, etc)
        for model_name, metric_dict in model_res_dict.items():
            y_proba_us = proba_df[f"y_proba_{model_name}"]
            ## Get swapped probas
            # syntax: np.where(condition, val if true, val if false)
            perm_nsqip = np.where(swap_mask, y_proba_us, y_proba_nsqip)
            perm_us = np.where(swap_mask, y_proba_nsqip, y_proba_us)
            # For each metric, calc delta metric and append to list
            for metric_name, metric_sub_dict in metric_dict.items():
                if metric_name == "auroc":
                    continue
                metric_func = metric_sub_dict["metric_fn"]
                obs_diff = metric_func(y_true, perm_us) - metric_func(
                    y_true, perm_nsqip
                )
                model_res_dict[model_name][metric_name]["perm_list"].append(obs_diff)
                # Ensure we're adding correctly
                assert len(model_res_dict[model_name][metric_name]["perm_list"]) == (
                    p + 1
                )
    return model_res_dict


def format_p(p_val):
    if p_val < 0.0001:
        p_val_str = "**<0.0001**"
    elif p_val < 0.05:
        p_val_str = f"**{p_val:.4f}**"
    else:
        p_val_str = f"{p_val:.2f}"
    return p_val_str


def plot_roc_prc(
    y_true, y_proba, roc_val_ci_str, prc_val_ci_str, outcome_name, res_dir
):
    """
    Params
    -----
    val_ci_str: str
        Value and CIs of
    """
    ### ROC
    fig, ax = plt.subplots(figsize=(12, 8))
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    ax.plot(
        fpr,
        tpr,
        lw=4,
        label=f"Test NSQIP AUROC = {roc_val_ci_str}",
    )
    ## Add meta
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=21, fontweight=550)
    ax.set_ylabel("True Positive Rate", fontsize=21, fontweight=550)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"NSQIP ROC", fontweight="semibold", fontsize=25)
    ax.legend(loc="lower right", prop={"size": 19, "weight": 550})
    ax.figure.tight_layout()
    ### Export plot
    roc_path = res_dir / "figures" / "ROC" / f"{outcome_name}_ROC.pdf"
    export_results(export_path=roc_path, data_to_export=ax)
    ## PRC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(12, 8))
    ## Get baseline
    test_pos_rate = np.mean(y_true)
    ax.hlines(
        test_pos_rate, 0, 1, color="gray", linestyle="--", label="Random Classifier"
    )
    ## plot
    ax.plot(
        recall,
        precision,
        lw=4,
        label=f"Test AUPRC = {prc_val_ci_str}",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ## Add meta
    ax.set_title(f"NSQIP Test PR Curve", fontweight="semibold", fontsize=25)
    ax.set_xlabel("Recall", fontsize=21, fontweight=550)
    ax.set_ylabel("Precision", fontsize=21, fontweight=550)
    ax.legend(loc="upper right", prop={"size": 19, "weight": 550})
    ax.figure.tight_layout()
    ## Export
    prc_path = res_dir / "figures" / "PRC" / f"{outcome_name}_PRC.pdf"
    export_results(export_path=prc_path, data_to_export=ax)


def fill_model_res_dict_update(
    proba_df, y_true, y_proba_nsqip, model_list, n_boot, n_perm, rand_state
):
    ### Get CIs ###
    metric_for_cis = ["roc_auc", "average_precision", "brier"]
    ci_dict = {
        metric: get_discrimination_str(
            y_true=y_true,
            y_proba=y_proba_nsqip,
            metric_str=metric,
            threshold=None,  # not used for AUROC, AUPRC, or brier
            n_bootstraps=n_boot,
            random_state=rand_state,
        )
        for metric in metric_for_cis
    }
    ## Create skeleton + get AUROC results
    model_res_dict = {
        model_name: construct_metric_dict(
            proba_df=proba_df,
            model_name=model_name,
            ci_dict=ci_dict,
            rand_state=rand_state,
        )
        for model_name in model_list
    }
    ## Bootstrap AUPRC + Brier
    model_res_dict_update = add_perm_deltas(
        proba_df=proba_df,
        model_res_dict=model_res_dict,
        n_perm=n_perm,
        rand_state=rand_state,
    )
    # get p-vals for each model/metric
    # some redundancy here but this is quick enough to be okay
    for model_name, metric_dict in model_res_dict_update.items():
        y_proba_us = proba_df[f"y_proba_{model_name}"]
        for metric_name, sub_dict in metric_dict.items():
            if metric_name == "auroc":
                continue
            perm_deltas = sub_dict["perm_list"]
            metric_fn = sub_dict["metric_fn"]
            obsv_us = metric_fn(y_true, y_proba_us)
            obsv_nsqip = metric_fn(y_true, y_proba_nsqip)
            # us - nsqip
            obsv_delta = obsv_us - obsv_nsqip
            p_val = np.mean(np.abs(perm_deltas) >= np.abs(obsv_delta))

            model_res_dict_update[model_name][metric_name]["nsqip_val"] = obsv_nsqip
            model_res_dict_update[model_name][metric_name]["us_val"] = obsv_us
            model_res_dict_update[model_name][metric_name]["p_val"] = p_val
    return model_res_dict_update

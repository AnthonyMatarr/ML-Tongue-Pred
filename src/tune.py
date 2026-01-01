# Packages/libraries
from src.config import SEED, DEVICE
from src.nn_models import TorchNNClassifier

import joblib
import json
import warnings
import logging
import numpy as np
import optuna

import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# Determinism
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


######################################## NEURAL NETWORK ########################################
def build_nn_estimator(trial):
    """
    Model builder used for neural network optuna training
    """
    n_layers = trial.suggest_int("n_layers", 2, 3)
    ### Hidden Layers ###
    hl_1 = trial.suggest_int("hl_1", 32, 512)
    hl_2 = trial.suggest_int("hl_2", 32, 512)
    h_sizes = [hl_1, hl_2]
    if n_layers == 3:
        hl_3 = trial.suggest_int("hl_3", 32, 512)
        h_sizes.append(hl_3)

    ### Dropouts ###
    dr_1 = trial.suggest_float("dr_1", 1e-5, 0.8)
    dr_2 = trial.suggest_float("dr_2", 1e-5, 0.8)
    dropouts = [dr_1, dr_2]
    if n_layers == 3:
        dr_3 = trial.suggest_float("dr_3", 1e-5, 0.8)
        dropouts.append(dr_3)

    ####Activation ###
    act_name = trial.suggest_categorical("act_func_str", ["relu", "leaky_relu"])
    ### Epochs ###
    num_epochs = trial.suggest_int("num_epochs", 10, 100)

    ### Optimizer ####
    # opt_choice = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    opt_choice = "adamw"
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    ##Batch size
    bs = trial.suggest_categorical("batch_size", [128, 256, 512])

    nn_clf = TorchNNClassifier(
        hidden_size_list=h_sizes,
        dropouts=dropouts,
        activation_name=act_name,
        lr=lr,
        weight_decay=wd,
        epochs=num_epochs,
        batch_size=bs,
        optimizer_str=opt_choice,
        device=DEVICE,
        seed=SEED,
    )
    return nn_clf


def make_objective_nn(X_train, y_train, scoring):
    """
    Creates objective for optuna neural network tuning
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        clf = build_nn_estimator(trial)
        scores = cross_val_score(
            clf, X_train, y_train, scoring=scoring, cv=skf, n_jobs=1
        )
        return np.round(np.mean(scores), 4)

    return objective


def tune_model_nn(*_, X_train, y_train, scoring, study, log_path, save_path, n_trials):
    """
    Tunes given model with Optuna and writes params/tuning results to memory.
    Used for NN

    Parameters
    ----------
    X_train: pandas dataframe
        Pandas dataframe containing training predictor features/data
    y_train: pandas dataframe
        Pandas data frame containing training labels.
    scoring: str
        String specifying which scoring metric to use for tuning.
        Ultimately passed into sklearn.model_selection.cross_val_score().
    study: optuna.Trial
        Optuna study object initialized by optuna.create_study().
    log_path: pathlib.Path
        Absolute path to file where tuning logs will be written to.
    save_path: pathlib.Path
        Absolute path to file where best CV score/params are written to in json format.
    n_trials: int
        Specifies number of trials to run optuna tuner
    Returns
    -------
    Nothing. This function only writes to memory
    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This func does not take positional args")
    ############ Set up paths ############
    if log_path.exists():
        log_path.unlink()
    log_path.parent.mkdir(exist_ok=True, parents=True)
    if save_path.exists():
        save_path.unlink()
    save_path.parent.mkdir(exist_ok=True, parents=True)

    ############ Set up logger ############
    file_handler = logging.FileHandler(log_path, mode="a")
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    ############ Run optimizer ############
    study.optimize(
        make_objective_nn(X_train, y_train, scoring),
        n_trials=n_trials,
    )
    ############ More logging ############
    root_logger.removeHandler(file_handler)
    file_handler.close()

    with open(log_path, "a") as f:
        f.write(f"{'*' * 100}\n")
        f.write(f"Best {scoring}: {study.best_value:.4f}\nparams={study.best_params}\n")
        f.write(f"{'*' * 100}\n")

    ############ Get dictionary to export/save ############
    result_dict = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        # "study": study, # NO need to save this, but including in case want to inspect later
    }
    ### Only needed if study is included --> remove study:<study> to export ###
    # result_dict = {k: v for k, v in result_dict.items() if k != "study"}
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    # return result_dict


######################################## SVC, LGBM, LR MODELS ########################################
def lr_model_builder(trial):
    """
    Logistic regression model builder for optuna tuning
    """
    ########## Get params ##########
    # C (regularization strength)
    C = trial.suggest_float("C", 1e-8, 1e2, log=True)
    # Regularization Penalty
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    # Class weight (to offset imbalance)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    # l1 ratio --> only useful for elasticnet (=0 means l2, =1 means l1) = None else
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:
        l1_ratio = None
    # Solver (dependent on penalty)
    # elasticnet --> ['saga]
    # l1 --> ['liblinear', 'saga']
    # l2 --> ALL
    if penalty == "elasticnet":
        solver = "saga"
    elif penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
    else:
        solver = trial.suggest_categorical(
            "solver_l2",
            ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        )

    # Intercept scaling (only useful if liblinear) = 1.0 else
    if solver == "liblinear":
        intercept_scaling = trial.suggest_float(
            "intercept_scaling", 1e-6, 1e2, log=True
        )
    else:
        intercept_scaling = 1.0

    # Need more iterations for saga/sag
    if solver in ["saga", "sag"]:
        max_iter = 10000
    else:
        max_iter = 5000
    ########## Build model ##########
    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        tol=1e-4,  # default
        fit_intercept=True,  # default
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        warm_start=False,
    )
    return clf


def lightgbm_model_builder(trial):
    """
    LightGBM model builder for optuna tuning
    """
    ######### Get params ###########
    ##Core params
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.4, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    ##Tree shape + complexity
    max_depth = trial.suggest_int("max_depth", -1, 20)  # -1 = no limit
    if max_depth < 0:
        max_leaves = 255
    else:
        max_leaves = min(255, 2 * max_depth)
    min_leaves = 12
    if min_leaves > max_leaves:
        max_leaves = 30
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)

    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 100)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 1.0)

    ## Subsampling/Imbalance
    feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
    # bagging_fraction = trial.suggest_float("bagging_fraction", 0.6, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 0, 20)
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 50.0, log=True)
    pos_bagging_fraction = trial.suggest_float(
        "pos_bagging_fraction", 0.7, 1.0
    )  # 1.0 samples all
    neg_bagging_fraction = trial.suggest_float(
        "neg_bagging_fraction", 0.1, 1.0
    )  # 1.0 samples all

    # Regularization
    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 10.0)
    lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 10.0)

    # Binning --> More bins may improve accuracy but cost more resources
    max_bin = trial.suggest_int("max_bin", 128, 511)

    ######### Build model ###########
    clf = LGBMClassifier(
        objective="binary",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_gain_to_split,  # alias of min_gain_to_split in sklearn wrapper
        feature_fraction=feature_fraction,
        pos_bagging_fraction=pos_bagging_fraction,
        neg_bagging_fraction=neg_bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        scale_pos_weight=scale_pos_weight,
        max_bin=max_bin,
        tree_learner="feature_parallel",
        n_jobs=2,
        seed=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
        # If you use categorical features with pandas dtypes 'category',
        # LightGBM will auto-detect them; otherwise pass categorical_feature explicitly in fit.
    )
    return clf


def svc_model_builder(trial):
    """
    Support Vector Classifier model builder for optuna training
    """
    ######### Get params ###########
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)

    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])

    # Degree only for poly kernel
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 6)
    else:
        degree = 3  # ignored by other kernels

    # Gamma only significant for 'rbf', 'poly', 'sigmoid'
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_choice = trial.suggest_categorical(
            "gamma_choice", ["scale_auto", "numerical"]
        )
        if gamma_choice == "scale_auto":
            gamma = trial.suggest_categorical("gamma_sa", ["scale", "auto"])
        else:
            gamma = trial.suggest_float("gamma_num", 1e-4, 10.0, log=True)
    else:
        gamma = "scale"

    # coef0 only significant for poly and sigmoid
    if kernel in ["poly", "sigmoid"]:
        coef0 = trial.suggest_float("coef0", 1e-5, 2.0, log=True)
    else:
        coef0 = 0.0

    shrinking = trial.suggest_categorical("shrinking", [True, False])

    # Weight given to positive class
    pos_weight = trial.suggest_float("pos_weight", 0.75, 20)

    ######### Build model ###########
    clf = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=False,  # Outputs will be calibrated later
        class_weight={0: 1, 1: pos_weight},
        random_state=SEED,
        cache_size=1000,
    )
    return clf


def make_objective(X_train, y_train, model_builder, scoring="roc_auc"):
    """
    Creates objective for LR, LGBM, SVC model tuning
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        model = model_builder(trial)
        scores = cross_val_score(
            model, X_train, y_train, scoring=scoring, cv=skf, n_jobs=1
        )
        return np.round(np.mean(scores), 4)

    return objective


def tune_model_mult_outcomes(
    *_,
    model_builder,
    model_abrv,
    outcome_dict,
    scoring,
    log_file_path,
    save_path,
    n_trials=500,
):
    """
    Tunes a given model for each given outcome using Optuna and writes params/tuning results to memory.
    Used for LR, SVC, LGBM

    Parameters
    ----------
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_abrv: string
        Abbreviation of model to be tuned, one of [lr, svc, lgbm]
    outcome_dict: dict
        Dictionary mapping outcome names to relevant data
        Format:
            {
                outcome_type (str): {
                    'X_train': pandas dataframe,
                    'y_train':pandas dataframe
                    'X_val': pandas dataframe
                    'y_val': pandas dataframe
                    'X_test': pandas dataframe
                    'y_test': pandas dataframe
                }
            }
    scoring: str
        String specifying which scoring metric to use for tuning.
        Ultimately passed into sklearn.model_selection.cross_val_score().
    log_file_path: pathlib.Path
        Absolute path to file where tuning logs will be written to.
    save_path: pathlib.Path
        Absolute path to directory where best CV score/params are written to in json format.
    n_trials: Optional int; defaults to 500
    Returns
    -------
    Nothing. This function only writes to memory
    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This func does not take positional args")

    ############ Set up paths ############
    if log_file_path.exists():
        warnings.warn(f"Over-writing log at path: {log_file_path}")
        log_file_path.unlink()
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    if save_path.exists():
        save_path.unlink()
    save_path.parent.mkdir(exist_ok=True, parents=True)

    ############ Set up logger ############
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode="a")
    root_logger.addHandler(file_handler)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    ############ Run for each outcome ############
    result_dict = {}
    for outcome_name, outcome_data in outcome_dict.items():
        with open(log_file_path, "a") as f:
            f.write(f"Working on outcome: {outcome_name}...\n")
        X_train = outcome_data["X_train"]
        y_train = outcome_data["y_train"].values.ravel()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            study = optuna.create_study(
                study_name=f"{model_abrv}_{outcome_name}_study",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.HyperbandPruner(),
            )
            study.optimize(
                make_objective(X_train, y_train, model_builder, scoring),  # type: ignore
                n_trials=n_trials,
            )
            result_dict[outcome_name] = {
                "best_score": study.best_value,
                "best_params": study.best_params,
                # "study": study, # NO need to save this, but including in case want to inspect later
            }
        with open(log_file_path, "a") as f:
            f.write(
                f"{outcome_name}: best {scoring}: {study.best_value:.4f}\nparams={study.best_params}\n"
            )
            f.write(f"{'*' * 100}\n")
    ### Only needed if study is included --> remove study:<study> to export ###
    # export_dict = {
    #     k: {kk: vv for kk, vv in v.items() if kk != "study"}
    #     for k, v in results_dict.items()
    # }
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    return result_dict


######################################## FOR ALL MODELS ########################################
def get_prelim_results(
    *_,
    results_dict,
    model_builder,
    model_abrv,
    outcome_dict,
    model_save_dir=None,
):
    """
    Calculates train + validation AUROC scores and prints them out.
    Also used to export models when model_save_dir is not None.

    Parameters
    ----------
    results_dict: dict
        Dictionary mapping of outcome types to results
        Format:
        {
            outcome_type (str): {
                'best_score': float,
                'best_params': dict
            }
        }
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator

    model_abrv: str
        Abbreviation of model used
    outcome_dict: dict
        Dictionary mapping outcome names to relevant data
        Format:
        {
            outcome_type (str): {
                'X_train': pandas dataframe,
                'y_train': pandas dataframe
                'X_val': pandas dataframe,
                'y_val': pandas dataframe
                'X_test': pandas dataframe,
                'y_test': pandas dataframe
            }
        }
    model_save_dir: pathlib.Path; defaults to None
        Directory to save models. If None, will not save models
    Returns
    --------
    Nothing, just prints out results and (optionally) saves models

    Raises
    --------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not take position arguments!")

    print(f"{'-'*30} {model_abrv} {'-'*30}")
    for outcome, results in results_dict.items():
        ##Get tuning results
        best_score = results["best_score"]
        best_params = results["best_params"]

        trial = optuna.trial.FixedTrial(best_params)
        model = model_builder(trial)

        ##### Train model #####
        X_train = outcome_dict[outcome]["X_train"]
        y_train = outcome_dict[outcome]["y_train"]
        model.fit(X_train, y_train.values.ravel())

        ##### Export model #####
        if model_save_dir:
            # n_layers only provided if using NN
            if model_abrv == "nn":
                save_path = model_save_dir / f"nn.pt"
                if save_path.exists():
                    save_path.unlink()
                save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(
                    {
                        "h_params": best_params,
                        "state_dict": model.model_.state_dict(),
                        "feature_names_in_": getattr(model, "feature_names_in_"),
                    },  # type: ignore
                    save_path,
                )
            # for all other models (lr, svc, lgbm)
            else:
                save_path = model_save_dir / outcome / f"{model_abrv}.joblib"
                if save_path.exists():
                    save_path.unlink()
                save_path.parent.mkdir(exist_ok=True, parents=True)
                joblib.dump(model, save_path)

        ##### Get prelim results #####
        X_val = outcome_dict[outcome]["X_val"]
        y_val = outcome_dict[outcome]["y_val"]

        ## Neural Network (model_abrv can be nn-2 or nn-3)
        if model_abrv == "nn":
            # nn class auto-implements AUROC for .score()
            train_auc = model.score(X_train, y_train)
            val_auc = model.score(X_val, y_val)

        else:
            ## SVC
            if model_abrv == "svc":
                # not probabilities, but appropriately ranked
                x_train_output = model.decision_function(X_train)  # type: ignore
                x_val_output = model.decision_function(X_val)  # type: ignore
            ## All other models (lr, lgbm)
            else:
                # probability of class 1
                x_train_output = model.predict_proba(X_train)[:, 1]
                x_val_output = model.predict_proba(X_val)[:, 1]
            ## Only compute these separately for non-NN models
            train_auc = roc_auc_score(y_train, x_train_output)
            val_auc = roc_auc_score(y_val, x_val_output)

        ##### Output results #####
        print(f"Outcome: \t{outcome}")
        print(f"Best CV AUROC: \t{best_score:.3f}")
        print(f"Train AUROC: \t{train_auc:.3f}")
        print(f"Val ROC AUC: \t{val_auc:.3f}")
        print(f"PARAMS: \t{best_params}")
        print("*" * 10)

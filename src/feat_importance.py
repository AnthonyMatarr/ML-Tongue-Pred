import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import shap
import numpy as np
import pandas as pd


def get_ohe_cols(df):
    """
    Helper function that returns original column names of features that were one-hot encoded
    Uses '_' as an indicator
    """
    ohe_cols = set()
    for col in df.columns:
        split_col = col.split("_")
        if len(split_col) > 1:
            ## These have _ in their names but were not one-hot encoded
            if split_col[0] in [
                "ETHNICITY",
                "Partial Glossectomy (Hemiglossectomy",
                "Composite",
                "Local",
            ]:
                continue
            ohe_cols.add(split_col[0])
    return ohe_cols


######## Combine one-hot-encoded #######
def combine_encoded(shap_values, name, mask, return_original=True):
    """
    Helper function to combine shap values for one-hot encoded columns
    Adapted from following repository: https://gist.github.com/peterdhansen/ca87cc1bfbc4c092f0872a3bfe3204b2
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]
    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_col_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
        instance_names=shap_values.instance_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)  # type: ignore
    svdata = np.concatenate(
        [shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1
    )

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate(
        [svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1
    )

    # Handle multi-class (3D) vs binary/regression (2D) SHAP arrays
    if len(shap_values.values.shape) == 3:  # Multi-class case
        # Sum encoded features while preserving class dimension
        new_values = sv_name.values.sum(axis=1, keepdims=True)  # type: ignore
        svvalues = np.concatenate([shap_values.values[:, ~mask, :], new_values], axis=1)
    else:  # Binary/regression case
        new_values = sv_name.values.sum(axis=1)  # type: ignore
        svvalues = np.concatenate(
            [shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1
        )

    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(
        svvalues,
        base_values=shap_values.base_values,
        data=svdata,
        display_data=svdisplay_data,
        instance_names=shap_values.instance_names,
        feature_names=svfeature_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    if return_original:
        return sv, sv_name
    else:
        return sv


def get_vals_to_plot(shap_vals):
    """
    Helper function to reformat the shap_vals object (return value of shap.explainer())
    """
    if len(shap_vals.values.shape) == 3:  # 3D array
        if shap_vals.values.shape[2] == 1:  # Binary classification with single output
            # DNN
            shap_vals_to_plot = shap_vals[:, :, 0]
        elif shap_vals.values.shape[2] >= 2:  # Binary with two outputs or multi-class
            shap_vals_to_plot = shap_vals[:, :, 1]  # Use positive class
        else:
            shap_vals_to_plot = shap_vals.mean(axis=2)  # Fallback
    else:  # 2D array
        # LightGBM, SVC, KNN, Stack, LR-Nomogram
        shap_vals_to_plot = shap_vals
    return shap_vals_to_plot


def get_shap(*_, X, model_dict, feat_order, result_path=None):
    """
    Generates SHAP values with kernel explainer, builds mean absolute value (MAV) and relative MAV shap tables, and exports the tables

    Parameters
    ----------
    X: pandas dataframe
        Tabular dataframe containing data for which SHAP values will be generated (excluding target variable)
        Test set is recommended for useful SHAP values, but can be any subset of the dataset
    model_dict: dict()
        Dictionary mapping model names to models
        Format:
            {
                <model_name> str: <model> sklearn model
            }
    feat_order: list[str]
        List of column names specifying desired order in SHAP table
    result_path: Optional pathlib.Path; defaults None
        Path to excel file where shap tables for all models will be written to
        If left None, will not export
    Returns
    -------
    Nothing

    Raises
    ------
    ValueError:
        If positional arguments are provided
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if result_path and result_path.exists():
        warnings.warn(
            category=Warning,
            message=f"Removing file at {result_path}. \nUnless this function fails, will over-write with new SHAP table",
        )
        result_path.unlink()
    ######################## Get SHAP values ########################
    for model_name, model in model_dict.items():
        print(f'Working on model: {model_name}...')
        explainer = shap.Explainer(
            model.predict_proba, X, feature_names=X.columns.tolist()
        )
        shap_old = explainer(X)
        ######################## Combine one-hot encoded ########################
        ohe_cols = get_ohe_cols(X)
        for col_name in ohe_cols:
            shap_values_new, _ = combine_encoded(
                shap_old, col_name, [col_name in n for n in shap_old.feature_names]
            )
            shap_old = shap_values_new
        combined_feat_names = shap_values_new.feature_names

        ######################## Generate MAV table ########################
        # Reformat shap values
        shap_vals_to_plot = get_vals_to_plot(shap_values_new)
        shap_df = pd.DataFrame(shap_vals_to_plot.values, columns=combined_feat_names)

        # Get absolute avg + relative abs avg
        absolute_mean_shap = shap_df.abs().mean().reset_index()
        absolute_mean_shap.columns = ["Feature", "MASV"]
        sum_vals = absolute_mean_shap["MASV"].sum()
        if sum_vals == 0:
            warnings.warn(
                message="All generated SHAP values are 0. Exiting...", category=Warning
            )
            return
        absolute_mean_shap["Relative_ MASV"] = np.round(
            (100 * absolute_mean_shap["MASV"] / absolute_mean_shap["MASV"].sum()), 2
        )
        # Ensure logic makes sense
        assert np.isclose(
            absolute_mean_shap["Relative_ MASV"].sum(), 100, atol=0.1
        ), f"Sum is instead {absolute_mean_shap['Relative_ MASV'].sum()}"
        # Reorder
        absolute_mean_shap["Feature"] = absolute_mean_shap["Feature"].astype(str)

        absolute_mean_shap_reordered = (
            absolute_mean_shap.set_index("Feature").loc[feat_order].reset_index()
        )

        ######################## Display + Export ########################
        if result_path:
            result_path.parent.mkdir(exist_ok=True, parents=True)
            with pd.ExcelWriter(
                result_path,
                engine="openpyxl",
                mode="a" if result_path.exists() else "w",
            ) as writer:
                absolute_mean_shap_reordered.to_excel(
                    writer, sheet_name=model_name, index=True
                )

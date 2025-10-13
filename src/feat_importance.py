import warnings
import copy
from src.config import SEED

warnings.filterwarnings("ignore", category=UserWarning)
import shap
import numpy as np
import pandas as pd


def get_ohe_cols(df):
    """
    Helper function that returns original column names of features that were one-hot encoded
    Uses '_' as an indicator
    """
    ohe_dict = {}
    for col in df.columns:
        col_split = col.split("_")
        if len(col_split) == 1 or col_split[0] in [
            "ETHNICITY",
            "Partial Glossectomy (Hemiglossectomy",
            "Composite",
            "Local",
        ]:
            continue
        col_name = col_split[0]
        instance_name = "_".join(col_split[1:])
        if col_name in ohe_dict.keys():

            ohe_dict[col_name].append(instance_name)
        else:
            ohe_dict[col_name] = [instance_name]
    return ohe_dict


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


def generate_MAV(shap_vals, feat_order, model_name, result_path=None):
    """
    Helper function to generate and optionally export mean absolute value table for shap values.

    Parameters
    ---------
    shap_vals: shap._explanation.Explanation
        Shap explanation object containing shap values
    feat_order:list[str]
        List of column names specifying desired order in SHAP table
    model_name: str
        Specify which model is being analyzed
    result_path: Optional pathlib.Path; defaults None
        Path to directory where shap tables for all models will be written to
        If left None, will not export
    """
    # if result_path and result_path.exists():
    #     warnings.warn(
    #         category=Warning,
    #         message=f"Removing file at {result_path}. \nUnless this function fails, will over-write with new SHAP table",
    #     )
    # result_path.unlink()
    feat_names = shap_vals.feature_names
    assert set(feat_names) == set(feat_order)
    shap_to_plot = get_vals_to_plot(shap_vals)
    shap_df = pd.DataFrame(shap_to_plot.values, columns=feat_names)
    absolute_mean_shap = shap_df.abs().mean().reset_index()
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


def get_shap(*_, X, model_dict, outcome_name, feat_order, result_path=None):
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
    outcome_name: str
        Specify outcome whose models are being analyzed
    feat_order: list[str]
        List of column names specifying desired order in SHAP table
    result_path: Optional pathlib.Path; defaults None
        Path to directory where shap tables for all models will be written to
        If left None, will not export

    Raises
    ------
    ValueError:
        If positional arguments are provided
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    ######################## Get SHAP values ########################
    for model_name, model in model_dict.items():
        print(f"Working on model: {model_name}...")
        explainer = shap.Explainer(
            model.predict_proba, X, feature_names=X.columns.tolist(), seed=SEED
        )
        shap_raw = explainer(X)
        ######################## Deal with one-hot encoded ########################
        ohe_dict = get_ohe_cols(X)
        ohe_cols = ohe_dict.keys()
        ### Get OHE feature order for raw
        raw_feat_order = []
        for col in feat_order:
            if col in ohe_cols:
                for sub_col in ohe_dict[col]:
                    raw_feat_order.append(f"{col}_{sub_col}")
            else:
                raw_feat_order.append(col)
        ### Combine ohe for combined
        shap_old = copy.deepcopy(shap_raw)
        for col_name in ohe_cols:
            shap_combined, _ = combine_encoded(
                shap_old, col_name, [col_name in n for n in shap_old.feature_names]
            )
            shap_old = shap_combined
        ######################## Generate + export MAV table ########################
        if result_path:
            raw_path = result_path / "raw" / f"{outcome_name}.xlsx"
            combined_path = result_path / "combined" / f"{outcome_name}.xlsx"
        else:
            raw_path = None
            combined_path = None
        generate_MAV(
            shap_raw, raw_feat_order, model_name=model_name, result_path=raw_path
        )
        generate_MAV(
            shap_combined, feat_order, model_name=model_name, result_path=combined_path
        )


# import warnings

# warnings.filterwarnings("ignore", category=UserWarning)
# import shap
# import numpy as np
# import pandas as pd


# def get_ohe_cols(df):
#     """
#     Helper function that returns original column names of features that were one-hot encoded
#     Uses '_' as an indicator
#     """
#     ohe_dict = {}
#     for col in df.columns:
#         col_split = col.split("_")
#         if len(col_split) == 1 or col_split[0] in [
#             "ETHNICITY",
#             "Partial Glossectomy (Hemiglossectomy",
#             "Composite",
#             "Local",
#         ]:
#             continue
#         col_name = col_split[0]
#         instance_name = "_".join(col_split[1:])
#         if col_name in ohe_dict.keys():

#             ohe_dict[col_name].append(instance_name)
#         else:
#             ohe_dict[col_name] = [instance_name]
#     return ohe_dict


# ######## Combine one-hot-encoded #######
# def combine_encoded(shap_values, name, mask, return_original=True):
#     """
#     Helper function to combine shap values for one-hot encoded columns
#     Adapted from following repository: https://gist.github.com/peterdhansen/ca87cc1bfbc4c092f0872a3bfe3204b2
#     """
#     mask = np.array(mask)
#     mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]
#     sv_name = shap.Explanation(
#         shap_values.values[:, mask],
#         feature_names=list(mask_col_names),
#         data=shap_values.data[:, mask],
#         base_values=shap_values.base_values,
#         display_data=shap_values.display_data,
#         instance_names=shap_values.instance_names,
#         output_names=shap_values.output_names,
#         output_indexes=shap_values.output_indexes,
#         lower_bounds=shap_values.lower_bounds,
#         upper_bounds=shap_values.upper_bounds,
#         main_effects=shap_values.main_effects,
#         hierarchical_values=shap_values.hierarchical_values,
#         clustering=shap_values.clustering,
#     )
#     new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)  # type: ignore
#     svdata = np.concatenate(
#         [shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1
#     )

#     if shap_values.display_data is None:
#         svdd = shap_values.data[:, ~mask]
#     else:
#         svdd = shap_values.display_data[:, ~mask]

#     svdisplay_data = np.concatenate(
#         [svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1
#     )

#     # Handle multi-class (3D) vs binary/regression (2D) SHAP arrays
#     if len(shap_values.values.shape) == 3:  # Multi-class case
#         # Sum encoded features while preserving class dimension
#         new_values = sv_name.values.sum(axis=1, keepdims=True)  # type: ignore
#         svvalues = np.concatenate([shap_values.values[:, ~mask, :], new_values], axis=1)
#     else:  # Binary/regression case
#         new_values = sv_name.values.sum(axis=1)  # type: ignore
#         svvalues = np.concatenate(
#             [shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1
#         )

#     svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

#     sv = shap.Explanation(
#         svvalues,
#         base_values=shap_values.base_values,
#         data=svdata,
#         display_data=svdisplay_data,
#         instance_names=shap_values.instance_names,
#         feature_names=svfeature_names,
#         output_names=shap_values.output_names,
#         output_indexes=shap_values.output_indexes,
#         lower_bounds=shap_values.lower_bounds,
#         upper_bounds=shap_values.upper_bounds,
#         main_effects=shap_values.main_effects,
#         hierarchical_values=shap_values.hierarchical_values,
#         clustering=shap_values.clustering,
#     )
#     if return_original:
#         return sv, sv_name
#     else:
#         return sv


# def get_vals_to_plot(shap_vals):
#     """
#     Helper function to reformat the shap_vals object (return value of shap.explainer())
#     """
#     if len(shap_vals.values.shape) == 3:  # 3D array
#         if shap_vals.values.shape[2] == 1:  # Binary classification with single output
#             # DNN
#             shap_vals_to_plot = shap_vals[:, :, 0]
#         elif shap_vals.values.shape[2] >= 2:  # Binary with two outputs or multi-class
#             shap_vals_to_plot = shap_vals[:, :, 1]  # Use positive class
#         else:
#             shap_vals_to_plot = shap_vals.mean(axis=2)  # Fallback
#     else:  # 2D array
#         # LightGBM, SVC, KNN, Stack, LR-Nomogram
#         shap_vals_to_plot = shap_vals
#     return shap_vals_to_plot


# def generate_MAV(shap_vals, feat_order, model_name, result_path=None):
#     """
#     Helper function to generate and optionally export mean absolute value table for shap values.

#     Parameters
#     ---------
#     shap_vals: shap._explanation.Explanation
#         Shap explanation object containing shap values
#     feat_order:list[str]
#         List of column names specifying desired order in SHAP table
#     model_name: str
#         Specify which model is being analyzed
#     result_path: Optional pathlib.Path; defaults None
#         Path to directory where shap tables for all models will be written to
#         If left None, will not export
#     """
#     feat_names = shap_vals.feature_names
#     try:
#         assert set(feat_names) == set(feat_order)
#     except AssertionError as e:
#         print(f"Feat name: {len(feat_names)}")
#         print(feat_names)
#         print(f"Feat order: {len(feat_order)}")
#         print(feat_order)
#         print("ERROR: Feature names does not match feat order")
#         return
#     shap_to_plot = get_vals_to_plot(shap_vals)
#     shap_df = pd.DataFrame(shap_to_plot.values, columns=feat_names)
#     absolute_mean_shap = shap_df.abs().mean().reset_index()
#     # Get absolute avg + relative abs avg
#     absolute_mean_shap = shap_df.abs().mean().reset_index()
#     absolute_mean_shap.columns = ["Feature", "MASV"]
#     sum_vals = absolute_mean_shap["MASV"].sum()
#     if sum_vals == 0:
#         warnings.warn(
#             message="All generated SHAP values are 0. Exiting...", category=Warning
#         )
#         return
#     absolute_mean_shap["Relative_ MASV"] = np.round(
#         (100 * absolute_mean_shap["MASV"] / absolute_mean_shap["MASV"].sum()), 2
#     )
#     # Ensure logic makes sense
#     assert np.isclose(
#         absolute_mean_shap["Relative_ MASV"].sum(), 100, atol=0.1
#     ), f"Sum is instead {absolute_mean_shap['Relative_ MASV'].sum()}"
#     # Reorder
#     absolute_mean_shap["Feature"] = absolute_mean_shap["Feature"].astype(str)

#     absolute_mean_shap_reordered = (
#         absolute_mean_shap.set_index("Feature").loc[feat_order].reset_index()
#     )
#     ######################## Display + Export ########################
#     if result_path:
#         result_path.parent.mkdir(exist_ok=True, parents=True)
#         with pd.ExcelWriter(
#             result_path,
#             engine="openpyxl",
#             mode="a" if result_path.exists() else "w",
#         ) as writer:
#             absolute_mean_shap_reordered.to_excel(
#                 writer, sheet_name=model_name, index=True
#             )


# def get_shap(*_, X, model_dict, outcome_name, feat_order, result_path=None):
#     """
#     Generates SHAP values with kernel explainer, builds mean absolute value (MAV) and relative MAV shap tables, and exports the tables

#     Parameters
#     ----------
#     X: pandas dataframe
#         Tabular dataframe containing data for which SHAP values will be generated (excluding target variable)
#         Test set is recommended for useful SHAP values, but can be any subset of the dataset
#     model_dict: dict()
#         Dictionary mapping model names to models
#         Format:
#             {
#                 <model_name> str: <model> sklearn model
#             }
#     outcome_name: str
#         Specify outcome whose models are being analyzed
#     feat_order: list[str]
#         List of column names specifying desired order in SHAP table
#     result_path: Optional pathlib.Path; defaults None
#         Path to directory where shap tables for all models will be written to
#         If left None, will not export

#     Raises
#     ------
#     ValueError:
#         If positional arguments are provided
#     """
#     if _ != tuple():
#         raise ValueError("This function does not take positional arguments")
#     if result_path and result_path.exists():
#         warnings.warn(
#             category=Warning,
#             message=f"Removing file at {result_path}. \nUnless this function fails, will over-write with new SHAP table",
#         )
#         result_path.unlink()
#     ######################## Get SHAP values ########################
#     for model_name, model in model_dict.items():
#         print(f"Working on model: {model_name}...")
#         explainer = shap.Explainer(
#             model.predict_proba, X, feature_names=X.columns.tolist()
#         )
#         shap_raw = explainer(X)
#         ######################## Deal with one-hot encoded ########################
#         ohe_dict = get_ohe_cols(X)
#         ohe_cols = ohe_dict.keys()
#         ### Get OHE feature order for raw
#         raw_feat_order = []
#         for col in feat_order:
#             if col in ohe_cols:
#                 for sub_col in ohe_dict[col]:
#                     raw_feat_order.append(f"{col}_{sub_col}")
#             else:
#                 raw_feat_order.append(col)
#         ### Combine ohe for combined
#         for col_name in ohe_cols:
#             shap_combined, _ = combine_encoded(
#                 shap_raw, col_name, [col_name in n for n in shap_raw.feature_names]
#             )
#         ######################## Generate + export MAV table ########################
#         if result_path:
#             raw_path = result_path / "raw" / f"{outcome_name}.xlsx"
#             combined_path = result_path / "combined" / f"{outcome_name}.xlsx"
#         generate_MAV(
#             shap_raw, raw_feat_order, model_name=model_name, result_path=raw_path
#         )
#         # generate_MAV(
#         #     shap_combined, feat_order, model_name=model_name, result_path=combined_path
#         # )

#         # # Reformat shap values
#         # shap_to_plot_combined = get_vals_to_plot(shap_combined)
#         # shap_df_combined = pd.DataFrame(shap_to_plot_combined.values, columns=combined_feat_names)

#         # shap_to_plot_raw = get_vals_to_plot(shap_raw)
#         # shap_df_raw = pd.DataFrame(shap_to_plot_raw.values, columns=raw_feat_names)

#         # # Get absolute avg + relative abs avg
#         # absolute_mean_shap = shap_df.abs().mean().reset_index()
#         # absolute_mean_shap.columns = ["Feature", "MASV"]
#         # sum_vals = absolute_mean_shap["MASV"].sum()
#         # if sum_vals == 0:
#         #     warnings.warn(
#         #         message="All generated SHAP values are 0. Exiting...", category=Warning
#         #     )
#         #     return
#         # absolute_mean_shap["Relative_ MASV"] = np.round(
#         #     (100 * absolute_mean_shap["MASV"] / absolute_mean_shap["MASV"].sum()), 2
#         # )
#         # # Ensure logic makes sense
#         # assert np.isclose(
#         #     absolute_mean_shap["Relative_ MASV"].sum(), 100, atol=0.1
#         # ), f"Sum is instead {absolute_mean_shap['Relative_ MASV'].sum()}"
#         # # Reorder
#         # absolute_mean_shap["Feature"] = absolute_mean_shap["Feature"].astype(str)

#         # absolute_mean_shap_reordered = (
#         #     absolute_mean_shap.set_index("Feature").loc[feat_order].reset_index()
#         # )

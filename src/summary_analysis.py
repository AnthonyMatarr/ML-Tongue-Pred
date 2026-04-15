from src.data_utils import get_feature_lists
from src.config import SEED
import sigfig
import warnings
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_numerical(df, numerical_cols):
    ## Impute
    imputer = IterativeImputer(
        estimator=None,  # default = BayesianRidge
        initial_strategy="median",
        imputation_order="random",
        max_iter=10,
        sample_posterior=False,  # deterministic
        random_state=SEED,
    )
    df_impute = df.copy()
    imputed_values = imputer.fit_transform(df[numerical_cols])
    df_impute[numerical_cols] = imputed_values
    assert df_impute.isna().sum().sum() == 0
    return df_impute


def generate_fish_list(df, outcome_dict, binary_cols, verbose=True):
    """
    Generates dictionary specifying, for each outcome, which features require fishers exact test due to low expected frequencies

    Returns
    -------
    Dictionary with format:
    {
        <outcome_name> str: List[<feat_name> str]
    }
    """
    fish_dict = {}
    for outcome_name, outcome in outcome_dict.items():
        if verbose:
            print(f"---------------{outcome_name}-------------------")
        fish_dict[outcome_name] = []
        for col in df:
            if col in binary_cols:
                contingency_table = pd.crosstab(df[col], outcome)
                _, _, _, expected_frequencies = chi2_contingency(contingency_table)
                if (expected_frequencies < 5).any():  # type: ignore
                    fish_dict[outcome_name].append(col)
                    if verbose:
                        print(col)
                        print(expected_frequencies)
                        print("-" * 20)
    return fish_dict


def generate_summary_column(
    df_impute,
    og_df,
    outcome,
    outcome_type,
    all_categories,
    feature_dict,
    set_type,
):
    """
    Generates a single column for the summary table. Usually one of [all, only_positive_instances, only_neg_instances]

    Parameters
    ----------
    df_impute: pd.DataFrame
        Dataframe for which summary values are to be generated. Usually a fully processed (imputed) dataset (excludes target variable).
    og_df: pd.DataFrame
        Original dataframe (before imputation), used only to extract missingness from numerical values
    outcome: TODO
    outcome_type: TODO
    all_categories: TODO
    feature_dict: dict
        Dictionary detailing the type of feature each column is (excluding target)
    """
    binary_cols = feature_dict["binary_cols"]
    numerical_cols = feature_dict["numerical_cols"]
    nominal_cols = feature_dict["nominal_cols"]
    ordinal_cols = feature_dict["ordinal_cols"]

    total_entries = len(df_impute)
    header = f"{outcome}-{outcome_type}  (n={total_entries})"
    summary_list = []
    for col in df_impute:  # Loop through all columns
        summary_list.append({"Feature": f"{col.upper()}", header: ""})
        if col in binary_cols + nominal_cols + ordinal_cols:  # Categorical
            # Get counts and percentages, add blank row for variable name
            counts = df_impute[col].value_counts()
            percentages = np.round(
                df_impute[col].value_counts(normalize=True) * 100, decimals=1
            )
            # For each instance in feature, add counts/percentages to df
            for entry in all_categories[col]:
                #### Special case for OperYR bc of seperation for year ####
                if set_type == "test" and col == "OPERYR":
                    continue
                elif set_type == "dev" and col == "OPERYR" and entry == 2024:
                    print(entry)
                    continue
                summary_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        # Get value count, if not existent, replace with 0
                        header: f"{counts.get(entry, 0)} ({percentages.get(entry, 0.0)})",
                    }
                )
        elif col in numerical_cols:  # Numerical
            # Get mean, stdev, quantiles
            avg = np.mean(df_impute[col])
            std = np.std(df_impute[col])
            quantiles = np.round(
                df_impute[col].quantile([0.25, 0.5, 0.75]).values.tolist(), 3
            )
            summary_list.append(
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header: f"{avg:.1f} ± {std:.1f} -- {quantiles}",
                }
            )
            n_missing = og_df[col].isna().sum()
            pct_missing = np.round(n_missing / total_entries * 100, 1)
            summary_list.append(
                {
                    "Feature": f"{col.upper()} Missing (% missing)",
                    header: f"{n_missing} ({pct_missing})",
                }
            )
        else:
            print(col)
            raise ValueError
    return pd.DataFrame(summary_list)


def generate_summary_table(
    *_,
    X_df_final,
    X_df_og,
    outcome_name,
    all_categories,
    set_type,
    feature_dict,
):
    """
        Parent/main function used to generate summary tables. Generates one column at a time (all, positives, negatives) and appends together.

        Parameters
        X_df_final: pandas dataframe
                Final tabular dataframe containing data (usually imputed) to be summarized
                INCLUDES target variable
        X_df_og: pandas dataframe
                Original tabular dataframe without any imputations ONLY used to calculate missingness
                INCLUDES target variable
        outcome_name: str
                Specifies name of target variable
        all_categories: dict
                Dictionary mapping feature names to all instances of that feature
    feature_dict: TODO
        Returns
        -------
        combined_df: pandas dataframe
            Columns:
                All/Total entries, Negative entries, Positive Entries
            Rows:
                Feature Names (+ feature instances if categorical)
            Values:
                Numerical Features: Mean +/- SD; [25%, 50%, 75%] quartiles

        Raises
        ------
        ValueError:
            If positional arguments are given
    """

    ## Get DFs of w/ and w/o outcome
    non_outcome_df = (
        X_df_final[X_df_final[outcome_name] == 0].drop(outcome_name, axis=1).copy()
    )
    non_outcome_df_og = (
        X_df_og[X_df_og[outcome_name] == 0].drop(outcome_name, axis=1).copy()
    )
    outcome_df = (
        X_df_final[X_df_final[outcome_name] == 1].drop(outcome_name, axis=1).copy()
    )
    outcome_df_og = (
        X_df_og[X_df_og[outcome_name] == 1].drop(outcome_name, axis=1).copy()
    )
    # Get total, neg-outcome, and pos-outcome summary tables
    total_col = generate_summary_column(
        X_df_final,
        X_df_og,
        outcome_name,
        "Total",
        all_categories,
        feature_dict,
        set_type,
    )
    non_outcome_col = generate_summary_column(
        non_outcome_df,
        non_outcome_df_og,
        outcome_name,
        "Negative",
        all_categories,
        feature_dict,
        set_type,
    )
    outcome_col = generate_summary_column(
        outcome_df,
        outcome_df_og,
        outcome_name,
        "Positive",
        all_categories,
        feature_dict,
        set_type,
    )

    ## Combine by feature column
    total_col = total_col.set_index("Feature")
    non_outcome_col = non_outcome_col.set_index("Feature")
    outcome_col = outcome_col.set_index("Feature")
    combined_df = pd.concat([total_col, non_outcome_col, outcome_col], axis=1)

    return combined_df


def add_dummy_rows(*_, df, col, result_list, header_ORs, header_p):
    """
    Creates dummy rows to be added to summary/analysis df in the event that a given feature should not be populated with values

    Parameters
    ---------
    df: pandas DataFrame
        Tabular dataframe containing both predictor and outcome variables
    col: string
        Name of the column for which dummy rows will be generated
    result_list: list[dict{}]
        List of dictionaries. Each dict pertains to a single row once converted into a dataframe
    header_ORs: string
        Header for Odds Ratios column in final summary/analysis dataframe
    header_p: string
        Header for p-value column in final summary/analysis dataframe
    """
    entries = sorted(df[col].unique())
    for entry in entries:
        entry_name = f"{col.upper()} {entry}"
        result_list.append(
            {
                "Feature": entry_name,
                header_ORs: "",
                header_p: "",
            }
        )
    return result_list


def format_p_val(p_val):
    """
    Re-formats p-values if <0.0001, and rounds otherwise. Converts to string
    """
    if p_val < 0.0001:
        return "<0.0001"
    elif p_val <= 0.05:
        return str(sigfig.round(p_val, sigfigs=2))
    else:
        return str(round(p_val, 1))


def get_cat_analysis(
    result_list,
    col,
    full_df,
    df_no_full,
    outcome_name,
    header_ORs,
    header_p,
    nominal_cols,
):
    """
    1) One-hot encode, excluding:
        Nominal: entry with highest frequency as a reference
        Ordinal: lowest-value entry (assuming no perfect seperation)
    2) attempt to id problematic/unstable entries and remove/mark unstable
    3) run log regression for p-val and ORs (CI)
        - returns early if logit fails (w/ all entries marked unstable)
    4) Return updated result_list

    """
    result_list.append({"Feature": col.upper(), header_ORs: "", header_p: ""})
    y = full_df[outcome_name].values
    # List of possible entries in a given column, sorted from low to HIGH
    entries = sorted(full_df[col].unique())
    # Create subset onehot-encoded temporary df
    temp_df = full_df[[col]]
    temp_df = pd.get_dummies(temp_df, columns=[col], drop_first=False, dtype=int)
    # If nominal, drop highest freq entry and make that reference
    if col in nominal_cols:
        val_counts = full_df[col].value_counts()
        max_freq_idx = val_counts.idxmax()  # Entry with the highest frequency
        drop_col = f"{col}_{max_freq_idx}"
    # If ordinal, drop the entry with the lowest value and make that reference
    else:
        # Ensure no perfect sep in reference instance
        ct = pd.crosstab(df_no_full[col], y)
        zero_entries = ct[ct[1] == 0].index.tolist()
        for i in range(len(entries)):
            drop_col = f"{col}_{entries[i]}"
            if entries[i] not in zero_entries:
                break
    temp_df.drop(drop_col, axis=1, inplace=True)
    X = sm.add_constant(temp_df)
    ## Identify problematic entries
    ct = pd.crosstab(df_no_full[col], y)
    zero_entries = ct[ct[1] == 0].index.tolist()
    bad_entry_list = [
        f"{col}_{entry}"
        for entry in zero_entries
        if f"{col}_{entry}" != drop_col and f"{col}_{entry}" in X.columns  # type: ignore
    ]
    if bad_entry_list:
        X = X.drop(bad_entry_list, axis=1)  # type: ignore
    model = None
    or_estimates = None
    conf_ints = None
    p_values = None

    # Try to run Logit
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = sm.Logit(y, X).fit(disp=0)
            or_estimates = np.exp(model.params)
            conf_ints = model.conf_int()
            p_values = model.pvalues
    # Except any issues with logit
    except (
        ConvergenceWarning,
        RuntimeWarning,
        ValueError,
        KeyError,
        OverflowError,
    ) as e:
        # Log the specific issue
        print(f"Failed to fit model for {col}: {type(e).__name__}: {e}")
        # Mark all entries as unstable
        for entry in entries:
            entry_name = f"{col}_{entry}"
            result_list.append(
                {
                    "Feature": f"{col.upper()} {entry}",
                    header_ORs: "UNSTABLE",
                    header_p: "UNSTABLE",
                }
            )
        return result_list  # Skip to next column
    if model is not None:
        # Loop through each possible entry of the feature
        for entry in entries:
            # Reformat to allow for indexing ex) 1.0 --> SITE_1.0
            entry_name = f"{col}_{entry}"
            if entry_name == drop_col:
                result_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        header_ORs: "Reference",
                        header_p: "Reference",
                    }
                )

            elif entry_name in bad_entry_list:
                result_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        header_ORs: "UNSTABLE",
                        header_p: "UNSTABLE",
                    }
                )
            # For BOTH: If not designated reference, get stat vals
            else:
                p_val_specific = format_p_val(p_values[entry_name])
                or_estimate = or_estimates.loc[entry_name]
                ci_lower = np.exp(conf_ints.loc[entry_name, 0])
                ci_upper = np.exp(conf_ints.loc[entry_name, 1])
                odds_conf = f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
                result_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        header_ORs: odds_conf,
                        header_p: p_val_specific,
                    }
                )
    return result_list


def get_analysis_df(
    *_, full_df, outcome_name, outcome_sub_cols, fish_dict, feature_dict, all_categories
):
    """
    Returns a dataframe containing p-values, odds ratios, and 95% CIs for a given outcome

    Parameters
    ----------
    full_df: pd.DataFrame
        Dataframe for which analysis values are being run on.
        Usually just the base tabular data (INCLUDES target variable)
    outcome_name: str
        String specifying the name of the given outcome (target variable)
    outcome_sub_cols: list
        List detailing raw columns used to create outcome col
    fish_dict: dict
        Dictionary containing, for each outcome, what features require fishers exact test. The return value of a call to generate_fish_list().
    feature_dict: TODO
    all_categories: TODO

    Raises:
    ------
    ValueError:
            If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not accept positional arguments!")
    ## Seperate outcome
    outcome_data = full_df[outcome_name]
    df_no_full = full_df.drop(outcome_name, axis=1)
    ## Set func globals
    binary_cols = feature_dict["binary_cols"]
    numerical_cols = feature_dict["numerical_cols"]
    nominal_cols = feature_dict["nominal_cols"]
    ordinal_cols = feature_dict["ordinal_cols"]
    header_ORs = "Odds Ratios (95% CI)"
    header_p = "P-Value"

    ##Analysis (p-vals + ORs)
    result_list = []
    ### Loop through all columns
    for col in full_df.columns:
        if col in outcome_sub_cols or col == outcome_name:
            result_list.append(
                {"Feature": f"{col.upper()}", header_ORs: "---", header_p: "---"}
            )
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
            continue
        elif col in binary_cols:

            contingency_table = pd.crosstab(df_no_full[col], outcome_data)
            if (contingency_table == 0).any().any():
                contingency_table += 1
            if col in fish_dict[outcome_name]:
                ## Fishers Exact for p-vals if expected freq < 5
                _, p_value = fisher_exact(contingency_table)
                result = odds_ratio(contingency_table, kind="conditional")
            else:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                result = odds_ratio(contingency_table, kind="sample")
            p_value = format_p_val(p_value)
            or_estimate = result.statistic
            ci_low, ci_high = result.confidence_interval(confidence_level=0.95)
            odds_conf = f"{or_estimate:.2f} ({ci_low:.2f}, {ci_high:.2f})"
            result_list.append(
                {"Feature": col.upper(), header_ORs: odds_conf, header_p: p_value}
            )
            ##Append empty rows just to ensure index matches with summary
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
        elif col in numerical_cols:
            ### Mann-Whitney U test for p-vals ###
            group1 = full_df[full_df[outcome_name] == 0][col]
            group2 = full_df[full_df[outcome_name] == 1][col]
            _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            p_value = format_p_val(p_value)
            ### Log Regresion for ORs and CIs ###
            X = sm.add_constant(full_df[col])
            y = full_df[outcome_name].values
            model = sm.Logit(y, X).fit(disp=0)
            or_estimate = np.exp(model.params[col])
            conf_int = model.conf_int().loc[col]
            ci_lower = np.exp(conf_int[0])
            ci_upper = np.exp(conf_int[1])
            logit_p = model.pvalues[col]
            logit_p = format_p_val(logit_p)
            odds_conf = f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
            list_adds = [
                {
                    "Feature": col.upper(),
                    header_ORs: odds_conf,
                    header_p: p_value,
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header_ORs: "",
                    header_p: "",
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + " Missing (% missing)",
                    header_ORs: "",
                    header_p: "",
                },
            ]
            result_list.extend(list_adds)
        elif col in nominal_cols + ordinal_cols:
            result_list = get_cat_analysis(
                result_list=result_list,
                col=col,
                full_df=full_df,
                df_no_full=df_no_full,
                outcome_name=outcome_name,
                header_ORs=header_ORs,
                header_p=header_p,
                nominal_cols=nominal_cols,
            )

    results_df = pd.DataFrame(result_list)
    return results_df.set_index("Feature")


def export_table(export_path, table):
    if export_path.exists():
        export_path.unlink()
        warnings.warn(f"Over-writing table at path {export_path}")
    export_path.parent.mkdir(exist_ok=True, parents=True)
    table.to_excel(export_path, index=True)


def get_summary_analysis(
    df, outcome_sub_cols_dict, col_order, set_type, export_path=None
):
    """
    Get summary + analysis of all variables

    df: pandas DF
        Contains all variables
    outcome_sub_cols_dict: dict{}
        Keys: outcome_name
        Values: Aggregated variables used to construct outcome
    col_order: list[str]
        Ordered list of variables
    set_type: str
        One of "dev", "test", "all"
    """
    #### SUBSET DF #####
    if set_type == "dev":
        df = df[df["OPERYR"] != 2024]
    elif set_type == "test":
        df = df[df["OPERYR"] == 2024]
        df = df.drop(["OPERYR"], axis=1)
        col_order = [col for col in col_order if col != "OPERYR"]
    elif set_type == "all":
        pass  # do nothing
    else:
        raise ValueError("Unrecognized summary type")

    ##### GET OUTCOME DICT
    outcome_dict = {}
    for col in outcome_sub_cols_dict.keys():
        outcome_dict[col] = df[col]
    ###N SUBSET COLS
    df_sub = df[col_order]
    ### Get feature groupings
    FEATURE_DICT = get_feature_lists(df_sub)
    binary_cols = FEATURE_DICT["binary_cols"]
    numerical_cols = FEATURE_DICT["numerical_cols"]
    nominal_cols = FEATURE_DICT["nominal_cols"]
    ordinal_cols = FEATURE_DICT["ordinal_cols"]
    if set_type == "dev":
        ## Make Year an ordinal variable instead of numerical --> just for this portion of proj
        numerical_cols.remove("OPERYR")
        ordinal_cols.append("OPERYR")
        FEATURE_DICT["numerical_cols"] = numerical_cols
        FEATURE_DICT["ordinal_cols"] = ordinal_cols

    #### GET ALL CATEGORIES
    all_categories = {}
    for col in nominal_cols + binary_cols + ordinal_cols:
        unique_entries = set(df_sub[col].unique())
        ## Dict: {column_names: [<unique_entries>]}
        all_categories[col] = unique_entries

    #### IMPUTE
    df_impute = impute_numerical(df=df_sub, numerical_cols=numerical_cols)
    #### GET FISH DICT (cols that req fischers exact)
    # Get features w/ expected freq < 5
    fish_dict = generate_fish_list(
        df_impute, outcome_dict, FEATURE_DICT["binary_cols"], verbose=False
    )
    ###### SUMMARY  + ANALYSIS PER OUTCOME ######
    final_tables = []
    full_len = len(df_impute)
    for i, outcome_name in enumerate(outcome_sub_cols_dict.keys()):
        print(f"{outcome_name}...")
        outcome_data = df_sub[outcome_name]
        assert len(outcome_data) == full_len
        ##Get summary
        summary_df = generate_summary_table(
            X_df_final=df_impute,
            X_df_og=df_sub,
            outcome_name=outcome_name,
            all_categories=all_categories,
            set_type=set_type,
            feature_dict=FEATURE_DICT,
        )
        print("\t Summary table done!")
        # Analysis- Df containing univariable values (p-values, ORs w/ CIs)
        analysis_df = get_analysis_df(
            full_df=df_impute,
            outcome_name=outcome_name,
            outcome_sub_cols=outcome_sub_cols_dict[outcome_name],
            fish_dict=fish_dict,
            feature_dict=FEATURE_DICT,
            all_categories=all_categories,
        )
        print("\t Analysis table done!")
        ## CHECKS
        try:
            assert set(analysis_df.index.to_list()) == set(summary_df.index.to_list())
        except AssertionError:
            print(set(analysis_df.index.to_list()) - set(summary_df.index.to_list()))
            print(set(summary_df.index.to_list()) - set(analysis_df.index.to_list()))
            raise AssertionError("Analysis and summary tables DO NOT match!")
        ## Append with summary
        final_table = summary_df.join(analysis_df, how="left").fillna("")
        # On first iteration, save the "all patients" column
        if i == 0:
            all_patients_col = final_table.iloc[:, 0].copy()  # first column
            n_total = len(df_impute)
            all_patients_col.name = f"Total Patients (n={n_total})"
            final_tables.append(all_patients_col)
        # Append only the outcome-specific columns (positive/negative)
        outcome_specific = final_table.iloc[:, 1:]  # all columns except first
        final_tables.append(outcome_specific)
    wide_table = pd.concat(final_tables, axis=1)
    if export_path is not None:
        export_table(export_path, wide_table)

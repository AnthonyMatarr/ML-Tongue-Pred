from shutil import rmtree
import numpy as np
import pandas as pd


##############################################################################
################################### CLEAN ####################################
##############################################################################
def get_code_cols(df, include_cpt=False):
    if include_cpt:
        code_cols = [
            col for col in df if (("PODIAG" in col or "CPT" in col) and col != "CPT")
        ]
    else:
        code_cols = [col for col in df if ("PODIAG" in col or "CPT" in col)]
    return code_cols


def combine_columns(row):
    """Combine 5 columns with hierarchy: Yes > No > NaN"""
    # Check if any value is "Yes"
    if row.isin(["Yes", "Ye"]).any():
        return "Yes"
    # Check if any value is "No"
    elif (row == "No").any():
        return "No"
    # All values are "NULL" or NaN
    else:
        return np.nan


def clean_08_10(df, include_cpt):
    df_w_codes = df.copy()
    ######################################
    ################ 2008 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    df_w_codes.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    df_w_codes.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## Readmission
    df_w_codes["ReAd"] = None
    ## Unplanned Readmission
    df_w_codes["UnplReAd"] = None
    ## Discharge Dest
    df_w_codes["DISCHDEST"] = None
    df_w_codes.reset_index(drop=True, inplace=True)
    # No code df
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_11(df, include_cpt):
    df_w_codes = df.copy()
    df_w_codes.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unpl_read_cols = ["RETURNOR", "REOPERATION"]
    df_w_codes["UnplReOp"] = df_w_codes[unpl_read_cols].apply(combine_columns, axis=1)
    df_w_codes.drop(unpl_read_cols, axis=1, inplace=True)
    ## Readmission
    df_w_codes.rename(columns={"READMISSION": "ReAd"}, inplace=True)
    ## Unplanned Readmission
    df_w_codes["UnplReAd"] = None
    df_w_codes.reset_index(drop=True, inplace=True)
    ## DROP codes
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_12_14(df, include_cpt):
    df_w_codes = df.copy()
    # Note that this is eventually called for 21-24
    # 2021-2024 dfs have no EMERGNCY col and
    # clean_22_24 renames `CASETYPE` --> `Urgency`, but this won't raise error
    df_w_codes.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    df_w_codes["UnplReOp"] = df_w_codes[unplanned_reop_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    df_w_codes["ReAd"] = df_w_codes[read_cols].apply(combine_columns, axis=1)
    df_w_codes.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    df_w_codes["UnplReAd"] = df_w_codes[unplanned_read_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_read_cols, axis=1, inplace=True)
    df_w_codes.reset_index(drop=True, inplace=True)
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_15_20(df, include_cpt):
    """
    Clean for years 15-20
    Note that eventually called from clean_21() & clean_22_24()
        - clean_22_24() renames CASETYPE --> EMERGENCY but will not raise error here
        - clean_21() calls clean_22_24() so will also do this^

        - years 22-24 also do not need to rename BLEEDIS--> BLEEDDIS, but will not raise error here either
    """
    df_w_codes = df.copy()
    # NOTE: Eventually called for 21, 22-24, who have CASETYPE instead of EMERGNCY
    # they did CASETYPE--> Urgency, but renaming will not raise error here for them
    df_w_codes.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    df_w_codes.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    ## Unplanned ReOp --> one less col than 12-14
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    df_w_codes["UnplReOp"] = df_w_codes[unplanned_reop_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    df_w_codes["ReAd"] = df_w_codes[read_cols].apply(combine_columns, axis=1)
    df_w_codes.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    df_w_codes["UnplReAd"] = df_w_codes[unplanned_read_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_read_cols, axis=1, inplace=True)
    df_w_codes.reset_index(drop=True, inplace=True)
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_22_24(df, include_cpt):
    df_w_codes = df.copy()
    ## Same as 15-20 w/ addition of:
    # 1) CASETYPE instead of EMERGNCY
    df_w_codes.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    # 2) adding missing cols
    df_w_codes["WTLOSS"] = None
    df_w_codes["WNDINF"] = None
    df_w_codes["DYSPNEA"] = None

    return clean_15_20(df_w_codes, include_cpt)


def clean_21(df, include_cpt):
    ## Same as 22-24 w/ addition of adding missing cols
    df_w_codes = df.copy()
    ## Add missing cols
    df_w_codes["RENAINSF"] = None
    df_w_codes["RENAFAIL"] = None
    return clean_22_24(df_w_codes, include_cpt)


def merge_dfs(data_dict, include_cpt, verbose=False):
    """
    Merges NSQIP dataframes from 2008-2024, normalizing values to append vertically

    Parameters
    ----------
    data_dict: dict
        Dictionary mapping NSQIP file name to pandas df
    """
    w_codes_dict = {}  # fill w/ cleaned data (including code cols)
    no_codes_dict = {}  # fill w/ cleaned data (excluding code cols)
    ################# #####################
    ################ 2008 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    df_w_codes_08, df_no_codes_08 = clean_08_10(data_dict["NSQIP_08_cpt"], include_cpt)
    w_codes_dict["08"] = df_w_codes_08
    no_codes_dict["08"] = df_no_codes_08
    if verbose:
        print(no_codes_dict["08"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2009 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    df_w_codes_09, df_no_codes_09 = clean_08_10(data_dict["NSQIP_09_cpt"], include_cpt)
    w_codes_dict["09"] = df_w_codes_09
    no_codes_dict["09"] = df_no_codes_09
    if verbose:
        print(no_codes_dict["09"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2010 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    df_w_codes_10, df_no_codes_10 = clean_08_10(data_dict["NSQIP_10_cpt"], include_cpt)
    w_codes_dict["10"] = df_w_codes_10
    no_codes_dict["10"] = df_no_codes_10
    if verbose:
        print(no_codes_dict["10"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2011 ################
    ######################################
    # {'UNPLREAD'}
    df_w_codes_11, df_no_codes_11 = clean_11(data_dict["NSQIP_11_cpt"], include_cpt)
    w_codes_dict["11"] = df_w_codes_11
    no_codes_dict["11"] = df_no_codes_11
    if verbose:
        print(no_codes_dict["11"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2012 ################
    ######################################
    df_w_codes_12, df_no_codes_12 = clean_12_14(data_dict["NSQIP_12_cpt"], include_cpt)
    w_codes_dict["12"] = df_w_codes_12
    no_codes_dict["12"] = df_no_codes_12
    if verbose:
        print(no_codes_dict["12"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2013 ################
    ######################################
    df_w_codes_13, df_no_codes_13 = clean_12_14(data_dict["NSQIP_13_cpt"], include_cpt)
    w_codes_dict["13"] = df_w_codes_13
    no_codes_dict["13"] = df_no_codes_13
    if verbose:
        print(no_codes_dict["13"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2014 ################
    ######################################
    df_w_codes_14, df_no_codes_14 = clean_12_14(data_dict["NSQIP_14_cpt"], include_cpt)
    w_codes_dict["14"] = df_w_codes_14
    no_codes_dict["14"] = df_no_codes_14
    if verbose:
        print(no_codes_dict["14"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2015 ################
    ######################################
    df_w_codes_15, df_no_codes_15 = clean_15_20(data_dict["NSQIP_15_cpt"], include_cpt)
    w_codes_dict["15"] = df_w_codes_15
    no_codes_dict["15"] = df_no_codes_15
    if verbose:
        print(no_codes_dict["15"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2016 ################
    ######################################
    df_w_codes_16, df_no_codes_16 = clean_15_20(data_dict["NSQIP_16_cpt"], include_cpt)
    w_codes_dict["16"] = df_w_codes_16
    no_codes_dict["16"] = df_no_codes_16
    if verbose:
        print(no_codes_dict["16"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2017 ################
    ######################################
    df_w_codes_17, df_no_codes_17 = clean_15_20(data_dict["NSQIP_17_cpt"], include_cpt)
    w_codes_dict["17"] = df_w_codes_17
    no_codes_dict["17"] = df_no_codes_17
    if verbose:
        print(no_codes_dict["17"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2018 ################
    ######################################
    df_w_codes_18, df_no_codes_18 = clean_15_20(data_dict["NSQIP_18_cpt"], include_cpt)
    w_codes_dict["18"] = df_w_codes_18
    no_codes_dict["18"] = df_no_codes_18
    if verbose:
        print(no_codes_dict["18"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2019 ################
    ######################################
    df_w_codes_19, df_no_codes_19 = clean_15_20(data_dict["NSQIP_19_cpt"], include_cpt)
    w_codes_dict["19"] = df_w_codes_19
    no_codes_dict["19"] = df_no_codes_19
    if verbose:
        print(no_codes_dict["19"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2020 ################
    ######################################
    df_w_codes_20, df_no_codes_20 = clean_15_20(data_dict["NSQIP_20_cpt"], include_cpt)
    w_codes_dict["20"] = df_w_codes_20
    no_codes_dict["20"] = df_no_codes_20
    if verbose:
        print(no_codes_dict["20"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2021 ################
    ######################################
    # {'RENAINSF', 'WTLOSS', 'WNDINF', 'RENAFAIL', 'DYSPNEA'}
    df_w_codes_21, df_no_codes_21 = clean_21(data_dict["NSQIP_21_cpt"], include_cpt)
    w_codes_dict["21"] = df_w_codes_21
    no_codes_dict["21"] = df_no_codes_21
    if verbose:
        print(no_codes_dict["21"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2022 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    df_w_codes_22, df_no_codes_22 = clean_22_24(data_dict["NSQIP_22_cpt"], include_cpt)
    w_codes_dict["22"] = df_w_codes_22
    no_codes_dict["22"] = df_no_codes_22
    if verbose:
        print(no_codes_dict["22"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2023 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    df_w_codes_23, df_no_codes_23 = clean_22_24(data_dict["NSQIP_23_cpt"], include_cpt)
    w_codes_dict["23"] = df_w_codes_23
    no_codes_dict["23"] = df_no_codes_23
    if verbose:
        print(no_codes_dict["23"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2024 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    df_w_codes_24, df_no_codes_24 = clean_22_24(data_dict["NSQIP_24_cpt"], include_cpt)
    w_codes_dict["24"] = df_w_codes_24
    no_codes_dict["24"] = df_no_codes_24
    if verbose:
        print(no_codes_dict["24"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ##################################################
    ########### ENSURE we did things right ###########
    ##################################################
    ###### ENSURE we did things right #####
    try:  ### Right number of dfs
        assert len(w_codes_dict) == len(data_dict)
        assert len(w_codes_dict) == len(no_codes_dict)
    except AssertionError:
        print("Dicts do not match in size...")
        print(f" New length w/ codes: {len(w_codes_dict)}")
        print(f" New length w/o codes: {len(no_codes_dict)}")
        print(f" OG length: {len(data_dict)}")
        raise AssertionError
    for year1, df1 in no_codes_dict.items():
        if (not include_cpt) and df1.shape[1] != 79:
            raise ValueError(f"Expected 79 rows, got {df1.shape[1]} instead")
        elif include_cpt and df1.shape[1] != 80:
            raise ValueError(f"Expected 80 rows, got {df1.shape[1]} instead")
        for year2, df2 in no_codes_dict.items():
            if year1 == year2:  # no need to compare the same dataset
                continue
            cols_1 = set(df1.columns)
            cols_2 = set(df2.columns)
            try:
                assert cols_2 - cols_1 == set()
                assert cols_1 - cols_2 == set()
            except AssertionError:
                print(f"In {year1} but not {year2}: {cols_1-cols_2}")
                print(f"In {year2} but not {year1}: {cols_2-cols_1}")
                raise AssertionError("Columns do not match in dfs!")
    ##### Combine
    combined_df_no_codes = pd.concat(no_codes_dict.values(), ignore_index=True)
    combined_df_w_codes = pd.concat(w_codes_dict.values(), ignore_index=True)
    print(f"Combined Shape No Codes: {combined_df_no_codes.shape}")
    print(f"Combined Shape With Codes: {combined_df_w_codes.shape}")
    return combined_df_no_codes, combined_df_w_codes


##############################################################################
################################## FILTER ####################################
##############################################################################
def extract_cols(import_df, new_cols_dict, target_cols_list, cpt_flag):
    """
    Extract binary indicator columns based on exact or prefix code matches.

    Creates new binary columns by searching for CPT or ICD codes across multiple
    source columns. Supports both exact matching and prefix matching depending on
    the code type and specified match strategy.

    Parameters
    ----------
    import_df : pd.DataFrame
        Raw tabular dataframe containing columns with CPT/ICD codes to search.
    new_cols_dict : dict of {str: list of tuple}
        Maps new column names to lists of (code, match_type) tuples, where:
        - code: CPT or ICD code to match (str or numeric)
        - match_type: Either 'exact' or 'prefix'
    target_cols_list : list of str
        Column names containing CPT/ICD codes to search within.
    cpt_flag : bool
        If True, applies CPT-specific normalization (converts to integer strings).
        If False, applies ICD-specific normalization (uppercase string conversion).

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with additional binary indicator columns (0/1)
        for each key in new_cols_dict.

    Notes
    -----
    - Missing values in target columns are treated as empty strings
    - CPT codes are converted to integer strings for normalization
    - ICD codes are uppercased for case-insensitive matching
    - A row receives 1 if any target column matches any specified code
    """
    df = import_df.copy()

    ## Normalize columns (make string and upper case; also make NA empty '')
    for col in target_cols_list:
        df[col] = df[col].fillna("")
        if cpt_flag:
            try:
                df[col] = df[col].astype(float).astype(int).astype(str)
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(str).str.upper()
    ## Check for matches
    for new_col, target_codes in new_cols_dict.items():
        # Exact matches (normalize in process)
        exact_codes = [
            str(code).upper()
            for code, match_type in target_codes
            if match_type == "exact"
        ]
        # Prefix matches (normalize in process)
        prefix_codes = [
            str(code).upper()
            for code, match_type in target_codes
            if match_type == "prefix"
        ]

        df[new_col] = (
            df[target_cols_list]
            .apply(
                lambda col: (
                    col.isin(exact_codes)
                    | col.str.startswith(tuple(prefix_codes), na=False)
                )
            )
            .any(axis=1)
            .astype(int)
        )
    return df


def create_and_filter_new_cols(
    *_,
    new_col_dict,
    old_df_dict,
    export_dir,
    target_cols,
    target_code_cols,
    filter_cols,
    cpt_flag,
    combine_col_name=None,
    cols_to_combine=None,
):
    """
    Create binary indicator columns from CPT/ICD codes, filter patients, and export results.

    This function processes multiple dataframes (typically by year) to:
    1. Extract binary columns based on code matches
    2. Filter patients who have at least one specified code
    3. Optionally combine mutually exclusive columns into a single categorical column
    4. Export filtered dataframes to parquet files

    Parameters
    ----------
    *_ : tuple
        Placeholder to prevent positional arguments (raises ValueError if used).
    new_col_dict : dict of {str: list of tuple}
        Maps new column names to lists of (code, match_type) tuples.
        Passed directly to extract_cols().
    old_df_dict : dict of {str: pd.DataFrame}
        Maps file identifiers (e.g., year labels) to their corresponding dataframes.
    export_dir : pathlib.Path
        Directory path where filtered dataframes will be exported as parquet files.
        Will be recreated if it already exists.
    target_cols : list of str
        Column names to retain from the original dataframe. Used to subset data
        and reduce computational overhead. Should include features, outcome columns,
        and newly created indicator columns.
    target_code_cols : list of str
        Column names containing CPT/ICD codes to search within for code matching.
    filter_cols : list of str
        Newly created binary columns used to filter patients. Patients are retained
        if they have at least one non-zero value in these columns.
    cpt_flag : bool
        If True, processes CPT codes. If False, processes ICD codes.
        Determines normalization strategy and export file naming.
    combine_col_name : str, optional
        Name for the combined categorical column. If provided, cols_to_combine
        must also be specified.
    cols_to_combine : list of str, optional
        Mutually exclusive binary columns to combine into a single categorical column.
        The combined column will contain the column name where the indicator is 1,
        or NA if none are present.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Maps file identifiers to filtered dataframes with new indicator columns.

    Raises
    ------
    ValueError
        If positional arguments are provided.
    ValueError
        If cols_to_combine are not mutually exclusive (more than one is 1 in any row).

    Notes
    -----
    - Column names are automatically uppercased for consistency
    - The function handles year-to-year variations in NSQIP column availability
    - Export files are named with pattern: {file_name}_{cpt|icd}.parquet
    - The export directory is deleted and recreated on each run
    - Progress and patient counts are printed for each file processed
    """

    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if cpt_flag:
        filter_type = "cpt"
    else:
        filter_type = "icd"
    new_df_dict = {}
    total_patients = 0
    ## Deal with export dir
    if export_dir.exists():
        rmtree(export_dir)
    export_dir.mkdir(exist_ok=True, parents=True)
    ## Loop through original dict of data files
    for file_name, file in old_df_dict.items():
        print(f"Working on {file_name}...")
        print(f"\t Initial number of patients: {len(file)}")
        ###########################################################
        ##################### Extract Cols ########################
        ###########################################################
        ## Make all columns upper case
        file.columns = file.columns.str.upper()
        # Subset df and col lists to match current df (can differ from year-year in NSQIP)
        # subset df on all cols
        df_sub = file[file.columns.intersection(target_cols)].copy()
        # get relevant code (ICD/CPT) cols
        target_code_cols_sub = df_sub.columns.intersection(target_code_cols)
        # Ensure code (ICD/CPT) cols are string in df
        df_sub[target_code_cols_sub] = df_sub[target_code_cols_sub].astype("string")
        ## Create new columns
        df_w_new_cols = extract_cols(
            df_sub, new_col_dict, target_code_cols_sub, cpt_flag=cpt_flag
        )

        ###########################################################
        ######################## Filter ###########################
        ###########################################################
        # convert to string first
        df_filtered = df_w_new_cols[
            df_w_new_cols[filter_cols].astype(str).ne("0").any(axis=1)
        ]
        total_patients += len(df_filtered)  # add total patients
        print(f"\t Remaining: {len(df_filtered)}")
        ###########################################################
        ################# Combine (optionally) ####################
        ###########################################################
        if combine_col_name is not None and cols_to_combine is not None:
            df_filtered = df_filtered.copy()
            ## Ensure mutually exlusive
            nonzero_count = df_filtered[cols_to_combine].astype(bool).sum(axis=1)
            is_mutually_exclusive = (nonzero_count <= 1).all()
            if not is_mutually_exclusive:
                raise ValueError(
                    f"The columns {cols_to_combine} are not mutually exclusive in {file_name}."
                )

            def combine_func(row):
                for col in cols_to_combine:
                    if str(row[col]) != "0":
                        return col
                return pd.NA  # No code present

            df_filtered.loc[:, combine_col_name] = df_filtered.apply(
                combine_func, axis=1
            )
            df_filtered = df_filtered.drop(cols_to_combine, axis=1)
        ###########################################################
        #################### Save + Export #######################
        ###########################################################
        new_df_dict[file_name] = df_filtered
        # Export
        export_path = export_dir / f"{file_name}_{filter_type}.parquet"
        df_filtered.to_parquet(export_path)

    print("*" * 30)
    print("*" * 30)
    print("*" * 30)
    print(f"TOTAL remaining patients post-{filter_type} filtering: {total_patients}")
    return new_df_dict

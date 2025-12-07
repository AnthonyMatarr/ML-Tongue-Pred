from shutil import rmtree
import numpy as np
import pandas as pd


##############################################################################
################################### CLEAN ####################################
##############################################################################
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


def merge_dfs(data_dict, verbose=False):
    """
    Merges NSQIP dataframes from 2008-2023, normalizing values to append vertically

    Parameters
    ----------
    data_dict: dict
        Dictionary mapping NSQIP file name to pandas df
    """
    data_dict_clean = {}
    no_codes_dict = {}
    ######################################
    ################ 2008 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_08_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## Readmission
    temp_df["ReAd"] = None
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["08"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["08"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["08"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2009 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_09_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## Readmission
    temp_df["ReAd"] = None
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["09"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["09"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["09"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2010 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_10_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## ReAd
    temp_df["ReAd"] = None
    ## Unplanned ReAd
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["10"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["10"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["10"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2011 ################
    ######################################
    # {'UNPLREAD'}
    temp_df = data_dict["NSQIP_11_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unpl_read_cols = ["RETURNOR", "REOPERATION"]
    temp_df["UnplReOp"] = temp_df[unpl_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unpl_read_cols, axis=1, inplace=True)
    ## Readmission
    temp_df.rename(columns={"READMISSION": "ReAd"}, inplace=True)
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["11"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["11"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["11"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2012 ################
    ######################################
    temp_df = data_dict["NSQIP_12_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["12"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["12"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["12"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2013 ################
    ######################################
    temp_df = data_dict["NSQIP_13_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["13"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["13"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["13"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2014 ################
    ######################################
    temp_df = data_dict["NSQIP_14_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["14"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["14"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["14"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2015 ################
    ######################################
    temp_df = data_dict["NSQIP_15_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["15"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["15"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["15"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2016 ################
    ######################################
    temp_df = data_dict["NSQIP_16_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["16"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["16"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["16"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2017 ################
    ######################################
    temp_df = data_dict["NSQIP_17_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["17"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["17"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["17"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2018 ################
    ######################################
    temp_df = data_dict["NSQIP_18_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["18"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["18"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["18"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2019 ################
    ######################################
    temp_df = data_dict["NSQIP_19_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["19"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["19"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["19"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2020 ################
    ######################################
    temp_df = data_dict["NSQIP_20_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["20"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["20"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["20"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2021 ################
    ######################################
    # {'RENAINSF', 'WTLOSS', 'WNDINF', 'RENAFAIL', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_21_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    ## Not included
    temp_df["RENAINSF"] = None
    temp_df["RENAFAIL"] = None
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["21"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["21"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["21"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2022 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_22_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["22"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["22"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["22"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2023 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_23_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["23"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["23"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["23"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2024 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_24_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["24"] = temp_df.drop(temp_drop_cols, axis=1)
    data_dict_clean["24"] = temp_df
    if verbose:
        print(no_codes_dict["24"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ###### ENSURE we did things right #####
    try:
        assert len(data_dict_clean) == len(data_dict)
        assert len(data_dict_clean) == len(no_codes_dict)
    except AssertionError:
        print("Dicts do not match in size...")
        print(f" New length w/ codes: {len(data_dict_clean)}")
        print(f" New length w/o codes: {len(no_codes_dict)}")
        print(f" OG length: {len(data_dict)}")
        raise AssertionError
    for year1, df1 in no_codes_dict.items():
        try:
            assert df1.shape[1] == 79
        except AssertionError:
            raise AssertionError(f"Expected 79 rows, got {df1.shape[1]} instead")
        for year2, df2 in no_codes_dict.items():
            cols_1 = set(df1.columns)
            cols_2 = set(df2.columns)
            assert cols_2 - cols_1 == set()
            assert cols_1 - cols_2 == set()
    ##### Combine
    combined_df_no_codes = pd.concat(no_codes_dict.values(), ignore_index=True)
    combined_df_w_codes = pd.concat(data_dict_clean.values(), ignore_index=True)
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

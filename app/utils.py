from pathlib import Path
import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).parent.parent


def load_population_probs(outcome):
    df = pd.read_parquet(BASE_PATH / "app" / "all_preds" / f"{outcome}.parquet")
    return df["prob"].values, df["label"].values


def load_bin_thresholds(outcome_name):
    """
    Loads bin thresholds for a given outcome/model from .npz files.
    Returns an array of bin edges.
    """
    thresholds_path = BASE_PATH / "app" / "bin_thresholds" / f"{outcome_name}.npz"
    npz_data = np.load(thresholds_path)
    return npz_data["thresholds"]


def get_risk_category(prob, outcome):
    """Assign outcome-specific risk category."""
    thresholds = load_bin_thresholds(outcome)

    if prob < thresholds[0]:
        return "Very Low", "🟢"
    elif prob < thresholds[1]:
        return "Low", "🟡"
    elif prob < thresholds[2]:
        return "Moderate", "🟠"
    else:
        return "High", "🔴"


def transform_yes_no(input_val):
    if input_val == "Yes":
        return 1
    elif input_val == "No":
        return 0
    else:
        raise ValueError(f"Invalid input: {input_val}. Expected 'Yes' or 'No'")


def transform_sex(input_val):
    if input_val == "Male":
        return 1
    elif input_val == "Female":
        return 0
    else:
        raise ValueError(f"Invalid input: {input_val}. Expected 'Male' or 'Female'")


def transform_hispanic(input_val):
    if input_val == "Hispanic":
        return 1
    elif input_val == "Not Hispanic/Unknown":
        return 0
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Hispanic' or 'Not Hispanic/Unknown'"
        )


def transform_unknown_other(input_val):
    if input_val == "Unknown/Other":
        return "Unknown_Other"
    else:
        return input_val


def transform_tumor_site(input_val):
    match input_val:
        case "Anterior two-thirds":
            return "Malignant neoplasm of anterior two-thirds of tongue unspecified"
        case "Base":
            return "Malignant neoplasm of base of tongue"
        case "Border":
            return "Malignant neoplasm of border of tongue"
        case "Junctional Zone":
            return "Malignant neoplasm of junctional zone of tongue"
        case "Surface":
            return "Malignant neoplasm of surface of tongue"
        case "Unspecified":
            return "Malignant neoplasm of tongue unspecified"
        case _:
            raise ValueError(f"Invalid input: {input_val}.")


def transform_func_status(input_val):
    if input_val == "Independent":
        return 1
    elif input_val == "Dependent":
        return 0
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Independent' or 'Dependent'"
        )


def transform_inout(input_val):
    if input_val == "Inpatient":
        return 1
    elif input_val == "Outpatient":
        return 0
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Inpatient' or 'Outpatient'"
        )


def transform_casetype(input_val):
    if input_val == "Urgent/Emergent":
        return "Urgent_Emergent"
    elif input_val in ["Unknown", "Elective"]:
        return input_val
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Urgent/Emergent', 'Elective', or 'Unknown'"
        )


def transform_ASA(input_val):
    match input_val:
        case "1-No Disturbance":
            return "1-No Disturb"
        case "2-Mild Disturbance":
            return "2-Mild Disturb"
        case "3-Severe Disturbance":
            return "3-Severe Disturb"
        case "4-Life Threatening Disturbance":
            return "4-Life Threat"
        case _:
            raise ValueError(f"Invalid input: {input_val}")

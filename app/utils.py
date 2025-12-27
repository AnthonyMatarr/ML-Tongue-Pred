from pathlib import Path
import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).parent.parent
from app.config import CHOSEN_MODEL_DICT


def load_population_probs(outcome_name):
    df = pd.read_parquet(
        BASE_PATH
        / "app"
        / "all_preds"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.parquet"
    )
    return df["prob"].values, df["label"].values


def load_bin_thresholds(outcome_name):
    """
    Loads bin thresholds for a given outcome/model from .npz files.
    Returns an array of bin edges.
    """
    thresholds_path = (
        BASE_PATH
        / "app"
        / "bin_thresholds"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.npz"
    )
    npz_data = np.load(thresholds_path)
    return npz_data["thresholds"]


def bin_occur_rates(outcome, thresholds):
    probs, true = load_population_probs(outcome)
    # thresholds = load_bin_thresholds(outcome)
    n_bins = len(thresholds) + 1
    bin_indices = np.digitize(probs, thresholds, right=False)  # type: ignore
    event_rates = []
    counts = []
    for b in range(n_bins):
        mask = bin_indices == b
        n = mask.sum()
        counts.append(n)
        if n == 0:
            event_rates.append(np.nan)
        else:
            event_rates.append(true[mask].mean())
    return event_rates


def get_risk_category(prob, outcome):
    """Assign outcome-specific risk category with emoji and color code."""
    thresholds = load_bin_thresholds(outcome)

    if prob < thresholds[0]:
        return "Very Low", "🟢", "#0ebd0d"  # Green
    elif prob < thresholds[1]:
        return "Low", "🟡", "#ffd401"  # Yellow
    elif prob < thresholds[2]:
        return "Moderate", "🟠", "#ee9410"  # Orange
    else:
        return "High", "🔴", "#c21615"  # Red


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
        case "Lingual Tonsil":
            return "Malignant neoplasm of lingual tonsil"
        case "Unspecified":
            return "Malignant neoplasm of tongue unspecified"
        case _:
            raise ValueError(f"Invalid input: {input_val}.")


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
        return 1
    elif input_val in ["Elective"]:
        return 0
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Urgent/Emergent', 'Elective', or 'Unknown'"
        )


def transform_ASA(input_val):
    input_val = input_val.strip()
    match input_val:
        case "I":
            return "1-No Disturb"
        case "II":
            return "2-Mild Disturb"
        case "III":
            return "3-Severe Disturb"
        case "IV/V":
            return "4/5-Life Threat/Moribund"
        case _:
            raise ValueError(f"Invalid input: {input_val}")


def transform_yes_no_unknown(input_val, col_name):
    if input_val in ["Yes", "No"]:
        return input_val
    elif input_val == "Unknown":
        if col_name == "RENAFAIL":
            yr = "21"
        elif col_name in ["DYSPNEA", "WNDINF", "WTLOSS"]:
            yr = "21-24"
        else:
            raise ValueError(
                f"Unrecognized column name for transforming yes/no/unknown: {col_name}"
            )
        return f"Unknown({yr})"
    else:
        raise ValueError(
            f"Unrecognized entry for transforming yes/no/unknown: {input_val}"
        )


def transform_unknown_other(input_val):
    if "UNKNOWN" in input_val.upper():
        return "otherUnknown"
    else:
        return input_val

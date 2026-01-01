from src.feat_importance import get_ohe_cols, combine_encoded
import copy
import numpy as np
import pandas as pd
import streamlit as st


def split_catgeorical(feature_name, model_value):
    try:
        code_str_split = str(model_value).split("_")
        if feature_name in ["ETHNICITY_HISPANIC", "RACE_NEW"]:
            colname = "_".join(code_str_split[:2])
            code_val = code_str_split[2]
        else:
            colname = code_str_split[0]
            code_val = code_str_split[1]
        try:
            assert colname == feature_name
        except AssertionError:
            raise ValueError(
                f"Split column name: {colname} does not match passed in column name: {feature_name}"
            )
        return code_val
    except IndexError:
        raise IndexError(f"Feat name: {feature_name}, value: {model_value}")


# --- inverse of transform_yes_no ---
def inv_yes_no(code):
    code = int(code)
    if code == 1:
        return "Yes"
    if code == 0:
        return "No"
    raise ValueError(f"Invalid yes/no code: {code}")


# --- inverse of transform_sex ---
def inv_sex(code):
    code = int(code)
    if code == 1:
        return "Male"
    if code == 0:
        return "Female"
    raise ValueError(f"Invalid sex code: {code}")


# --- inverse of transform_hispanic ---
def inv_hispanic(code):
    code = int(code)
    if code == 1:
        return "Hispanic"
    if code == 0:
        return "Not Hispanic/Unknown"
    raise ValueError(f"Invalid ETHNICITY_HISPANIC code: {code}")


# --- inverse of transform_yes_no_unknown ---
def inv_yes_no_unknown(feature_name, code_str):
    code_str_parsed = split_catgeorical(feature_name, code_str)
    if code_str_parsed in ["Yes", "No"]:
        return code_str_parsed
    if "UNKNOWN" in code_str_parsed.upper():
        return "Unknown"
    raise ValueError(
        f"Invalid yes/no/unknown code for {feature_name}: {code_str}->{code_str_parsed}"
    )


def inv_tumor_site(feature_name, code_str):
    code_str_parsed = split_catgeorical(feature_name, code_str)
    match code_str_parsed:
        case "Malignant neoplasm of anterior two-thirds of tongue unspecified":
            return "Anterior two-thirds"
        case "Malignant neoplasm of base of tongue":
            return "Base"
        case "Malignant neoplasm of border of tongue":
            return "Border"
        case "Malignant neoplasm of junctional zone of tongue":
            return "Junctional Zone"
        case "Malignant neoplasm of surface of tongue":
            return "Surface"
        case "Malignant neoplasm of lingual tonsil":
            return "Lingual Tonsil"
        case "Malignant neoplasm of tongue unspecified":
            return "Unspecified"
        case _:
            raise ValueError(
                f"Invalid yes/no/unknown code for {feature_name}: {code_str}->{code_str_parsed}"
            )


# --- inverse of transform_inout ---
def inv_inout(code):
    code = int(code)
    if code == 1:
        return "Inpatient"
    if code == 0:
        return "Outpatient"
    raise ValueError(f"Invalid INOUT code: {code}")


# --- inverse of transform_casetype ---
def inv_casetype(code):
    code = int(code)
    if code == 0:
        return "Elective"
    if code == 1:
        return "Urgent/Emergent"
    raise ValueError(f"Invalid URGENCY code: {code}")


# --- inverse of transform_asa ---
def inv_asa(entry):
    m = {
        0: "I",
        1: "II",
        2: "III",
        3: "IV/V",
    }
    try:
        return m[entry]
    except KeyError:
        raise ValueError(f"Invalid ASA code: {entry}")


# --- inverse of transform_race ---
def inv_race(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    m = {
        "White": "White",
        "Black or African American": "Black or African American",
        "Asian": "Asian",
        "American Indian or Alaska Native": "American Indian/Alaska Native",
        "Native Hawaiian or Pacific Islander": "Native Hawaiian/Pacific Islander",
        "otherUnknown": "Unknown/Other",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid race code: {code_str}")


# --- inverse of transform_asa ---
def inv_fnstatus(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    if code_str == "otherUnknown":
        return "Unknown"
    elif code_str in ["Independent", "Dependent"]:
        return code_str
    else:
        raise ValueError(f"Invalid Functional Status entry: {code_str}")


def combine_encoded_for_app(input_data, shap_raw):
    ohe_dict = get_ohe_cols(input_data)
    ohe_cols = ohe_dict.keys()
    raw_feat_order = []
    for col in input_data.columns.to_list():
        if col in ohe_cols:
            for sub_col in ohe_dict[col]:
                raw_feat_order.append(f"{col}_{sub_col}")
        else:
            raw_feat_order.append(col)

    shap_old = copy.deepcopy(shap_raw)
    for col_name in ohe_cols:
        shap_combined, _ = combine_encoded(
            shap_old, col_name, [col_name in n for n in shap_old.feature_names]  # type: ignore
        )
        shap_old = shap_combined
    return shap_combined


############################################################################################################
############################################################################################################
############################################################################################################
def get_explainer_name(model_name):
    if model_name in ["xgb", "lgbm"]:
        return "TreeExplainer"
    elif model_name in ["svc", "lr"]:
        return "LinearExplainer"
    else:
        return "KernelExplainer"


def decode_cat_for_display(feature_name, model_value):
    """
    Map model-level codes back to the same labels used in the Streamlit UI.
    feature_name: column name in the model input / SHAP features
    model_value:  value in shap_combined.display_data (or original data) for that feature
    """
    # normalize once
    code_val = str(model_value)
    # demographics
    if feature_name == "SEX":
        return inv_sex(float(code_val))
    if feature_name == "ETHNICITY_HISPANIC":
        return inv_hispanic(float(code_val))
    if feature_name == "RACE_NEW":
        return inv_race(feature_name, model_value)
    ## Pre-Op
    if feature_name == "ASACLAS":
        return inv_asa(float(code_val))
    if feature_name == "FNSTATUS2":
        return inv_fnstatus(feature_name, code_val)
    # Intra-op
    if feature_name == "Malignant neoplasm".upper():
        return inv_tumor_site(feature_name, model_value)
    if feature_name == "INOUT":
        return inv_inout(float(code_val))
    if feature_name == "URGENCY":
        return inv_casetype(float(code_val))
    # simple yes/no binaries
    yes_no_cols = [
        # Pre-op
        "DIABETES",
        "SMOKE",
        "VENTILAT",
        "HXCOPD",
        "ASCITES",
        "HXCHF",
        "HYPERMED",
        "DIALYSIS",
        "DISCANCR",
        "STEROID",
        "BLEEDDIS",
        "TRANSFUS",
        "PRSEPIS",
        ## Head+Neck
        "PARTIAL GLOSSECTOMY (HEMIGLOSSECTOMY_SUBTOTAL)",
        "COMPOSITE_EXTENDED GLOSSECTOMY",
        "TOTAL GLOSSECTOMY (COMPLETE TONGUE REMOVAL)",
        "EXCISION OF TONGUE LESIONS (MINOR)",
        "LOCAL_REGIONAL TISSUE FLAPS FOR ORAL CAVITY RECONSTRUCTION",
        "FREE TISSUE TRANSFER (MICROVASCULAR FREE FLAPS) AND COMPLEX FLAP RECONSTRUCTION",
        "SKIN AUTOGRAFTS FOR HEAD AND NECK RECONSTRUCTION",
        "NECK DISSECTION AND LYMPHADENECTOMY PROCEDURES",
        "ALVEOLAR RIDGE AND GINGIVAL PROCEDURES",
        "MANDIBULAR RESECTION AND RECONSTRUCTION PROCEDURES",
        "PERIPHERAL NERVE REPAIR AND NEUROPLASTY",
        "TRACHEOSTOMY PROCEDURES",
        "GASTROSTOMY AND ESOPHAGEAL ACCESS PROCEDURES",
        "SUBMANDIBULAR GLAND EXCISION",
        "PAROTID GLAND EXCISION",
        "LARYNGEAL RESECTION AND RECONSTRUCTION PROCEDURES",
        "PHARYNGEAL RESECTION AND RECONSTRUCTION PROCEDURES",
        "TONSILLECTOMY AND TONSILLAR REGION PROCEDURES",
    ]
    if feature_name in [col.upper() for col in yes_no_cols]:
        return inv_yes_no(float(code_val))

    # yes/no/unknown columns
    if feature_name in [
        ## Pre Op
        "RACE_NEW",
        "DYSPNEA",
        "RENAFAIL",
        "WNDINF",
        "WTLOSS",
    ]:
        return inv_yes_no_unknown(feature_name, code_val)
    # default: just cast to string
    raise ValueError(
        f"Unrecognized feature name: {feature_name} with value: {code_val}"
    )


def feature_value_label(name, disp_value, num_series):
    # numeric features: as before using num_original_series
    if name in num_series.index:
        val = num_series[name]
        if name in ["AGE", "OPERYR"]:
            return f"{val:.0f}"
        elif name == "BMI":
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"

    # categorical / binary / ordinal: decode to Streamlit label
    return decode_cat_for_display(name, disp_value)


def pretty_feature_name(name):
    feature_label_map = {
        ## Col 1
        "AGE": "Age (years)",
        "SEX": "Sex",
        "BMI": "BMI (kg/m²)",
        "ETHNICITY_HISPANIC": "Ethnicity",
        "RACE_NEW": "Race",
        ## Pre op
        "DIABETES": "Diabetes status",
        "SMOKE": "Current smoker",
        "DYSPNEA": "Dyspnea",
        "VENTILAT": "Ventilator dependence",
        "HXCOPD": "COPD",
        "ASCITES": "Ascites",
        "HXCHF": "CHF",
        "HYPERMED": "Hypertension",
        "RENAFAIL": "Renal failure",
        "DIALYSIS": "Dialysis",
        "DISCANCR": "Disseminated cancer",
        "WNDINF": "Wound infection",
        "STEROID": "Chronic steroid use",
        "WTLOSS": "Significant weight loss",
        "BLEEDDIS": "Bleeding disorder",
        "TRANSFUS": "Preop transfusion",
        "PRSEPIS": "Preop sepsis",
        "FNSTATUS2": "Functional status",
        "ASACLAS": "ASA class",
        ## Blood
        "PRALBUM": "Albumin (g/dL)",
        "PRWBC": "WBC (×10³/µL)",
        "PRHCT": "Hematocrit (%)",
        "PRPLATE": "Platelet (×10³/µL)",
        ## Intra-op
        "OPERYR": "Surgery Year",
        "MALIGNANT NEOPLASM": "Tumor site",
        "INOUT": "Setting",
        "URGENCY": "Case type",
        "OPTIME": "Operative time (min)",
        # Head and neck procedure variables
        "PARTIAL GLOSSECTOMY (HEMIGLOSSECTOMY_SUBTOTAL)": "Partial glossectomy",
        "COMPOSITE_EXTENDED GLOSSECTOMY": "Composite/extended glossectomy",
        "TOTAL GLOSSECTOMY (COMPLETE TONGUE REMOVAL)": "Total glossectomy",
        "EXCISION OF TONGUE LESIONS (MINOR)": "Tongue lesion excision",
        "LOCAL_REGIONAL TISSUE FLAPS FOR ORAL CAVITY RECONSTRUCTION": "Local/regional oral cavity flap",
        "FREE TISSUE TRANSFER (MICROVASCULAR FREE FLAPS) AND COMPLEX FLAP RECONSTRUCTION": "Free tissue transfer",
        "SKIN AUTOGRAFTS FOR HEAD AND NECK RECONSTRUCTION": "Skin autograft",
        "NECK DISSECTION AND LYMPHADENECTOMY PROCEDURES": "Neck dissection",
        "ALVEOLAR RIDGE AND GINGIVAL PROCEDURES": "Alveolar ridge/gingival procedure",
        "MANDIBULAR RESECTION AND RECONSTRUCTION PROCEDURES": "Mandibular resection/reconstruction",
        "PERIPHERAL NERVE REPAIR AND NEUROPLASTY": "Peripheral nerve repair",
        "TRACHEOSTOMY PROCEDURES": "Tracheostomy",
        "GASTROSTOMY AND ESOPHAGEAL ACCESS PROCEDURES": "Gastrostomy/esophageal access",
        "SUBMANDIBULAR GLAND EXCISION": "Submandibular gland excision",
        "PAROTID GLAND EXCISION": "Parotid gland excision",
        "LARYNGEAL RESECTION AND RECONSTRUCTION PROCEDURES": "Laryngeal resection/reconstruction",
        "PHARYNGEAL RESECTION AND RECONSTRUCTION PROCEDURES": "Pharyngeal resection/reconstruction",
        "TONSILLECTOMY AND TONSILLAR REGION PROCEDURES": "Tonsillectomy",
    }

    return feature_label_map.get(name, name)


@st.cache_data
def compute_shap_data(
    _explainer, _input_data, _pipeline, processed_data_hash, outcome_name
):
    """
    Compute SHAP values once and cache them.
    processed_data_hash and outcome_nameis used to invalidate cache when input changes.

    All parameters prefixed with _ to tell Streamlit not to hash them directly.
    """
    expected_features = _explainer.feature_names
    input_data = _input_data[expected_features].copy()
    shap_raw = _explainer(input_data)

    # Combine one-hot encoded values
    shap_combined = combine_encoded_for_app(input_data, shap_raw)

    # Get scaler for inverse transform
    num_name, num_pipe, num_cols = _pipeline.transformers_[0]
    assert num_name == "num"
    scaler = num_pipe.named_steps["scaler"]

    # numeric outputs after BMI step (hard-coded order)
    num_out_cols = [
        "AGE",
        "PRALBUM",
        "PRWBC",
        "PRHCT",
        "PRPLATE",
        "OPERYR",
        "OPTIME",
        "BMI",
    ]

    # Inverse transform numeric features
    feat_names = list(shap_combined.feature_names)
    num_indices = [feat_names.index(col) for col in num_out_cols]
    x_trans_row = shap_combined.data[0]
    x_num_scaled = np.array([x_trans_row[i] for i in num_indices]).reshape(1, -1)
    x_num_original = scaler.inverse_transform(x_num_scaled)
    num_original_series = pd.Series(x_num_original.ravel(), index=num_out_cols)

    # Return all data needed for plotting
    return {
        "phi": shap_combined.values[0],
        "feat_names": np.array(shap_combined.feature_names),
        "disp_row": (
            shap_combined.display_data[0]
            if shap_combined.display_data is not None
            else shap_combined.data[0]
        ),
        "num_original_series": num_original_series,
    }

## Append path to root
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))

## Other imports
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import utils as util
from src.preprocess import remove_prefix

# Configuration
OUTCOMES = {
    "Aspiration-related Complications": "asp",
    "Bleeding Transfusion": "bleed",
    "Unplanned Reoperation": "reop",
    # "Death": "mort",  # binning is so bad for this, would be detrimental to include
    "Surgical Complications": "surg",
}


# Cache model loading for performance
@st.cache_resource
def load_model_pipeline(outcome_name):
    """Load model and preprocessor for a specific outcome."""
    model_path = BASE_PATH / "app" / "models" / f"{outcome_name}_stack.joblib"
    model = joblib.load(model_path)
    preprocessor = joblib.load(
        BASE_PATH / "app" / "preprocessors" / f"{outcome_name}_pipeline.joblib"
    )
    return model, preprocessor


def main():
    st.set_page_config(
        page_title="PRO-TONGUE",
        page_icon="🏥",
        layout="wide",
    )

    st.title("PRO-TONGUE: A Post-Resection Outcome Prediction Tool for Tongue Cancer")
    st.markdown(
        "Predict 30-day complications after head and neck surgery for tongue cancer"
    )
    st.info(
        "Adjust all fields to match your patient. Default values are set arbitrarily. To reset to default values, refresh the page. "
    )

    #################################################################################################################
    ################################################### Side Bar ####################################################
    #################################################################################################################
    st.sidebar.header("Select Outcomes to Predict")
    selected_outcomes = []
    for display_name, folder_name in OUTCOMES.items():
        if st.sidebar.checkbox(display_name, value=True):
            selected_outcomes.append((display_name, folder_name))

    if not selected_outcomes:
        st.warning("Please select at least one outcome to predict")
        return

    #################################################################################################################
    ################################################# Input Section #################################################
    #################################################################################################################
    st.header("Patient Information")
    col1, col2, col3, col4, col5 = st.columns(5)
    # ================== Demographics ===================
    with col1:
        with st.expander("**Demographics**", expanded=True):
            sex = st.selectbox("Sex", ["Male", "Female"], index=0)  # --> 1/0
            # ================== Dynamic Weight ===================
            # Initialize session state
            if "prev_weight_unit" not in st.session_state:
                st.session_state.prev_weight_unit = "kg"
            if "weight_kg" not in st.session_state:
                st.session_state.weight_kg = 77.11
            if "weight_lbs" not in st.session_state:
                st.session_state.weight_lbs = 170.0
            # Nested columns: input on left, checkbox on right
            input_col, check_col = st.columns([3, 2])

            with check_col:
                st.write("")  # Spacer to align with input
                st.write("")  # Spacer to align with input
                weight_unknown = st.checkbox("N/A", key="weight_unknown")
            # Unit selector below both
            if not weight_unknown:
                weight_unit = st.radio(
                    "Weight unit",
                    ["lbs", "kg"],
                    index=1,
                    key="weight_unit",
                    horizontal=True,
                )
                # Detect if the unit changed and convert b4 rendering inputs
                if weight_unit != st.session_state.prev_weight_unit:
                    if weight_unit == "lbs":
                        # KG -> LBS
                        st.session_state.weight_lbs = (
                            st.session_state.weight_kg * 2.20462
                        )
                    else:
                        # LBS -> KG
                        st.session_state.weight_kg = (
                            st.session_state.weight_lbs / 2.20462
                        )
                    st.session_state.prev_weight_unit = weight_unit
            ## Render input box
            with input_col:
                if weight_unknown:
                    st.number_input("Weight", value=0.0, disabled=True)
                    weight = None
                else:
                    weight_unit = st.session_state.get("weight_unit", "kg")
                    if weight_unit == "lbs":
                        weight = st.number_input(
                            "Weight (lbs)", min_value=2.20462, key="weight_lbs"
                        )
                    else:
                        weight_kg = st.number_input(
                            "Weight (kg)", min_value=1.0, key="weight_kg"
                        )
                        weight = weight_kg * 2.20462

            # ================== Dynamic Height ===================
            # Initialize session state
            if "prev_height_unit" not in st.session_state:
                st.session_state.prev_height_unit = "m"
            if "height_m" not in st.session_state:
                st.session_state.height_m = 1.68
            if "height_in" not in st.session_state:
                st.session_state.height_in = 66.0

            # Nested columns: input on left, checkbox on right
            height_input_col, height_check_col = st.columns([3, 2])

            with height_check_col:
                st.write("")  # Spacer to align with input
                st.write("")  # Spacer to align with input
                height_unknown = st.checkbox("N/A", key="height_unknown")

            # Unit selector below both
            if not height_unknown:
                height_unit = st.radio(
                    "Height unit",
                    ["in", "m"],
                    index=1,
                    key="height_unit",
                    horizontal=True,
                )
                # Detect if the unit changed and convert b4 rendering inputs
                if height_unit != st.session_state.prev_height_unit:
                    if height_unit == "in":
                        # Meters -> Inches
                        st.session_state.height_in = st.session_state.height_m * 39.3701
                    else:
                        # Inches -> Meters
                        st.session_state.height_m = st.session_state.height_in / 39.3701
                    st.session_state.prev_height_unit = height_unit

            ## Render input box
            with height_input_col:
                if height_unknown:
                    st.number_input("Height", value=0.0, disabled=True)
                    height = None
                else:
                    height_unit = st.session_state.get("height_unit", "m")
                    if height_unit == "in":
                        height = st.number_input(
                            "Height (in)", min_value=39.3701, key="height_in"
                        )
                    else:
                        height_m = st.number_input(
                            "Height (m)", min_value=1.0, key="height_m"
                        )
                        height = height_m * 39.3701

            # ================== Others ===================
            ## AGE
            # Nested columns: input on left, checkbox on right
            age_input_col, age_check_col = st.columns([3, 2])
            with age_check_col:
                st.write("")  # Spacer to align with input
                st.write("")  # Spacer to align with input
                age_unknown = st.checkbox("N/A", key="age_unknown")
            with age_input_col:
                if age_unknown:
                    st.number_input("Age", value=0.0, disabled=True)
                    age = None
                else:
                    age = st.number_input("Age", min_value=18, max_value=90, value=63)

            hispanic = st.selectbox(
                "Ethnicity", ["Hispanic", "Not Hispanic/Unknown"], index=1  # --> 1/0
            )  # fix to be 1/0
            race = st.selectbox(
                "Race",
                [
                    "White",
                    "Black or African American",
                    "Asian",
                    "Unknown/Other",  # --> Unknown_Other
                ],
                index=0,
            )
    # ================== Pre-Op ===================
    with col2:
        with st.expander("**Pre-Operative Health Status**", expanded=True):
            diabetes = st.selectbox("Diabetes", ["Yes", "No"], index=1)
            smoke = st.selectbox("Current Smoker", ["Yes", "No"], index=1)
            dyspnea = st.selectbox("Dyspnea", ["Yes", "No", "Unknown"], index=2)
            vent = st.selectbox("Ventilator >48 Hours", ["Yes", "No"], index=1)
            hxcopd = st.selectbox("COPD", ["Yes", "No"], index=1)
            ascites = st.selectbox("Ascites", ["Yes", "No"])
            hxchf = st.selectbox("Congestive Heart Failure", ["Yes", "No"], index=1)
            hypermed = st.selectbox("Hypertension", ["Yes", "No"], index=1)
            renal_failure = st.selectbox(
                "Acute Renal Failure", ["Yes", "No", "Unknown"], index=2
            )
            dialysis = st.selectbox("Dialysis", ["Yes", "No"], index=1)
            discancr = st.selectbox("Disseminated Cancer", ["Yes", "No"], index=1)
            wndinf = st.selectbox("Wound Infection", ["Yes", "No", "Unknown"], index=2)
            steroid = st.selectbox("Corticosteroid Use", ["Yes", "No"], index=1)
            wtloss = st.selectbox(f"Weight Loss", ["Yes", "No", "Unknown"], index=2)
            bleed = st.selectbox("Bleeding Disorder", ["Yes", "No"], index=1)
            transfus = st.selectbox("Blood Transfusion", ["Yes", "No"], index=1)
            prsepis = st.selectbox("Sepsis", ["Yes", "No"], index=1)
            func_stat = st.selectbox(
                "Functional Status", ["Independent", "Dependent", "Unknown"], index=0
            )  # --> 1/0
            asa_class = st.selectbox(
                "ASA Class",
                [
                    "1-No Disturbance",  # --> 1-No Disturb
                    "2-Mild Disturbance",  # --> 2-Mild Disturb
                    "3-Severe Disturbance",  # --> 3-Severe Disturb
                    "4-Life Threatening Disturbance/5-Moribund",  # --> 4-Life Threat
                ],
                index=2,
            )
    # ================== Blood ===================
    with col3:
        with st.expander("**Pre-Operative Blood Labs**", expanded=True):
            # Albumin
            alb_input_col, alb_check_col = st.columns([3, 2])

            with alb_check_col:
                st.write("")
                st.write("")
                alb_unknown = st.checkbox("N/A", key="alb_unknown")

            with alb_input_col:
                if alb_unknown:
                    st.number_input("Albumin (g/dL)", value=0.0, disabled=True)
                    pralbumin = None
                else:
                    pralbumin = st.number_input(
                        "Albumin (g/dL)", min_value=0.0, max_value=None, value=4.2
                    )

            # WBC
            wbc_input_col, wbc_check_col = st.columns([3, 2])

            with wbc_check_col:
                st.write("")
                st.write("")
                wbc_unknown = st.checkbox("N/A", key="wbc_unknown")

            with wbc_input_col:
                if wbc_unknown:
                    st.number_input(
                        "White Blood Cell Count (*10^9/L)", value=0.0, disabled=True
                    )
                    prwbc = None
                else:
                    prwbc = st.number_input(
                        "White Blood Cell Count (*10^9/L)",
                        min_value=0.0,
                        max_value=None,
                        value=7.0,
                    )

            # HCT
            hct_input_col, hct_check_col = st.columns([3, 2])

            with hct_check_col:
                st.write("")
                st.write("")
                hct_unknown = st.checkbox("N/A", key="hct_unknown")

            with hct_input_col:
                if hct_unknown:
                    st.number_input("Hematocrit (%)", value=0.0, disabled=True)
                    prhct = None
                else:
                    prhct = st.number_input(
                        "Hematocrit (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=41.0,
                    )

            # PLATE
            plate_input_col, plate_check_col = st.columns([3, 2])

            with plate_check_col:
                st.write("")
                st.write("")
                plate_unknown = st.checkbox("N/A", key="plate_unknown")

            with plate_input_col:
                if plate_unknown:
                    st.number_input(
                        "Platelet Count (*10^9/L)", value=0.0, disabled=True
                    )
                    prplate = None
                else:
                    prplate = st.number_input(
                        "Platelet Count (*10^9/L)",
                        min_value=0.0,
                        max_value=None,
                        value=238.0,
                    )

        # ================== Intra-Op ===================
    # ================== Intra-Op ===================
    with col4:
        with st.expander("**Intra-Operative Characteristics**", expanded=True):
            operyr = st.number_input(
                "Operation Year", min_value=2008, max_value=2025, value=2018
            )
            mal_neoplasm = st.selectbox(
                "Location of Tongue Tumor",
                [
                    "Anterior two-thirds",  # --> Malignant neoplasm of anterior two-thirds of tongue unspecified
                    "Base",  # --> Malignant neoplasm of base of tongue
                    "Border",  # --> Malignant neoplasm of border of tongue
                    "Junctional Zone",  # --> Malignant neoplasm of junctional zone of tongue
                    "Surface",  # --> Malignant neoplasm of surface of tongue
                    "Lingual Tonsil",  # --> Malignant neoplasm of lingual tonsil
                    "Unspecified",  # --> Malignant neoplasm of tongue unspecified
                ],
                index=5,
            )
            inout = st.selectbox(
                "Setting", ["Inpatient", "Outpatient"], index=0
            )  # --> 1/0
            elect_surg = st.selectbox(
                "Case Type",
                [
                    "Elective",  # --> Elective
                    "Urgent/Emergent",  # --> Urgent_Emergent
                    "Unknown",  # --> Unknown
                ],
                index=1,
            )
            # Operation Time
            optime_input_col, optime_check_col = st.columns([3, 2])

            with optime_check_col:
                st.write("")  # Spacer to align with input
                st.write("")
                optime_unknown = st.checkbox("N/A", key="optime_unknown")

            with optime_input_col:
                if optime_unknown:
                    st.number_input(
                        "Operation Time (minutes)",
                        value=0.0,
                        disabled=True,
                    )
                    optime = None
                else:
                    optime = st.number_input(
                        "Operation Time (minutes)",
                        min_value=0.0,
                        max_value=None,
                        value=214.0,
                    )
    # ================== Proc ===================
    with col5:
        with st.expander("**Head and Neck Procedures**", expanded=True):
            part_gloss = st.selectbox("Partial Glossectomy", ["Yes", "No"], index=1)
            comp_ext_gloss = st.selectbox(
                "Composite Extended Glossectomy", ["Yes", "No"], index=1
            )
            total_gloss = st.selectbox("Total Glossectomy", ["Yes", "No"], index=1)
            tongue_exc = st.selectbox(
                "Excision of Tongue Lesions", ["Yes", "No"], index=1
            )
            oral_cav_recon = st.selectbox(
                "Local/Regional Tissue Flaps",
                ["Yes", "No"],
                index=1,
            )
            free_tissue_transfer = st.selectbox(
                "Free Tissue Transfer", ["Yes", "No"], index=1
            )
            skin_auto = st.selectbox("Skin Autograft", ["Yes", "No"], index=1)
            neck_diss = st.selectbox(
                "Lymphadenectomy Procedure", ["Yes", "No"], index=1
            )
            alv_ridge = st.selectbox(
                "Alveolar Ridge and Gingival Procedure", ["Yes", "No"], index=1
            )
            mand_res = st.selectbox(
                "Mandibular Resection/Reconstruction", ["Yes", "No"], index=1
            )
            peri_nerve = st.selectbox("Peripheral Nerve Repair", ["Yes", "No"], index=1)
            trach_proc = st.selectbox("Tracheostomy Procedure", ["Yes", "No"], index=1)
            gast_eso_proc = st.selectbox(
                "Gastrostomy and Esophageal Access Procedure", ["Yes", "No"], index=1
            )
            sub_gland = st.selectbox(
                "Submandibular Gland Excision", ["Yes", "No"], index=1
            )
            parotid = st.selectbox("Parotid Gland Excision", ["Yes", "No"], index=1)
            laryngeal = st.selectbox(
                "Laryngeal Resection/Reconstruction", ["Yes", "No"], index=1
            )
            pharyngeal = st.selectbox(
                "Pharyngeal Resection/Reconstruction", ["Yes", "No"], index=1
            )
            tonsil = st.selectbox(
                "Tonsillectomy and Tonsillar Region Procedure", ["Yes", "No"], index=1
            )

    # ================== Create input DF ===================
    input_data = pd.DataFrame(
        {
            ## Col 1
            "SEX": [util.transform_sex(sex)],
            "WEIGHT": [weight],
            "HEIGHT": [height],
            "Age": [age],
            "ETHNICITY_HISPANIC": [util.transform_hispanic(hispanic)],
            "RACE_NEW": [util.transform_unknown_other(race)],
            "OPERYR": [operyr],
            ## Col 2
            "Malignant neoplasm": [util.transform_tumor_site(mal_neoplasm)],
            "DIABETES": [util.transform_yes_no(diabetes)],
            "SMOKE": [util.transform_yes_no(smoke)],
            "DYSPNEA": [util.transform_unknown_other(dyspnea)],
            "VENTILAT": [util.transform_yes_no(vent)],
            "HXCOPD": [util.transform_yes_no(hxcopd)],
            "ASCITES": [util.transform_yes_no(ascites)],
            "HXCHF": [util.transform_yes_no(hxchf)],
            "HYPERMED": [util.transform_yes_no(hypermed)],
            "RENAFAIL": [util.transform_unknown_other(renal_failure)],
            "DIALYSIS": [util.transform_yes_no(dialysis)],
            "DISCANCR": [util.transform_yes_no(discancr)],
            "WNDINF": [util.transform_unknown_other(wndinf)],
            "STEROID": [util.transform_yes_no(steroid)],
            "WTLOSS": [util.transform_unknown_other(wtloss)],
            "BLEEDDIS": [util.transform_yes_no(bleed)],
            "TRANSFUS": [util.transform_yes_no(transfus)],
            "PRSEPIS": [util.transform_yes_no(prsepis)],
            "PRALBUM": [pralbumin],
            "PRWBC": [prwbc],
            "PRHCT": [prhct],
            "PRPLATE": [prplate],
            "FNSTATUS2": [util.transform_unknown_other(func_stat)],
            ## Col 3
            "INOUT": [util.transform_inout(inout)],
            "URGENCY": [util.transform_casetype(elect_surg)],
            "ASACLAS": [util.transform_ASA(asa_class)],
            "OPTIME": [optime],
            ## Col 4
            "Partial Glossectomy (Hemiglossectomy_Subtotal)": [
                util.transform_yes_no(part_gloss)
            ],
            "Composite_Extended Glossectomy": [util.transform_yes_no(comp_ext_gloss)],
            "Total Glossectomy (Complete Tongue Removal)": [
                util.transform_yes_no(total_gloss)
            ],
            "Excision of Tongue Lesions (Minor)": [util.transform_yes_no(tongue_exc)],
            "Local_Regional Tissue Flaps for Oral Cavity Reconstruction": [
                util.transform_yes_no(oral_cav_recon)
            ],
            "Free Tissue Transfer (Microvascular Free Flaps) and Complex Flap Reconstruction": [
                util.transform_yes_no(free_tissue_transfer)
            ],
            "Skin Autografts for Head and Neck Reconstruction": [
                util.transform_yes_no(skin_auto)
            ],
            "Neck Dissection and Lymphadenectomy Procedures": [
                util.transform_yes_no(neck_diss)
            ],
            "Alveolar Ridge and Gingival Procedures": [
                util.transform_yes_no(alv_ridge)
            ],
            "Mandibular Resection and Reconstruction Procedures": [
                util.transform_yes_no(mand_res)
            ],
            "Peripheral Nerve Repair and Neuroplasty": [
                util.transform_yes_no(peri_nerve)
            ],
            "Tracheostomy Procedures": [util.transform_yes_no(trach_proc)],
            "Gastrostomy and Esophageal Access Procedures": [
                util.transform_yes_no(gast_eso_proc)
            ],
            "Submandibular Gland Excision": [util.transform_yes_no(sub_gland)],
            "Parotid Gland Excision": [util.transform_yes_no(parotid)],
            "Laryngeal Resection and Reconstruction Procedures": [
                util.transform_yes_no(laryngeal)
            ],
            "Pharyngeal Resection and Reconstruction Procedures": [
                util.transform_yes_no(pharyngeal)
            ],
            "Tonsillectomy and Tonsillar Region Procedures": [
                util.transform_yes_no(tonsil)
            ],
        }
    )
    input_data.columns = input_data.columns.str.upper()
    #################################################################################################################
    ############################################### Output Section ##################################################
    #################################################################################################################
    ## Maps column name to unknown
    ### This order is hard-coded and matches order passed into imputer
    num_dict = {
        "AGE": {
            "Value": age,
            "Display Name": "Age",
            "round rule": lambda x: int(round(x)),
        },
        "HEIGHT": {
            "Value": weight,
            "Display Name": "Height (in)",
            "round rule": lambda x: round(x, 2),
        },
        "WEIGHT": {
            "Value": height,
            "Display Name": "Weight (lbs)",
            "round rule": lambda x: round(x, 2),
        },
        "PRALBUM": {
            "Value": pralbumin,
            "Display Name": "Albumin (g/dL)",
            "round rule": lambda x: round(x, 2),
        },
        "PRWBC": {
            "Value": prwbc,
            "Display Name": "White Blood Cell Count (*10^9/L)",
            "round rule": lambda x: round(x, 2),
        },
        "PRHCT": {
            "Value": prhct,
            "Display Name": "Platelet Count (*10^9/L)",
            "round rule": lambda x: round(x, 2),
        },
        "PRPLATE": {
            "Value": prplate,
            "Display Name": "Hematocrit (%)",
            "round rule": lambda x: round(x, 2),
        },
        # this shouldnt run, dont give option
        ## just including for completeness
        "OPERYR": {
            "Value": operyr,
            "Display Name": "Operation Year",
            "round rule": lambda x: int(round(x)),
        },
        "OPTIME": {
            "Value": optime,
            "Display Name": "Operation Time (min)",
            "round rule": lambda x: round(x, 2),
        },
    }

    # Get list of imputed vals
    imp_cols = [
        col_name for col_name, sub_dict in num_dict.items() if sub_dict["Value"] is None
    ]
    if st.button("Predict Outcomes", type="primary", key="pred_btn"):
        st.header("Prediction Results")

        # Process each selected outcome
        for display_name, folder_name in selected_outcomes:
            with st.expander(f"📊 {display_name}", expanded=True):
                try:
                    # ================== Get model output ===================
                    # Load model and preprocessor
                    model, preprocessor = load_model_pipeline(folder_name)

                    ## Preprocess
                    feature_names = preprocessor.get_feature_names_out()
                    data_transformed = np.array(preprocessor.transform(input_data))
                    processed_data = pd.DataFrame(
                        data_transformed, columns=feature_names
                    )
                    processed_data = remove_prefix(processed_data)
                    for col in processed_data.columns:
                        processed_data[col] = pd.to_numeric(processed_data[col])

                    ## predict
                    probabilities = model.predict_proba(processed_data)[:, 1]

                    # Extract scalar probability
                    if probabilities.ndim == 1:
                        prob_positive = float(probabilities[0])
                    else:
                        # output is 2D
                        prob_positive = float(probabilities[0, 1])

                    # ================== Display Results ===================
                    col_a, col_b, col_c = st.columns(3)

                    ## Model output as %
                    with col_a:
                        st.metric(
                            label="Risk Probability",
                            value=f"{prob_positive:.2%}",
                            delta=None,
                        )
                    ## Risk Bins
                    with col_b:
                        # Display appropriate risk category
                        bin_thresholds = util.load_bin_thresholds(folder_name)
                        risk, color = util.get_risk_category(prob_positive, folder_name)
                        st.metric(label="Risk Category", value=f"{color} {risk}")
                        # Display bin thresholds
                        try:
                            labels = ["Very Low", "Low", "Moderate", "High"]
                            cutoffs = [0.0] + list(bin_thresholds) + [1.0]
                            # returns bin index
                            bin_idx = np.digitize([prob_positive], cutoffs[1:])[0]

                            # Build output lines with bold for the active bin
                            lines = []
                            for i, lab in enumerate(labels):
                                line = f"{lab}: {cutoffs[i]:.1%} – {cutoffs[i+1]:.1%}"
                                if i == bin_idx:
                                    # Bold the whole line
                                    line = f"<b>{line}</b>"
                                lines.append(line)
                            # <br> for newlines
                            st.markdown("<br>".join(lines), unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not load thresholds: {str(e)}")
                    ## Percentiles
                    with col_c:
                        all_probs, all_labels = util.load_population_probs(folder_name)

                        # Overall percentile
                        n_overall = len(all_probs)
                        overall_pctile = (all_probs < prob_positive).mean() * 100  # type: ignore

                        # Percentile among patients WITHOUT the outcome (label==0)
                        neg_patients = all_probs[all_labels == 0]
                        n_neg = len(neg_patients)
                        neg_pctile = (neg_patients < prob_positive).mean() * 100

                        # Percentile among patients WITH the outcome (label==1)
                        pos_patients = all_probs[all_labels == 1]
                        n_pos = len(pos_patients)
                        pos_pctile = (pos_patients < prob_positive).mean() * 100

                        st.markdown(
                            f"<b>{overall_pctile:.1f}%</b> of <b>all patients sampled</b> (n={n_overall}) received a lower risk score than this patient</b>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<b>{neg_pctile:.1f}%</b> of patients sampled who <b>did not suffer this outcome</b> (n={n_neg}) received a lower risk score than this patient</b>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<b>{pos_pctile:.1f}%</b> of patients who <b>did suffer this outcome</b> (n={n_pos}) received a lower risk score than this patient",
                            unsafe_allow_html=True,
                        )

                    # Progress bar visualization
                    st.progress(float(prob_positive))
                except Exception as e:
                    st.error(f"Error predicting {display_name}: {str(e)}")

        # Display imputed values
        if len(imp_cols) > 0:
            st.header("Imputed Values")
            st.info(
                """
            When patient data is missing, the modeling pipeline uses an iterative regression-based imputation method to estimate those values.
            It models each incomplete variable using other available patient characteristics, ***defined in each outcome's respective train set,*** and refines these estimates over several rounds.
            Imputed values are statistical estimates, not actual measurements.
            """
            )
            for display_name, folder_name in selected_outcomes:
                _, preprocessor = load_model_pipeline(folder_name)
                with st.expander(f"📋 Imputed Values for {display_name}"):
                    ## Get pipeline steps
                    num_pipe = preprocessor.named_transformers_["num"]
                    imputer = num_pipe.named_steps["imputer"]
                    # bmi_step = num_pipe.named_steps["bmi"]
                    # scaler = num_pipe.named_steps["scaler"]
                    ## Get intermediate values
                    X_num_raw = input_data[num_dict.keys()].to_numpy()
                    X_imputed = imputer.transform(X_num_raw)
                    # all_X_bmi_unscaled = bmi_step.transform(X_imputed)
                    # bmi_unscaled = all_X_bmi_unscaled[:, -1]
                    # Build a small df for display
                    imp_display = {}
                    for col in imp_cols:
                        ## Find index amongst num cols
                        col_idx = 0
                        for col_name in num_dict.keys():
                            if col == col_name:
                                break
                            else:
                                col_idx += 1
                        raw_val = X_imputed[0, col_idx]
                        round_rule = num_dict[col]["round rule"]
                        imp_display[num_dict[col]["Display Name"]] = round_rule(raw_val)

                    display_df = pd.DataFrame.from_dict(
                        imp_display, orient="index", columns=["Value"]
                    )
                    st.dataframe(display_df, width="content")


if __name__ == "__main__":
    main()

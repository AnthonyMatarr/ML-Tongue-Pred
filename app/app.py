## CPU not MPS
import os

os.environ["PYTORCH_MPS_ENABLE"] = "0"
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
import torch
from src.preprocess import remove_prefix

# Configuration
OUTCOMES = {
    "Aspiration Related Complications": "asp",
    "Bleeding Transfusion": "bleed",
    "Death": "mort",
    "Surgical Wound Complications": "surg",
}


# Cache model loading for performance
@st.cache_resource
def load_model_pipeline(outcome_name):
    """Load model and preprocessor for a specific outcome."""
    model_path = BASE_PATH / "app" / "models" / f"{outcome_name}_stack.joblib"
    # force CPU during load
    with torch.device("cpu"):
        # Use stack model
        model = joblib.load(model_path)
    model = joblib.load(model_path)
    preprocessor = joblib.load(
        BASE_PATH / "preprocessors" / f"preprocessor_outcome_{outcome_name}.joblib"
    )
    return model, preprocessor


def main():
    st.set_page_config(
        page_title="PRO-TONGUE",
        page_icon="🏥",
        layout="wide",
    )

    st.title("PRO-TONGUE: Post-Resection Outcomes for tongue tumor excision")
    st.markdown(
        "Predict 30-day complications after head and neck surgery for tongue neoplasm"
    )
    st.info(
        "Default values represent median values for numerical variables, but are meaningless for categorical variables. "
        "Adjust all fields to match your patient."
    )

    # Sidebar: Outcome selection
    st.sidebar.header("Select Outcomes to Predict")
    selected_outcomes = []
    for display_name, folder_name in OUTCOMES.items():
        if st.sidebar.checkbox(display_name, value=True):
            selected_outcomes.append((display_name, folder_name))

    if not selected_outcomes:
        st.warning("Please select at least one outcome to predict")
        return
    ## Reset button
    if st.sidebar.button("🔄 Clear Form"):
        st.rerun()

    # Main input section
    st.header("Patient Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Demographics")
        sex = st.selectbox("Sex", ["Male", "Female"])  # --> 1/0
        weight = st.number_input(
            "Weight (lbs)", min_value=0.0, max_value=None, value=66.0
        )
        height = st.number_input(
            "Height (in)", min_value=0.0, max_value=None, value=170.0
        )

        # bmi = st.number_input(
        #     "BMI (kg/m^2)", min_value=11.0, max_value=82.0, value=26.7
        # )
        age = st.number_input("Age", min_value=1, max_value=90, value=63)
        hispanic = st.selectbox(
            "Ethnicity", ["Hispanic", "Not Hispanic/Unknown"]  # --> 1/0
        )  # fix to be 1/0
        race = st.selectbox(
            "Race",
            [
                "White",
                "Black or African American",
                "Asian",
                "Unknown/Other",  # --> Unknown_Other
            ],
        )

    with col2:
        st.subheader("Pre-Op Chatacteristics")
        mal_neoplasm = st.selectbox(
            "Tumor Site",
            [
                "Anterior two-thirds",  # --> Malignant neoplasm of anterior two-thirds of tongue unspecified
                "Base",  # --> Malignant neoplasm of base of tongue
                "Border",  # --> Malignant neoplasm of border of tongue
                "Junctional Zone",  # --> Malignant neoplasm of junctional zone of tongue
                "Surface",  # --> Malignant neoplasm of surface of tongue
                "Unspecified",  # --> Malignant neoplasm of surface of tongue
            ],
        )
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        smoke = st.selectbox("Current Smoker", ["Yes", "No"])
        dyspnea = st.selectbox(
            "Dyspnea", ["Yes", "No", "Unknown/Other"]
        )  # Unknown_Other
        vent = st.selectbox("Ventilator >48 Hours", ["Yes", "No"])
        hxcopd = st.selectbox("COPD", ["Yes", "No"])
        ascites = st.selectbox("Ascites", ["Yes", "No"])
        hxchf = st.selectbox("Congestive Heart Failure", ["Yes", "No"])
        hypermed = st.selectbox("Hypertension", ["Yes", "No"])
        renal_failure = st.selectbox("Acute Renal Failure", ["Yes", "No"])
        dialysis = st.selectbox("Dialysis", ["Yes", "No"])
        discancr = st.selectbox("Disseminated Cancer", ["Yes", "No"])
        wndinf = st.selectbox("Wound Infection", ["Yes", "No", "Unknown"])
        steroid = st.selectbox("Corticosteroid Use", ["Yes", "No"])
        wtloss = st.selectbox(
            f"Weight Loss (>10% body weight loss in last 6 months)",
            ["Yes", "No", "Unknown/Other"],  # --> Unknown_Other
        )
        bleed = st.selectbox("Bleeding Disorder", ["Yes", "No"])
        transfus = st.selectbox("Blood Transfusion", ["Yes", "No"])
        prsepis = st.selectbox("Sepsis", ["Yes", "No"])
        pralbumin = st.number_input(
            "Albumin (g/dL)", min_value=0.0, max_value=None, value=4.2
        )
        prwbc = st.number_input(
            "White Blood Cell Count (*10^9/L)",
            min_value=0.0,
            max_value=None,
            value=7.0,
        )
        func_stat = st.selectbox(
            "Functional Status", ["Independent", "Dependent"]
        )  # --> 1/0
    with col3:
        st.subheader("Intra-Op Characteristics")
        inout = st.selectbox("Setting", ["Inpatient", "Outpatient"])  # --> 1/0
        elect_surg = st.selectbox(
            "Case Type",
            [
                "Elective",  # --> Elective
                "Urgent/Emergent",  # --> Urgent_Emergent
                "Unknown",  # --> Unknown
            ],
        )
        asa_class = st.selectbox(
            "ASA Class",
            [
                "1-No Disturbance",  # --> 1-No Disturb
                "2-Mild Disturbance",  # --> 2-Mild Disturb
                "3-Severe Disturbance",  # --> 3-Severe Disturb
                "4-Life Threatening Disturbance",  # --> 4-Life Threat
            ],
        )
        optime = st.number_input(
            "Operation Time (minutes)", min_value=0.0, max_value=None, value=214.0
        )

    with col4:
        st.subheader("Head and Neck Procedures")
        part_gloss = st.selectbox("Partial Glossectomy", ["Yes", "No"])
        comp_ext_gloss = st.selectbox("Composite Extended Glossectomy", ["Yes", "No"])
        total_gloss = st.selectbox("Total Glossectomy", ["Yes", "No"])
        tongue_exc = st.selectbox("Excision of Tongue Lesions", ["Yes", "No"])
        oral_cav_recon = st.selectbox(
            "Local/Regional Tissue Flaps for Oral Cavity Reconstruction", ["Yes", "No"]
        )
        free_tissue_transfer = st.selectbox("Free Tissue Transfer", ["Yes", "No"])
        skin_auto = st.selectbox("Skin Autograft", ["Yes", "No"])
        neck_diss = st.selectbox(
            "Neck Dissection and Lymphadenectomy Procedure", ["Yes", "No"]
        )
        alv_ridge = st.selectbox("Alveolar Ridge and Gingival Procedure", ["Yes", "No"])
        mand_res = st.selectbox(
            "Mandibular Resection and Reconstruction Procedure", ["Yes", "No"]
        )
        peri_nerve = st.selectbox(
            "Peripheral Nerve Repair and Neuroplasty", ["Yes", "No"]
        )
        trach_proc = st.selectbox("Tracheostomy Procedure", ["Yes", "No"])
        gast_eso_proc = st.selectbox(
            "Gastrostomy and Esophageal Access Procedure", ["Yes", "No"]
        )
        sub_gland = st.selectbox("Submandibular Gland Excision", ["Yes", "No"])
        parotid = st.selectbox("Parotid Gland Excision", ["Yes", "No"])
        laryngeal = st.selectbox(
            "Laryngeal Resection and Reconstruction Procedure", ["Yes", "No"]
        )
        pharyngeal = st.selectbox(
            "Pharyngeal Resection and Reconstruction Procedure", ["Yes", "No"]
        )
        tonsil = st.selectbox(
            "Tonsillectomy and Tonsillar Region Procedure", ["Yes", "No"]
        )

    # Create input dataframe
    input_data = pd.DataFrame(
        {
            ## Col 1
            "SEX": [util.transform_sex(sex)],
            "WEIGHT": [weight],
            "HEIGHT": [height],
            "Age": [age],
            "ETHNICITY_HISPANIC": [util.transform_hispanic(hispanic)],
            "RACE_NEW": [util.transform_unknown_other(race)],
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
            "RENAFAIL": [util.transform_yes_no(renal_failure)],
            "DIALYSIS": [util.transform_yes_no(dialysis)],
            "DISCANCR": [util.transform_yes_no(discancr)],
            "WNDINF": [wndinf],
            "STEROID": [util.transform_yes_no(steroid)],
            "WTLOSS": [util.transform_unknown_other(wtloss)],
            "BLEEDDIS": [util.transform_yes_no(bleed)],
            "TRANSFUS": [util.transform_yes_no(transfus)],
            "PRSEPIS": [util.transform_yes_no(prsepis)],
            "PRALBUM": [pralbumin],
            "PRWBC": [prwbc],
            "FNSTATUS2": [util.transform_func_status(func_stat)],
            ## Col 3
            "INOUT": [util.transform_inout(inout)],
            "ELECTSURG": [util.transform_casetype(elect_surg)],
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

    # Predict button
    if st.button("🔮 Predict Outcomes", type="primary"):
        st.header("Prediction Results")

        # Process each selected outcome
        for display_name, folder_name in selected_outcomes:
            with st.expander(f"📊 {display_name}", expanded=True):
                try:
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
                        prob_positive = float(
                            probabilities[0]
                        )  # ← Convert to Python float
                    else:
                        prob_positive = float(
                            probabilities[0, 1]
                        )  # ← Extract and convert

                    # Display results
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric(
                            label="Risk Probability",
                            value=f"{prob_positive:.2%}",
                            delta=None,
                        )

                    with col_b:
                        risk, color = util.get_risk_category(prob_positive, folder_name)
                        st.metric(label="Risk Category", value=f"{color} {risk}")

                    # Progress bar visualization
                    st.progress(float(prob_positive))

                except Exception as e:
                    st.error(f"Error predicting {display_name}: {str(e)}")

        # Optional: Display input summary
        with st.expander("📋 Input Summary"):
            display_df = input_data.T.astype(str)
            st.dataframe(display_df, width="stretch")  # Updated API


if __name__ == "__main__":
    main()

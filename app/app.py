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
    # "Death": "mort", #binning is so bad for this, would be detrimental to include
    "Surgical Complications": "surg",
}


# Cache model loading for performance
@st.cache_resource
def load_model_pipeline(outcome_name):
    """Load model and preprocessor for a specific outcome."""
    model_path = BASE_PATH / "app" / "models" / f"{outcome_name}_stack.joblib"
    # # force CPU during load
    # with torch.device("cpu"):
    #     # Use stack model
    # model = joblib.load(model_path)
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

    col1, col2, col3, col4 = st.columns(4)
    # ================== Demographics ===================
    with col1:
        st.subheader("Demographics")
        sex = st.selectbox("Sex", ["Male", "Female"], index=0)  # --> 1/0

        # ================== Dynamic Weight ===================
        # Store the previous unit in session state
        if "prev_weight_unit" not in st.session_state:
            st.session_state.prev_weight_unit = "kg"
        if "weight_kg" not in st.session_state:
            st.session_state.weight_kg = 77.11
        if "weight_lbs" not in st.session_state:
            st.session_state.weight_lbs = 170.0

        weight_unit = st.radio("Weight unit", ["lbs", "kg"], index=1, key="weight_unit")

        # Detect if the unit changed
        if weight_unit != st.session_state.prev_weight_unit:
            if weight_unit == "lbs":
                # KG -> LBS
                st.session_state.weight_lbs = st.session_state.weight_kg * 2.20462
            else:
                # LBS -> KG
                st.session_state.weight_kg = st.session_state.weight_lbs / 2.20462
            st.session_state.prev_weight_unit = weight_unit  # Update the tracker

        # Render input
        if weight_unit == "lbs":
            weight = st.number_input("Weight (lbs)", min_value=0.0, key="weight_lbs")
        else:
            weight_kg = st.number_input("Weight (kg)", min_value=0.0, key="weight_kg")
            weight = weight_kg * 2.20462

        # ================== Dynamic Height ===================
        # Store the previous unit in session state
        if "prev_height_unit" not in st.session_state:
            st.session_state.prev_height_unit = "m"
        if "height_m" not in st.session_state:
            st.session_state.height_m = 1.68
        if "height_in" not in st.session_state:
            st.session_state.height_in = 66.0

        height_unit = st.radio("Height unit", ["in", "m"], index=1, key="height_unit")

        # Detect if the unit changed
        if height_unit != st.session_state.prev_height_unit:
            if height_unit == "in":
                # Meters -> Inches
                st.session_state.height_in = st.session_state.height_m * 39.3701
            else:
                # Inches -> Meters
                st.session_state.height_m = st.session_state.height_in / 39.3701
            st.session_state.prev_height_unit = height_unit  # Update the tracker

        # Render input
        if height_unit == "in":
            height = st.number_input("Height (in)", min_value=0.0, key="height_in")
        else:
            height_m = st.number_input("Height (m)", min_value=0.0, key="height_m")
            height = height_m * 39.3701
        # ================== Others ===================
        age = st.number_input("Age", min_value=1, max_value=90, value=63)

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
        st.subheader("Pre-Operative Characteristics")
        diabetes = st.selectbox("Diabetes", ["Yes", "No"], index=1)
        smoke = st.selectbox("Current Smoker", ["Yes", "No"], index=1)
        dyspnea = st.selectbox(
            "Dyspnea", ["Yes", "No", "Unknown/Other"], index=2
        )  # Unknown_Other
        vent = st.selectbox("Ventilator >48 Hours", ["Yes", "No"], index=1)
        hxcopd = st.selectbox("COPD", ["Yes", "No"], index=1)
        ascites = st.selectbox("Ascites", ["Yes", "No"])
        hxchf = st.selectbox("Congestive Heart Failure", ["Yes", "No"], index=1)
        hypermed = st.selectbox("Hypertension", ["Yes", "No"], index=1)
        renal_failure = st.selectbox("Acute Renal Failure", ["Yes", "No"], index=1)
        dialysis = st.selectbox("Dialysis", ["Yes", "No"], index=1)
        discancr = st.selectbox("Disseminated Cancer", ["Yes", "No"], index=1)
        wndinf = st.selectbox("Wound Infection", ["Yes", "No", "Unknown"], index=2)
        steroid = st.selectbox("Corticosteroid Use", ["Yes", "No"], index=1)
        wtloss = st.selectbox(
            f"Weight Loss (>10% body weight loss in last 6 months)",
            ["Yes", "No", "Unknown/Other"],
            index=2,  # --> Unknown_Other
        )
        bleed = st.selectbox("Bleeding Disorder", ["Yes", "No"], index=1)
        transfus = st.selectbox("Blood Transfusion", ["Yes", "No"], index=1)
        prsepis = st.selectbox("Sepsis", ["Yes", "No"], index=1)
        ## Albumin
        pralbumin = st.number_input(
            "Albumin (g/dL)", min_value=0.0, max_value=None, value=4.2
        )

        ## WBC
        prwbc = st.number_input(
            "White Blood Cell Count (*10^9/L)",
            min_value=0.0,
            max_value=None,
            value=7.0,
        )

        func_stat = st.selectbox(
            "Functional Status", ["Independent", "Dependent"], index=0
        )  # --> 1/0
    # ================== Intra-Op ===================
    with col3:
        st.subheader("Intra-Operative Characteristics")
        mal_neoplasm = st.selectbox(
            "Location of Tongue Tumor",
            [
                "Anterior two-thirds",  # --> Malignant neoplasm of anterior two-thirds of tongue unspecified
                "Base",  # --> Malignant neoplasm of base of tongue
                "Border",  # --> Malignant neoplasm of border of tongue
                "Junctional Zone",  # --> Malignant neoplasm of junctional zone of tongue
                "Surface",  # --> Malignant neoplasm of surface of tongue
                "Unspecified",  # --> Malignant neoplasm of surface of tongue
            ],
            index=5,
        )
        inout = st.selectbox("Setting", ["Inpatient", "Outpatient"], index=0)  # --> 1/0
        elect_surg = st.selectbox(
            "Case Type",
            [
                "Elective",  # --> Elective
                "Urgent/Emergent",  # --> Urgent_Emergent
                "Unknown",  # --> Unknown
            ],
            index=1,
        )
        asa_class = st.selectbox(
            "ASA Class",
            [
                "1-No Disturbance",  # --> 1-No Disturb
                "2-Mild Disturbance",  # --> 2-Mild Disturb
                "3-Severe Disturbance",  # --> 3-Severe Disturb
                "4-Life Threatening Disturbance",  # --> 4-Life Threat
            ],
            index=2,
        )
        ##Op-time
        optime = st.number_input(
            "Operation Time (minutes)", min_value=0.0, max_value=None, value=214.0
        )
    # ================== Proc ===================
    with col4:
        st.subheader("Head and Neck Procedures")
        part_gloss = st.selectbox("Partial Glossectomy", ["Yes", "No"], index=1)
        comp_ext_gloss = st.selectbox(
            "Composite Extended Glossectomy", ["Yes", "No"], index=1
        )
        total_gloss = st.selectbox("Total Glossectomy", ["Yes", "No"], index=1)
        tongue_exc = st.selectbox("Excision of Tongue Lesions", ["Yes", "No"], index=1)
        oral_cav_recon = st.selectbox(
            "Local/Regional Tissue Flaps for Oral Cavity Reconstruction",
            ["Yes", "No"],
            index=1,
        )
        free_tissue_transfer = st.selectbox(
            "Free Tissue Transfer", ["Yes", "No"], index=1
        )
        skin_auto = st.selectbox("Skin Autograft", ["Yes", "No"], index=1)
        neck_diss = st.selectbox(
            "Neck Dissection and Lymphadenectomy Procedure", ["Yes", "No"], index=1
        )
        alv_ridge = st.selectbox(
            "Alveolar Ridge and Gingival Procedure", ["Yes", "No"], index=1
        )
        mand_res = st.selectbox(
            "Mandibular Resection and Reconstruction Procedure", ["Yes", "No"], index=1
        )
        peri_nerve = st.selectbox(
            "Peripheral Nerve Repair and Neuroplasty", ["Yes", "No"], index=1
        )
        trach_proc = st.selectbox("Tracheostomy Procedure", ["Yes", "No"], index=1)
        gast_eso_proc = st.selectbox(
            "Gastrostomy and Esophageal Access Procedure", ["Yes", "No"], index=1
        )
        sub_gland = st.selectbox("Submandibular Gland Excision", ["Yes", "No"], index=1)
        parotid = st.selectbox("Parotid Gland Excision", ["Yes", "No"], index=1)
        laryngeal = st.selectbox(
            "Laryngeal Resection and Reconstruction Procedure", ["Yes", "No"], index=1
        )
        pharyngeal = st.selectbox(
            "Pharyngeal Resection and Reconstruction Procedure", ["Yes", "No"], index=1
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

    #################################################################################################################
    ############################################### Output Section ##################################################
    #################################################################################################################
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


if __name__ == "__main__":
    main()

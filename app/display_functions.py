## Append path to root
import sys
from pathlib import Path

## Project imports
BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
from src.preprocess import remove_prefix
from app.shap_utils import (
    pretty_feature_name,
    feature_value_label,
    combine_encoded_for_app,
)
from app.config import CHOSEN_MODEL_DICT
import app.utils as util

## Other imports
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches


#################### MAIN CLINICAL RESULTS ####################
# Cache loading for performance
@st.cache_resource
def load_model_pipeline_explainer(outcome_name):
    """Load model and preprocessor for a specific outcome."""
    ## MODEL
    model_path = (
        BASE_PATH
        / "app"
        / "models"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.joblib"
    )
    model = joblib.load(model_path)
    ## PIPELINE
    preprocessor = joblib.load(
        BASE_PATH / "app" / "preprocessors" / f"{outcome_name}_pipeline.joblib"
    )
    ## EXPLAINER
    explainer = joblib.load(
        BASE_PATH / "app" / "shap_explainers" / f"{outcome_name}.joblib"
    )
    feat_names = joblib.load(
        BASE_PATH / "app" / "shap_explainers" / "feature_names.joblib"
    )
    explainer.feature_names = feat_names
    return model, preprocessor, explainer


def plot_risk_bins(bin_occur_rates, bin_idx, folder_name, color):
    """
    Plot occurrence rates per bin with patient's bin highlighted.

    Args:
        bin_occur_rates: List of occurrence rates [0.001, 0.004, 0.009, 0.018]
        bin_idx: Index of patient's assigned bin (0-3)
        display_name: Name of outcome for labeling
    """
    labels = ["Very Low", "Low", "Moderate", "High"]

    # Create color list: highlight patient's bin
    colors = ["#BDBDBD"] * len(labels)  # default grey
    colors[bin_idx] = color

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [rate * 100 for rate in bin_occur_rates], color=colors)

    # Add percentage labels on top of bars
    for i, (bar, rate) in enumerate(zip(bars, bin_occur_rates)):
        height = bar.get_height()
        label = f"{rate:.1%}"

        # Make highlighted bar's label bold
        if i == bin_idx:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                color="black",  # Black for selected
                zorder=10,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontweight="normal",
                fontsize=10,
                color="#BDBDBD",  # Grey for non-selected
                zorder=10,
            )
    # Grey out x-axis tick labels for non-selected bins
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    # Modify tick label colors
    for i, tick_label in enumerate(ax.get_xticklabels()):
        if i == bin_idx:
            tick_label.set_color("black")
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color("#BDBDBD")  # Grey for non-selected
    # horizontal line for overall population rate
    _, true = util.load_population_probs(folder_name)
    tot_occur_rate = true.mean()  # type: ignore
    ax.axhline(
        y=tot_occur_rate * 100,
        color="#467fb8",  # Blue to contrast with grey/colored bars
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Average risk across all breast surgery patients: {tot_occur_rate:.1%}",
        zorder=5,  # Ensure line appears above grid
    )
    # legend for the reference line
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_ylabel(
        "Complication Rate (%)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Risk Category",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "Observed Complication Rate by Risk Category",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, max([r * 100 for r in bin_occur_rates]) * 1.15)  # Add headroom

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    return fig


def show_imputed(display_name, folder_name, input_data, num_dict, imp_cols):
    """
    Display imputed values if applicable
    """
    _, preprocessor, _ = load_model_pipeline_explainer(folder_name)
    with st.expander(f"📋 Imputed Values for {display_name}"):
        ## Get pipeline steps
        num_pipe = preprocessor.named_transformers_["num"]
        imputer = num_pipe.named_steps["imputer"]
        ## Get intermediate values
        X_num_raw = input_data[list(num_dict.keys())]
        X_imputed = imputer.transform(X_num_raw)
        # Build a small df for display
        num_cols_list = list(num_dict.keys())
        imp_display = {}
        for col in imp_cols:
            ## Find index amongst num cols
            col_idx = num_cols_list.index(col)
            raw_val = X_imputed[0, col_idx]
            round_rule = num_dict[col]["round rule"]
            imp_display[num_dict[col]["Display Name"]] = round_rule(raw_val)

        display_df = pd.DataFrame.from_dict(
            imp_display, orient="index", columns=["Value"]
        )
        st.dataframe(display_df, width="content")


def get_input_data():
    st.header("Patient Information")
    col1, col2, col3, col4, col5 = st.columns(5)
    # ================== Demographics ===================
    with col1:
        with st.expander("**Demographics**", expanded=True):
            # --> 1/0
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                index=0,
                help="Patient-reported biological sex at the time of surgery.",
            )
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
                    help="Unit used for weight entry (kg or lbs).",
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
                    st.number_input(
                        "Weight",
                        value=0.0,
                        disabled=True,
                        help="""
                        Patient’s body weight, recorded preoperatively. Used to calculate BMI.

                        *Currently marked as unknown*
                        """,
                    )
                    weight = None
                else:
                    weight_unit = st.session_state.get("weight_unit", "kg")
                    if weight_unit == "lbs":
                        weight = st.number_input(
                            "Weight (lbs)",
                            min_value=2.20462,
                            key="weight_lbs",
                            help="Patient’s body weight, recorded preoperatively. Used to calculate BMI.",
                        )
                    else:
                        weight_kg = st.number_input(
                            "Weight (kg)",
                            min_value=1.0,
                            key="weight_kg",
                            help="Patient’s body weight, recorded preoperatively. Used to calculate BMI.",
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
                    help="Unit used for height entry (m or inches).",
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
                    st.number_input(
                        "Height",
                        value=0.0,
                        disabled=True,
                        help="""
                            Patient’s height as recorded preoperatively.

                            *Currently marked as unknown*""",
                    )
                    height = None
                else:
                    height_unit = st.session_state.get("height_unit", "m")
                    if height_unit == "in":
                        height = st.number_input(
                            "Height (in)",
                            min_value=39.3701,
                            key="height_in",
                            help="Patient’s height as recorded preoperatively.",
                        )
                    else:
                        height_m = st.number_input(
                            "Height (m)",
                            min_value=1.0,
                            key="height_m",
                            help="Patient’s height as recorded preoperatively.",
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
                    st.number_input(
                        "Age",
                        value=0.0,
                        disabled=True,
                        help="""
                            Age in years on the date of the principal operative procedure.

                            *Currently marked as unknown*
                            """,
                    )
                    age = None
                else:
                    age = st.number_input(
                        "Age",
                        min_value=18,
                        max_value=90,
                        value=63,
                        help="Age in years on the date of the principal operative procedure.",
                    )
            # --> 1/0
            hispanic = st.selectbox(
                "Ethnicity",
                ["Hispanic", "Not Hispanic/Unknown"],
                index=1,
                help="Self-reported ethnicity, categorized as Hispanic or Not Hispanic/Unknown.",
            )
            race = st.selectbox(
                "Race",
                [
                    "White",
                    "Black or African American",
                    "Asian",
                    "Unknown/Other",  # --> Unknown_Other
                ],
                index=0,
                help="Self-reported race category documented in the medical record.",
            )
    # ================== Pre-Op ===================
    with col2:
        with st.expander("**Pre-Operative Health Status**", expanded=True):
            diabetes = st.selectbox(
                "Diabetes",
                ["Yes", "No"],
                index=1,
                help="The patient requires daily exogenous insulin or oral hypoglycemic agents. Diet-controlled diabetes does not qualify.",
            )
            smoke = st.selectbox(
                "Current Smoker (within 1 year)",
                ["Yes", "No"],
                index=1,
                help="The patient has smoked cigarettes within 1 year before surgery. Use of cigars, pipes, or smokeless tobacco is not included.",
            )
            dyspnea = st.selectbox(
                "Dyspnea",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Shortness of breath noted at rest or on exertion within 30 days pre-op, based on clinical documentation.",
            )
            vent = st.selectbox(
                "Ventilator >48 Hours",
                ["Yes", "No"],
                index=1,
                help="The patient requires mechanical ventilation for any duration during the 48 hours immediately preceding surgery. CPAP for sleep apnea is excluded.",
            )
            hxcopd = st.selectbox(
                "COPD (Severe Chronic Obstructive Pulmonary Disease)",
                ["Yes", "No"],
                index=1,
                help="""
                    A history of emphysema and/or chronic bronchitis meeting at least one of the following:

                    &nbsp;&nbsp;&nbsp;&nbsp;•Functional limitation requiring oxygen or limiting ADLs

                    &nbsp;&nbsp;&nbsp;&nbsp;•Prior hospitalization for COPD

                    &nbsp;&nbsp;&nbsp;&nbsp;•Chronic bronchodilator therapy

                    &nbsp;&nbsp;&nbsp;&nbsp;•FEV1 <75% predicted

                    &nbsp;&nbsp;&nbsp;&nbsp;•Asthma, interstitial fibrosis, or sarcoidosis are excluded.
                """,
            )
            ascites = st.selectbox(
                "Ascites (Within 30 Days Prior to Surgery)",
                ["Yes", "No"],
                help="Clinically detectable or radiographically confirmed peritoneal fluid within 30 days pre-op. Must be associated with liver disease or malignancy unless otherwise documented.",
                index=1,
            )
            hxchf = st.selectbox(
                "Congestive Heart Failure (Within 30 Days Prior to Surgery)",
                ["Yes", "No"],
                index=1,
                help="Newly diagnosed CHF within 30 days or chronic CHF with active symptoms/signs in the 30 days prior to surgery.",
            )
            hypermed = st.selectbox(
                "Hypertension Requiring Medication",
                ["Yes", "No"],
                index=1,
                help="Documented diagnosis of hypertension requiring antihypertensive medication within 30 days before surgery.",
            )
            renal_failure = st.selectbox(
                "Acute Renal Failure",
                ["Yes", "No", "Unknown"],
                index=1,
                help="""
                    Preoperative renal dysfunction defined as Stage 2/3 AKI:

                    &nbsp;&nbsp;&nbsp;&nbsp;•Stage 2: Serum creatinine 2.0–<3.0× baseline within 7 days

                    &nbsp;&nbsp;&nbsp;&nbsp;•Stage 3: Any of the following:

                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•≥3.0× baseline within 7 days

                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•Rise of ≥0.3 mg/dL to ≥4.0 mg/dL within 48 hours

                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•≥1.5× baseline to ≥4.0 mg/dL within 7 days
                """,
            )
            dialysis = st.selectbox(
                "Dialysis",
                ["Yes", "No"],
                index=1,
                help="Patient has required peritoneal dialysis, hemodialysis, hemofiltration, hemodiafiltration, or ultrafiltration within 2 weeks prior to surgery. Patients who refuse indicated dialysis are coded “Yes.",
            )
            discancr = st.selectbox(
                "Disseminated Cancer",
                ["Yes", "No"],
                index=1,
                help="""
                    Metastatic cancer to at least one major organ and ONE of the following:

                    &nbsp;&nbsp;&nbsp;&nbsp;•Receiving active treatment within the past year\n

                    &nbsp;&nbsp;&nbsp;&nbsp;•Declined treatment

                    &nbsp;&nbsp;&nbsp;&nbsp;•Deemed untreatable


                    **Includes**: ALL, AML, Stage IV lymphomas.

                    **Excludes**: CLL, CML, lymphomas stages I–III, multiple myeloma.

                    """,
            )
            wndinf = st.selectbox(
                "Wound Infection",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Presence of superficial, deep, or organ/space infection documented preoperatively.",
            )
            steroid = st.selectbox(
                "Corticosteroid",
                ["Yes", "No"],
                index=1,
                help="Use of systemic corticosteroids, anti-rejection agents, DMARDs, or other immunosuppressants for ≥10 days within 30 days pre-op, or on an active long-interval regimen extending into the surgical period.",
            )
            wtloss = st.selectbox(
                "Weight Loss",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Unintentional weight loss >10% of body weight in the 6 months prior to surgery.",
            )
            bleed = st.selectbox(
                "Bleeding Disorder",
                ["Yes", "No"],
                index=1,
                help="History of congenital or acquired bleeding diathesis, anticoagulation therapy, or clinical coagulopathy documented preoperatively.",
            )
            transfus = st.selectbox(
                "Blood Transfusion (Preoperative)",
                ["Yes", "No"],
                index=1,
                help="Receipt of ≥1 unit of packed RBCs within the 72 hours prior to surgery.",
            )
            prsepis = st.selectbox(
                "Sepsis (Within 48 Hours Prior to Surgery)",
                ["Yes", "No"],
                index=1,
                help="Presence of SIRS, sepsis, or septic shock documented within 48 hours prior to the operation.",
            )
            func_stat = st.selectbox(
                "Functional Status (30 Days Pre-Op)",
                ["Independent", "Dependent", "Unknown"],
                index=0,
                help="""
                    The highest level of independence in ADLs within 30 days prior to surgery:

                    &nbsp;&nbsp;&nbsp;&nbsp;•Independent: No assistance required; may use prosthetics/devices.

                    &nbsp;&nbsp;&nbsp;&nbsp;• Dependent: Requires some or complete assistance with ADLs.
                    """,
            )
            asa_class = st.selectbox(
                "ASA Physical Status Classification",
                [
                    "I",  # --> 1-No Disturb
                    "II",  # --> 2-Mild Disturb
                    "III",  # --> 3-Severe Disturb
                    "IV/V",  # --> 4-Life Threat
                ],
                index=2,
                help="""
                    •ASA I: Normal healthy patient

                    •ASA II: Mild systemic disease

                    •ASA III: Severe systemic disease

                    •ASA IV/V: Severe systemic disease that is life-threatening (ASA IV) or moribund with minimal chance of survival without surgery (ASA V).
                    """,
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
                    st.number_input(
                        "Albumin (g/dL)",
                        value=0.0,
                        disabled=True,
                        help="""
                            Serum albumin drawn within 90 days pre-op.

                            *Currently marked as unknown*
                            """,
                    )
                    pralbumin = None
                else:
                    pralbumin = st.number_input(
                        "Albumin (g/dL)",
                        min_value=0.0,
                        max_value=None,
                        value=4.2,
                        help="Serum albumin drawn within 90 days pre-op.",
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
                        "White Blood Cell Count",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent WBC value (*10^9/L) within 90 days prior to surgery.

                            *Currently marked as unknown*
                            """,
                    )
                    prwbc = None
                else:
                    prwbc = st.number_input(
                        "White Blood Cell Count",
                        min_value=0.0,
                        max_value=None,
                        value=7.0,
                        help="Most recent WBC value (*10^9/L) within 90 days prior to surgery.",
                    )

            # HCT
            hct_input_col, hct_check_col = st.columns([3, 2])

            with hct_check_col:
                st.write("")
                st.write("")
                hct_unknown = st.checkbox("N/A", key="hct_unknown")

            with hct_input_col:
                if hct_unknown:
                    st.number_input(
                        "Hematocrit (%)",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent Hct (%) within 90 days prior to surgery.

                            *Currently marked as unknown*
                            """,
                    )
                    prhct = None
                else:
                    prhct = st.number_input(
                        "Hematocrit (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=41.0,
                        help="Most recent Hct (%) within 90 days prior to surgery.",
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
                        "Platelet Count",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent platelet level (*10^9/L) within 90 days prior to surgery.

                            *Currently marked as unknown*
                            """,
                    )
                    prplate = None
                else:
                    prplate = st.number_input(
                        "Platelet Count",
                        min_value=0.0,
                        max_value=None,
                        value=238.0,
                        help="Most recent platelet level (*10^9/L) within 90 days prior to surgery.",
                    )
    # ================== Intra-Op ===================
    with col4:
        with st.expander("**Intra-Operative Characteristics**", expanded=True):
            operyr = st.number_input(
                "Operation Year",
                min_value=2008,
                max_value=2025,
                value=2018,
                help="Calendar year during which the operation occurred.",
            )
            # --> 1/0
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
                help="Primary subsites of tongue malignancy.",
            )

            inout = st.selectbox(
                "Setting (Inpatient or Outpatient)",
                ["Inpatient", "Outpatient"],
                index=0,
                help="Hospital admission status at time of the surgical procedure.",
            )
            elect_surg = st.selectbox(
                "Case Type",
                [
                    "Elective",  # --> Elective
                    "Urgent/Emergent",  # --> Urgent_Emergent
                ],
                index=1,
                help="""
                    Determination of operative priority based on surgeon/anesthesiologist classification.

                    &nbsp;&nbsp;&nbsp;&nbsp;•Urgent/Emergent: Must occur during same admission or within 48 hours and documented as urgent/emergent.

                    &nbsp;&nbsp;&nbsp;&nbsp;•Elective: Planned procedure, scheduled in advance for non-life-threatening issues or quality of life.
                    """,
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
                        help="""
                            Total operative duration in minutes for the principal procedure and related components.

                            *Currently marked as unknown*
                            """,
                    )
                    optime = None
                else:
                    optime = st.number_input(
                        "Operation Time (minutes)",
                        min_value=0.0,
                        max_value=None,
                        value=214.0,
                        help="Total operative duration in minutes for the principal procedure and related components.",
                    )
    # ================== Proc ===================
    with col5:
        with st.expander("**Head and Neck Procedures**", expanded=True):
            part_gloss = st.selectbox(
                "Partial Glossectomy",
                ["Yes", "No"],
                index=1,
                help="""
                    Surgical excision of a portion of the tongue for benign or malignant disease, including unilateral or bilateral limited resections.

                    ***CPT codes: 41120, 41130, 41135***
                """,
            )
            comp_ext_gloss = st.selectbox(
                "Composite Extended Glossectomy",
                ["Yes", "No"],
                index=1,
                help="""
                    En bloc resection of the tongue with contiguous structures such as the floor of mouth, base of tongue, or mandible for advanced lesions .

                    ***CPT codes: 41150, 41153, 41155***
                    """,
            )
            total_gloss = st.selectbox(
                "Total Glossectomy",
                ["Yes", "No"],
                index=1,
                help="""
                    Complete surgical removal of the entire tongue, often requiring reconstruction of oral continuity.

                    ***CPT codes: 41140, 41145***
                    """,
            )
            tongue_exc = st.selectbox(
                "Excision of Tongue Lesions",
                ["Yes", "No"],
                index=1,
                help="""
                    Local or limited excision of superficial, benign, dysplastic, or focal tongue lesions not requiring major resection.

                    ***CPT codes: 41100, 41105, 41110, 41112, 41113, 41114, 41116***
                """,
            )
            oral_cav_recon = st.selectbox(
                "Local/Regional Tissue Flaps",
                ["Yes", "No"],
                index=1,
                help="""
                    Reconstruction using advancement, rotation, transposition, or other pedicled flaps from adjacent oral, cervical, or facial tissues.

                    ***CPT codes: 14020, 14021, 14040, 14041, 14301, 14302, 15733, 15740***
                """,
            )
            free_tissue_transfer = st.selectbox(
                "Free Tissue Transfer",
                ["Yes", "No"],
                index=1,
                help="""
                    Microvascular transfer of soft tissue or vascularized bone free flaps for reconstruction following tumor resection or composite defect repair.

                    ***CPT codes: 15732, 15734, 15736, 15738, 15750, 15756, 15757, 15758, 15770, 20902, 20955, 20962, 20969, 21215, 21230, 42894***
                    """,
            )
            skin_auto = st.selectbox(
                "Skin Autograft",
                ["Yes", "No"],
                index=1,
                help="""
                    Application of split- or full-thickness autologous skin grafts for reconstruction or coverage of surgical defects.

                    ***CPT codes: 15004, 15100, 15101, 15120, 15121, 15200, 15220, 15240, 15241, 15271, 15272, 15273, 15275***.
                    """,
            )
            neck_diss = st.selectbox(
                "Lymphadenectomy Procedure",
                ["Yes", "No"],
                index=1,
                help="""
                    Cervical lymph node dissection—selective, modified radical, or radical—for oncologic staging or control.

                    ***CPT codes: 31365, 38500, 38510, 38542, 38700, 38720, 38724, 41135, 41140, 41145, 41153, 41155***.
                    """,
            )
            alv_ridge = st.selectbox(
                "Alveolar Ridge and Gingival Procedure",
                ["Yes", "No"],
                index=1,
                help="""
                    Excision, resection, or reconstruction involving gingival tissue or the alveolar ridge for neoplastic or structural indications.

                    ***CPT codes: 40840, 40845, 41874)***
                    """,
            )
            mand_res = st.selectbox(
                "Mandibular Resection/Reconstruction",
                ["Yes", "No"],
                index=1,
                help="""
                    Segmental or marginal mandibulectomy with or without reconstruction using plates or vascularized/non-vascularized bone grafts.

                    ***CPT codes: 21025, 21044, 21045, 21047, 21196, 21198, 21244, 21245, 21461***
                    """,
            )
            peri_nerve = st.selectbox(
                "Peripheral Nerve Repair",
                ["Yes", "No"],
                index=1,
                help="""
                    Microsurgical neurolysis, primary nerve repair, or nerve grafting for peripheral nerves in the head and neck region.

                    ***CPT codes: 64716, 64740, 64864, 64885, 64886***
                    """,
            )
            trach_proc = st.selectbox(
                "Tracheostomy Procedure",
                ["Yes", "No"],
                index=1,
                help="""
                    Creation of a temporary or permanent surgical airway through tracheal incision.

                    ***CPT codes: 31600, 31603, 31610, 31611***
                    """,
            )
            gast_eso_proc = st.selectbox(
                "Gastrostomy and Esophageal Access Procedure",
                ["Yes", "No"],
                index=1,
                help="""
                    Placement of a gastrostomy tube or surgical creation of an esophageal access point for enteral feeding.

                    ***CPT codes: 43030, 43830, 43832, 44120, 44500, 49440***
                    """,
            )
            sub_gland = st.selectbox(
                "Submandibular Gland Excision",
                ["Yes", "No"],
                index=1,
                help="""
                    Removal of the submandibular gland for neoplasm, chronic infection, or sialolithiasis.

                    ***CPT codes: 42420, 42440, 42450)***
                    """,
            )
            parotid = st.selectbox(
                "Parotid Gland Excision",
                ["Yes", "No"],
                index=1,
                help="""
                    Superficial or total parotidectomy with or without facial nerve dissection or preservation.

                    ***CPT codes: 42410, 42415, 42505***
                    """,
            )
            laryngeal = st.selectbox(
                "Laryngeal Resection/Reconstruction",
                ["Yes", "No"],
                index=1,
                help="""
                    Partial or total laryngectomy or reconstruction of the laryngeal framework following tumor resection.

                    ***CPT codes: 31360, 31365, 31367, 31395, 31599***
                    """,
            )
            pharyngeal = st.selectbox(
                "Pharyngeal Resection/Reconstruction",
                ["Yes", "No"],
                index=1,
                help="""
                    Surgical excision or reconstruction of the pharyngeal wall for neoplastic, structural, or post-radiation indications.

                    ***CPT codes: 31395, 42808, 42890, 42892, 42894, 42950, 42953, 42962***
                    """,
            )
            tonsil = st.selectbox(
                "Tonsillectomy and Tonsillar Region Procedure",
                ["Yes", "No"],
                index=1,
                help="""
                    Excision of tonsillar tissue or resection of adjacent peritonsillar or oropharyngeal structures.

                    ***CPT codes: 42821, 42826, 42842, 42844, 42845, 42870, 42961***""",
            )

    # ================== Create input DF ===================
    input_data = pd.DataFrame(
        {
            ## Col 1 demographics
            "SEX": [util.transform_sex(sex)],  #
            "WEIGHT": [weight],
            "HEIGHT": [height],
            "Age": [age],
            "ETHNICITY_HISPANIC": [util.transform_hispanic(hispanic)],  #
            "RACE_NEW": [util.transform_unknown_other(race)],  #
            ## Col 2 Pre-Op Health
            "DIABETES": [util.transform_yes_no(diabetes)],
            "SMOKE": [util.transform_yes_no(smoke)],
            "DYSPNEA": [util.transform_yes_no_unknown(dyspnea, "DYSPNEA")],
            "VENTILAT": [util.transform_yes_no(vent)],
            "HXCOPD": [util.transform_yes_no(hxcopd)],
            "ASCITES": [util.transform_yes_no(ascites)],
            "HXCHF": [util.transform_yes_no(hxchf)],
            "HYPERMED": [util.transform_yes_no(hypermed)],
            "RENAFAIL": [util.transform_yes_no_unknown(renal_failure, "RENAFAIL")],
            "DIALYSIS": [util.transform_yes_no(dialysis)],
            "DISCANCR": [util.transform_yes_no(discancr)],
            "WNDINF": [util.transform_yes_no_unknown(wndinf, "WNDINF")],
            "STEROID": [util.transform_yes_no(steroid)],
            "WTLOSS": [util.transform_yes_no_unknown(wtloss, "WTLOSS")],
            "BLEEDDIS": [util.transform_yes_no(bleed)],
            "TRANSFUS": [util.transform_yes_no(transfus)],
            "PRSEPIS": [util.transform_yes_no(prsepis)],
            "FNSTATUS2": [util.transform_unknown_other(func_stat)],
            "ASACLAS": [util.transform_ASA(asa_class)],
            ## Col 3 Pre-Op Blood
            "PRALBUM": [pralbumin],
            "PRWBC": [prwbc],
            "PRHCT": [prhct],
            "PRPLATE": [prplate],
            ## Col 4 Intra-Op
            "OPERYR": [operyr],
            "Malignant neoplasm": [util.transform_tumor_site(mal_neoplasm)],
            "INOUT": [util.transform_inout(inout)],
            "URGENCY": [util.transform_casetype(elect_surg)],
            "OPTIME": [optime],
            ## Col 5 head + neck
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
    ## Maps column name to unknown
    ### This order is hard-coded and matches order passed into imputer
    num_dict = {
        "AGE": {
            "Value": age,
            "Display Name": "Age",
            "round rule": lambda x: int(round(x)),
        },
        "HEIGHT": {
            "Value": height,
            "Display Name": "Height (in)",
            "round rule": lambda x: round(x, 2),
        },
        "WEIGHT": {
            "Value": weight,
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
            "Display Name": "Hematocrit (%)",
            "round rule": lambda x: round(x, 2),
        },
        "PRPLATE": {
            "Value": prplate,
            "Display Name": "Platelet Count (*10^9/L)",
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
    return input_data, num_dict, imp_cols


def get_shap_plot(explainer, input_data, num_feats, pipeline):
    expected_features = explainer.feature_names
    input_data = input_data[expected_features]
    shap_raw = explainer(input_data)
    ## Combine one-hot encodeed values
    shap_combined = combine_encoded_for_app(input_data, shap_raw)

    n_feats = min(num_feats, input_data.shape[1])
    num_name, num_pipe, num_cols = pipeline.transformers_[0]
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

    # SHAP feature names after full pipeline
    feat_names = list(shap_combined.feature_names)
    # indices of numeric outputs in SHAP explanation
    num_indices = [feat_names.index(col) for col in num_out_cols]
    x_trans_row = shap_combined.data[0]
    x_num_scaled = np.array([x_trans_row[i] for i in num_indices]).reshape(1, -1)  # type: ignore
    # inverse MinMax scaling
    x_num_original = scaler.inverse_transform(x_num_scaled)
    num_original_series = pd.Series(x_num_original.ravel(), index=num_out_cols)

    ##### MAKE PLOT ######
    # phi contains SHAP values in log-odds scale
    phi = shap_combined.values[0]
    feat_names = np.array(shap_combined.feature_names)
    disp_row = (
        shap_combined.display_data[0]
        if shap_combined.display_data is not None
        else shap_combined.data[0]
    )

    # sort by absolute impact
    order = np.argsort(-np.abs(phi))
    phi = phi[order]
    feat_names = feat_names[order]
    disp_row = disp_row[order]

    # choose k "explicit" features; aggregate the rest
    k = min(n_feats, len(phi))

    phi_main = phi[:k]
    feat_main = feat_names[:k]
    disp_main = disp_row[:k]

    phi_tail = phi[k:]

    if len(phi_tail) > 0:
        # sum remaining SHAP values (same units as phi)
        tail_sum = phi_tail.sum()
        tail_count = len(phi_tail)

        # append aggregated "other features" entry
        phi_all = np.concatenate([phi_main, np.array([tail_sum])])
        feat_all = np.concatenate(
            [feat_main, np.array([f"{tail_count} other features"])]
        )
        disp_all = np.concatenate([disp_main, np.array(["(aggregated)"])])
    else:
        phi_all = phi_main
        feat_all = feat_main
        disp_all = disp_main

    # NORMALIZE: Convert log-odds to percentage contributions
    total_abs_contribution = np.sum(np.abs(phi_all))

    if total_abs_contribution > 0:
        # Each feature's contribution as % of total absolute deviation
        phi_top = (phi_all / total_abs_contribution) * 100.0
    else:
        # Edge case: all SHAP values are zero
        phi_top = np.zeros_like(phi_all)

    feat_top = feat_all
    disp_top = disp_all

    # Calculate dynamic font sizes based on number of bars
    num_bars = len(phi_top)

    # Y-tick label size: scale inversely with number of bars
    # Range: 14pt (few bars) down to 9pt (many bars)
    ytick_fontsize = max(9, min(14, 14 - (num_bars - 5) * 0.15))

    # Bar label size: slightly smaller than y-ticks
    # Range: 12pt (few bars) down to 8pt (many bars)
    bar_label_fontsize = max(8, min(12, 12 - (num_bars - 5) * 0.15))

    fig, ax = plt.subplots(figsize=(12, 14))

    colors = np.where(phi_top >= 0, "#ff006e", "#118ab2")
    bars = ax.barh(range(len(phi_top)), phi_top, color=colors)

    ax.set_yticks(range(len(phi_top)))
    ax.set_yticklabels(
        [
            (
                f"{pretty_feature_name(name)}: {feature_value_label(name, val, num_original_series)}"
                if "(other features)" not in name
                and name != f"{len(phi_tail)} other features"
                else f"{name}"
            )  # label for the aggregated bar
            for name, val in zip(feat_top, disp_top)
        ],
        fontsize=ytick_fontsize,
    )

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.set_xlabel(
        "Relative contribution to risk prediction (%)",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_ylabel(
        f"Top {k} factors ranked by impact on risk",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_title(
        "Which patient factors increase or decrease this patient's risk \nCalculated using SHAP (SHapley Additive exPlanations)",
        fontweight="bold",
        pad=15,
        fontsize=20,
    )

    values = phi_top  # normalized percentage contributions
    bar_lengths = np.array(values)

    # extremes on each side
    max_right = np.max(bar_lengths[bar_lengths > 0], initial=0.0)
    max_left = np.min(bar_lengths[bar_lengths < 0], initial=0.0)

    # Calculate total range
    value_range = max_right - max_left

    # Use percentage-based padding (15% of total range)
    # If range is 0 (all bars same sign), use absolute padding
    if value_range > 0:
        padding_pct = 0.15  # 15% padding on each side
        padding = value_range * padding_pct
    else:
        # Fallback for edge case: all bars same sign
        max_abs = max(abs(max_right), abs(max_left))
        padding = max(0.5, max_abs * 0.2)  # 20% of max value or minimum 0.5

    # Calculate limits
    x_min = max_left - padding
    x_max = max_right + padding

    # ensure non-zero width
    if x_max <= x_min:
        center = (max_left + max_right) / 2
        half_width = 0.5
        x_min, x_max = center - half_width, center + half_width

    ax.set_xlim(x_min, x_max)

    # add labels anchored at bar tips
    for bar, value in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ha = "left" if x >= 0 else "right"

        ax.text(
            x,
            y,
            f"{value:+.1f}%",
            va="center",
            ha=ha,
            fontsize=bar_label_fontsize,
            color="black",
        )

    red_patch = mpatches.Patch(color="#ff006e", label="Increases predicted risk")
    blue_patch = mpatches.Patch(color="#118ab2", label="Decreases predicted risk")

    ax.legend(
        handles=[red_patch, blue_patch],
        loc="best",
        frameon=True,
        title="Direction of effect",
        fontsize=13,
        title_fontsize=15,
    )
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=2.0)
    return fig


def show_clinical_results(display_name, folder_name, input_data):
    """
    Render clinical output.

    Params
    ------
    display_name: str
        Full name of outcome displayed on interface
    folder_name: str
        Outcome abbreviation
    input_data: pd.DataFrame
        User input single patient data

    """
    with st.expander(f"📊 {display_name}", expanded=True):
        try:
            # ================== Get model output ===================
            # Load model and preprocessor
            model, preprocessor, explainer = load_model_pipeline_explainer(folder_name)

            ## Preprocess
            feature_names = preprocessor.get_feature_names_out()
            data_transformed = np.array(preprocessor.transform(input_data))
            processed_data = pd.DataFrame(data_transformed, columns=feature_names)
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
            ## Get all test output
            all_probs, all_labels = util.load_population_probs(folder_name)
            tot_patients = len(all_labels)
            # ================== Display Results ===================
            # ========== ROW 1: Risk Assessment ==========
            st.markdown("## Risk Assessment")

            col_a1, col_b1 = st.columns(2)

            with col_a1:
                # Display appropriate risk category
                bin_thresholds = util.load_bin_thresholds(folder_name)
                risk, color, color_code = util.get_risk_category(
                    prob_positive, folder_name
                )
                st.metric(label="Risk Category", value=f"{color} {risk}")

                # Display bin context
                labels = ["Very Low", "Low", "Moderate", "High"]
                cutoffs = [0.0] + list(bin_thresholds) + [1.0]
                bin_idx = np.digitize([prob_positive], cutoffs[1:])[0]
                bin_occur_rate_list = util.bin_occur_rates(folder_name, bin_thresholds)

                st.markdown(
                    f"In our independent test cohort of {tot_patients:,} breast surgery patients, <b>~{bin_occur_rate_list[bin_idx]:.1%}</b> of those with similar characteristics developed this complication",
                    unsafe_allow_html=True,
                )

            with col_b1:
                # RISK PLOT
                fig = plot_risk_bins(
                    bin_occur_rate_list,
                    bin_idx,
                    folder_name,
                    color_code,
                )
                st.pyplot(fig, bbox_inches="tight")
                plt.close()

            # ========== ROW 2: Feature Contributions ==========
            st.markdown("### What Drives This Patient's Risk")

            col_a2, col_b2 = st.columns(2)

            with col_a2:
                # SHAP feature count input
                n_feats = st.number_input(
                    "Number of top features to display",
                    min_value=5,
                    max_value=input_data.shape[1],
                    value=10,
                    step=1,
                    key=f"n_feats_{folder_name}",
                )
                # n_feats = 10
                st.info(
                    """
                    **How to read this chart**

                    This chart explains *why* the model gave this patient their risk estimate.

                    - Each bar represents one patient factor (age, lab value, operative detail, etc.).
                    - Bars to the **right** (pink) show factors that **increase** this patient's risk.
                    - Bars to the **left** (blue) show factors that **decrease** this patient's risk.
                    - Longer bars mean a **greater influence** on the risk estimate.
                    - Factors are ranked from **most influential at the top** to least influential at the bottom.

                    The percentages show each factor's **relative contribution** to why this patient's risk differs 
                    from the average patient in our test cohort. For example, if "Setting: Inpatient" shows "-12%", 
                    this factor accounts for 12% of the total model influence that makes this patient higher or lower 
                    risk than average.

                    
                    **Important:** The absolute values of all bars sum to 100%, representing the complete explanation 
                    of how this patient compares to the average.

                    ---

                    *Technical note for interested users:*  
                    Feature contributions are computed using SHAP (SHapley Additive exPlanations) values in log-odds 
                    space from the uncalibrated model, then normalized such that absolute values sum to 100%. 
                    This normalization maintains the sign, ranking, and relative magnitude of each feature's impact 
                    while providing scale-invariant interpretability across different models and outcomes. Calibrated 
                    risk predictions used to allocate patients into risk categories apply a monotonic transformation 
                    (Platt scaling) on raw model output that preserves feature importance ranking and direction.
                    """
                )

            with col_b2:
                # SHAP plot
                shap_fig = get_shap_plot(
                    explainer,
                    processed_data,
                    n_feats,
                    preprocessor,
                )
                st.pyplot(shap_fig, bbox_inches="tight")
                plt.close(shap_fig)

            # ================== Dropdown to Show Model Details ===================
            st.markdown("---")
            with st.expander(f"Extra information on model output", expanded=False):
                show_model_details(
                    display_name,
                    folder_name,
                    prob_positive,
                )

        except Exception as e:
            st.error(f"Error predicting {display_name}: {str(e)}")


#################### EXTRA INFO ON MODEL ####################
def get_full_model_name(model_abrv):
    if model_abrv == "xgb":
        return "XGBoost"
    elif model_abrv == "lgbm":
        return "LightGBM"
    elif model_abrv == "stack":
        return "Stacked Generalization"
    elif model_abrv == "lr":
        return "Logistic Regression"
    elif model_abrv == "nn":
        return "Neural Network"
    elif model_abrv == "svc":
        return "Support Vector Classifier"
    else:
        raise ValueError(f"Unrecognized model name: {model_abrv}")


def show_model_details(display_name, outcome_abrv, prob_positive):
    """
    Display advanced model information: raw output, percentiles, thresholds, and SHAP.

    Params
    ------
    display_name: str
        Full name of outcome
    outcome_abrv: str
        Outcome abbreviation
    prob_positive: float
        Model prediction probability output for input data
    """
    model_abrv = CHOSEN_MODEL_DICT[outcome_abrv]
    chosen_model = get_full_model_name(model_abrv)
    st.subheader(f"{display_name} Model Details")
    # Load population data
    all_probs, all_labels = util.load_population_probs(outcome_abrv)
    bin_thresholds = util.load_bin_thresholds(outcome_abrv)
    # === Raw Model Output & Percentiles ===
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Calibrated Model Output")
        if outcome_abrv == "mort":
            value = f"{prob_positive:.3%}"
        else:
            value = f"{prob_positive:.2%}"
        st.metric(
            label="Model-assigned risk score",
            value=value,
            delta=None,
        )
        with st.expander("ℹ️ How to interpret this score"):
            st.write(
                "Given the limits of probability calibration for this model, the raw output is used primarily to place patients into risk categories. It should not be taken as an exact individual probability, but interpreted alongside the percentiles and the observed complication rates in each risk category."
            )
        st.markdown("##### Model Used:")
        st.markdown(f"##### {chosen_model}")

    with col_b:
        st.markdown("#### Risk Score Percentiles")

        # Overall percentile
        n_overall = len(all_probs)
        overall_pctile = (all_probs < prob_positive).mean() * 100

        # Percentile among patients WITHOUT the outcome
        neg_patients = all_probs[all_labels == 0]
        n_neg = len(neg_patients)
        neg_pctile = (neg_patients < prob_positive).mean() * 100

        # Percentile among patients WITH the outcome
        pos_patients = all_probs[all_labels == 1]
        n_pos = len(pos_patients)
        pos_pctile = (pos_patients < prob_positive).mean() * 100
        st.markdown(
            f"**{overall_pctile:.1f}%** of all patients (n={n_overall:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{neg_pctile:.1f}%** of patients who did not develop this complication (n={n_neg:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{pos_pctile:.1f}%** of patients who developed this complication (n={n_pos:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        with st.expander("ℹ️ How to interpret these percentiles"):
            st.write(
                f"Model output is compared with {chosen_model}'s predicted scores "
                f"in the independent test cohort (n={n_overall:,}). These percentiles show where this patient’s score falls relative to all patients, to those who did not develop the outcome, and to those who did."
            )
    # === Risk Bin Thresholds ===
    st.markdown("---")
    st.markdown("#### Risk Category Thresholds")
    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["Very Low", "Low", "Moderate", "High"]
        cutoffs = [0.0] + list(bin_thresholds) + [1.0]
        bin_idx = np.digitize([prob_positive], cutoffs[1:])[0]
        threshold_data = []
        for i, lab in enumerate(labels):
            if outcome_abrv == "mort":
                threshold_range = f"{cutoffs[i]:.3%} – {cutoffs[i+1]:.3%}"
            else:
                threshold_range = f"{cutoffs[i]:.2%} – {cutoffs[i+1]:.2%}"
            threshold_data.append(
                {"Risk Category": lab, "Model Output Range": threshold_range}
            )
        threshold_df = pd.DataFrame(threshold_data)

        # Styler to bold active bin
        def highlight_active(row):
            if row.name == bin_idx:
                return ["font-weight: bold;" for _ in row]
            else:
                return ["" for _ in row]

        styled = threshold_df.style.apply(highlight_active, axis=1)

        # Generate HTML to bold selected bin
        html_table = styled.to_html(index=False)
        # render
        st.markdown(html_table, unsafe_allow_html=True)
    with col_b:
        st.markdown(
            f"Risk categories are defined using cutoffs taken from the {chosen_model} model’s predicted scores in the training and validation cohorts (n=7,026) for the {display_name} outcome. These cutoffs follow a logarithmic scale so that higher‑risk ranges are more finely separated than very low‑risk ranges."
        )

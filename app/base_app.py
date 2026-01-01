## Append path to root
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))

## Other imports
import streamlit as st
import display_functions as display
from app.config import OUTCOMES


def init_session_state():
    default_keys = {
        "predictions_made": False,
        "last_input_hash": None,
        "selected_outcomes": [],
        "input_data": None,
        "num_dict": None,
        "imp_cols": [],
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title="PRO-TONGUE",
        page_icon="🏥",
        layout="wide",
    )
    init_session_state()
    st.title("PRO-TONGUE: Post-resection Outcome prediction for Tongue cancer")
    st.markdown("Predict 30-day complications after glossectomy for tongue cancer.")
    st.info(
        "Adjust all fields to match your patient. Default values are set arbitrarily. To reset to default values, refresh the page. "
    )

    #################################################################################################################
    ################################################### Side Bar ####################################################
    #################################################################################################################
    outcome_info_dict = {
        "asp": """
            A postoperative “aspiration” complication is present if a patient experiences any of the following NSQIP-defined events within 30 days of the index operation:
                
            &nbsp;&nbsp;&nbsp;&nbsp;•**Postoperative pneumonia**, as defined using CDC/NHSN-based criteria for new infectious pulmonary infiltrate treated with antibiotics. 
                
            &nbsp;&nbsp;&nbsp;&nbsp;•**Unplanned postoperative reintubation**, defined as unplanned endotracheal intubation and mechanical ventilation after initial extubation due to respiratory or cardiac decompensation. 
                
            &nbsp;&nbsp;&nbsp;&nbsp;•**Prolonged postoperative mechanical ventilation**, defined as requirement for invasive ventilatory support for more than 48 hours after anesthesia end time. 
            """,
        "bleed": "Postoperative bleeding event requiring transfusion of packed red blood cells or whole blood within 72 hours of the end of surgery, recorded when transfusion is given to treat or in response to postoperative hemorrhage.",
        "reop": "Any unplanned return to the operating room for a surgical procedure related to the index or concurrent procedure within 30 days, at any facility; planned/staged procedures are excluded. ",
        "surg": """
            A postoperative “surgical complication” is present if a patient has any surgical site infection or disruption within 30 days of surgery, including:
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Superficial incisional SSI** (skin or subcutaneous tissue only). 
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Deep incisional SSI** (involving fascia or muscle of the incision). 
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Organ/space SSI** (infection involving any organ or space opened or manipulated during the operation, excluding the incision itself). 
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Wound disruption/dehiscence** requiring clinical intervention
            """,
    }
    st.sidebar.header("Select Outcomes to Predict")
    selected_outcomes = []
    for display_name, folder_name in OUTCOMES.items():
        if st.sidebar.checkbox(
            display_name, value=True, help=outcome_info_dict[folder_name]
        ):
            selected_outcomes.append((display_name, folder_name))

    if not selected_outcomes:
        st.warning("Please select at least one outcome to predict")
        return

    ############# Input Section #############
    input_data, num_dict, imp_cols = display.get_input_data()
    current_input_hash = hash(input_data.to_json())
    # If inputs changed compared to the last prediction, reset predictions
    if (
        st.session_state.last_input_hash is not None
        and current_input_hash != st.session_state.last_input_hash
    ):
        st.session_state.predictions_made = False

    ############# Output Section #############
    # Button triggers prediction and stores results in session state
    if st.button("Predict Outcomes", type="primary", key="pred_btn"):
        if display.check_filter_cols(input_data):
            st.session_state.predictions_made = True
            st.session_state.selected_outcomes = selected_outcomes
            st.session_state.input_data = input_data
            st.session_state.num_dict = num_dict
            st.session_state.imp_cols = imp_cols
            st.session_state.last_input_hash = current_input_hash
        else:
            st.error(
                "At least one *Resection Procedure* must be selected before predicting outcomes."
            )
            st.session_state.predictions_made = False

    # Display results if predictions have been made
    if st.session_state.predictions_made:
        st.header("Prediction Results")

        # Process each selected outcome
        for display_name, folder_name in selected_outcomes:
            display.show_clinical_results(display_name, folder_name, input_data)

        # Display imputed values
        if len(st.session_state.imp_cols) > 0:
            st.header("Imputed Values")
            st.info(
                """
            When patient data is missing, the modeling pipeline uses an iterative regression-based imputation method to estimate those values.
            It models each incomplete variable using other available patient characteristics, ***defined in each outcome's respective train set,*** and refines these estimates over several rounds.
            Imputed values are statistical estimates, not actual measurements.
            """
            )
            for display_name, folder_name in selected_outcomes:
                display.show_imputed(
                    display_name, folder_name, input_data, num_dict, imp_cols
                )


if __name__ == "__main__":
    main()
